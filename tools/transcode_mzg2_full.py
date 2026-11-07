#!/usr/bin/env python3
"""Transcode and independently verify a resumable complete Kimi K3.1 MZG2 store."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import struct
import subprocess
import threading
import time
from pathlib import Path

from k31_checkpoint import (
    ARCHITECTURE,
    EXPERT_LAYOUT,
    EXPERT_SPAN_BYTES,
    MODEL_TYPE,
    OFFICIAL_EXPERTS_PER_LAYER,
    OFFICIAL_ROUTED_LAYERS,
    OFFICIAL_SHARDS,
    SCHEMA as CHECKPOINT_SCHEMA,
    UPSTREAM_ID,
    UPSTREAM_REVISION,
    inspect_checkpoint,
)

STORE_SCHEMA = "moonshine-k31-mzg2-full-v1"
PROGRESS_SCHEMA = "moonshine-k31-mzg2-full-progress-v1"
FILE_HEADER = struct.Struct("<8s9I6Q32s")
INDEX_ENTRY = struct.Struct("<HHIQ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_geometry(source: object) -> tuple[dict[str, object], dict[str, object], int, int]:
    """Validate the K3.1 identity and exact routed geometry used by the C tool."""
    if not isinstance(source, dict) or source.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("source is not a K3.1 checkpoint inspection manifest")
    if source.get("upstream") != {
        "id": UPSTREAM_ID,
        "revision": UPSTREAM_REVISION,
        "architecture": ARCHITECTURE,
        "model_type": MODEL_TYPE,
    }:
        raise ValueError("source checkpoint model identity mismatch")
    digest = source.get("source_layout_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("invalid source layout SHA-256")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError("invalid source layout SHA-256") from error
    geometry = source.get("expert_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("missing K3.1 expert geometry")
    layers = geometry.get("routed_layers")
    experts = geometry.get("experts_per_layer")
    if (
        type(layers) is not int
        or layers <= 0
        or type(experts) is not int
        or experts <= 0
        or geometry.get("expert_count") != layers * experts
        or geometry.get("tensors_per_expert") != len(EXPERT_LAYOUT)
        or geometry.get("routed_tensor_count") != layers * experts * len(EXPERT_LAYOUT)
    ):
        raise ValueError("invalid K3.1 routed expert geometry")
    span = geometry.get("bytes_per_expert")
    if span != EXPERT_SPAN_BYTES:
        raise ValueError(f"expert span mismatch: {span} != {EXPERT_SPAN_BYTES}")
    if geometry.get("routed_bytes") != layers * experts * EXPERT_SPAN_BYTES:
        raise ValueError("routed byte count does not match the exact expert span")
    layout = geometry.get("tensor_layout")
    if not isinstance(layout, list) or len(layout) != len(EXPERT_LAYOUT):
        raise ValueError("invalid K3.1 expert tensor layout")
    offset = 0
    for record, (name, size) in zip(layout, EXPERT_LAYOUT):
        if (
            not isinstance(record, dict)
            or record.get("name") != name
            or record.get("dtype") != "U8"
            or record.get("offset") != offset
            or record.get("bytes") != size
        ):
            raise ValueError("K3.1 expert tensor layout mismatch")
        offset += size
    if offset != EXPERT_SPAN_BYTES:
        raise ValueError("expert span mismatch in K3.1 tensor layout")
    return source, geometry, layers, experts


def inspect_source(
    model: Path,
    *,
    expected_shards: int = OFFICIAL_SHARDS,
    expected_routed_layers: int = OFFICIAL_ROUTED_LAYERS,
    expected_experts_per_layer: int = OFFICIAL_EXPERTS_PER_LAYER,
) -> dict[str, object]:
    source = inspect_checkpoint(
        model,
        expected_shards=expected_shards,
        expected_routed_layers=expected_routed_layers,
        expected_experts_per_layer=expected_experts_per_layer,
    )
    source_geometry(source)
    return source


def load_progress(
    path: Path,
    source: dict[str, object],
    layers_count: int,
    experts_per_layer: int,
) -> dict[str, object]:
    if not path.exists():
        return {
            "schema": PROGRESS_SCHEMA,
            "base_model": UPSTREAM_ID,
            "base_revision": UPSTREAM_REVISION,
            "source_layout_sha256": source["source_layout_sha256"],
            "layers_count": layers_count,
            "experts_per_layer": experts_per_layer,
            "expert_span_bytes": EXPERT_SPAN_BYTES,
            "layers": {},
        }
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema") != PROGRESS_SCHEMA
        or value.get("base_model") != UPSTREAM_ID
        or value.get("base_revision") != UPSTREAM_REVISION
        or value.get("source_layout_sha256") != source["source_layout_sha256"]
        or value.get("layers_count") != layers_count
        or value.get("experts_per_layer") != experts_per_layer
        or value.get("expert_span_bytes") != EXPERT_SPAN_BYTES
        or not isinstance(value.get("layers"), dict)
    ):
        raise RuntimeError(f"invalid or stale K3.1 MZG2 progress file: {path}")
    return value


def validate_record(root: Path, layer: int, record: object) -> bool:
    if not isinstance(record, dict):
        return False
    path = root / f"layer-{layer:03d}.mzg2"
    return (
        path.is_file()
        and path.stat().st_size == record.get("bytes")
        and sha256_file(path) == record.get("sha256")
    )


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("wb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def parse_sidecar(
    path: Path,
    expected_layer: int,
    expected_experts: int,
    source_layout_sha256: str,
) -> dict[str, int]:
    file_bytes = path.stat().st_size
    with path.open("rb") as source:
        header_bytes = source.read(FILE_HEADER.size)
        if len(header_bytes) != FILE_HEADER.size:
            raise RuntimeError(f"short MZG2 header: {path}")
        (
            magic,
            version,
            header_size,
            alignment,
            layer,
            expert_count,
            index_entry_bytes,
            model_entry_bytes,
            tile_bytes,
            flags,
            index_offset,
            index_bytes,
            model_offset,
            model_bytes,
            data_offset,
            max_block_bytes,
            manifest_digest,
        ) = FILE_HEADER.unpack(header_bytes)
        if (
            magic != b"K3MZG2\0\0"
            or version != 2
            or header_size != 4096
            or alignment != 4096
            or layer != expected_layer
            or expert_count != expected_experts
            or index_entry_bytes != INDEX_ENTRY.size
            or model_entry_bytes != 0
            or tile_bytes != 16_384
            or flags != 0
            or index_offset != 4096
            or index_bytes != expected_experts * INDEX_ENTRY.size
            or model_offset != 0
            or model_bytes != 0
            or data_offset < index_offset + index_bytes
            or data_offset % 4096
            or max_block_bytes <= 0
            or max_block_bytes % 4096
            or manifest_digest.hex() != source_layout_sha256
        ):
            raise RuntimeError(f"invalid or stale K3.1 MZG2 header: {path}")
        source.seek(index_offset)
        index = source.read(index_bytes)
        if len(index) != index_bytes:
            raise RuntimeError(f"short MZG2 index: {path}")
    block_bytes = 0
    previous_end = data_offset
    for expert in range(expected_experts):
        item_layer, item_expert, item_bytes, item_offset = INDEX_ENTRY.unpack_from(
            index, expert * INDEX_ENTRY.size
        )
        if (
            item_layer != expected_layer
            or item_expert != expert
            or item_bytes <= 0
            or item_bytes % 4096
            or item_bytes > max_block_bytes
            or item_offset != previous_end
            or item_offset + item_bytes > file_bytes
        ):
            raise RuntimeError(f"invalid MZG2 index: {path} expert {expert}")
        block_bytes += item_bytes
        previous_end = item_offset + item_bytes
    if previous_end != file_bytes:
        raise RuntimeError(f"trailing MZG2 bytes: {path}")
    return {
        "file_bytes": file_bytes,
        "block_bytes": block_bytes,
        "max_block_bytes": max_block_bytes,
        "experts": expert_count,
    }


def transcode_store(
    model: Path,
    out: Path,
    *,
    jobs: int = 24,
    tool: Path | None = None,
    source_manifest: dict[str, object] | None = None,
    expected_shards: int = OFFICIAL_SHARDS,
    expected_routed_layers: int = OFFICIAL_ROUTED_LAYERS,
    expected_experts_per_layer: int = OFFICIAL_EXPERTS_PER_LAYER,
) -> dict[str, object]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    tool = tool if tool is not None else Path(__file__).resolve().with_name(
        "transcode_mzg2_layer"
    )
    if not tool.is_file():
        raise ValueError(f"transcoder not found: {tool}")
    if out.exists():
        raise ValueError(f"published output already exists: {out}")
    source = source_manifest if source_manifest is not None else inspect_source(
        model,
        expected_shards=expected_shards,
        expected_routed_layers=expected_routed_layers,
        expected_experts_per_layer=expected_experts_per_layer,
    )
    _, _, layers_count, experts_per_layer = source_geometry(source)
    source_layout_sha256 = str(source["source_layout_sha256"])

    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    logs = partial / "logs"
    logs.mkdir(exist_ok=True)
    progress_path = partial / "progress.json"
    progress = load_progress(
        progress_path, source, layers_count, experts_per_layer
    )
    progress_lock = threading.Lock()
    layers = progress["layers"]
    assert isinstance(layers, dict)

    pending: list[int] = []
    for layer in range(1, layers_count + 1):
        record = layers.get(str(layer))
        if validate_record(partial, layer, record):
            print(f"MZG2 resume layer={layer}: verified existing sidecar")
            continue
        final_path = partial / f"layer-{layer:03d}.mzg2"
        temporary = partial / f"layer-{layer:03d}.mzg2.partial"
        final_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        layers.pop(str(layer), None)
        pending.append(layer)
    atomic_json(progress_path, progress)

    started = time.monotonic()

    def transcode(layer: int) -> None:
        temporary = partial / f"layer-{layer:03d}.mzg2.partial"
        final_path = partial / f"layer-{layer:03d}.mzg2"
        log_path = logs / f"layer-{layer:03d}-transcode.log"
        run_logged(
            [
                str(tool),
                str(model),
                str(temporary),
                str(layer),
                str(experts_per_layer),
                source_layout_sha256,
            ],
            log_path,
        )
        os.replace(temporary, final_path)
        fsync_file(final_path)
        record = {
            "bytes": final_path.stat().st_size,
            "sha256": sha256_file(final_path),
            "transcoded": True,
            "verified": False,
        }
        with progress_lock:
            layers[str(layer)] = record
            atomic_json(progress_path, progress)
        print(f"MZG2 transcode layer={layer}: PASS bytes={record['bytes']}")

    if pending:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(jobs, len(pending))
        ) as pool:
            futures = {pool.submit(transcode, layer): layer for layer in pending}
            for future in concurrent.futures.as_completed(futures):
                future.result()

    verification_pending = [
        layer
        for layer in range(1, layers_count + 1)
        if not bool(layers[str(layer)].get("verified"))
    ]

    def verify(layer: int) -> None:
        sidecar = partial / f"layer-{layer:03d}.mzg2"
        log_path = logs / f"layer-{layer:03d}-verify.log"
        run_logged([str(tool), "--verify", str(model), str(sidecar)], log_path)
        digest = sha256_file(sidecar)
        with progress_lock:
            record = layers[str(layer)]
            if (
                record.get("bytes") != sidecar.stat().st_size
                or record.get("sha256") != digest
            ):
                raise RuntimeError(f"sidecar changed during verify: {sidecar}")
            record["verified"] = True
            atomic_json(progress_path, progress)
        print(f"MZG2 verify layer={layer}: PASS")

    if verification_pending:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(jobs, len(verification_pending))
        ) as pool:
            futures = {
                pool.submit(verify, layer): layer
                for layer in verification_pending
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()

    files: list[dict[str, object]] = []
    total_file_bytes = 0
    total_block_bytes = 0
    max_block_bytes = 0
    total_experts = 0
    for layer in range(1, layers_count + 1):
        path = partial / f"layer-{layer:03d}.mzg2"
        parsed = parse_sidecar(
            path, layer, experts_per_layer, source_layout_sha256
        )
        record = layers[str(layer)]
        if not record.get("verified"):
            raise RuntimeError(f"unverified MZG2 layer {layer}")
        if parsed["file_bytes"] != record.get("bytes"):
            raise RuntimeError(f"sidecar size changed before publish: {path}")
        total_file_bytes += parsed["file_bytes"]
        total_block_bytes += parsed["block_bytes"]
        total_experts += parsed["experts"]
        max_block_bytes = max(max_block_bytes, parsed["max_block_bytes"])
        files.append({
            "layer": layer,
            "path": path.name,
            "bytes": parsed["file_bytes"],
            "sha256": record["sha256"],
        })
    if total_experts != layers_count * experts_per_layer:
        raise RuntimeError("MZG2 expert coverage mismatch")
    raw_bytes = layers_count * experts_per_layer * EXPERT_SPAN_BYTES
    manifest = {
        "schema": STORE_SCHEMA,
        "version": 2,
        "base_model": UPSTREAM_ID,
        "base_revision": UPSTREAM_REVISION,
        "source_layout_sha256": source_layout_sha256,
        "source_checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_tensor_count": source.get("tensor_count"),
        "source_shard_count": len(source.get("shards", [])),
        "layers": layers_count,
        "experts_per_layer": experts_per_layer,
        "experts": total_experts,
        "expert_span_bytes": EXPERT_SPAN_BYTES,
        "tile_bytes": 16_384,
        "raw_bytes": raw_bytes,
        "stored_file_bytes": total_file_bytes,
        "stored_block_bytes": total_block_bytes,
        "reduction_pct": 100.0 * (1.0 - total_file_bytes / raw_bytes),
        "max_block_bytes": max_block_bytes,
        "transcode_and_verify_seconds": time.monotonic() - started,
        "files": files,
    }
    manifest_path = partial / "expert-store.json"
    atomic_json(manifest_path, manifest)
    progress_path.unlink()
    fsync_directory(partial)
    os.replace(partial, out)
    fsync_directory(out.parent)
    print(
        "MZG2 full store: PASS "
        f"experts={total_experts} raw={raw_bytes} stored={total_file_bytes} "
        f"reduction={manifest['reduction_pct']:.9f}% out={out}"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument(
        "--tool",
        type=Path,
        default=Path(__file__).resolve().with_name("transcode_mzg2_layer"),
    )
    args = parser.parse_args()
    try:
        transcode_store(
            args.model,
            args.out,
            jobs=args.jobs,
            tool=args.tool,
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
