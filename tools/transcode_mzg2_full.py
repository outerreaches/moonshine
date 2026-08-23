#!/usr/bin/env python3
"""Transcode and independently verify a resumable complete K3 MZG2 store."""

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

LAYERS = 92
EXPERTS = 896
EXPERT_BYTES = 17_547_264
MANIFEST_SHA256 = "476fa0ba64e3233cbb9ca0642327361a73f6807e751edb071c92fa2216b202a4"
SCHEMA = "moonshine-mzg2-full-progress-v1"
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


def load_progress(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "schema": SCHEMA,
            "source_manifest_sha256": MANIFEST_SHA256,
            "layers": {},
        }
    value = json.loads(path.read_text())
    if (
        value.get("schema") != SCHEMA
        or value.get("source_manifest_sha256") != MANIFEST_SHA256
        or not isinstance(value.get("layers"), dict)
    ):
        raise RuntimeError(f"invalid MZG2 progress file: {path}")
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
        raise RuntimeError(
            f"command failed ({result.returncode}); see {log_path}"
        )


def parse_sidecar(path: Path, expected_layer: int) -> dict[str, int]:
    file_bytes = path.stat().st_size
    with path.open("rb") as source:
        header_bytes = source.read(FILE_HEADER.size)
        if len(header_bytes) != FILE_HEADER.size:
            raise RuntimeError(f"short MZG2 header: {path}")
        values = FILE_HEADER.unpack(header_bytes)
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
        ) = values
        if (
            magic != b"K3MZG2\0\0"
            or version != 2
            or header_size != 4096
            or alignment != 4096
            or layer != expected_layer
            or expert_count != EXPERTS
            or index_entry_bytes != INDEX_ENTRY.size
            or model_entry_bytes != 0
            or tile_bytes != 16_384
            or flags != 0
            or index_offset != 4096
            or index_bytes != EXPERTS * INDEX_ENTRY.size
            or model_offset != 0
            or model_bytes != 0
            or data_offset < index_offset + index_bytes
            or data_offset % 4096
            or max_block_bytes <= 0
            or max_block_bytes % 4096
            or manifest_digest.hex() != MANIFEST_SHA256
        ):
            raise RuntimeError(f"invalid MZG2 header: {path}")
        source.seek(index_offset)
        index = source.read(index_bytes)
        if len(index) != index_bytes:
            raise RuntimeError(f"short MZG2 index: {path}")
    block_bytes = 0
    previous_end = data_offset
    for expert in range(EXPERTS):
        item = INDEX_ENTRY.unpack_from(index, expert * INDEX_ENTRY.size)
        item_layer, item_expert, item_bytes, item_offset = item
        if (
            item_layer != expected_layer
            or item_expert != expert
            or item_bytes <= 0
            or item_bytes % 4096
            or item_bytes > max_block_bytes
            or item_offset != previous_end
            or item_offset + item_bytes > file_bytes
        ):
            raise RuntimeError(
                f"invalid MZG2 index: {path} expert {expert}"
            )
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument(
        "--tool",
        type=Path,
        default=Path(__file__).resolve().with_name("transcode_mzg2_layer"),
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not args.tool.is_file():
        parser.error(f"transcoder not found: {args.tool}")
    if args.out.exists():
        parser.error(f"published output already exists: {args.out}")

    partial = args.out.with_name(args.out.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    logs = partial / "logs"
    logs.mkdir(exist_ok=True)
    progress_path = partial / "progress.json"
    progress = load_progress(progress_path)
    progress_lock = threading.Lock()
    layers = progress["layers"]
    assert isinstance(layers, dict)

    pending: list[int] = []
    for layer in range(1, LAYERS + 1):
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
                str(args.tool),
                str(args.model),
                str(temporary),
                str(layer),
                str(EXPERTS),
                MANIFEST_SHA256,
            ],
            log_path,
        )
        os.replace(temporary, final_path)
        record = {
            "bytes": final_path.stat().st_size,
            "sha256": sha256_file(final_path),
            "transcoded": True,
            "verified": False,
        }
        with progress_lock:
            layers[str(layer)] = record
            atomic_json(progress_path, progress)
        print(
            f"MZG2 transcode layer={layer}: PASS "
            f"bytes={record['bytes']}"
        )

    if pending:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.jobs, len(pending))
        ) as pool:
            futures = {pool.submit(transcode, layer): layer for layer in pending}
            for future in concurrent.futures.as_completed(futures):
                future.result()

    verification_pending = [
        layer
        for layer in range(1, LAYERS + 1)
        if not bool(layers[str(layer)].get("verified"))
    ]

    def verify(layer: int) -> None:
        sidecar = partial / f"layer-{layer:03d}.mzg2"
        log_path = logs / f"layer-{layer:03d}-verify.log"
        run_logged(
            [str(args.tool), "--verify", str(args.model), str(sidecar)],
            log_path,
        )
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
            max_workers=min(args.jobs, len(verification_pending))
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
    for layer in range(1, LAYERS + 1):
        path = partial / f"layer-{layer:03d}.mzg2"
        parsed = parse_sidecar(path, layer)
        record = layers[str(layer)]
        if not record.get("verified"):
            raise RuntimeError(f"unverified MZG2 layer {layer}")
        total_file_bytes += parsed["file_bytes"]
        total_block_bytes += parsed["block_bytes"]
        total_experts += parsed["experts"]
        max_block_bytes = max(max_block_bytes, parsed["max_block_bytes"])
        files.append(
            {
                "layer": layer,
                "path": path.name,
                "bytes": parsed["file_bytes"],
                "sha256": record["sha256"],
            }
        )
    raw_bytes = LAYERS * EXPERTS * EXPERT_BYTES
    manifest = {
        "schema": "moonshine-mzg2-full-v1",
        "version": 2,
        "source_manifest_sha256": MANIFEST_SHA256,
        "layers": LAYERS,
        "experts_per_layer": EXPERTS,
        "experts": total_experts,
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
    os.replace(partial, args.out)
    print(
        "MZG2 full store: PASS "
        f"experts={total_experts} raw={raw_bytes} stored={total_file_bytes} "
        f"reduction={manifest['reduction_pct']:.9f}% out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
