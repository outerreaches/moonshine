#!/usr/bin/env python3
"""Build the source-precision static tier for a standalone Kimi K3.1 MZG2 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import BinaryIO

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
    canonical_json,
    inspect_checkpoint,
)

STATIC_SCHEMA = "moonshine-k31-static-pack-v1"
BUNDLE_SCHEMA = "moonshine-k31-mzg2-bundle-v1"
MZG2_STORE_SCHEMA = "moonshine-k31-mzg2-full-v1"
BASE_MODEL = UPSTREAM_ID
BASE_REVISION = UPSTREAM_REVISION
CHUNK_BYTES = 16 * 1024 * 1024
AUXILIARY_FILES = (
    ("config.json", "config.json"),
    ("generation_config.json", "generation_config.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("tiktoken.model", "tiktoken.model"),
    ("encoding_k3.py", "encoding_k3.py"),
    ("LICENSE", "LICENSE"),
    ("model.safetensors.index.json", "model.safetensors.index.json"),
)
HEADER_ALIGNMENT = 4096


def align_up(value: int, alignment: int = HEADER_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_header(path: Path) -> tuple[int, dict[str, object]]:
    """Read a SafeTensors header, used for generated static-pack verification."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated SafeTensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or header_bytes > path.stat().st_size - 8:
            raise ValueError(f"invalid SafeTensors header length: {path}")
        encoded = stream.read(header_bytes)
        if len(encoded) != header_bytes:
            raise ValueError(f"truncated SafeTensors header: {path}")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError(f"SafeTensors header is not an object: {path}")
    return 8 + header_bytes, value


def _integer(value: object) -> bool:
    return type(value) is int


def _valid_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def validate_source_manifest(source: object) -> None:
    """Validate the inspector contract consumed by all bundle stages."""
    if not isinstance(source, dict) or source.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("source is not a K3.1 checkpoint inspection manifest")
    upstream = source.get("upstream")
    if upstream != {
        "id": UPSTREAM_ID,
        "revision": UPSTREAM_REVISION,
        "architecture": ARCHITECTURE,
        "model_type": MODEL_TYPE,
    }:
        raise ValueError("source checkpoint model identity mismatch")
    if source.get("verification") not in {"metadata-only", "full-sha256"}:
        raise ValueError("invalid source checkpoint verification mode")
    if not _valid_digest(source.get("source_layout_sha256")):
        raise ValueError("invalid source layout SHA-256")
    layout_crc = source.get("source_model_layout_crc64")
    if not isinstance(layout_crc, str) or len(layout_crc) != 16:
        raise ValueError("invalid source model layout CRC64")
    try:
        if int(layout_crc, 16) == 0 or layout_crc != layout_crc.lower():
            raise ValueError("invalid source model layout CRC64")
    except ValueError as error:
        raise ValueError("invalid source model layout CRC64") from error

    integer_fields = (
        "tensor_count", "total_bytes", "file_bytes",
        "static_tensor_count", "static_bytes",
    )
    if any(not _integer(source.get(name)) or int(source[name]) < 0 for name in integer_fields):
        raise ValueError("invalid source checkpoint inventory counts")

    raw_shards = source.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("source checkpoint has no shard metadata")
    shards: dict[str, dict[str, object]] = {}
    for record in raw_shards:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("invalid source shard record")
        name = str(record["name"])
        if name in shards or Path(name).name != name:
            raise ValueError("duplicate or unsafe source shard name")
        if any(
            not _integer(record.get(field)) or int(record[field]) < 0
            for field in ("bytes", "header_bytes", "data_offset", "tensor_count")
        ):
            raise ValueError("invalid source shard metadata")
        if (
            int(record["bytes"]) <= 0
            or int(record["data_offset"]) != int(record["header_bytes"]) + 8
            or int(record["data_offset"]) > int(record["bytes"])
        ):
            raise ValueError("invalid source shard offsets")
        shards[name] = record
    if (
        sum(int(record["tensor_count"]) for record in raw_shards) != source["tensor_count"]
        or sum(int(record["bytes"]) for record in raw_shards) != source["file_bytes"]
    ):
        raise ValueError("source shard totals do not match checkpoint inventory")

    geometry = source.get("expert_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("missing K3.1 expert geometry")
    geometry_fields = (
        "routed_layers", "experts_per_layer", "expert_count",
        "tensors_per_expert", "bytes_per_expert", "routed_tensor_count",
        "routed_bytes",
    )
    if any(
        not _integer(geometry.get(field)) or int(geometry[field]) < 0
        for field in geometry_fields
    ):
        raise ValueError("invalid K3.1 expert geometry")
    if geometry["bytes_per_expert"] != EXPERT_SPAN_BYTES:
        raise ValueError(
            f"expert span mismatch: {geometry['bytes_per_expert']} != {EXPERT_SPAN_BYTES}"
        )
    experts = int(geometry["routed_layers"]) * int(geometry["experts_per_layer"])
    if (
        geometry["expert_count"] != experts
        or geometry["tensors_per_expert"] != len(EXPERT_LAYOUT)
        or geometry["routed_tensor_count"] != experts * len(EXPERT_LAYOUT)
        or geometry["routed_bytes"] != experts * EXPERT_SPAN_BYTES
    ):
        raise ValueError("inconsistent K3.1 routed expert inventory")
    layout = geometry.get("tensor_layout")
    if not isinstance(layout, list) or len(layout) != len(EXPERT_LAYOUT):
        raise ValueError("invalid K3.1 expert tensor layout")
    offset = 0
    for record, (expected_name, expected_bytes) in zip(layout, EXPERT_LAYOUT):
        if (
            not isinstance(record, dict)
            or record.get("name") != expected_name
            or record.get("dtype") != "U8"
            or record.get("offset") != offset
            or record.get("bytes") != expected_bytes
        ):
            raise ValueError("K3.1 expert tensor layout mismatch")
        offset += expected_bytes
    if offset != EXPERT_SPAN_BYTES:
        raise ValueError("K3.1 expert layout does not fill the exact expert span")

    static = source.get("static_tensors")
    if not isinstance(static, list) or len(static) != source["static_tensor_count"]:
        raise ValueError("static tensor inventory count mismatch")
    seen: set[str] = set()
    static_bytes = 0
    for tensor in static:
        if not isinstance(tensor, dict):
            raise ValueError("invalid static tensor record")
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        size = tensor.get("bytes")
        shard_name = tensor.get("source_shard")
        offsets = tensor.get("data_offsets")
        source_offset = tensor.get("source_offset")
        if (
            not isinstance(name, str)
            or not name.startswith("language_model.")
            or name in seen
            or dtype not in {"BF16", "F32"}
            or not isinstance(shape, list)
            or not shape
            or any(not _integer(value) or value <= 0 for value in shape)
            or not _integer(size)
            or int(size) <= 0
            or not isinstance(shard_name, str)
            or shard_name not in shards
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not _integer(value) for value in offsets)
            or not _integer(source_offset)
        ):
            raise ValueError("invalid K3.1 static tensor inventory")
        start, end = offsets
        shard = shards[shard_name]
        if (
            start < 0
            or end != start + int(size)
            or source_offset != int(shard["data_offset"]) + start
            or int(source_offset) + int(size) > int(shard["bytes"])
        ):
            raise ValueError(f"invalid static tensor source span: {name}")
        elements = 1
        for dimension in shape:
            elements *= int(dimension)
        if elements * (2 if dtype == "BF16" else 4) != size:
            raise ValueError(f"static tensor shape/size mismatch: {name}")
        seen.add(name)
        static_bytes += int(size)
    if static_bytes != source["static_bytes"]:
        raise ValueError("static tensor inventory byte count mismatch")
    if (
        int(source["static_tensor_count"]) + int(geometry["routed_tensor_count"])
            > int(source["tensor_count"])
        or int(source["static_bytes"]) + int(geometry["routed_bytes"])
            > int(source["total_bytes"])
    ):
        raise ValueError("checkpoint inventory is smaller than its static and routed tiers")


def inspect_source(
    model: Path,
    *,
    verify_sha256: bool = False,
    expected_shards: int = OFFICIAL_SHARDS,
    expected_routed_layers: int = OFFICIAL_ROUTED_LAYERS,
    expected_experts_per_layer: int = OFFICIAL_EXPERTS_PER_LAYER,
) -> dict[str, object]:
    source = inspect_checkpoint(
        model,
        verify_sha256=verify_sha256,
        expected_shards=expected_shards,
        expected_routed_layers=expected_routed_layers,
        expected_experts_per_layer=expected_experts_per_layer,
    )
    validate_source_manifest(source)
    return source


def static_tensors_from_source(
    model: Path,
    source: dict[str, object],
) -> list[dict[str, object]]:
    validate_source_manifest(source)
    tensors: list[dict[str, object]] = []
    for record in source["static_tensors"]:
        assert isinstance(record, dict)
        item = dict(record)
        item["source_path"] = str(model / str(record["source_shard"]))
        tensors.append(item)
    if not tensors:
        raise ValueError("empty static tensor inventory")
    if sum(int(item["bytes"]) for item in tensors) != source["static_bytes"]:
        raise ValueError("unexpected static payload bytes")
    return tensors


def make_static_header(
    tensors: list[dict[str, object]],
) -> tuple[bytes, list[dict[str, object]]]:
    metadata: dict[str, object] = {}
    cursor = 0
    planned: list[dict[str, object]] = []
    for source in tensors:
        size = int(source["bytes"])
        metadata[str(source["name"])] = {
            "dtype": source["dtype"],
            "shape": source["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        planned.append({**source, "output_offset": cursor})
        cursor += size
    encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    header_bytes = align_up(8 + len(encoded), HEADER_ALIGNMENT) - 8
    encoded += b" " * (header_bytes - len(encoded))
    return struct.pack("<Q", header_bytes) + encoded, planned


def copy_tensor(
    source: BinaryIO,
    destination: BinaryIO,
    source_offset: int,
    destination_offset: int,
    size: int,
) -> str:
    source.seek(source_offset)
    destination.seek(destination_offset)
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = source.read(min(CHUNK_BYTES, remaining))
        if not chunk:
            raise IOError("short source tensor read")
        destination.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def verify_tensor(
    source_path: Path,
    source_offset: int,
    output: BinaryIO,
    output_offset: int,
    size: int,
    expected_sha256: str | None = None,
) -> str:
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    with source_path.open("rb") as source:
        source.seek(source_offset)
        output.seek(output_offset)
        remaining = size
        while remaining:
            request = min(CHUNK_BYTES, remaining)
            left = source.read(request)
            right = output.read(request)
            if len(left) != request or len(right) != request or left != right:
                raise ValueError(f"static tensor verification failed: {source_path}")
            source_digest.update(left)
            output_digest.update(right)
            remaining -= request
    digest = output_digest.hexdigest()
    if digest != source_digest.hexdigest() or (
        expected_sha256 is not None and digest != expected_sha256
    ):
        raise ValueError("static tensor SHA-256 mismatch")
    return digest


def _progress_identity(
    source: dict[str, object],
    planned: list[dict[str, object]],
) -> str:
    records = [
        {
            key: item[key]
            for key in (
                "name", "dtype", "shape", "bytes", "source_shard",
                "source_offset", "output_offset",
            )
        }
        for item in planned
    ]
    return hashlib.sha256(canonical_json({
        "source_layout_sha256": source["source_layout_sha256"],
        "static_tensors": records,
    }).encode("utf-8")).hexdigest()


def _source_fields(source: dict[str, object]) -> dict[str, object]:
    return {
        "source_layout_sha256": source["source_layout_sha256"],
        "source_manifest_sha256": source["source_layout_sha256"],
        "source_model_layout_crc64": source["source_model_layout_crc64"],
        "source_checkpoint_schema": source["schema"],
        "source_verification": source["verification"],
        "source_tensor_count": source["tensor_count"],
        "source_total_bytes": source["total_bytes"],
        "source_file_bytes": source["file_bytes"],
        "source_shard_count": len(source["shards"]),
        "source_shards": source["shards"],
    }


def build_static_pack(
    model: Path,
    output: Path,
    *,
    source_manifest: dict[str, object] | None = None,
    expected_shards: int = OFFICIAL_SHARDS,
    expected_routed_layers: int = OFFICIAL_ROUTED_LAYERS,
    expected_experts_per_layer: int = OFFICIAL_EXPERTS_PER_LAYER,
    verify_source_hashes: bool = False,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"published static pack already exists: {output}")
    source = source_manifest if source_manifest is not None else inspect_source(
        model,
        verify_sha256=verify_source_hashes,
        expected_shards=expected_shards,
        expected_routed_layers=expected_routed_layers,
        expected_experts_per_layer=expected_experts_per_layer,
    )
    tensors = static_tensors_from_source(model, source)
    header, planned = make_static_header(tensors)
    partial = output.with_name(output.name + ".partial")
    progress_path = output.with_name(output.name + ".progress.json")
    identity = _progress_identity(source, planned)
    completed: list[dict[str, object]] = []
    if partial.exists() or progress_path.exists():
        if not partial.exists() or not progress_path.exists():
            raise ValueError("incomplete static-pack resume pair")
        progress = json.loads(progress_path.read_text())
        if (
            progress.get("schema") != STATIC_SCHEMA
            or progress.get("identity") != identity
            or progress.get("source_layout_sha256") != source["source_layout_sha256"]
        ):
            raise ValueError("static-pack progress identity mismatch")
        raw_completed = progress.get("completed")
        if not isinstance(raw_completed, list) or len(raw_completed) > len(planned):
            raise ValueError("invalid static-pack completed inventory")
        completed = raw_completed
        with partial.open("rb") as output_stream:
            if output_stream.read(len(header)) != header:
                raise ValueError("static-pack partial header mismatch")
            for index, record in enumerate(completed):
                item = planned[index]
                if not isinstance(record, dict) or not _valid_digest(record.get("sha256")):
                    raise ValueError("invalid static-pack completed tensor record")
                digest = verify_tensor(
                    Path(str(item["source_path"])),
                    int(item["source_offset"]),
                    output_stream,
                    len(header) + int(item["output_offset"]),
                    int(item["bytes"]),
                    str(record["sha256"]),
                )
                if record.get("name") != item["name"] or digest != record["sha256"]:
                    raise ValueError("static-pack completed tensor mismatch")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("wb") as stream:
            stream.write(header)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(progress_path, {
            "schema": STATIC_SCHEMA,
            "identity": identity,
            "source_layout_sha256": source["source_layout_sha256"],
            "completed": [],
        })
    with partial.open("r+b", buffering=0) as destination:
        for index in range(len(completed), len(planned)):
            item = planned[index]
            with Path(str(item["source_path"])).open("rb", buffering=0) as source_stream:
                digest = copy_tensor(
                    source_stream,
                    destination,
                    int(item["source_offset"]),
                    len(header) + int(item["output_offset"]),
                    int(item["bytes"]),
                )
            completed.append({"name": item["name"], "sha256": digest})
            if len(completed) % 16 == 0 or len(completed) == len(planned):
                destination.flush()
                os.fsync(destination.fileno())
                atomic_json(progress_path, {
                    "schema": STATIC_SCHEMA,
                    "identity": identity,
                    "source_layout_sha256": source["source_layout_sha256"],
                    "completed": completed,
                })
        destination.flush()
        os.fsync(destination.fileno())
    if len(completed) != len(planned):
        raise ValueError("static-pack conversion did not complete")
    with partial.open("rb") as output_stream:
        for index, item in enumerate(planned):
            record = completed[index]
            if not isinstance(record, dict):
                raise ValueError("invalid completed static tensor")
            verify_tensor(
                Path(str(item["source_path"])),
                int(item["source_offset"]),
                output_stream,
                len(header) + int(item["output_offset"]),
                int(item["bytes"]),
                str(record["sha256"]),
            )
    expected_file_bytes = len(header) + int(source["static_bytes"])
    if partial.stat().st_size != expected_file_bytes:
        raise ValueError("static-pack file size mismatch")
    file_sha256 = sha256_file(partial)
    os.replace(partial, output)
    progress_path.unlink()
    manifest = {
        "schema": STATIC_SCHEMA,
        "version": 1,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        **_source_fields(source),
        "path": output.name,
        "tensor_count": source["static_tensor_count"],
        "payload_bytes": source["static_bytes"],
        "file_bytes": expected_file_bytes,
        "sha256": file_sha256,
        "header_bytes": len(header),
        "tensors": completed,
    }
    atomic_json(output.with_name("static-store.json"), manifest)
    return manifest


def verify_static_pack(
    model: Path,
    output: Path,
    manifest_path: Path,
    *,
    source_manifest: dict[str, object] | None = None,
    expected_shards: int = OFFICIAL_SHARDS,
    expected_routed_layers: int = OFFICIAL_ROUTED_LAYERS,
    expected_experts_per_layer: int = OFFICIAL_EXPERTS_PER_LAYER,
    verify_source_hashes: bool = False,
) -> dict[str, object]:
    source = source_manifest if source_manifest is not None else inspect_source(
        model,
        verify_sha256=verify_source_hashes,
        expected_shards=expected_shards,
        expected_routed_layers=expected_routed_layers,
        expected_experts_per_layer=expected_experts_per_layer,
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != STATIC_SCHEMA
        or manifest.get("version") != 1
        or manifest.get("base_model") != BASE_MODEL
        or manifest.get("base_revision") != BASE_REVISION
    ):
        raise ValueError("invalid K3.1 static-store manifest")
    if manifest.get("source_layout_sha256") != source["source_layout_sha256"]:
        raise ValueError("static-store source layout mismatch")
    tensors = static_tensors_from_source(model, source)
    if (
        manifest.get("tensor_count") != source["static_tensor_count"]
        or manifest.get("payload_bytes") != source["static_bytes"]
    ):
        raise ValueError("static-store manifest inventory mismatch")
    header, planned = make_static_header(tensors)
    records = manifest.get("tensors")
    if not isinstance(records, list) or len(records) != len(planned):
        raise ValueError("static-store manifest tensor count mismatch")
    with output.open("rb") as stream:
        if stream.read(len(header)) != header:
            raise ValueError("static-store header mismatch")
        for index, item in enumerate(planned):
            record = records[index]
            if not isinstance(record, dict) or record.get("name") != item["name"]:
                raise ValueError("static-store manifest order mismatch")
            verify_tensor(
                Path(str(item["source_path"])),
                int(item["source_offset"]),
                stream,
                len(header) + int(item["output_offset"]),
                int(item["bytes"]),
                str(record.get("sha256")),
            )
    if (
        output.stat().st_size != manifest.get("file_bytes")
        or manifest.get("file_bytes") != len(header) + int(source["static_bytes"])
    ):
        raise ValueError("static-store verified size mismatch")
    if sha256_file(output) != manifest.get("sha256"):
        raise ValueError("static-store file SHA-256 mismatch")
    return manifest


def validate_mzg2_store(
    root: Path,
    *,
    source_layout_sha256: str,
    expected_layers: int,
    expected_experts_per_layer: int,
    verify_hashes: bool = True,
) -> dict[str, object]:
    manifest = json.loads((root / "expert-store.json").read_text())
    expected_experts = expected_layers * expected_experts_per_layer
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != MZG2_STORE_SCHEMA
        or manifest.get("version") != 2
        or manifest.get("base_model") != BASE_MODEL
        or manifest.get("base_revision") != BASE_REVISION
        or manifest.get("source_layout_sha256") != source_layout_sha256
        or manifest.get("layers") != expected_layers
        or manifest.get("experts_per_layer") != expected_experts_per_layer
        or manifest.get("experts") != expected_experts
        or manifest.get("expert_span_bytes") != EXPERT_SPAN_BYTES
        or manifest.get("raw_bytes") != expected_experts * EXPERT_SPAN_BYTES
        or manifest.get("tile_bytes") != 16_384
    ):
        raise ValueError("invalid or stale K3.1 MZG2 full-store manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != expected_layers:
        raise ValueError("invalid MZG2 sidecar inventory")
    seen: set[int] = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("invalid MZG2 sidecar record")
        layer = record.get("layer")
        relative = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (
            not _integer(layer)
            or not isinstance(layer, int)
            or layer < 1
            or layer > expected_layers
            or layer in seen
            or not isinstance(relative, str)
            or relative != f"layer-{layer:03d}.mzg2"
            or not _integer(size)
            or not isinstance(size, int)
            or size <= 0
            or not _valid_digest(digest)
        ):
            raise ValueError("invalid MZG2 sidecar record")
        path = root / relative
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"MZG2 sidecar size mismatch: {relative}")
        if verify_hashes and sha256_file(path) != digest:
            raise ValueError(f"MZG2 sidecar SHA-256 mismatch: {relative}")
        seen.add(layer)
    if seen != set(range(1, expected_layers + 1)):
        raise ValueError("incomplete MZG2 layer coverage")
    return manifest


def link_mzg2_store(source: Path, destination: Path, layers: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "expert-store.json", destination / "expert-store.json")
    fsync_file(destination / "expert-store.json")
    for layer in range(1, layers + 1):
        source_file = source / f"layer-{layer:03d}.mzg2"
        destination_file = destination / source_file.name
        if destination_file.exists():
            if (
                destination_file.stat().st_ino != source_file.stat().st_ino
                or destination_file.stat().st_dev != source_file.stat().st_dev
            ):
                raise ValueError(f"existing MZG2 link differs: {destination_file}")
            continue
        os.link(source_file, destination_file)


def copy_auxiliary_files(model: Path, output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for source_name, destination_name in AUXILIARY_FILES:
        source = model / source_name
        if not source.is_file():
            raise ValueError(f"missing bundle auxiliary file: {source_name}")
        destination = output / destination_name
        shutil.copy2(source, destination)
        fsync_file(destination)
        result[destination_name] = sha256_file(destination)
    notice = Path(__file__).resolve().parent.parent / "NOTICE"
    if not notice.is_file():
        raise ValueError("missing Moonshine NOTICE")
    shutil.copy2(notice, output / "NOTICE")
    fsync_file(output / "NOTICE")
    result["NOTICE"] = sha256_file(output / "NOTICE")
    return result


def write_bundle_manifest(
    output: Path,
    static_manifest: dict[str, object],
    mzg2_manifest: dict[str, object],
    auxiliary: dict[str, str],
    source_manifest: dict[str, object],
) -> dict[str, object]:
    validate_source_manifest(source_manifest)
    geometry = source_manifest["expert_geometry"]
    assert isinstance(geometry, dict)
    if (
        static_manifest.get("source_layout_sha256")
            != source_manifest["source_layout_sha256"]
        or mzg2_manifest.get("source_layout_sha256")
            != source_manifest["source_layout_sha256"]
    ):
        raise ValueError("bundle tiers have different source checkpoint identities")
    routed_manifest_path = output / "expert-store-mzg2" / "expert-store.json"
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "version": 1,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "license": "kimi-k3.1",
        **_source_fields(source_manifest),
        "static_store": {
            key: static_manifest[key]
            for key in (
                "path", "tensor_count", "payload_bytes", "file_bytes", "sha256",
            )
        },
        "routed_store": {
            "format": "mzg2",
            "path": "expert-store-mzg2",
            "layers": mzg2_manifest["layers"],
            "experts_per_layer": mzg2_manifest["experts_per_layer"],
            "experts": mzg2_manifest["experts"],
            "expert_span_bytes": mzg2_manifest["expert_span_bytes"],
            "tile_bytes": mzg2_manifest["tile_bytes"],
            "manifest_sha256": sha256_file(routed_manifest_path),
        },
        "expert_geometry": geometry,
        "expert_layout": {
            "bytes": geometry["bytes_per_expert"],
            "tensors": [
                {
                    "name": record["name"],
                    "offset": record["offset"],
                    "bytes": record["bytes"],
                }
                for record in geometry["tensor_layout"]
            ],
        },
        "auxiliary_sha256": auxiliary,
    }
    atomic_json(output / "moonshine-bundle.json", manifest)
    return manifest


def build_bundle(
    model: Path,
    output: Path,
    *,
    jobs: int,
    reuse_mzg2: Path | None,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"published output already exists: {output}")
    native_transcoder = Path(__file__).resolve().with_name("transcode_mzg2_layer")
    if reuse_mzg2 is None and not native_transcoder.is_file():
        raise ValueError(
            f"native transcoder is missing: {native_transcoder}; "
            "run `make tools/transcode_mzg2_layer` before bundle conversion"
        )
    source = inspect_source(model, verify_sha256=True)
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    static_path = partial / "model-static.safetensors"
    if static_path.exists():
        static_manifest = verify_static_pack(
            model,
            static_path,
            partial / "static-store.json",
            source_manifest=source,
        )
    else:
        static_manifest = build_static_pack(model, static_path, source_manifest=source)
    mzg2_path = partial / "expert-store-mzg2"
    geometry = source["expert_geometry"]
    assert isinstance(geometry, dict)
    if reuse_mzg2 is not None:
        validate_mzg2_store(
            reuse_mzg2,
            source_layout_sha256=str(source["source_layout_sha256"]),
            expected_layers=int(geometry["routed_layers"]),
            expected_experts_per_layer=int(geometry["experts_per_layer"]),
            verify_hashes=False,
        )
        link_mzg2_store(reuse_mzg2, mzg2_path, int(geometry["routed_layers"]))
    elif not mzg2_path.exists():
        tool = Path(__file__).resolve().with_name("transcode_mzg2_full.py")
        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--model",
                str(model),
                "--out",
                str(mzg2_path),
                "--jobs",
                str(jobs),
            ],
            check=True,
        )
    mzg2_manifest = validate_mzg2_store(
        mzg2_path,
        source_layout_sha256=str(source["source_layout_sha256"]),
        expected_layers=int(geometry["routed_layers"]),
        expected_experts_per_layer=int(geometry["experts_per_layer"]),
        verify_hashes=True,
    )
    auxiliary = copy_auxiliary_files(model, partial)
    manifest = write_bundle_manifest(
        partial, static_manifest, mzg2_manifest, auxiliary, source
    )
    fsync_file(partial / "moonshine-bundle.json")
    fsync_directory(partial / "expert-store-mzg2")
    fsync_directory(partial)
    fsync_directory(partial.parent)
    os.replace(partial, output)
    fsync_directory(output.parent)
    return manifest


def verify_bundle(
    root: Path,
    *,
    model: Path | None,
    verify_hashes: bool,
    source_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = json.loads((root / "moonshine-bundle.json").read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("version") != 1
        or manifest.get("base_model") != BASE_MODEL
        or manifest.get("base_revision") != BASE_REVISION
        or not _valid_digest(manifest.get("source_layout_sha256"))
    ):
        raise ValueError("invalid K3.1 standalone bundle manifest")
    static = manifest.get("static_store")
    if not isinstance(static, dict):
        raise ValueError("missing static store manifest")
    static_path = root / str(static.get("path"))
    if static_path.stat().st_size != static.get("file_bytes"):
        raise ValueError("bundle static file size mismatch")
    if verify_hashes and sha256_file(static_path) != static.get("sha256"):
        raise ValueError("bundle static file SHA-256 mismatch")
    static_manifest = json.loads((root / "static-store.json").read_text())
    if (
        not isinstance(static_manifest, dict)
        or static_manifest.get("schema") != STATIC_SCHEMA
        or static_manifest.get("source_layout_sha256")
            != manifest["source_layout_sha256"]
        or any(static_manifest.get(key) != static.get(key) for key in (
            "path", "tensor_count", "payload_bytes", "file_bytes", "sha256",
        ))
    ):
        raise ValueError("bundle and static-store manifests disagree")

    source = source_manifest
    if model is not None:
        source = source if source is not None else inspect_source(model)
        if source["source_layout_sha256"] != manifest["source_layout_sha256"]:
            raise ValueError("bundle source checkpoint mismatch")
        verify_static_pack(
            model,
            static_path,
            root / "static-store.json",
            source_manifest=source,
        )
    elif source_manifest is not None:
        raise ValueError("a source manifest requires the source model path")

    routed = manifest.get("routed_store")
    if not isinstance(routed, dict):
        raise ValueError("missing routed store manifest")
    geometry = source["expert_geometry"] if source is not None else routed
    assert isinstance(geometry, dict)
    expected_layers = geometry.get("routed_layers", geometry.get("layers"))
    expected_experts = geometry.get("experts_per_layer")
    if not _integer(expected_layers) or not _integer(expected_experts):
        raise ValueError("invalid bundle routed geometry")
    if source is not None and (
        routed.get("layers") != expected_layers
        or routed.get("experts_per_layer") != expected_experts
    ):
        raise ValueError("bundle routed geometry differs from source checkpoint")
    mzg2_root = root / str(routed.get("path"))
    mzg2 = validate_mzg2_store(
        mzg2_root,
        source_layout_sha256=str(manifest["source_layout_sha256"]),
        expected_layers=int(expected_layers),
        expected_experts_per_layer=int(expected_experts),
        verify_hashes=verify_hashes,
    )
    if sha256_file(mzg2_root / "expert-store.json") != routed.get("manifest_sha256"):
        raise ValueError("bundle MZG2 manifest SHA-256 mismatch")
    auxiliary = manifest.get("auxiliary_sha256")
    if not isinstance(auxiliary, dict):
        raise ValueError("invalid bundle auxiliary inventory")
    for name, digest in auxiliary.items():
        path = root / name
        if not path.is_file() or (verify_hashes and sha256_file(path) != digest):
            raise ValueError(f"bundle auxiliary mismatch: {name}")
    return {"bundle": manifest, "mzg2": mzg2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--model", required=True, type=Path)
    inspect.add_argument("--verify-sha256", action="store_true")
    build = subparsers.add_parser("build-static")
    build.add_argument("--model", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    verify = subparsers.add_parser("verify-static")
    verify.add_argument("--model", required=True, type=Path)
    verify.add_argument("--static", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    full = subparsers.add_parser("build")
    full.add_argument("--model", required=True, type=Path)
    full.add_argument("--out", required=True, type=Path)
    full.add_argument("--jobs", type=int, default=24)
    full.add_argument("--reuse-mzg2", type=Path)
    check = subparsers.add_parser("verify")
    check.add_argument("--bundle", required=True, type=Path)
    check.add_argument("--model", type=Path)
    check.add_argument("--skip-large-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inspect":
        source = inspect_source(args.model, verify_sha256=args.verify_sha256)
        print(json.dumps({
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "source_layout_sha256": source["source_layout_sha256"],
            "source_tensor_count": source["tensor_count"],
            "source_total_bytes": source["total_bytes"],
            "source_shard_count": len(source["shards"]),
            "tensor_count": source["static_tensor_count"],
            "payload_bytes": source["static_bytes"],
            "expert_geometry": source["expert_geometry"],
        }, sort_keys=True))
        return 0
    if args.command == "build-static":
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = build_static_pack(
            args.model, args.out / "model-static.safetensors"
        )
        print(json.dumps({key: manifest[key] for key in (
            "tensor_count", "payload_bytes", "file_bytes", "sha256",
            "source_layout_sha256",
        )}, sort_keys=True))
        return 0
    if args.command == "verify-static":
        manifest = verify_static_pack(args.model, args.static, args.manifest)
        print(json.dumps(
            {"decision": "PASS", "sha256": manifest["sha256"]},
            sort_keys=True,
        ))
        return 0
    if args.command == "build":
        manifest = build_bundle(
            args.model,
            args.out,
            jobs=args.jobs,
            reuse_mzg2=args.reuse_mzg2,
        )
        print(json.dumps({
            "decision": "PASS",
            "schema": manifest["schema"],
            "out": str(args.out),
        }, sort_keys=True))
        return 0
    result = verify_bundle(
        args.bundle,
        model=args.model,
        verify_hashes=not args.skip_large_hashes,
    )
    print(json.dumps({
        "decision": "PASS",
        "schema": result["bundle"]["schema"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
