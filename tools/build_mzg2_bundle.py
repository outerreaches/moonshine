#!/usr/bin/env python3
"""Build the source-precision static tier for a standalone Moonshine MZG2 bundle."""

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

PINNED_SHARDS = 96
PINNED_TENSORS = 2_460
PINNED_PAYLOAD_BYTES = 113_509_540_864
PINNED_SOURCE_MANIFEST_SHA256 = (
    "476fa0ba64e3233cbb9ca0642327361a73f6807e751edb071c92fa2216b202a4"
)
STATIC_SCHEMA = "moonshine-k3-static-pack-v1"
DTYPE_CODE = {"BF16": 1, "F32": 2, "U8": 3}
CRC64_POLYNOMIAL = 0x42F0E1EBA9EA3693
BUNDLE_SCHEMA = "moonshine-k3-mzg2-bundle-v1"
BASE_MODEL = "moonshotai/Kimi-K3"
BASE_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
CHUNK_BYTES = 16 * 1024 * 1024
AUXILIARY_FILES = (
    ("config.json", "config.json"),
    ("generation_config.json", "generation_config.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("tiktoken.model", "tiktoken.model"),
    ("encoding_k3.py", "encoding_k3.py"),
    ("LICENSE", "LICENSE"),
    ("manifest.tsv", "source-manifest.tsv"),
)
EXPERT_LAYOUT = (
    ("w1.weight_packed", 0, 5_505_024),
    ("w1.weight_scale", 5_505_024, 344_064),
    ("w2.weight_packed", 5_849_088, 5_505_024),
    ("w2.weight_scale", 11_354_112, 344_064),
    ("w3.weight_packed", 11_698_176, 5_505_024),
    ("w3.weight_scale", 17_203_200, 344_064),
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



def crc64_table() -> list[int]:
    mask = (1 << 64) - 1
    table: list[int] = []
    for value in range(256):
        crc = value << 56
        for _ in range(8):
            crc = (
                ((crc << 1) & mask) ^ CRC64_POLYNOMIAL
                if crc & (1 << 63)
                else (crc << 1) & mask
            )
        table.append(crc)
    return table


CRC64_TABLE = crc64_table()


def crc64_update(crc: int, data: bytes) -> int:
    mask = (1 << 64) - 1
    for byte in data:
        crc = CRC64_TABLE[((crc >> 56) ^ byte) & 0xFF] ^ ((crc << 8) & mask)
    return crc


def crc64_integer(crc: int, value: int, byte_count: int) -> int:
    return crc64_update(crc, value.to_bytes(byte_count, "little"))


def source_model_layout_crc64(model: Path) -> int:
    shard_metadata: list[tuple[int, int]] = []
    tensors: list[tuple[str, int, int, int, int, int, list[int]]] = []
    for shard_index, shard in enumerate(shard_paths(model)):
        data_offset, header = load_header(shard)
        shard_metadata.append((shard.stat().st_size, data_offset))
        for name, raw_meta in header.items():
            if name == "__metadata__" or not isinstance(raw_meta, dict):
                continue
            offsets = raw_meta.get("data_offsets")
            shape = raw_meta.get("shape")
            dtype = raw_meta.get("dtype")
            size = tensor_bytes(raw_meta)
            if (
                not isinstance(offsets, list)
                or not isinstance(shape, list)
                or dtype not in DTYPE_CODE
            ):
                raise ValueError(f"invalid source tensor metadata: {name}")
            tensors.append((
                name,
                data_offset + offsets[0],
                size,
                shard_index,
                len(shape),
                DTYPE_CODE[dtype],
                shape,
            ))
    tensors.sort(key=lambda item: item[0])
    crc = crc64_integer(0, len(shard_metadata), 8)
    for file_bytes, data_offset in shard_metadata:
        crc = crc64_integer(crc, file_bytes, 8)
        crc = crc64_integer(crc, data_offset, 8)
    crc = crc64_integer(crc, len(tensors), 8)
    for name, physical_offset, size, shard, ndim, dtype, shape in tensors:
        encoded = name.encode()
        crc = crc64_integer(crc, len(encoded), 8)
        crc = crc64_update(crc, encoded)
        crc = crc64_integer(crc, physical_offset, 8)
        crc = crc64_integer(crc, size, 8)
        crc = crc64_integer(crc, shard, 2)
        crc = crc64_integer(crc, ndim, 4)
        crc = crc64_integer(crc, dtype, 4)
        for dimension in shape:
            crc = crc64_integer(crc, int(dimension), 8)
    return crc

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def shard_paths(model: Path) -> list[Path]:
    return sorted(model.glob("model-*.safetensors"))


def load_header(path: Path) -> tuple[int, dict[str, object]]:
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


def tensor_bytes(meta: dict[str, object]) -> int:
    offsets = meta.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(item, int) for item in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise ValueError("invalid SafeTensors data_offsets")
    return offsets[1] - offsets[0]


def is_required_tensor(name: str) -> bool:
    return name.startswith("language_model.") and \
        ".block_sparse_moe.experts." not in name


def discover_static_tensors(model: Path) -> list[dict[str, object]]:
    tensors: list[dict[str, object]] = []
    seen: set[str] = set()
    for shard_index, shard in enumerate(shard_paths(model)):
        data_offset, header = load_header(shard)
        file_bytes = shard.stat().st_size
        for name, raw_meta in header.items():
            if name == "__metadata__" or not is_required_tensor(name):
                continue
            if name in seen or not isinstance(raw_meta, dict):
                raise ValueError(f"duplicate/invalid required tensor: {name}")
            meta = dict(raw_meta)
            offsets = meta.get("data_offsets")
            size = tensor_bytes(meta)
            assert isinstance(offsets, list)
            physical_offset = data_offset + offsets[0]
            if physical_offset + size > file_bytes:
                raise ValueError(f"tensor exceeds source shard: {name}")
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            if (
                dtype not in {"BF16", "F32"}
                or not isinstance(shape, list)
                or not shape
                or not all(
                    isinstance(value, int) and value > 0 for value in shape
                )
            ):
                raise ValueError(f"unsupported required tensor: {name}")
            elements = 1
            for dimension in shape:
                elements *= dimension
            scalar_bytes = 2 if dtype == "BF16" else 4
            if elements * scalar_bytes != size:
                raise ValueError(
                    f"required tensor shape/size mismatch: {name}"
                )
            tensors.append({
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "bytes": size,
                "source_shard": shard_index,
                "source_path": str(shard),
                "source_offset": physical_offset,
            })
            seen.add(name)
    tensors.sort(key=lambda item: str(item["name"]))
    return tensors


def make_static_header(
    tensors: list[dict[str, object]],
) -> tuple[bytes, list[dict[str, object]]]:
    metadata: dict[str, object] = {}
    cursor = 0
    planned: list[dict[str, object]] = []
    for source in tensors:
        size = int(source["bytes"])
        entry = {
            "dtype": source["dtype"],
            "shape": source["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        metadata[str(source["name"])] = entry
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
                raise ValueError(
                    f"static tensor verification failed: {source_path}"
                )
            source_digest.update(left)
            output_digest.update(right)
            remaining -= request
    digest = output_digest.hexdigest()
    if digest != source_digest.hexdigest() or (
        expected_sha256 is not None and digest != expected_sha256
    ):
        raise ValueError("static tensor SHA-256 mismatch")
    return digest


def validate_source_manifest(
    model: Path,
    expected_sha256: str | None,
    *,
    verify_shard_hashes: bool = False,
) -> str:
    if expected_sha256 is None:
        return "synthetic"
    manifest_path = model / "manifest.tsv"
    digest = sha256_file(manifest_path)
    if digest != expected_sha256:
        raise ValueError("source manifest SHA-256 mismatch")
    entries: dict[str, tuple[int, str]] = {}
    for line in manifest_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[1].isdigit() or len(parts[2]) != 64:
            raise ValueError("invalid source manifest entry")
        entries[parts[0]] = (int(parts[1]), parts[2])
    paths = shard_paths(model)
    if len(entries) != len(paths):
        raise ValueError("source manifest shard count mismatch")
    for path in paths:
        entry = entries.get(path.name)
        if entry is None or entry[0] != path.stat().st_size:
            raise ValueError(
                f"source shard size/manifest mismatch: {path.name}"
            )
        if verify_shard_hashes and sha256_file(path) != entry[1]:
            raise ValueError(f"source shard SHA-256 mismatch: {path.name}")
    return digest


def validate_inventory(
    tensors: list[dict[str, object]],
    expected_shards: int | None,
    expected_tensors: int | None,
    expected_payload_bytes: int | None,
) -> None:
    if not tensors:
        raise ValueError("empty static tensor inventory")
    payload_bytes = sum(int(item["bytes"]) for item in tensors)
    model_root = Path(str(tensors[0]["source_path"])).parent
    if expected_shards is not None and len(shard_paths(model_root)) != expected_shards:
        raise ValueError("unexpected source shard count")
    if expected_tensors is not None and len(tensors) != expected_tensors:
        raise ValueError("unexpected static tensor count")
    if expected_payload_bytes is not None and payload_bytes != expected_payload_bytes:
        raise ValueError("unexpected static payload bytes")


def build_static_pack(
    model: Path,
    output: Path,
    *,
    expected_shards: int | None = PINNED_SHARDS,
    expected_tensors: int | None = PINNED_TENSORS,
    expected_payload_bytes: int | None = PINNED_PAYLOAD_BYTES,
    expected_source_manifest_sha256: str | None =
        PINNED_SOURCE_MANIFEST_SHA256,
) -> dict[str, object]:
    source_manifest_sha256 = validate_source_manifest(
        model, expected_source_manifest_sha256
    )
    tensors = discover_static_tensors(model)
    validate_inventory(
        tensors, expected_shards, expected_tensors, expected_payload_bytes
    )
    source_model_crc64 = source_model_layout_crc64(model)
    header, planned = make_static_header(tensors)
    partial = output.with_name(output.name + ".partial")
    progress_path = output.with_name(output.name + ".progress.json")
    identity = hashlib.sha256(
        json.dumps(
            [{key: item[key] for key in ("name", "dtype", "shape", "bytes", "source_shard", "source_offset")} for item in planned],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    completed: list[dict[str, object]] = []
    if partial.exists() or progress_path.exists():
        if not partial.exists() or not progress_path.exists():
            raise ValueError("incomplete static-pack resume pair")
        progress = json.loads(progress_path.read_text())
        if progress.get("schema") != STATIC_SCHEMA or progress.get("identity") != identity:
            raise ValueError("static-pack progress identity mismatch")
        completed = list(progress.get("completed", []))
        with partial.open("rb") as output_stream:
            if output_stream.read(len(header)) != header:
                raise ValueError("static-pack partial header mismatch")
            for index, record in enumerate(completed):
                item = planned[index]
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
            "completed": [],
        })
    with partial.open("r+b", buffering=0) as destination:
        for index in range(len(completed), len(planned)):
            item = planned[index]
            with Path(str(item["source_path"])).open("rb", buffering=0) as source:
                digest = copy_tensor(
                    source,
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
                    "completed": completed,
                })
        destination.flush()
        os.fsync(destination.fileno())
    with partial.open("rb") as output_stream:
        for index, item in enumerate(planned):
            verify_tensor(
                Path(str(item["source_path"])),
                int(item["source_offset"]),
                output_stream,
                len(header) + int(item["output_offset"]),
                int(item["bytes"]),
                str(completed[index]["sha256"]),
            )
    expected_file_bytes = len(header) + sum(int(item["bytes"]) for item in planned)
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
        "source_manifest_sha256": source_manifest_sha256,
        "path": output.name,
        "tensor_count": len(planned),
        "payload_bytes": sum(int(item["bytes"]) for item in planned),
        "source_model_layout_crc64": f"{source_model_crc64:016x}",
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
    expected_shards: int | None = PINNED_SHARDS,
    expected_tensors: int | None = PINNED_TENSORS,
    expected_payload_bytes: int | None = PINNED_PAYLOAD_BYTES,
    expected_source_manifest_sha256: str | None =
        PINNED_SOURCE_MANIFEST_SHA256,
) -> dict[str, object]:
    source_manifest_sha256 = validate_source_manifest(
        model, expected_source_manifest_sha256
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != STATIC_SCHEMA or manifest.get("version") != 1:
        raise ValueError("invalid static-store manifest")
    if manifest.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("static-store source manifest mismatch")
    tensors = discover_static_tensors(model)
    validate_inventory(tensors, expected_shards, expected_tensors, expected_payload_bytes)
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
    if output.stat().st_size != manifest.get("file_bytes"):
        raise ValueError("static-store verified size mismatch")
    if sha256_file(output) != manifest.get("sha256"):
        raise ValueError("static-store file SHA-256 mismatch")
    return manifest


def validate_mzg2_store(
    root: Path,
    *,
    expected_layers: int = 92,
    expected_experts: int = 82_432,
    expected_source_manifest_sha256: str =
        PINNED_SOURCE_MANIFEST_SHA256,
    verify_hashes: bool = True,
) -> dict[str, object]:
    manifest_path = root / "expert-store.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "moonshine-mzg2-full-v1"
        or manifest.get("version") != 2
        or manifest.get("source_manifest_sha256")
            != expected_source_manifest_sha256
        or manifest.get("layers") != expected_layers
        or manifest.get("experts") != expected_experts
        or manifest.get("tile_bytes") != 16_384
    ):
        raise ValueError("invalid MZG2 full-store manifest")
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
            not isinstance(layer, int)
            or layer < 1
            or layer > expected_layers
            or layer in seen
            or not isinstance(relative, str)
            or relative != f"layer-{layer:03d}.mzg2"
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("invalid MZG2 sidecar record")
        path = root / relative
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"MZG2 sidecar size mismatch: {relative}")
        if verify_hashes and sha256_file(path) != digest:
            raise ValueError(f"MZG2 sidecar SHA-256 mismatch: {relative}")
        seen.add(layer)
    return manifest


def link_mzg2_store(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "expert-store.json", destination / "expert-store.json")
    fsync_file(destination / "expert-store.json")
    for layer in range(1, 93):
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
) -> dict[str, object]:
    routed_manifest_path = output / "expert-store-mzg2" / "expert-store.json"
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "version": 1,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "license": "kimi-k3",
        "source_manifest_sha256": PINNED_SOURCE_MANIFEST_SHA256,
        "source_model_layout_crc64":
            static_manifest["source_model_layout_crc64"],
        "static_store": {
            key: static_manifest[key]
            for key in (
                "path",
                "tensor_count",
                "payload_bytes",
                "file_bytes",
                "sha256",
            )
        },
        "routed_store": {
            "format": "mzg2",
            "path": "expert-store-mzg2",
            "layers": mzg2_manifest["layers"],
            "experts_per_layer": mzg2_manifest["experts_per_layer"],
            "experts": mzg2_manifest["experts"],
            "tile_bytes": mzg2_manifest["tile_bytes"],
            "manifest_sha256": sha256_file(routed_manifest_path),
        },
        "expert_layout": {
            "bytes": 17_547_264,
            "tensors": [
                {"name": name, "offset": offset, "bytes": size}
                for name, offset, size in EXPERT_LAYOUT
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
    validate_source_manifest(
        model,
        PINNED_SOURCE_MANIFEST_SHA256,
        verify_shard_hashes=True,
    )
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    static_path = partial / "model-static.safetensors"
    if static_path.exists():
        static_manifest = verify_static_pack(
            model, static_path, partial / "static-store.json"
        )
    else:
        static_manifest = build_static_pack(model, static_path)
    mzg2_path = partial / "expert-store-mzg2"
    if reuse_mzg2 is not None:
        validate_mzg2_store(reuse_mzg2, verify_hashes=False)
        link_mzg2_store(reuse_mzg2, mzg2_path)
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
    mzg2_manifest = validate_mzg2_store(mzg2_path, verify_hashes=True)
    auxiliary = copy_auxiliary_files(model, partial)
    manifest = write_bundle_manifest(
        partial, static_manifest, mzg2_manifest, auxiliary
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
) -> dict[str, object]:
    manifest = json.loads((root / "moonshine-bundle.json").read_text())
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("version") != 1
        or manifest.get("base_model") != BASE_MODEL
        or manifest.get("base_revision") != BASE_REVISION
        or manifest.get("source_manifest_sha256")
            != PINNED_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("invalid standalone bundle manifest")
    static = manifest.get("static_store")
    if not isinstance(static, dict):
        raise ValueError("missing static store manifest")
    static_path = root / str(static.get("path"))
    if static_path.stat().st_size != static.get("file_bytes"):
        raise ValueError("bundle static file size mismatch")
    if verify_hashes and sha256_file(static_path) != static.get("sha256"):
        raise ValueError("bundle static file SHA-256 mismatch")
    if model is not None:
        verify_static_pack(model, static_path, root / "static-store.json")
    routed = manifest.get("routed_store")
    if not isinstance(routed, dict):
        raise ValueError("missing routed store manifest")
    mzg2_root = root / str(routed.get("path"))
    mzg2 = validate_mzg2_store(mzg2_root, verify_hashes=verify_hashes)
    if sha256_file(mzg2_root / "expert-store.json") != routed.get(
        "manifest_sha256"
    ):
        raise ValueError("bundle MZG2 manifest SHA-256 mismatch")
    for name, digest in manifest.get("auxiliary_sha256", {}).items():
        path = root / name
        if not path.is_file() or (
            verify_hashes and sha256_file(path) != digest
        ):
            raise ValueError(f"bundle auxiliary mismatch: {name}")
    return {"bundle": manifest, "mzg2": mzg2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--model", required=True, type=Path)
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
        source_manifest = validate_source_manifest(
            args.model, PINNED_SOURCE_MANIFEST_SHA256
        )
        tensors = discover_static_tensors(args.model)
        validate_inventory(
            tensors, PINNED_SHARDS, PINNED_TENSORS, PINNED_PAYLOAD_BYTES
        )
        print(json.dumps({
            "source_manifest_sha256": source_manifest,
            "source_model_layout_crc64":
                f"{source_model_layout_crc64(args.model):016x}",
            "tensor_count": len(tensors),
            "payload_bytes": sum(int(item["bytes"]) for item in tensors),
        }, sort_keys=True))
        return 0
    if args.command == "build-static":
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = build_static_pack(
            args.model, args.out / "model-static.safetensors"
        )
        print(json.dumps({key: manifest[key] for key in (
            "tensor_count", "payload_bytes", "file_bytes", "sha256"
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
