#!/usr/bin/env python3
"""Transcode Kimi K3 routed experts into Moonshine MZG1 sidecars.

The official SafeTensors remain untouched.  The output store contains only
random-access expert sidecars plus an expert-store.json manifest referring to
the source model. Each expert is one 4 KiB-aligned block containing twelve
independent two-stripe Zstandard level-1 frames in native tensor order. E2M1
negative zero is canonicalized to positive zero in packed planes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import struct
import threading
import time
from pathlib import Path

import zstandard as zstd

FORMAT_NAME = "moonshine-mzg1"
FILE_MAGIC = b"K3MZG1\0\0"
BLOCK_MAGIC = b"MZGB"
FORMAT_VERSION = 1
FILE_HEADER_BYTES = 4096
BLOCK_ALIGNMENT = 4096
CODEC_ZSTD1 = 1
FILE_FLAG_ZERO_GAUGE = 1 << 0
FILE_FLAG_ZSTD_CHECKSUM = 1 << 1
BLOCK_FLAG_ZERO_GAUGE = 1 << 0
FRAME_RAW_FLAG = 1 << 31
FRAME_SIZE_MASK = FRAME_RAW_FLAG - 1
STRIPES_PER_PLANE = 2
EXPERT_BYTES = 17_547_264
PACKED_BYTES = 5_505_024
SCALE_BYTES = 344_064
PLANE_OUTPUT_BYTES = (
    PACKED_BYTES,
    SCALE_BYTES,
    PACKED_BYTES,
    SCALE_BYTES,
    PACKED_BYTES,
    SCALE_BYTES,
)
FRAME_KINDS = (
    "w1.weight_packed",
    "w1.weight_scale",
    "w2.weight_packed",
    "w2.weight_scale",
    "w3.weight_packed",
    "w3.weight_scale",
)
PLANE_OUTPUT_OFFSETS = (
    0,
    PACKED_BYTES,
    PACKED_BYTES + SCALE_BYTES,
    2 * PACKED_BYTES + SCALE_BYTES,
    2 * (PACKED_BYTES + SCALE_BYTES),
    3 * PACKED_BYTES + 2 * SCALE_BYTES,
)
FRAME_OUTPUT_BYTES = tuple(
    size // STRIPES_PER_PLANE
    for size in PLANE_OUTPUT_BYTES
    for _ in range(STRIPES_PER_PLANE)
)
FRAME_OUTPUT_OFFSETS = tuple(
    base + stripe * (size // STRIPES_PER_PLANE)
    for base, size in zip(PLANE_OUTPUT_OFFSETS, PLANE_OUTPUT_BYTES)
    for stripe in range(STRIPES_PER_PLANE)
)
FRAME_COUNT = len(FRAME_OUTPUT_BYTES)
FILE_HEADER = struct.Struct("<8s9I4Q32s32s")
INDEX_ENTRY = struct.Struct("<HHIQ")
BLOCK_PREFIX = struct.Struct("<4sHHIIII")
FRAME_DESCRIPTOR = struct.Struct("<III")
BLOCK_HEADER_BYTES = (
    (BLOCK_PREFIX.size + FRAME_COUNT * FRAME_DESCRIPTOR.size + 63) // 64 * 64
)
TENSOR_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe"
    r"\.experts\.(\d+)\.(w[123])\.(weight_packed|weight_scale)$"
)
SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{6})\.safetensors$")
ZERO_GAUGE_TABLE = bytes(
    ((0 if (value & 0x0F) == 8 else value & 0x0F) |
     ((0 if (value >> 4) == 8 else value >> 4) << 4))
    for value in range(256)
)
_THREAD_LOCAL = threading.local()


@dataclasses.dataclass(frozen=True)
class TensorRef:
    name: str
    path: Path
    shard_index: int
    physical_offset: int
    byte_length: int
    shape: tuple[int, ...]
    dtype: str


@dataclasses.dataclass(frozen=True)
class ExpertRef:
    layer: int
    expert: int
    shard_index: int
    path: Path
    physical_offset: int
    frames: tuple[TensorRef, ...]


@dataclasses.dataclass(frozen=True)
class IndexRecord:
    layer: int
    expert: int
    block_bytes: int
    block_offset: int


@dataclasses.dataclass(frozen=True)
class SidecarResult:
    path: Path
    source_path: Path
    source_sha256: str
    file_sha256: str
    expert_count: int
    raw_bytes: int
    stored_bytes: int
    max_block_bytes: int
    elapsed_seconds: float


def align_up(value: int, alignment: int = BLOCK_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_set(value: str) -> set[int]:
    result = {int(item) for item in value.split(",") if item}
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated set")
    return result


def parse_manifest(model: Path) -> tuple[bytes, dict[str, tuple[int, str]]]:
    path = model / "manifest.tsv"
    data = path.read_bytes()
    entries: dict[str, tuple[int, str]] = {}
    for line_number, raw_line in enumerate(data.decode().splitlines(), 1):
        fields = raw_line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected three fields")
        name, size_text, digest = fields
        size = int(size_text)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}:{line_number}: invalid SHA-256")
        entries[name] = (size, digest)
    return data, entries


def parse_safetensors_header(path: Path, shard_index: int) -> dict[str, TensorRef]:
    file_size = path.stat().st_size
    with path.open("rb", buffering=0) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated SafeTensors prefix")
        header_length = struct.unpack("<Q", prefix)[0]
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{path}: truncated SafeTensors header")
    metadata = json.loads(header_bytes)
    metadata.pop("__metadata__", None)
    data_start = 8 + header_length
    tensors: dict[str, TensorRef] = {}
    for name, item in metadata.items():
        start, end = item["data_offsets"]
        if start < 0 or end < start or data_start + end > file_size:
            raise ValueError(f"{path}: invalid data_offsets for {name}")
        tensors[name] = TensorRef(
            name=name,
            path=path,
            shard_index=shard_index,
            physical_offset=data_start + start,
            byte_length=end - start,
            shape=tuple(item["shape"]),
            dtype=item["dtype"],
        )
    return tensors


def discover_experts(
    model: Path,
    layers: set[int] | None = None,
    experts: set[int] | None = None,
) -> tuple[dict[int, list[ExpertRef]], int, bytes, dict[str, tuple[int, str]]]:
    manifest_bytes, manifest_entries = parse_manifest(model)
    shard_paths = sorted(model.glob("model-*-of-*.safetensors"))
    if not shard_paths:
        raise ValueError(f"{model}: no SafeTensors shards")
    shard_count = 0
    grouped: dict[tuple[int, int], dict[str, TensorRef]] = {}
    for path in shard_paths:
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        shard_number = int(match.group(1))
        declared_count = int(match.group(2))
        if shard_count and shard_count != declared_count:
            raise ValueError(f"{path}: inconsistent shard count")
        shard_count = declared_count
        if shard_number < 1 or shard_number > shard_count:
            raise ValueError(f"{path}: invalid shard number")
        if path.name not in manifest_entries:
            raise ValueError(f"{path}: absent from manifest.tsv")
        expected_size, _ = manifest_entries[path.name]
        if path.stat().st_size != expected_size:
            raise ValueError(f"{path}: size differs from manifest.tsv")
        for tensor in parse_safetensors_header(path, shard_number - 1).values():
            match = TENSOR_RE.match(tensor.name)
            if not match:
                continue
            layer = int(match.group(1))
            expert = int(match.group(2))
            if layers is not None and layer not in layers:
                continue
            if experts is not None and expert not in experts:
                continue
            key = (layer, expert)
            suffix = f"{match.group(3)}.{match.group(4)}"
            slot = grouped.setdefault(key, {})
            if suffix in slot:
                raise ValueError(f"duplicate tensor L{layer} E{expert} {suffix}")
            slot[suffix] = tensor
    if len(shard_paths) != shard_count:
        raise ValueError(f"expected {shard_count} shards, found {len(shard_paths)}")

    by_shard: dict[int, list[ExpertRef]] = {}
    for (layer, expert), tensors in sorted(grouped.items()):
        missing = [kind for kind in FRAME_KINDS if kind not in tensors]
        if missing:
            raise ValueError(f"L{layer} E{expert}: missing {missing}")
        frames = tuple(tensors[kind] for kind in FRAME_KINDS)
        if any(frame.dtype != "U8" for frame in frames):
            raise ValueError(f"L{layer} E{expert}: non-U8 expert tensor")
        if tuple(frame.byte_length for frame in frames) != PLANE_OUTPUT_BYTES:
            raise ValueError(f"L{layer} E{expert}: unexpected frame sizes")
        if len({frame.shard_index for frame in frames}) != 1:
            raise ValueError(f"L{layer} E{expert}: crosses source shards")
        start = frames[0].physical_offset
        expected = start
        for frame in frames:
            if frame.physical_offset != expected:
                raise ValueError(f"L{layer} E{expert}: non-contiguous tensors")
            expected += frame.byte_length
        if expected - start != EXPERT_BYTES:
            raise ValueError(f"L{layer} E{expert}: invalid expert span")
        ref = ExpertRef(
            layer=layer,
            expert=expert,
            shard_index=frames[0].shard_index,
            path=frames[0].path,
            physical_offset=start,
            frames=frames,
        )
        by_shard.setdefault(ref.shard_index, []).append(ref)
    for refs in by_shard.values():
        refs.sort(key=lambda item: item.physical_offset)
    return by_shard, shard_count, manifest_bytes, manifest_entries


def read_expert(ref: ExpertRef) -> tuple[bytes, ...]:
    descriptor = os.open(ref.path, os.O_RDONLY)
    try:
        raw = os.pread(descriptor, EXPERT_BYTES, ref.physical_offset)
    finally:
        os.close(descriptor)
    if len(raw) != EXPERT_BYTES:
        raise OSError(f"L{ref.layer} E{ref.expert}: short source read")
    planes: list[bytes] = []
    offset = 0
    for index, size in enumerate(PLANE_OUTPUT_BYTES):
        plane = raw[offset:offset + size]
        if index % 2 == 0:
            plane = plane.translate(ZERO_GAUGE_TABLE)
        planes.append(plane)
        offset += size
    return tuple(planes)


def thread_compressor() -> zstd.ZstdCompressor:
    compressor = getattr(_THREAD_LOCAL, "compressor", None)
    if compressor is None:
        compressor = zstd.ZstdCompressor(
            level=1,
            write_checksum=True,
            write_content_size=True,
            write_dict_id=False,
        )
        _THREAD_LOCAL.compressor = compressor
    return compressor


def compress_frame(frame: bytes) -> tuple[int, bytes]:
    encoded = thread_compressor().compress(frame)
    if len(encoded) >= len(frame):
        return FRAME_RAW_FLAG | len(frame), frame
    return len(encoded), encoded


def decode_frame(encoded_size: int, payload: bytes, output_size: int) -> bytes:
    stored_size = encoded_size & FRAME_SIZE_MASK
    if stored_size != len(payload):
        raise ValueError("frame payload length mismatch")
    if encoded_size & FRAME_RAW_FLAG:
        if len(payload) != output_size:
            raise ValueError("raw frame output length mismatch")
        return payload
    return zstd.ZstdDecompressor().decompress(payload, max_output_size=output_size)


def encode_expert(
    ref: ExpertRef,
    pool: concurrent.futures.ThreadPoolExecutor,
) -> tuple[bytes, bytes]:
    planes = read_expert(ref)
    frames = tuple(
        plane[stripe * (len(plane) // STRIPES_PER_PLANE):
              (stripe + 1) * (len(plane) // STRIPES_PER_PLANE)]
        for plane in planes
        for stripe in range(STRIPES_PER_PLANE)
    )
    encoded = list(pool.map(compress_frame, frames))
    frame_sizes = tuple(item[0] for item in encoded)
    payloads = tuple(item[1] for item in encoded)
    payload_bytes = sum(len(payload) for payload in payloads)
    prefix = BLOCK_PREFIX.pack(
        BLOCK_MAGIC,
        ref.layer,
        ref.expert,
        payload_bytes,
        BLOCK_FLAG_ZERO_GAUGE,
        FRAME_COUNT,
        BLOCK_HEADER_BYTES,
    )
    descriptors = b"".join(
        FRAME_DESCRIPTOR.pack(encoded_size, output_offset, output_size)
        for encoded_size, output_offset, output_size in zip(
            frame_sizes, FRAME_OUTPUT_OFFSETS, FRAME_OUTPUT_BYTES
        )
    )
    header = prefix + descriptors
    header += bytes(BLOCK_HEADER_BYTES - len(header))
    block = header + b"".join(payloads)
    block += bytes(align_up(len(block)) - len(block))

    # Immediate source-buffer verification, including Zstd frame checksums.
    decoded = bytearray(EXPERT_BYTES)
    for encoded_size, payload, output_offset, output_size in zip(
        frame_sizes, payloads, FRAME_OUTPUT_OFFSETS, FRAME_OUTPUT_BYTES
    ):
        decoded[output_offset:output_offset + output_size] = decode_frame(
            encoded_size, payload, output_size
        )
    expected = b"".join(planes)
    if decoded != expected:
        raise ValueError(f"L{ref.layer} E{ref.expert}: immediate round trip failed")
    return block, expected


def pack_file_header(
    shard_index: int,
    shard_count: int,
    expert_count: int,
    index_offset: int,
    index_bytes: int,
    data_offset: int,
    max_block_bytes: int,
    source_manifest_sha256: bytes,
    source_shard_sha256: bytes,
) -> bytes:
    prefix = FILE_HEADER.pack(
        FILE_MAGIC,
        FORMAT_VERSION,
        FILE_HEADER_BYTES,
        BLOCK_ALIGNMENT,
        CODEC_ZSTD1,
        FILE_FLAG_ZERO_GAUGE | FILE_FLAG_ZSTD_CHECKSUM,
        shard_index,
        shard_count,
        expert_count,
        INDEX_ENTRY.size,
        index_offset,
        index_bytes,
        data_offset,
        max_block_bytes,
        source_manifest_sha256,
        source_shard_sha256,
    )
    if len(prefix) > FILE_HEADER_BYTES:
        raise AssertionError("MZG file header exceeds reserved bytes")
    return prefix + bytes(FILE_HEADER_BYTES - len(prefix))


def unpack_file_header(data: bytes) -> dict[str, object]:
    if len(data) < FILE_HEADER_BYTES:
        raise ValueError("truncated MZG file header")
    fields = FILE_HEADER.unpack_from(data)
    (
        magic,
        version,
        header_bytes,
        alignment,
        codec,
        flags,
        shard_index,
        shard_count,
        expert_count,
        index_entry_bytes,
        index_offset,
        index_bytes,
        data_offset,
        max_block_bytes,
        source_manifest_sha256,
        source_shard_sha256,
    ) = fields
    if magic != FILE_MAGIC or version != FORMAT_VERSION:
        raise ValueError("unsupported MZG file")
    if header_bytes != FILE_HEADER_BYTES or alignment != BLOCK_ALIGNMENT:
        raise ValueError("invalid MZG alignment/header size")
    if codec != CODEC_ZSTD1 or index_entry_bytes != INDEX_ENTRY.size:
        raise ValueError("unsupported MZG codec/index")
    if flags != FILE_FLAG_ZERO_GAUGE | FILE_FLAG_ZSTD_CHECKSUM:
        raise ValueError("unsupported MZG flags")
    return {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "expert_count": expert_count,
        "index_offset": index_offset,
        "index_bytes": index_bytes,
        "data_offset": data_offset,
        "max_block_bytes": max_block_bytes,
        "source_manifest_sha256": source_manifest_sha256,
        "source_shard_sha256": source_shard_sha256,
    }


def parse_block(block: bytes, expected_layer: int, expected_expert: int) -> bytes:
    if len(block) < BLOCK_PREFIX.size:
        raise ValueError("truncated MZG block")
    (
        magic,
        layer,
        expert,
        payload_bytes,
        flags,
        frame_count,
        header_bytes,
    ) = BLOCK_PREFIX.unpack_from(block)
    if magic != BLOCK_MAGIC or flags != BLOCK_FLAG_ZERO_GAUGE:
        raise ValueError("invalid MZG block header")
    if (layer, expert) != (expected_layer, expected_expert):
        raise ValueError("MZG block identity mismatch")
    if frame_count != FRAME_COUNT or header_bytes != BLOCK_HEADER_BYTES:
        raise ValueError("MZG block frame/header count mismatch")
    if header_bytes + payload_bytes > len(block):
        raise ValueError("MZG block payload exceeds aligned block")
    descriptor_end = BLOCK_PREFIX.size + frame_count * FRAME_DESCRIPTOR.size
    if descriptor_end > header_bytes or any(block[descriptor_end:header_bytes]):
        raise ValueError("MZG block has invalid header padding")
    descriptors = [
        FRAME_DESCRIPTOR.unpack_from(
            block, BLOCK_PREFIX.size + index * FRAME_DESCRIPTOR.size
        )
        for index in range(frame_count)
    ]
    coverage = sorted(
        (output_offset, output_offset + output_size)
        for _, output_offset, output_size in descriptors
    )
    if not coverage or coverage[0][0] != 0 or coverage[-1][1] != EXPERT_BYTES:
        raise ValueError("MZG block output coverage mismatch")
    if any(left[1] != right[0] for left, right in zip(coverage, coverage[1:])):
        raise ValueError("MZG block output ranges overlap or have gaps")
    stored_sizes = [encoded_size & FRAME_SIZE_MASK for encoded_size, _, _ in descriptors]
    if payload_bytes != sum(stored_sizes):
        raise ValueError("MZG block payload size mismatch")
    cursor = header_bytes
    decoded = bytearray(EXPERT_BYTES)
    for (encoded_size, output_offset, output_size), stored_size in zip(
        descriptors, stored_sizes
    ):
        payload = block[cursor:cursor + stored_size]
        decoded[output_offset:output_offset + output_size] = decode_frame(
            encoded_size, payload, output_size
        )
        cursor += stored_size
    if any(block[cursor:]):
        raise ValueError("MZG block has nonzero alignment padding")
    return bytes(decoded)


def verify_sidecar(path: Path, deep: bool = True) -> dict[str, object]:
    file_size = path.stat().st_size
    with path.open("rb", buffering=0) as stream:
        header_bytes = stream.read(FILE_HEADER_BYTES)
        header = unpack_file_header(header_bytes)
        index_offset = int(header["index_offset"])
        index_bytes = int(header["index_bytes"])
        data_offset = int(header["data_offset"])
        expert_count = int(header["expert_count"])
        if index_offset != FILE_HEADER_BYTES or index_bytes != expert_count * INDEX_ENTRY.size:
            raise ValueError(f"{path}: invalid index bounds")
        if data_offset != align_up(index_offset + index_bytes):
            raise ValueError(f"{path}: invalid data offset")
        if data_offset > file_size:
            raise ValueError(f"{path}: data offset beyond file")
        stream.seek(index_offset)
        index_data = stream.read(index_bytes)
        if len(index_data) != index_bytes:
            raise ValueError(f"{path}: truncated index")
        records: list[IndexRecord] = []
        previous_key = (-1, -1)
        maximum = 0
        for offset in range(0, index_bytes, INDEX_ENTRY.size):
            layer, expert, block_bytes, block_offset = INDEX_ENTRY.unpack_from(index_data, offset)
            key = (layer, expert)
            if key <= previous_key:
                raise ValueError(f"{path}: unsorted/duplicate index")
            previous_key = key
            if block_offset % BLOCK_ALIGNMENT or block_bytes % BLOCK_ALIGNMENT:
                raise ValueError(f"{path}: unaligned block")
            if block_offset < data_offset or block_offset + block_bytes > file_size:
                raise ValueError(f"{path}: block outside file")
            maximum = max(maximum, block_bytes)
            records.append(IndexRecord(layer, expert, block_bytes, block_offset))
        expected_offset = data_offset
        for record in sorted(records, key=lambda item: item.block_offset):
            if record.block_offset != expected_offset:
                raise ValueError(f"{path}: block gap/overlap")
            expected_offset += record.block_bytes
        if expected_offset != file_size:
            raise ValueError(f"{path}: trailing/unindexed bytes")
        if maximum != int(header["max_block_bytes"]):
            raise ValueError(f"{path}: max block mismatch")
        if deep:
            for record in records:
                stream.seek(record.block_offset)
                block = stream.read(record.block_bytes)
                if len(block) != record.block_bytes:
                    raise ValueError(f"{path}: truncated block")
                parse_block(block, record.layer, record.expert)
    return {
        "expert_count": expert_count,
        "max_block_bytes": maximum,
        "file_bytes": file_size,
        "source_manifest_sha256": bytes(header["source_manifest_sha256"]).hex(),
        "source_shard_sha256": bytes(header["source_shard_sha256"]).hex(),
    }


def transcode_sidecar(
    refs: list[ExpertRef],
    destination: Path,
    shard_count: int,
    manifest_sha256: bytes,
    source_shard_sha256: bytes,
    jobs: int,
) -> SidecarResult:
    if not refs:
        raise ValueError("cannot encode empty MZG sidecar")
    started = time.monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()
    index_offset = FILE_HEADER_BYTES
    index_bytes = len(refs) * INDEX_ENTRY.size
    data_offset = align_up(index_offset + index_bytes)
    records: list[IndexRecord] = []
    raw_bytes = 0
    maximum = 0
    with partial.open("w+b", buffering=0) as output, concurrent.futures.ThreadPoolExecutor(
        max_workers=jobs
    ) as pool:
        output.truncate(data_offset)
        output.seek(data_offset)
        for position, ref in enumerate(refs, 1):
            block, _ = encode_expert(ref, pool)
            block_offset = output.tell()
            if block_offset % BLOCK_ALIGNMENT:
                raise AssertionError("unaligned MZG output cursor")
            output.write(block)
            records.append(IndexRecord(ref.layer, ref.expert, len(block), block_offset))
            raw_bytes += EXPERT_BYTES
            maximum = max(maximum, len(block))
            if position % 64 == 0 or position == len(refs):
                print(
                    f"  {destination.name}: {position}/{len(refs)} experts "
                    f"raw={raw_bytes / 2**30:.3f} GiB "
                    f"stored={(output.tell() - data_offset) / 2**30:.3f} GiB",
                    flush=True,
                )
        output.seek(index_offset)
        for record in sorted(records, key=lambda item: (item.layer, item.expert)):
            output.write(INDEX_ENTRY.pack(
                record.layer, record.expert, record.block_bytes, record.block_offset
            ))
        output.seek(0)
        output.write(pack_file_header(
            refs[0].shard_index,
            shard_count,
            len(refs),
            index_offset,
            index_bytes,
            data_offset,
            maximum,
            manifest_sha256,
            source_shard_sha256,
        ))
        output.flush()
        os.fsync(output.fileno())
    partial.replace(destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    verification = verify_sidecar(destination, deep=True)
    file_sha256 = sha256_file(destination)
    return SidecarResult(
        path=destination,
        source_path=refs[0].path,
        source_sha256=source_shard_sha256.hex(),
        file_sha256=file_sha256,
        expert_count=len(refs),
        raw_bytes=raw_bytes,
        stored_bytes=int(verification["file_bytes"]),
        max_block_bytes=maximum,
        elapsed_seconds=time.monotonic() - started,
    )


def write_store_manifest(
    output: Path,
    model: Path,
    shard_count: int,
    source_manifest_sha256: str,
    results: list[SidecarResult],
    selected_layers: set[int] | None,
    selected_experts: set[int] | None,
    complete: bool,
) -> Path:
    raw_bytes = sum(result.raw_bytes for result in results)
    stored_bytes = sum(result.stored_bytes for result in results)
    manifest = {
        "schema": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_model": str(model.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "shard_count": shard_count,
        "complete": complete,
        "selected_layers": sorted(selected_layers) if selected_layers else None,
        "selected_experts": sorted(selected_experts) if selected_experts else None,
        "codec": {
            "name": "zstd",
            "level": 1,
            "checksum": True,
            "content_size": True,
            "zstandard_python": zstd.__version__,
        },
        "zero_gauge": "E2M1 0x8 -> 0x0 in packed frames",
        "block_alignment": BLOCK_ALIGNMENT,
        "expert_count": sum(result.expert_count for result in results),
        "raw_expert_bytes": raw_bytes,
        "stored_bytes": stored_bytes,
        "reduction_pct": 100.0 * (1.0 - stored_bytes / raw_bytes) if raw_bytes else 0.0,
        "max_block_bytes": max((result.max_block_bytes for result in results), default=0),
        "sidecars": [
            {
                "file": result.path.name,
                "source_file": result.source_path.name,
                "source_sha256": result.source_sha256,
                "sha256": result.file_sha256,
                "experts": result.expert_count,
                "raw_bytes": result.raw_bytes,
                "stored_bytes": result.stored_bytes,
                "max_block_bytes": result.max_block_bytes,
                "elapsed_seconds": result.elapsed_seconds,
            }
            for result in sorted(results, key=lambda item: item.path.name)
        ],
    }
    path = output / "expert-store.json"
    temporary = output / "expert-store.json.partial"
    temporary.write_text(json.dumps(manifest, indent=1) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def verify_store(
    output: Path, deep: bool = True, jobs: int = 1
) -> dict[str, object]:
    manifest_path = output / "expert-store.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != FORMAT_NAME or manifest.get("version") != FORMAT_VERSION:
        raise ValueError(f"{output}: unsupported store manifest")

    def verify_item(item: dict[str, object]) -> dict[str, object]:
        path = output / str(item["file"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"{path}: SHA-256 mismatch")
        result = verify_sidecar(path, deep=deep)
        if result["source_shard_sha256"] != item["source_sha256"]:
            raise ValueError(f"{path}: source SHA-256 mismatch")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        verified = list(pool.map(verify_item, manifest["sidecars"]))
    expert_count = sum(int(result["expert_count"]) for result in verified)
    stored_bytes = sum(int(result["file_bytes"]) for result in verified)
    maximum = max(
        (int(result["max_block_bytes"]) for result in verified),
        default=0,
    )
    if expert_count != manifest["expert_count"]:
        raise ValueError(f"{output}: expert count mismatch")
    if stored_bytes != manifest["stored_bytes"]:
        raise ValueError(f"{output}: stored byte count mismatch")
    if maximum != manifest["max_block_bytes"]:
        raise ValueError(f"{output}: max block mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=parse_int_set)
    parser.add_argument("--experts", type=parse_int_set)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--shard-jobs", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--shallow-verify", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.shard_jobs < 1:
        parser.error("--jobs and --shard-jobs must be at least one")
    if args.verify_only:
        manifest = verify_store(
            args.out,
            deep=not args.shallow_verify,
            jobs=args.shard_jobs,
        )
        print(json.dumps({
            "store": str(args.out),
            "experts": manifest["expert_count"],
            "stored_bytes": manifest["stored_bytes"],
            "reduction_pct": manifest["reduction_pct"],
            "max_block_bytes": manifest["max_block_bytes"],
            "verification": "pass",
        }, indent=1))
        return 0
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")

    by_shard, shard_count, manifest_bytes, manifest_entries = discover_experts(
        args.model, args.layers, args.experts
    )
    expected_complete = args.layers is None and args.experts is None
    expected_experts = 92 * 896 if expected_complete else sum(map(len, by_shard.values()))
    if sum(map(len, by_shard.values())) != expected_experts:
        raise ValueError(
            f"expert inventory mismatch: {sum(map(len, by_shard.values()))} != {expected_experts}"
        )
    partial_root = args.out.with_name(args.out.name + ".partial")
    partial_root.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = hashlib.sha256(manifest_bytes).digest()
    results: list[SidecarResult] = []
    started = time.monotonic()
    existing: list[tuple[list[ExpertRef], Path, str]] = []
    pending: list[tuple[list[ExpertRef], Path, bytes]] = []
    for _, refs in sorted(by_shard.items()):
        source = refs[0].path
        source_size, source_digest = manifest_entries[source.name]
        if source.stat().st_size != source_size:
            raise ValueError(f"{source}: size changed during inventory")
        destination = partial_root / source.name.replace(".safetensors", ".mzg")
        if destination.exists():
            existing.append((refs, destination, source_digest))
        else:
            pending.append((refs, destination, bytes.fromhex(source_digest)))

    def resume_existing(
        item: tuple[list[ExpertRef], Path, str]
    ) -> SidecarResult:
        refs, destination, source_digest = item
        verification = verify_sidecar(destination, deep=True)
        if verification["source_manifest_sha256"] != manifest_sha256.hex():
            raise ValueError(f"{destination}: source manifest mismatch")
        result = SidecarResult(
            path=destination,
            source_path=refs[0].path,
            source_sha256=source_digest,
            file_sha256=sha256_file(destination),
            expert_count=int(verification["expert_count"]),
            raw_bytes=int(verification["expert_count"]) * EXPERT_BYTES,
            stored_bytes=int(verification["file_bytes"]),
            max_block_bytes=int(verification["max_block_bytes"]),
            elapsed_seconds=0.0,
        )
        print(f"resume verified: {destination.name}", flush=True)
        return result

    def transcode_pending(
        item: tuple[list[ExpertRef], Path, bytes]
    ) -> SidecarResult:
        refs, destination, source_digest = item
        print(
            f"transcoding {refs[0].path.name}: {len(refs)} experts",
            flush=True,
        )
        return transcode_sidecar(
            refs,
            destination,
            shard_count,
            manifest_sha256,
            source_digest,
            args.jobs,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.shard_jobs
    ) as shard_pool:
        if existing:
            results.extend(shard_pool.map(resume_existing, existing))
        if pending:
            results.extend(shard_pool.map(transcode_pending, pending))
    manifest_path = write_store_manifest(
        partial_root,
        args.model,
        shard_count,
        manifest_sha256.hex(),
        results,
        args.layers,
        args.experts,
        expected_complete,
    )
    verify_store(partial_root, deep=True, jobs=args.shard_jobs)
    partial_root.replace(args.out)
    parent_fd = os.open(args.out.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    elapsed = time.monotonic() - started
    manifest = json.loads((args.out / manifest_path.name).read_text())
    print("\n==== MZG TRANSCODE COMPLETE ====")
    print(f"store:           {args.out}")
    print(f"experts:         {manifest['expert_count']}")
    print(f"raw bytes:       {manifest['raw_expert_bytes']}")
    print(f"stored bytes:    {manifest['stored_bytes']}")
    print(f"reduction:       {manifest['reduction_pct']:.3f}%")
    print(f"max block:       {manifest['max_block_bytes']}")
    print(f"elapsed:         {elapsed:.1f}s")
    print("verification:    PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
