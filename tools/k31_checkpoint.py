#!/usr/bin/env python3
"""Inspect a complete, pinned Kimi K3.1 SafeTensors checkpoint offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any

UPSTREAM_ID = "moonshotai/Kimi-K3.1-Pro"
UPSTREAM_REVISION = "6d0a0202956796543364d666ab19c6ad1a494ee1"
ARCHITECTURE = "KimiK31ForConditionalGeneration"
MODEL_TYPE = "kimi_k31"
OFFICIAL_SHARDS = 96
OFFICIAL_ROUTED_LAYERS = 92
OFFICIAL_EXPERTS_PER_LAYER = 896
EXPERT_SPAN_BYTES = 17_547_264
SCHEMA = "moonshine-k31-checkpoint-v1"
INDEX_NAME = "model.safetensors.index.json"
EXPERT_MARKER = ".block_sparse_moe.experts."
EXPERT_NAME = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe"
    r"\.experts\.(\d+)\.(w[123])\.(weight_packed|weight_scale)$"
)
EXPERT_LAYOUT = (
    ("w1.weight_packed", 5_505_024),
    ("w1.weight_scale", 344_064),
    ("w2.weight_packed", 5_505_024),
    ("w2.weight_scale", 344_064),
    ("w3.weight_packed", 5_505_024),
    ("w3.weight_scale", 344_064),
)
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
DSA_LAYERS = [
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
    52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93,
]
DSA_FIELDS = {
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}
PARTIAL_SUFFIXES = (".partial", ".part", ".tmp", ".incomplete")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


class CheckpointError(ValueError):
    """The checkpoint is incomplete, unsafe, or not the pinned K3.1 model."""


def canonical_json(value: object) -> str:
    """Return the stable JSON representation used by manifests and digests."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _crc64_update(crc: int, data: bytes) -> int:
    polynomial = 0x42F0E1EBA9EA3693
    for byte in data:
        crc ^= byte << 56
        for _ in range(8):
            crc = ((crc << 1) ^ polynomial) & 0xFFFFFFFFFFFFFFFF \
                if crc & (1 << 63) else (crc << 1) & 0xFFFFFFFFFFFFFFFF
    return crc


def _crc_u64(crc: int, value: int) -> int:
    return _crc64_update(crc, struct.pack("<Q", value))


def _crc_u32(crc: int, value: int) -> int:
    return _crc64_update(crc, struct.pack("<I", value))


def source_layout_crc64(
    tensors: list[_Tensor],
    shards: list[dict[str, object]],
    upstream: dict[str, object],
    config: dict[str, object],
) -> str:
    """Match the engine's tensor-layout plus pinned-profile identity CRC."""
    crc = _crc_u64(0, len(shards))
    for shard in shards:
        crc = _crc_u64(crc, int(shard["bytes"]))
        crc = _crc_u64(crc, int(shard["data_offset"]))
    crc = _crc_u64(crc, len(tensors))
    shard_ids = {str(shard["name"]): index for index, shard in enumerate(shards)}
    dtype_ids = {"BF16": 1, "F32": 2, "U8": 3}
    for tensor in sorted(tensors, key=lambda item: item.name):
        encoded = tensor.name.encode("utf-8")
        crc = _crc_u64(crc, len(encoded))
        crc = _crc64_update(crc, encoded)
        crc = _crc_u64(crc, tensor.physical_offset)
        crc = _crc_u64(crc, tensor.bytes)
        crc = _crc64_update(crc, struct.pack("<H", shard_ids[tensor.shard]))
        crc = _crc_u32(crc, len(tensor.shape))
        crc = _crc_u32(crc, dtype_ids[tensor.dtype])
        for dimension in tensor.shape:
            crc = _crc_u64(crc, dimension)
    for key in ("id", "revision"):
        encoded = str(upstream[key]).encode("utf-8")
        crc = _crc_u64(crc, len(encoded))
        crc = _crc64_update(crc, encoded)
    text = config["text_config"]
    dsa = text["dsa_config"]
    for key in (
        "index_topk", "index_head_dim", "index_num_heads", "index_rope_dim",
        "index_k_norm_type", "index_topk_freq", "index_skip_topk_offset",
    ):
        value = 1 if key == "index_k_norm_type" else dsa[key]
        crc = _crc_u32(crc, int(value))
    return f"{crc:016x}"


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_json(encoded: bytes, source: str) -> dict[str, object]:
    try:
        text = encoded.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CheckpointError(f"non-finite JSON number in {source}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"invalid JSON in {source}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"JSON root is not an object: {source}")
    return value


def _regular_file(path: Path) -> int:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise CheckpointError(f"missing checkpoint file: {path.name}") from error
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointError(f"checkpoint file is not a regular file: {path.name}")
    return info.st_size


def _load_json(path: Path) -> dict[str, object]:
    size = _regular_file(path)
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise CheckpointError(f"cannot read {path.name}: {error}") from error
    if len(encoded) != size:
        raise CheckpointError(f"file changed while being read: {path.name}")
    return _decode_json(encoded, path.name)


def _require_int(value: object, description: str) -> int:
    if type(value) is not int:
        raise CheckpointError(f"{description} must be an integer")
    return value


def _validate_config(config: dict[str, object]) -> None:
    if config.get("architectures") != [ARCHITECTURE]:
        raise CheckpointError(f"architecture is not {ARCHITECTURE}")
    if config.get("model_type") != MODEL_TYPE:
        raise CheckpointError(f"model_type is not {MODEL_TYPE}")
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("num_hidden_layers") != 93:
        raise CheckpointError("text_config.num_hidden_layers is not 93")
    dsa = text.get("dsa_config")
    if not isinstance(dsa, dict):
        raise CheckpointError("missing text_config.dsa_config")
    if dsa.get("dsa_layers") != DSA_LAYERS:
        raise CheckpointError("DSA layer schedule does not match the pinned model")
    for name, expected in DSA_FIELDS.items():
        if dsa.get(name) != expected or type(dsa.get(name)) is not type(expected):
            raise CheckpointError(f"DSA field {name} does not match the pinned model")
    linear = text.get("linear_attn_config")
    if not isinstance(linear, dict):
        raise CheckpointError("missing text_config.linear_attn_config")
    if linear.get("full_attn_layers") != DSA_LAYERS:
        raise CheckpointError("linear-attention full_attn_layers mismatch")
    dsa_set = set(DSA_LAYERS)
    expected_kda = [layer for layer in range(1, 94) if layer not in dsa_set]
    if linear.get("kda_layers") != expected_kda:
        raise CheckpointError("linear-attention kda_layers mismatch")


def _reject_partials(root: Path) -> None:
    try:
        paths = root.rglob("*")
        for path in paths:
            if path.name.lower().endswith(PARTIAL_SUFFIXES):
                raise CheckpointError(f"incomplete checkpoint artifact: {path.name}")
    except OSError as error:
        raise CheckpointError(f"cannot scan checkpoint directory: {error}") from error


def _shard_names(count: int) -> list[str]:
    return [
        f"model-{number:05d}-of-{count:06d}.safetensors"
        for number in range(1, count + 1)
    ]


def _validate_tensor_name(name: object) -> str:
    if not isinstance(name, str) or not name or name in {".", "..", "__metadata__"}:
        raise CheckpointError(f"unsafe tensor name: {name!r}")
    if (
        name != name.strip()
        or "/" in name
        or "\\" in name
        or any(ord(character) < 0x21 or ord(character) == 0x7F for character in name)
    ):
        raise CheckpointError(f"unsafe tensor name: {name!r}")
    return name


def _read_header(path: Path) -> tuple[int, int, dict[str, object]]:
    file_bytes = _regular_file(path)
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise CheckpointError(f"truncated SafeTensors prefix: {path.name}")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if header_bytes == 0 or header_bytes > file_bytes - 8:
                raise CheckpointError(f"invalid SafeTensors header length: {path.name}")
            encoded = stream.read(header_bytes)
            if len(encoded) != header_bytes:
                raise CheckpointError(f"truncated SafeTensors header: {path.name}")
    except OSError as error:
        raise CheckpointError(f"cannot read {path.name}: {error}") from error
    return file_bytes, 8 + header_bytes, _decode_json(encoded, path.name)


class _Tensor:
    __slots__ = (
        "name", "dtype", "shape", "shard", "start", "end", "physical_offset",
    )

    def __init__(
        self,
        name: str,
        dtype: str,
        shape: list[int],
        shard: str,
        start: int,
        end: int,
        physical_offset: int,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.shard = shard
        self.start = start
        self.end = end
        self.physical_offset = physical_offset

    @property
    def bytes(self) -> int:
        return self.end - self.start

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "bytes": self.bytes,
            "source_shard": self.shard,
            "data_offsets": [self.start, self.end],
            "source_offset": self.physical_offset,
        }


def _parse_tensor(
    name: str,
    raw: object,
    shard: str,
    data_offset: int,
    file_bytes: int,
) -> _Tensor:
    if not isinstance(raw, dict):
        raise CheckpointError(f"tensor metadata is not an object: {name}")
    dtype = raw.get("dtype")
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        raise CheckpointError(f"unsupported tensor dtype: {name}")
    raw_shape = raw.get("shape")
    if not isinstance(raw_shape, list):
        raise CheckpointError(f"invalid tensor shape: {name}")
    shape = [_require_int(value, f"shape dimension for {name}") for value in raw_shape]
    if any(value < 0 for value in shape):
        raise CheckpointError(f"negative tensor shape: {name}")
    offsets = raw.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(value) is not int for value in offsets)
    ):
        raise CheckpointError(f"invalid tensor data_offsets: {name}")
    start, end = offsets
    if start < 0 or end < start or data_offset + end > file_bytes:
        raise CheckpointError(f"tensor exceeds shard data span: {name}")
    elements = 1
    for dimension in shape:
        elements *= dimension
    if elements * DTYPE_BYTES[dtype] != end - start:
        raise CheckpointError(f"tensor shape/byte length mismatch: {name}")
    return _Tensor(name, dtype, shape, shard, start, end, data_offset + start)


def _sha256_file(path: Path, expected_bytes: int) -> str:
    try:
        before = path.lstat()
    except OSError as error:
        raise CheckpointError(f"cannot stat {path.name}: {error}") from error
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
        raise CheckpointError(f"file changed before SHA-256 verification: {path.name}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
        after = path.lstat()
    except OSError as error:
        raise CheckpointError(f"cannot hash {path.name}: {error}") from error
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_size != expected_bytes
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_size != before.st_size
    ):
        raise CheckpointError(f"file changed during SHA-256 verification: {path.name}")
    return digest.hexdigest()


def _static_tensor(tensor: _Tensor) -> bool:
    return tensor.name.startswith("language_model.") and EXPERT_MARKER not in tensor.name


def _expert_geometry(
    tensors: list[_Tensor],
    routed_layers: int,
    experts_per_layer: int,
) -> dict[str, object]:
    groups: dict[tuple[int, int], dict[str, _Tensor]] = {}
    for tensor in tensors:
        if EXPERT_MARKER not in tensor.name:
            continue
        match = EXPERT_NAME.fullmatch(tensor.name)
        if match is None:
            raise CheckpointError(f"unrecognized routed expert tensor: {tensor.name}")
        layer, expert = int(match.group(1)), int(match.group(2))
        suffix = f"{match.group(3)}.{match.group(4)}"
        group = groups.setdefault((layer, expert), {})
        if suffix in group:
            raise CheckpointError(f"duplicate routed expert tensor: {tensor.name}")
        group[suffix] = tensor

    expected = {
        (layer, expert)
        for layer in range(1, routed_layers + 1)
        for expert in range(experts_per_layer)
    }
    if set(groups) != expected:
        missing = sorted(expected - set(groups))
        extra = sorted(set(groups) - expected)
        raise CheckpointError(
            "routed expert coverage mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    reference_shapes: dict[str, list[int]] = {}
    for key in sorted(groups):
        group = groups[key]
        if set(group) != {name for name, _ in EXPERT_LAYOUT}:
            raise CheckpointError(f"expert {key} does not have exactly six tensors")
        ordered = [group[name] for name, _ in EXPERT_LAYOUT]
        shard = ordered[0].shard
        span_start = ordered[0].physical_offset
        cursor = span_start
        for tensor, (suffix, expected_bytes) in zip(ordered, EXPERT_LAYOUT):
            if tensor.shard != shard:
                raise CheckpointError(f"expert {key} crosses shard boundaries")
            if tensor.dtype != "U8":
                raise CheckpointError(f"expert tensor is not U8: {tensor.name}")
            if tensor.bytes != expected_bytes:
                raise CheckpointError(f"expert tensor has wrong size: {tensor.name}")
            if tensor.physical_offset != cursor:
                raise CheckpointError(f"expert tensors are not contiguous: {key}")
            shape = reference_shapes.setdefault(suffix, tensor.shape)
            if tensor.shape != shape:
                raise CheckpointError(f"expert tensor shape differs: {tensor.name}")
            cursor += tensor.bytes
        if cursor - span_start != EXPERT_SPAN_BYTES:
            raise CheckpointError(f"expert {key} has wrong physical span")

    layout = []
    offset = 0
    for suffix, size in EXPERT_LAYOUT:
        layout.append({
            "name": suffix,
            "dtype": "U8",
            "shape": reference_shapes.get(suffix, []),
            "offset": offset,
            "bytes": size,
        })
        offset += size
    return {
        "routed_layers": routed_layers,
        "experts_per_layer": experts_per_layer,
        "expert_count": len(groups),
        "tensors_per_expert": len(EXPERT_LAYOUT),
        "bytes_per_expert": EXPERT_SPAN_BYTES,
        "routed_tensor_count": len(groups) * len(EXPERT_LAYOUT),
        "routed_bytes": len(groups) * EXPERT_SPAN_BYTES,
        "tensor_layout": layout,
    }


def inspect_checkpoint(
    model: str | Path,
    *,
    verify_sha256: bool = False,
    expected_shards: int = OFFICIAL_SHARDS,
    expected_routed_layers: int = OFFICIAL_ROUTED_LAYERS,
    expected_experts_per_layer: int = OFFICIAL_EXPERTS_PER_LAYER,
) -> dict[str, Any]:
    """Inspect *model* and return a JSON-serializable canonical source manifest.

    The defaults enforce the official 96-shard checkpoint and full routed expert
    matrix. The expected-count arguments exist for small synthetic fixtures and
    format tests; callers inspecting K3.1 should retain the defaults. Metadata
    mode reads only JSON and shard headers. ``verify_sha256=True`` additionally
    streams every input file and adds full-content SHA-256 identities.
    """
    root = Path(model)
    if not root.is_dir():
        raise CheckpointError(f"checkpoint directory not found: {root}")
    if expected_shards < 1 or expected_routed_layers < 0 or expected_experts_per_layer < 0:
        raise CheckpointError("expected checkpoint geometry must be non-negative")
    if expected_routed_layers == 0 and expected_experts_per_layer != 0:
        raise CheckpointError("expert count must be zero when there are no routed layers")
    _reject_partials(root)

    config_path = root / "config.json"
    index_path = root / INDEX_NAME
    config = _load_json(config_path)
    _validate_config(config)
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise CheckpointError("index weight_map is not an object")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or "total_size" not in metadata:
        raise CheckpointError("index metadata.total_size is missing")
    index_total_size = _require_int(metadata["total_size"], "index total_size")
    if index_total_size < 0:
        raise CheckpointError("index total_size is negative")

    names = _shard_names(expected_shards)
    expected_name_set = set(names)
    index_shards: set[str] = set()
    for raw_name, raw_shard in weight_map.items():
        _validate_tensor_name(raw_name)
        if not isinstance(raw_shard, str) or raw_shard not in expected_name_set:
            raise CheckpointError(f"unsafe or unexpected shard in index: {raw_shard!r}")
        index_shards.add(raw_shard)
    if index_shards != expected_name_set or len(index_shards) != expected_shards:
        raise CheckpointError(f"index does not reference exactly {expected_shards} shards")

    actual_names = {path.name for path in root.glob("*.safetensors")}
    if actual_names != expected_name_set:
        missing = sorted(expected_name_set - actual_names)
        extra = sorted(actual_names - expected_name_set)
        raise CheckpointError(
            f"shard file coverage mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    tensors: list[_Tensor] = []
    seen: set[str] = set()
    shard_records: list[dict[str, object]] = []
    for shard_name in names:
        path = root / shard_name
        file_bytes, data_offset, header = _read_header(path)
        shard_tensors: list[_Tensor] = []
        for raw_name, raw_metadata in header.items():
            if raw_name == "__metadata__":
                if not isinstance(raw_metadata, dict) or not all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in raw_metadata.items()
                ):
                    raise CheckpointError(f"invalid SafeTensors __metadata__: {shard_name}")
                continue
            name = _validate_tensor_name(raw_name)
            if name in seen:
                raise CheckpointError(f"duplicate tensor across shards: {name}")
            if weight_map.get(name) != shard_name:
                raise CheckpointError(f"index/header shard mismatch: {name}")
            tensor = _parse_tensor(
                name, raw_metadata, shard_name, data_offset, file_bytes
            )
            seen.add(name)
            tensors.append(tensor)
            shard_tensors.append(tensor)
        cursor = 0
        for tensor in sorted(shard_tensors, key=lambda item: item.start):
            if tensor.start != cursor:
                raise CheckpointError(
                    f"overlapping, gapped, or unsafely spaced tensors: {shard_name}"
                )
            cursor = tensor.end
        if cursor != file_bytes - data_offset:
            raise CheckpointError(f"trailing or uncovered shard data: {shard_name}")
        shard_records.append({
            "name": shard_name,
            "bytes": file_bytes,
            "header_bytes": data_offset - 8,
            "data_offset": data_offset,
            "tensor_count": len(shard_tensors),
        })

    index_names = set(weight_map)
    if index_names != seen:
        missing = sorted(index_names - seen)
        unexpected = sorted(seen - index_names)
        raise CheckpointError(
            f"index/tensor coverage mismatch: missing={missing[:3]} "
            f"unexpected={unexpected[:3]}"
        )
    total_bytes = sum(tensor.bytes for tensor in tensors)
    if total_bytes != index_total_size:
        raise CheckpointError("index total_size does not match tensor payload bytes")

    static_tensors = []
    for tensor in sorted(tensors, key=lambda item: item.name):
        if not _static_tensor(tensor):
            continue
        if tensor.dtype not in {"BF16", "F32"} or not tensor.shape or any(
            dimension <= 0 for dimension in tensor.shape
        ):
            raise CheckpointError(f"unsupported static tensor: {tensor.name}")
        static_tensors.append(tensor.record())
    geometry = _expert_geometry(
        tensors, expected_routed_layers, expected_experts_per_layer
    )
    ordered_records = [tensor.record() for tensor in sorted(
        tensors, key=lambda item: item.name
    )]
    upstream = {
        "id": UPSTREAM_ID,
        "revision": UPSTREAM_REVISION,
        "architecture": ARCHITECTURE,
        "model_type": MODEL_TYPE,
    }
    layout_digest = hashlib.sha256(canonical_json({
        "upstream": upstream,
        "shards": shard_records,
        "tensors": ordered_records,
    }).encode("utf-8")).hexdigest()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "upstream": upstream,
        "verification": "full-sha256" if verify_sha256 else "metadata-only",
        "tensor_count": len(tensors),
        "total_bytes": total_bytes,
        "file_bytes": sum(int(record["bytes"]) for record in shard_records),
        "static_tensor_count": len(static_tensors),
        "static_bytes": sum(int(record["bytes"]) for record in static_tensors),
        "static_tensors": static_tensors,
        "expert_geometry": geometry,
        "shards": shard_records,
        "source_layout_sha256": layout_digest,
        "source_model_layout_crc64": source_layout_crc64(
            tensors, shard_records, upstream, config
        ),
    }
    if verify_sha256:
        input_hashes = {
            "config.json": _sha256_file(config_path, _regular_file(config_path)),
            INDEX_NAME: _sha256_file(index_path, _regular_file(index_path)),
        }
        for record in shard_records:
            name = str(record["name"])
            digest = _sha256_file(root / name, int(record["bytes"]))
            record["sha256"] = digest
            input_hashes[name] = digest
        manifest["input_sha256"] = input_hashes
        manifest["source_content_sha256"] = hashlib.sha256(canonical_json(
            input_hashes
        ).encode("utf-8")).hexdigest()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="complete local K3.1 checkpoint directory")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="stream every input file and include full-content SHA-256 identities",
    )
    args = parser.parse_args(argv)
    try:
        manifest = inspect_checkpoint(args.model, verify_sha256=args.verify_sha256)
    except (CheckpointError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(canonical_json(manifest) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
