#!/usr/bin/env python3
"""Behavioral tests for the Moonshine MZG1 transcoder."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "tools" / "transcode_mzg.py"
SPEC = importlib.util.spec_from_file_location("transcode_mzg", TOOL)
assert SPEC and SPEC.loader
mzg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mzg
SPEC.loader.exec_module(mzg)


def synthetic_frame(index: int, size: int) -> bytes:
    if index % 2 == 0:
        # Both nibbles include negative zero often; remaining values ensure the
        # transform cannot pass by replacing the entire packed plane.
        pattern = bytes((0x88, 0x81, 0x28, 0xF8, 0x37, 0x80))
    else:
        pattern = bytes((120 + index, 120 + index, 121 + index))
    return (pattern * (size // len(pattern) + 1))[:size]


def write_model(root: Path) -> tuple[Path, tuple[bytes, ...]]:
    root.mkdir(parents=True)
    frames = tuple(
        synthetic_frame(index, size)
        for index, size in enumerate(mzg.PLANE_OUTPUT_BYTES)
    )
    metadata = {}
    cursor = 0
    for kind, frame, size in zip(mzg.FRAME_KINDS, frames, mzg.PLANE_OUTPUT_BYTES):
        matrix, tensor_kind = kind.split(".")
        if tensor_kind == "weight_packed":
            shape = [3072, 1792] if matrix != "w2" else [3584, 1536]
        else:
            shape = [3072, 112] if matrix != "w2" else [3584, 96]
        name = (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            f"{kind}"
        )
        metadata[name] = {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    encoded_header = json.dumps(metadata, separators=(",", ":")).encode()
    header_length = mzg.align_up(len(encoded_header), 8)
    encoded_header += b" " * (header_length - len(encoded_header))
    shard = root / "model-00001-of-000001.safetensors"
    with shard.open("wb") as stream:
        stream.write(struct.pack("<Q", header_length))
        stream.write(encoded_header)
        for frame in frames:
            stream.write(frame)
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    (root / "manifest.tsv").write_text(
        f"{shard.name}\t{shard.stat().st_size}\t{digest}\n"
    )
    return shard, frames


class MzgFormatTests(unittest.TestCase):
    def test_header_sizes_are_stable(self):
        self.assertEqual(mzg.FILE_HEADER.size, 140)
        self.assertEqual(mzg.INDEX_ENTRY.size, 16)
        self.assertEqual(mzg.BLOCK_PREFIX.size, 24)
        self.assertEqual(mzg.FRAME_DESCRIPTOR.size, 12)
        self.assertEqual(mzg.BLOCK_HEADER_BYTES, 192)
        self.assertEqual(mzg.align_up(1), 4096)
        self.assertEqual(mzg.align_up(4096), 4096)

    def test_zero_gauge_changes_only_negative_zero(self):
        source = bytes(range(256))
        transformed = source.translate(mzg.ZERO_GAUGE_TABLE)
        for before, after in zip(source, transformed):
            expected_low = 0 if (before & 0x0F) == 8 else before & 0x0F
            expected_high = 0 if (before >> 4) == 8 else before >> 4
            self.assertEqual(after, expected_low | expected_high << 4)

    def test_sidecar_round_trip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            shard, source_frames = write_model(model)
            by_shard, shard_count, manifest_bytes, entries = mzg.discover_experts(
                model, {1}, {0}
            )
            refs = by_shard[0]
            self.assertEqual(len(refs), 1)
            destination = root / "model-00001-of-000001.mzg"
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                block, canonical_expert = mzg.encode_expert(refs[0], pool)
            expected_expert = b"".join(
                frame.translate(mzg.ZERO_GAUGE_TABLE)
                if index % 2 == 0 else frame
                for index, frame in enumerate(source_frames)
            )
            self.assertEqual(canonical_expert, expected_expert)
            self.assertEqual(mzg.parse_block(block, 1, 0), expected_expert)

            result = mzg.transcode_sidecar(
                refs,
                destination,
                shard_count,
                hashlib.sha256(manifest_bytes).digest(),
                bytes.fromhex(entries[shard.name][1]),
                jobs=2,
            )
            verified = mzg.verify_sidecar(destination, deep=True)
            self.assertEqual(result.expert_count, 1)
            self.assertEqual(verified["expert_count"], 1)
            self.assertEqual(verified["source_shard_sha256"], entries[shard.name][1])

            with destination.open("r+b", buffering=0) as stream:
                stream.seek(mzg.FILE_HEADER_BYTES)
                index = stream.read(mzg.INDEX_ENTRY.size)
                _, _, block_bytes, block_offset = mzg.INDEX_ENTRY.unpack(index)
                stream.seek(block_offset)
                stored = bytearray(stream.read(block_bytes))
                prefix = mzg.BLOCK_PREFIX.unpack_from(stored)
                header_bytes = prefix[6]
                first = mzg.FRAME_DESCRIPTOR.unpack_from(
                    stored, mzg.BLOCK_PREFIX.size
                )
                first_size = first[0] & mzg.FRAME_SIZE_MASK
                self.assertFalse(first[0] & mzg.FRAME_RAW_FLAG)
                corrupt_at = header_bytes + first_size // 2
                stored[corrupt_at] ^= 0x01
                stream.seek(block_offset)
                stream.write(stored)
                stream.flush()
                os.fsync(stream.fileno())
            with self.assertRaises(Exception):
                mzg.verify_sidecar(destination, deep=True)


if __name__ == "__main__":
    unittest.main()
