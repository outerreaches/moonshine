#!/usr/bin/env python3
"""Behavioral tests for offline Kimi K3.1 checkpoint inspection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "tools" / "k31_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("k31_checkpoint", TOOL)
assert SPEC and SPEC.loader
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)

TensorSpec = tuple[str, str, list[int], int]


def config() -> dict[str, object]:
    dsa_layers = list(checkpoint.DSA_LAYERS)
    return {
        "architectures": [checkpoint.ARCHITECTURE],
        "model_type": checkpoint.MODEL_TYPE,
        "text_config": {
            "num_hidden_layers": 93,
            "dsa_config": {
                "dsa_layers": dsa_layers,
                **checkpoint.DSA_FIELDS,
            },
            "linear_attn_config": {
                "full_attn_layers": dsa_layers,
                "kda_layers": [
                    layer for layer in range(1, 94) if layer not in dsa_layers
                ],
            },
        },
    }


def write_shard(path: Path, tensors: list[TensorSpec]) -> None:
    metadata: dict[str, object] = {}
    cursor = 0
    for name, dtype, shape, byte_length in tensors:
        metadata[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + byte_length],
        }
        cursor += byte_length
    encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        if cursor:
            stream.seek(cursor - 1, 1)
            stream.write(b"\0")


def write_index(root: Path, assignments: dict[str, str]) -> None:
    (root / checkpoint.INDEX_NAME).write_text(json.dumps({
        "metadata": {"total_size": sum(
            _indexed_tensor_size(name) for name in assignments
        )},
        "weight_map": assignments,
    }), encoding="utf-8")


def _indexed_tensor_size(name: str) -> int:
    for suffix, size in checkpoint.EXPERT_LAYOUT:
        if name.endswith("." + suffix):
            return size
    if name.endswith("norm.weight"):
        return 4
    if name.endswith("embed_tokens.weight"):
        return 8
    if name.endswith("separator.weight"):
        return 4
    if name.endswith("escape"):
        return 2
    raise AssertionError(f"unknown fixture tensor: {name}")


def expert_tensors(sizes: list[int] | None = None) -> list[TensorSpec]:
    if sizes is None:
        sizes = [size for _, size in checkpoint.EXPERT_LAYOUT]
    prefix = "language_model.model.layers.1.block_sparse_moe.experts.0."
    return [
        (prefix + suffix, "U8", [size], size)
        for (suffix, _), size in zip(checkpoint.EXPERT_LAYOUT, sizes)
    ]


def write_model(
    root: Path,
    *,
    expert_sizes: list[int] | None = None,
    separate_expert: bool = False,
    duplicate_tensor: bool = False,
    unsafe_tensor: bool = False,
) -> None:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config()), encoding="utf-8")
    first = expert_tensors(expert_sizes)
    if separate_expert:
        first.insert(1, (
            "language_model.model.layers.1.separator.weight", "BF16", [2], 4,
        ))
    first.append(("language_model.model.norm.weight", "F32", [1], 4))
    second: list[TensorSpec] = [
        ("language_model.model.embed_tokens.weight", "BF16", [4], 8),
    ]
    if duplicate_tensor:
        second.append(first[0])
    if unsafe_tensor:
        second.append(("../escape", "BF16", [1], 2))
    first_name, second_name = checkpoint._shard_names(2)
    write_shard(root / first_name, first)
    write_shard(root / second_name, second)
    assignments = {tensor[0]: first_name for tensor in first}
    for tensor in second:
        assignments.setdefault(tensor[0], second_name)
    write_index(root, assignments)


def inspect(root: Path, **kwargs: object) -> dict[str, object]:
    return checkpoint.inspect_checkpoint(
        root,
        expected_shards=2,
        expected_routed_layers=1,
        expected_experts_per_layer=1,
        **kwargs,
    )


class CheckpointInspectionTests(unittest.TestCase):
    def test_valid_manifest_and_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root)
            manifest = inspect(root)
            self.assertEqual(manifest["schema"], checkpoint.SCHEMA)
            self.assertEqual(manifest["upstream"], {
                "id": checkpoint.UPSTREAM_ID,
                "revision": checkpoint.UPSTREAM_REVISION,
                "architecture": checkpoint.ARCHITECTURE,
                "model_type": checkpoint.MODEL_TYPE,
            })
            self.assertEqual(manifest["tensor_count"], 8)
            self.assertEqual(
                manifest["total_bytes"], checkpoint.EXPERT_SPAN_BYTES + 12
            )
            self.assertEqual(manifest["static_tensor_count"], 2)
            self.assertEqual(len(manifest["shards"]), 2)
            geometry = manifest["expert_geometry"]
            self.assertIsInstance(geometry, dict)
            assert isinstance(geometry, dict)
            self.assertEqual(geometry["expert_count"], 1)
            self.assertEqual(geometry["bytes_per_expert"], 17_547_264)
            self.assertEqual(geometry["routed_tensor_count"], 6)
            self.assertEqual(
                sum(item["bytes"] for item in geometry["tensor_layout"]),
                checkpoint.EXPERT_SPAN_BYTES,
            )
            encoded = checkpoint.canonical_json(manifest)
            self.assertEqual(json.loads(encoded), manifest)
            self.assertNotIn("\n", encoded)

    def test_full_sha256_adds_content_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root)
            manifest = inspect(root, verify_sha256=True)
            self.assertEqual(manifest["verification"], "full-sha256")
            hashes = manifest["input_sha256"]
            self.assertIsInstance(hashes, dict)
            assert isinstance(hashes, dict)
            self.assertEqual(len(hashes), 4)
            self.assertTrue(all(
                len(value) == 64 for value in hashes.values()
            ))
            for shard in manifest["shards"]:
                self.assertEqual(shard["sha256"], hashes[shard["name"]])

    def test_resume_partial_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root)
            (root / "model-00002-of-000002.safetensors.partial").write_bytes(b"x")
            with self.assertRaisesRegex(checkpoint.CheckpointError, "incomplete"):
                inspect(root)

    def test_missing_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root)
            (root / checkpoint.INDEX_NAME).unlink()
            with self.assertRaisesRegex(checkpoint.CheckpointError, "missing"):
                inspect(root)

    def test_bad_dsa_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root)
            value = config()
            text = value["text_config"]
            assert isinstance(text, dict)
            dsa = text["dsa_config"]
            assert isinstance(dsa, dict)
            dsa["index_topk"] = 1024
            (root / "config.json").write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(checkpoint.CheckpointError, "index_topk"):
                inspect(root)

    def test_missing_and_duplicate_shards_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root)
            second = root / "model-00002-of-000002.safetensors"
            second.unlink()
            with self.assertRaisesRegex(checkpoint.CheckpointError, "coverage"):
                inspect(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root)
            (root / "model-00003-of-000002.safetensors").write_bytes(b"duplicate")
            with self.assertRaisesRegex(checkpoint.CheckpointError, "coverage"):
                inspect(root)

    def test_duplicate_and_unsafe_tensor_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root, duplicate_tensor=True)
            with self.assertRaisesRegex(checkpoint.CheckpointError, "duplicate tensor"):
                inspect(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root, unsafe_tensor=True)
            with self.assertRaisesRegex(checkpoint.CheckpointError, "unsafe tensor"):
                inspect(root)

    def test_noncontiguous_expert_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_model(root, separate_expert=True)
            with self.assertRaisesRegex(checkpoint.CheckpointError, "not contiguous"):
                inspect(root)

    def test_wrong_size_expert_span_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            sizes = [size for _, size in checkpoint.EXPERT_LAYOUT]
            sizes[-1] -= 1
            write_model(root, expert_sizes=sizes)
            index_path = root / checkpoint.INDEX_NAME
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["metadata"]["total_size"] -= 1
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(checkpoint.CheckpointError, "wrong size"):
                inspect(root)

    def test_same_size_metadata_change_during_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard"
            path.write_bytes(b"abc")
            before = path.lstat()
            after = types.SimpleNamespace(
                st_mode=before.st_mode,
                st_size=before.st_size,
                st_dev=before.st_dev,
                st_ino=before.st_ino,
                st_mtime_ns=before.st_mtime_ns + 1,
            )
            with mock.patch.object(
                Path, "lstat", side_effect=[before, after]
            ):
                with self.assertRaisesRegex(
                    checkpoint.CheckpointError, "changed during SHA-256"
                ):
                    checkpoint._sha256_file(path, 3)

    def test_source_layout_digest_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            write_model(first)
            write_model(second)
            first_manifest = inspect(first)
            second_manifest = inspect(second)
            self.assertEqual(
                first_manifest["source_layout_sha256"],
                second_manifest["source_layout_sha256"],
            )
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                first_manifest["source_model_layout_crc64"],
                "0e85db34387f4e0e",
            )
            self.assertRegex(
                str(first_manifest["source_layout_sha256"]), r"^[0-9a-f]{64}$"
            )


if __name__ == "__main__":
    unittest.main()
