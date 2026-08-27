#!/usr/bin/env python3
"""Behavioral tests for the standalone MZG2 static-pack builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "tools" / "build_mzg2_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_mzg2_bundle", TOOL)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def write_shard(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    metadata: dict[str, object] = {}
    cursor = 0
    for name, dtype, shape, payload in tensors:
        metadata[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
    encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    header_bytes = bundle.align_up(8 + len(encoded), 8) - 8
    encoded += b" " * (header_bytes - len(encoded))
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", header_bytes))
        stream.write(encoded)
        for _, _, _, payload in tensors:
            stream.write(payload)


def write_model(root: Path) -> dict[str, bytes]:
    root.mkdir()
    expected = {
        "language_model.model.layers.0.input_layernorm.weight": bytes(range(16)),
        "language_model.model.embed_tokens.weight": b"embedding-payload!",
        "language_model.model.norm.weight": b"\x01\x02\x03\x04",
    }
    write_shard(
        root / "model-00001-of-000002.safetensors",
        [
            (
                "language_model.model.layers.0.input_layernorm.weight",
                "BF16",
                [8],
                expected["language_model.model.layers.0.input_layernorm.weight"],
            ),
            (
                "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
                "U8",
                [8],
                b"routed!!",
            ),
            ("vision_tower.layer.weight", "BF16", [4], b"vision!!"),
        ],
    )
    write_shard(
        root / "model-00002-of-000002.safetensors",
        [
            (
                "language_model.model.embed_tokens.weight",
                "BF16",
                [9],
                expected["language_model.model.embed_tokens.weight"],
            ),
            (
                "language_model.model.norm.weight",
                "F32",
                [1],
                expected["language_model.model.norm.weight"],
            ),
        ],
    )
    return expected


def write_mzg2_store(root: Path) -> dict[str, object]:
    root.mkdir()
    files = []
    total = 0
    for layer in range(1, 93):
        path = root / f"layer-{layer:03d}.mzg2"
        payload = f"layer-{layer:03d}".encode()
        path.write_bytes(payload)
        total += len(payload)
        files.append({
            "layer": layer,
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest = {
        "schema": "moonshine-mzg2-full-v1",
        "version": 2,
        "source_manifest_sha256": bundle.PINNED_SOURCE_MANIFEST_SHA256,
        "layers": 92,
        "experts_per_layer": 896,
        "experts": 82432,
        "tile_bytes": 16384,
        "stored_file_bytes": total,
        "files": files,
    }
    (root / "expert-store.json").write_text(
        json.dumps(manifest, sort_keys=True)
    )
    return manifest


class StaticPackTests(unittest.TestCase):


    def test_static_pack_is_deterministic_and_excludes_routed_and_vision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            expected = write_model(model)
            payload_bytes = sum(map(len, expected.values()))
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = first_root / "model-static.safetensors"
            second = second_root / "model-static.safetensors"
            first_manifest = bundle.build_static_pack(
                model,
                first,
                expected_shards=2,
                expected_tensors=3,
                expected_payload_bytes=payload_bytes,
                expected_source_manifest_sha256=None,
            )
            second_manifest = bundle.build_static_pack(
                model,
                second,
                expected_shards=2,
                expected_tensors=3,
                expected_payload_bytes=payload_bytes,
                expected_source_manifest_sha256=None,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first_manifest["source_model_layout_crc64"],
                second_manifest["source_model_layout_crc64"],
            )
            self.assertRegex(
                str(first_manifest["source_model_layout_crc64"]),
                r"^[0-9a-f]{16}$",
            )
            self.assertEqual(first_manifest["sha256"], second_manifest["sha256"])
            self.assertEqual(first_manifest["tensor_count"], 3)
            self.assertEqual(first_manifest["payload_bytes"], payload_bytes)
            data_offset, header = bundle.load_header(first)
            names = sorted(name for name in header if name != "__metadata__")
            self.assertEqual(names, sorted(expected))
            self.assertEqual(data_offset % 4096, 0)
            with first.open("rb") as stream:
                for name in names:
                    meta = header[name]
                    start, end = meta["data_offsets"]
                    stream.seek(data_offset + start)
                    self.assertEqual(stream.read(end - start), expected[name])
            verified = bundle.verify_static_pack(
                model,
                first,
                first_root / "static-store.json",
                expected_shards=2,
                expected_tensors=3,
                expected_payload_bytes=payload_bytes,
                expected_source_manifest_sha256=None,
            )
            self.assertEqual(verified["sha256"], hashlib.sha256(first.read_bytes()).hexdigest())

    def test_source_manifest_is_pinned_and_size_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            write_model(model)
            lines = []
            for shard in sorted(model.glob("model-*.safetensors")):
                lines.append(
                    f"{shard.name}\t{shard.stat().st_size}\t"
                    f"{hashlib.sha256(shard.read_bytes()).hexdigest()}\n"
                )
            manifest = model / "manifest.tsv"
            manifest.write_text("".join(lines))
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(
                bundle.validate_source_manifest(model, digest), digest
            )
            manifest.write_text(lines[0].replace(
                f"\t{sorted(model.glob('model-*.safetensors'))[0].stat().st_size}\t",
                "\t1\t",
            ) + "".join(lines[1:]))
            bad_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                bundle.validate_source_manifest(model, bad_digest)

    def test_mzg2_validation_linking_and_bundle_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-mzg2"
            linked = root / "bundle" / "expert-store-mzg2"
            output = linked.parent
            mzg2_manifest = write_mzg2_store(source)
            self.assertEqual(
                bundle.validate_mzg2_store(source)["experts"], 82432
            )
            bundle.link_mzg2_store(source, linked)
            self.assertEqual(
                (source / "layer-001.mzg2").stat().st_ino,
                (linked / "layer-001.mzg2").stat().st_ino,
            )
            static_file = output / "model-static.safetensors"
            static_file.write_bytes(b"static")
            static_manifest = {
                "path": static_file.name,
                "tensor_count": 2460,
                "payload_bytes": 113509540864,
                "file_bytes": len(b"static"),
                "sha256": hashlib.sha256(b"static").hexdigest(),
                "source_model_layout_crc64": "d17f7f2aad23c9c9",
            }
            manifest = bundle.write_bundle_manifest(
                output, static_manifest, mzg2_manifest, {}
            )
            self.assertEqual(
                manifest["source_model_layout_crc64"],
                "d17f7f2aad23c9c9",
            )
            self.assertEqual(manifest["routed_store"]["experts"], 82432)
            with (linked / "layer-001.mzg2").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                bundle.validate_mzg2_store(linked)

    def test_corruption_and_wrong_inventory_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            expected = write_model(model)
            output_root = root / "out"
            output_root.mkdir()
            output = output_root / "model-static.safetensors"
            bundle.build_static_pack(
                model,
                output,
                expected_shards=2,
                expected_tensors=3,
                expected_payload_bytes=sum(map(len, expected.values())),
                expected_source_manifest_sha256=None,
            )
            with output.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 1]))
            with self.assertRaises(ValueError):
                bundle.verify_static_pack(
                    model,
                    output,
                    output_root / "static-store.json",
                    expected_shards=2,
                    expected_tensors=3,
                    expected_payload_bytes=sum(map(len, expected.values())),
                    expected_source_manifest_sha256=None,
                )
            with self.assertRaises(ValueError):
                bundle.validate_inventory(
                    bundle.discover_static_tensors(model),
                    expected_shards=2,
                    expected_tensors=4,
                    expected_payload_bytes=None,
                )


if __name__ == "__main__":
    unittest.main()
