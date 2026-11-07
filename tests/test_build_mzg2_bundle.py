#!/usr/bin/env python3
"""Behavioral tests for the Kimi K3.1 MZG2 bundle and transcode drivers."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
sys.path.insert(0, str(TOOLS))
import k31_checkpoint as k31


def load_tool(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bundle = load_tool("build_mzg2_bundle")
transcode = load_tool("transcode_mzg2_full")


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


def write_checkpoint(
    root: Path,
    shards: list[list[tuple[str, str, list[int], bytes]]],
) -> None:
    root.mkdir()
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_names = [
        f"model-{index:05d}-of-{len(shards):06d}.safetensors"
        for index in range(1, len(shards) + 1)
    ]
    for shard_name, tensors in zip(shard_names, shards):
        write_shard(root / shard_name, tensors)
        for name, _, _, payload in tensors:
            weight_map[name] = shard_name
            total_size += len(payload)
    dsa = {"dsa_layers": k31.DSA_LAYERS, **k31.DSA_FIELDS}
    dsa_set = set(k31.DSA_LAYERS)
    config = {
        "architectures": [k31.ARCHITECTURE],
        "model_type": k31.MODEL_TYPE,
        "text_config": {
            "num_hidden_layers": 93,
            "dsa_config": dsa,
            "linear_attn_config": {
                "full_attn_layers": k31.DSA_LAYERS,
                "kda_layers": [
                    layer for layer in range(1, 94) if layer not in dsa_set
                ],
            },
        },
    }
    (root / "config.json").write_text(json.dumps(config, sort_keys=True))
    (root / k31.INDEX_NAME).write_text(json.dumps({
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }, sort_keys=True))


def write_model(
    root: Path,
    *,
    variant: bool = False,
) -> tuple[dict[str, bytes], dict[str, object]]:
    indexer = "language_model.model.layers.4.self_attn.indexer.k_norm.weight"
    embedding = "language_model.model.embed_tokens.weight"
    norm = "language_model.model.norm.weight"
    expected = {
        indexer: bytes(range(16)),
        embedding: b"embedding-payload!",
        norm: b"\x01\x02\x03\x04",
    }
    vision_shape = [5] if variant else [4]
    vision_payload = b"vision!!" + (b"xx" if variant else b"")
    write_checkpoint(root, [
        [
            (indexer, "BF16", [8], expected[indexer]),
            ("vision_tower.layer.weight", "BF16", vision_shape, vision_payload),
        ],
        [
            (embedding, "BF16", [9], expected[embedding]),
            (norm, "F32", [1], expected[norm]),
        ],
    ])
    source = bundle.inspect_source(
        root,
        expected_shards=2,
        expected_routed_layers=0,
        expected_experts_per_layer=0,
    )
    return expected, source


def write_resume_model(root: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    tensors = []
    expected = {}
    for index in range(17):
        name = f"language_model.model.layers.{index}.input_layernorm.weight"
        payload = bytes([index, 255 - index])
        expected[name] = payload
        tensors.append((name, "BF16", [1], payload))
    write_checkpoint(root, [tensors])
    source = bundle.inspect_source(
        root,
        expected_shards=1,
        expected_routed_layers=0,
        expected_experts_per_layer=0,
    )
    return expected, source


def synthetic_source(layers: int = 2, experts: int = 2) -> dict[str, object]:
    layout = []
    offset = 0
    for name, size in k31.EXPERT_LAYOUT:
        layout.append({
            "name": name,
            "dtype": "U8",
            "shape": [size],
            "offset": offset,
            "bytes": size,
        })
        offset += size
    expert_count = layers * experts
    routed_tensors = expert_count * len(k31.EXPERT_LAYOUT)
    routed_bytes = expert_count * k31.EXPERT_SPAN_BYTES
    shard_name = "model-00001-of-000001.safetensors"
    total_bytes = routed_bytes + 2
    digest = hashlib.sha256(
        k31.canonical_json({"synthetic-k31": [layers, experts]}).encode()
    ).hexdigest()
    return {
        "schema": k31.SCHEMA,
        "upstream": {
            "id": k31.UPSTREAM_ID,
            "revision": k31.UPSTREAM_REVISION,
            "architecture": k31.ARCHITECTURE,
            "model_type": k31.MODEL_TYPE,
        },
        "verification": "metadata-only",
        "tensor_count": routed_tensors + 1,
        "total_bytes": total_bytes,
        "file_bytes": total_bytes + 8,
        "static_tensor_count": 1,
        "static_bytes": 2,
        "static_tensors": [{
            "name": "language_model.model.norm.weight",
            "dtype": "BF16",
            "shape": [1],
            "bytes": 2,
            "source_shard": shard_name,
            "data_offsets": [0, 2],
            "source_offset": 8,
        }],
        "expert_geometry": {
            "routed_layers": layers,
            "experts_per_layer": experts,
            "expert_count": expert_count,
            "tensors_per_expert": len(k31.EXPERT_LAYOUT),
            "bytes_per_expert": k31.EXPERT_SPAN_BYTES,
            "routed_tensor_count": routed_tensors,
            "routed_bytes": routed_bytes,
            "tensor_layout": layout,
        },
        "shards": [{
            "name": shard_name,
            "bytes": total_bytes + 8,
            "header_bytes": 0,
            "data_offset": 8,
            "tensor_count": routed_tensors + 1,
        }],
        "source_layout_sha256": digest,
        "source_model_layout_crc64": "0123456789abcdef",
    }


def write_mzg2_store(
    root: Path,
    source: dict[str, object],
) -> dict[str, object]:
    root.mkdir()
    geometry = source["expert_geometry"]
    assert isinstance(geometry, dict)
    layers = int(geometry["routed_layers"])
    experts_per_layer = int(geometry["experts_per_layer"])
    files = []
    total = 0
    for layer in range(1, layers + 1):
        path = root / f"layer-{layer:03d}.mzg2"
        payload = f"k31-layer-{layer:03d}".encode()
        path.write_bytes(payload)
        total += len(payload)
        files.append({
            "layer": layer,
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest = {
        "schema": bundle.MZG2_STORE_SCHEMA,
        "version": 2,
        "base_model": k31.UPSTREAM_ID,
        "base_revision": k31.UPSTREAM_REVISION,
        "source_layout_sha256": source["source_layout_sha256"],
        "layers": layers,
        "experts_per_layer": experts_per_layer,
        "experts": layers * experts_per_layer,
        "expert_span_bytes": k31.EXPERT_SPAN_BYTES,
        "tile_bytes": 16_384,
        "raw_bytes": layers * experts_per_layer * k31.EXPERT_SPAN_BYTES,
        "stored_file_bytes": total,
        "files": files,
    }
    (root / "expert-store.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest


FAKE_TRANSCODER = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import struct
import sys

state = Path(__file__).resolve().parent
if len(sys.argv) == 4 and sys.argv[1] == "--verify":
    raise SystemExit(0)
if len(sys.argv) != 6:
    raise SystemExit(2)
output = Path(sys.argv[2])
layer = int(sys.argv[3])
experts = int(sys.argv[4])
digest = bytes.fromhex(sys.argv[5])
with (state / "invocations.jsonl").open("a") as stream:
    stream.write(json.dumps({"layer": layer}) + "\n")
marker = state / "failed-layer-2"
if layer == 2 and not marker.exists():
    marker.write_text("interrupted")
    raise SystemExit(17)
header = struct.Struct("<8s9I6Q32s")
entry = struct.Struct("<HHIQ")
index_offset = 4096
index_bytes = experts * entry.size
data_offset = (index_offset + index_bytes + 4095) // 4096 * 4096
block_bytes = 4096
values = (
    b"K3MZG2\0\0", 2, 4096, 4096, layer, experts, entry.size, 0,
    16384, 0, index_offset, index_bytes, 0, 0, data_offset, block_bytes,
    digest,
)
with output.open("wb") as stream:
    stream.write(header.pack(*values))
    stream.seek(index_offset)
    for expert in range(experts):
        stream.write(entry.pack(
            layer, expert, block_bytes, data_offset + expert * block_bytes
        ))
    stream.seek(data_offset)
    stream.write(bytes([layer]) * (experts * block_bytes))
    stream.flush()
    os.fsync(stream.fileno())
'''


def write_fake_transcoder(root: Path) -> Path:
    path = root / "fake-transcoder.py"
    path.write_text(FAKE_TRANSCODER)
    path.chmod(0o755)
    return path


class StaticPackTests(unittest.TestCase):
    def test_static_pack_is_manifest_driven_and_preserves_dsa_bf16(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            expected, source = write_model(model)
            payload_bytes = sum(map(len, expected.values()))
            self.assertEqual(source["static_tensor_count"], len(expected))
            self.assertEqual(source["static_bytes"], payload_bytes)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = first_root / "model-static.safetensors"
            second = second_root / "model-static.safetensors"
            first_manifest = bundle.build_static_pack(
                model, first, source_manifest=source
            )
            second_manifest = bundle.build_static_pack(
                model, second, source_manifest=source
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_manifest["sha256"], second_manifest["sha256"])
            self.assertEqual(first_manifest["schema"], "moonshine-k31-static-pack-v1")
            self.assertEqual(first_manifest["base_revision"], k31.UPSTREAM_REVISION)
            self.assertEqual(
                first_manifest["source_layout_sha256"],
                source["source_layout_sha256"],
            )
            self.assertEqual(first_manifest["source_tensor_count"], source["tensor_count"])
            self.assertEqual(first_manifest["source_shard_count"], 2)
            self.assertEqual(first_manifest["tensor_count"], len(expected))
            self.assertEqual(first_manifest["payload_bytes"], payload_bytes)
            data_offset, header = bundle.load_header(first)
            self.assertEqual(
                sorted(name for name in header if name != "__metadata__"),
                sorted(expected),
            )
            self.assertEqual(data_offset % 4096, 0)
            indexer = "language_model.model.layers.4.self_attn.indexer.k_norm.weight"
            self.assertEqual(header[indexer]["dtype"], "BF16")
            source_indexer = next(
                record for record in source["static_tensors"]
                if record["name"] == indexer
            )
            self.assertEqual(source_indexer["dtype"], "BF16")
            with first.open("rb") as stream:
                for name in expected:
                    meta = header[name]
                    start, end = meta["data_offsets"]
                    stream.seek(data_offset + start)
                    self.assertEqual(stream.read(end - start), expected[name])
            verified = bundle.verify_static_pack(
                model,
                first,
                first_root / "static-store.json",
                source_manifest=source,
            )
            self.assertEqual(verified["sha256"], hashlib.sha256(first.read_bytes()).hexdigest())

    def test_static_resume_reuses_fsynced_completed_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            expected, source = write_resume_model(model)
            output = root / "out" / "model-static.safetensors"
            original_copy = bundle.copy_tensor
            calls = 0

            def interrupt_at_seventeen(*args):
                nonlocal calls
                calls += 1
                digest = original_copy(*args)
                if calls == 17:
                    raise RuntimeError("simulated interruption")
                return digest

            bundle.copy_tensor = interrupt_at_seventeen
            try:
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    bundle.build_static_pack(model, output, source_manifest=source)
            finally:
                bundle.copy_tensor = original_copy
            progress_path = output.with_name(output.name + ".progress.json")
            progress = json.loads(progress_path.read_text())
            self.assertEqual(len(progress["completed"]), 16)
            self.assertTrue(output.with_name(output.name + ".partial").is_file())

            resumed_calls = 0

            def count_resume(*args):
                nonlocal resumed_calls
                resumed_calls += 1
                return original_copy(*args)

            bundle.copy_tensor = count_resume
            try:
                manifest = bundle.build_static_pack(
                    model, output, source_manifest=source
                )
            finally:
                bundle.copy_tensor = original_copy
            self.assertEqual(resumed_calls, 1)
            self.assertEqual(manifest["tensor_count"], 17)
            self.assertFalse(progress_path.exists())
            bundle.verify_static_pack(
                model,
                output,
                output.parent / "static-store.json",
                source_manifest=source,
            )
            data_offset, header = bundle.load_header(output)
            with output.open("rb") as stream:
                for name, payload in expected.items():
                    start, end = header[name]["data_offsets"]
                    stream.seek(data_offset + start)
                    self.assertEqual(stream.read(end - start), payload)

    def test_corruption_and_source_identity_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            expected, source = write_model(model)
            output = root / "out" / "model-static.safetensors"
            bundle.build_static_pack(model, output, source_manifest=source)
            with output.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 1]))
            with self.assertRaises(ValueError):
                bundle.verify_static_pack(
                    model,
                    output,
                    output.parent / "static-store.json",
                    source_manifest=source,
                )

            # A valid new inspection with a different source layout cannot reuse it.
            second_model = root / "second-model"
            second_expected, second_source = write_model(second_model, variant=True)
            self.assertEqual(second_expected, expected)
            self.assertNotEqual(
                source["source_layout_sha256"], second_source["source_layout_sha256"]
            )
            with self.assertRaisesRegex(ValueError, "source layout mismatch"):
                bundle.verify_static_pack(
                    second_model,
                    output,
                    output.parent / "static-store.json",
                    source_manifest=second_source,
                )

    def test_k3_checkpoint_and_static_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            _, source = write_model(model)
            config_path = model / "config.json"
            config = json.loads(config_path.read_text())
            config["architectures"] = ["KimiK3ForConditionalGeneration"]
            config["model_type"] = "kimi_k3"
            config_path.write_text(json.dumps(config))
            with self.assertRaises(k31.CheckpointError):
                bundle.build_static_pack(
                    model,
                    root / "k3.safetensors",
                    expected_shards=2,
                    expected_routed_layers=0,
                    expected_experts_per_layer=0,
                )

            output = root / "static" / "model-static.safetensors"
            bundle.build_static_pack(model, output, source_manifest=source)
            manifest_path = output.parent / "static-store.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema"] = "moonshine-k3-static-pack-v1"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "invalid K3.1 static-store"):
                bundle.verify_static_pack(
                    model, output, manifest_path, source_manifest=source
                )


class BundleAndTranscodeTests(unittest.TestCase):
    def test_mzg2_linking_bundle_fields_and_cross_generation_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = synthetic_source()
            store = root / "source-mzg2"
            linked = root / "bundle" / "expert-store-mzg2"
            output = linked.parent
            mzg2_manifest = write_mzg2_store(store, source)
            validated = bundle.validate_mzg2_store(
                store,
                source_layout_sha256=str(source["source_layout_sha256"]),
                expected_layers=2,
                expected_experts_per_layer=2,
            )
            self.assertEqual(validated["experts"], 4)
            bundle.link_mzg2_store(store, linked, 2)
            self.assertEqual(
                (store / "layer-002.mzg2").stat().st_ino,
                (linked / "layer-002.mzg2").stat().st_ino,
            )
            static_file = output / "model-static.safetensors"
            static_file.write_bytes(b"static")
            static_manifest = {
                "path": static_file.name,
                "tensor_count": source["static_tensor_count"],
                "payload_bytes": source["static_bytes"],
                "file_bytes": len(b"static"),
                "sha256": hashlib.sha256(b"static").hexdigest(),
                "source_layout_sha256": source["source_layout_sha256"],
            }
            manifest = bundle.write_bundle_manifest(
                output, static_manifest, mzg2_manifest, {}, source
            )
            self.assertEqual(manifest["schema"], "moonshine-k31-mzg2-bundle-v1")
            self.assertEqual(manifest["base_model"], k31.UPSTREAM_ID)
            self.assertEqual(manifest["base_revision"], k31.UPSTREAM_REVISION)
            self.assertEqual(
                manifest["source_manifest_sha256"],
                source["source_layout_sha256"],
            )
            self.assertEqual(
                manifest["source_model_layout_crc64"],
                source["source_model_layout_crc64"],
            )
            self.assertEqual(manifest["source_tensor_count"], source["tensor_count"])
            self.assertEqual(manifest["source_shard_count"], 1)
            self.assertEqual(
                manifest["static_store"]["tensor_count"],
                source["static_tensor_count"],
            )
            self.assertEqual(manifest["routed_store"]["experts"], 4)
            self.assertEqual(
                manifest["expert_layout"]["bytes"], k31.EXPERT_SPAN_BYTES
            )
            self.assertEqual(
                manifest["expert_layout"]["tensors"][-1]["offset"]
                    + manifest["expert_layout"]["tensors"][-1]["bytes"],
                k31.EXPERT_SPAN_BYTES,
            )
            with (linked / "layer-001.mzg2").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                bundle.validate_mzg2_store(
                    linked,
                    source_layout_sha256=str(source["source_layout_sha256"]),
                    expected_layers=2,
                    expected_experts_per_layer=2,
                )

            stale = json.loads((store / "expert-store.json").read_text())
            stale["source_layout_sha256"] = "0" * 64
            (store / "expert-store.json").write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, "invalid or stale K3.1"):
                bundle.validate_mzg2_store(
                    store,
                    source_layout_sha256=str(source["source_layout_sha256"]),
                    expected_layers=2,
                    expected_experts_per_layer=2,
                )
            stale["schema"] = "moonshine-mzg2-full-v1"
            (store / "expert-store.json").write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, "invalid or stale K3.1"):
                bundle.validate_mzg2_store(
                    store,
                    source_layout_sha256="0" * 64,
                    expected_layers=2,
                    expected_experts_per_layer=2,
                )

            k3_bundle = root / "k3-bundle"
            k3_bundle.mkdir()
            (k3_bundle / "moonshine-bundle.json").write_text(json.dumps({
                "schema": "moonshine-k3-mzg2-bundle-v1",
                "version": 1,
            }))
            with self.assertRaisesRegex(ValueError, "invalid K3.1 standalone bundle"):
                bundle.verify_bundle(k3_bundle, model=None, verify_hashes=False)

    def test_interrupted_transcode_recovers_verified_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = write_fake_transcoder(root)
            source = synthetic_source()
            model = root / "synthetic-model"
            model.mkdir()
            output = root / "expert-store-mzg2"
            with self.assertRaisesRegex(RuntimeError, r"command failed \(17\)"):
                transcode.transcode_store(
                    model,
                    output,
                    jobs=1,
                    tool=tool,
                    source_manifest=source,
                )
            partial = output.with_name(output.name + ".partial")
            progress = json.loads((partial / "progress.json").read_text())
            self.assertIn("1", progress["layers"])
            self.assertNotIn("2", progress["layers"])
            self.assertTrue((partial / "layer-001.mzg2").is_file())

            manifest = transcode.transcode_store(
                model,
                output,
                jobs=1,
                tool=tool,
                source_manifest=source,
            )
            self.assertEqual(manifest["schema"], transcode.STORE_SCHEMA)
            self.assertEqual(manifest["base_revision"], k31.UPSTREAM_REVISION)
            self.assertEqual(manifest["source_layout_sha256"], source["source_layout_sha256"])
            self.assertEqual(manifest["experts"], 4)
            self.assertEqual(manifest["expert_span_bytes"], k31.EXPERT_SPAN_BYTES)
            invocations = [
                json.loads(line)["layer"]
                for line in (root / "invocations.jsonl").read_text().splitlines()
            ]
            self.assertEqual(invocations, [1, 2, 2])
            bundle.validate_mzg2_store(
                output,
                source_layout_sha256=str(source["source_layout_sha256"]),
                expected_layers=2,
                expected_experts_per_layer=2,
            )

    def test_transcoder_refuses_inexact_expert_span_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = write_fake_transcoder(root)
            source = synthetic_source()
            geometry = source["expert_geometry"]
            assert isinstance(geometry, dict)
            bad_source = copy.deepcopy(source)
            bad_geometry = bad_source["expert_geometry"]
            assert isinstance(bad_geometry, dict)
            bad_geometry["bytes_per_expert"] = k31.EXPERT_SPAN_BYTES + 1
            with self.assertRaisesRegex(ValueError, "expert span mismatch"):
                transcode.transcode_store(
                    root,
                    root / "out",
                    jobs=1,
                    tool=tool,
                    source_manifest=bad_source,
                )
            self.assertFalse((root / "invocations.jsonl").exists())
            self.assertFalse((root / "out.partial").exists())
            self.assertEqual(geometry["bytes_per_expert"], k31.EXPERT_SPAN_BYTES)


if __name__ == "__main__":
    unittest.main()
