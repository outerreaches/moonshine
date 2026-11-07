#!/usr/bin/env python3
"""Negative tests for audit_mimo26_checkpoint.

A validator that only ever passes proves nothing, so every check gets a
synthesized fault. Builds tiny fake checkpoints in a temp dir -- no real
weights needed, so this runs anywhere.
"""
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_mimo26_checkpoint as audit


def write_shard(path, entries, *, raw_header=None, pad=0):
    """entries: {name: (dtype, shape, start, end)}."""
    if raw_header is None:
        header = {name: {"dtype": dtype, "shape": list(shape),
                         "data_offsets": [start, end]}
                  for name, (dtype, shape, start, end) in entries.items()}
        blob = json.dumps(header).encode()
    else:
        blob = raw_header
    payload = max((e[3] for e in entries.values()), default=0) + pad
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * payload)


def build(root, entries, *, weight_map=None, raw_header=None, pad=0,
          shard="model_pp0_ep0_shard0.safetensors"):
    root.mkdir(parents=True, exist_ok=True)
    write_shard(root / shard, entries, raw_header=raw_header, pad=pad)
    names = list(entries) if raw_header is None else json.loads(raw_header).keys()
    mapping = weight_map if weight_map is not None else {n: shard for n in names}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping}))


def categories(root):
    return set(audit.audit(root)["problems"].keys())


def expect(root, category, label):
    found = categories(root)
    assert category in found, f"{label}: expected {category}, got {sorted(found)}"
    print(f"  ok  {label} -> {category}")


def main():
    work = Path(tempfile.mkdtemp(prefix="mimo26-audit-test-"))
    try:
        norm = ("model.norm.weight", "BF16", [4096], 0, 8192)

        # Contract table checks, exercised directly.
        assert audit.check_contract("model.norm.weight", "BF16", [4096]) is None
        assert audit.check_contract("model.norm.weight", "F32", [4096])
        assert audit.check_contract("model.norm.weight", "BF16", [4095])
        # Layer 0 is dense and has no experts or router.
        assert audit.check_contract(
            "model.layers.0.mlp.experts.0.gate_proj.weight", "U8",
            [2048, 2048])
        assert audit.check_contract("model.layers.0.mlp.gate.weight", "BF16",
                                    [256, 4096])
        assert audit.check_contract("model.layers.1.mlp.gate.weight", "BF16",
                                    [256, 4096]) is None
        # SWA vs global geometry must not be interchangeable.
        assert audit.check_contract(
            "model.layers.5.self_attn.qkv_proj.weight", "F8_E4M3",
            [13568, 4096]) is None
        assert audit.check_contract(
            "model.layers.5.self_attn.qkv_proj.weight", "F8_E4M3",
            [14848, 4096])
        assert audit.check_contract(
            "model.layers.5.self_attn.attention_sink_bias", "BF16", [64])
        assert audit.check_contract(
            "model.layers.1.self_attn.attention_sink_bias", "BF16",
            [64]) is None
        # The global scale-grid overhang is the contract, not an exact ceil.
        assert audit.check_contract(
            "model.layers.5.self_attn.qkv_proj.weight_scale_inv", "F32",
            [108, 32]) is None
        assert audit.check_contract(
            "model.layers.5.self_attn.qkv_proj.weight_scale_inv", "F32",
            [106, 32])
        # Non-canonical indices are rejected rather than normalized.
        assert audit.check_contract("model.layers.01.input_layernorm.weight",
                                    "BF16", [4096])
        assert audit.check_contract("model.layers.48.input_layernorm.weight",
                                    "BF16", [4096])
        # MTP lives at 0..2, not continuing the text numbering.
        assert audit.check_contract("model.mtp.layers.0.enorm.weight", "BF16",
                                    [4096]) is None
        assert audit.check_contract("model.mtp.layers.48.enorm.weight", "BF16",
                                    [4096])
        # The non-periodic pattern must not be derivable from a period.
        assert audit.layer_kind(12) == "moe_swa"
        assert audit.layer_kind(11) == "moe_global"
        assert audit.layer_kind(48) == "invalid"
        assert sum(audit.SWA_PATTERN) == 39
        print("  ok  contract and pattern unit checks")

        # A minimal well-formed shard is structurally clean: it fails only the
        # coverage assertions, which is what isolates the faults injected below.
        root = work / "baseline"
        build(root, {norm[0]: norm[1:]})
        found = categories(root)
        assert found == {"coverage_mismatch", "expert_pair_count"}, sorted(found)
        print("  ok  minimal shard -> coverage failures only")

        root = work / "byte_length"
        build(root, {norm[0]: ("BF16", [4096], 0, 8191)})
        expect(root, "byte_length_mismatch", "shape/dtype vs byte span")

        root = work / "overlap"
        build(root, {
            "model.norm.weight": ("BF16", [4096], 0, 8192),
            "model.layers.1.input_layernorm.weight": ("BF16", [4096], 4096,
                                                      12288),
        })
        expect(root, "span_overlap", "overlapping spans")

        root = work / "past_end"
        build(root, {norm[0]: ("BF16", [4096], 0, 8192)})
        # Truncate the payload so the declared span runs past the file end.
        shard = root / "model_pp0_ep0_shard0.safetensors"
        data = shard.read_bytes()
        shard.write_bytes(data[:-16])
        expect(root, "offset_past_end_of_file", "span past end of file")

        root = work / "bad_range"
        build(root, {norm[0]: ("BF16", [4096], 100, 50)})
        expect(root, "offset_range_invalid", "end before start")

        root = work / "unknown_dtype"
        build(root, {norm[0]: ("F16", [4096], 0, 8192)})
        expect(root, "unknown_dtype", "unrecognized dtype")

        root = work / "duplicate_key"
        raw = (b'{"model.norm.weight": {"dtype": "BF16", "shape": [4096], '
               b'"data_offsets": [0, 8192]}, "model.norm.weight": '
               b'{"dtype": "BF16", "shape": [4096], "data_offsets": [0, 8192]}}')
        build(root, {norm[0]: norm[1:]}, raw_header=raw,
              weight_map={"model.norm.weight": "model_pp0_ep0_shard0.safetensors"})
        expect(root, "duplicate_json_key", "duplicate JSON key in header")

        root = work / "traversal"
        build(root, {norm[0]: norm[1:]},
              weight_map={"model.norm.weight": "../escape.safetensors"})
        expect(root, "shard_path_unsafe", "index escapes the checkpoint dir")

        root = work / "missing_shard"
        build(root, {norm[0]: norm[1:]},
              weight_map={"model.norm.weight": "absent.safetensors"})
        expect(root, "referenced_shard_missing", "index names a missing shard")

        root = work / "stray_shard"
        build(root, {norm[0]: norm[1:]})
        write_shard(root / "stray.safetensors", {norm[0]: norm[1:]})
        expect(root, "shard_not_referenced", "unreferenced shard on disk")

        root = work / "wrong_association"
        build(root, {norm[0]: norm[1:]},
              weight_map={"model.norm.weight": "other.safetensors"})
        write_shard(root / "other.safetensors", {})
        expect(root, "index_association_wrong", "index points at wrong shard")

        root = work / "unclassified"
        build(root, {"totally.unknown.tensor": ("BF16", [4096], 0, 8192)})
        expect(root, "unclassified_tensor", "unknown namespace")

        root = work / "contract"
        build(root, {"model.layers.1.self_attn.qkv_proj.weight":
                     ("F8_E4M3", [13568, 4096], 0, 13568 * 4096)})
        expect(root, "contract_violation", "global geometry on an SWA layer")

        root = work / "incomplete_expert"
        build(root, {"model.layers.1.mlp.experts.0.gate_proj.weight":
                     ("U8", [2048, 2048], 0, 2048 * 2048)})
        expect(root, "incomplete_expert", "expert missing five of six tensors")

        print("test_mimo26_audit: ok")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
