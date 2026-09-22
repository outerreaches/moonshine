#!/usr/bin/env python3
"""Compare the C dequantizer against the independent NumPy one, bit-exactly.

Both sides decode the same checkpoint bytes and round F32 to BF16 with
round-to-nearest-even, so the BF16 output must be bit-identical. The comparison
is an FNV-1a checksum over those bytes rather than a tolerance: a tolerance
would hide exactly the kind of disagreement this is meant to catch -- a wrong
nibble order, a mis-associated scale block, or a different rounding rule.

The NumPy path is written from the OCP bit fields and shares no constants with
the C implementation.

  MIMO26_ROOT=/path/to/checkpoint python3 tests/test_mimo26_weights_vs_reference.py
"""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOOL = ROOT / "tools" / "mimo26_dump_weights"
DEFAULT_ROOT = "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL"

FP8_BLOCK = 128
TP_SIZE = 4  # metadata.tp_size; the checkpoint is a TP=4 save (see fp8_qkv)
MXFP4_BLOCK = 32


def fnv1a(data):
    hash_value = 1469598103934665603
    mask = (1 << 64) - 1
    for byte in memoryview(data).cast("B"):
        hash_value = ((hash_value ^ byte) * 1099511628211) & mask
    return hash_value


def fnv1a_fast(data):
    """Same result as fnv1a, but vectorized over 8-bit lanes is not possible
    for a sequential hash, so chunk through bytes() for speed."""
    hash_value = 1469598103934665603
    mask = (1 << 64) - 1
    prime = 1099511628211
    for byte in bytes(data):
        hash_value = ((hash_value ^ byte) * prime) & mask
    return hash_value


def e2m1_table():
    values = np.zeros(16, dtype=np.float32)
    for code in range(16):
        sign = -1.0 if (code & 0x8) else 1.0
        exponent = (code >> 1) & 0x3
        mantissa = code & 0x1
        if exponent == 0:
            magnitude = mantissa * 0.5
        else:
            magnitude = (1.0 + 0.5 * mantissa) * (2.0 ** (exponent - 1))
        values[code] = sign * magnitude
    return values


def e4m3_table():
    values = np.zeros(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if (code >> 7) else 1.0
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0xF and mantissa == 0x7:
            values[code] = np.nan
            continue
        if exponent == 0:
            magnitude = mantissa * (2.0 ** -9)
        else:
            magnitude = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values[code] = sign * magnitude
    return values


def f32_to_bf16_bits(values):
    """Round-to-nearest-even, matching mimo26_f32_to_bf16 and hip_bfloat16."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    lower = bits & 0x0000FFFF
    upper = bits >> 16
    round_up = (lower > 0x8000) | ((lower == 0x8000) & ((upper & 1) == 1))
    return (upper + round_up.astype(np.uint32)).astype(np.uint16)


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        index = json.loads(
            (self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers = {}

    def _header(self, shard):
        if shard not in self._headers:
            with open(self.root / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self._headers[shard] = (json.loads(handle.read(length)),
                                        8 + length)
        return self._headers[shard]

    def raw(self, name):
        shard = self.weight_map[name]
        header, base = self._header(shard)
        entry = header[name]
        start, end = entry["data_offsets"]
        with open(self.root / shard, "rb") as handle:
            handle.seek(base + start)
            blob = handle.read(end - start)
        return blob, entry["dtype"], list(entry["shape"])

    def bf16_bits(self, name):
        blob, dtype, shape = self.raw(name)
        assert dtype == "BF16", f"{name} is {dtype}"
        return np.frombuffer(blob, dtype=np.uint16).copy()

    def fp8_qkv_bf16_bits(self, name, heads, kv_heads, head_dim, v_head_dim,
                          tp=TP_SIZE):
        """TP-sharded QKV: TP_SIZE rank slices, each already [Q | K | V], with
        the FP8 block scales tiling each slice padded to a whole block."""
        blob, dtype, shape = self.raw(name)
        assert dtype == "F8_E4M3", f"{name} is {dtype}"
        codes = np.frombuffer(blob, dtype=np.uint8).reshape(shape)
        scale_blob, scale_dtype, scale_shape = self.raw(name + "_scale_inv")
        assert scale_dtype == "F32"
        scales = np.frombuffer(scale_blob, dtype=np.float32).reshape(scale_shape)
        rows, cols = shape
        q_rows = (heads // tp) * head_dim
        k_rows = (kv_heads // tp) * head_dim
        v_rows = (kv_heads // tp) * v_head_dim
        rows_per_rank = q_rows + k_rows + v_rows
        assert rows_per_rank * tp == rows, f"{name}: {rows} vs {tp}x{rows_per_rank}"
        per = (rows_per_rank + FP8_BLOCK - 1) // FP8_BLOCK
        assert per * tp == scale_shape[0], f"{name}: {scale_shape[0]} vs {tp}x{per}"
        values = e4m3_table()[codes]
        assert not np.isnan(values).any()
        bc = (cols + FP8_BLOCK - 1) // FP8_BLOCK
        qs, ks, vs = [], [], []
        for s in range(tp):
            r0 = s * rows_per_rank
            grid = scales[s * per:(s + 1) * per, :bc]
            exp = np.repeat(np.repeat(grid, FP8_BLOCK, axis=0), FP8_BLOCK,
                            axis=1)[:rows_per_rank, :cols]
            blk = values[r0:r0 + rows_per_rank] * exp
            qs.append(blk[:q_rows])
            ks.append(blk[q_rows:q_rows + k_rows])
            vs.append(blk[q_rows + k_rows:])
        out = np.concatenate(qs + ks + vs, axis=0).astype(np.float32)
        return f32_to_bf16_bits(out).ravel()

    def fp8_bf16_bits(self, name):
        blob, dtype, shape = self.raw(name)
        assert dtype == "F8_E4M3", f"{name} is {dtype}"
        codes = np.frombuffer(blob, dtype=np.uint8).reshape(shape)
        scale_blob, scale_dtype, scale_shape = self.raw(name + "_scale_inv")
        assert scale_dtype == "F32"
        scales = np.frombuffer(scale_blob, dtype=np.float32).reshape(scale_shape)
        rows, cols = shape
        values = e4m3_table()[codes]
        assert not np.isnan(values).any(), f"{name} has an FP8 NaN"
        block_rows = (rows + FP8_BLOCK - 1) // FP8_BLOCK
        block_cols = (cols + FP8_BLOCK - 1) // FP8_BLOCK
        # Surplus scale rows are ignored, which the global QKV grid needs.
        live = scales[:block_rows, :block_cols]
        expanded = np.repeat(np.repeat(live, FP8_BLOCK, axis=0), FP8_BLOCK,
                             axis=1)[:rows, :cols]
        return f32_to_bf16_bits((values * expanded).astype(np.float32)).ravel()

    def mxfp4_bf16_bits(self, prefix):
        blob, dtype, shape = self.raw(prefix + ".weight")
        assert dtype == "U8", f"{prefix}.weight is {dtype}"
        packed = np.frombuffer(blob, dtype=np.uint8).reshape(shape)
        scale_blob, scale_dtype, scale_shape = self.raw(prefix + ".weight_scale")
        assert scale_dtype == "U8"
        scales = np.frombuffer(scale_blob, dtype=np.uint8).reshape(scale_shape)
        rows, packed_cols = shape
        cols = packed_cols * 2
        codes = np.empty((rows, cols), dtype=np.uint8)
        codes[:, 0::2] = packed & 0x0F
        codes[:, 1::2] = packed >> 4
        values = e2m1_table()[codes]
        assert not (scales == 0xFF).any(), f"{prefix} has an E8M0 NaN scale"
        exponents = scales.astype(np.int32) - 127
        block = np.ldexp(np.ones_like(exponents, dtype=np.float32),
                         exponents).astype(np.float32)
        block = np.where(scales == 0, np.float32(2.0 ** -127), block)
        expanded = np.repeat(block, MXFP4_BLOCK, axis=1)
        return f32_to_bf16_bits((values * expanded).astype(np.float32)).ravel()


def parse_tool(output):
    checksums = {}
    for line in output.splitlines():
        if "fnv1a=" not in line:
            continue
        parts = line.split()
        name = parts[0]
        for part in parts:
            if part.startswith("fnv1a="):
                checksums[name] = int(part.split("=", 1)[1], 16)
    return checksums


def main():
    root = Path(os.environ.get("MIMO26_ROOT", DEFAULT_ROOT))
    if not (root / "model.safetensors.index.json").exists():
        print(f"skip: no checkpoint at {root}")
        return 0
    if not TOOL.exists():
        print(f"skip: {TOOL} not built (make tools/mimo26_dump_weights)")
        return 0

    layer = int(os.environ.get("MIMO26_LAYER", "1"))
    expert = int(os.environ.get("MIMO26_EXPERT", "0"))
    completed = subprocess.run(
        [str(TOOL), str(root), str(layer), str(expert)],
        check=True, capture_output=True, text=True)
    checksums = parse_tool(completed.stdout)
    if not checksums:
        print("failed to parse tool output", file=sys.stderr)
        print(completed.stdout, file=sys.stderr)
        return 1

    checkpoint = Checkpoint(root)
    prefix = f"model.layers.{layer}"
    expected = {}

    expected["input_layernorm"] = checkpoint.bf16_bits(
        f"{prefix}.input_layernorm.weight")
    expected["post_attention_layernorm"] = checkpoint.bf16_bits(
        f"{prefix}.post_attention_layernorm.weight")
    expected["o_proj"] = checkpoint.bf16_bits(
        f"{prefix}.self_attn.o_proj.weight")
    swa = layer not in (0, 5, 11, 17, 23, 29, 35, 41, 47)
    expected["qkv_proj"] = checkpoint.fp8_qkv_bf16_bits(
        f"{prefix}.self_attn.qkv_proj.weight", 64, 8 if swa else 4, 192, 128)
    if "attention_sink_bias" in checksums:
        expected["attention_sink_bias"] = checkpoint.bf16_bits(
            f"{prefix}.self_attn.attention_sink_bias")
    if "mlp.gate.weight" in checksums:
        expected["mlp.gate.weight"] = checkpoint.bf16_bits(
            f"{prefix}.mlp.gate.weight")
    if "expert.gate_proj" in checksums:
        base = f"{prefix}.mlp.experts.{expert}"
        expected["expert.gate_proj"] = checkpoint.mxfp4_bf16_bits(
            f"{base}.gate_proj")
        expected["expert.up_proj"] = checkpoint.mxfp4_bf16_bits(
            f"{base}.up_proj")
        expected["expert.down_proj"] = checkpoint.mxfp4_bf16_bits(
            f"{base}.down_proj")

    failures = []
    print(f"layer {layer}, expert {expert}: comparing {len(expected)} tensors "
          f"bit-exactly")
    for name, bits in expected.items():
        mine = fnv1a_fast(bits.tobytes())
        theirs = checksums.get(name)
        if theirs is None:
            failures.append(f"{name}: tool produced no checksum")
            continue
        if mine != theirs:
            failures.append(f"{name}: numpy {mine:016x} != C {theirs:016x} "
                            f"over {bits.size} values")
        else:
            print(f"  ok  {name:<26} {bits.size:>11} values  "
                  f"{mine:016x}")

    if failures:
        print("\nfailures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("test_mimo26_weights_vs_reference: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
