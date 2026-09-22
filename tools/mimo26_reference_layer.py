#!/usr/bin/env python3
"""Run one real MiMo layer through the checkpoint's own reference modules.

This is the independent full-layer reference. Generic `from_pretrained` cannot
load this checkpoint because nothing in the reference code dequantizes the
packed MXFP4 experts, but that only blocks automatic loading: dequantizing with
our own verified decoders and injecting the tensors into the reference classes
runs the author's arithmetic on real weights.

Two deliberate mechanics, both documented rather than hidden:

  * MiMoV2MoE.__init__ would allocate all 256 experts (about 12.9 GiB in
    BF16). Its moe() loop only invokes experts that own at least one token, so
    the object is built without __init__ and given a plain list holding real
    MLPs only at the routed indices. The routing and combination code that
    runs is the reference's, unmodified.
  * The decoder layer is composed here following the reference's own forward
    (pre-norm, attention, residual, post-norm, MoE, residual) because
    MiMoV2DecoderLayer would construct the full MoE. Each component is the
    reference's.

Dequantization is independent of the C implementation: MXFP4 from the OCP bit
fields, FP8 E4M3 likewise, so agreement is not a shared-constant artifact.

  python3 tools/mimo26_reference_layer.py --layer 1 --tokens 3 --out fixture.bin
"""
import argparse
import hashlib
import json
import struct
import sys
import warnings
from pathlib import Path

import numpy as np

DEFAULT_ROOT = "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL"


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
    """All 256 OCP E4M3FN codes; the two NaN patterns become NaN."""
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


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
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
        if len(blob) != end - start:
            raise RuntimeError(f"short read for {name}")
        return blob, entry["dtype"], list(entry["shape"])

    def bf16(self, name):
        blob, dtype, shape = self.raw(name)
        if dtype != "BF16":
            raise RuntimeError(f"{name} is {dtype}, expected BF16")
        bits = np.frombuffer(blob, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape)

    def f32(self, name):
        blob, dtype, shape = self.raw(name)
        if dtype != "F32":
            raise RuntimeError(f"{name} is {dtype}, expected F32")
        return np.frombuffer(blob, dtype=np.float32).reshape(shape)

    def fp8_block(self, name):
        """Dequantize an F8_E4M3 matrix with its 128x128 weight_scale_inv."""
        blob, dtype, shape = self.raw(name)
        if dtype != "F8_E4M3":
            raise RuntimeError(f"{name} is {dtype}, expected F8_E4M3")
        codes = np.frombuffer(blob, dtype=np.uint8).reshape(shape)
        scale_blob, scale_dtype, scale_shape = self.raw(name + "_scale_inv")
        if scale_dtype != "F32":
            raise RuntimeError(f"{name}_scale_inv is {scale_dtype}")
        scales = np.frombuffer(scale_blob, dtype=np.float32).reshape(scale_shape)

        rows, cols = shape
        values = e4m3_table()[codes]
        if np.isnan(values).any():
            raise RuntimeError(f"{name} contains an FP8 NaN encoding")
        # Flat r/128, c/128 indexing. The global QKV grid is over-provisioned
        # ([108,32] against 106 needed) and the surplus rows are simply unused.
        block_rows = (rows + 127) // 128
        block_cols = (cols + 127) // 128
        if scale_shape[0] < block_rows or scale_shape[1] < block_cols:
            raise RuntimeError(f"{name}_scale_inv {scale_shape} too small for "
                               f"{shape}")
        expanded = np.repeat(np.repeat(scales[:block_rows, :block_cols], 128,
                                       axis=0), 128, axis=1)[:rows, :cols]
        return (values * expanded).astype(np.float32)

    def mxfp4_expert(self, prefix):
        """Dequantize one packed expert projection plus its E8M0 scales."""
        blob, dtype, shape = self.raw(prefix + ".weight")
        if dtype != "U8":
            raise RuntimeError(f"{prefix}.weight is {dtype}, expected U8")
        packed = np.frombuffer(blob, dtype=np.uint8).reshape(shape)
        scale_blob, scale_dtype, scale_shape = self.raw(prefix + ".weight_scale")
        if scale_dtype != "U8":
            raise RuntimeError(f"{prefix}.weight_scale is {scale_dtype}")
        scales = np.frombuffer(scale_blob, dtype=np.uint8).reshape(scale_shape)

        rows, packed_cols = shape
        cols = packed_cols * 2
        codes = np.empty((rows, cols), dtype=np.uint8)
        # Element 2k is the low nibble of byte k, per SGLang's interleave.
        # MIMO26_SWAP_NIBBLES=1 tests the opposite convention end to end.
        import os
        if os.environ.get("MIMO26_SWAP_NIBBLES") == "1":
            codes[:, 0::2] = packed >> 4
            codes[:, 1::2] = packed & 0x0F
        else:
            codes[:, 0::2] = packed & 0x0F
            codes[:, 1::2] = packed >> 4
        values = e2m1_table()[codes]
        if (scales == 0xFF).any():
            raise RuntimeError(f"{prefix} has an E8M0 NaN scale byte")
        exponents = scales.astype(np.int32) - 127
        block_scales = np.ldexp(np.ones_like(exponents, dtype=np.float32),
                                exponents).astype(np.float32)
        block_scales = np.where(scales == 0, np.float32(2.0 ** -127),
                                block_scales)
        return (values * np.repeat(block_scales, 32, axis=1)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    import shutil
    import tempfile
    import torch

    root = Path(args.root)
    if not (root / "modeling_mimo_v2.py").exists():
        print(f"skip: no checkpoint at {root}")
        return 0

    workdir = Path(tempfile.mkdtemp(prefix="mimo26-reflayer-"))
    try:
        package = workdir / "mimo_reference"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name in ("modeling_mimo_v2.py", "configuration_mimo_v2.py"):
            shutil.copy2(root / name, package / name)
        sys.path.insert(0, str(workdir))
        from mimo_reference.configuration_mimo_v2 import MiMoV2Config
        import mimo_reference.modeling_mimo_v2 as reference
        import torch.nn as nn

        raw_config = json.loads((root / "config.json").read_text())
        config = MiMoV2Config(**raw_config)
        layer = args.layer
        is_swa = bool(config.hybrid_layer_pattern[layer])
        is_moe = bool(config.moe_layer_freq[layer])
        checkpoint = Checkpoint(root)
        prefix = f"model.layers.{layer}"

        print(f"layer {layer}: {'SWA' if is_swa else 'GLOBAL'} attention, "
              f"{'MoE' if is_moe else 'dense'} MLP")

        def to_bf16(array):
            return torch.from_numpy(np.ascontiguousarray(array)).to(torch.bfloat16)

        # ---- norms ----
        input_norm = reference.MiMoV2RMSNorm(config.hidden_size,
                                             eps=config.layernorm_epsilon)
        post_norm = reference.MiMoV2RMSNorm(config.hidden_size,
                                            eps=config.layernorm_epsilon)
        input_norm.eval()
        post_norm.eval()
        with torch.no_grad():
            input_norm.weight.data = to_bf16(
                checkpoint.bf16(f"{prefix}.input_layernorm.weight"))
            post_norm.weight.data = to_bf16(
                checkpoint.bf16(f"{prefix}.post_attention_layernorm.weight"))

        # ---- attention ----
        attention = reference.MiMoV2Attention(
            config, is_swa, layer, projection_layout="fused_qkv")
        attention.eval()
        with torch.no_grad():
            attention.qkv_proj.weight.data = to_bf16(
                checkpoint.fp8_block(f"{prefix}.self_attn.qkv_proj.weight"))
            attention.o_proj.weight.data = to_bf16(
                checkpoint.bf16(f"{prefix}.self_attn.o_proj.weight"))
            if attention.attention_sink_bias is not None:
                attention.attention_sink_bias.data = to_bf16(
                    checkpoint.bf16(f"{prefix}.self_attn.attention_sink_bias"))
        print(f"  attention: qkv {tuple(attention.qkv_proj.weight.shape)}, "
              f"o_proj {tuple(attention.o_proj.weight.shape)}, "
              f"sink {attention.attention_sink_bias is not None}")

        # ---- router, then only the routed experts ----
        torch.manual_seed(args.seed)
        hidden = (torch.randn(1, args.tokens, config.hidden_size,
                              dtype=torch.float32) * 0.5).to(torch.bfloat16)

        rotary = reference.MiMoV2RotaryEmbedding(config, is_swa=is_swa)
        position_ids = torch.arange(args.tokens).unsqueeze(0)
        position_embeddings = rotary(hidden, position_ids)

        # Build the additive mask explicitly. Passing attention_mask=None
        # leaves eager attention completely unmasked, which is invisible at one
        # token and silently bidirectional beyond that -- every position but
        # the last then disagrees with causal decoding.
        window = config.sliding_window if is_swa else None
        q = torch.arange(args.tokens).view(-1, 1)
        kv = torch.arange(args.tokens).view(1, -1)
        visible = kv <= q
        if window is not None:
            visible = visible & (kv > q - window)
        attention_mask = torch.zeros(1, 1, args.tokens, args.tokens,
                                     dtype=hidden.dtype)
        attention_mask = attention_mask.masked_fill(
            ~visible.view(1, 1, args.tokens, args.tokens),
            torch.finfo(hidden.dtype).min)

        residual = hidden
        normed = input_norm(hidden)
        with torch.no_grad():
            attention_out, _ = attention(
                normed,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=None,
                cache_position=position_ids[0],
                position_ids=position_ids,
            )
        after_attention = residual + attention_out

        residual = after_attention
        normed = post_norm(after_attention)

        if is_moe:
            gate = reference.MiMoV2MoEGate(config)
            gate.eval()
            with torch.no_grad():
                gate.weight.data = torch.from_numpy(
                    np.ascontiguousarray(
                        checkpoint.bf16(f"{prefix}.mlp.gate.weight"))
                ).to(torch.bfloat16)
                gate.e_score_correction_bias.data = torch.from_numpy(
                    np.ascontiguousarray(checkpoint.f32(
                        f"{prefix}.mlp.gate.e_score_correction_bias"))
                ).to(torch.float32)
                topk_indices, topk_weights = gate(normed)

            routed = sorted({int(x) for x in topk_indices.flatten().tolist()})
            print(f"  router selected {len(routed)} distinct experts across "
                  f"{args.tokens} tokens: {routed}")

            experts = [None] * config.n_routed_experts
            with torch.no_grad():
                for expert_id in routed:
                    mlp = reference.MiMoV2MLP(
                        config, intermediate_size=config.moe_intermediate_size)
                    mlp.eval()
                    base = f"{prefix}.mlp.experts.{expert_id}"
                    mlp.gate_proj.weight.data = to_bf16(
                        checkpoint.mxfp4_expert(f"{base}.gate_proj"))
                    mlp.up_proj.weight.data = to_bf16(
                        checkpoint.mxfp4_expert(f"{base}.up_proj"))
                    mlp.down_proj.weight.data = to_bf16(
                        checkpoint.mxfp4_expert(f"{base}.down_proj"))
                    experts[expert_id] = mlp

            # Build MiMoV2MoE without __init__ so the 248 unrouted experts are
            # never allocated. moe() only calls experts that own a token.
            moe = reference.MiMoV2MoE.__new__(reference.MiMoV2MoE)
            nn.Module.__init__(moe)
            moe.config = config
            moe.gate = gate
            moe.experts = experts
            moe.eval()
            with torch.no_grad():
                mlp_out = moe(normed)
        else:
            dense = reference.MiMoV2MLP(config)
            dense.eval()
            with torch.no_grad():
                dense.gate_proj.weight.data = to_bf16(
                    checkpoint.fp8_block(f"{prefix}.mlp.gate_proj.weight"))
                dense.up_proj.weight.data = to_bf16(
                    checkpoint.fp8_block(f"{prefix}.mlp.up_proj.weight"))
                dense.down_proj.weight.data = to_bf16(
                    checkpoint.fp8_block(f"{prefix}.mlp.down_proj.weight"))
                mlp_out = dense(normed)
            topk_indices = None
            routed = []

        final = residual + mlp_out

        def summary(name, tensor):
            flat = tensor.detach().float().flatten()
            return {
                "name": name,
                "shape": list(tensor.shape),
                "min": float(flat.min()),
                "max": float(flat.max()),
                "mean": float(flat.mean()),
                "rms": float(flat.pow(2).mean().sqrt()),
            }

        stages = [
            summary("hidden_in", hidden),
            summary("attention_out", attention_out),
            summary("after_attention", after_attention),
            summary("mlp_out", mlp_out),
            summary("final", final),
        ]
        print()
        print(f"  {'stage':<18}{'min':>12}{'max':>12}{'mean':>12}{'rms':>12}")
        for stage in stages:
            print(f"  {stage['name']:<18}{stage['min']:>12.5f}"
                  f"{stage['max']:>12.5f}{stage['mean']:>12.5f}"
                  f"{stage['rms']:>12.5f}")

        if not all(np.isfinite([s["rms"] for s in stages])):
            print("\nnon-finite activation detected", file=sys.stderr)
            return 1

        def bf16_bits(tensor):
            return tensor.detach().to(torch.bfloat16).view(torch.int16).numpy(
            ).astype(np.uint16).tobytes()

        if args.out:
            with open(args.out, "wb") as handle:
                handle.write(struct.pack("<IIII", layer, args.tokens,
                                         config.hidden_size,
                                         1 if is_swa else 0))
                handle.write(bf16_bits(hidden))
                handle.write(bf16_bits(after_attention))
                handle.write(bf16_bits(final))
                ids = (np.array(topk_indices.flatten().tolist(), dtype=np.uint32)
                       if topk_indices is not None
                       else np.zeros(0, dtype=np.uint32))
                handle.write(struct.pack("<I", ids.size))
                handle.write(ids.tobytes())
            print(f"\nwrote fixture {args.out}")

        if args.report:
            Path(args.report).write_text(json.dumps({
                "layer": layer,
                "is_swa": is_swa,
                "is_moe": is_moe,
                "tokens": args.tokens,
                "seed": args.seed,
                "routed_experts": routed,
                "stages": stages,
                "config_sha256": hashlib.sha256(
                    (root / "config.json").read_bytes()).hexdigest(),
                "modeling_sha256": hashlib.sha256(
                    (root / "modeling_mimo_v2.py").read_bytes()).hexdigest(),
            }, indent=1) + "\n")
            print(f"wrote report {args.report}")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
