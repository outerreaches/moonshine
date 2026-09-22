#!/usr/bin/env python3
"""Run the whole 48-layer model through the reference modules, streaming.

This is the independent full-model evidence M4 requires. Materializing every
layer's experts is impossible, so layers are processed one at a time: load,
dequantize, run, free. Peak footprint is one layer's non-expert weights plus
the experts that layer actually routed, a few hundred megabytes, rather than
the hundreds of gigabytes a resident model would need.

Every module executed is the checkpoint's own. Dequantization is the
independent NumPy path shared with tools/mimo26_reference_layer.py, written
from the OCP bit fields rather than from the C implementation.

  python3 tools/mimo26_reference_model.py --tokens 151644 872 198 --out logits.bin
"""
import argparse
import json
import struct
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mimo26_reference_layer import Checkpoint  # noqa: E402

DEFAULT_ROOT = "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--layers", type=int, default=None,
                        help="run only the first N layers, for localizing")
    parser.add_argument("--rank-of", type=int, nargs="*", default=None,
                        help="true next-token ids, one per position")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    import shutil
    import tempfile
    import torch
    import torch.nn as nn

    root = Path(args.root)
    workdir = Path(tempfile.mkdtemp(prefix="mimo26-refmodel-"))
    try:
        package = workdir / "mimo_reference"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name in ("modeling_mimo_v2.py", "configuration_mimo_v2.py"):
            shutil.copy2(root / name, package / name)
        sys.path.insert(0, str(workdir))
        from mimo_reference.configuration_mimo_v2 import MiMoV2Config
        import mimo_reference.modeling_mimo_v2 as reference

        _original_rope = reference.apply_rotary_pos_emb
        import os
        config = MiMoV2Config(**json.loads((root / "config.json").read_text()))
        checkpoint = Checkpoint(root)
        tokens = list(args.tokens)
        count = len(tokens)

        def to_bf16(array):
            return torch.from_numpy(np.ascontiguousarray(array)).to(torch.bfloat16)

        embed = to_bf16(checkpoint.bf16("model.embed_tokens.weight"))
        hidden = embed[torch.tensor(tokens, dtype=torch.long)].unsqueeze(0)
        del embed
        print(f"{count} tokens, hidden {tuple(hidden.shape)}")

        position_ids = torch.arange(count).unsqueeze(0)
        q = torch.arange(count).view(-1, 1)
        kv = torch.arange(count).view(1, -1)
        causal = kv <= q

        started = time.monotonic()
        layer_budget = (config.num_hidden_layers if args.layers is None
                        else min(args.layers, config.num_hidden_layers))
        for layer in range(layer_budget):
            is_swa = bool(config.hybrid_layer_pattern[layer])
            is_moe = bool(config.moe_layer_freq[layer])
            prefix = f"model.layers.{layer}"

            visible = causal.clone()
            if is_swa:
                visible = visible & (kv > q - config.sliding_window)
            mask = torch.zeros(1, 1, count, count, dtype=hidden.dtype)
            mask = mask.masked_fill(~visible.view(1, 1, count, count),
                                    torch.finfo(hidden.dtype).min)

            input_norm = reference.MiMoV2RMSNorm(
                config.hidden_size, eps=config.layernorm_epsilon).eval()
            post_norm = reference.MiMoV2RMSNorm(
                config.hidden_size, eps=config.layernorm_epsilon).eval()
            attention = reference.MiMoV2Attention(
                config, is_swa, layer, projection_layout="fused_qkv").eval()
            with torch.no_grad():
                input_norm.weight.data = to_bf16(
                    checkpoint.bf16(f"{prefix}.input_layernorm.weight"))
                post_norm.weight.data = to_bf16(
                    checkpoint.bf16(f"{prefix}.post_attention_layernorm.weight"))
                attention.qkv_proj.weight.data = to_bf16(
                    checkpoint.fp8_block(f"{prefix}.self_attn.qkv_proj.weight"))
                attention.o_proj.weight.data = to_bf16(
                    checkpoint.bf16(f"{prefix}.self_attn.o_proj.weight"))
                if attention.attention_sink_bias is not None:
                    attention.attention_sink_bias.data = to_bf16(
                        checkpoint.bf16(
                            f"{prefix}.self_attn.attention_sink_bias"))

            # Diagnostic switches. Each isolates one unusual contract item so
            # a wrong choice shows up as an end-to-end behaviour change.
            import os
            if os.environ.get("MIMO26_NO_VALUE_SCALE") == "1":
                attention.v_scale = None
            if os.environ.get("MIMO26_VDIM_SCALING") == "1":
                attention.scaling = float(config.swa_v_head_dim) ** -0.5
            if os.environ.get("MIMO26_NO_SINK") == "1" and \
                    attention.attention_sink_bias is not None:
                attention.attention_sink_bias.data = torch.full_like(
                    attention.attention_sink_bias.data, -1.0e4)

            rotary = reference.MiMoV2RotaryEmbedding(config, is_swa=is_swa)
            position_embeddings = rotary(hidden, position_ids)

            if os.environ.get("MIMO26_ROPE_INTERLEAVED") == "1":
                # The shipped reference uses split-half (NeoX) pairing. Test
                # interleaved (GPT-J) pairing instead: pairs (2p, 2p+1) rather
                # than (p, p+32). Identity at distance 0 either way, so only a
                # distance-dependent test can tell them apart.
                def interleaved_rope(q, k, cos, sin, position_ids=None,
                                     unsqueeze_dim=1):
                    cos = cos.unsqueeze(unsqueeze_dim)
                    sin = sin.unsqueeze(unsqueeze_dim)
                    half = cos.shape[-1] // 2
                    c = cos[..., :half]
                    s = sin[..., :half]
                    def rot(x):
                        even = x[..., 0::2]
                        odd = x[..., 1::2]
                        out = torch.empty_like(x)
                        out[..., 0::2] = even * c - odd * s
                        out[..., 1::2] = odd * c + even * s
                        return out
                    return rot(q), rot(k)
                reference.apply_rotary_pos_emb = interleaved_rope
            else:
                reference.apply_rotary_pos_emb = _original_rope

            residual = hidden
            normed = input_norm(hidden)
            with torch.no_grad():
                attention_out, _ = attention(
                    normed, position_embeddings=position_embeddings,
                    attention_mask=mask, past_key_values=None,
                    cache_position=position_ids[0], position_ids=position_ids)
            attn_contrib = attention_out
            hidden = residual + attention_out
            del attention, input_norm, normed

            residual = hidden
            normed = post_norm(hidden)
            if is_moe:
                gate = reference.MiMoV2MoEGate(config).eval()
                with torch.no_grad():
                    gate.weight.data = to_bf16(
                        checkpoint.bf16(f"{prefix}.mlp.gate.weight"))
                    gate.e_score_correction_bias.data = torch.from_numpy(
                        np.ascontiguousarray(checkpoint.f32(
                            f"{prefix}.mlp.gate.e_score_correction_bias"))
                    ).to(torch.float32)
                    topk_indices, _ = gate(normed)
                routed = sorted({int(x) for x in topk_indices.flatten().tolist()})
                experts = [None] * config.n_routed_experts
                with torch.no_grad():
                    for expert_id in routed:
                        mlp = reference.MiMoV2MLP(
                            config,
                            intermediate_size=config.moe_intermediate_size).eval()
                        base = f"{prefix}.mlp.experts.{expert_id}"
                        mlp.gate_proj.weight.data = to_bf16(
                            checkpoint.mxfp4_expert(f"{base}.gate_proj"))
                        mlp.up_proj.weight.data = to_bf16(
                            checkpoint.mxfp4_expert(f"{base}.up_proj"))
                        mlp.down_proj.weight.data = to_bf16(
                            checkpoint.mxfp4_expert(f"{base}.down_proj"))
                        experts[expert_id] = mlp
                moe = reference.MiMoV2MoE.__new__(reference.MiMoV2MoE)
                nn.Module.__init__(moe)
                moe.config = config
                moe.gate = gate
                moe.experts = experts
                moe.eval()
                with torch.no_grad():
                    mlp_out = moe(normed)
                    if os.environ.get("MIMO26_MOE_DETAIL") == str(layer):
                        idx, wts = gate(normed)
                        print(f"      normed rms "
                              f"{float(normed.float().pow(2).mean().sqrt()):.4f}",
                              flush=True)
                        for slot in range(idx.shape[-1]):
                            e = int(idx[0, slot])
                            w = float(wts[0, slot])
                            out = experts[e](normed[:, -1:])
                            print(f"      expert {e:3d} w={w:.4f} "
                                  f"out_rms {float(out.float().pow(2).mean().sqrt()):9.4f} "
                                  f"weighted {w * float(out.float().pow(2).mean().sqrt()):9.4f}",
                                  flush=True)
                        print(f"      weight sum {float(wts.sum()):.4f}", flush=True)
                del moe, experts, gate
            else:
                dense = reference.MiMoV2MLP(config).eval()
                with torch.no_grad():
                    dense.gate_proj.weight.data = to_bf16(
                        checkpoint.fp8_block(f"{prefix}.mlp.gate_proj.weight"))
                    dense.up_proj.weight.data = to_bf16(
                        checkpoint.fp8_block(f"{prefix}.mlp.up_proj.weight"))
                    dense.down_proj.weight.data = to_bf16(
                        checkpoint.fp8_block(f"{prefix}.mlp.down_proj.weight"))
                    mlp_out = dense(normed)
                del dense
            hidden = residual + mlp_out
            del normed, post_norm

            elapsed = time.monotonic() - started
            def rms(x):
                return float(x.float().pow(2).mean().sqrt())
            print(f"  layer {layer:2d} {'SWA' if is_swa else 'GLB'} "
                  f"{'MoE' if is_moe else 'dense'}  "
                  f"attn {rms(attn_contrib):8.4f}  mlp {rms(mlp_out):8.4f}  "
                  f"resid {rms(hidden):9.4f}  {elapsed:6.1f}s", flush=True)
            if layer in (2, 3, 4, 47):
                v = hidden[0, -1].float().abs()
                top = torch.topk(v, 6)
                share = float(top.values.pow(2).sum() / v.pow(2).sum())
                print("      top |dims|: " + ", ".join(
                    f"{int(i)}:{float(x):.2f}" for x, i in
                    zip(top.values, top.indices)) +
                    f"   top6 carry {100*share:.1f}% of the energy", flush=True)

        final_norm = reference.MiMoV2RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon).eval()
        with torch.no_grad():
            final_norm.weight.data = to_bf16(checkpoint.bf16("model.norm.weight"))
            normed = final_norm(hidden)
            head = to_bf16(checkpoint.bf16("lm_head.weight"))
            logits = torch.nn.functional.linear(normed, head)

        values = logits[0].float()
        print(f"\nlogits {tuple(values.shape)} after {layer_budget} layers")
        if args.rank_of:
            print("  true-next-token rank (1 is best):")
            for position, true_id in enumerate(args.rank_of):
                if position >= count:
                    break
                row = values[position].clone()
                row[151675:] = float("-inf")
                rank = int((row > row[true_id]).sum()) + 1
                print(f"    position {position:3d} true {true_id:<7} "
                      f"rank {rank:<8} logit {float(row[true_id]):8.3f}")
        for position in range(count):
            row = values[position].clone()
            row[151675:] = float("-inf")
            top = torch.topk(row, args.top)
            pairs = ", ".join(f"{int(i)}:{float(v):.3f}"
                              for v, i in zip(top.values, top.indices))
            print(f"  position {position:3d} fed {tokens[position]:<7} top: {pairs}")

        if args.out:
            with open(args.out, "wb") as handle:
                handle.write(struct.pack("<II", count, values.shape[1]))
                handle.write(values.numpy().astype(np.float32).tobytes())
            print(f"\nwrote {args.out}")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
