# MiMo full-model divergence: one cause found, one remaining

Status 2026-09-22. **A real cause was found and fixed**: the fused QKV
projection is stored grouped by KV head, not as `[all Q | all K | all V]`.
Output improved dramatically but is still not correct, so at least one issue
remains. M4 is still **not qualified**.

## The fix: fused QKV is grouped by KV head

vLLM 0.30 ships a native MiMo V2 (`model_executor/models/mimo_v2.py`) whose
`_shard_fp8_qkv_proj` documents the real layout:

> The checkpoint stores the fused QKV as `num_kv_heads` contiguous groups (one
> per KV head), each ordered `[Q | K | V]`:
> `[Q_1 | K_1 | V_1 | Q_2 | K_2 | V_2 | ... | Q_n | K_n | V_n]`

The arithmetic is exact — global: 4 × (16·192 + 192 + 128) = 4 × 3392 =
13568; SWA: 8 × (8·192 + 192 + 128) = 8 × 1856 = 14848. The shipped
`modeling_mimo_v2.py` does a plain three-way split, which scrambles every
head. This is the staleness this lane suspected from the start, now located:
that file also claims Flash uses split q/k/v while the tensors are fused.

It also explains the `[108,32]` scale grid that never added up. Global groups
are 26.5 blocks, padded per group to 27, so 4 × 27 = 108 rather than the 106
a flat layout needs — and flat indexing misassociated the scales of every
group after the first. SWA groups are 14.5 blocks and total exactly 116, so
that grid really is flat. Both readings are implemented, selected by whether
the grid divides evenly by the group count.

### Effect

Rank of the true next token, "The capital of X is Y" three times:

| Position | Token | Before | After |
| ---: | --- | ---: | ---: |
| 1 | ` of` | 10 | **1** |
| 8 | ` of` | 10 | **2** |
| 12 | `.` | 20,817 | **9** |
| 13 | ` The` | **100,421** | **12** |
| 3 | ` is` | 90,433 | **65** |

`assistant` after `<|im_start|>` went 60,627 → 3,494. The unexplained layer-3
blow-up also disappeared: residual rms at layer 9 is now 1.42 rather than
14.9, which is independent confirmation the fix was right.

Verified in C as well: the C and NumPy dequantizers agree bit-exactly on the
grouped path for both a SWA layer (flat scales) and a global layer (grouped
scales).

## What is still wrong

Function words are now consistently well ranked (1-14), but content words are
not: France 34,140, Berlin 47,458, Rome 48,102, Paris 23,248. Greedy
generation degenerates into ` the` and a couple of other high-frequency
tokens, which is exactly what a model biased toward function words does.

Ruled out since the fix:

| Hypothesis | Evidence |
| --- | --- |
| Router dtype should be BF16 per `moe_router_dtype` | vLLM honours it and the shipped reference hardcodes F32, so vLLM is likely right — but switching changes ranks only marginally (1→2, 2→4, 72→54). Worth adopting for fidelity, not the bug |
| The MoE config differs | vLLM uses sigmoid scoring, grouped top-k with `n_group`/`topk_group`, the correction bias and `renormalize=norm_topk_prob` — all matching |
| Expert tensors need special handling | vLLM's loader maps `gate_proj`/`up_proj`/`down_proj` normally |
| The decoder layer structure differs | vLLM uses the standard fused add-norm pre-norm pattern |

## Where to look next

vLLM's MiMo implementation is now extracted under `/tmp/vllm_src` and is the
best available oracle. The parts not yet compared line by line are the
attention backend's handling of `v_head_dim != head_dim` (vLLM uses a
dedicated DiffKV backend), the sink's exact placement in that backend, and
the FusedMoE kernel path. Read those before writing more code.

