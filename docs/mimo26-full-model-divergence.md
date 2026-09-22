# MiMo full-model divergence: resolved

Status 2026-09-22. **Resolved.** There was one cause, found in two stages,
and both stages were the same tensor: the fused QKV projection's row layout.
The model now predicts correctly and shows textbook induction. Kept as a
record because the false intermediate conclusion in stage one is instructive,
and because the vLLM docstring that produced it is still wrong for this
checkpoint.

## Stage one: not `[all Q | all K | all V]`

vLLM 0.30 ships a native MiMo V2 (`model_executor/models/mimo_v2.py`) whose
`_shard_fp8_qkv_proj` says the checkpoint stores the fused QKV as
`num_kv_heads` contiguous groups, each ordered `[Q_g | K_g | V_g]`. The
shipped `modeling_mimo_v2.py` does a plain three-way split instead, which
scrambles every head — the staleness this lane suspected from the start,
located at last. That file also claims Flash uses split q/k/v while the
config and tensors are fused.

Implementing the grouped reading moved the symptom sharply: ` The` at
position 13 went from rank 100,421 to 12, ` of` to rank 1, and the
unexplained layer-3 blow-up disappeared. That improvement was real but the
diagnosis was only half right, and the half that was wrong was invisible
because the two readings **coincide on global layers**.

## Stage two: the grouping is by TP rank, not by KV head

The checkpoint's own `metadata.tp_size` is 4. It was written by a
tensor-parallel deployment, so the fused QKV is 4 rank slices concatenated,
each already in the de-interleaved `[Q | K | V]` layout that rank's forward
expects. Rank `s` owns query heads `[16s, 16s+16)` and KV heads
`[s·kv/4, ...)`, which preserves the standard `repeat_kv` mapping.

A rank owns `kv_heads / 4` KV heads. For the 9 global layers that is exactly
one, so a rank slice *is* a KV-head group and stage one's reading happens to
be correct. For the 39 SWA layers a rank owns two, and the two readings give
genuinely different permutations. Stage one therefore left 39 of 48 layers
scrambled — including every layer that does local copying.

The FP8 block scales settle it. They tile each rank slice independently,
padded up to a whole block, and that one rule reproduces both grids exactly:

| Layer | Rank slice | Blocks | × 4 | Observed |
| --- | --- | --- | ---: | ---: |
| global | 16·192 + 1·192 + 1·128 = 3392 | 26.5 → 27 | 108 | `[108, 32]` |
| SWA | 16·192 + 2·192 + 2·128 = 3712 | 24 + 3 + 2 = **29 exactly** | 116 | `[116, 32]` |

Under the by-KV-head reading a SWA group is 1856 rows = **14.5 blocks**, so
8 × 14.5 = 116 only by coincidence of the total, and no block grid can
address such a group at all. vLLM's own code cannot execute this path: it
computes `scale_rows_per_group = s_full.shape[0] // num_kv_heads` = 116 // 8
= 14, expands to 1792 rows, and multiplies against an 1856-row group. It
would raise. The docstring describes a format this checkpoint does not use,
which is why reading code is not the same as checking it against the bytes.

The de-interleaved layout is also the one whose parts are block-aligned —
3072 = 24 blocks, 384 = 3, 256 = 2 — which is precisely what that same
docstring says the re-quantization exists to achieve.

## Effect

Greedy prediction on "The quick brown fox jumps." repeated, rank of the true
next token:

| Position | Token | Before stage 1 | After stage 1 | After stage 2 |
| ---: | --- | ---: | ---: | ---: |
| 3 | ` jumps` | 36,800 | 36,800 | **1** |
| 9 | ` jumps` | 26,097 | 26,097 | **1** |
| 15 | ` jumps` | 27,667 | 27,667 | **1** |

Every position in that prompt is now rank 1 except position 10, which is
rank 2 (`.` against ` is` — a real ambiguity). Induction strengthens with
repetition the way it should: the logit for the same prediction rises
16.375 → 17.750 → 21.625 across the three occurrences. Before stage two
there was no improvement across repetitions at all, which was the clearest
sign that the failure was in copying rather than in the MoE or the sampler.

The residual now grows smoothly (0.031 → 0.051 → 0.108 → …) with the massive
activation appearing at layer 16 concentrated in a handful of dimensions,
the usual trained-model pattern. Earlier the same trace blew up at layer 3
spread across all 4096 dimensions.

C and NumPy dequantizers agree **bit-exactly** on the new layout for both a
global layer (padded per-rank scales) and a SWA layer (60,817,408 values,
FNV-1a `8b9157fd724a0818`). The C worker's logits match the Python
reference's rankings at every position checked.

## What this cost, and the cheaper path

Roughly twenty hypotheses were ruled out by measurement while the real fault
sat in a tensor that two separate sources described incorrectly. The probe
that would have found it fastest was not a hypothesis test at all: **derive
the block-scale grid from first principles and require it to come out
exactly.** `[108, 32]` and `[116, 32]` are only four numbers, and no reading
but the right one reproduces both. The reading that survived arithmetic was
the reading that was true.

One test that actively misled: boosting the attention logits by 2× and 4× to
see whether diffuse attention was merely under-scaled. Both made predictions
worse, which correctly ruled out a missing scale factor — but the diffuse
attention it was probing was a *symptom* of the scrambled SWA heads, not a
cause. Measuring a symptom's response to a change tells you about the
symptom.

## Adopted separately

`moe_router_dtype` is BF16 in the config; vLLM honours it and the shipped
reference hardcodes F32. It changes ranks only marginally and was never the
bug, but the checkpoint says what it says.
