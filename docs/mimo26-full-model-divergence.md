# MiMo full-model output is wrong: what is ruled out, what is left

Status as of 2026-09-22. The 48-layer worker runs and is transactional, and
every per-layer parity check passes, but generated text is not sensible. The
C worker and the Python reference **agree** with each other, so this is a
shared interpretation error, not an implementation divergence.

M4 is therefore built but **not qualified**. Do not promote it.

## The symptom, stated precisely

On text with an obvious repeating structure — "The capital of France is Paris.
The capital of Germany is Berlin. The capital of Italy is Rome." — a healthy
model nails the later repetitions. Rank of the true next token:

| Position | True token | Rank |
| ---: | --- | ---: |
| 1 | ` of` | 10 |
| 6 | ` The` | 5 |
| 8 | ` of` | 10 |
| 13 | ` The` | **100,421** |
| 14 | ` capital` | **17,134** |
| 15 | ` of` | 9 |
| 17 | ` is` | 64 |
| 19 | `.` | 1,082 |

So some syntax works — function words land around rank 5-10 — while induction
and copying fail completely. Ranks get *worse* in later repetitions, not
better. A pure repetition test (` the` eight times) gives rank 1 at position 1
and then 4, 132, 20, 461, 109, 222: correct at short range, erratic beyond.

Depth does help, so the layers compute something real: ` of` after
"The capital" moves from rank 19,540 with 4 layers to rank 10 with 48.

## Ruled out by direct measurement

Each of these was tested, not assumed. Recorded so they are not re-tried.

| Hypothesis | Evidence against it |
| --- | --- |
| BF16 tensor reading is wrong | Embedding neighbourhoods are clean: ` Paris` → Paris, London, Berlin, France; ` Monday` → every weekday |
| The padded vocabulary tail is real data | Rows 151,675+ have absmax 1.3e-5, numerically zero |
| FP8 block geometry or scale association | Every sampled 128×128 block has amax/448 exactly 1.0000, so blocks use the full E4M3 range |
| MXFP4 is packed as two half-rows | Interleaved layouts leave 100% of blocks normalized to max code 4 or 6; half-split layouts only 95.58%, with ragged maxima |
| MXFP4 within-byte nibble order is reversed | Swapping it makes full-model output clearly worse — rare high-id tokens instead of common ones |
| E8M0 bias is not 127 | Dequantized expert magnitudes (rms 0.003-0.021) match the dense FP8 MLPs (0.010-0.037); a wrong bias would show as a clean factor of two |
| RoPE uses interleaved (GPT-J) pairing | Interleaved destroys the one prediction that currently works — position 1 goes from rank 1 to rank 513 |
| RoPE theta is not per layer type | Measured from the reference's own `inv_freq`: 1e7 global, 1e4 SWA, dim 64 |
| The QKV split order is not `[q, k, v]` | Read directly from the reference forward |
| `attention_value_scale` should not be applied | Removing it does not fix output |
| The softmax scale should use the V dimension | Using 128^-0.5 does not fix output |
| The SWA sink is mishandled | Suppressing it does not fix output |
| Routing is degenerate | Three tokens select 20-22 distinct experts of 256; router weights sum to 1.0000 |
| Expert weights are corrupt | On a unit-rms random input an expert gives out rms 0.066, exactly what its weight magnitudes predict |
| The architecture is misread | The shipped technical report's table matches on every parameter: 48/39/9 layers, 64/8 SWA heads, 64/4 GA, QK/V 192/128, window 128, 256/8 experts |

## Two real bugs found and fixed on the way

**The reference harness left attention unmasked.** Passing
`attention_mask=None` makes eager attention bidirectional. That is invisible
at one token and wrong at every position but the last beyond it. Fixed by
building the additive mask explicitly.

**Layer parity only ever tested position 0.** Which is exactly why a
position-dependent error could hide behind a passing test. It now feeds every
position, and passes.

Note that the 3-token activation tables in `Evidence/mimo26-m1-m2-20260921/`
were produced before the mask fix and are therefore bidirectional. The
single-token parity numbers are unaffected.

## The one unexplained measurement

Layer 3's MoE output has rms **14.67** against 0.1-1.6 for every other layer,
and the residual stream stays near 14.9 for the remaining 44 layers. The
energy is spread across all 4096 dimensions — the top six carry only 2.6% —
so this is *not* the "massive activations" pattern that would be benign.

The experts routed at that layer produce outputs of rms 15.8 to 58.9 from an
input of rms 1.26, while the same experts on a random unit-rms input produce
rms 0.066. So the blow-up comes from the input being aligned with high-gain
directions of the expert, not from the weights. Dimension 2743 is both a top
residual dimension after layer 2 and the maximum column norm of a routed
expert's `gate_proj`.

Whether that alignment is the disease or a symptom is the open question.

## Where to look next

1. **An independent implementation.** The strongest remaining move. The shipped
   `modeling_mimo_v2.py` is demonstrably stale for this variant — it states
   Flash uses split q/k/v while the config and tensors are fused — so agreeing
   with it is not sufficient evidence. vLLM's MXFP4 "Humming" kernels are what
   the report says the quantization targets and have never been read.
2. **Layers 1-3 specifically.** The residual grows 0.18 → 0.58 → 1.91 → 14.67
   across them and then stays flat. Compare against a known-good run rather
   than judging plausibility.
3. **The attention path across positions**, since the failure is
   distance-dependent and RoPE has now been eliminated as the cause.
