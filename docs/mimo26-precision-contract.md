# MiMo V2.6 Flash precision contract

Target checkpoint: `XiaomiMiMo/MiMo-V2.6-Flash-RL`, revision
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`.

This records the arithmetic Moonshine commits to executing, with the evidence
for each item. Config flags alone do not define executed arithmetic: where the
config and the reference implementation disagree, the reference forward pass
wins and the disagreement is noted. Unresolved items are marked OPEN and must
not be silently decided in code.

## Status legend

- **SETTLED** — verified against the checkpoint or a reference, with evidence.
- **DECLARED** — chosen by us, must be asserted in code, not inferred.
- **OPEN** — not yet decided; code must fail closed rather than guess.

## Storage formats

| Item | Status | Value |
| --- | --- | --- |
| Routed expert weights | SETTLED | U8-packed MXFP4, two E2M1 codes per byte |
| Routed expert scales | SETTLED | U8 OCP E8M0, one per 32-element block |
| Within-byte code order | SETTLED | element 2k is the **low** nibble of byte k |
| E2M1 magnitudes | SETTLED | 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign |
| E8M0 value | SETTLED | `2**(byte-127)`; byte 0 is `2**-127` |
| E8M0 byte 255 | DECLARED | OCP says NaN; K3 currently yields `+Inf` |
| Dense and QKV weights | SETTLED | F8_E4M3 with F32 128x128 `weight_scale_inv` |
| FP8 activation scheme | SETTLED | `dynamic`, per `quantization_config` |
| o_proj, router, embeddings | SETTLED | BF16; all 48 `o_proj` are FP8-ignored |
| Expert compute output | SETTLED | BF16, round-to-nearest-even |
| Embedding rows | SETTLED | 152,576 rows for a 151,675-token vocabulary |

The embedding and `lm_head` carry **901 more rows than the tokenizer has
tokens** — 152,576 is 151,675 padded up to a multiple of 128. Ids
151,675..152,575 decode to no token, so sampling must mask that tail. An
unmasked argmax or a top-p tail can otherwise select an id the detokenizer
cannot map, which surfaces as a crash or a dropped token rather than as bad
text. `tests/test_mimo26_tokenizer.py` asserts the surplus stays known.

Code order is settled externally, not from the checkpoint: swapping the nibble
order permutes elements inside a shared-scale block and leaves every marginal
distribution and scale association unchanged, so the checkpoint cannot
adjudicate. The technical report shipped with the weights states the
quantization "follows the numeric constraints of the MXFP4 Humming GEMM kernels
(vLLM Project, 2026)", and SGLang's dequant for the same family builds elements
as `stack([table[low], table[high]], -1).flatten()`. Checked elementwise on
8,388,608 real codes. The wrong order shifts results by a median relative 1.41,
so this is not a tolerance question. The cited vLLM kernel itself is unread —
that is the residual.

E8M0 byte 255 is DECLARED rather than SETTLED because K3 maps it to `+Inf`
while OCP specifies NaN. No expert scale byte in the checkpoint's sampled
tensors falls outside `0x76`–`0x7D`, so the divergence is currently inert. Fix
or assert it; do not rely on it.

Signed zero: nibble code 0 does not occur anywhere in the sampled expert
tensors, while code 8 (negative zero) occurs in 11.623%. Preserve both codes
byte-exactly. K3 decodes code 8 to `-0.0f`, contributing nothing to the dot
product, and SGLang's table carries an explicit `-0.0` at index 8, so the
encoding is expected rather than anomalous. Do **not** import K3's 8-to-0
canonicalization: it is not byte-exact and buys no entropy here.

## Attention

| Item | Status | Value |
| --- | --- | --- |
| Softmax scale | SETTLED | `head_dim ** -0.5` = `192 ** -0.5` |
| QK head dim | SETTLED | 192, both global and SWA |
| V head dim | SETTLED | 128, both global and SWA |
| Query heads | SETTLED | 64 |
| KV heads | SETTLED | 4 global, 8 SWA |
| Fused QKV width | SETTLED | 13568 global, 14848 SWA |
| Rotary coordinates | SETTLED | first 64 of 192, `int(192 * 0.334)` |
| RoPE theta | SETTLED | 1e7 global, 1e4 SWA |
| Value scale | SETTLED | 0.707, applied **before** the cache write |
| Softmax accumulation | SETTLED | F32, then cast back to the query dtype |
| Attention sink | SETTLED | per-head learned logit, SWA layers only |
| Q/K norm | SETTLED | none exists |
| Sliding window | SETTLED | 128 |
| Window boundary semantics | SETTLED | `q - 128 < kv <= q`, inclusive of the current token |

The softmax scale derives from the **QK** dimension, not the V dimension. With
asymmetric 192/128 head dims this is the easiest silent error available.

`attention_value_scale` is applied to V before `past_key_values.update`, so the
KV cache stores pre-scaled V. Any parity harness comparing cached tensors, and
any rollback format, must match that. The constant is literally 0.707 — not
`1/sqrt(2)`; do not "correct" it.

Three config fields carry the window length: `sliding_window`,
`sliding_window_size` and `attention_chunk_size`, all 128. The reference
config and model never read `attention_chunk_size`. Read one field, cross-check
the others, and reject disagreement rather than picking silently.

**Window bounds, settled against the masking implementation.** The reference
uses HF's `sliding_window_causal_mask_function`, which is
`and_masks(kv_idx > q_idx - sliding_window, kv_idx <= q_idx)` — so a query at
position *q* attends to *q-127 .. q*: **128 positions including itself**.
Verified by evaluating the mask directly:

| Position | Attends | First | Last | Ring slot |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 0 | 0 |
| 127 | 128 | 0 | 127 | 127 |
| 128 | 128 | 1 | 128 | 0 |
| 129 | 128 | 2 | 129 | 1 |
| 257 | 128 | 130 | 257 | 1 |

Consequences for the KV implementation: exactly 128 ring slots suffice, the
slot for position *p* is `p mod 128`, and eviction begins at position 128 —
not 127 and not 129. Positions 127/128/129 and 255/256/257 are the boundary
cases any window test must cover, because each sits on a different side of a
wrap or an eviction.

## RoPE and attention arithmetic

Implemented in `mimo26_attention.{c,h}`, tested by
`tests/test_mimo26_attention.c` and cross-checked against torch by
`tests/test_mimo26_rope_vs_torch.py`, which reaches **bit-exact** agreement on
both the cos/sin tables and the rotated output across 11 positions from 0 to
1,048,575.

| Item | Status | Value |
| --- | --- | --- |
| `inv_freq` precision | SETTLED | **F32**, matching `1/(theta**(arange(0,64,2)/64))` |
| Angle product | SETTLED | F32 `position * inv_freq` |
| cos/sin dtype | SETTLED | computed F32, then rounded to BF16 before use |
| Rotation arithmetic | SETTLED | BF16: each product rounds, then the sum rounds |
| Rotation convention | SETTLED | split-half (NeoX), not interleaved |
| Score dtype | SETTLED | BF16 after scaling |
| Max subtraction | SETTLED | explicit, in BF16, before the softmax |
| Softmax | SETTLED | F32, then probabilities round to BF16 |
| Sink probability | SETTLED | computed, then **discarded** |
| Output | SETTLED | F32 accumulation, one BF16 rounding |

**Compute `inv_freq` in F32, not double.** Double is *more accurate* and
therefore wrong here: the reference's F32 `inv_freq` carries up to 6.4e-8
relative error, which the position multiplies. Using double instead broke
agreement at long positions, and switching to F32 made the tables bit-exact.

**Long-context RoPE is ill-conditioned, and this bounds useful context.**
Because the angle error scales with position, the absolute error in cos and
sin grows accordingly:

| Position | Angle error | cos error | vs one BF16 ulp (3.9e-3) |
| ---: | ---: | ---: | --- |
| 1,024 | 1.8e-5 rad | 4.3e-6 | negligible |
| 65,535 | 1.4e-3 rad | 8.8e-4 | approaching |
| 1,048,575 | 2.5e-2 rad | 8.1e-3 | **2x over** |

Those figures are for an F32-versus-double `inv_freq` difference, but the same
amplification applies to *any* last-ulp difference in `inv_freq` or in the
platform's `cos`/`sin`. So exact parity at extreme positions requires a
bit-identical `inv_freq` and a bit-identical trig implementation — not
something to assume across platforms. This is an independent argument for
qualifying context well below the advertised 1M, separate from memory and
latency, and it is why the context ladder should carry a parity check at each
rung rather than only a retrieval check.

**The rotation rounds three times.** `(q * cos) + (rotate_half(q) * sin)` has
every operand in BF16, so each product rounds and then the sum rounds.
Accumulating in F32 and rounding once left 21 of 192 coordinates differing;
matching the reference brought it to zero.

**The sink's probability mass is discarded.** Its logit is appended before the
softmax and its column is dropped after, so the surviving probabilities
deliberately sum to less than one and the output is a partial weighted sum. A
dominant sink drives the output toward zero, which the test asserts. Treating
the sink as a cached token, or renormalizing after dropping it, are both wrong.

## Routing

| Item | Status | Value |
| --- | --- | --- |
| Router projection dtype | SETTLED | F32, despite `moe_router_dtype: bfloat16` |
| Scoring | SETTLED | sigmoid |
| Selection scores | SETTLED | sigmoid + `e_score_correction_bias` |
| Mixing weights | SETTLED | uncorrected sigmoid, gathered at chosen indices |
| Normalization | SETTLED | `norm_topk_prob` true, denominator `+1e-20` |
| `routed_scaling_factor` | DECLARED | absent from config, so 1.0 |
| Group masking | DECLARED | `n_group = topk_group = 1`; assert, do not assume |
| Expert accumulation order | DECLARED | ascending expert id |
| Expert accumulation dtype | SETTLED | F32 accumulator, **one** BF16 cast at the end |

`moe_router_dtype: bfloat16` contradicts the reference forward, which does
`F.linear(x.float(), w.float())` unconditionally. The reference wins.

The reference calls `topk(..., sorted=False)`, leaving expert order
unspecified. Floating-point accumulation is order-sensitive, so an order is
declared here rather than inherited from a library's scheduling.

## Norms, activations, accumulation and residual

Implemented in `mimo26_ops.{c,h}`, tested by `tests/test_mimo26_ops.c`. Each
row was confirmed against a double-precision reference, and each is also shown
to differ measurably from its natural-looking alternative — the contract
distinctions are demonstrated, not asserted.

| Item | Status | Value |
| --- | --- | --- |
| RMSNorm epsilon | SETTLED | 1e-6, from `layernorm_epsilon` |
| RMSNorm accumulation | SETTLED | F32 mean of squares, index order |
| RMSNorm cast order | SETTLED | round to BF16 **then** multiply by weight |
| Activation | SETTLED | plain `silu(gate) * up`, unclamped |
| Expert accumulation | SETTLED | F32, one BF16 cast after the weighted sum |
| Residual | SETTLED | BF16 + BF16, single rounding |
| BF16 rounding | SETTLED | round-to-nearest-even everywhere |

**RMSNorm rounds twice.** The reference is
`return self.weight * hidden_states.to(input_dtype)` — the normalized value is
cast back to BF16 *before* the weight multiply, and the weight is BF16, so the
product is BF16×BF16. Multiplying in F32 and rounding once is the natural
implementation and it is wrong: **1029 of 4096** elements differ in the test.

**GLM's limited SwiGLU is not reusable.** `glm53_limited_swiglu_f32` caps gate
above and clamps up to a symmetric range; MiMo's reference MLP does neither.
They differ on **1681 of 4096** elements at a limit of 7.0. This is the clearest
example of a primitive that looks reusable and is not.

**Expert outputs accumulate in F32 with exactly one cast.** The reference
allocates the accumulator as `zeros_like(hidden_states, dtype=topk_weights.dtype)`
and the gate runs in F32, so the accumulator is F32; each BF16 expert output is
promoted, multiplied by its F32 router weight, added in F32, and the result is
cast once by `final_hidden_states.type(hidden_states.dtype)`. Rounding after
each expert instead changes **35 of 64** elements. This is what the feasibility
note meant by not inheriting GLM's BF16 accumulation boundaries.

## Numerical evidence

The committed K3 MXFP4 kernels were exercised on real expert bytes from layers
1, 24 and 47, all three projections, at both geometries (gate/up 2048x4096,
down 4096x2048), across `gemv`, `gemm` and `gemm_tiled` at tiles 16/32/64.

- Against a float64 reference built from an independent spec-derived decoder,
  RMS relative error is 1.65e-3 to 1.70e-3.
- Restricted to outputs at or above median magnitude, max relative error is
  3.873e-3 to 3.890e-3 — one BF16 ulp (2^-8 = 3.906e-3).
- Larger headline figures, up to 1.121e-2, occur only where the reference
  output is 4e-5 to 5e-5, four to five orders below median: cancellation, not
  decode error.
- Column 0 of `gemm` and of every `gemm_tiled` tile reproduces single-vector
  `gemv` bit for bit, confirming the reduction-order invariant documented in
  `k3_rocm_ops.h` holds at MiMo geometry.
- `hip_bfloat16(float)` was tested directly on tie, above-tie and below-tie
  patterns of both signs: round-to-nearest-even, not truncation.

This establishes kernel-versus-oracle agreement and geometry compatibility. It
is not a full-model qualification: coverage is three experts, and M1's
independent full-model reference does not yet exist.

## Tokenization and turn rendering

Pinned in `tests/fixtures/mimo26_tokenizer_v1.json` from `tokenizer.json` and
`chat_template.jinja`, generated without `trust_remote_code` and verified by
`tests/test_mimo26_tokenizer.py` against the checkpoint's file hashes.

| Item | Status | Value |
| --- | --- | --- |
| Tokenizer vocabulary | SETTLED | 151,675 including added tokens |
| `model_max_length` | SETTLED | 1,048,576 (upstream claim, not qualified) |
| `eos_token_id` | SETTLED | **three** ids: 151643, 151645, 151672 |
| Default sampling | SETTLED | `do_sample` false, temperature 1.0, top_p 0.95 |
| Turn framing | SETTLED | `<\|im_start\|>role\n` ... `<\|im_end\|>` |
| Generation prompt | SETTLED | `<\|im_start\|>assistant\n` |
| `enable_thinking: false` | SETTLED | appends an empty `<think></think>` |
| Tool call surface | SETTLED | `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` |

Two consequences worth stating before any server work:

**Finish detection needs all three eos ids.** Treating `eos_token_id` as a
scalar leaves two stop conditions unhandled.

**Assistant turns always carry a think block.** The template emits
`<think>` + `reasoning_content` + `</think>` for every assistant message, so a
replayed turn with no reasoning renders as `<think></think>text`. History
replay must reproduce that byte-exactly or the KV prefix diverges — the same
failure class as the earlier Hermes reasoning-replay prefix miss.

Tool calls are **not** JSON: arguments are nested parameter tags. A parser and
a round-trip test are required before tool support is offered, and rendering
fixtures are not evidence that tool execution works.

## Reuse decisions

| Primitive | Verdict |
| --- | --- |
| `k3_rocm_mxfp4_*` expert kernels | Reusable, verified on real bytes at both geometries |
| `glm53_fp8_oracle` block-FP8 | Reusable, verified at all four MiMo geometries including the `[108,32]` overhang |
| `glm53_router_topk_f32` | Contract-identical including the `1e-20` epsilon; used as a cross-check oracle, but MiMo declares ascending-id order so `mimo26_router` owns the accumulation |
| `glm53_limited_swiglu_f32` | **Not reusable** — clamps where MiMo does not |
| `k3_st_model_open*` | **Not applicable** — MiMo's shard names need `k3_st_model_open_paths` |

## Open items blocking M1 exit

1. The independent full-model reference path. An opaque chat API cannot
   establish logit, route or precision parity, and generic HF loading is not
   assumed to work since the reference modeling code carries no packed-expert
   dequantization. Needs a decision, not more code.

Every operator-level item is now settled. What remains is composition: M2's
transactional KV, and M3's layer assembly against captured inputs.
