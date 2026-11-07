#ifndef MIMO26_ROCM_OPS_H
#define MIMO26_ROCM_OPS_H

/*
 * MiMo-specific GPU primitives.
 *
 * Deliberately separate from k3_rocm_ops: three of MiMo's elementwise
 * contracts differ from K3's in ways that are invisible in a tolerance test
 * and poison a 48-layer graph slowly.
 *
 *   RMSNorm  K3 multiplies by the weight in F32 and rounds once. MiMo rounds
 *            to BF16 *before* the weight multiply -- two roundings. Measured
 *            on the CPU side: 1029 of 4096 elements differ.
 *   SwiGLU   K3's is GLM's limited form, which caps gate above and clamps up
 *            to a symmetric range. MiMo's reference MLP does neither.
 *            Measured: 1681 of 4096 elements differ.
 *   Router   K3 emits top-k in score-descending order and accumulates the
 *            normalization denominator in that order. MiMo declares ascending
 *            expert id for both, and float summation is order-dependent.
 *
 * What is NOT duplicated here is k3_rocm_mxfp4_gemv_bf16, which was checked
 * against MiMo's dequantizer and matched on every one of 8192 values, and
 * k3_rocm_bf16_gemv_bf16, checked the same way. Reuse where the contract is
 * genuinely shared; fork where it is not.
 *
 * Every entry point takes device pointers and an optional stream. All are
 * host-callable from C.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MIMO26_ROCM_TOP_K 8u
#define MIMO26_ROCM_EXPERTS 256u
#define MIMO26_ROCM_HIDDEN 4096u

#ifdef __cplusplus
extern "C" {
#endif

/*
 * out = weight * (bf16)(x * rsqrt(mean(x^2) + epsilon))
 *
 * vectors rows of hidden_size BF16 elements each, normalized independently.
 */
bool mimo26_rocm_rmsnorm_bf16(void *output, const void *input,
                              const void *weight, uint32_t vectors,
                              uint32_t hidden_size, float epsilon,
                              void *stream);

/*
 * out = silu(gate) * up, silu(x) = x * sigmoid(x), plain and unclamped,
 * with silu rounded to BF16 before the product and the product rounded
 * again -- the two roundings the reference MLP performs.
 */
bool mimo26_rocm_silu_product_bf16(void *output, const void *gate,
                                   const void *up, uint64_t count,
                                   void *stream);

/* out = residual + delta, BF16 + BF16 -> BF16, one rounding. */
bool mimo26_rocm_residual_add_bf16(void *output, const void *residual,
                                   const void *delta, uint64_t count,
                                   void *stream);

/*
 * Router logits in F32, summed in ascending column order so the result
 * matches mimo26_router_logits_f32 on the CPU.
 *
 * Pinning the order matters here specifically. The logits land around -2
 * while their summands total far more in absolute value, so the sum
 * cancels heavily and a tree reduction diverges much further than the
 * condition of a well-behaved row would suggest: measured through the
 * mixing weights it sat 4.5e-04 from the CPU, roughly 3800x libm's last
 * ulp, which is enough to shift every element of the expert-weighted sum.
 * The exactness is free -- the router is negligible beside the 4.68 GiB of
 * expert reads in the same token.
 */
bool mimo26_rocm_ordered_gemv_f32(float *output, const void *weights,
                                  const void *input, uint32_t rows,
                                  uint32_t columns, void *stream);

/* Batched form: `count` inputs against the same weights, each row summed in
 * its own ascending order. */
bool mimo26_rocm_ordered_gemv_f32_batch(float *output, const void *weights,
                                        const void *input, uint32_t rows,
                                        uint32_t columns, uint32_t count,
                                        void *stream);

/* Whether router logits are rounded to BF16, per MIMO26_ROUTER_BF16. Read
 * once; reported when enabled so a run cannot be silently different. */
uint32_t mimo26_rocm_router_bf16_enabled(void);

/* The router's use of it, named for what it computes. */
bool mimo26_rocm_router_logits_f32(float *logits, const void *weight,
                                   const void *hidden, uint32_t experts,
                                   uint32_t hidden_size, void *stream);

/*
 * Select top_k of expert_count by sigmoid(logit) + bias, returning the
 * *uncorrected* sigmoid values as mixing weights, normalized by their sum
 * plus 1e-20 and scaled.
 *
 * Indices come back in ascending expert id, and the denominator accumulates
 * in that same order, because the CPU contract declares both. Selection ties
 * resolve to the lower expert id.
 */
bool mimo26_rocm_router_topk(uint32_t *expert_ids, float *expert_weights,
                             const float *logits, const float *correction_bias,
                             uint32_t vectors, uint32_t expert_count,
                             uint32_t top_k, float scale, void *stream);

/*
 * accumulator[count] += weight * expert_output[count], in F32, then
 * finalize rounds once to BF16. Call accumulate once per selected expert in
 * ascending expert-id order, then finalize.
 */
bool mimo26_rocm_expert_accumulate_f32(float *accumulator,
                                       const void *expert_output, float weight,
                                       uint64_t count, void *stream);
bool mimo26_rocm_expert_finalize_bf16(void *output, const float *accumulator,
                                      uint64_t count, void *stream);
bool mimo26_rocm_zero_f32(float *accumulator, uint64_t count, void *stream);

/*
 * One decode step of MiMo attention for all 64 query heads.
 *
 * Mirrors mimo26_attention_decode exactly, including its cast points: the
 * score is rounded to BF16 after scaling, the row max is subtracted in BF16,
 * the softmax denominator accumulates in double, probabilities round to BF16
 * before weighting the values, and the value sum accumulates in F32 and
 * rounds once on the way out.
 *
 * The parallel decomposition is chosen so the summation ORDER matches the CPU
 * too, which is what makes bit-exactness reachable rather than hopeful:
 * each thread computes a whole 192-term dot sequentially, the denominator is
 * summed by one thread in ascending slot order, and the value accumulation
 * splits over the 128 output dimensions so every thread walks history in
 * ascending order. Tree-reducing any of those three would reassociate the
 * additions and lose the guarantee.
 *
 * query is [64][192] already rotated, keys [history][kv_heads][192], values
 * [history][kv_heads][128], out [64][128]. current_keys/current_values carry
 * the uncommitted token at query_position, or both NULL. sink_bias is [64]
 * BF16 and required for windowed layers; its probability is discarded, so
 * the surviving probabilities sum to less than one.
 *
 * scratch is caller-owned device storage of at least 64 * (history + 2)
 * floats, so the entry point performs no allocation.
 */
bool mimo26_rocm_attention_decode(void *out, const void *query,
                                  const void *keys, const void *values,
                                  const void *current_keys,
                                  const void *current_values,
                                  const void *sink_bias, float *scratch,
                                  uint32_t kv_heads, uint32_t kv_groups,
                                  uint32_t window, uint64_t history,
                                  uint64_t first_position,
                                  uint64_t query_position, float scale,
                                  void *stream);

/*
 * Attention for a whole chunk of queries at once.
 *
 * query is [count][64][192], out [count][64][128]. keys and values must
 * already include the chunk's own entries appended to the prior history,
 * and `history` counts both -- causal masking within the chunk then falls
 * out of the same absolute-position predicate the decode path uses, so
 * prefilling N tokens is bit-identical to decoding them one at a time.
 *
 * scratch_floats says how large the scratch actually is. A full chunk wants
 * count * 64 * (history + 2) floats, which grows with the context capacity
 * and reaches 8 GiB at a 262144 context and a 128 chunk. When the buffer is
 * smaller than that, the queries are attended in sub-batches narrow enough to
 * fit, each passing the shorter history its own last query can see.
 *
 * That is arithmetically invisible: a sub-batch sees the same visible slots in
 * the same order, and the slots it no longer iterates were masked out and
 * contributed nothing. Splitting is therefore bit-exact, and the equality gate
 * is what holds it to that.
 *
 * Fails if the scratch cannot hold even a single query's row.
 */
bool mimo26_rocm_attention_prefill(void *out, const void *query,
                                   const void *keys, const void *values,
                                   const void *sink_bias, float *scratch,
                                   uint64_t scratch_floats,
                                   uint32_t kv_heads, uint32_t kv_groups,
                                   uint32_t window, uint64_t history,
                                   uint64_t first_position,
                                   uint64_t first_query_position,
                                   uint32_t query_count, float scale,
                                   void *stream);

/*
 * Split a fused QKV projection output into per-head Q, K and V, scaling V by
 * 0.707 on the way so a cache downstream holds pre-scaled V -- as the
 * reference does. Layout in is [q | k | v] with q = 64*192, k = kv*192,
 * v = kv*128.
 */
bool mimo26_rocm_split_qkv(void *q, void *k, void *v, const void *fused,
                           uint32_t kv_heads, void *stream);

/*
 * Rotate the first 64 coordinates of each 192-wide head in place, split-half
 * (NeoX), leaving the other 128 untouched.
 *
 * The cos/sin tables are built on the HOST by mimo26_rope_table and uploaded.
 * They need powf, cosf and sinf, and host and device libm disagree in the
 * last ulp, so computing them on device would import that difference into
 * every rotated coordinate for nothing -- the tables are 64 entries and are
 * built once per position. Keeping them on the host is what lets the rotation
 * itself be bit-exact.
 */
bool mimo26_rocm_rope_apply(void *heads, const void *cos_table,
                            const void *sin_table, uint32_t head_count,
                            void *stream);

/* Batched forms for prefill: one chunk of tokens at a time. The cos/sin
 * tables are [count][64], one pair per token, built on the host. */
bool mimo26_rocm_split_qkv_batch(void *q, void *k, void *v,
                                 const void *fused, uint32_t kv_heads,
                                 uint32_t qkv_width, uint32_t count,
                                 void *stream);
bool mimo26_rocm_rope_apply_batch(void *heads, const void *cos_tables,
                                  const void *sin_tables,
                                  uint32_t head_count, uint32_t count,
                                  void *stream);
/* Append a chunk's rotated keys and scaled values into the history at
 * `offset`, so attention sees them through the same arrays decode uses. */
bool mimo26_rocm_append_kv(void *keys, void *values, const void *chunk_keys,
                           const void *chunk_values, uint32_t kv_heads,
                           uint64_t offset, uint32_t count, void *stream);

/*
 * Run only the score pass, leaving the raw per-slot scores in scratch:
 * -INFINITY for a masked slot, the BF16-rounded scaled dot product for a
 * visible one, and the sink logit at index slot_total when sink_bias is
 * given. No transcendental is involved, so this is exactly specified and can
 * be held to bit-exactness against the CPU -- which is the point. Device and
 * host libm disagree by up to 1 ulp on roughly 6% of expf inputs, so the full
 * decode cannot be; checking the scores separately keeps a strict gate on
 * every layout, masking and ordering decision rather than letting a single
 * output tolerance cover all of them.
 */
bool mimo26_rocm_attention_scores(const void *query, const void *keys,
                                  const void *current_keys,
                                  const void *sink_bias, float *scratch,
                                  uint32_t kv_heads, uint32_t kv_groups,
                                  uint32_t window, uint64_t history,
                                  uint64_t first_position,
                                  uint64_t query_position, float scale,
                                  void *stream);

/* Floats of scratch mimo26_rocm_attention_decode needs for a given history. */
uint64_t mimo26_rocm_attention_scratch_floats(uint64_t history);

#ifdef __cplusplus
}
#endif

#endif
