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

#ifdef __cplusplus
}
#endif

#endif
