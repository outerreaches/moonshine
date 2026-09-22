#ifndef MIMO26_OPS_H
#define MIMO26_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIMO26_LAYERNORM_EPSILON 1.0e-6f

typedef enum {
    MIMO26_OPS_OK = 0,
    MIMO26_OPS_INVALID_ARGUMENT,
    MIMO26_OPS_NONFINITE_VALUE
} mimo26_ops_status;

/*
 * BF16 is carried as uint16_t. f32 -> bf16 is round-to-nearest-even, matching
 * hip_bfloat16(float) as verified directly on gfx1151, so the CPU reference
 * and the GPU kernels round identically. NaN payloads are preserved as
 * quiet NaN rather than normalized.
 */
float mimo26_bf16_to_f32(uint16_t value);
uint16_t mimo26_f32_to_bf16(float value);

/*
 * RMSNorm following the reference's cast sequence exactly:
 *
 *   x32      = (f32)x
 *   variance = mean(x32^2)                     in f32
 *   x32      = x32 * rsqrt(variance + eps)     in f32
 *   out      = weight * (bf16)x32              bf16 * bf16 -> bf16
 *
 * The cast back to BF16 happens **before** the weight multiply, so the value
 * is rounded twice. Multiplying in f32 and rounding once gives a different
 * result and is wrong here, however natural it looks.
 */
mimo26_ops_status mimo26_rmsnorm_bf16(uint16_t *out, const uint16_t *in,
                                      const uint16_t *weight, size_t count,
                                      float epsilon);

/*
 * out = silu(gate) * up, with silu(x) = x * sigmoid(x).
 *
 * Plain, unclamped. GLM's limited SwiGLU caps gate above and clamps up to a
 * symmetric range; MiMo's reference MLP does neither, so that primitive is
 * NOT interchangeable with this one.
 */
mimo26_ops_status mimo26_silu_product_f32(float *out, const float *gate,
                                          const float *up, size_t count);

/*
 * Weighted expert accumulation. The reference allocates the accumulator with
 * the router weight dtype, which is f32, multiplies each BF16 expert output
 * by its f32 weight, accumulates in f32, and casts to BF16 once at the end.
 *
 * Accumulate every selected expert with mimo26_expert_accumulate_f32, in the
 * ascending expert-id order the router declares, then finalize once. Rounding
 * per expert instead would be a different and incorrect contract.
 */
mimo26_ops_status mimo26_expert_accumulate_f32(float *accumulator,
                                               const uint16_t *expert_output,
                                               float weight, size_t count);
mimo26_ops_status mimo26_expert_finalize_bf16(uint16_t *out,
                                              const float *accumulator,
                                              size_t count);

/* Pre-norm residual: BF16 + BF16 -> BF16, one rounding. */
mimo26_ops_status mimo26_residual_add_bf16(uint16_t *out, const uint16_t *residual,
                                           const uint16_t *delta, size_t count);

#ifdef __cplusplus
}
#endif

#endif
