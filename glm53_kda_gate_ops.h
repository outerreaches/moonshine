#ifndef GLM53_KDA_GATE_OPS_H
#define GLM53_KDA_GATE_OPS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_KDA_GATE_HEAD_DIM 128u
#define GLM53_KDA_GATE_OFFICIAL_HEADS 64u
#define GLM53_KDA_GATE_OFFICIAL_LOWER_BOUND (-5.0f)

/* Allocation-free asynchronous preparation of the KDA recurrence gates.
 * Pointers name contiguous device storage, counts are elements, and stream is
 * a hipStream_t passed as void *. Every buffer in one call must be pairwise
 * disjoint. Calls enqueue only on the caller's stream. */

/* raw_forget is BF16 [heads,head_dim], dt_bias and output are F32 with the
 * same shape, and A_log is F32 [heads]. head_dim must be 128. Computes:
 *   output[h,d] = lower_bound * sigmoid(exp(A_log[h]) *
 *                                         (float(raw_forget[h,d])+dt_bias[h,d]))
 * Production uses heads=64 and lower_bound=-5. lower_bound may be any finite
 * negative value. */
bool glm53_kda_prepare_forget_f32(
    void *output, size_t output_count,
    const void *raw_forget, size_t raw_forget_count,
    const void *dt_bias, size_t dt_bias_count,
    const void *A_log, size_t A_log_count,
    size_t heads, size_t head_dim, float lower_bound, void *stream);

/* raw_beta is BF16 [heads], output is F32 [heads]. This intentionally rounds
 * sigmoid(float(raw_beta)) to BF16 (round-to-nearest-even) and then promotes
 * that BF16 value to F32, matching torch.sigmoid on a BF16 tensor. */
bool glm53_kda_prepare_beta_f32(
    void *output, size_t output_count,
    const void *raw_beta, size_t raw_beta_count,
    size_t heads, void *stream);

#ifdef __cplusplus
}
#endif
#endif
