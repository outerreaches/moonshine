#ifndef GLM53_DENSE_OPS_H
#define GLM53_DENSE_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One official row-major OCP E4M3FN matrix and its row-major F32
 * dequant-multiplier grid [ceil(rows/128), ceil(columns/128)]. Counts are in
 * elements (bytes for weights, floats for inverse_scales). */
typedef struct glm53_dense_fp8_matrix {
    const void *weights;
    size_t weights_count;
    const void *inverse_scales;
    size_t inverse_scales_count;
} glm53_dense_fp8_matrix;

/* Caller-owned device workspace. Counts are in elements of the documented
 * type. q8 is raw OCP E4M3FN bytes. dynamic_inverse_scales is F32. */
typedef struct glm53_dense_scratch {
    void *input_f32;                 /* F32 [hidden] */
    size_t input_f32_count;
    void *gate_f32;                  /* F32 [intermediate] */
    size_t gate_f32_count;
    void *up_f32;                    /* F32 [intermediate] */
    size_t up_f32_count;
    void *activation_f32;            /* F32 [intermediate] */
    size_t activation_f32_count;
    void *down_f32;                  /* F32 [hidden] */
    size_t down_f32_count;
    void *q8;                        /* uint8_t [max(hidden, intermediate)] */
    size_t q8_count;
    void *dynamic_inverse_scales;    /* F32 [max(hidden, intermediate)/128] */
    size_t dynamic_inverse_scale_count;
} glm53_dense_scratch;

/*
 * Allocation-free asynchronous batch-1 dense MLP:
 *   gate = gate_proj(input), up = up_proj(input)
 *   activation = silu(min(gate, limit)) * clamp(up, -limit, limit)
 *   output = down_proj(activation)
 *
 * input/output are BF16 [hidden]. gate/up are [intermediate, hidden] and down
 * is [hidden, intermediate]. Both dimensions must be nonzero multiples of
 * 128. Every complete accessible span named by the arguments and scratch must
 * be pairwise disjoint. All nine kernels are enqueued, in order, on the caller
 * stream (hipStream_t passed as void *); the call allocates and synchronizes
 * nothing. It reports validation and immediate kernel-launch status only.
 */
bool glm53_dense_mlp_fp8_bf16(
    void *output, size_t output_count,
    const void *input, size_t input_count,
    const glm53_dense_fp8_matrix *gate,
    const glm53_dense_fp8_matrix *up,
    const glm53_dense_fp8_matrix *down,
    uint32_t hidden, uint32_t intermediate, float limit,
    glm53_dense_scratch *scratch, void *stream);

#ifdef __cplusplus
}
#endif
#endif
