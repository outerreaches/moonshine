#ifndef GLM53_ROCM_OPS_H
#define GLM53_ROCM_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * All tensors are contiguous device storage. weights is OCP E4M3FN raw bytes
 * in row-major [rows, columns] order. inverse_scales is F32 with shape
 * [ceil(rows / 128), ceil(columns / 128)]. input is BF16 [columns], and
 * output is F32 [rows]. The stream is a hipStream_t passed as void *; NULL
 * selects the default stream. The call is asynchronous and reports only
 * argument/geometry and kernel-launch status.
 */
bool glm53_rocm_fp8_gemv_f32(void       *output,
                              const void *weights,
                              const void *inverse_scales,
                              const void *input,
                              uint32_t    rows,
                              uint32_t    columns,
                              void       *stream);


/*
 * Batch-1 correctness path for activation_scheme=dynamic. input is contiguous
 * F32 [columns]. The first kernel writes OCP E4M3FN activation bytes
 * [columns] and F32 dequant multipliers [columns / 128]. The second kernel
 * consumes those buffers and official row-major OCP E4M3FN weights
 * [rows, columns] with F32 [ceil(rows / 128), columns / 128] dequant multipliers, and writes F32 [rows]. columns must be a multiple of 128;
 * rows may be any nonzero value.
 *
 * Every count is in elements of its pointed-to type. Buffers must cover the
 * complete logical tensor. The six accessible spans described by the counts
 * must be pairwise non-overlapping. The function
 * allocates no memory, enqueues both kernels in order on stream, does not
 * synchronize, clears stale HIP launch status, and reports argument,
 * geometry, capacity, address-overflow, non-overlap, and launch errors.
 * Input activations and weight scales must be finite; weight scales must be
 * positive; weight bytes must not use the E4M3FN NaN codes 0x7f/0xff. These
 * are asynchronous data preconditions and are not validated by the wrapper.
 */
bool glm53_rocm_fp8_dynamic_gemv_f32(
    void *output, size_t output_count,
    void *quantized_input, size_t quantized_input_count,
    void *input_inverse_scales, size_t input_inverse_scale_count,
    const void *weights, size_t weights_count,
    const void *weight_inverse_scales, size_t weight_inverse_scale_count,
    const void *input, size_t input_count,
    uint32_t rows, uint32_t columns, void *stream);

/*
 * Dequantize the same contiguous block-scaled representation to contiguous
 * BF16 [rows, columns] workspace. No allocation or synchronization occurs.
 */
bool glm53_rocm_fp8_dequantize_bf16(void       *output,
                                     const void *weights,
                                     const void *inverse_scales,
                                     uint32_t    rows,
                                     uint32_t    columns,
                                     void       *stream);

#ifdef __cplusplus
}
#endif

#endif
