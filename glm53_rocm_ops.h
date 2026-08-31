#ifndef GLM53_ROCM_OPS_H
#define GLM53_ROCM_OPS_H

#include <stdbool.h>
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
