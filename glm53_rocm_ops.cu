#include "glm53_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

enum {
    GLM53_ROCM_THREADS = 256,
    GLM53_ROCM_BLOCK = 128,
};

/* Exact OCP E4M3FN decode. Only 0x7f and 0xff are NaNs. */
__device__ static inline float glm53_e4m3fn_to_float(uint8_t encoded) {
    const uint32_t sign = (uint32_t)(encoded & UINT8_C(0x80)) << 24u;
    const uint32_t exponent = ((uint32_t)encoded >> 3u) & 15u;
    const uint32_t fraction = (uint32_t)encoded & 7u;

    if (exponent == 15u && fraction == 7u) {
        return __uint_as_float(UINT32_C(0x7fc00000));
    }
    if (exponent == 0u) {
        if (fraction == 0u) return __uint_as_float(sign);
        const uint32_t top = fraction >= 4u ? 2u :
                             (fraction >= 2u ? 1u : 0u);
        const uint32_t ieee_exponent = 118u + top;
        const uint32_t ieee_fraction =
            (fraction - (UINT32_C(1) << top)) << (23u - top);
        return __uint_as_float(sign | (ieee_exponent << 23u) |
                               ieee_fraction);
    }
    return __uint_as_float(sign | ((exponent + 120u) << 23u) |
                           (fraction << 20u));
}

__global__ static void glm53_fp8_gemv_f32_kernel(
        float               *output,
        const uint8_t       *weights,
        const float         *inverse_scales,
        const hip_bfloat16  *input,
        uint32_t columns,
        uint32_t scale_columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t weight_base = (uint64_t)row * columns;
    const uint64_t scale_base =
        (uint64_t)(row / GLM53_ROCM_BLOCK) * scale_columns;
    float partial = 0.0f;
    __shared__ float reduction[GLM53_ROCM_THREADS];

    for (uint32_t column = tid; column < columns;
         column += GLM53_ROCM_THREADS) {
        const float weight =
            glm53_e4m3fn_to_float(weights[weight_base + column]);
        const float scale =
            inverse_scales[scale_base + column / GLM53_ROCM_BLOCK];
        partial += (weight * scale) * (float)input[column];
    }
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = GLM53_ROCM_THREADS / 2u; width != 0u;
         width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0u) output[row] = reduction[0];
}

__global__ static void glm53_fp8_dequantize_bf16_kernel(
        hip_bfloat16 *output,
        const uint8_t *weights,
        const float *inverse_scales,
        uint64_t element_count,
        uint32_t columns,
        uint32_t scale_columns) {
    const uint64_t index =
        (uint64_t)blockIdx.x * GLM53_ROCM_THREADS + threadIdx.x;
    if (index >= element_count) return;
    const uint32_t row = (uint32_t)(index / columns);
    const uint32_t column = (uint32_t)(index - (uint64_t)row * columns);
    const uint64_t scale_index =
        (uint64_t)(row / GLM53_ROCM_BLOCK) * scale_columns +
        column / GLM53_ROCM_BLOCK;
    output[index] = hip_bfloat16(
        glm53_e4m3fn_to_float(weights[index]) * inverse_scales[scale_index]);
}

static bool glm53_checked_geometry(uint32_t rows, uint32_t columns,
                                   uint64_t *elements,
                                   uint32_t *scale_columns) {
    if (rows == 0u || columns == 0u || elements == NULL ||
        scale_columns == NULL) return false;
    const uint64_t product = (uint64_t)rows * (uint64_t)columns;
    if (product == 0u || product > (uint64_t)SIZE_MAX / sizeof(hip_bfloat16))
        return false;
    const uint64_t scale_cols =
        ((uint64_t)columns + GLM53_ROCM_BLOCK - 1u) / GLM53_ROCM_BLOCK;
    const uint64_t scale_rows =
        ((uint64_t)rows + GLM53_ROCM_BLOCK - 1u) / GLM53_ROCM_BLOCK;
    if (scale_cols == 0u || scale_cols > UINT32_MAX ||
        scale_rows > UINT64_MAX / scale_cols ||
        scale_rows * scale_cols > (uint64_t)SIZE_MAX / sizeof(float) ||
        (uint64_t)rows > (uint64_t)SIZE_MAX / sizeof(float)) return false;
    *elements = product;
    *scale_columns = (uint32_t)scale_cols;
    return true;
}

extern "C" bool glm53_rocm_fp8_gemv_f32(
        void *output, const void *weights, const void *inverse_scales,
        const void *input, uint32_t rows, uint32_t columns,
        void *stream_pointer) {
    uint64_t elements;
    uint32_t scale_columns;
    if (output == NULL || weights == NULL || inverse_scales == NULL ||
        input == NULL ||
        !glm53_checked_geometry(rows, columns, &elements, &scale_columns)) {
        return false;
    }
    (void)elements;
    /* A one-dimensional HIP grid dimension is represented by uint32_t. */
    if (rows > (uint32_t)INT_MAX) return false;
    (void)hipGetLastError(); /* clear unrelated stale thread-local status */
    hipLaunchKernelGGL(glm53_fp8_gemv_f32_kernel, dim3(rows),
                       dim3(GLM53_ROCM_THREADS), 0,
                       (hipStream_t)stream_pointer,
                       (float *)output, (const uint8_t *)weights,
                       (const float *)inverse_scales,
                       (const hip_bfloat16 *)input, columns, scale_columns);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool glm53_rocm_fp8_dequantize_bf16(
        void *output, const void *weights, const void *inverse_scales,
        uint32_t rows, uint32_t columns, void *stream_pointer) {
    uint64_t elements;
    uint32_t scale_columns;
    if (output == NULL || weights == NULL || inverse_scales == NULL ||
        !glm53_checked_geometry(rows, columns, &elements, &scale_columns)) {
        return false;
    }
    const uint64_t blocks =
        (elements + GLM53_ROCM_THREADS - 1u) / GLM53_ROCM_THREADS;
    if (blocks == 0u || blocks > (uint64_t)INT_MAX) return false;
    (void)hipGetLastError(); /* clear unrelated stale thread-local status */
    hipLaunchKernelGGL(glm53_fp8_dequantize_bf16_kernel,
                       dim3((uint32_t)blocks), dim3(GLM53_ROCM_THREADS), 0,
                       (hipStream_t)stream_pointer,
                       (hip_bfloat16 *)output, (const uint8_t *)weights,
                       (const float *)inverse_scales, elements, columns,
                       scale_columns);
    return hipGetLastError() == hipSuccess;
}
