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

__device__ static inline float glm53_e4m3fn_finite_magnitude(
        uint32_t code) {
    const uint32_t exponent = (code >> 3u) & 15u;
    const uint32_t fraction = code & 7u;
    if (exponent == 0u) return ldexpf((float)fraction, -9);
    return ldexpf(1.0f + (float)fraction * 0.125f,
                  (int)exponent - 7);
}

/* Software OCP E4M3FN RNE. The kernel clamps to 448 before this call. */
__device__ static inline uint8_t glm53_e4m3fn_encode_finite(float value) {
    const uint32_t bits = __float_as_uint(value);
    const float magnitude = fabsf(value);
    uint32_t best = 0u;
    float best_distance = magnitude;
    for (uint32_t code = 1u; code <= 0x7eu; ++code) {
        const float candidate = glm53_e4m3fn_finite_magnitude(code);
        const float distance = fabsf(magnitude - candidate);
        if (distance < best_distance ||
            (distance == best_distance && (code & 1u) == 0u &&
             (best & 1u) != 0u)) {
            best = code;
            best_distance = distance;
        }
    }
    return (uint8_t)(best | ((bits >> 24u) & 0x80u));
}

__global__ static void glm53_fp8_dynamic_quantize_f32_kernel(
        uint8_t *quantized, float *inverse_scales, const float *input) {
    const uint32_t group = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t column = group * GLM53_ROCM_BLOCK + tid;
    __shared__ float magnitudes[GLM53_ROCM_BLOCK];
    __shared__ float scale;

    magnitudes[tid] = fabsf(input[column]);
    __syncthreads();
    for (uint32_t width = GLM53_ROCM_BLOCK / 2u; width != 0u;
         width /= 2u) {
        if (tid < width && magnitudes[tid + width] > magnitudes[tid])
            magnitudes[tid] = magnitudes[tid + width];
        __syncthreads();
    }
    if (tid == 0u) {
        float absmax = magnitudes[0];
        if (absmax < 1.0e-10f) absmax = 1.0e-10f;
        const float value = absmax * (1.0f / 448.0f);
        scale = value;
        inverse_scales[group] = value;
    }
    __syncthreads();
    float scaled = input[column] / scale;
    if (scaled > 448.0f) scaled = 448.0f;
    else if (scaled < -448.0f) scaled = -448.0f;
    {
        uint8_t code = glm53_e4m3fn_encode_finite(scaled);
        const uint32_t input_bits = __float_as_uint(input[column]);
        code = (uint8_t)((code & 0x7fu) | ((input_bits >> 24u) & 0x80u));
        quantized[column] = code;
    }
}

__global__ static void glm53_fp8_dynamic_gemv_f32_kernel(
        float *output, const uint8_t *weights,
        const float *weight_inverse_scales,
        const uint8_t *quantized_input,
        const float *input_inverse_scales,
        uint32_t columns, uint32_t scale_columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t weight_base = (uint64_t)row * columns;
    const uint64_t scale_base =
        (uint64_t)(row / GLM53_ROCM_BLOCK) * scale_columns;
    float partial = 0.0f;
    __shared__ float reduction[GLM53_ROCM_THREADS];
    for (uint32_t column = tid; column < columns;
         column += GLM53_ROCM_THREADS) {
        const uint32_t group = column / GLM53_ROCM_BLOCK;
        const float w = glm53_e4m3fn_to_float(
            weights[weight_base + column]);
        const float x = glm53_e4m3fn_to_float(quantized_input[column]);
        partial += (w * weight_inverse_scales[scale_base + group]) *
                   (x * input_inverse_scales[group]);
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

static bool glm53_required_byte_span(const void *pointer, size_t count,
                                      size_t element_size,
                                      uintptr_t *begin, uintptr_t *end) {
    if (pointer == NULL || count == 0u ||
        count > SIZE_MAX / element_size) return false;
    const size_t bytes = count * element_size;
    const uintptr_t first = (uintptr_t)pointer;
    if (first > UINTPTR_MAX - bytes) return false;
    *begin = first;
    *end = first + bytes;
    return true;
}

static bool glm53_spans_do_not_overlap(const uintptr_t *begin,
                                        const uintptr_t *end,
                                        size_t count) {
    for (size_t i = 0u; i < count; ++i) {
        for (size_t j = i + 1u; j < count; ++j) {
            if (begin[i] < end[j] && begin[j] < end[i]) return false;
        }
    }
    return true;
}

extern "C" bool glm53_rocm_fp8_dynamic_gemv_f32(
        void *output, size_t output_count,
        void *quantized_input, size_t quantized_input_count,
        void *input_inverse_scales, size_t input_inverse_scale_count,
        const void *weights, size_t weights_count,
        const void *weight_inverse_scales, size_t weight_inverse_scale_count,
        const void *input, size_t input_count,
        uint32_t rows, uint32_t columns, void *stream_pointer) {
    if (output == NULL || quantized_input == NULL ||
        input_inverse_scales == NULL || weights == NULL ||
        weight_inverse_scales == NULL || input == NULL || rows == 0u ||
        columns == 0u || columns % GLM53_ROCM_BLOCK != 0u ||
        rows > (uint32_t)INT_MAX) {
        return false;
    }
    const size_t row_count = (size_t)rows;
    const size_t column_count = (size_t)columns;
    if (row_count > SIZE_MAX / column_count) return false;
    const size_t weight_elements = row_count * column_count;
    const size_t scale_columns = column_count / GLM53_ROCM_BLOCK;
    const size_t scale_rows =
        (row_count + GLM53_ROCM_BLOCK - 1u) / GLM53_ROCM_BLOCK;
    if (scale_rows > SIZE_MAX / scale_columns) return false;
    const size_t weight_scale_elements = scale_rows * scale_columns;
    if (output_count < row_count || quantized_input_count < column_count ||
        input_inverse_scale_count < scale_columns ||
        weights_count < weight_elements ||
        weight_inverse_scale_count < weight_scale_elements ||
        input_count < column_count) return false;

    uintptr_t begin[6], end[6];
    if (!glm53_required_byte_span(output, output_count, sizeof(float),
                                  &begin[0], &end[0]) ||
        !glm53_required_byte_span(quantized_input, quantized_input_count,
                                  sizeof(uint8_t), &begin[1], &end[1]) ||
        !glm53_required_byte_span(input_inverse_scales, input_inverse_scale_count,
                                  sizeof(float), &begin[2], &end[2]) ||
        !glm53_required_byte_span(weights, weights_count,
                                  sizeof(uint8_t), &begin[3], &end[3]) ||
        !glm53_required_byte_span(weight_inverse_scales,
                                  weight_inverse_scale_count, sizeof(float),
                                  &begin[4], &end[4]) ||
        !glm53_required_byte_span(input, input_count, sizeof(float),
                                  &begin[5], &end[5]) ||
        !glm53_spans_do_not_overlap(begin, end, 6u)) return false;

    (void)hipGetLastError();
    hipLaunchKernelGGL(glm53_fp8_dynamic_quantize_f32_kernel,
                       dim3((uint32_t)scale_columns),
                       dim3(GLM53_ROCM_BLOCK), 0,
                       (hipStream_t)stream_pointer,
                       (uint8_t *)quantized_input,
                       (float *)input_inverse_scales,
                       (const float *)input);
    if (hipGetLastError() != hipSuccess) return false;
    hipLaunchKernelGGL(glm53_fp8_dynamic_gemv_f32_kernel, dim3(rows),
                       dim3(GLM53_ROCM_THREADS), 0,
                       (hipStream_t)stream_pointer,
                       (float *)output, (const uint8_t *)weights,
                       (const float *)weight_inverse_scales,
                       (const uint8_t *)quantized_input,
                       (const float *)input_inverse_scales, columns,
                       (uint32_t)scale_columns);
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
