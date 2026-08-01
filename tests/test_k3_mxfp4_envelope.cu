#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                     \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                    \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    K3_ENVELOPE_ROWS = 512,
    K3_ENVELOPE_VECTORS = 256,
    K3_SCALAR_THREADS = 256,
};

static uint32_t rng_state = UINT32_C(0x6d786670);

static uint32_t next_random_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float random_symmetric(float scale) {
    const float unit =
        (float)(next_random_u32() & UINT32_C(0x00ffffff)) /
        (float)UINT32_C(0x01000000);
    return (unit * 2.0f - 1.0f) * scale;
}

static inline float bf16_to_float(hip_bfloat16 value) {
    return (float)value;
}

static inline hip_bfloat16 float_to_bf16(float value) {
    return hip_bfloat16(value);
}

__device__ static inline float scalar_e8m0_to_float(uint8_t exponent) {
    return exponent == 0u ?
        __uint_as_float(UINT32_C(0x00400000)) :
        __uint_as_float((uint32_t)exponent << 23u);
}

__device__ static inline float scalar_e2m1_to_float(uint8_t nibble) {
    const float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float value = magnitude[nibble & 7u];
    return (nibble & 8u) ? -value : value;
}

/* Exact diagnostic copy of the pre-f68aa08 scalar/256-thread schedule. */
__global__ static void scalar_mxfp4_gemv_kernel(
        hip_bfloat16       *output,
        const uint8_t      *packed,
        const uint8_t      *scales,
        const hip_bfloat16 *input,
        uint32_t columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t vector = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    output += (uint64_t)vector * gridDim.x;
    input += (uint64_t)vector * columns;
    const uint64_t packed_base = (uint64_t)row * (columns / 2u);
    const uint64_t scale_base = (uint64_t)row * (columns / 32u);
    __shared__ float reduction[K3_SCALAR_THREADS];

    float partial = 0.0f;
    for (uint32_t column = tid;
         column < columns;
         column += blockDim.x) {
        const uint8_t byte = packed[packed_base + column / 2u];
        const uint8_t nibble =
            (column & 1u) ? byte >> 4u : byte & 0x0fu;
        const float weight = scalar_e2m1_to_float(nibble) *
            scalar_e8m0_to_float(
                scales[scale_base + column / 32u]);
        partial += weight * (float)input[column];
    }
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u;
         width > 0u;
         width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0u) output[row] = hip_bfloat16(reduction[0]);
}

static float host_e8m0_to_float(uint8_t exponent) {
    union {
        uint32_t bits;
        float value;
    } decoded;
    decoded.bits = exponent == 0u ?
        UINT32_C(0x00400000) : (uint32_t)exponent << 23u;
    return decoded.value;
}

static float host_e2m1_to_float(uint8_t nibble) {
    static const float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float value = magnitude[nibble & 7u];
    return (nibble & 8u) ? -value : value;
}

static uint16_t bf16_bits(hip_bfloat16 value) {
    uint16_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint32_t ordered_bf16(hip_bfloat16 value) {
    const uint16_t bits = bf16_bits(value);
    return (bits & UINT16_C(0x8000)) != 0u ?
        (uint32_t)(UINT16_C(0xffff) - bits) :
        (uint32_t)bits + UINT32_C(0x8000);
}

static uint32_t bf16_ulp_distance(hip_bfloat16 left,
                                  hip_bfloat16 right) {
    const uint32_t a = ordered_bf16(left);
    const uint32_t b = ordered_bf16(right);
    return a > b ? a - b : b - a;
}

static void fill_inputs(hip_bfloat16 *input, uint32_t columns) {
    for (uint32_t vector = 0u;
         vector < K3_ENVELOPE_VECTORS;
         vector++) {
        const uint32_t pattern = vector % 6u;
        const float scale =
            ldexpf(1.0f, (int)((vector / 6u) % 4u) - 3);
        for (uint32_t column = 0u; column < columns; column++) {
            float value = 0.0f;
            switch (pattern) {
                case 0u:
                    value = random_symmetric(scale);
                    break;
                case 1u:
                    value = (column & 1u) ? -scale : scale;
                    break;
                case 2u:
                    value = (column % 29u == 0u) ?
                        random_symmetric(scale * 8.0f) : 0.0f;
                    break;
                case 3u: {
                    const float magnitude =
                        (float)(column % 17u + 1u) * scale / 17.0f;
                    value = (column % 4u < 2u) ? magnitude : -magnitude;
                    break;
                }
                case 4u:
                    value = ((int32_t)(column % 31u) - 15) *
                        scale / 16.0f;
                    break;
                default:
                    value = random_symmetric(scale) +
                        ((column & 1u) ? -0.00390625f : 0.00390625f) *
                        scale;
                    break;
            }
            input[(uint64_t)vector * columns + column] =
                float_to_bf16(value);
        }
    }
}

static int run_shape(uint32_t columns) {
    const uint32_t rows = K3_ENVELOPE_ROWS;
    const uint32_t vectors = K3_ENVELOPE_VECTORS;
    const size_t packed_bytes = (size_t)rows * columns / 2u;
    const size_t scale_bytes = (size_t)rows * columns / 32u;
    const size_t input_bytes =
        (size_t)vectors * columns * sizeof(hip_bfloat16);
    const size_t output_count = (size_t)vectors * rows;
    const size_t output_bytes = output_count * sizeof(hip_bfloat16);
    uint8_t *packed = (uint8_t *)malloc(packed_bytes);
    uint8_t *scales = (uint8_t *)malloc(scale_bytes);
    hip_bfloat16 *input = (hip_bfloat16 *)malloc(input_bytes);
    hip_bfloat16 *scalar = (hip_bfloat16 *)malloc(output_bytes);
    hip_bfloat16 *vectorized = (hip_bfloat16 *)malloc(output_bytes);
    CHECK(packed && scales && input && scalar && vectorized,
          "MXFP4 envelope host allocation");

    for (size_t i = 0u; i < packed_bytes; i++) {
        packed[i] = (uint8_t)next_random_u32();
    }
    for (size_t i = 0u; i < scale_bytes; i++) {
        scales[i] = (uint8_t)(123u + (next_random_u32() % 9u));
    }
    fill_inputs(input, columns);

    void *d_packed = NULL;
    void *d_scales = NULL;
    void *d_input = NULL;
    void *d_scalar = NULL;
    void *d_vectorized = NULL;
    HIP_CHECK(hipMalloc(&d_packed, packed_bytes));
    HIP_CHECK(hipMalloc(&d_scales, scale_bytes));
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_scalar, output_bytes));
    HIP_CHECK(hipMalloc(&d_vectorized, output_bytes));
    HIP_CHECK(hipMemcpy(
        d_packed, packed, packed_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_scales, scales, scale_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_input, input, input_bytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(
        scalar_mxfp4_gemv_kernel,
        dim3(rows, vectors), dim3(K3_SCALAR_THREADS), 0u, NULL,
        (hip_bfloat16 *)d_scalar,
        (const uint8_t *)d_packed,
        (const uint8_t *)d_scales,
        (const hip_bfloat16 *)d_input,
        columns);
    HIP_CHECK(hipGetLastError());
    CHECK(k3_rocm_mxfp4_gemm_bf16(
              d_vectorized, d_packed, d_scales, d_input,
              vectors, rows, columns, NULL),
          "production MXFP4 envelope launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(
        scalar, d_scalar, output_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        vectorized, d_vectorized, output_bytes, hipMemcpyDeviceToHost));

    uint64_t differing = 0u;
    uint32_t maximum_ulp = 0u;
    float maximum_cross_absolute = 0.0f;
    float maximum_cross_relative = 0.0f;
    float maximum_scalar_reference_absolute = 0.0f;
    float maximum_vector_reference_absolute = 0.0f;
    float maximum_scalar_reference_relative = 0.0f;
    float maximum_vector_reference_relative = 0.0f;
    for (uint32_t vector = 0u; vector < vectors; vector++) {
        for (uint32_t row = 0u; row < rows; row++) {
            double reference = 0.0;
            for (uint32_t column = 0u; column < columns; column++) {
                const uint8_t byte =
                    packed[(uint64_t)row * columns / 2u + column / 2u];
                const uint8_t nibble =
                    (column & 1u) ? byte >> 4u : byte & 0x0fu;
                const double weight =
                    (double)host_e2m1_to_float(nibble) *
                    (double)host_e8m0_to_float(
                        scales[(uint64_t)row * columns / 32u +
                               column / 32u]);
                reference += weight * (double)bf16_to_float(
                    input[(uint64_t)vector * columns + column]);
            }
            const size_t index = (size_t)vector * rows + row;
            const float scalar_value = bf16_to_float(scalar[index]);
            const float vector_value = bf16_to_float(vectorized[index]);
            CHECK(isfinite(scalar_value) && isfinite(vector_value) &&
                      isfinite(reference),
                  "MXFP4 envelope produced a non-finite value");
            const float cross_absolute =
                fabsf(scalar_value - vector_value);
            const float cross_relative = cross_absolute /
                fmaxf(fmaxf(fabsf(scalar_value), fabsf(vector_value)),
                      1e-3f);
            const float scalar_reference_absolute =
                fabsf(scalar_value - (float)reference);
            const float vector_reference_absolute =
                fabsf(vector_value - (float)reference);
            const float reference_denominator =
                fmaxf(fabsf((float)reference), 1e-3f);
            const float scalar_reference_relative =
                scalar_reference_absolute / reference_denominator;
            const float vector_reference_relative =
                vector_reference_absolute / reference_denominator;
            if (bf16_bits(scalar[index]) != bf16_bits(vectorized[index])) {
                differing++;
            }
            const uint32_t ulp =
                bf16_ulp_distance(scalar[index], vectorized[index]);
            if (ulp > maximum_ulp) maximum_ulp = ulp;
            CHECK(ulp <= 1u,
                  "MXFP4 schedules differ by more than one BF16 ULP");
            if (cross_absolute > maximum_cross_absolute) {
                maximum_cross_absolute = cross_absolute;
            }
            if (cross_relative > maximum_cross_relative) {
                maximum_cross_relative = cross_relative;
            }
            if (scalar_reference_absolute >
                    maximum_scalar_reference_absolute) {
                maximum_scalar_reference_absolute =
                    scalar_reference_absolute;
            }
            if (vector_reference_absolute >
                    maximum_vector_reference_absolute) {
                maximum_vector_reference_absolute =
                    vector_reference_absolute;
            }
            if (scalar_reference_relative >
                    maximum_scalar_reference_relative) {
                maximum_scalar_reference_relative =
                    scalar_reference_relative;
            }
            if (vector_reference_relative >
                    maximum_vector_reference_relative) {
                maximum_vector_reference_relative =
                    vector_reference_relative;
            }
            CHECK(cross_absolute <= 0.25f || cross_relative <= 0.01f,
                  "MXFP4 schedules exceed the cross-schedule envelope");
            CHECK(scalar_reference_absolute <= 0.25f ||
                      scalar_reference_relative <= 0.01f,
                  "scalar MXFP4 exceeds the FP64 envelope");
            CHECK(vector_reference_absolute <= 0.25f ||
                      vector_reference_relative <= 0.01f,
                  "vector MXFP4 exceeds the FP64 envelope");
        }
    }
    CHECK(differing != 0u,
          "MXFP4 envelope did not cross any BF16 rounding boundary");
    printf(
        "  columns=%u vectors=%u outputs=%zu changed=%" PRIu64
        " max_ulp=%u cross_abs=%.6f cross_rel=%.6f "
        "scalar_ref=%.6f/%.6f vector_ref=%.6f/%.6f\n",
        columns, vectors, output_count, differing, maximum_ulp,
        maximum_cross_absolute, maximum_cross_relative,
        maximum_scalar_reference_absolute,
        maximum_scalar_reference_relative,
        maximum_vector_reference_absolute,
        maximum_vector_reference_relative);

    HIP_CHECK(hipFree(d_vectorized));
    HIP_CHECK(hipFree(d_scalar));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_scales));
    HIP_CHECK(hipFree(d_packed));
    free(vectorized);
    free(scalar);
    free(input);
    free(scales);
    free(packed);
    return 0;
}

int main(void) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 MXFP4 numerical envelope: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    printf("K3 MXFP4 scalar/vector numerical envelope\n");
    if (run_shape(3584u) != 0) return 1;
    if (run_shape(3072u) != 0) return 1;
    printf("K3 MXFP4 numerical envelope: PASS\n");
    return 0;
}
