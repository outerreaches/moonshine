#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <stdint.h>
#include <stdlib.h>

struct k3_rocm_blas_context {
    hipblasHandle_t handle;
};

enum {
    K3_ROCM_THREADS = 256,
    K3_ROCM_GEMV_THREADS = 256,
    K3_ROCM_MXFP4_THREADS = 128,
    K3_ROCM_BATCH_TILE = 16,
    K3_Q8_BLOCK_SIZE = 128,
    K3_ATTN_RES_MAX_SOURCES = 16,
    K3_KDA_HEAD_DIM = 128,
    K3_KDA_VALUE_TILE = 32,
    K3_MLA_NOPE_DIM = 128,
    K3_MLA_PASS_DIM = 64,
    K3_MLA_Q_DIM = K3_MLA_NOPE_DIM + K3_MLA_PASS_DIM,
    K3_MLA_LATENT_DIM = 512,
    K3_MLA_CACHE_DIM = K3_MLA_LATENT_DIM + K3_MLA_PASS_DIM,
    K3_MLA_V_DIM = 128,
    K3_MLA_KV_B_HEAD_STRIDE = K3_MLA_NOPE_DIM + K3_MLA_V_DIM,
};

__host__ __device__ static inline float
k3_bf16_to_float(hip_bfloat16 value) {
    return (float)value;
}

__host__ __device__ static inline hip_bfloat16
k3_float_to_bf16(float value) {
    return hip_bfloat16(value);
}

__device__ static inline float k3_packed_bf16x2_dot(uint32_t weights,
                                                    uint32_t input) {
    return
        __uint_as_float((weights & UINT32_C(0x0000ffff)) << 16u) *
        __uint_as_float((input & UINT32_C(0x0000ffff)) << 16u) +
        __uint_as_float(weights & UINT32_C(0xffff0000)) *
        __uint_as_float(input & UINT32_C(0xffff0000));
}

__device__ static inline float k3_packed_bf16_value(uint4 input,
                                                    uint32_t index) {
    uint32_t word;
    switch (index / 2u) {
        case 0: word = input.x; break;
        case 1: word = input.y; break;
        case 2: word = input.z; break;
        default: word = input.w; break;
    }
    uint32_t bits = (index & 1u) ?
        word & UINT32_C(0xffff0000) :
        (word & UINT32_C(0x0000ffff)) << 16u;
    return __uint_as_float(bits);
}

__global__ static void k3_situ_kernel(hip_bfloat16       *output,
                                      const hip_bfloat16 *gate,
                                      const hip_bfloat16 *up,
                                      uint64_t element_count,
                                      float beta,
                                      float linear_beta) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; index < element_count; index += stride) {
        float gate_value = k3_bf16_to_float(gate[index]);
        float up_value = k3_bf16_to_float(up[index]);
        float bounded_gate = beta * tanhf(gate_value / beta);
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_value));
        float bounded_up = linear_beta * tanhf(up_value / linear_beta);
        output[index] =
            k3_float_to_bf16(bounded_gate * sigmoid_gate * bounded_up);
    }
}

__global__ static void k3_attn_res_kernel(
        hip_bfloat16       *output,
        const hip_bfloat16 *prefix,
        const hip_bfloat16 *blocks,
        const hip_bfloat16 *norm_weight,
        const hip_bfloat16 *qk_weight,
        uint32_t block_capacity,
        uint32_t num_blocks,
        uint32_t hidden_size,
        float epsilon) {
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t num_sources = num_blocks + 1u;
    __shared__ float reduction[K3_ROCM_THREADS];
    __shared__ float logits[K3_ATTN_RES_MAX_SOURCES];
    __shared__ float probabilities[K3_ATTN_RES_MAX_SOURCES];

    const uint64_t prefix_base = (uint64_t)token * hidden_size;
    const uint64_t block_base =
        (uint64_t)token * block_capacity * hidden_size;

    for (uint32_t source = 0; source < num_sources; source++) {
        float sum_squares = 0.0f;
        const bool is_prefix = source == num_blocks;
        const uint64_t value_base =
            is_prefix ? prefix_base :
                        block_base + (uint64_t)source * hidden_size;
        for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
            float value = is_prefix ?
                k3_bf16_to_float(prefix[value_base + d]) :
                k3_bf16_to_float(blocks[value_base + d]);
            sum_squares += value * value;
        }
        reduction[tid] = sum_squares;
        __syncthreads();
        for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
            if (tid < width) reduction[tid] += reduction[tid + width];
            __syncthreads();
        }
        float reciprocal_std =
            rsqrtf(reduction[0] / (float)hidden_size + epsilon);

        float dot = 0.0f;
        for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
            float value = is_prefix ?
                k3_bf16_to_float(prefix[value_base + d]) :
                k3_bf16_to_float(blocks[value_base + d]);
            float key_weight = k3_bf16_to_float(norm_weight[d]) *
                               k3_bf16_to_float(qk_weight[d]);
            dot += value * key_weight;
        }
        reduction[tid] = dot;
        __syncthreads();
        for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
            if (tid < width) reduction[tid] += reduction[tid + width];
            __syncthreads();
        }
        if (tid == 0) logits[source] = reduction[0] * reciprocal_std;
        __syncthreads();
    }

    if (tid == 0) {
        float maximum = logits[0];
        for (uint32_t source = 1; source < num_sources; source++) {
            maximum = fmaxf(maximum, logits[source]);
        }
        float denominator = 0.0f;
        for (uint32_t source = 0; source < num_sources; source++) {
            float score = expf(logits[source] - maximum);
            probabilities[source] = score;
            denominator += score;
        }
        for (uint32_t source = 0; source < num_sources; source++) {
            probabilities[source] /= denominator;
        }
    }
    __syncthreads();

    for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
        float mixed = 0.0f;
        for (uint32_t source = 0; source < num_sources; source++) {
            bool is_prefix = source == num_blocks;
            uint64_t value_index = is_prefix ?
                prefix_base + d :
                block_base + (uint64_t)source * hidden_size + d;
            float value = is_prefix ?
                k3_bf16_to_float(prefix[value_index]) :
                k3_bf16_to_float(blocks[value_index]);
            mixed += probabilities[source] * value;
        }
        output[prefix_base + d] = k3_float_to_bf16(mixed);
    }
}

__device__ static inline float k3_e8m0_to_float(uint8_t exponent) {
    uint32_t bits = exponent == 0u ?
        UINT32_C(0x00400000) : (uint32_t)exponent << 23u;
    return __uint_as_float(bits);
}

__device__ static inline float k3_e2m1_to_float(uint8_t nibble) {
    const float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    float value = magnitude[nibble & 7u];
    return (nibble & 8u) ? -value : value;
}

__device__ static inline float k3_mxfp4_dot8(uint32_t packed,
                                             uint4 input,
                                             float scale) {
    float partial = 0.0f;
#pragma unroll
    for (uint32_t index = 0; index < 8u; index++) {
        const uint8_t nibble =
            (uint8_t)((packed >> (index * 4u)) & 0x0fu);
        partial += k3_e2m1_to_float(nibble) * scale *
                   k3_packed_bf16_value(input, index);
    }
    return partial;
}

__device__ static inline uint4 k3_mxfp4_load_group(
        const uint8_t *source) {
    if (((uintptr_t)source & 15u) == 0u) {
        return *(const uint4 *)source;
    }
    /*
     * About a quarter of the shipped expert tensors start at offset 8 mod 16.
     * Preserve their guaranteed 8-byte alignment instead of relying on an
     * undefined misaligned uint4 dereference.
     */
    const uint64_t *halves = (const uint64_t *)source;
    const uint64_t low = halves[0];
    const uint64_t high = halves[1];
    return make_uint4(
        (uint32_t)low, (uint32_t)(low >> 32u),
        (uint32_t)high, (uint32_t)(high >> 32u));
}

__global__ static void k3_mxfp4_gemv_kernel(
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
    __shared__ float reduction[K3_ROCM_MXFP4_THREADS];

    float partial = 0.0f;
    /*
     * One native group is one E8M0 scale, 16 packed bytes, and 32 BF16
     * activations. Assign a whole group to each thread so weights use one
     * 16-byte load and the scale is fetched once instead of 32 times.
     */
    const uint32_t group_count = columns / 32u;
    const uint8_t *packed_groups = packed + packed_base;
    const uint4 *input_octets = (const uint4 *)input;
    for (uint32_t group = tid;
         group < group_count;
         group += blockDim.x) {
        const uint4 group_weights =
            k3_mxfp4_load_group(packed_groups + group * sizeof(uint4));
        const uint4 input0 = input_octets[group * 4u];
        const uint4 input1 = input_octets[group * 4u + 1u];
        const uint4 input2 = input_octets[group * 4u + 2u];
        const uint4 input3 = input_octets[group * 4u + 3u];
        const float scale =
            k3_e8m0_to_float(scales[scale_base + group]);
        partial +=
            k3_mxfp4_dot8(group_weights.x, input0, scale) +
            k3_mxfp4_dot8(group_weights.y, input1, scale) +
            k3_mxfp4_dot8(group_weights.z, input2, scale) +
            k3_mxfp4_dot8(group_weights.w, input3, scale);
    }
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) output[row] = k3_float_to_bf16(reduction[0]);
}

template <uint32_t BatchTile>
__global__ static void k3_mxfp4_gemm_tiled_kernel(
        hip_bfloat16       *output,
        const uint8_t      *packed,
        const uint8_t      *scales,
        const hip_bfloat16 *input,
        uint32_t vector_count,
        uint32_t rows,
        uint32_t columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t vector0 =
        blockIdx.y * BatchTile;
    const uint32_t tid = threadIdx.x;
    const uint32_t active =
        vector_count - vector0 < BatchTile ?
            vector_count - vector0 :
            BatchTile;
    const uint64_t packed_base =
        (uint64_t)row * (columns / 2u);
    const uint64_t scale_base =
        (uint64_t)row * (columns / 32u);
    __shared__ float
        reduction[BatchTile]
                 [K3_ROCM_MXFP4_THREADS];
    float partial[BatchTile] = {0.0f};
    const uint32_t group_count = columns / 32u;
    const uint8_t *packed_groups = packed + packed_base;
    for (uint32_t group = tid;
         group < group_count;
         group += blockDim.x) {
        const uint4 group_weights =
            k3_mxfp4_load_group(
                packed_groups + group * sizeof(uint4));
        const float scale =
            k3_e8m0_to_float(
                scales[scale_base + group]);
        for (uint32_t tile = 0u;
             tile < active; tile++) {
            const uint4 *input_octets =
                (const uint4 *)(input +
                    (uint64_t)(vector0 + tile) * columns);
            const uint4 input0 =
                input_octets[group * 4u];
            const uint4 input1 =
                input_octets[group * 4u + 1u];
            const uint4 input2 =
                input_octets[group * 4u + 2u];
            const uint4 input3 =
                input_octets[group * 4u + 3u];
            partial[tile] +=
                k3_mxfp4_dot8(
                    group_weights.x, input0, scale) +
                k3_mxfp4_dot8(
                    group_weights.y, input1, scale) +
                k3_mxfp4_dot8(
                    group_weights.z, input2, scale) +
                k3_mxfp4_dot8(
                    group_weights.w, input3, scale);
        }
    }
    for (uint32_t tile = 0u; tile < active; tile++) {
        reduction[tile][tid] = partial[tile];
    }
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u;
         width > 0u; width /= 2u) {
        if (tid < width) {
            for (uint32_t tile = 0u;
                 tile < active; tile++) {
                reduction[tile][tid] +=
                    reduction[tile][tid + width];
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        for (uint32_t tile = 0u;
             tile < active; tile++) {
            output[(uint64_t)(vector0 + tile) * rows + row] =
                k3_float_to_bf16(reduction[tile][0]);
        }
    }
}

template <typename Output>
__global__ static void k3_bf16_gemv_kernel(
        Output             *output,
        const hip_bfloat16 *weights,
        const hip_bfloat16 *input,
        uint32_t columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t vector = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    output += (uint64_t)vector * gridDim.x;
    input += (uint64_t)vector * columns;
    const uint64_t weight_base = (uint64_t)row * columns;
    __shared__ float reduction[K3_ROCM_THREADS];

    float partial = 0.0f;
    if ((columns & 3u) == 0) {
        const uint64_t *weight_quads =
            (const uint64_t *)(weights + weight_base);
        const uint64_t *input_quads =
            (const uint64_t *)input;
        const uint32_t quad_count = columns / 4u;
        for (uint32_t quad = tid; quad < quad_count; quad += blockDim.x) {
            uint64_t weight_value = weight_quads[quad];
            uint64_t input_value = input_quads[quad];
            uint32_t weight_low = (uint32_t)weight_value;
            uint32_t weight_high = (uint32_t)(weight_value >> 32u);
            uint32_t input_low = (uint32_t)input_value;
            uint32_t input_high = (uint32_t)(input_value >> 32u);
            partial +=
                __uint_as_float((weight_low & UINT32_C(0x0000ffff)) << 16u) *
                __uint_as_float((input_low & UINT32_C(0x0000ffff)) << 16u) +
                __uint_as_float(weight_low & UINT32_C(0xffff0000)) *
                __uint_as_float(input_low & UINT32_C(0xffff0000)) +
                __uint_as_float((weight_high & UINT32_C(0x0000ffff)) << 16u) *
                __uint_as_float((input_high & UINT32_C(0x0000ffff)) << 16u) +
                __uint_as_float(weight_high & UINT32_C(0xffff0000)) *
                __uint_as_float(input_high & UINT32_C(0xffff0000));
        }
    } else if ((columns & 1u) == 0) {
        const uint32_t *weight_pairs =
            (const uint32_t *)(weights + weight_base);
        const uint32_t *input_pairs =
            (const uint32_t *)input;
        const uint32_t pair_count = columns / 2u;
        for (uint32_t pair = tid; pair < pair_count; pair += blockDim.x) {
            uint32_t weight_value = weight_pairs[pair];
            uint32_t input_value = input_pairs[pair];
            partial +=
                __uint_as_float((weight_value & UINT32_C(0x0000ffff)) << 16u) *
                __uint_as_float((input_value & UINT32_C(0x0000ffff)) << 16u) +
                __uint_as_float(weight_value & UINT32_C(0xffff0000)) *
                __uint_as_float(input_value & UINT32_C(0xffff0000));
        }
    } else {
        for (uint32_t column = tid; column < columns; column += blockDim.x) {
            partial += k3_bf16_to_float(weights[weight_base + column]) *
                       k3_bf16_to_float(input[column]);
        }
    }
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) output[row] = (Output)reduction[0];
}

template <>
__global__ void k3_bf16_gemv_kernel<hip_bfloat16>(
        hip_bfloat16       *output,
        const hip_bfloat16 *weights,
        const hip_bfloat16 *input,
        uint32_t columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t vector = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    output += (uint64_t)vector * gridDim.x;
    input += (uint64_t)vector * columns;
    const uint64_t weight_base = (uint64_t)row * columns;
    __shared__ float reduction[K3_ROCM_GEMV_THREADS];

    float partial = 0.0f;
    if ((columns & 7u) == 0) {
        const uint4 *weight_octets =
            (const uint4 *)(weights + weight_base);
        const uint4 *input_octets =
            (const uint4 *)input;
        const uint32_t octet_count = columns / 8u;
        for (uint32_t octet = tid;
             octet < octet_count;
             octet += blockDim.x) {
            uint4 weight_value = weight_octets[octet];
            uint4 input_value = input_octets[octet];
            partial +=
                k3_packed_bf16x2_dot(weight_value.x, input_value.x) +
                k3_packed_bf16x2_dot(weight_value.y, input_value.y) +
                k3_packed_bf16x2_dot(weight_value.z, input_value.z) +
                k3_packed_bf16x2_dot(weight_value.w, input_value.w);
        }
    } else if ((columns & 3u) == 0) {
        const uint64_t *weight_quads =
            (const uint64_t *)(weights + weight_base);
        const uint64_t *input_quads =
            (const uint64_t *)input;
        const uint32_t quad_count = columns / 4u;
        for (uint32_t quad = tid; quad < quad_count; quad += blockDim.x) {
            uint64_t weight_value = weight_quads[quad];
            uint64_t input_value = input_quads[quad];
            uint32_t weight_low = (uint32_t)weight_value;
            uint32_t weight_high = (uint32_t)(weight_value >> 32u);
            uint32_t input_low = (uint32_t)input_value;
            uint32_t input_high = (uint32_t)(input_value >> 32u);
            partial +=
                __uint_as_float((weight_low & UINT32_C(0x0000ffff)) << 16u) *
                __uint_as_float((input_low & UINT32_C(0x0000ffff)) << 16u) +
                __uint_as_float(weight_low & UINT32_C(0xffff0000)) *
                __uint_as_float(input_low & UINT32_C(0xffff0000)) +
                __uint_as_float((weight_high & UINT32_C(0x0000ffff)) << 16u) *
                __uint_as_float((input_high & UINT32_C(0x0000ffff)) << 16u) +
                __uint_as_float(weight_high & UINT32_C(0xffff0000)) *
                __uint_as_float(input_high & UINT32_C(0xffff0000));
        }
    } else if ((columns & 1u) == 0) {
        const uint32_t *weight_pairs =
            (const uint32_t *)(weights + weight_base);
        const uint32_t *input_pairs =
            (const uint32_t *)input;
        const uint32_t pair_count = columns / 2u;
        for (uint32_t pair = tid; pair < pair_count; pair += blockDim.x) {
            uint32_t weight_value = weight_pairs[pair];
            uint32_t input_value = input_pairs[pair];
            partial +=
                __uint_as_float((weight_value & UINT32_C(0x0000ffff)) << 16u) *
                __uint_as_float((input_value & UINT32_C(0x0000ffff)) << 16u) +
                __uint_as_float(weight_value & UINT32_C(0xffff0000)) *
                __uint_as_float(input_value & UINT32_C(0xffff0000));
        }
    } else {
        for (uint32_t column = tid; column < columns; column += blockDim.x) {
            partial += k3_bf16_to_float(weights[weight_base + column]) *
                       k3_bf16_to_float(input[column]);
        }
    }
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) output[row] = k3_float_to_bf16(reduction[0]);
}

template <uint32_t BatchTile, typename Output, bool UseOctets>
__global__ static void k3_bf16_gemm_tiled_kernel(
        Output             *output,
        const hip_bfloat16 *weights,
        const hip_bfloat16 *input,
        uint32_t vector_count,
        uint32_t rows,
        uint32_t columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t vector0 =
        blockIdx.y * BatchTile;
    const uint32_t tid = threadIdx.x;
    const uint32_t active =
        vector_count - vector0 < BatchTile ?
            vector_count - vector0 :
            BatchTile;
    const uint64_t weight_base =
        (uint64_t)row * columns;
    __shared__ float
        reduction[BatchTile]
                 [K3_ROCM_GEMV_THREADS];
    float partial[BatchTile] = {0.0f};
    if (UseOctets && (columns & 7u) == 0u) {
        const uint4 *weight_octets =
            (const uint4 *)(weights + weight_base);
        const uint32_t count = columns / 8u;
        for (uint32_t index = tid;
             index < count; index += blockDim.x) {
            const uint4 weight = weight_octets[index];
            for (uint32_t tile = 0u;
                 tile < active; tile++) {
                const uint4 input_value =
                    ((const uint4 *)(input +
                        (uint64_t)(vector0 + tile) *
                            columns))[index];
                partial[tile] +=
                    k3_packed_bf16x2_dot(
                        weight.x, input_value.x) +
                    k3_packed_bf16x2_dot(
                        weight.y, input_value.y) +
                    k3_packed_bf16x2_dot(
                        weight.z, input_value.z) +
                    k3_packed_bf16x2_dot(
                        weight.w, input_value.w);
            }
        }
    } else if ((columns & 3u) == 0u) {
        const uint64_t *weight_quads =
            (const uint64_t *)(weights + weight_base);
        const uint32_t count = columns / 4u;
        for (uint32_t index = tid;
             index < count; index += blockDim.x) {
            const uint64_t weight = weight_quads[index];
            const uint32_t weight_low = (uint32_t)weight;
            const uint32_t weight_high =
                (uint32_t)(weight >> 32u);
            for (uint32_t tile = 0u;
                 tile < active; tile++) {
                const uint64_t input_value =
                    ((const uint64_t *)(input +
                        (uint64_t)(vector0 + tile) *
                            columns))[index];
                const uint32_t input_low =
                    (uint32_t)input_value;
                const uint32_t input_high =
                    (uint32_t)(input_value >> 32u);
                partial[tile] +=
                    __uint_as_float(
                        (weight_low &
                         UINT32_C(0x0000ffff)) << 16u) *
                    __uint_as_float(
                        (input_low &
                         UINT32_C(0x0000ffff)) << 16u) +
                    __uint_as_float(
                        weight_low &
                        UINT32_C(0xffff0000)) *
                    __uint_as_float(
                        input_low &
                        UINT32_C(0xffff0000)) +
                    __uint_as_float(
                        (weight_high &
                         UINT32_C(0x0000ffff)) << 16u) *
                    __uint_as_float(
                        (input_high &
                         UINT32_C(0x0000ffff)) << 16u) +
                    __uint_as_float(
                        weight_high &
                        UINT32_C(0xffff0000)) *
                    __uint_as_float(
                        input_high &
                        UINT32_C(0xffff0000));
            }
        }
    } else if ((columns & 1u) == 0u) {
        const uint32_t *weight_pairs =
            (const uint32_t *)(weights + weight_base);
        const uint32_t count = columns / 2u;
        for (uint32_t index = tid;
             index < count; index += blockDim.x) {
            const uint32_t weight = weight_pairs[index];
            for (uint32_t tile = 0u;
                 tile < active; tile++) {
                const uint32_t input_value =
                    ((const uint32_t *)(input +
                        (uint64_t)(vector0 + tile) *
                            columns))[index];
                partial[tile] +=
                    k3_packed_bf16x2_dot(
                        weight, input_value);
            }
        }
    } else {
        for (uint32_t column = tid;
             column < columns; column += blockDim.x) {
            const float weight =
                k3_bf16_to_float(
                    weights[weight_base + column]);
            for (uint32_t tile = 0u;
                 tile < active; tile++) {
                partial[tile] +=
                    weight * k3_bf16_to_float(
                        input[(uint64_t)(vector0 + tile) *
                                  columns +
                              column]);
            }
        }
    }
    for (uint32_t tile = 0u; tile < active; tile++) {
        reduction[tile][tid] = partial[tile];
    }
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u;
         width > 0u; width /= 2u) {
        if (tid < width) {
            for (uint32_t tile = 0u;
                 tile < active; tile++) {
                reduction[tile][tid] +=
                    reduction[tile][tid + width];
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        for (uint32_t tile = 0u;
             tile < active; tile++) {
            output[(uint64_t)(vector0 + tile) * rows + row] =
                (Output)reduction[tile][0];
        }
    }
}

__global__ static void k3_bf16_quantize_q8_128_kernel(
        int8_t             *quantized,
        float              *scales,
        const hip_bfloat16 *weights,
        uint32_t blocks_per_row) {
    const uint32_t row = blockIdx.x;
    const uint32_t block = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint64_t block_index =
        (uint64_t)row * blocks_per_row + block;
    const uint64_t base = block_index * K3_Q8_BLOCK_SIZE;
    __shared__ float reduction[K3_Q8_BLOCK_SIZE];
    __shared__ float scale;

    const float value = k3_bf16_to_float(weights[base + tid]);
    reduction[tid] = fabsf(value);
    __syncthreads();
    for (uint32_t width = K3_Q8_BLOCK_SIZE / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) {
            reduction[tid] =
                fmaxf(reduction[tid], reduction[tid + width]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        scale = reduction[0] > 0.0f ? reduction[0] / 127.0f : 1.0f;
        scales[block_index] = scale;
    }
    __syncthreads();
    int quantized_value = __float2int_rn(value / scale);
    quantized_value = max(-127, min(127, quantized_value));
    quantized[base + tid] = (int8_t)quantized_value;
}

__global__ static void k3_q8_128_dequantize_bf16_kernel(
        hip_bfloat16 *weights,
        const int8_t *quantized,
        const float  *scales,
        uint32_t      blocks_per_row) {
    const uint64_t block_index =
        (uint64_t)blockIdx.x * blocks_per_row + blockIdx.y;
    const uint64_t index =
        block_index * K3_Q8_BLOCK_SIZE + threadIdx.x;
    weights[index] = k3_float_to_bf16(
        (float)quantized[index] * scales[block_index]);
}

__global__ static void k3_q8_128_gemv_bf16_kernel(
        hip_bfloat16       *output,
        const int8_t       *quantized,
        const float        *scales,
        const hip_bfloat16 *input,
        uint32_t columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t vector = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    output += (uint64_t)vector * gridDim.x;
    input += (uint64_t)vector * columns;
    const uint32_t octet_count = columns / 8u;
    const uint32_t blocks_per_row = columns / K3_Q8_BLOCK_SIZE;
    const uint64_t row_base = (uint64_t)row * columns;
    const uint64_t scale_base = (uint64_t)row * blocks_per_row;
    const uint64_t *weight_octets =
        (const uint64_t *)(quantized + row_base);
    const uint4 *input_octets = (const uint4 *)input;
    __shared__ float reduction[K3_ROCM_GEMV_THREADS];

    float partial = 0.0f;
    for (uint32_t octet = tid;
         octet < octet_count;
         octet += blockDim.x) {
        uint64_t packed_weights = weight_octets[octet];
        uint4 packed_input = input_octets[octet];
        const float scale =
            scales[scale_base + octet / (K3_Q8_BLOCK_SIZE / 8u)];
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const int8_t q =
                (int8_t)(packed_weights >> (i * 8u));
            partial +=
                (float)q * scale *
                k3_packed_bf16_value(packed_input, i);
        }
    }
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) output[row] = k3_float_to_bf16(reduction[0]);
}

template <uint32_t BatchTile>
__global__ static void k3_q8_128_gemm_tiled_kernel(
        hip_bfloat16       *output,
        const int8_t       *quantized,
        const float        *scales,
        const hip_bfloat16 *input,
        uint32_t vector_count,
        uint32_t rows,
        uint32_t columns) {
    const uint32_t row = blockIdx.x;
    const uint32_t vector0 =
        blockIdx.y * BatchTile;
    const uint32_t tid = threadIdx.x;
    const uint32_t active =
        vector_count - vector0 < BatchTile ?
            vector_count - vector0 :
            BatchTile;
    const uint32_t octet_count = columns / 8u;
    const uint32_t blocks_per_row =
        columns / K3_Q8_BLOCK_SIZE;
    const uint64_t row_base = (uint64_t)row * columns;
    const uint64_t scale_base =
        (uint64_t)row * blocks_per_row;
    const uint64_t *weight_octets =
        (const uint64_t *)(quantized + row_base);
    __shared__ float
        reduction[BatchTile]
                 [K3_ROCM_GEMV_THREADS];
    float partial[BatchTile] = {0.0f};
    for (uint32_t octet = tid;
         octet < octet_count;
         octet += blockDim.x) {
        const uint64_t packed_weights =
            weight_octets[octet];
        const float scale =
            scales[scale_base +
                   octet / (K3_Q8_BLOCK_SIZE / 8u)];
        for (uint32_t tile = 0u;
             tile < active; tile++) {
            const uint4 packed_input =
                ((const uint4 *)(input +
                    (uint64_t)(vector0 + tile) *
                        columns))[octet];
#pragma unroll
            for (uint32_t index = 0u; index < 8u; index++) {
                const int8_t q =
                    (int8_t)(packed_weights >>
                             (index * 8u));
                partial[tile] +=
                    (float)q * scale *
                    k3_packed_bf16_value(
                        packed_input, index);
            }
        }
    }
    for (uint32_t tile = 0u; tile < active; tile++) {
        reduction[tile][tid] = partial[tile];
    }
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u;
         width > 0u; width /= 2u) {
        if (tid < width) {
            for (uint32_t tile = 0u;
                 tile < active; tile++) {
                reduction[tile][tid] +=
                    reduction[tile][tid + width];
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        for (uint32_t tile = 0u;
             tile < active; tile++) {
            output[(uint64_t)(vector0 + tile) * rows + row] =
                k3_float_to_bf16(reduction[tile][0]);
        }
    }
}

__global__ static void k3_rms_norm_kernel(
        hip_bfloat16       *output,
        const hip_bfloat16 *input,
        const hip_bfloat16 *weight,
        uint32_t hidden_size,
        float epsilon) {
    const uint32_t vector = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)vector * hidden_size;
    __shared__ float reduction[K3_ROCM_THREADS];

    float sum_squares = 0.0f;
    for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
        float value = k3_bf16_to_float(input[base + d]);
        sum_squares += value * value;
    }
    reduction[tid] = sum_squares;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    float reciprocal_std =
        rsqrtf(reduction[0] / (float)hidden_size + epsilon);
    for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
        float value = k3_bf16_to_float(input[base + d]) *
                      reciprocal_std *
                      k3_bf16_to_float(weight[d]);
        output[base + d] = k3_float_to_bf16(value);
    }
}

__global__ static void k3_mla_pack_k_weight_kernel(
        hip_bfloat16       *packed_k,
        const hip_bfloat16 *kv_b,
        uint64_t element_count) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; index < element_count; index += stride) {
        const uint32_t nope = index % K3_MLA_NOPE_DIM;
        const uint64_t outer = index / K3_MLA_NOPE_DIM;
        const uint32_t latent = outer % K3_MLA_LATENT_DIM;
        const uint32_t head = outer / K3_MLA_LATENT_DIM;
        packed_k[index] = kv_b[
            ((uint64_t)head * K3_MLA_KV_B_HEAD_STRIDE + nope) *
            K3_MLA_LATENT_DIM + latent];
    }
}

__global__ static void k3_mla_absorb_q_kernel(
        hip_bfloat16       *absorbed_q,
        const hip_bfloat16 *q,
        const hip_bfloat16 *packed_k) {
    const uint32_t head = blockIdx.x;
    const uint32_t latent = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint64_t weight_base =
        ((uint64_t)head * K3_MLA_LATENT_DIM + latent) *
        K3_MLA_NOPE_DIM;
    const uint64_t q_base = (uint64_t)head * K3_MLA_Q_DIM;
    __shared__ float reduction[K3_MLA_NOPE_DIM];
    reduction[tid] =
        k3_bf16_to_float(packed_k[weight_base + tid]) *
        k3_bf16_to_float(q[q_base + tid]);
    __syncthreads();
    for (uint32_t width = K3_MLA_NOPE_DIM / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) {
        absorbed_q[
            (uint64_t)head * K3_MLA_CACHE_DIM + latent] =
            k3_float_to_bf16(reduction[0]);
    }
}

__global__ static void k3_mla_copy_pass_q_kernel(
        hip_bfloat16       *absorbed_q,
        const hip_bfloat16 *q,
        uint32_t head_count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count = head_count * K3_MLA_PASS_DIM;
    if (index >= count) return;
    uint32_t head = index / K3_MLA_PASS_DIM;
    uint32_t d = index % K3_MLA_PASS_DIM;
    absorbed_q[(uint64_t)head * K3_MLA_CACHE_DIM +
               K3_MLA_LATENT_DIM + d] =
        q[(uint64_t)head * K3_MLA_Q_DIM + K3_MLA_NOPE_DIM + d];
}

__global__ static void k3_mla_decompress_v_kernel(
        hip_bfloat16       *output,
        const hip_bfloat16 *latent,
        const hip_bfloat16 *kv_b) {
    const uint32_t head = blockIdx.x;
    const uint32_t value = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint64_t weight_base =
        ((uint64_t)head * K3_MLA_KV_B_HEAD_STRIDE +
         K3_MLA_NOPE_DIM + value) * K3_MLA_LATENT_DIM;
    const uint64_t latent_base =
        (uint64_t)head * K3_MLA_LATENT_DIM;
    const uint32_t *weight_pairs =
        (const uint32_t *)(kv_b + weight_base);
    const uint32_t *latent_pairs =
        (const uint32_t *)(latent + latent_base);
    __shared__ float reduction[K3_ROCM_THREADS];
    uint32_t weight_value = weight_pairs[tid];
    uint32_t latent_value = latent_pairs[tid];
    reduction[tid] =
        k3_packed_bf16x2_dot(weight_value, latent_value);
    __syncthreads();
    for (uint32_t width = K3_ROCM_THREADS / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) {
        output[(uint64_t)head * K3_MLA_V_DIM + value] =
            k3_float_to_bf16(reduction[0]);
    }
}

__global__ static void k3_mla_absorb_q_batch_kernel(
        hip_bfloat16       *absorbed_q,
        const hip_bfloat16 *q,
        const hip_bfloat16 *packed_k,
        uint32_t            head_count) {
    const uint32_t head = blockIdx.x;
    const uint32_t latent = blockIdx.y;
    const uint32_t query = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint64_t weight_base =
        ((uint64_t)head * K3_MLA_LATENT_DIM + latent) *
        K3_MLA_NOPE_DIM;
    const uint64_t q_base =
        ((uint64_t)query * head_count + head) * K3_MLA_Q_DIM;
    const uint64_t output_base =
        ((uint64_t)query * head_count + head) * K3_MLA_CACHE_DIM;
    __shared__ float reduction[K3_MLA_NOPE_DIM];
    reduction[tid] =
        k3_bf16_to_float(packed_k[weight_base + tid]) *
        k3_bf16_to_float(q[q_base + tid]);
    __syncthreads();
    for (uint32_t width = K3_MLA_NOPE_DIM / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) {
        absorbed_q[output_base + latent] =
            k3_float_to_bf16(reduction[0]);
    }
}

__global__ static void k3_mla_copy_pass_q_batch_kernel(
        hip_bfloat16       *absorbed_q,
        const hip_bfloat16 *q,
        uint32_t            head_count,
        uint64_t            count) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t d = index % K3_MLA_PASS_DIM;
    const uint64_t outer = index / K3_MLA_PASS_DIM;
    const uint32_t head = outer % head_count;
    const uint64_t query = outer / head_count;
    absorbed_q[(query * head_count + head) * K3_MLA_CACHE_DIM +
               K3_MLA_LATENT_DIM + d] =
        q[(query * head_count + head) * K3_MLA_Q_DIM +
          K3_MLA_NOPE_DIM + d];
}

__global__ static void k3_mla_decompress_v_batch_kernel(
        hip_bfloat16       *output,
        const hip_bfloat16 *latent,
        const hip_bfloat16 *kv_b,
        uint32_t            head_count) {
    const uint32_t head = blockIdx.x;
    const uint32_t value = blockIdx.y;
    const uint32_t query = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint64_t weight_base =
        ((uint64_t)head * K3_MLA_KV_B_HEAD_STRIDE +
         K3_MLA_NOPE_DIM + value) * K3_MLA_LATENT_DIM;
    const uint64_t latent_base =
        ((uint64_t)query * head_count + head) * K3_MLA_LATENT_DIM;
    const uint64_t output_base =
        ((uint64_t)query * head_count + head) * K3_MLA_V_DIM;
    const uint32_t *weight_pairs =
        (const uint32_t *)(kv_b + weight_base);
    const uint32_t *latent_pairs =
        (const uint32_t *)(latent + latent_base);
    __shared__ float reduction[K3_ROCM_THREADS];
    const uint32_t weight_value = weight_pairs[tid];
    const uint32_t latent_value = latent_pairs[tid];
    reduction[tid] =
        k3_packed_bf16x2_dot(weight_value, latent_value);
    __syncthreads();
    for (uint32_t width = K3_ROCM_THREADS / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) {
        output[output_base + value] =
            k3_float_to_bf16(reduction[0]);
    }
}

__global__ static void k3_mla_softmax_kernel(
        hip_bfloat16 *probabilities,
        const float  *scores,
        uint32_t token_count) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)head * token_count;
    __shared__ float reduction[K3_ROCM_THREADS];
    __shared__ float maximum;
    __shared__ float denominator;
    float local_maximum = -INFINITY;
    for (uint32_t token = tid;
         token < token_count;
         token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, scores[base + token]);
    }
    reduction[tid] = local_maximum;
    __syncthreads();
    for (uint32_t width = K3_ROCM_THREADS / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) {
            reduction[tid] =
                fmaxf(reduction[tid], reduction[tid + width]);
        }
        __syncthreads();
    }
    if (tid == 0) maximum = reduction[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (uint32_t token = tid;
         token < token_count;
         token += blockDim.x) {
        local_sum += expf(scores[base + token] - maximum);
    }
    reduction[tid] = local_sum;
    __syncthreads();
    for (uint32_t width = K3_ROCM_THREADS / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) denominator = reduction[0];
    __syncthreads();
    for (uint32_t token = tid;
         token < token_count;
         token += blockDim.x) {
        probabilities[base + token] = k3_float_to_bf16(
            expf(scores[base + token] - maximum) / denominator);
    }
}

__global__ static void k3_mla_causal_batch_softmax_kernel(
        hip_bfloat16 *probabilities,
        const float  *scores,
        uint32_t      head_count,
        uint32_t      first_token_count,
        uint32_t      maximum_token_count) {
    const uint32_t query = blockIdx.x / head_count;
    const uint32_t head = blockIdx.x % head_count;
    const uint32_t tid = threadIdx.x;
    const uint32_t token_count = first_token_count + query;
    const uint64_t base =
        ((uint64_t)query * head_count + head) * maximum_token_count;
    __shared__ float reduction[K3_ROCM_THREADS];
    __shared__ float maximum;
    __shared__ float denominator;
    float local_maximum = -INFINITY;
    for (uint32_t token = tid;
         token < token_count;
         token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, scores[base + token]);
    }
    reduction[tid] = local_maximum;
    __syncthreads();
    for (uint32_t width = K3_ROCM_THREADS / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) {
            reduction[tid] =
                fmaxf(reduction[tid], reduction[tid + width]);
        }
        __syncthreads();
    }
    if (tid == 0) maximum = reduction[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (uint32_t token = tid;
         token < token_count;
         token += blockDim.x) {
        local_sum += expf(scores[base + token] - maximum);
    }
    reduction[tid] = local_sum;
    __syncthreads();
    for (uint32_t width = K3_ROCM_THREADS / 2u;
         width > 0;
         width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) denominator = reduction[0];
    __syncthreads();
    for (uint32_t token = tid;
         token < maximum_token_count;
         token += blockDim.x) {
        probabilities[base + token] = token < token_count ?
            k3_float_to_bf16(
                expf(scores[base + token] - maximum) / denominator) :
            k3_float_to_bf16(0.0f);
    }
}

__device__ static inline float k3_kda_block_sum(float value,
                                                float *wave_sums,
                                                float *block_sum) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t wave = threadIdx.x / warpSize;
    for (uint32_t offset = warpSize / 2u; offset > 0; offset /= 2u) {
        value += __shfl_down(value, offset);
    }
    if (lane == 0) wave_sums[wave] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t wave_count =
            (blockDim.x + warpSize - 1u) / warpSize;
        float sum = 0.0f;
        for (uint32_t i = 0; i < wave_count; i++) sum += wave_sums[i];
        *block_sum = sum;
    }
    __syncthreads();
    return *block_sum;
}

/*
 * K3 decode recurrence, matching FLA fused_recurrent_kda with:
 *   use_qk_l2norm_in_kernel=True
 *   use_gate_in_kernel=True
 *   use_beta_sigmoid_in_kernel=True
 *   state_v_first=True
 *
 * One 128-thread block owns 32 value rows of one head. The persistent state
 * is V-first so every update and reduction reads a contiguous key row.
 */
__global__ static void k3_kda_recurrent_kernel(
        hip_bfloat16       *output,
        float              *state,
        const hip_bfloat16 *q,
        const hip_bfloat16 *k,
        const hip_bfloat16 *v,
        const hip_bfloat16 *raw_gate,
        const hip_bfloat16 *raw_beta,
        const float        *A_log,
        const float        *dt_bias,
        float lower_bound) {
    const uint32_t head = blockIdx.x;
    const uint32_t value_base = blockIdx.y * K3_KDA_VALUE_TILE;
    const uint32_t key = threadIdx.x;
    const uint64_t vector_base = (uint64_t)head * K3_KDA_HEAD_DIM;
    __shared__ float wave_sums[4];
    __shared__ float reduction;

    const float q_raw = k3_bf16_to_float(q[vector_base + key]);
    const float k_raw = k3_bf16_to_float(k[vector_base + key]);
    const float q_norm_sq =
        k3_kda_block_sum(q_raw * q_raw, wave_sums, &reduction);
    const float k_norm_sq =
        k3_kda_block_sum(k_raw * k_raw, wave_sums, &reduction);
    const float q_value =
        q_raw * rsqrtf(q_norm_sq + 1e-6f) *
        rsqrtf((float)K3_KDA_HEAD_DIM);
    const float k_value = k_raw * rsqrtf(k_norm_sq + 1e-6f);

    const float gate_input =
        k3_bf16_to_float(raw_gate[vector_base + key]) +
        dt_bias[vector_base + key];
    const float gate =
        lower_bound /
        (1.0f + expf(-expf(A_log[head]) * gate_input));
    const float decay = expf(gate);
    const float beta =
        1.0f /
        (1.0f + expf(-k3_bf16_to_float(raw_beta[head])));

#pragma unroll
    for (uint32_t offset = 0; offset < K3_KDA_VALUE_TILE; offset++) {
        const uint32_t value_index = value_base + offset;
        const uint64_t state_index =
            ((uint64_t)head * K3_KDA_HEAD_DIM + value_index) *
            K3_KDA_HEAD_DIM + key;
        float state_value = state[state_index] * decay;
        const float prediction =
            k3_kda_block_sum(state_value * k_value,
                             wave_sums, &reduction);
        const float delta =
            beta *
            (k3_bf16_to_float(v[vector_base + value_index]) - prediction);
        state_value += delta * k_value;
        state[state_index] = state_value;
        const float result =
            k3_kda_block_sum(state_value * q_value,
                             wave_sums, &reduction);
        if (key == 0) {
            output[vector_base + value_index] =
                k3_float_to_bf16(result);
        }
    }
}

__global__ static void k3_short_conv4_silu_kernel(
        hip_bfloat16       *output,
        hip_bfloat16       *cache,
        const hip_bfloat16 *input,
        const float        *weight,
        uint32_t channel_count) {
    uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; channel < channel_count; channel += stride) {
        const uint64_t base = (uint64_t)channel * 4u;
        const hip_bfloat16 current = input[channel];
        const hip_bfloat16 x0 = cache[base + 1u];
        const hip_bfloat16 x1 = cache[base + 2u];
        const hip_bfloat16 x2 = cache[base + 3u];
        cache[base] = x0;
        cache[base + 1u] = x1;
        cache[base + 2u] = x2;
        cache[base + 3u] = current;
        const float convolution =
            k3_bf16_to_float(x0) * weight[base] +
            k3_bf16_to_float(x1) * weight[base + 1u] +
            k3_bf16_to_float(x2) * weight[base + 2u] +
            k3_bf16_to_float(current) * weight[base + 3u];
        output[channel] = k3_float_to_bf16(
            convolution / (1.0f + expf(-convolution)));
    }
}

__global__ static void k3_short_conv4_silu_sequence_kernel(
        hip_bfloat16       *output,
        hip_bfloat16       *cache,
        const hip_bfloat16 *input,
        const float        *weight,
        uint32_t token_count,
        uint32_t channel_count) {
    uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; channel < channel_count; channel += stride) {
        const uint64_t base = (uint64_t)channel * 4u;
        hip_bfloat16 x0 = cache[base];
        hip_bfloat16 x1 = cache[base + 1u];
        hip_bfloat16 x2 = cache[base + 2u];
        hip_bfloat16 x3 = cache[base + 3u];
        for (uint32_t token = 0u;
             token < token_count; token++) {
            const uint64_t index =
                (uint64_t)token * channel_count + channel;
            const hip_bfloat16 current = input[index];
            const float convolution =
                k3_bf16_to_float(x1) * weight[base] +
                k3_bf16_to_float(x2) * weight[base + 1u] +
                k3_bf16_to_float(x3) * weight[base + 2u] +
                k3_bf16_to_float(current) * weight[base + 3u];
            output[index] = k3_float_to_bf16(
                convolution / (1.0f + expf(-convolution)));
            x0 = x1;
            x1 = x2;
            x2 = x3;
            x3 = current;
        }
        cache[base] = x0;
        cache[base + 1u] = x1;
        cache[base + 2u] = x2;
        cache[base + 3u] = x3;
    }
}

__global__ static void k3_rms_norm_sigmoid_gate_kernel(
        hip_bfloat16       *output,
        const hip_bfloat16 *input,
        const hip_bfloat16 *gate,
        const float        *weight,
        uint32_t hidden_size,
        float epsilon) {
    const uint32_t vector = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)vector * hidden_size;
    __shared__ float reduction[K3_ROCM_THREADS];

    float sum_squares = 0.0f;
    for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
        float value = k3_bf16_to_float(input[base + d]);
        sum_squares += value * value;
    }
    reduction[tid] = sum_squares;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    const float reciprocal_std =
        rsqrtf(reduction[0] / (float)hidden_size + epsilon);
    for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
        const float gate_value = k3_bf16_to_float(gate[base + d]);
        const float sigmoid_gate = 1.0f / (1.0f + expf(-gate_value));
        output[base + d] = k3_float_to_bf16(
            k3_bf16_to_float(input[base + d]) *
            reciprocal_std * weight[d] * sigmoid_gate);
    }
}

__global__ static void k3_router_topk_kernel(
        uint32_t    *expert_ids,
        float       *expert_weights,
        const float *logits,
        const float *correction_bias,
        uint32_t expert_count,
        uint32_t top_k,
        float scaling_factor) {
    const uint32_t vector = blockIdx.x;
    expert_ids += (uint64_t)vector * top_k;
    expert_weights += (uint64_t)vector * top_k;
    logits += (uint64_t)vector * expert_count;
    /*
     * K3 has 896 experts. Parallelize the expensive sigmoid/correction pass,
     * then retain a deterministic single-thread insertion merge so ties and
     * output order remain bit-for-bit compatible with the CPU oracle.
     */
    __shared__ float scores[1024];
    __shared__ float choices[1024];
    __shared__ uint32_t ids[1024];
    for (uint32_t slot = threadIdx.x;
         slot < 1024u;
         slot += blockDim.x) {
        if (slot < expert_count) {
            const float score =
                1.0f / (1.0f + expf(-logits[slot]));
            scores[slot] = score;
            choices[slot] = score + correction_bias[slot];
            ids[slot] = slot;
        } else {
            scores[slot] = 0.0f;
            choices[slot] = -INFINITY;
            ids[slot] = UINT32_MAX;
        }
    }
    __syncthreads();

    /*
     * Deterministic 1024-entry bitonic sort in shared memory. The ascending
     * order is (choice ascending, expert ID descending), so reading from the
     * end yields the reference order (choice descending, ID ascending).
     */
    for (uint32_t width = 2u; width <= 1024u; width <<= 1u) {
        for (uint32_t stride = width >> 1u;
             stride > 0;
             stride >>= 1u) {
            for (uint32_t left = threadIdx.x;
                 left < 1024u;
                 left += blockDim.x) {
                const uint32_t right = left ^ stride;
                if (right <= left) continue;
                const bool ascending = (left & width) == 0;
                const float left_choice = choices[left];
                const float right_choice = choices[right];
                const uint32_t left_id = ids[left];
                const uint32_t right_id = ids[right];
                const bool left_after_right =
                    left_choice > right_choice ||
                    (left_choice == right_choice && left_id < right_id);
                const bool left_before_right =
                    left_choice < right_choice ||
                    (left_choice == right_choice && left_id > right_id);
                const bool swap = ascending ?
                    left_after_right : left_before_right;
                if (swap) {
                    choices[left] = right_choice;
                    choices[right] = left_choice;
                    const float left_score = scores[left];
                    scores[left] = scores[right];
                    scores[right] = left_score;
                    ids[left] = right_id;
                    ids[right] = left_id;
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x != 0) return;
    float denominator = 1e-20f;
    for (uint32_t rank = 0; rank < top_k; rank++) {
        denominator += scores[1023u - rank];
    }
    for (uint32_t rank = 0; rank < top_k; rank++) {
        const uint32_t source = 1023u - rank;
        expert_ids[rank] = ids[source];
        expert_weights[rank] =
            scores[source] / denominator * scaling_factor;
    }
}

__global__ static void k3_weighted_sum_kernel(
        hip_bfloat16       *output,
        const hip_bfloat16 *vectors,
        const float        *weights,
        uint32_t vector_count,
        uint32_t hidden_size) {
    const uint32_t token = blockIdx.y;
    output += (uint64_t)token * hidden_size;
    vectors +=
        (uint64_t)token * vector_count * hidden_size;
    weights += (uint64_t)token * vector_count;
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; index < hidden_size; index += stride) {
        float sum = 0.0f;
        for (uint32_t vector = 0; vector < vector_count; vector++) {
            sum += weights[vector] *
                   k3_bf16_to_float(
                       vectors[(uint64_t)vector * hidden_size + index]);
        }
        output[index] = k3_float_to_bf16(sum);
    }
}

template <bool Scatter>
__global__ static void k3_copy_indexed_rows_kernel(
        hip_bfloat16       *output,
        const hip_bfloat16 *input,
        const uint32_t     *row_ids,
        uint64_t element_count,
        uint32_t row_size) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (; index < element_count; index += stride) {
        const uint32_t packed_row =
            (uint32_t)(index / row_size);
        const uint32_t column =
            (uint32_t)(index % row_size);
        const uint64_t indexed =
            (uint64_t)row_ids[packed_row] * row_size + column;
        if (Scatter) output[indexed] = input[index];
        else output[index] = input[indexed];
    }
}

__global__ static void k3_add_kernel(hip_bfloat16       *output,
                                     const hip_bfloat16 *left,
                                     const hip_bfloat16 *right,
                                     uint64_t element_count) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; index < element_count; index += stride) {
        output[index] = k3_float_to_bf16(
            k3_bf16_to_float(left[index]) +
            k3_bf16_to_float(right[index]));
    }
}

__global__ static void k3_sigmoid_mul_kernel(
        hip_bfloat16       *output,
        const hip_bfloat16 *input,
        const hip_bfloat16 *gate,
        uint64_t element_count) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; index < element_count; index += stride) {
        const float gate_value = k3_bf16_to_float(gate[index]);
        output[index] = k3_float_to_bf16(
            k3_bf16_to_float(input[index]) /
            (1.0f + expf(-gate_value)));
    }
}

__global__ static void k3_argmax_bf16_kernel(
        uint32_t           *token_id,
        float              *token_value,
        const hip_bfloat16 *logits,
        uint32_t count) {
    const uint32_t tid = threadIdx.x;
    float best_value = -INFINITY;
    uint32_t best_id = UINT32_MAX;
    for (uint32_t index = tid; index < count; index += blockDim.x) {
        const float value = k3_bf16_to_float(logits[index]);
        if (value > best_value ||
            (value == best_value && index < best_id)) {
            best_value = value;
            best_id = index;
        }
    }
    __shared__ float values[K3_ROCM_THREADS];
    __shared__ uint32_t ids[K3_ROCM_THREADS];
    values[tid] = best_value;
    ids[tid] = best_id;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0; width /= 2u) {
        if (tid < width) {
            const float other_value = values[tid + width];
            const uint32_t other_id = ids[tid + width];
            if (other_value > values[tid] ||
                (other_value == values[tid] && other_id < ids[tid])) {
                values[tid] = other_value;
                ids[tid] = other_id;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *token_id = ids[0];
        *token_value = values[0];
    }
}

extern "C" bool k3_rocm_situ_bf16(void       *output,
                                  const void *gate,
                                  const void *up,
                                  uint64_t    element_count,
                                  float       beta,
                                  float       linear_beta,
                                  void       *stream_pointer) {
    if (!output || !gate || !up || element_count == 0 ||
        beta <= 0.0f || linear_beta <= 0.0f) {
        return false;
    }
    uint64_t block_count =
        (element_count + K3_ROCM_THREADS - 1u) / K3_ROCM_THREADS;
    if (block_count > 65535u) block_count = 65535u;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_situ_kernel,
                       dim3((uint32_t)block_count),
                       dim3(K3_ROCM_THREADS),
                       0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)gate,
                       (const hip_bfloat16 *)up,
                       element_count, beta, linear_beta);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_mxfp4_gemv_bf16(void       *output,
                                        const void *packed,
                                        const void *scales,
                                        const void *input,
                                        uint32_t    rows,
                                        uint32_t    columns,
                                        void       *stream_pointer) {
    return k3_rocm_mxfp4_gemm_bf16(
        output, packed, scales, input, 1u,
        rows, columns, stream_pointer);
}

template <uint32_t BatchTile>
static bool k3_launch_mxfp4_gemm_tiled_bf16(
        void       *output,
        const void *packed,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        hipStream_t stream) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            k3_mxfp4_gemm_tiled_kernel<BatchTile>),
        dim3(rows, (vector_count + BatchTile - 1u) / BatchTile),
        dim3(K3_ROCM_MXFP4_THREADS), 0, stream,
        (hip_bfloat16 *)output,
        (const uint8_t *)packed,
        (const uint8_t *)scales,
        (const hip_bfloat16 *)input,
        vector_count, rows, columns);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_mxfp4_gemm_tiled_bf16(
        void       *output,
        const void *packed,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        uint32_t    batch_tile,
        void       *stream_pointer) {
    if (!output || !packed || !scales || !input ||
        vector_count == 0u || vector_count > 65535u ||
        rows == 0u || columns == 0u || columns % 32u != 0u) {
        return false;
    }
    const hipStream_t stream = (hipStream_t)stream_pointer;
    switch (batch_tile) {
        case 16u:
            return k3_launch_mxfp4_gemm_tiled_bf16<16u>(
                output, packed, scales, input,
                vector_count, rows, columns, stream);
        case 32u:
            return k3_launch_mxfp4_gemm_tiled_bf16<32u>(
                output, packed, scales, input,
                vector_count, rows, columns, stream);
        case 64u:
            return k3_launch_mxfp4_gemm_tiled_bf16<64u>(
                output, packed, scales, input,
                vector_count, rows, columns, stream);
        default:
            return false;
    }
}

extern "C" bool k3_rocm_mxfp4_gemm_bf16(
        void       *output,
        const void *packed,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream_pointer) {
    if (!output || !packed || !scales || !input ||
        vector_count == 0u || vector_count > 65535u ||
        rows == 0 || columns == 0 || columns % 32u != 0) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    if (vector_count == 1u) {
        hipLaunchKernelGGL(k3_mxfp4_gemv_kernel,
                           dim3(rows),
                           dim3(K3_ROCM_MXFP4_THREADS),
                           0, stream,
                           (hip_bfloat16 *)output,
                           (const uint8_t *)packed,
                           (const uint8_t *)scales,
                           (const hip_bfloat16 *)input,
                           columns);
        return hipGetLastError() == hipSuccess;
    }
    return k3_rocm_mxfp4_gemm_tiled_bf16(
        output, packed, scales, input,
        vector_count, rows, columns,
        K3_ROCM_BATCH_TILE, stream_pointer);
}

extern "C" bool k3_rocm_bf16_gemv_bf16(void       *output,
                                        const void *weights,
                                        const void *input,
                                        uint32_t    rows,
                                        uint32_t    columns,
                                        void       *stream_pointer) {
    return k3_rocm_bf16_gemm_bf16(
        output, weights, input, 1u,
        rows, columns, stream_pointer);
}

template <uint32_t BatchTile>
static bool k3_launch_bf16_gemm_tiled_bf16(
        void       *output,
        const void *weights,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        hipStream_t stream) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            k3_bf16_gemm_tiled_kernel<
                BatchTile, hip_bfloat16, true>),
        dim3(rows, (vector_count + BatchTile - 1u) / BatchTile),
        dim3(K3_ROCM_GEMV_THREADS), 0, stream,
        (hip_bfloat16 *)output,
        (const hip_bfloat16 *)weights,
        (const hip_bfloat16 *)input,
        vector_count, rows, columns);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_bf16_gemm_tiled_bf16(
        void       *output,
        const void *weights,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        uint32_t    batch_tile,
        void       *stream_pointer) {
    if (!output || !weights || !input ||
        vector_count == 0u || vector_count > 65535u ||
        rows == 0u || columns == 0u) {
        return false;
    }
    const hipStream_t stream = (hipStream_t)stream_pointer;
    switch (batch_tile) {
        case 16u:
            return k3_launch_bf16_gemm_tiled_bf16<16u>(
                output, weights, input,
                vector_count, rows, columns, stream);
        case 32u:
            return k3_launch_bf16_gemm_tiled_bf16<32u>(
                output, weights, input,
                vector_count, rows, columns, stream);
        case 64u:
            return k3_launch_bf16_gemm_tiled_bf16<64u>(
                output, weights, input,
                vector_count, rows, columns, stream);
        default:
            return false;
    }
}

extern "C" bool k3_rocm_bf16_gemm_bf16(
        void       *output,
        const void *weights,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream_pointer) {
    if (!output || !weights || !input || vector_count == 0u ||
        vector_count > 65535u || rows == 0 || columns == 0) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    if (vector_count == 1u) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(
                               k3_bf16_gemv_kernel<
                                   hip_bfloat16>),
                           dim3(rows),
                           dim3(K3_ROCM_GEMV_THREADS),
                           0, stream,
                           (hip_bfloat16 *)output,
                           (const hip_bfloat16 *)weights,
                           (const hip_bfloat16 *)input,
                           columns);
        return hipGetLastError() == hipSuccess;
    }
    return k3_rocm_bf16_gemm_tiled_bf16(
        output, weights, input, vector_count,
        rows, columns, K3_ROCM_BATCH_TILE,
        stream_pointer);
}

extern "C" bool k3_rocm_bf16_gemv_f32(void       *output,
                                       const void *weights,
                                       const void *input,
                                       uint32_t    rows,
                                       uint32_t    columns,
                                       void       *stream_pointer) {
    return k3_rocm_bf16_gemm_f32(
        output, weights, input, 1u,
        rows, columns, stream_pointer);
}

template <uint32_t BatchTile>
static bool k3_launch_bf16_gemm_tiled_f32(
        void       *output,
        const void *weights,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        hipStream_t stream) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            k3_bf16_gemm_tiled_kernel<
                BatchTile, float, false>),
        dim3(rows, (vector_count + BatchTile - 1u) / BatchTile),
        dim3(K3_ROCM_THREADS), 0, stream,
        (float *)output,
        (const hip_bfloat16 *)weights,
        (const hip_bfloat16 *)input,
        vector_count, rows, columns);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_bf16_gemm_tiled_f32(
        void       *output,
        const void *weights,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        uint32_t    batch_tile,
        void       *stream_pointer) {
    if (!output || !weights || !input ||
        vector_count == 0u || vector_count > 65535u ||
        rows == 0u || columns == 0u) {
        return false;
    }
    const hipStream_t stream = (hipStream_t)stream_pointer;
    switch (batch_tile) {
        case 16u:
            return k3_launch_bf16_gemm_tiled_f32<16u>(
                output, weights, input,
                vector_count, rows, columns, stream);
        case 32u:
            return k3_launch_bf16_gemm_tiled_f32<32u>(
                output, weights, input,
                vector_count, rows, columns, stream);
        case 64u:
            return k3_launch_bf16_gemm_tiled_f32<64u>(
                output, weights, input,
                vector_count, rows, columns, stream);
        default:
            return false;
    }
}

extern "C" bool k3_rocm_bf16_gemm_f32(
        void       *output,
        const void *weights,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream_pointer) {
    if (!output || !weights || !input || vector_count == 0u ||
        vector_count > 65535u || rows == 0 || columns == 0) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    if (vector_count == 1u) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(k3_bf16_gemv_kernel<float>),
            dim3(rows), dim3(K3_ROCM_THREADS), 0, stream,
            (float *)output,
            (const hip_bfloat16 *)weights,
                           (const hip_bfloat16 *)input,
                           columns);
        return hipGetLastError() == hipSuccess;
    }
    return k3_rocm_bf16_gemm_tiled_f32(
        output, weights, input, vector_count,
        rows, columns, K3_ROCM_BATCH_TILE,
        stream_pointer);
}

extern "C" bool k3_rocm_bf16_quantize_q8_128(
        void       *quantized,
        void       *scales,
        const void *weights,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream_pointer) {
    if (!quantized || !scales || !weights || rows == 0 ||
        columns == 0 || columns % K3_Q8_BLOCK_SIZE != 0) {
        return false;
    }
    const uint32_t blocks_per_row = columns / K3_Q8_BLOCK_SIZE;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_bf16_quantize_q8_128_kernel,
                       dim3(rows, blocks_per_row),
                       dim3(K3_Q8_BLOCK_SIZE), 0, stream,
                       (int8_t *)quantized,
                       (float *)scales,
                       (const hip_bfloat16 *)weights,
                       blocks_per_row);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_q8_128_dequantize_bf16(
        void       *weights,
        const void *quantized,
        const void *scales,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream_pointer) {
    if (!weights || !quantized || !scales || rows == 0u ||
        columns == 0u ||
        columns % K3_Q8_BLOCK_SIZE != 0u) {
        return false;
    }
    const uint32_t blocks_per_row =
        columns / K3_Q8_BLOCK_SIZE;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(
        k3_q8_128_dequantize_bf16_kernel,
        dim3(rows, blocks_per_row),
        dim3(K3_Q8_BLOCK_SIZE), 0, stream,
        (hip_bfloat16 *)weights,
        (const int8_t *)quantized,
        (const float *)scales,
        blocks_per_row);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_q8_128_gemv_bf16(
        void       *output,
        const void *quantized,
        const void *scales,
        const void *input,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream_pointer) {
    return k3_rocm_q8_128_gemm_bf16(
        output, quantized, scales, input, 1u,
        rows, columns, stream_pointer);
}

template <uint32_t BatchTile>
static bool k3_launch_q8_128_gemm_tiled_bf16(
        void       *output,
        const void *quantized,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        hipStream_t stream) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            k3_q8_128_gemm_tiled_kernel<BatchTile>),
        dim3(rows, (vector_count + BatchTile - 1u) / BatchTile),
        dim3(K3_ROCM_GEMV_THREADS), 0, stream,
        (hip_bfloat16 *)output,
        (const int8_t *)quantized,
        (const float *)scales,
        (const hip_bfloat16 *)input,
        vector_count, rows, columns);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_q8_128_gemm_tiled_bf16(
        void       *output,
        const void *quantized,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        uint32_t    batch_tile,
        void       *stream_pointer) {
    if (!output || !quantized || !scales || !input ||
        vector_count == 0u || vector_count > 65535u ||
        rows == 0u || columns == 0u ||
        columns % K3_Q8_BLOCK_SIZE != 0u) {
        return false;
    }
    const hipStream_t stream = (hipStream_t)stream_pointer;
    switch (batch_tile) {
        case 16u:
            return k3_launch_q8_128_gemm_tiled_bf16<16u>(
                output, quantized, scales, input,
                vector_count, rows, columns, stream);
        case 32u:
            return k3_launch_q8_128_gemm_tiled_bf16<32u>(
                output, quantized, scales, input,
                vector_count, rows, columns, stream);
        case 64u:
            return k3_launch_q8_128_gemm_tiled_bf16<64u>(
                output, quantized, scales, input,
                vector_count, rows, columns, stream);
        default:
            return false;
    }
}

extern "C" bool k3_rocm_q8_128_gemm_bf16(
        void       *output,
        const void *quantized,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream_pointer) {
    if (!output || !quantized || !scales || !input ||
        vector_count == 0u || vector_count > 65535u || rows == 0 ||
        columns == 0 || columns % K3_Q8_BLOCK_SIZE != 0) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    if (vector_count == 1u) {
        hipLaunchKernelGGL(
            k3_q8_128_gemv_bf16_kernel,
            dim3(rows), dim3(K3_ROCM_GEMV_THREADS),
            0, stream,
            (hip_bfloat16 *)output,
            (const int8_t *)quantized,
            (const float *)scales,
                           (const hip_bfloat16 *)input,
                           columns);
        return hipGetLastError() == hipSuccess;
    }
    return k3_rocm_q8_128_gemm_tiled_bf16(
        output, quantized, scales, input,
        vector_count, rows, columns,
        K3_ROCM_BATCH_TILE, stream_pointer);
}

extern "C" bool k3_rocm_blas_context_create(
        k3_rocm_blas_context **context) {
    if (!context) return false;
    *context = NULL;
    k3_rocm_blas_context *created =
        (k3_rocm_blas_context *)calloc(1, sizeof(*created));
    if (!created) return false;
    if (hipblasCreate(&created->handle) != HIPBLAS_STATUS_SUCCESS) {
        free(created);
        return false;
    }
    *context = created;
    return true;
}

extern "C" void k3_rocm_blas_context_destroy(
        k3_rocm_blas_context *context) {
    if (!context) return;
    if (context->handle) (void)hipblasDestroy(context->handle);
    free(context);
}

extern "C" bool k3_rocm_blas_bf16_gemv_bf16(
        k3_rocm_blas_context *context,
        void                 *output,
        const void           *weights,
        const void           *input,
        uint32_t              rows,
        uint32_t              columns,
        void                 *stream_pointer) {
    return k3_rocm_blas_bf16_gemm_bf16(
        context, output, weights, input, 1u,
        rows, columns, stream_pointer);
}

extern "C" bool k3_rocm_blas_bf16_gemm_bf16(
        k3_rocm_blas_context *context,
        void                 *output,
        const void           *weights,
        const void           *input,
        uint32_t              vector_count,
        uint32_t              rows,
        uint32_t              columns,
        void                 *stream_pointer) {
    if (!context || !context->handle || !output || !weights || !input ||
        vector_count == 0u || rows == 0u || columns == 0u ||
        vector_count > INT32_MAX ||
        rows > INT32_MAX || columns > INT32_MAX) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    if (hipblasSetStream(context->handle, stream) !=
        HIPBLAS_STATUS_SUCCESS) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    /*
     * SafeTensors weights are row-major [rows, columns]. Treat the same
     * storage as column-major [columns, rows] and transpose it.
     */
    return hipblasGemmEx(
               context->handle,
               HIPBLAS_OP_T, HIPBLAS_OP_N,
               (int)rows, (int)vector_count, (int)columns,
               &alpha,
               weights, HIP_R_16BF, (int)columns,
               input, HIP_R_16BF, (int)columns,
               &beta,
               output, HIP_R_16BF, (int)rows,
               HIPBLAS_COMPUTE_32F,
               HIPBLAS_GEMM_DEFAULT) == HIPBLAS_STATUS_SUCCESS;
}

extern "C" bool k3_rocm_blas_bf16_gemm_f32(
        k3_rocm_blas_context *context,
        void                 *output,
        const void           *weights,
        const void           *input,
        uint32_t              vector_count,
        uint32_t              rows,
        uint32_t              columns,
        void                 *stream_pointer) {
    if (!context || !context->handle || !output || !weights || !input ||
        vector_count == 0u || rows == 0u || columns == 0u ||
        vector_count > INT32_MAX ||
        rows > INT32_MAX || columns > INT32_MAX) {
        return false;
    }
    const hipStream_t stream = (hipStream_t)stream_pointer;
    if (hipblasSetStream(context->handle, stream) !=
        HIPBLAS_STATUS_SUCCESS) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return hipblasGemmEx(
               context->handle,
               HIPBLAS_OP_T, HIPBLAS_OP_N,
               (int)rows, (int)vector_count, (int)columns,
               &alpha,
               weights, HIP_R_16BF, (int)columns,
               input, HIP_R_16BF, (int)columns,
               &beta,
               output, HIP_R_32F, (int)rows,
               HIPBLAS_COMPUTE_32F,
               HIPBLAS_GEMM_DEFAULT) == HIPBLAS_STATUS_SUCCESS;
}

extern "C" bool k3_rocm_mla_pack_k_weight_bf16(
        void       *packed_k,
        const void *kv_b,
        uint32_t    head_count,
        void       *stream_pointer) {
    if (!packed_k || !kv_b || head_count == 0) return false;
    const uint64_t count =
        (uint64_t)head_count * K3_MLA_LATENT_DIM * K3_MLA_NOPE_DIM;
    uint64_t blocks =
        (count + K3_ROCM_THREADS - 1u) / K3_ROCM_THREADS;
    if (blocks > 65535u) blocks = 65535u;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_mla_pack_k_weight_kernel,
                       dim3((uint32_t)blocks), dim3(K3_ROCM_THREADS),
                       0, stream,
                       (hip_bfloat16 *)packed_k,
                       (const hip_bfloat16 *)kv_b,
                       count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_mla_absorb_q_bf16(
        void       *absorbed_q,
        const void *q,
        const void *packed_k,
        uint32_t    head_count,
        void       *stream_pointer) {
    if (!absorbed_q || !q || !packed_k || head_count == 0) return false;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_mla_absorb_q_kernel,
                       dim3(head_count, K3_MLA_LATENT_DIM),
                       dim3(K3_MLA_NOPE_DIM), 0, stream,
                       (hip_bfloat16 *)absorbed_q,
                       (const hip_bfloat16 *)q,
                       (const hip_bfloat16 *)packed_k);
    uint32_t pass_count = head_count * K3_MLA_PASS_DIM;
    uint32_t blocks =
        (pass_count + K3_ROCM_THREADS - 1u) / K3_ROCM_THREADS;
    hipLaunchKernelGGL(k3_mla_copy_pass_q_kernel,
                       dim3(blocks), dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)absorbed_q,
                       (const hip_bfloat16 *)q,
                       head_count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_mla_decompress_v_bf16(
        void       *output,
        const void *latent,
        const void *kv_b,
        uint32_t    head_count,
        void       *stream_pointer) {
    if (!output || !latent || !kv_b || head_count == 0) return false;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_mla_decompress_v_kernel,
                       dim3(head_count, K3_MLA_V_DIM),
                       dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)latent,
                       (const hip_bfloat16 *)kv_b);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_mla_absorb_q_batch_bf16(
        void       *absorbed_q,
        const void *q,
        const void *packed_k,
        uint32_t    head_count,
        uint32_t    query_count,
        void       *stream_pointer) {
    if (!absorbed_q || !q || !packed_k || head_count == 0 ||
        query_count == 0 || query_count > UINT16_MAX) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_mla_absorb_q_batch_kernel,
                       dim3(head_count, K3_MLA_LATENT_DIM, query_count),
                       dim3(K3_MLA_NOPE_DIM), 0, stream,
                       (hip_bfloat16 *)absorbed_q,
                       (const hip_bfloat16 *)q,
                       (const hip_bfloat16 *)packed_k,
                       head_count);
    if (hipGetLastError() != hipSuccess) return false;
    const uint64_t count =
        (uint64_t)query_count * head_count * K3_MLA_PASS_DIM;
    const uint64_t blocks =
        (count + K3_ROCM_THREADS - 1u) / K3_ROCM_THREADS;
    if (blocks > UINT32_MAX) return false;
    hipLaunchKernelGGL(k3_mla_copy_pass_q_batch_kernel,
                       dim3((uint32_t)blocks),
                       dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)absorbed_q,
                       (const hip_bfloat16 *)q,
                       head_count, count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_mla_decompress_v_batch_bf16(
        void       *output,
        const void *latent,
        const void *kv_b,
        uint32_t    head_count,
        uint32_t    query_count,
        void       *stream_pointer) {
    if (!output || !latent || !kv_b || head_count == 0 ||
        query_count == 0 || query_count > UINT16_MAX) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_mla_decompress_v_batch_kernel,
                       dim3(head_count, K3_MLA_V_DIM, query_count),
                       dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)latent,
                       (const hip_bfloat16 *)kv_b,
                       head_count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_blas_mla_attention_bf16(
        k3_rocm_blas_context *context,
        void                 *output,
        void                 *score_workspace,
        void                 *probability_workspace,
        const void           *absorbed_q,
        const void           *cache,
        uint32_t              head_count,
        uint32_t              token_count,
        void                 *stream_pointer) {
    if (!context || !context->handle || !output || !score_workspace ||
        !probability_workspace || !absorbed_q || !cache ||
        head_count == 0 || token_count == 0 ||
        head_count > INT32_MAX || token_count > INT32_MAX) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    if (hipblasSetStream(context->handle, stream) !=
        HIPBLAS_STATUS_SUCCESS) {
        return false;
    }
    const float score_alpha = 0.07216878364870322f; /* 1 / sqrt(192) */
    const float alpha = 1.0f;
    const float beta = 0.0f;
    hipblasStatus_t status = hipblasGemmEx(
        context->handle,
        HIPBLAS_OP_T, HIPBLAS_OP_N,
        (int)token_count, (int)head_count, K3_MLA_CACHE_DIM,
        &score_alpha,
        cache, HIP_R_16BF, K3_MLA_CACHE_DIM,
        absorbed_q, HIP_R_16BF, K3_MLA_CACHE_DIM,
        &beta,
        score_workspace, HIP_R_32F, (int)token_count,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT);
    if (status != HIPBLAS_STATUS_SUCCESS) return false;
    hipLaunchKernelGGL(k3_mla_softmax_kernel,
                       dim3(head_count), dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)probability_workspace,
                       (const float *)score_workspace,
                       token_count);
    if (hipGetLastError() != hipSuccess) return false;
    status = hipblasGemmEx(
        context->handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        K3_MLA_LATENT_DIM, (int)head_count, (int)token_count,
        &alpha,
        cache, HIP_R_16BF, K3_MLA_CACHE_DIM,
        probability_workspace, HIP_R_16BF, (int)token_count,
        &beta,
        output, HIP_R_16BF, K3_MLA_LATENT_DIM,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT);
    return status == HIPBLAS_STATUS_SUCCESS;
}

extern "C" bool k3_rocm_blas_mla_attention_batch_bf16(
        k3_rocm_blas_context *context,
        void                 *output,
        void                 *score_workspace,
        void                 *probability_workspace,
        const void           *absorbed_q,
        const void           *cache,
        uint32_t              head_count,
        uint32_t              first_token_count,
        uint32_t              query_count,
        void                 *stream_pointer) {
    if (!context || !context->handle || !output || !score_workspace ||
        !probability_workspace || !absorbed_q || !cache ||
        head_count == 0 || first_token_count == 0 || query_count == 0 ||
        query_count > UINT16_MAX ||
        first_token_count > UINT32_MAX - query_count + 1u) {
        return false;
    }
    const uint32_t maximum_token_count =
        first_token_count + query_count - 1u;
    if (head_count > INT32_MAX || maximum_token_count > INT32_MAX) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    if (hipblasSetStream(context->handle, stream) !=
        HIPBLAS_STATUS_SUCCESS) {
        return false;
    }
    const float score_alpha = 0.07216878364870322f;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStride query_stride =
        (hipblasStride)head_count * K3_MLA_CACHE_DIM;
    const hipblasStride workspace_stride =
        (hipblasStride)head_count * maximum_token_count;
    const hipblasStride output_stride =
        (hipblasStride)head_count * K3_MLA_LATENT_DIM;
    hipblasStatus_t status = hipblasGemmStridedBatchedEx(
        context->handle,
        HIPBLAS_OP_T, HIPBLAS_OP_N,
        (int)maximum_token_count, (int)head_count,
        K3_MLA_CACHE_DIM,
        &score_alpha,
        cache, HIP_R_16BF, K3_MLA_CACHE_DIM, 0,
        absorbed_q, HIP_R_16BF, K3_MLA_CACHE_DIM, query_stride,
        &beta,
        score_workspace, HIP_R_32F,
        (int)maximum_token_count, workspace_stride,
        (int)query_count,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    if (status != HIPBLAS_STATUS_SUCCESS) return false;
    hipLaunchKernelGGL(k3_mla_causal_batch_softmax_kernel,
                       dim3(query_count * head_count),
                       dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)probability_workspace,
                       (const float *)score_workspace,
                       head_count, first_token_count,
                       maximum_token_count);
    if (hipGetLastError() != hipSuccess) return false;
    status = hipblasGemmStridedBatchedEx(
        context->handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        K3_MLA_LATENT_DIM, (int)head_count,
        (int)maximum_token_count,
        &alpha,
        cache, HIP_R_16BF, K3_MLA_CACHE_DIM, 0,
        probability_workspace, HIP_R_16BF,
        (int)maximum_token_count, workspace_stride,
        &beta,
        output, HIP_R_16BF, K3_MLA_LATENT_DIM, output_stride,
        (int)query_count,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    return status == HIPBLAS_STATUS_SUCCESS;
}

extern "C" bool k3_rocm_rms_norm_bf16(void       *output,
                                      const void *input,
                                      const void *weight,
                                      uint32_t    vector_count,
                                      uint32_t    hidden_size,
                                      float       epsilon,
                                      void       *stream_pointer) {
    if (!output || !input || !weight || vector_count == 0 ||
        hidden_size == 0 || epsilon <= 0.0f) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_rms_norm_kernel,
                       dim3(vector_count), dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)input,
                       (const hip_bfloat16 *)weight,
                       hidden_size, epsilon);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_kda_recurrent_bf16_f32_state(
        void       *output,
        void       *state,
        const void *q,
        const void *k,
        const void *v,
        const void *raw_gate,
        const void *raw_beta,
        const void *A_log,
        const void *dt_bias,
        uint32_t    head_count,
        uint32_t    head_dim,
        float       lower_bound,
        void       *stream_pointer) {
    if (!output || !state || !q || !k || !v || !raw_gate || !raw_beta ||
        !A_log || !dt_bias || head_count == 0 ||
        head_dim != K3_KDA_HEAD_DIM || lower_bound >= 0.0f) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_kda_recurrent_kernel,
                       dim3(head_count,
                            K3_KDA_HEAD_DIM / K3_KDA_VALUE_TILE),
                       dim3(K3_KDA_HEAD_DIM), 0, stream,
                       (hip_bfloat16 *)output,
                       (float *)state,
                       (const hip_bfloat16 *)q,
                       (const hip_bfloat16 *)k,
                       (const hip_bfloat16 *)v,
                       (const hip_bfloat16 *)raw_gate,
                       (const hip_bfloat16 *)raw_beta,
                       (const float *)A_log,
                       (const float *)dt_bias,
                       lower_bound);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_short_conv4_silu_bf16_f32_weight(
        void       *output,
        void       *cache,
        const void *input,
        const void *weight,
        uint32_t    channel_count,
        void       *stream_pointer) {
    if (!output || !cache || !input || !weight || channel_count == 0) {
        return false;
    }
    uint32_t blocks =
        (channel_count + K3_ROCM_THREADS - 1u) / K3_ROCM_THREADS;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_short_conv4_silu_kernel,
                       dim3(blocks), dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)output,
                       (hip_bfloat16 *)cache,
                       (const hip_bfloat16 *)input,
                       (const float *)weight,
                       channel_count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_short_conv4_silu_sequence_bf16_f32_weight(
        void       *output,
        void       *cache,
        const void *input,
        const void *weight,
        uint32_t    token_count,
        uint32_t    channel_count,
        void       *stream_pointer) {
    if (!output || !cache || !input || !weight ||
        token_count == 0u || channel_count == 0u) {
        return false;
    }
    uint32_t blocks =
        (channel_count + K3_ROCM_THREADS - 1u) /
        K3_ROCM_THREADS;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(
        k3_short_conv4_silu_sequence_kernel,
        dim3(blocks), dim3(K3_ROCM_THREADS), 0, stream,
        (hip_bfloat16 *)output,
        (hip_bfloat16 *)cache,
        (const hip_bfloat16 *)input,
        (const float *)weight,
        token_count, channel_count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_rms_norm_sigmoid_gate_bf16_f32_weight(
        void       *output,
        const void *input,
        const void *gate,
        const void *weight,
        uint32_t    vector_count,
        uint32_t    hidden_size,
        float       epsilon,
        void       *stream_pointer) {
    if (!output || !input || !gate || !weight || vector_count == 0 ||
        hidden_size == 0 || hidden_size > K3_ROCM_THREADS ||
        epsilon <= 0.0f) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_rms_norm_sigmoid_gate_kernel,
                       dim3(vector_count), dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)input,
                       (const hip_bfloat16 *)gate,
                       (const float *)weight,
                       hidden_size, epsilon);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_router_topk_f32(void       *expert_ids,
                                        void       *expert_weights,
                                        const void *logits,
                                        const void *correction_bias,
                                        uint32_t    expert_count,
                                        uint32_t    top_k,
                                        float       scaling_factor,
                                        void       *stream_pointer) {
    return k3_rocm_router_topk_f32_batch(
        expert_ids, expert_weights, logits, correction_bias,
        1u, expert_count, top_k, scaling_factor, stream_pointer);
}

extern "C" bool k3_rocm_router_topk_f32_batch(
        void       *expert_ids,
        void       *expert_weights,
        const void *logits,
        const void *correction_bias,
        uint32_t    vector_count,
        uint32_t    expert_count,
        uint32_t    top_k,
        float       scaling_factor,
        void       *stream_pointer) {
    if (!expert_ids || !expert_weights || !logits || !correction_bias ||
        vector_count == 0u || vector_count > 65535u ||
        expert_count == 0 || top_k == 0 || top_k > expert_count ||
        expert_count > 1024u || top_k > 32u || scaling_factor <= 0.0f) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_router_topk_kernel,
                       dim3(vector_count),
                       dim3(K3_ROCM_THREADS), 0, stream,
                       (uint32_t *)expert_ids,
                       (float *)expert_weights,
                       (const float *)logits,
                       (const float *)correction_bias,
                       expert_count, top_k, scaling_factor);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_weighted_sum_bf16(void       *output,
                                          const void *vectors,
                                          const void *weights,
                                          uint32_t    vector_count,
                                          uint32_t    hidden_size,
                                          void       *stream_pointer) {
    return k3_rocm_weighted_sum_bf16_batch(
        output, vectors, weights, 1u,
        vector_count, hidden_size, stream_pointer);
}

extern "C" bool k3_rocm_weighted_sum_bf16_batch(
        void       *output,
        const void *vectors,
        const void *weights,
        uint32_t    token_count,
        uint32_t    vector_count,
        uint32_t    hidden_size,
        void       *stream_pointer) {
    if (!output || !vectors || !weights ||
        token_count == 0u || token_count > 65535u ||
        vector_count == 0 || hidden_size == 0) {
        return false;
    }
    uint32_t blocks =
        (hidden_size + K3_ROCM_THREADS - 1u) / K3_ROCM_THREADS;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_weighted_sum_kernel,
                       dim3(blocks, token_count),
                       dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)vectors,
                       (const float *)weights,
                       vector_count, hidden_size);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_gather_rows_bf16(
        void       *output,
        const void *input,
        const void *row_ids,
        uint32_t    row_count,
        uint32_t    row_size,
        void       *stream_pointer) {
    if (!output || !input || !row_ids ||
        row_count == 0u || row_size == 0u) {
        return false;
    }
    const uint64_t elements =
        (uint64_t)row_count * row_size;
    uint64_t blocks =
        (elements + K3_ROCM_THREADS - 1u) /
        K3_ROCM_THREADS;
    if (blocks > 65535u) blocks = 65535u;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(k3_copy_indexed_rows_kernel<false>),
        dim3((uint32_t)blocks), dim3(K3_ROCM_THREADS),
        0, stream,
        (hip_bfloat16 *)output,
        (const hip_bfloat16 *)input,
        (const uint32_t *)row_ids,
        elements, row_size);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_scatter_rows_bf16(
        void       *output,
        const void *input,
        const void *row_ids,
        uint32_t    row_count,
        uint32_t    row_size,
        void       *stream_pointer) {
    if (!output || !input || !row_ids ||
        row_count == 0u || row_size == 0u) {
        return false;
    }
    const uint64_t elements =
        (uint64_t)row_count * row_size;
    uint64_t blocks =
        (elements + K3_ROCM_THREADS - 1u) /
        K3_ROCM_THREADS;
    if (blocks > 65535u) blocks = 65535u;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(k3_copy_indexed_rows_kernel<true>),
        dim3((uint32_t)blocks), dim3(K3_ROCM_THREADS),
        0, stream,
        (hip_bfloat16 *)output,
        (const hip_bfloat16 *)input,
        (const uint32_t *)row_ids,
        elements, row_size);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_add_bf16(void       *output,
                                  const void *left,
                                  const void *right,
                                  uint64_t    element_count,
                                  void       *stream_pointer) {
    if (!output || !left || !right || element_count == 0) return false;
    uint64_t block_count =
        (element_count + K3_ROCM_THREADS - 1u) / K3_ROCM_THREADS;
    if (block_count > 65535u) block_count = 65535u;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_add_kernel,
                       dim3((uint32_t)block_count),
                       dim3(K3_ROCM_THREADS), 0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)left,
                       (const hip_bfloat16 *)right,
                       element_count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_sigmoid_mul_bf16(
        void       *output,
        const void *input,
        const void *gate,
        uint64_t    element_count,
        void       *stream_pointer) {
    if (!output || !input || !gate || element_count == 0) return false;
    uint64_t blocks =
        (element_count + K3_ROCM_THREADS - 1u) / K3_ROCM_THREADS;
    if (blocks > 65535u) blocks = 65535u;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_sigmoid_mul_kernel,
                       dim3((uint32_t)blocks), dim3(K3_ROCM_THREADS),
                       0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)input,
                       (const hip_bfloat16 *)gate,
                       element_count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_argmax_bf16(
        void       *token_id,
        void       *token_value,
        const void *logits,
        uint32_t    count,
        void       *stream_pointer) {
    if (!token_id || !token_value || !logits || count == 0) return false;
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_argmax_bf16_kernel,
                       dim3(1), dim3(K3_ROCM_THREADS), 0, stream,
                       (uint32_t *)token_id,
                       (float *)token_value,
                       (const hip_bfloat16 *)logits,
                       count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k3_rocm_attn_res_bf16(void       *output,
                                      const void *prefix,
                                      const void *blocks,
                                      const void *norm_weight,
                                      const void *qk_weight,
                                      uint32_t    token_count,
                                      uint32_t    block_capacity,
                                      uint32_t    num_blocks,
                                      uint32_t    hidden_size,
                                      float       epsilon,
                                      void       *stream_pointer) {
    if (!output || !prefix || !blocks || !norm_weight || !qk_weight ||
        token_count == 0 || hidden_size == 0 ||
        num_blocks == 0 || num_blocks > block_capacity ||
        num_blocks + 1u > K3_ATTN_RES_MAX_SOURCES || epsilon <= 0.0f) {
        return false;
    }
    hipStream_t stream = (hipStream_t)stream_pointer;
    hipLaunchKernelGGL(k3_attn_res_kernel,
                       dim3(token_count),
                       dim3(K3_ROCM_THREADS),
                       0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)prefix,
                       (const hip_bfloat16 *)blocks,
                       (const hip_bfloat16 *)norm_weight,
                       (const hip_bfloat16 *)qk_weight,
                       block_capacity, num_blocks, hidden_size, epsilon);
    return hipGetLastError() == hipSuccess;
}
