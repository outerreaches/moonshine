#include "mimo26_rocm_ops.h"

#include <hip/hip_runtime.h>

#define MIMO26_ROCM_THREADS 256u
#define MIMO26_ROCM_MAX_EXPERTS 1024u

/*
 * BF16 conversion is written out rather than taken from hip_bfloat16 so the
 * rounding is visibly round-to-nearest-even and identical to the CPU path's
 * mimo26_f32_to_bf16. The contracts this file exists to honour are rounding
 * contracts; hiding the rounding behind an operator would defeat the point.
 */
__device__ static inline float mimo26_bf16_to_f32_d(uint16_t bits)
{
    const uint32_t widened = (uint32_t)bits << 16;
    return __uint_as_float(widened);
}

__device__ static inline uint16_t mimo26_f32_to_bf16_d(float value)
{
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        return (uint16_t)((bits >> 16) | 0x0040u);   /* quiet NaN */
    }
    const uint32_t rounding = ((bits >> 16) & 1u) + 0x7FFFu;
    return (uint16_t)((bits + rounding) >> 16);
}

__global__ static void mimo26_rmsnorm_kernel(uint16_t *output,
                                             const uint16_t *input,
                                             const uint16_t *weight,
                                             uint32_t hidden_size,
                                             float epsilon)
{
    const uint32_t vector = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)vector * hidden_size;
    __shared__ float reduction[MIMO26_ROCM_THREADS];

    float sum_squares = 0.0f;
    for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
        const float value = mimo26_bf16_to_f32_d(input[base + d]);
        sum_squares += value * value;
    }
    reduction[tid] = sum_squares;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
        if (tid < width) {
            reduction[tid] += reduction[tid + width];
        }
        __syncthreads();
    }
    const float reciprocal =
        rsqrtf(reduction[0] / (float)hidden_size + epsilon);

    for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
        const float scaled = mimo26_bf16_to_f32_d(input[base + d]) * reciprocal;
        /* The rounding that distinguishes this from K3's kernel: the
         * normalized value becomes BF16 before it meets the weight. */
        const float rounded = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(scaled));
        const float product = rounded * mimo26_bf16_to_f32_d(weight[d]);
        output[base + d] = mimo26_f32_to_bf16_d(product);
    }
}

__global__ static void mimo26_silu_product_kernel(uint16_t *output,
                                                  const uint16_t *gate,
                                                  const uint16_t *up,
                                                  uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float g = mimo26_bf16_to_f32_d(gate[index]);
    /* expf, not __expf: the fast intrinsic is a different function and this
     * module exists to match the CPU path's arithmetic exactly. */
    const float activated = g / (1.0f + expf(-g));
    /* silu rounds to BF16, then the product rounds again. */
    const float rounded =
        mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(activated));
    output[index] =
        mimo26_f32_to_bf16_d(rounded * mimo26_bf16_to_f32_d(up[index]));
}

__global__ static void mimo26_residual_add_kernel(uint16_t *output,
                                                  const uint16_t *residual,
                                                  const uint16_t *delta,
                                                  uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = mimo26_f32_to_bf16_d(
        mimo26_bf16_to_f32_d(residual[index]) +
        mimo26_bf16_to_f32_d(delta[index]));
}

__global__ static void mimo26_router_topk_kernel(uint32_t *expert_ids,
                                                 float *expert_weights,
                                                 const float *logits,
                                                 const float *correction_bias,
                                                 uint32_t expert_count,
                                                 uint32_t top_k, float scale)
{
    const uint32_t vector = blockIdx.x;
    expert_ids += (uint64_t)vector * top_k;
    expert_weights += (uint64_t)vector * top_k;
    logits += (uint64_t)vector * expert_count;

    __shared__ float scores[MIMO26_ROCM_MAX_EXPERTS];
    __shared__ float choices[MIMO26_ROCM_MAX_EXPERTS];
    __shared__ uint32_t ids[MIMO26_ROCM_MAX_EXPERTS];

    for (uint32_t slot = threadIdx.x; slot < MIMO26_ROCM_MAX_EXPERTS;
         slot += blockDim.x) {
        if (slot < expert_count) {
            const float score = 1.0f / (1.0f + expf(-logits[slot]));
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

    /* Deterministic bitonic sort, ascending by (choice, then id descending),
     * so the tail holds the selection in choice-descending, id-ascending
     * order and a tie resolves to the lower id. */
    for (uint32_t width = 2u; width <= MIMO26_ROCM_MAX_EXPERTS;
         width <<= 1u) {
        for (uint32_t stride = width >> 1u; stride > 0u; stride >>= 1u) {
            for (uint32_t left = threadIdx.x; left < MIMO26_ROCM_MAX_EXPERTS;
                 left += blockDim.x) {
                const uint32_t right = left ^ stride;
                if (right <= left) {
                    continue;
                }
                const bool ascending = (left & width) == 0u;
                const float left_choice = choices[left];
                const float right_choice = choices[right];
                const uint32_t left_id = ids[left];
                const uint32_t right_id = ids[right];
                const bool left_after =
                    left_choice > right_choice ||
                    (left_choice == right_choice && left_id < right_id);
                const bool left_before =
                    left_choice < right_choice ||
                    (left_choice == right_choice && left_id > right_id);
                if (ascending ? left_after : left_before) {
                    choices[left] = right_choice;
                    choices[right] = left_choice;
                    const float swap_score = scores[left];
                    scores[left] = scores[right];
                    scores[right] = swap_score;
                    ids[left] = right_id;
                    ids[right] = left_id;
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x != 0u) {
        return;
    }
    /*
     * MiMo declares ascending expert id for the returned indices AND for the
     * order the denominator accumulates in. Sorting the selection by id
     * before summing is the whole difference from K3's kernel, and it is a
     * real one: float addition is not associative, so summing in
     * score-descending order gives a denominator that can differ in the last
     * ulp, and that difference reaches every mixing weight.
     */
    uint32_t chosen[64];
    float chosen_score[64];
    for (uint32_t rank = 0; rank < top_k; rank++) {
        const uint32_t source = MIMO26_ROCM_MAX_EXPERTS - 1u - rank;
        uint32_t id = ids[source];
        float score = scores[source];
        uint32_t slot = rank;
        while (slot > 0u && chosen[slot - 1u] > id) {
            chosen[slot] = chosen[slot - 1u];
            chosen_score[slot] = chosen_score[slot - 1u];
            slot--;
        }
        chosen[slot] = id;
        chosen_score[slot] = score;
    }

    float denominator = 1e-20f;
    for (uint32_t rank = 0; rank < top_k; rank++) {
        denominator += chosen_score[rank];
    }
    for (uint32_t rank = 0; rank < top_k; rank++) {
        expert_ids[rank] = chosen[rank];
        expert_weights[rank] = chosen_score[rank] / denominator * scale;
    }
}

__global__ static void mimo26_expert_accumulate_kernel(
        float *accumulator, const uint16_t *expert_output, float weight,
        uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    accumulator[index] += weight * mimo26_bf16_to_f32_d(expert_output[index]);
}

__global__ static void mimo26_expert_finalize_kernel(uint16_t *output,
                                                     const float *accumulator,
                                                     uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = mimo26_f32_to_bf16_d(accumulator[index]);
}

__global__ static void mimo26_zero_kernel(float *accumulator, uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        accumulator[index] = 0.0f;
    }
}

static inline uint32_t blocks_for(uint64_t count)
{
    return (uint32_t)((count + MIMO26_ROCM_THREADS - 1u) /
                      MIMO26_ROCM_THREADS);
}

extern "C" {

bool mimo26_rocm_rmsnorm_bf16(void *output, const void *input,
                              const void *weight, uint32_t vectors,
                              uint32_t hidden_size, float epsilon,
                              void *stream)
{
    if (output == NULL || input == NULL || weight == NULL || vectors == 0u ||
        hidden_size == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_rmsnorm_kernel, dim3(vectors),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)output, (const uint16_t *)input,
                       (const uint16_t *)weight, hidden_size, epsilon);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_silu_product_bf16(void *output, const void *gate,
                                   const void *up, uint64_t count,
                                   void *stream)
{
    if (output == NULL || gate == NULL || up == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_silu_product_kernel, dim3(blocks_for(count)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)output, (const uint16_t *)gate,
                       (const uint16_t *)up, count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_residual_add_bf16(void *output, const void *residual,
                                   const void *delta, uint64_t count,
                                   void *stream)
{
    if (output == NULL || residual == NULL || delta == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_residual_add_kernel, dim3(blocks_for(count)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)output, (const uint16_t *)residual,
                       (const uint16_t *)delta, count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_router_topk(uint32_t *expert_ids, float *expert_weights,
                             const float *logits, const float *correction_bias,
                             uint32_t vectors, uint32_t expert_count,
                             uint32_t top_k, float scale, void *stream)
{
    if (expert_ids == NULL || expert_weights == NULL || logits == NULL ||
        correction_bias == NULL || vectors == 0u || top_k == 0u ||
        top_k > 64u || expert_count == 0u ||
        expert_count > MIMO26_ROCM_MAX_EXPERTS) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_router_topk_kernel, dim3(vectors),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       expert_ids, expert_weights, logits, correction_bias,
                       expert_count, top_k, scale);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_expert_accumulate_f32(float *accumulator,
                                       const void *expert_output, float weight,
                                       uint64_t count, void *stream)
{
    if (accumulator == NULL || expert_output == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_expert_accumulate_kernel,
                       dim3(blocks_for(count)), dim3(MIMO26_ROCM_THREADS), 0,
                       (hipStream_t)stream, accumulator,
                       (const uint16_t *)expert_output, weight, count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_expert_finalize_bf16(void *output, const float *accumulator,
                                      uint64_t count, void *stream)
{
    if (output == NULL || accumulator == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_expert_finalize_kernel, dim3(blocks_for(count)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)output, accumulator, count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_zero_f32(float *accumulator, uint64_t count, void *stream)
{
    if (accumulator == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_zero_kernel, dim3(blocks_for(count)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       accumulator, count);
    return hipGetLastError() == hipSuccess;
}

}  /* extern "C" */
