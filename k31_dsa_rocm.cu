#include "k31_dsa_rocm.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#include <limits.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr uint32_t kThreads = 256;
constexpr size_t kAlignment = 256;

__host__ __device__ inline float to_float(hip_bfloat16 value) {
    return static_cast<float>(value);
}

__host__ __device__ inline hip_bfloat16 to_bf16(float value) {
    return hip_bfloat16(value);
}

bool add_aligned(size_t *cursor, size_t bytes, size_t *offset) {
    if (*cursor > SIZE_MAX - (kAlignment - 1u)) return false;
    *cursor = (*cursor + kAlignment - 1u) & ~(kAlignment - 1u);
    if (*cursor > SIZE_MAX - bytes) return false;
    *offset = *cursor;
    *cursor += bytes;
    return true;
}

bool topk_layout(uint32_t token_count,
                 void *stream_pointer,
                 size_t *sorted_scores_offset,
                 size_t *sorted_ids_offset,
                 size_t *initial_ids_offset,
                 size_t *sort_offset,
                 size_t *total_bytes) {
    if (token_count == 0u || token_count > INT_MAX) return false;
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    size_t sort_bytes = 0;
    hipError_t status = hipcub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_bytes,
        static_cast<const float *>(nullptr),
        static_cast<float *>(nullptr),
        static_cast<const uint32_t *>(nullptr),
        static_cast<uint32_t *>(nullptr),
        static_cast<int>(token_count), 0, 32, stream);
    if (status != hipSuccess) return false;
    size_t cursor = 0;
    if (!add_aligned(&cursor, (size_t)token_count * sizeof(float),
                     sorted_scores_offset) ||
        !add_aligned(&cursor, (size_t)token_count * sizeof(uint32_t),
                     sorted_ids_offset) ||
        !add_aligned(&cursor, (size_t)token_count * sizeof(uint32_t),
                     initial_ids_offset) ||
        !add_aligned(&cursor, sort_bytes, sort_offset)) {
        return false;
    }
    *total_bytes = cursor;
    return true;
}

__global__ void layer_norm_kernel(
        hip_bfloat16 *output,
        const hip_bfloat16 *input,
        const hip_bfloat16 *gamma,
        const hip_bfloat16 *beta,
        uint32_t head_dim,
        float epsilon) {
    const uint32_t vector = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)vector * head_dim;
    __shared__ float reduction[kThreads];
    float partial = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        partial += to_float(input[base + d]);
    }
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    const float mean = reduction[0] / (float)head_dim;
    partial = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        const float delta = to_float(input[base + d]) - mean;
        partial += delta * delta;
    }
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    const float inverse_stddev = rsqrtf(reduction[0] / (float)head_dim + epsilon);
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        output[base + d] = to_bf16(
            (to_float(input[base + d]) - mean) * inverse_stddev *
                to_float(gamma[d]) + to_float(beta[d]));
    }
}

__global__ void index_scores_kernel(
        float *scores,
        const hip_bfloat16 *queries,
        const hip_bfloat16 *keys,
        const hip_bfloat16 *head_weights,
        const float *query_scales,
        const float *key_scales,
        uint32_t index_heads,
        uint32_t index_head_dim) {
    const uint32_t key = blockIdx.x;
    const uint32_t query = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const hip_bfloat16 *key_row = keys + (uint64_t)key * index_head_dim;
    __shared__ float reduction[kThreads];
    __shared__ float score;
    if (tid == 0u) score = 0.0f;
    __syncthreads();
    for (uint32_t head = 0; head < index_heads; head++) {
        const hip_bfloat16 *query_row = queries +
            ((uint64_t)query * index_heads + head) * index_head_dim;
        float partial = 0.0f;
        for (uint32_t d = tid; d < index_head_dim; d += blockDim.x) {
            partial += to_float(query_row[d]) * to_float(key_row[d]);
        }
        reduction[tid] = partial;
        __syncthreads();
        for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
            if (tid < width) reduction[tid] += reduction[tid + width];
            __syncthreads();
        }
        if (tid == 0u) {
            const float query_scale = query_scales ?
                query_scales[(uint64_t)query * index_heads + head] : 1.0f;
            const float weight = to_float(head_weights[
                (uint64_t)query * index_heads + head]);
            score += weight * query_scale * fmaxf(reduction[0], 0.0f);
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const float key_scale = key_scales ? key_scales[key] : 1.0f;
        scores[(uint64_t)query * gridDim.x + key] =
            score * key_scale * rsqrtf((float)index_heads * index_head_dim);
    }
}

__global__ void initial_ids_kernel(uint32_t *ids, uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) ids[index] = index;
}

__global__ void index_reduce_kernel(
        float *scores,
        const float *dots,
        const hip_bfloat16 *head_weights,
        const float *query_scales,
        const float *key_scales,
        uint32_t index_heads,
        uint32_t index_head_dim) {
    const uint32_t key = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    float partial = 0.0f;
    for (uint32_t head = tid; head < index_heads; head += blockDim.x) {
        const float scale = query_scales ? query_scales[head] : 1.0f;
        partial += to_float(head_weights[head]) * scale *
            fmaxf(dots[(uint64_t)head * gridDim.x + key], 0.0f);
    }
    __shared__ float reduction[kThreads];
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0u) {
        const float key_scale = key_scales ? key_scales[key] : 1.0f;
        scores[key] = reduction[0] * key_scale *
            rsqrtf((float)index_heads * index_head_dim);
    }
}

__global__ void mask_padding_kernel(
        uint32_t *selected_ids,
        const float *sorted_scores,
        uint32_t selected_count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < selected_count && sorted_scores[index] == -INFINITY) {
        selected_ids[index] = UINT32_MAX;
    }
}

__global__ void sparse_scores_kernel(
        float *scores,
        const hip_bfloat16 *query,
        const hip_bfloat16 *cache,
        const uint32_t *selected_ids,
        uint32_t selected_count,
        uint32_t cache_dim,
        float attention_scale) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const hip_bfloat16 *head_query = query + (uint64_t)head * cache_dim;
    __shared__ float reduction[kThreads];
    for (uint32_t selected = 0; selected < selected_count; selected++) {
        const uint32_t token = selected_ids ? selected_ids[selected] : selected;
        if (token == UINT32_MAX) {
            if (tid == 0u) {
                scores[(uint64_t)head * selected_count + selected] = -INFINITY;
            }
            __syncthreads();
            continue;
        }
        const hip_bfloat16 *row = cache + (uint64_t)token * cache_dim;
        float partial = 0.0f;
        for (uint32_t d = tid; d < cache_dim; d += blockDim.x) {
            partial += to_float(head_query[d]) * to_float(row[d]);
        }
        reduction[tid] = partial;
        __syncthreads();
        for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
            if (tid < width) reduction[tid] += reduction[tid + width];
            __syncthreads();
        }
        if (tid == 0u) {
            scores[(uint64_t)head * selected_count + selected] =
                reduction[0] * attention_scale;
        }
        __syncthreads();
    }
}

__global__ void sparse_softmax_kernel(
        hip_bfloat16 *probabilities,
        const float *scores,
        uint32_t selected_count) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)head * selected_count;
    __shared__ float reduction[kThreads];
    __shared__ float maximum;
    __shared__ float denominator;
    float local_maximum = -INFINITY;
    for (uint32_t selected = tid; selected < selected_count;
         selected += blockDim.x) {
        local_maximum = fmaxf(local_maximum, scores[base + selected]);
    }
    reduction[tid] = local_maximum;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
        if (tid < width) reduction[tid] = fmaxf(
            reduction[tid], reduction[tid + width]);
        __syncthreads();
    }
    if (tid == 0u) maximum = reduction[0];
    __syncthreads();
    if (maximum == -INFINITY) {
        for (uint32_t selected = tid; selected < selected_count;
             selected += blockDim.x) {
            probabilities[base + selected] = to_bf16(0.0f);
        }
        return;
    }
    float local_sum = 0.0f;
    for (uint32_t selected = tid; selected < selected_count;
         selected += blockDim.x) {
        local_sum += expf(scores[base + selected] - maximum);
    }
    reduction[tid] = local_sum;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0u) denominator = reduction[0];
    __syncthreads();
    for (uint32_t selected = tid; selected < selected_count;
         selected += blockDim.x) {
        probabilities[base + selected] = to_bf16(
            expf(scores[base + selected] - maximum) / denominator);
    }
}

__global__ void sparse_values_kernel(
        hip_bfloat16 *output,
        const hip_bfloat16 *probabilities,
        const hip_bfloat16 *cache,
        const uint32_t *selected_ids,
        uint32_t selected_count,
        uint32_t cache_dim,
        uint32_t latent_dim) {
    const uint32_t latent = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (latent >= latent_dim) return;
    float partial = 0.0f;
    for (uint32_t selected = tid; selected < selected_count;
         selected += blockDim.x) {
        const uint32_t token = selected_ids ? selected_ids[selected] : selected;
        if (token == UINT32_MAX) continue;
        partial += to_float(probabilities[
                       (uint64_t)head * selected_count + selected]) *
                   to_float(cache[(uint64_t)token * cache_dim + latent]);
    }
    __shared__ float reduction[kThreads];
    reduction[tid] = partial;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0u) {
        output[(uint64_t)head * latent_dim + latent] = to_bf16(reduction[0]);
    }
}

}  // namespace

extern "C" bool k31_dsa_layer_norm_bf16(
        void *output,
        const void *input,
        const void *gamma,
        const void *beta,
        uint32_t vector_count,
        uint32_t head_dim,
        float epsilon,
        void *stream_pointer) {
    if (head_dim == 0u || !isfinite(epsilon) || epsilon < 0.0f) return false;
    if (vector_count == 0u) return true;
    if (!output || !input || !gamma || !beta || vector_count > 65535u) {
        return false;
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    hipLaunchKernelGGL(layer_norm_kernel, dim3(vector_count), dim3(kThreads),
                       0, stream,
                       static_cast<hip_bfloat16 *>(output),
                       static_cast<const hip_bfloat16 *>(input),
                       static_cast<const hip_bfloat16 *>(gamma),
                       static_cast<const hip_bfloat16 *>(beta),
                       head_dim, epsilon);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k31_dsa_index_scores_f32(
        void *scores,
        const void *queries,
        const void *keys,
        const void *head_weights,
        const void *query_scales,
        const void *key_scales,
        uint32_t query_count,
        uint32_t key_count,
        uint32_t index_heads,
        uint32_t index_head_dim,
        void *stream_pointer) {
    if (index_heads == 0u || index_head_dim == 0u) return false;
    if (query_count == 0u || key_count == 0u) return true;
    if (!scores || !queries || !keys || !head_weights ||
        query_count > 65535u) {
        return false;
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    hipLaunchKernelGGL(index_scores_kernel, dim3(key_count, query_count),
                       dim3(kThreads), 0, stream,
                       static_cast<float *>(scores),
                       static_cast<const hip_bfloat16 *>(queries),
                       static_cast<const hip_bfloat16 *>(keys),
                       static_cast<const hip_bfloat16 *>(head_weights),
                       static_cast<const float *>(query_scales),
                       static_cast<const float *>(key_scales),
                       index_heads, index_head_dim);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k31_dsa_index_reduce_f32(
        void *scores,
        const void *head_dot_products,
        const void *head_weights,
        const void *query_scales,
        const void *key_scales,
        uint32_t key_count,
        uint32_t index_heads,
        uint32_t index_head_dim,
        void *stream_pointer) {
    if (key_count == 0u) return true;
    if (!scores || !head_dot_products || !head_weights ||
        index_heads == 0u || index_head_dim == 0u) return false;
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    hipLaunchKernelGGL(index_reduce_kernel, dim3(key_count), dim3(kThreads),
                       0, stream, static_cast<float *>(scores),
                       static_cast<const float *>(head_dot_products),
                       static_cast<const hip_bfloat16 *>(head_weights),
                       static_cast<const float *>(query_scales),
                       static_cast<const float *>(key_scales),
                       index_heads, index_head_dim);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k31_dsa_topk_workspace_bytes(
        uint32_t token_count,
        void *stream,
        size_t *workspace_bytes) {
    if (!workspace_bytes) return false;
    *workspace_bytes = 0;
    size_t sorted_scores, sorted_ids, initial_ids, sort;
    return topk_layout(token_count, stream, &sorted_scores, &sorted_ids,
                       &initial_ids, &sort, workspace_bytes);
}

extern "C" bool k31_dsa_topk_f32(
        void *selected_ids,
        const void *scores,
        uint32_t token_count,
        uint32_t top_k,
        void *workspace,
        size_t workspace_bytes,
        void *stream_pointer) {
    if (!selected_ids || !scores || top_k == 0u) return false;
    size_t sorted_scores_offset, sorted_ids_offset, initial_ids_offset;
    size_t sort_offset, required;
    if (!topk_layout(token_count, stream_pointer, &sorted_scores_offset,
                     &sorted_ids_offset, &initial_ids_offset, &sort_offset,
                     &required) ||
        !workspace || workspace_bytes < required) {
        return false;
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    uint8_t *bytes = static_cast<uint8_t *>(workspace);
    float *sorted_scores = reinterpret_cast<float *>(bytes + sorted_scores_offset);
    uint32_t *sorted_ids = reinterpret_cast<uint32_t *>(bytes + sorted_ids_offset);
    uint32_t *initial_ids = reinterpret_cast<uint32_t *>(bytes + initial_ids_offset);
    void *sort_workspace = bytes + sort_offset;
    size_t sort_bytes = required - sort_offset;
    hipLaunchKernelGGL(initial_ids_kernel,
                       dim3((token_count + kThreads - 1u) / kThreads),
                       dim3(kThreads), 0, stream, initial_ids, token_count);
    if (hipGetLastError() != hipSuccess) return false;
    const size_t selected_bytes = (size_t)top_k * sizeof(uint32_t);
    if (hipMemsetAsync(selected_ids, 0xff, selected_bytes, stream) != hipSuccess) {
        return false;
    }
    if (hipcub::DeviceRadixSort::SortPairsDescending(
            sort_workspace, sort_bytes,
            static_cast<const float *>(scores), sorted_scores,
            initial_ids, sorted_ids, static_cast<int>(token_count),
            0, 32, stream) != hipSuccess) {
        return false;
    }
    const uint32_t selected_count =
        token_count < top_k ? token_count : top_k;
    if (hipMemcpyAsync(selected_ids, sorted_ids,
                       (size_t)selected_count * sizeof(uint32_t),
                       hipMemcpyDeviceToDevice, stream) != hipSuccess) {
        return false;
    }
    hipLaunchKernelGGL(mask_padding_kernel,
                       dim3((selected_count + kThreads - 1u) / kThreads),
                       dim3(kThreads), 0, stream,
                       static_cast<uint32_t *>(selected_ids), sorted_scores,
                       selected_count);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool k31_dsa_sparse_attention_bf16(
        void *latent_output,
        void *score_workspace,
        void *probability_workspace,
        const void *absorbed_q,
        const void *cache,
        const void *selected_ids,
        uint32_t selected_count,
        uint32_t attention_heads,
        uint32_t cache_dim,
        uint32_t latent_dim,
        float attention_scale,
        void *stream_pointer) {
    if (!latent_output || attention_heads == 0u || cache_dim == 0u ||
        latent_dim == 0u || latent_dim > cache_dim ||
        attention_heads > 65535u || latent_dim > 65535u ||
        !isfinite(attention_scale) || attention_scale <= 0.0f) {
        return false;
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    if (selected_count == 0u) {
        const uint64_t bytes = (uint64_t)attention_heads * latent_dim *
                               sizeof(hip_bfloat16);
        return bytes <= SIZE_MAX &&
               hipMemsetAsync(latent_output, 0, (size_t)bytes, stream) ==
                   hipSuccess;
    }
    if (!score_workspace || !probability_workspace || !absorbed_q || !cache) {
        return false;
    }
    hipLaunchKernelGGL(sparse_scores_kernel, dim3(attention_heads),
                       dim3(kThreads), 0, stream,
                       static_cast<float *>(score_workspace),
                       static_cast<const hip_bfloat16 *>(absorbed_q),
                       static_cast<const hip_bfloat16 *>(cache),
                       static_cast<const uint32_t *>(selected_ids),
                       selected_count, cache_dim, attention_scale);
    if (hipGetLastError() != hipSuccess) return false;
    hipLaunchKernelGGL(sparse_softmax_kernel, dim3(attention_heads),
                       dim3(kThreads), 0, stream,
                       static_cast<hip_bfloat16 *>(probability_workspace),
                       static_cast<const float *>(score_workspace),
                       selected_count);
    if (hipGetLastError() != hipSuccess) return false;
    hipLaunchKernelGGL(sparse_values_kernel,
                       dim3(latent_dim, attention_heads), dim3(kThreads),
                       0, stream,
                       static_cast<hip_bfloat16 *>(latent_output),
                       static_cast<const hip_bfloat16 *>(probability_workspace),
                       static_cast<const hip_bfloat16 *>(cache),
                       static_cast<const uint32_t *>(selected_ids),
                       selected_count, cache_dim, latent_dim);
    return hipGetLastError() == hipSuccess;
}
