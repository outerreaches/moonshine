#include "glm53_kda_aux_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
constexpr unsigned kThreads = 256u;
constexpr unsigned kHeadDim = GLM53_KDA_AUX_HEAD_DIM;

struct Buffer { const void *pointer; size_t count; };

static bool checked_mul(size_t a, size_t b, size_t *result) {
    if (a != 0u && b > SIZE_MAX / a) return false;
    *result = a * b;
    return true;
}

static bool disjoint_bf16(const Buffer *buffers, size_t count) {
    if (buffers == nullptr || count == 0u || count > 5u) return false;
    uintptr_t first[5], last[5];
    for (size_t i = 0; i < count; ++i) {
        if (buffers[i].pointer == nullptr || buffers[i].count == 0u ||
            buffers[i].count > SIZE_MAX / sizeof(hip_bfloat16)) return false;
        const size_t bytes = buffers[i].count * sizeof(hip_bfloat16);
        first[i] = reinterpret_cast<uintptr_t>(buffers[i].pointer);
        if (first[i] > UINTPTR_MAX - bytes) return false;
        last[i] = first[i] + bytes;
    }
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i + 1u; j < count; ++j)
            if (first[i] < last[j] && first[j] < last[i]) return false;
    return true;
}

static bool grid_for(size_t elements, unsigned *grid) {
    if (elements == 0u || elements > SIZE_MAX - (kThreads - 1u)) return false;
    const size_t blocks = (elements + kThreads - 1u) / kThreads;
    if (blocks == 0u || blocks > UINT32_MAX) return false;
    *grid = static_cast<unsigned>(blocks);
    return true;
}

__device__ __forceinline__ float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__global__ void conv4_silu_kernel(hip_bfloat16 *output,
                                  hip_bfloat16 *dst_cache,
                                  const hip_bfloat16 *input,
                                  const hip_bfloat16 *src_cache,
                                  const hip_bfloat16 *weight,
                                  size_t channels) {
    const size_t channel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const size_t base = channel * 4u;
    const hip_bfloat16 x0 = src_cache[base + 1u];
    const hip_bfloat16 x1 = src_cache[base + 2u];
    const hip_bfloat16 x2 = src_cache[base + 3u];
    const hip_bfloat16 x3 = input[channel];
    dst_cache[base] = x0;
    dst_cache[base + 1u] = x1;
    dst_cache[base + 2u] = x2;
    dst_cache[base + 3u] = x3;
    float convolution = static_cast<float>(x0) * static_cast<float>(weight[base]);
    convolution += static_cast<float>(x1) * static_cast<float>(weight[base + 1u]);
    convolution += static_cast<float>(x2) * static_cast<float>(weight[base + 2u]);
    convolution += static_cast<float>(x3) * static_cast<float>(weight[base + 3u]);
    const hip_bfloat16 staged(convolution);
    const float rounded = static_cast<float>(staged);
    output[channel] = hip_bfloat16(rounded * stable_sigmoid(rounded));
}

__global__ void rmsnorm_gated_kernel(hip_bfloat16 *output,
                                     const hip_bfloat16 *input,
                                     const hip_bfloat16 *gate,
                                     const hip_bfloat16 *weight,
                                     float eps) {
    const unsigned d = threadIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.x) * kHeadDim;
    __shared__ float reduction[kHeadDim];
    const float x = static_cast<float>(input[base + d]);
    reduction[d] = x * x;
    __syncthreads();
    for (unsigned width = kHeadDim / 2u; width != 0u; width >>= 1u) {
        if (d < width) reduction[d] += reduction[d + width];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(reduction[0] / static_cast<float>(kHeadDim) + eps);
    const float g = static_cast<float>(gate[base + d]);
    output[base + d] = hip_bfloat16(
        x * inv_rms * static_cast<float>(weight[d]) * stable_sigmoid(g));
}

static void clear_status() { (void)hipGetLastError(); }
static bool launch_ok() { return hipGetLastError() == hipSuccess; }
} // namespace

extern "C" bool glm53_kda_conv4_silu_bf16(
    void *output, size_t output_count, void *dst_cache, size_t dst_cache_count,
    const void *input, size_t input_count, const void *src_cache,
    size_t src_cache_count, const void *weight, size_t weight_count,
    size_t channels, void *stream) {
    size_t cache_elements;
    if (channels == 0u || !checked_mul(channels, 4u, &cache_elements) ||
        output_count < channels || dst_cache_count < cache_elements ||
        input_count < channels || src_cache_count < cache_elements ||
        weight_count < cache_elements) return false;
    const Buffer buffers[] = {{output, output_count}, {dst_cache, dst_cache_count},
        {input, input_count}, {src_cache, src_cache_count}, {weight, weight_count}};
    unsigned grid;
    if (!disjoint_bf16(buffers, 5u) || !grid_for(channels, &grid)) return false;
    clear_status();
    hipLaunchKernelGGL(conv4_silu_kernel, dim3(grid), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream), static_cast<hip_bfloat16 *>(output),
        static_cast<hip_bfloat16 *>(dst_cache),
        static_cast<const hip_bfloat16 *>(input),
        static_cast<const hip_bfloat16 *>(src_cache),
        static_cast<const hip_bfloat16 *>(weight), channels);
    return launch_ok();
}

extern "C" bool glm53_kda_rmsnorm_gated_bf16(
    void *output, size_t output_count, const void *input, size_t input_count,
    const void *gate, size_t gate_count, const void *weight, size_t weight_count,
    size_t heads, float eps, void *stream) {
    size_t elements;
    if (heads == 0u || !checked_mul(heads, kHeadDim, &elements) ||
        output_count < elements || input_count < elements || gate_count < elements ||
        weight_count < kHeadDim || !std::isfinite(eps) || !(eps > 0.0f) ||
        heads > UINT32_MAX) return false;
    const Buffer buffers[] = {{output, output_count}, {input, input_count},
        {gate, gate_count}, {weight, weight_count}};
    if (!disjoint_bf16(buffers, 4u)) return false;
    clear_status();
    hipLaunchKernelGGL(rmsnorm_gated_kernel, dim3(static_cast<unsigned>(heads)),
        dim3(kHeadDim), 0, static_cast<hipStream_t>(stream),
        static_cast<hip_bfloat16 *>(output),
        static_cast<const hip_bfloat16 *>(input),
        static_cast<const hip_bfloat16 *>(gate),
        static_cast<const hip_bfloat16 *>(weight), eps);
    return launch_ok();
}
