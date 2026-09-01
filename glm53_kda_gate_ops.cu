#include "glm53_kda_gate_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
constexpr unsigned kThreads = 256u;

struct Buffer {
    const void *pointer;
    size_t count;
    size_t element_size;
};

static bool pairwise_disjoint(const Buffer *buffers, size_t count) {
    if (buffers == nullptr || count == 0u || count > 4u) return false;
    uintptr_t begin[4], end[4];
    for (size_t i = 0; i < count; ++i) {
        const Buffer &b = buffers[i];
        if (b.pointer == nullptr || b.count == 0u || b.element_size == 0u ||
            b.count > SIZE_MAX / b.element_size) return false;
        const size_t bytes = b.count * b.element_size;
        begin[i] = reinterpret_cast<uintptr_t>(b.pointer);
        if (begin[i] > UINTPTR_MAX - bytes) return false;
        end[i] = begin[i] + bytes;
    }
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i + 1u; j < count; ++j)
            if (begin[i] < end[j] && begin[j] < end[i]) return false;
    return true;
}

static bool grid_for(size_t count, unsigned *grid) {
    if (count == 0u || count > SIZE_MAX - (kThreads - 1u)) return false;
    const size_t blocks = (count + kThreads - 1u) / kThreads;
    if (blocks == 0u || blocks > UINT32_MAX) return false;
    *grid = static_cast<unsigned>(blocks);
    return true;
}

__device__ __forceinline__ float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__global__ void prepare_forget_kernel(float *output,
                                      const hip_bfloat16 *raw_forget,
                                      const float *dt_bias,
                                      const float *A_log,
                                      size_t elements) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const size_t head = i / GLM53_KDA_GATE_HEAD_DIM;
    const float argument = expf(A_log[head]) *
        (static_cast<float>(raw_forget[i]) + dt_bias[i]);
    output[i] = GLM53_KDA_GATE_OFFICIAL_LOWER_BOUND * stable_sigmoid(argument);
}

__global__ void prepare_forget_generic_kernel(float *output,
                                              const hip_bfloat16 *raw_forget,
                                              const float *dt_bias,
                                              const float *A_log,
                                              size_t elements,
                                              float lower_bound) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const size_t head = i / GLM53_KDA_GATE_HEAD_DIM;
    const float argument = expf(A_log[head]) *
        (static_cast<float>(raw_forget[i]) + dt_bias[i]);
    output[i] = lower_bound * stable_sigmoid(argument);
}

__global__ void prepare_beta_kernel(float *output,
                                    const hip_bfloat16 *raw_beta,
                                    size_t heads) {
    const size_t h = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    const hip_bfloat16 rounded(stable_sigmoid(static_cast<float>(raw_beta[h])));
    output[h] = static_cast<float>(rounded);
}

static void clear_status() { (void)hipGetLastError(); }
static bool launch_ok() { return hipGetLastError() == hipSuccess; }
} // namespace

extern "C" bool glm53_kda_prepare_forget_f32(
    void *output, size_t output_count,
    const void *raw_forget, size_t raw_forget_count,
    const void *dt_bias, size_t dt_bias_count,
    const void *A_log, size_t A_log_count,
    size_t heads, size_t head_dim, float lower_bound, void *stream) {
    if (heads == 0u || head_dim != GLM53_KDA_GATE_HEAD_DIM ||
        heads > SIZE_MAX / head_dim || !std::isfinite(lower_bound) ||
        !(lower_bound < 0.0f)) return false;
    const size_t elements = heads * head_dim;
    if (output_count < elements || raw_forget_count < elements ||
        dt_bias_count < elements || A_log_count < heads) return false;
    const Buffer buffers[] = {
        {output, output_count, sizeof(float)},
        {raw_forget, raw_forget_count, sizeof(hip_bfloat16)},
        {dt_bias, dt_bias_count, sizeof(float)},
        {A_log, A_log_count, sizeof(float)}};
    unsigned grid;
    if (!pairwise_disjoint(buffers, 4u) || !grid_for(elements, &grid)) return false;
    clear_status();
    if (lower_bound == GLM53_KDA_GATE_OFFICIAL_LOWER_BOUND) {
        hipLaunchKernelGGL(prepare_forget_kernel, dim3(grid), dim3(kThreads), 0,
            static_cast<hipStream_t>(stream), static_cast<float *>(output),
            static_cast<const hip_bfloat16 *>(raw_forget),
            static_cast<const float *>(dt_bias), static_cast<const float *>(A_log),
            elements);
    } else {
        hipLaunchKernelGGL(prepare_forget_generic_kernel, dim3(grid), dim3(kThreads), 0,
            static_cast<hipStream_t>(stream), static_cast<float *>(output),
            static_cast<const hip_bfloat16 *>(raw_forget),
            static_cast<const float *>(dt_bias), static_cast<const float *>(A_log),
            elements, lower_bound);
    }
    return launch_ok();
}

extern "C" bool glm53_kda_prepare_beta_f32(
    void *output, size_t output_count,
    const void *raw_beta, size_t raw_beta_count,
    size_t heads, void *stream) {
    if (heads == 0u || output_count < heads || raw_beta_count < heads) return false;
    const Buffer buffers[] = {
        {output, output_count, sizeof(float)},
        {raw_beta, raw_beta_count, sizeof(hip_bfloat16)}};
    unsigned grid;
    if (!pairwise_disjoint(buffers, 2u) || !grid_for(heads, &grid)) return false;
    clear_status();
    hipLaunchKernelGGL(prepare_beta_kernel, dim3(grid), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream), static_cast<float *>(output),
        static_cast<const hip_bfloat16 *>(raw_beta), heads);
    return launch_ok();
}
