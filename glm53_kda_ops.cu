#include "glm53_kda_ops.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
constexpr size_t kDim = GLM53_KDA_HEAD_DIM;
constexpr float kEpsilon = 1.0e-6f;

struct Buffer {
    const void *pointer;
    size_t count;
};

static bool checked_mul(size_t left, size_t right, size_t *result) {
    if (left != 0u && right > SIZE_MAX / left) return false;
    *result = left * right;
    return true;
}

static bool pairwise_disjoint(const Buffer *buffers, size_t count) {
    if (buffers == nullptr || count == 0u || count > 8u) return false;
    uintptr_t begin[8], end[8];
    for (size_t i = 0u; i < count; ++i) {
        if (buffers[i].pointer == nullptr || buffers[i].count == 0u ||
            buffers[i].count > SIZE_MAX / sizeof(float)) return false;
        const size_t bytes = buffers[i].count * sizeof(float);
        begin[i] = reinterpret_cast<uintptr_t>(buffers[i].pointer);
        if (begin[i] > UINTPTR_MAX - bytes) return false;
        end[i] = begin[i] + bytes;
    }
    for (size_t i = 0u; i < count; ++i)
        for (size_t j = i + 1u; j < count; ++j)
            if (begin[i] < end[j] && begin[j] < end[i]) return false;
    return true;
}

/* A single lane owns a head.  This is the correctness core: all reductions
 * have the fixture's left-to-right binary32 order, and heads are independent.
 * State layout is [head,key,value]. */
__global__ void recurrent_kernel(float *output, float *destination,
                                 const float *q, const float *k,
                                 const float *v, const float *g,
                                 const float *beta, const float *source,
                                 size_t heads) {
    const size_t head = static_cast<size_t>(blockIdx.x);
    if (head >= heads || threadIdx.x != 0u) return;
    const size_t vector_base = head * kDim;
    const size_t state_base = head * kDim * kDim;

    float qss = 0.0f;
    float kss = 0.0f;
    for (size_t key = 0u; key < kDim; ++key) {
        const float qvalue = q[vector_base + key];
        const float kvalue = k[vector_base + key];
        qss += qvalue * qvalue;
        kss += kvalue * kvalue;
    }
    const float qden = sqrtf(qss + kEpsilon);
    const float kden = sqrtf(kss + kEpsilon);
    const float qdim = sqrtf(static_cast<float>(kDim));

    for (size_t key = 0u; key < kDim; ++key) {
        const float decay = expf(g[vector_base + key]);
        const size_t row = state_base + key * kDim;
        for (size_t value = 0u; value < kDim; ++value)
            destination[row + value] = source[row + value] * decay;
    }

    for (size_t value = 0u; value < kDim; ++value) {
        float prediction = 0.0f;
        for (size_t key = 0u; key < kDim; ++key) {
            const float kn = k[vector_base + key] / kden;
            prediction += destination[state_base + key * kDim + value] * kn;
        }
        const float delta =
            (v[vector_base + value] - prediction) * beta[head];
        for (size_t key = 0u; key < kDim; ++key) {
            const float kn = k[vector_base + key] / kden;
            const size_t index = state_base + key * kDim + value;
            destination[index] += kn * delta;
        }
        float result = 0.0f;
        for (size_t key = 0u; key < kDim; ++key) {
            const float qn = (q[vector_base + key] / qden) / qdim;
            result += qn * destination[state_base + key * kDim + value];
        }
        output[vector_base + value] = result;
    }
}

static void clear_status() { (void)hipGetLastError(); }
static bool launch_ok() { return hipGetLastError() == hipSuccess; }
} // namespace

extern "C" bool glm53_kda_recurrent_f32(
    void *output, size_t output_count,
    void *destination_state, size_t destination_state_count,
    const void *q, size_t q_count,
    const void *k, size_t k_count,
    const void *v, size_t v_count,
    const void *g, size_t g_count,
    const void *beta, size_t beta_count,
    const void *source_state, size_t source_state_count,
    size_t heads, void *stream) {
    size_t vector_count, state_count;
    if (heads == 0u || !checked_mul(heads, kDim, &vector_count) ||
        !checked_mul(vector_count, kDim, &state_count) ||
        output_count < vector_count || destination_state_count < state_count ||
        q_count < vector_count || k_count < vector_count ||
        v_count < vector_count || g_count < vector_count ||
        beta_count < heads || source_state_count < state_count ||
        heads > UINT32_MAX) return false;
    const Buffer buffers[] = {
        {output, output_count}, {destination_state, destination_state_count},
        {q, q_count}, {k, k_count}, {v, v_count}, {g, g_count},
        {beta, beta_count}, {source_state, source_state_count}};
    if (!pairwise_disjoint(buffers, 8u)) return false;

    clear_status();
    hipLaunchKernelGGL(recurrent_kernel, dim3(static_cast<unsigned>(heads)),
        dim3(1), 0, static_cast<hipStream_t>(stream),
        static_cast<float *>(output), static_cast<float *>(destination_state),
        static_cast<const float *>(q), static_cast<const float *>(k),
        static_cast<const float *>(v), static_cast<const float *>(g),
        static_cast<const float *>(beta),
        static_cast<const float *>(source_state), heads);
    return launch_ok();
}
