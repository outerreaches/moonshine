#include "glm53_mhc_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
constexpr unsigned kThreads = 256u;
constexpr float kRmsEps = 1.0e-5f;
constexpr float kHcEps = 1.0e-6f;

struct Buffer { const void *p; size_t count; size_t element; };

static bool checked_mul(size_t a, size_t b, size_t *out) {
    if (a != 0u && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool disjoint(const Buffer *b, size_t n) {
    if (b == nullptr || n == 0u || n > 9u) return false;
    uintptr_t first[9], last[9];
    for (size_t i = 0; i < n; ++i) {
        if (b[i].p == nullptr || b[i].count == 0u || b[i].element == 0u ||
            b[i].count > SIZE_MAX / b[i].element) return false;
        const size_t bytes = b[i].count * b[i].element;
        first[i] = reinterpret_cast<uintptr_t>(b[i].p);
        if (first[i] > UINTPTR_MAX - bytes) return false;
        last[i] = first[i] + bytes;
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1u; j < n; ++j)
            if (first[i] < last[j] && first[j] < last[i]) return false;
    return true;
}

static bool grid_for(size_t n, unsigned *grid) {
    if (n == 0u || n > SIZE_MAX - (kThreads - 1u)) return false;
    const size_t q = (n + kThreads - 1u) / kThreads;
    if (q == 0u || q > UINT32_MAX) return false;
    *grid = static_cast<unsigned>(q);
    return true;
}

__device__ __forceinline__ float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__global__ void replicate_kernel(hip_bfloat16 *out,
                                 const hip_bfloat16 *in, size_t width) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < 4u * width) out[i] = in[i % width];
}

/* One thread is deliberate. It fixes the device binary32 reduction order for
 * the flat RMS, all 24 matvec rows, Sinkhorn reductions, and collapse. Device
 * rsqrt/exp rounding can differ from a host serial oracle; official gates use
 * narrow numerical tolerances and at most one adjacent BF16 code at a final
 * rounding boundary. */
__global__ void prepare_kernel(float *mix, float *pre, hip_bfloat16 *post,
                               hip_bfloat16 *comb, hip_bfloat16 *collapsed,
                               const hip_bfloat16 *streams,
                               const hip_bfloat16 *fn, const float *base,
                               const float *scale, size_t width) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    const size_t flat = 4u * width;
    float squares = 0.0f;
    for (size_t i = 0; i < flat; ++i) {
        const float x = static_cast<float>(streams[i]);
        squares += x * x;
    }
    const float inv_rms = rsqrtf(squares / static_cast<float>(flat) + kRmsEps);
    for (size_t row = 0; row < 24u; ++row) {
        float sum = 0.0f;
        const size_t row_base = row * flat;
        for (size_t i = 0; i < flat; ++i)
            sum += static_cast<float>(fn[row_base + i]) *
                   (static_cast<float>(streams[i]) * inv_rms);
        mix[row] = sum;
    }

    float p[4], q[4], c[16];
    for (size_t i = 0; i < 4u; ++i) {
        p[i] = sigmoid(mix[i] * scale[0] + base[i]) + kHcEps;
        q[i] = 2.0f * sigmoid(mix[4u + i] * scale[1] + base[4u + i]);
        pre[i] = p[i];
        post[i] = hip_bfloat16(q[i]);
    }
    for (size_t src = 0; src < 4u; ++src) {
        float maximum = -INFINITY;
        for (size_t dst = 0; dst < 4u; ++dst) {
            const size_t i = 8u + src * 4u + dst;
            c[src * 4u + dst] = mix[i] * scale[2] + base[i];
            if (c[src * 4u + dst] > maximum) maximum = c[src * 4u + dst];
        }
        float denom = 0.0f;
        for (size_t dst = 0; dst < 4u; ++dst) {
            const size_t i = src * 4u + dst;
            c[i] = expf(c[i] - maximum);
            denom += c[i];
        }
        for (size_t dst = 0; dst < 4u; ++dst)
            c[src * 4u + dst] = c[src * 4u + dst] / denom + kHcEps;
    }
    for (size_t dst = 0; dst < 4u; ++dst) {
        float denom = kHcEps;
        for (size_t src = 0; src < 4u; ++src) denom += c[src * 4u + dst];
        for (size_t src = 0; src < 4u; ++src) c[src * 4u + dst] /= denom;
    }
    for (size_t iteration = 1u; iteration < 20u; ++iteration) {
        for (size_t src = 0; src < 4u; ++src) {
            float denom = kHcEps;
            for (size_t dst = 0; dst < 4u; ++dst) denom += c[src * 4u + dst];
            for (size_t dst = 0; dst < 4u; ++dst) c[src * 4u + dst] /= denom;
        }
        for (size_t dst = 0; dst < 4u; ++dst) {
            float denom = kHcEps;
            for (size_t src = 0; src < 4u; ++src) denom += c[src * 4u + dst];
            for (size_t src = 0; src < 4u; ++src) c[src * 4u + dst] /= denom;
        }
    }
    for (size_t i = 0; i < 16u; ++i) comb[i] = hip_bfloat16(c[i]);
    for (size_t d = 0; d < width; ++d) {
        float sum = 0.0f;
        for (size_t src = 0; src < 4u; ++src)
            sum += p[src] * static_cast<float>(streams[src * width + d]);
        collapsed[d] = hip_bfloat16(sum);
    }
}

__global__ void expand_kernel(hip_bfloat16 *out,
                              const hip_bfloat16 *branch,
                              const hip_bfloat16 *residual,
                              const hip_bfloat16 *post,
                              const hip_bfloat16 *comb, size_t width) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 4u * width) return;
    const size_t dst = i / width, d = i - dst * width;
    const hip_bfloat16 branch_term = hip_bfloat16(
        static_cast<float>(post[dst]) * static_cast<float>(branch[d]));
    float residual_sum = 0.0f;
    for (size_t src = 0; src < 4u; ++src)
        residual_sum += static_cast<float>(comb[src * 4u + dst]) *
                        static_cast<float>(residual[src * width + d]);
    const hip_bfloat16 residual_term = hip_bfloat16(residual_sum);
    out[i] = hip_bfloat16(static_cast<float>(branch_term) +
                          static_cast<float>(residual_term));
}

__global__ void mean_kernel(hip_bfloat16 *out,
                            const hip_bfloat16 *streams, size_t width) {
    const size_t d = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d >= width) return;
    float sum = 0.0f;
    for (size_t src = 0; src < 4u; ++src)
        sum += static_cast<float>(streams[src * width + d]);
    out[d] = hip_bfloat16(sum * 0.25f);
}

static bool launch_ok() { return hipGetLastError() == hipSuccess; }
static void clear_status() { (void)hipGetLastError(); }
} // namespace

extern "C" bool glm53_mhc_replicate_bf16(
    void *out, size_t oc, const void *in, size_t ic, size_t width, void *stream) {
    size_t flat;
    if (width == 0u || !checked_mul(4u, width, &flat) || oc < flat || ic < width)
        return false;
    const Buffer b[] = {{out, oc, sizeof(hip_bfloat16)},
                        {in, ic, sizeof(hip_bfloat16)}};
    unsigned grid;
    if (!disjoint(b, 2u) || !grid_for(flat, &grid)) return false;
    clear_status();
    hipLaunchKernelGGL(replicate_kernel, dim3(grid), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream), static_cast<hip_bfloat16 *>(out),
        static_cast<const hip_bfloat16 *>(in), width);
    return launch_ok();
}

extern "C" bool glm53_mhc_prepare_bf16(
    void *mix, size_t mixc, void *pre, size_t prec, void *post, size_t postc,
    void *comb, size_t combc, void *collapsed, size_t collc,
    const void *streams, size_t streamc, const void *fn, size_t fnc,
    const void *base, size_t basec, const void *scale, size_t scalec,
    size_t width, void *stream) {
    size_t flat, fn_need;
    if (width == 0u || !checked_mul(4u, width, &flat) ||
        !checked_mul(24u, flat, &fn_need) || mixc < 24u || prec < 4u ||
        postc < 4u || combc < 16u || collc < width || streamc < flat ||
        fnc < fn_need || basec < 24u || scalec < 3u) return false;
    const Buffer b[] = {
        {mix,mixc,sizeof(float)}, {pre,prec,sizeof(float)},
        {post,postc,sizeof(hip_bfloat16)}, {comb,combc,sizeof(hip_bfloat16)},
        {collapsed,collc,sizeof(hip_bfloat16)},
        {streams,streamc,sizeof(hip_bfloat16)}, {fn,fnc,sizeof(hip_bfloat16)},
        {base,basec,sizeof(float)}, {scale,scalec,sizeof(float)}};
    if (!disjoint(b, 9u)) return false;
    clear_status();
    hipLaunchKernelGGL(prepare_kernel, dim3(1), dim3(1), 0,
        static_cast<hipStream_t>(stream), static_cast<float *>(mix),
        static_cast<float *>(pre), static_cast<hip_bfloat16 *>(post),
        static_cast<hip_bfloat16 *>(comb), static_cast<hip_bfloat16 *>(collapsed),
        static_cast<const hip_bfloat16 *>(streams),
        static_cast<const hip_bfloat16 *>(fn), static_cast<const float *>(base),
        static_cast<const float *>(scale), width);
    return launch_ok();
}

extern "C" bool glm53_mhc_expand_bf16(
    void *out, size_t oc, const void *branch, size_t bc,
    const void *residual, size_t rc, const void *post, size_t pc,
    const void *comb, size_t cc, size_t width, void *stream) {
    size_t flat;
    if (width == 0u || !checked_mul(4u, width, &flat) || oc < flat ||
        bc < width || rc < flat || pc < 4u || cc < 16u) return false;
    const Buffer b[] = {{out,oc,2u},{branch,bc,2u},{residual,rc,2u},
                        {post,pc,2u},{comb,cc,2u}};
    unsigned grid;
    if (!disjoint(b, 5u) || !grid_for(flat, &grid)) return false;
    clear_status();
    hipLaunchKernelGGL(expand_kernel, dim3(grid), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream), static_cast<hip_bfloat16 *>(out),
        static_cast<const hip_bfloat16 *>(branch),
        static_cast<const hip_bfloat16 *>(residual),
        static_cast<const hip_bfloat16 *>(post),
        static_cast<const hip_bfloat16 *>(comb), width);
    return launch_ok();
}

extern "C" bool glm53_mhc_hyper_mean_bf16(
    void *out, size_t oc, const void *streams, size_t sc,
    size_t width, void *stream) {
    size_t flat;
    if (width == 0u || !checked_mul(4u, width, &flat) || oc < width || sc < flat)
        return false;
    const Buffer b[] = {{out,oc,2u},{streams,sc,2u}};
    unsigned grid;
    if (!disjoint(b, 2u) || !grid_for(width, &grid)) return false;
    clear_status();
    hipLaunchKernelGGL(mean_kernel, dim3(grid), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream), static_cast<hip_bfloat16 *>(out),
        static_cast<const hip_bfloat16 *>(streams), width);
    return launch_ok();
}
