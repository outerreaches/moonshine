#include "k31_dsa_rocm.h"
#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            std::fprintf(stderr, "FAIL: %s: %s\n", #call,                  \
                         hipGetErrorString(status_));                        \
            return 1;                                                       \
        }                                                                   \
    } while (0)

namespace {

constexpr uint32_t kIndexHeads = 32;
constexpr uint32_t kIndexDim = 128;
constexpr uint32_t kHeads = 96;
constexpr uint32_t kCacheDim = 576;
constexpr uint32_t kLatentDim = 512;
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kIterations = 20;
constexpr float kAttentionScale = 0.07216878364870322f;
uint32_t rng_state = UINT32_C(0x8d5a11);

float random_unit(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return static_cast<float>(rng_state & UINT32_C(0x00ffffff)) /
           static_cast<float>(UINT32_C(0x01000000));
}

template <typename F>
bool measure(F &&call, double *milliseconds) {
    hipEvent_t start = nullptr;
    hipEvent_t end = nullptr;
    if (hipEventCreate(&start) != hipSuccess ||
        hipEventCreate(&end) != hipSuccess) return false;
    float total = 0.0f;
    for (uint32_t iteration = 0; iteration < kIterations; iteration++) {
        if (hipEventRecord(start) != hipSuccess) return false;
        if (!call()) return false;
        if (hipEventRecord(end) != hipSuccess ||
            hipEventSynchronize(end) != hipSuccess) return false;
        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, start, end) != hipSuccess) return false;
        total += elapsed;
    }
    if (hipEventDestroy(start) != hipSuccess ||
        hipEventDestroy(end) != hipSuccess) return false;
    *milliseconds = total / kIterations;
    return true;
}

}  // namespace

int main(void) {
    constexpr uint32_t context = 8192;
    std::vector<hip_bfloat16> host(kHeads * kCacheDim);
    for (auto &value : host) value = hip_bfloat16(random_unit() - 0.5f);

    hip_bfloat16 *index_q = nullptr;
    hip_bfloat16 *index_k = nullptr;
    hip_bfloat16 *weights = nullptr;
    hip_bfloat16 *absorbed_q = nullptr;
    hip_bfloat16 *cache = nullptr;
    hip_bfloat16 *probabilities = nullptr;
    hip_bfloat16 *latent = nullptr;
    float *index_scores = nullptr;
    float *candidate_index_scores = nullptr;
    float *dot_products = nullptr;
    float *attention_scores = nullptr;
    uint32_t *selected = nullptr;
    hip_bfloat16 *gathered = nullptr;
    hip_bfloat16 *candidate = nullptr;
    k3_rocm_blas_context *blas = nullptr;
    void *topk_workspace = nullptr;
    HIP_CHECK(hipMalloc(&index_q,
                        kIndexHeads * kIndexDim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&index_k, context * kIndexDim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&weights, kIndexHeads * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&absorbed_q, host.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&cache, context * kCacheDim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&probabilities,
                        kHeads * kTopK * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&latent, kHeads * kLatentDim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&index_scores, context * sizeof(float)));
    HIP_CHECK(hipMalloc(&candidate_index_scores, context * sizeof(float)));
    HIP_CHECK(hipMalloc(&dot_products, context * kIndexHeads * sizeof(float)));
    HIP_CHECK(hipMalloc(&attention_scores, kHeads * kTopK * sizeof(float)));
    HIP_CHECK(hipMalloc(&selected, kTopK * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&gathered, kTopK * kCacheDim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&candidate, kHeads * kLatentDim * sizeof(hip_bfloat16)));
    if (!k3_rocm_blas_context_create(&blas)) return 1;
    size_t topk_bytes = 0;
    if (!k31_dsa_topk_workspace_bytes(context, nullptr, &topk_bytes)) return 1;
    HIP_CHECK(hipMalloc(&topk_workspace, topk_bytes));

    auto upload = [&](void *destination, size_t bytes) {
        size_t copied = 0;
        while (copied < bytes) {
            const size_t chunk = std::min(bytes - copied, host.size() * sizeof(hip_bfloat16));
            if (hipMemcpy(static_cast<uint8_t *>(destination) + copied,
                          host.data(), chunk, hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
            copied += chunk;
        }
        return true;
    };
    if (!upload(index_q, kIndexHeads * kIndexDim * sizeof(hip_bfloat16)) ||
        !upload(index_k, context * kIndexDim * sizeof(hip_bfloat16)) ||
        !upload(weights, kIndexHeads * sizeof(hip_bfloat16)) ||
        !upload(absorbed_q, host.size() * sizeof(hip_bfloat16)) ||
        !upload(cache, context * kCacheDim * sizeof(hip_bfloat16))) return 1;

    std::puts("tokens,index_ms,blas_index_ms,index_max_abs,topk_ms,sparse_ms,dense_ms,dense_max_abs,total_ms");
    const uint32_t widths[] = {512, 768, 1024, 1536, 2048, 4096, 8192, 65536};
    for (const uint32_t tokens : widths) {
        const uint32_t selected_count = std::min(tokens, kTopK);
        bool ok = k31_dsa_index_scores_f32(
            index_scores, index_q, index_k, weights, nullptr, nullptr,
            1, tokens, kIndexHeads, kIndexDim, nullptr) &&
            k31_dsa_topk_f32(selected, index_scores, tokens, kTopK,
                             topk_workspace, topk_bytes, nullptr) &&
            k31_dsa_sparse_attention_bf16(
                latent, attention_scores, probabilities, absorbed_q, cache,
                selected, selected_count, kHeads, kCacheDim, kLatentDim,
                kAttentionScale, nullptr);
        if (!ok) return 1;
        HIP_CHECK(hipDeviceSynchronize());
        double index_ms = 0, topk_ms = 0, sparse_ms = 0;
        if (!measure([&] {
                return k31_dsa_index_scores_f32(
                    index_scores, index_q, index_k, weights, nullptr, nullptr,
                    1, tokens, kIndexHeads, kIndexDim, nullptr);
            }, &index_ms) ||
            !measure([&] {
                return k31_dsa_topk_f32(selected, index_scores, tokens, kTopK,
                                        topk_workspace, topk_bytes, nullptr);
            }, &topk_ms) ||
            !measure([&] {
                return k31_dsa_sparse_attention_bf16(
                    latent, attention_scores, probabilities, absorbed_q, cache,
                    selected, selected_count, kHeads, kCacheDim, kLatentDim,
                    kAttentionScale, nullptr);
            }, &sparse_ms)) {
            return 1;
        }
        double blas_index_ms = 0;
        if (!measure([&] {
                return k3_rocm_blas_bf16_gemm_f32(
                           blas, dot_products, index_k, index_q,
                           kIndexHeads, tokens, kIndexDim, nullptr) &&
                       k31_dsa_index_reduce_f32(
                           candidate_index_scores, dot_products, weights,
                           nullptr, nullptr, tokens,
                           kIndexHeads, kIndexDim, nullptr);
            }, &blas_index_ms)) {
            return 1;
        }
        std::vector<float> reference_scores(tokens), candidate_scores(tokens);
        HIP_CHECK(hipMemcpy(reference_scores.data(), index_scores,
                            tokens * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(candidate_scores.data(), candidate_index_scores,
                            tokens * sizeof(float), hipMemcpyDeviceToHost));
        float index_max_abs = 0.0f;
        for (uint32_t index = 0; index < tokens; index++) {
            index_max_abs = std::max(index_max_abs,
                std::fabs(reference_scores[index] - candidate_scores[index]));
        }
        if (index_max_abs > 0.0001f) {
            std::fprintf(stderr, "BLAS indexer mismatch: %g\n", index_max_abs);
            return 1;
        }
        double dense_ms = 0;
        if (!measure([&] {
                if (selected_count != tokens &&
                    !k3_rocm_gather_rows_bf16(
                        gathered, cache, selected, selected_count,
                        kCacheDim, nullptr)) {
                    return false;
                }
                return k3_rocm_blas_mla_attention_bf16(
                    blas, candidate, attention_scores, probabilities,
                    absorbed_q, selected_count == tokens ? cache : gathered,
                    kHeads, selected_count, nullptr);
            }, &dense_ms)) {
            return 1;
        }
        std::vector<hip_bfloat16> baseline_host(kHeads * kLatentDim);
        std::vector<hip_bfloat16> candidate_host(kHeads * kLatentDim);
        HIP_CHECK(hipMemcpy(baseline_host.data(), latent,
                            baseline_host.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(candidate_host.data(), candidate,
                            candidate_host.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        float maximum_absolute = 0.0f;
        for (size_t index = 0; index < baseline_host.size(); index++) {
            maximum_absolute = std::max(maximum_absolute,
                std::fabs(static_cast<float>(baseline_host[index]) -
                          static_cast<float>(candidate_host[index])));
        }
        if (maximum_absolute > 0.0078125f) {
            std::fprintf(stderr, "dense candidate mismatch: %g\n", maximum_absolute);
            return 1;
        }
        std::printf("%u,%.6f,%.6f,%g,%.6f,%.6f,%.6f,%g,%.6f\n",
                    tokens, index_ms, blas_index_ms, index_max_abs, topk_ms,
                    sparse_ms, dense_ms, maximum_absolute,
                    index_ms + topk_ms + sparse_ms);
    }

    k3_rocm_blas_context_destroy(blas);
    HIP_CHECK(hipFree(candidate));
    HIP_CHECK(hipFree(gathered));
    HIP_CHECK(hipFree(topk_workspace));
    HIP_CHECK(hipFree(selected));
    HIP_CHECK(hipFree(attention_scores));
    HIP_CHECK(hipFree(dot_products));
    HIP_CHECK(hipFree(candidate_index_scores));
    HIP_CHECK(hipFree(index_scores));
    HIP_CHECK(hipFree(latent));
    HIP_CHECK(hipFree(probabilities));
    HIP_CHECK(hipFree(cache));
    HIP_CHECK(hipFree(absorbed_q));
    HIP_CHECK(hipFree(weights));
    HIP_CHECK(hipFree(index_k));
    HIP_CHECK(hipFree(index_q));
    return 0;
}
