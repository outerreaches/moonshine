#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                      \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define BLAS_CHECK(call)                                                    \
    do {                                                                    \
        hipblasStatus_t status_ = (call);                                   \
        if (status_ != HIPBLAS_STATUS_SUCCESS) {                            \
            fprintf(stderr, "FAIL: %s: hipBLAS status %d\n", #call,       \
                    (int)status_);                                          \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    HEADS = 96,
    LATENT = 512,
    CACHE_DIM = 576,
    QUERIES = 32,
    MAX_TOKENS = 8192,
    BASE_TOKENS = MAX_TOKENS - QUERIES,
    THREADS = 256,
    REPEATS = 4,
};

__global__ static void fill_bf16(hip_bfloat16 *output,
                                 uint64_t count,
                                 uint32_t salt) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t bits = (uint32_t)index ^ salt;
    bits ^= bits << 13;
    bits ^= bits >> 17;
    bits ^= bits << 5;
    float value = ((float)(bits & UINT32_C(0xffff)) / 32768.0f - 1.0f) *
                  0.08f;
    output[index] = hip_bfloat16(value);
}

/*
 * Reproduce the engine's reduction order. The looped layout packs each
 * head at its causal width; the batched layout pads every head to max_tokens.
 */
__global__ static void softmax_causal(
        hip_bfloat16 *probabilities,
        const float  *scores,
        uint32_t      max_tokens,
        uint32_t      base_tokens,
        bool          padded_layout) {
    const uint32_t query = blockIdx.x / HEADS;
    const uint32_t head = blockIdx.x % HEADS;
    const uint32_t tid = threadIdx.x;
    const uint32_t token_count = base_tokens + query + 1u;
    const uint64_t query_base =
        (uint64_t)query * HEADS * max_tokens;
    const uint64_t base = query_base +
        (uint64_t)head * (padded_layout ? max_tokens : token_count);
    __shared__ float reduction[THREADS];
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
    for (uint32_t width = THREADS / 2u; width > 0; width /= 2u) {
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
    for (uint32_t width = THREADS / 2u; width > 0; width /= 2u) {
        if (tid < width) reduction[tid] += reduction[tid + width];
        __syncthreads();
    }
    if (tid == 0) denominator = reduction[0];
    __syncthreads();
    const uint32_t stored_tokens = padded_layout ? max_tokens : token_count;
    for (uint32_t token = tid;
         token < stored_tokens;
         token += blockDim.x) {
        probabilities[base + token] = token < token_count ?
            hip_bfloat16(expf(scores[base + token] - maximum) /
                         denominator) : hip_bfloat16(0.0f);
    }
}

__global__ static void compact_probabilities(
        hip_bfloat16       *compact,
        const hip_bfloat16 *padded,
        uint32_t            max_tokens,
        uint32_t            base_tokens) {
    const uint32_t query = blockIdx.y;
    const uint32_t head = blockIdx.x;
    const uint32_t token_count = base_tokens + query + 1u;
    const uint64_t query_base =
        (uint64_t)query * HEADS * max_tokens;
    const uint64_t compact_base =
        query_base + (uint64_t)head * token_count;
    const uint64_t padded_base =
        query_base + (uint64_t)head * max_tokens;
    for (uint32_t token = threadIdx.x;
         token < token_count;
         token += blockDim.x) {
        compact[compact_base + token] = padded[padded_base + token];
    }
}

static int run_looped(hipblasHandle_t handle,
                      void *output,
                      void *scores,
                      void *probabilities,
                      const void *queries,
                      const void *cache) {
    const float score_alpha = 0.07216878364870322f;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const uint64_t query_stride = (uint64_t)HEADS * CACHE_DIM;
    const uint64_t workspace_stride =
        (uint64_t)HEADS * MAX_TOKENS;
    const uint64_t output_stride = (uint64_t)HEADS * LATENT;
    for (uint32_t query = 0; query < QUERIES; query++) {
        const int token_count = (int)(BASE_TOKENS + query + 1u);
        hipblasStatus_t status = hipblasGemmEx(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
            token_count, HEADS, CACHE_DIM,
            &score_alpha,
            cache, HIP_R_16BF, CACHE_DIM,
            (const hip_bfloat16 *)queries + query * query_stride,
            HIP_R_16BF, CACHE_DIM,
            &beta,
            (float *)scores + query * workspace_stride,
            HIP_R_32F, token_count,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        if (status != HIPBLAS_STATUS_SUCCESS) return 1;
    }
    hipLaunchKernelGGL(softmax_causal,
                       dim3(QUERIES * HEADS), dim3(THREADS), 0, 0,
                       (hip_bfloat16 *)probabilities,
                       (const float *)scores,
                       MAX_TOKENS, BASE_TOKENS, false);
    if (hipGetLastError() != hipSuccess) return 1;
    for (uint32_t query = 0; query < QUERIES; query++) {
        const int token_count = (int)(BASE_TOKENS + query + 1u);
        hipblasStatus_t status = hipblasGemmEx(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            LATENT, HEADS, token_count,
            &alpha,
            cache, HIP_R_16BF, CACHE_DIM,
            (const hip_bfloat16 *)probabilities +
                query * workspace_stride,
            HIP_R_16BF, token_count,
            &beta,
            (hip_bfloat16 *)output + query * output_stride,
            HIP_R_16BF, LATENT,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        if (status != HIPBLAS_STATUS_SUCCESS) return 1;
    }
    return 0;
}

static int run_strided(hipblasHandle_t handle,
                       void *output,
                       void *scores,
                       void *probabilities,
                       const void *queries,
                       const void *cache) {
    const float score_alpha = 0.07216878364870322f;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStride query_stride =
        (hipblasStride)HEADS * CACHE_DIM;
    const hipblasStride workspace_stride =
        (hipblasStride)HEADS * MAX_TOKENS;
    const hipblasStride output_stride =
        (hipblasStride)HEADS * LATENT;
    hipblasStatus_t status = hipblasGemmStridedBatchedEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
        MAX_TOKENS, HEADS, CACHE_DIM,
        &score_alpha,
        cache, HIP_R_16BF, CACHE_DIM, 0,
        queries, HIP_R_16BF, CACHE_DIM, query_stride,
        &beta,
        scores, HIP_R_32F, MAX_TOKENS, workspace_stride,
        QUERIES, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    if (status != HIPBLAS_STATUS_SUCCESS) return 1;
    hipLaunchKernelGGL(softmax_causal,
                       dim3(QUERIES * HEADS), dim3(THREADS), 0, 0,
                       (hip_bfloat16 *)probabilities,
                       (const float *)scores,
                       MAX_TOKENS, BASE_TOKENS, true);
    if (hipGetLastError() != hipSuccess) return 1;
    status = hipblasGemmStridedBatchedEx(
        handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        LATENT, HEADS, MAX_TOKENS,
        &alpha,
        cache, HIP_R_16BF, CACHE_DIM, 0,
        probabilities, HIP_R_16BF, MAX_TOKENS, workspace_stride,
        &beta,
        output, HIP_R_16BF, LATENT, output_stride,
        QUERIES, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    return status == HIPBLAS_STATUS_SUCCESS ? 0 : 1;
}

static int run_hybrid(hipblasHandle_t handle,
                      void *output,
                      void *scores,
                      void *probabilities,
                      const void *queries,
                      const void *cache) {
    const float score_alpha = 0.07216878364870322f;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStride query_stride =
        (hipblasStride)HEADS * CACHE_DIM;
    const hipblasStride workspace_stride =
        (hipblasStride)HEADS * MAX_TOKENS;
    const uint64_t output_stride = (uint64_t)HEADS * LATENT;
    hipblasStatus_t status = hipblasGemmStridedBatchedEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
        MAX_TOKENS, HEADS, CACHE_DIM,
        &score_alpha,
        cache, HIP_R_16BF, CACHE_DIM, 0,
        queries, HIP_R_16BF, CACHE_DIM, query_stride,
        &beta,
        scores, HIP_R_32F, MAX_TOKENS, workspace_stride,
        QUERIES, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    if (status != HIPBLAS_STATUS_SUCCESS) return 1;
    hipLaunchKernelGGL(softmax_causal,
                       dim3(QUERIES * HEADS), dim3(THREADS), 0, 0,
                       (hip_bfloat16 *)probabilities,
                       (const float *)scores,
                       MAX_TOKENS, BASE_TOKENS, true);
    if (hipGetLastError() != hipSuccess) return 1;
    /* Scores are dead after softmax; reuse their larger allocation as BF16. */
    hipLaunchKernelGGL(compact_probabilities,
                       dim3(HEADS, QUERIES), dim3(THREADS), 0, 0,
                       (hip_bfloat16 *)scores,
                       (const hip_bfloat16 *)probabilities,
                       MAX_TOKENS, BASE_TOKENS);
    if (hipGetLastError() != hipSuccess) return 1;
    for (uint32_t query = 0; query < QUERIES; query++) {
        const int token_count = (int)(BASE_TOKENS + query + 1u);
        status = hipblasGemmEx(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            LATENT, HEADS, token_count,
            &alpha,
            cache, HIP_R_16BF, CACHE_DIM,
            (const hip_bfloat16 *)scores +
                query * workspace_stride,
            HIP_R_16BF, token_count,
            &beta,
            (hip_bfloat16 *)output + query * output_stride,
            HIP_R_16BF, LATENT,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        if (status != HIPBLAS_STATUS_SUCCESS) return 1;
    }
    return 0;
}

static float time_path(hipblasHandle_t handle,
                       int (*path)(hipblasHandle_t, void *, void *, void *,
                                   const void *, const void *),
                       void *output,
                       void *scores,
                       void *probabilities,
                       const void *queries,
                       const void *cache) {
    hipEvent_t start;
    hipEvent_t stop;
    if (hipEventCreate(&start) != hipSuccess ||
        hipEventCreate(&stop) != hipSuccess) return -1.0f;
    if (path(handle, output, scores, probabilities, queries, cache) != 0 ||
        hipDeviceSynchronize() != hipSuccess) return -1.0f;
    if (hipEventRecord(start, 0) != hipSuccess) return -1.0f;
    for (uint32_t repeat = 0; repeat < REPEATS; repeat++) {
        if (path(handle, output, scores, probabilities,
                 queries, cache) != 0) return -1.0f;
    }
    if (hipEventRecord(stop, 0) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) return -1.0f;
    float elapsed = 0.0f;
    if (hipEventElapsedTime(&elapsed, start, stop) != hipSuccess) {
        return -1.0f;
    }
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    return elapsed / REPEATS;
}

static void compare_scores(const float *looped,
                           const float *batched,
                           uint64_t *mismatches,
                           float *maximum_absolute) {
    *mismatches = 0;
    *maximum_absolute = 0.0f;
    const uint64_t stride = (uint64_t)HEADS * MAX_TOKENS;
    for (uint32_t query = 0; query < QUERIES; query++) {
        const uint32_t count = BASE_TOKENS + query + 1u;
        for (uint32_t head = 0; head < HEADS; head++) {
            const float *left = looped + query * stride + head * count;
            const float *right = batched + query * stride +
                                 head * MAX_TOKENS;
            for (uint32_t token = 0; token < count; token++) {
                float difference = fabsf(left[token] - right[token]);
                if (difference != 0.0f) (*mismatches)++;
                if (difference > *maximum_absolute) {
                    *maximum_absolute = difference;
                }
            }
        }
    }
}

static void compare_bf16(const hip_bfloat16 *left,
                         const hip_bfloat16 *right,
                         size_t count,
                         uint64_t *mismatches,
                         float *maximum_absolute) {
    *mismatches = 0;
    *maximum_absolute = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float difference = fabsf((float)left[i] - (float)right[i]);
        if (difference != 0.0f) (*mismatches)++;
        if (difference > *maximum_absolute) {
            *maximum_absolute = difference;
        }
    }
}

int main(void) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 MLA batch determinism: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 MLA batch determinism on %s (%s)\n",
           properties.name, properties.gcnArchName);

    hipblasHandle_t handle = NULL;
    BLAS_CHECK(hipblasCreate(&handle));
    hipblasAtomicsMode_t atomics;
    BLAS_CHECK(hipblasGetAtomicsMode(handle, &atomics));
    printf("  initial atomics mode: %s\n",
           atomics == HIPBLAS_ATOMICS_NOT_ALLOWED ?
               "not-allowed" : "allowed");

    const size_t cache_count = (size_t)MAX_TOKENS * CACHE_DIM;
    const size_t query_count = (size_t)QUERIES * HEADS * CACHE_DIM;
    const size_t workspace_count =
        (size_t)QUERIES * HEADS * MAX_TOKENS;
    const size_t output_count = (size_t)QUERIES * HEADS * LATENT;
    void *d_cache = NULL;
    void *d_queries = NULL;
    void *d_loop_scores = NULL;
    void *d_batch_scores = NULL;
    void *d_loop_probabilities = NULL;
    void *d_batch_probabilities = NULL;
    void *d_loop_output = NULL;
    void *d_batch_output = NULL;
    HIP_CHECK(hipMalloc(&d_cache, cache_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_queries, query_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_loop_scores, workspace_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_batch_scores, workspace_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_loop_probabilities,
                        workspace_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_batch_probabilities,
                        workspace_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_loop_output,
                        output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_batch_output,
                        output_count * sizeof(hip_bfloat16)));
    hipLaunchKernelGGL(fill_bf16,
                       dim3((cache_count + THREADS - 1u) / THREADS),
                       dim3(THREADS), 0, 0,
                       (hip_bfloat16 *)d_cache, cache_count,
                       UINT32_C(0x6b336361));
    hipLaunchKernelGGL(fill_bf16,
                       dim3((query_count + THREADS - 1u) / THREADS),
                       dim3(THREADS), 0, 0,
                       (hip_bfloat16 *)d_queries, query_count,
                       UINT32_C(0x6b337175));
    HIP_CHECK(hipGetLastError());

    float loop_ms = time_path(handle, run_looped,
                              d_loop_output, d_loop_scores,
                              d_loop_probabilities, d_queries, d_cache);
    float batch_ms = time_path(handle, run_strided,
                               d_batch_output, d_batch_scores,
                               d_batch_probabilities, d_queries, d_cache);
    if (loop_ms < 0.0f || batch_ms < 0.0f) {
        fprintf(stderr, "FAIL: MLA path timing\n");
        return 1;
    }
    printf("  looped %.3f ms, strided %.3f ms, speedup %.3fx\n",
           loop_ms, batch_ms, loop_ms / batch_ms);

    float *loop_scores =
        (float *)malloc(workspace_count * sizeof(float));
    float *batch_scores =
        (float *)malloc(workspace_count * sizeof(float));
    hip_bfloat16 *loop_output = (hip_bfloat16 *)malloc(
        output_count * sizeof(hip_bfloat16));
    hip_bfloat16 *batch_output = (hip_bfloat16 *)malloc(
        output_count * sizeof(hip_bfloat16));
    hip_bfloat16 *repeat_output = (hip_bfloat16 *)malloc(
        output_count * sizeof(hip_bfloat16));
    if (!loop_scores || !batch_scores || !loop_output ||
        !batch_output || !repeat_output) {
        fprintf(stderr, "FAIL: host comparison allocation\n");
        return 1;
    }
    HIP_CHECK(hipMemcpy(loop_scores, d_loop_scores,
                        workspace_count * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(batch_scores, d_batch_scores,
                        workspace_count * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(loop_output, d_loop_output,
                        output_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(batch_output, d_batch_output,
                        output_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    uint64_t score_mismatches;
    float score_maximum;
    compare_scores(loop_scores, batch_scores,
                   &score_mismatches, &score_maximum);
    uint64_t output_mismatches;
    float output_maximum;
    compare_bf16(loop_output, batch_output, output_count,
                 &output_mismatches, &output_maximum);
    printf("  loop/strided valid scores: %llu mismatches, max_abs %.9g\n",
           (unsigned long long)score_mismatches, score_maximum);
    printf("  loop/strided latent BF16:  %llu/%zu mismatches, "
           "max_abs %.9g\n",
           (unsigned long long)output_mismatches, output_count,
           output_maximum);

    k3_rocm_blas_context *api_blas = NULL;
    if (!k3_rocm_blas_context_create(&api_blas) ||
        !k3_rocm_blas_mla_attention_batch_bf16(
            api_blas, d_batch_output, d_batch_scores,
            d_batch_probabilities, d_queries, d_cache,
            HEADS, BASE_TOKENS + 1u, QUERIES, NULL)) {
        fprintf(stderr, "FAIL: public strided MLA batch API\n");
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(repeat_output, d_batch_output,
                        output_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    const bool api_exact =
        memcmp(batch_output, repeat_output,
               output_count * sizeof(hip_bfloat16)) == 0;
    printf("  public batch API: %s direct strided fixture\n",
           api_exact ? "matches" : "DIFFERS FROM");
    k3_rocm_blas_context_destroy(api_blas);

    float hybrid_ms = time_path(handle, run_hybrid,
                                d_batch_output, d_batch_scores,
                                d_batch_probabilities, d_queries, d_cache);
    if (hybrid_ms < 0.0f) {
        fprintf(stderr, "FAIL: hybrid MLA path timing\n");
        return 1;
    }
    HIP_CHECK(hipMemcpy(repeat_output, d_batch_output,
                        output_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    uint64_t hybrid_mismatches;
    float hybrid_maximum;
    compare_bf16(loop_output, repeat_output, output_count,
                 &hybrid_mismatches, &hybrid_maximum);
    printf("  hybrid %.3f ms, speedup %.3fx; latent BF16 %llu/%zu "
           "mismatches, max_abs %.9g\n",
           hybrid_ms, loop_ms / hybrid_ms,
           (unsigned long long)hybrid_mismatches, output_count,
           hybrid_maximum);

    bool repeatable = true;
    for (uint32_t repeat = 0; repeat < REPEATS; repeat++) {
        if (run_strided(handle, d_batch_output, d_batch_scores,
                        d_batch_probabilities, d_queries, d_cache) != 0) {
            fprintf(stderr, "FAIL: repeated strided MLA path\n");
            return 1;
        }
        HIP_CHECK(hipMemcpy(repeat_output, d_batch_output,
                            output_count * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        if (memcmp(batch_output, repeat_output,
                   output_count * sizeof(hip_bfloat16)) != 0) {
            repeatable = false;
        }
    }
    printf("  strided latent repeatability: %s across %u repeats\n",
           repeatable ? "bitwise" : "NONDETERMINISTIC", REPEATS);

    BLAS_CHECK(hipblasSetAtomicsMode(handle,
                                     HIPBLAS_ATOMICS_NOT_ALLOWED));
    hipblasAtomicsMode_t locked_atomics;
    BLAS_CHECK(hipblasGetAtomicsMode(handle, &locked_atomics));
    if (run_strided(handle, d_batch_output, d_batch_scores,
                    d_batch_probabilities, d_queries, d_cache) != 0) {
        fprintf(stderr, "FAIL: atomics-disabled strided MLA path\n");
        return 1;
    }
    HIP_CHECK(hipMemcpy(repeat_output, d_batch_output,
                        output_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    bool atomics_same = memcmp(batch_output, repeat_output,
                               output_count * sizeof(hip_bfloat16)) == 0;
    printf("  explicit atomics not-allowed: %s; output %s initial mode\n",
           locked_atomics == HIPBLAS_ATOMICS_NOT_ALLOWED ?
               "confirmed" : "FAILED",
           atomics_same ? "matches" : "differs from");

    free(repeat_output);
    free(batch_output);
    free(loop_output);
    free(batch_scores);
    free(loop_scores);
    HIP_CHECK(hipFree(d_batch_output));
    HIP_CHECK(hipFree(d_loop_output));
    HIP_CHECK(hipFree(d_batch_probabilities));
    HIP_CHECK(hipFree(d_loop_probabilities));
    HIP_CHECK(hipFree(d_batch_scores));
    HIP_CHECK(hipFree(d_loop_scores));
    HIP_CHECK(hipFree(d_queries));
    HIP_CHECK(hipFree(d_cache));
    BLAS_CHECK(hipblasDestroy(handle));
    if (score_mismatches != 0u || hybrid_mismatches != 0u || !api_exact ||
        !repeatable || !atomics_same ||
        locked_atomics != HIPBLAS_ATOMICS_NOT_ALLOWED) {
        fprintf(stderr, "K3 MLA batch determinism: FAIL\n");
        return 1;
    }
    printf("K3 MLA batch determinism: PASS\n");
    return 0;
}
