#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                        \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    K3_NOPE = 128,
    K3_PASS = 64,
    K3_Q_DIM = 192,
    K3_LATENT = 512,
    K3_CACHE_DIM = 576,
    K3_VALUE = 128,
    K3_KV_HEAD_STRIDE = 256,
    CORRECTNESS_HEADS = 4,
    CORRECTNESS_TOKENS = 257,
    MODEL_HEADS = 96,
    MODEL_TOKENS = 8192,
    BENCHMARK_STEPS = 20,
};

static uint32_t rng_state = UINT32_C(0x4b334d4c);

static float random_unit(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state & UINT32_C(0x00ffffff)) /
           (float)UINT32_C(0x01000000);
}

static float random_symmetric(float scale) {
    return (random_unit() * 2.0f - 1.0f) * scale;
}

static inline float bf16_to_float(hip_bfloat16 value) {
    return (float)value;
}

static inline hip_bfloat16 float_to_bf16(float value) {
    return hip_bfloat16(value);
}

static bool compare_vector(const char *label,
                           const hip_bfloat16 *actual,
                           const hip_bfloat16 *expected,
                           size_t count,
                           float absolute_tolerance,
                           float relative_tolerance) {
    float maximum_absolute = 0.0f;
    float maximum_relative = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float got = bf16_to_float(actual[i]);
        float want = bf16_to_float(expected[i]);
        float absolute = fabsf(got - want);
        float relative = absolute / fmaxf(fabsf(want), 1e-3f);
        if (absolute > maximum_absolute) maximum_absolute = absolute;
        if (relative > maximum_relative) maximum_relative = relative;
        if (!isfinite(got) ||
            (absolute > absolute_tolerance &&
             relative > relative_tolerance)) {
            fprintf(stderr,
                    "FAIL: %s[%zu] got=%f expected=%f abs=%f rel=%f\n",
                    label, i, got, want, absolute, relative);
            return false;
        }
    }
    printf("  %-12s max_abs=%-10.7f max_rel=%.7f\n",
           label, maximum_absolute, maximum_relative);
    return true;
}

static void reference_absorb_q(hip_bfloat16 *output,
                               const hip_bfloat16 *q,
                               const hip_bfloat16 *kv_b,
                               uint32_t heads) {
    for (uint32_t head = 0; head < heads; head++) {
        for (uint32_t latent = 0; latent < K3_LATENT; latent++) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < K3_NOPE; d++) {
                sum += bf16_to_float(q[(uint64_t)head * K3_Q_DIM + d]) *
                       bf16_to_float(kv_b[
                           ((uint64_t)head * K3_KV_HEAD_STRIDE + d) *
                           K3_LATENT + latent]);
            }
            output[(uint64_t)head * K3_CACHE_DIM + latent] =
                float_to_bf16(sum);
        }
        for (uint32_t d = 0; d < K3_PASS; d++) {
            output[(uint64_t)head * K3_CACHE_DIM + K3_LATENT + d] =
                q[(uint64_t)head * K3_Q_DIM + K3_NOPE + d];
        }
    }
}

static bool reference_attention(hip_bfloat16 *output,
                                hip_bfloat16 *probabilities,
                                const hip_bfloat16 *query,
                                const hip_bfloat16 *cache,
                                uint32_t heads,
                                uint32_t tokens) {
    float *scores = (float *)malloc((size_t)tokens * sizeof(float));
    if (!scores) return false;
    const float scale = 1.0f / sqrtf((float)K3_Q_DIM);
    for (uint32_t head = 0; head < heads; head++) {
        float maximum = -INFINITY;
        for (uint32_t token = 0; token < tokens; token++) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < K3_CACHE_DIM; d++) {
                sum += bf16_to_float(
                           query[(uint64_t)head * K3_CACHE_DIM + d]) *
                       bf16_to_float(
                           cache[(uint64_t)token * K3_CACHE_DIM + d]);
            }
            scores[token] = sum * scale;
            maximum = fmaxf(maximum, scores[token]);
        }
        float denominator = 0.0f;
        for (uint32_t token = 0; token < tokens; token++) {
            scores[token] = expf(scores[token] - maximum);
            denominator += scores[token];
        }
        for (uint32_t token = 0; token < tokens; token++) {
            probabilities[(uint64_t)head * tokens + token] =
                float_to_bf16(scores[token] / denominator);
        }
        for (uint32_t d = 0; d < K3_LATENT; d++) {
            float sum = 0.0f;
            for (uint32_t token = 0; token < tokens; token++) {
                sum += bf16_to_float(
                           probabilities[(uint64_t)head * tokens + token]) *
                       bf16_to_float(
                           cache[(uint64_t)token * K3_CACHE_DIM + d]);
            }
            output[(uint64_t)head * K3_LATENT + d] =
                float_to_bf16(sum);
        }
    }
    free(scores);
    return true;
}

static void reference_decompress_v(hip_bfloat16 *output,
                                   const hip_bfloat16 *latent,
                                   const hip_bfloat16 *kv_b,
                                   uint32_t heads) {
    for (uint32_t head = 0; head < heads; head++) {
        for (uint32_t value = 0; value < K3_VALUE; value++) {
            float sum = 0.0f;
            const uint64_t weight_base =
                ((uint64_t)head * K3_KV_HEAD_STRIDE +
                 K3_NOPE + value) * K3_LATENT;
            for (uint32_t d = 0; d < K3_LATENT; d++) {
                sum += bf16_to_float(kv_b[weight_base + d]) *
                       bf16_to_float(
                           latent[(uint64_t)head * K3_LATENT + d]);
            }
            output[(uint64_t)head * K3_VALUE + value] =
                float_to_bf16(sum);
        }
    }
}

static int test_correctness(k3_rocm_blas_context *blas) {
    const uint32_t heads = CORRECTNESS_HEADS;
    const uint32_t tokens = CORRECTNESS_TOKENS;
    const size_t q_count = (size_t)heads * K3_Q_DIM;
    const size_t absorbed_count = (size_t)heads * K3_CACHE_DIM;
    const size_t kv_b_count =
        (size_t)heads * K3_KV_HEAD_STRIDE * K3_LATENT;
    const size_t packed_count =
        (size_t)heads * K3_LATENT * K3_NOPE;
    const size_t cache_count = (size_t)tokens * K3_CACHE_DIM;
    const size_t probability_count = (size_t)heads * tokens;
    const size_t latent_count = (size_t)heads * K3_LATENT;
    const size_t output_count = (size_t)heads * K3_VALUE;
    hip_bfloat16 *q =
        (hip_bfloat16 *)malloc(q_count * sizeof(*q));
    hip_bfloat16 *kv_b =
        (hip_bfloat16 *)malloc(kv_b_count * sizeof(*kv_b));
    hip_bfloat16 *cache =
        (hip_bfloat16 *)malloc(cache_count * sizeof(*cache));
    hip_bfloat16 *expected_absorbed =
        (hip_bfloat16 *)malloc(absorbed_count * sizeof(*expected_absorbed));
    hip_bfloat16 *expected_probabilities =
        (hip_bfloat16 *)malloc(
            probability_count * sizeof(*expected_probabilities));
    hip_bfloat16 *expected_latent =
        (hip_bfloat16 *)malloc(latent_count * sizeof(*expected_latent));
    hip_bfloat16 *expected_output =
        (hip_bfloat16 *)malloc(output_count * sizeof(*expected_output));
    hip_bfloat16 *got =
        (hip_bfloat16 *)malloc(absorbed_count * sizeof(*got));
    CHECK(q && kv_b && cache && expected_absorbed &&
          expected_probabilities && expected_latent &&
          expected_output && got,
          "MLA correctness host allocation");
    for (size_t i = 0; i < q_count; i++) {
        q[i] = float_to_bf16(random_symmetric(0.2f));
    }
    for (size_t i = 0; i < kv_b_count; i++) {
        kv_b[i] = float_to_bf16(random_symmetric(0.05f));
    }
    for (size_t i = 0; i < cache_count; i++) {
        cache[i] = float_to_bf16(random_symmetric(0.2f));
    }
    reference_absorb_q(expected_absorbed, q, kv_b, heads);
    CHECK(reference_attention(expected_latent, expected_probabilities,
                              expected_absorbed, cache, heads, tokens),
          "MLA reference attention");
    reference_decompress_v(
        expected_output, expected_latent, kv_b, heads);

    void *d_q = NULL;
    void *d_kv_b = NULL;
    void *d_packed = NULL;
    void *d_absorbed = NULL;
    void *d_cache = NULL;
    void *d_scores = NULL;
    void *d_probabilities = NULL;
    void *d_latent = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_q, q_count * sizeof(*q)));
    HIP_CHECK(hipMalloc(&d_kv_b, kv_b_count * sizeof(*kv_b)));
    HIP_CHECK(hipMalloc(&d_packed, packed_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_absorbed,
                        absorbed_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_cache, cache_count * sizeof(*cache)));
    HIP_CHECK(hipMalloc(&d_scores,
                        probability_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_probabilities,
                        probability_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_latent,
                        latent_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_output,
                        output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMemcpy(d_q, q, q_count * sizeof(*q),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_kv_b, kv_b, kv_b_count * sizeof(*kv_b),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cache, cache, cache_count * sizeof(*cache),
                        hipMemcpyHostToDevice));
    CHECK(k3_rocm_mla_pack_k_weight_bf16(
              d_packed, d_kv_b, heads, NULL),
          "MLA K-weight pack launch");
    CHECK(k3_rocm_mla_absorb_q_bf16(
              d_absorbed, d_q, d_packed, heads, NULL),
          "MLA query absorption launch");
    CHECK(k3_rocm_blas_mla_attention_bf16(
              blas, d_latent, d_scores, d_probabilities,
              d_absorbed, d_cache, heads, tokens, NULL),
          "MLA compressed attention launch");
    CHECK(k3_rocm_mla_decompress_v_bf16(
              d_output, d_latent, d_kv_b, heads, NULL),
          "MLA value decompression launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got, d_absorbed,
                        absorbed_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("absorbed_q", got, expected_absorbed,
                         absorbed_count, 0.0078125f, 0.01f),
          "MLA absorbed query mismatch");
    HIP_CHECK(hipMemcpy(got, d_latent,
                        latent_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("latent_out", got, expected_latent,
                         latent_count, 0.0078125f, 0.02f),
          "MLA latent attention mismatch");
    HIP_CHECK(hipMemcpy(got, d_output,
                        output_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("value_out", got, expected_output,
                         output_count, 0.015625f, 0.03f),
          "MLA value decompression mismatch");

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_latent));
    HIP_CHECK(hipFree(d_probabilities));
    HIP_CHECK(hipFree(d_scores));
    HIP_CHECK(hipFree(d_cache));
    HIP_CHECK(hipFree(d_absorbed));
    HIP_CHECK(hipFree(d_packed));
    HIP_CHECK(hipFree(d_kv_b));
    HIP_CHECK(hipFree(d_q));
    free(got);
    free(expected_output);
    free(expected_latent);
    free(expected_probabilities);
    free(expected_absorbed);
    free(cache);
    free(kv_b);
    free(q);
    return 0;
}

static int benchmark_model_shape(k3_rocm_blas_context *blas) {
    const uint32_t heads = MODEL_HEADS;
    const uint32_t tokens = MODEL_TOKENS;
    const size_t q_bytes =
        (size_t)heads * K3_Q_DIM * sizeof(hip_bfloat16);
    const size_t kv_b_bytes =
        (size_t)heads * K3_KV_HEAD_STRIDE * K3_LATENT *
        sizeof(hip_bfloat16);
    const size_t packed_bytes =
        (size_t)heads * K3_LATENT * K3_NOPE * sizeof(hip_bfloat16);
    const size_t absorbed_bytes =
        (size_t)heads * K3_CACHE_DIM * sizeof(hip_bfloat16);
    const size_t cache_bytes =
        (size_t)tokens * K3_CACHE_DIM * sizeof(hip_bfloat16);
    const size_t score_bytes =
        (size_t)heads * tokens * sizeof(float);
    const size_t probability_bytes =
        (size_t)heads * tokens * sizeof(hip_bfloat16);
    const size_t latent_bytes =
        (size_t)heads * K3_LATENT * sizeof(hip_bfloat16);
    const size_t output_bytes =
        (size_t)heads * K3_VALUE * sizeof(hip_bfloat16);
    void *d_q = NULL;
    void *d_kv_b = NULL;
    void *d_packed = NULL;
    void *d_absorbed = NULL;
    void *d_cache = NULL;
    void *d_scores = NULL;
    void *d_probabilities = NULL;
    void *d_latent = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_q, q_bytes));
    HIP_CHECK(hipMalloc(&d_kv_b, kv_b_bytes));
    HIP_CHECK(hipMalloc(&d_packed, packed_bytes));
    HIP_CHECK(hipMalloc(&d_absorbed, absorbed_bytes));
    HIP_CHECK(hipMalloc(&d_cache, cache_bytes));
    HIP_CHECK(hipMalloc(&d_scores, score_bytes));
    HIP_CHECK(hipMalloc(&d_probabilities, probability_bytes));
    HIP_CHECK(hipMalloc(&d_latent, latent_bytes));
    HIP_CHECK(hipMalloc(&d_output, output_bytes));
    HIP_CHECK(hipMemset(d_q, 0, q_bytes));
    HIP_CHECK(hipMemset(d_kv_b, 0, kv_b_bytes));
    HIP_CHECK(hipMemset(d_cache, 0, cache_bytes));
    CHECK(k3_rocm_mla_pack_k_weight_bf16(
              d_packed, d_kv_b, heads, NULL),
          "MLA benchmark K-weight pack");
    CHECK(k3_rocm_mla_absorb_q_bf16(
              d_absorbed, d_q, d_packed, heads, NULL),
          "MLA benchmark query absorption");
    CHECK(k3_rocm_blas_mla_attention_bf16(
              blas, d_latent, d_scores, d_probabilities,
              d_absorbed, d_cache, heads, tokens, NULL),
          "MLA benchmark attention warmup");
    CHECK(k3_rocm_mla_decompress_v_bf16(
              d_output, d_latent, d_kv_b, heads, NULL),
          "MLA benchmark value warmup");
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t i = 0; i < BENCHMARK_STEPS; i++) {
        CHECK(k3_rocm_mla_absorb_q_bf16(
                  d_absorbed, d_q, d_packed, heads, NULL),
              "timed MLA query absorption");
        CHECK(k3_rocm_blas_mla_attention_bf16(
                  blas, d_latent, d_scores, d_probabilities,
                  d_absorbed, d_cache, heads, tokens, NULL),
              "timed MLA compressed attention");
        CHECK(k3_rocm_mla_decompress_v_bf16(
                  d_output, d_latent, d_kv_b, heads, NULL),
              "timed MLA value decompression");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("  model MLA core, ctx=8192 cache=9.000 MiB/layer: "
           "%.3f ms/token/layer (mean of %u)\n",
           elapsed_ms / (float)BENCHMARK_STEPS,
           BENCHMARK_STEPS);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_latent));
    HIP_CHECK(hipFree(d_probabilities));
    HIP_CHECK(hipFree(d_scores));
    HIP_CHECK(hipFree(d_cache));
    HIP_CHECK(hipFree(d_absorbed));
    HIP_CHECK(hipFree(d_packed));
    HIP_CHECK(hipFree(d_kv_b));
    HIP_CHECK(hipFree(d_q));
    return 0;
}

int main(void) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 MLA decode: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 MLA decode on %s (%s)\n",
           properties.name, properties.gcnArchName);
    k3_rocm_blas_context *blas = NULL;
    CHECK(k3_rocm_blas_context_create(&blas),
          "MLA rocBLAS context creation");
    if (test_correctness(blas) != 0) return 1;
    if (benchmark_model_shape(blas) != 0) return 1;
    k3_rocm_blas_context_destroy(blas);
    printf("K3 MLA decode: PASS\n");
    return 0;
}
