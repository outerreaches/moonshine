#include "k31_dsa_rocm.h"
#include "k31_dsa_reference.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
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

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL: %s\n", message);                  \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static uint32_t rng_state = UINT32_C(0x31d5a202);

static float random_unit(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return static_cast<float>(rng_state & UINT32_C(0x00ffffff)) /
           static_cast<float>(UINT32_C(0x01000000));
}

static hip_bfloat16 random_bf16(float scale) {
    return hip_bfloat16((random_unit() * 2.0f - 1.0f) * scale);
}

static std::vector<uint32_t> reference_topk(
        const std::vector<float> &scores,
        uint32_t top_k) {
    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(scores.size());
    for (uint32_t index = 0; index < scores.size(); index++) {
        if (scores[index] != -INFINITY) {
            candidates.emplace_back(scores[index], index);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &left, const auto &right) {
                  return left.first != right.first ?
                      left.first > right.first : left.second < right.second;
              });
    std::vector<uint32_t> result(top_k, UINT32_MAX);
    const size_t count = std::min<size_t>(top_k, candidates.size());
    for (size_t index = 0; index < count; index++) {
        result[index] = candidates[index].second;
    }
    return result;
}

static int run_topk_case(uint32_t token_count, uint32_t top_k, int mode) {
    std::vector<float> scores(token_count);
    for (uint32_t index = 0; index < token_count; index++) {
        scores[index] = mode == 2 ?
            (index % 3u == 0u ? -INFINITY : static_cast<float>(index % 7)) :
            mode == 1 ? static_cast<float>(index % 7) - 3.0f :
            random_unit() * 2.0f - 1.0f;
    }
    const std::vector<uint32_t> expected = reference_topk(scores, top_k);
    float *device_scores = nullptr;
    uint32_t *device_ids = nullptr;
    void *workspace = nullptr;
    HIP_CHECK(hipMalloc(&device_scores, scores.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_ids, (size_t)top_k * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(device_scores, scores.data(),
                        scores.size() * sizeof(float), hipMemcpyHostToDevice));
    size_t workspace_bytes = 0;
    CHECK(k31_dsa_topk_workspace_bytes(token_count, nullptr, &workspace_bytes),
          "query top-k workspace");
    CHECK(workspace_bytes > 0, "empty top-k workspace");
    CHECK(!k31_dsa_topk_f32(device_ids, device_scores, token_count, top_k,
                            nullptr, 0, nullptr),
          "top-k accepted missing workspace");
    HIP_CHECK(hipMalloc(&workspace, workspace_bytes));
    CHECK(k31_dsa_topk_f32(device_ids, device_scores, token_count, top_k,
                           workspace, workspace_bytes, nullptr),
          "top-k launch");
    std::vector<uint32_t> actual(top_k);
    HIP_CHECK(hipMemcpy(actual.data(), device_ids,
                        (size_t)top_k * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    CHECK(actual == expected, "top-k order, tie-break, or padding mismatch");
    HIP_CHECK(hipFree(workspace));
    HIP_CHECK(hipFree(device_ids));
    HIP_CHECK(hipFree(device_scores));
    return 0;
}

static int test_topk(void) {
    CHECK(!k31_dsa_topk_workspace_bytes(0, nullptr, nullptr),
          "zero-length top-k query accepted null output");
    size_t bytes = 0;
    CHECK(!k31_dsa_topk_workspace_bytes(0, nullptr, &bytes),
          "zero-length top-k accepted");
    CHECK(run_topk_case(1, 2048, 0) == 0, "single-token top-k");
    CHECK(run_topk_case(2047, 2048, 0) == 0, "2047-token top-k");
    CHECK(run_topk_case(2048, 2048, 0) == 0, "2048-token top-k");
    CHECK(run_topk_case(2049, 2048, 0) == 0, "2049-token top-k");
    CHECK(run_topk_case(2051, 2048, 1) == 0, "deterministic tied top-k");
    CHECK(run_topk_case(1048576, 2048, 0) == 0,
          "architectural-width top-k");
    CHECK(run_topk_case(5, 8, 1) == 0, "padded top-k");
    CHECK(run_topk_case(8, 8, 2) == 0, "negative-infinity padding");
    return 0;
}

static int test_indexer(void) {
    constexpr uint32_t queries_count = 2;
    constexpr uint32_t keys_count = 5;
    constexpr uint32_t heads = 2;
    constexpr uint32_t dim = 4;
    constexpr float epsilon = 1e-5f;
    std::vector<hip_bfloat16> raw_keys(keys_count * dim);
    std::vector<hip_bfloat16> gamma(dim);
    std::vector<hip_bfloat16> beta(dim);
    std::vector<hip_bfloat16> queries(queries_count * heads * dim);
    std::vector<hip_bfloat16> weights(queries_count * heads);
    for (auto &value : raw_keys) value = random_bf16(0.5f);
    for (auto &value : gamma) value = random_bf16(0.2f) + hip_bfloat16(1.0f);
    for (auto &value : beta) value = random_bf16(0.05f);
    for (auto &value : queries) value = random_bf16(0.3f);
    for (auto &value : weights) value = random_bf16(0.8f);
    const float query_scales[] = {1.0f, 0.5f, 1.25f, 0.75f};
    const float key_scales[] = {1.0f, 1.5f, 0.75f, 1.25f, 0.5f};

    auto as_float = [](const std::vector<hip_bfloat16> &input) {
        std::vector<float> output;
        output.reserve(input.size());
        for (const auto value : input) output.push_back(static_cast<float>(value));
        return output;
    };
    const std::vector<float> raw_float = as_float(raw_keys);
    const std::vector<float> gamma_float = as_float(gamma);
    const std::vector<float> beta_float = as_float(beta);
    std::vector<float> expected_norm(raw_keys.size());
    CHECK(k31_dsa_normalize_keys(raw_float.data(), keys_count, dim, 0,
                                 gamma_float.data(), beta_float.data(),
                                 epsilon, expected_norm.data()),
          "CPU key normalization");
    std::vector<hip_bfloat16> normalized_keys(raw_keys.size());
    for (size_t index = 0; index < raw_keys.size(); index++) {
        normalized_keys[index] = hip_bfloat16(expected_norm[index]);
    }
    const std::vector<float> query_float = as_float(queries);
    const std::vector<float> key_float = as_float(normalized_keys);
    const std::vector<float> weight_float = as_float(weights);
    std::vector<float> expected_scores(queries_count * keys_count);
    CHECK(k31_dsa_scores(query_float.data(), queries_count, heads, dim,
                         key_float.data(), keys_count, weight_float.data(),
                         query_scales, key_scales, expected_scores.data()),
          "CPU indexer scoring");

    hip_bfloat16 *device_raw = nullptr;
    hip_bfloat16 *device_gamma = nullptr;
    hip_bfloat16 *device_beta = nullptr;
    hip_bfloat16 *device_norm = nullptr;
    hip_bfloat16 *device_queries = nullptr;
    hip_bfloat16 *device_weights = nullptr;
    float *device_query_scales = nullptr;
    float *device_key_scales = nullptr;
    float *device_scores = nullptr;
    HIP_CHECK(hipMalloc(&device_raw, raw_keys.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_gamma, gamma.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_beta, beta.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_norm, raw_keys.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_queries, queries.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_weights, weights.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_query_scales, sizeof(query_scales)));
    HIP_CHECK(hipMalloc(&device_key_scales, sizeof(key_scales)));
    HIP_CHECK(hipMalloc(&device_scores, expected_scores.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(device_raw, raw_keys.data(),
                        raw_keys.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_gamma, gamma.data(),
                        gamma.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_beta, beta.data(),
                        beta.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_queries, queries.data(),
                        queries.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_weights, weights.data(),
                        weights.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_query_scales, query_scales,
                        sizeof(query_scales), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_key_scales, key_scales,
                        sizeof(key_scales), hipMemcpyHostToDevice));
    CHECK(k31_dsa_layer_norm_bf16(device_norm, device_raw, device_gamma,
                                  device_beta, keys_count, dim, epsilon, nullptr),
          "GPU key normalization launch");
    CHECK(k31_dsa_index_scores_f32(device_scores, device_queries, device_norm,
                                   device_weights, device_query_scales,
                                   device_key_scales, queries_count, keys_count,
                                   heads, dim, nullptr),
          "GPU indexer scoring launch");
    std::vector<hip_bfloat16> actual_norm(raw_keys.size());
    std::vector<float> actual_scores(expected_scores.size());
    HIP_CHECK(hipMemcpy(actual_norm.data(), device_norm,
                        actual_norm.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_scores.data(), device_scores,
                        actual_scores.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    for (size_t index = 0; index < actual_norm.size(); index++) {
        CHECK(std::fabs(static_cast<float>(actual_norm[index]) -
                        static_cast<float>(normalized_keys[index])) <= 0.0078125f,
              "GPU key LayerNorm mismatch");
    }
    float maximum_absolute = 0.0f;
    for (size_t index = 0; index < actual_scores.size(); index++) {
        maximum_absolute = std::max(maximum_absolute,
            std::fabs(actual_scores[index] - expected_scores[index]));
    }
    CHECK(maximum_absolute <= 0.0001f, "GPU indexer score mismatch");
    CHECK(k31_dsa_index_scores_f32(nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, 0, 0, heads, dim, nullptr),
          "empty indexer invocation rejected");

    HIP_CHECK(hipFree(device_scores));
    HIP_CHECK(hipFree(device_key_scales));
    HIP_CHECK(hipFree(device_query_scales));
    HIP_CHECK(hipFree(device_weights));
    HIP_CHECK(hipFree(device_queries));
    HIP_CHECK(hipFree(device_norm));
    HIP_CHECK(hipFree(device_beta));
    HIP_CHECK(hipFree(device_gamma));
    HIP_CHECK(hipFree(device_raw));
    std::printf("indexer scores max_abs=%g\n", maximum_absolute);
    return 0;
}

static int test_sparse_attention(void) {
    constexpr uint32_t heads = 3;
    constexpr uint32_t cache_dim = 16;
    constexpr uint32_t latent_dim = 8;
    constexpr uint32_t tokens = 7;
    const uint32_t selected_ids[] = {5, 0, 6, 2, UINT32_MAX, UINT32_MAX};
    constexpr uint32_t selected = sizeof(selected_ids) / sizeof(selected_ids[0]);
    const float scale = 0.25f;
    std::vector<hip_bfloat16> query(heads * cache_dim);
    std::vector<hip_bfloat16> cache(tokens * cache_dim);
    for (auto &value : query) value = random_bf16(0.25f);
    for (auto &value : cache) value = random_bf16(0.25f);

    std::vector<float> scores(heads * selected);
    std::vector<hip_bfloat16> probabilities(heads * selected);
    std::vector<hip_bfloat16> expected(heads * latent_dim);
    for (uint32_t head = 0; head < heads; head++) {
        float maximum = -INFINITY;
        for (uint32_t slot = 0; slot < selected; slot++) {
            if (selected_ids[slot] == UINT32_MAX) {
                scores[head * selected + slot] = -INFINITY;
                continue;
            }
            float sum = 0.0f;
            for (uint32_t d = 0; d < cache_dim; d++) {
                sum += static_cast<float>(query[head * cache_dim + d]) *
                    static_cast<float>(cache[
                        (uint64_t)selected_ids[slot] * cache_dim + d]);
            }
            scores[head * selected + slot] = sum * scale;
            maximum = std::max(maximum, scores[head * selected + slot]);
        }
        float denominator = 0.0f;
        for (uint32_t slot = 0; slot < selected; slot++) {
            if (selected_ids[slot] == UINT32_MAX) continue;
            denominator += std::exp(scores[head * selected + slot] - maximum);
        }
        for (uint32_t slot = 0; slot < selected; slot++) {
            probabilities[head * selected + slot] = hip_bfloat16(
                selected_ids[slot] == UINT32_MAX ? 0.0f :
                std::exp(scores[head * selected + slot] - maximum) /
                denominator);
        }
        for (uint32_t latent = 0; latent < latent_dim; latent++) {
            float sum = 0.0f;
            for (uint32_t slot = 0; slot < selected; slot++) {
                if (selected_ids[slot] == UINT32_MAX) continue;
                sum += static_cast<float>(probabilities[head * selected + slot]) *
                    static_cast<float>(cache[
                        (uint64_t)selected_ids[slot] * cache_dim + latent]);
            }
            expected[head * latent_dim + latent] = hip_bfloat16(sum);
        }
    }

    hip_bfloat16 *device_query = nullptr;
    hip_bfloat16 *device_cache = nullptr;
    uint32_t *device_ids = nullptr;
    float *device_scores = nullptr;
    hip_bfloat16 *device_probabilities = nullptr;
    hip_bfloat16 *device_output = nullptr;
    HIP_CHECK(hipMalloc(&device_query, query.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_cache, cache.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_ids, sizeof(selected_ids)));
    HIP_CHECK(hipMalloc(&device_scores, scores.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_probabilities,
                        probabilities.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_output, expected.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMemcpy(device_query, query.data(),
                        query.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_cache, cache.data(),
                        cache.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_ids, selected_ids, sizeof(selected_ids),
                        hipMemcpyHostToDevice));
    CHECK(k31_dsa_sparse_attention_bf16(
              device_output, device_scores, device_probabilities,
              device_query, device_cache, device_ids, selected,
              heads, cache_dim, latent_dim, scale, nullptr),
          "sparse attention launch");
    std::vector<hip_bfloat16> actual(expected.size());
    HIP_CHECK(hipMemcpy(actual.data(), device_output,
                        actual.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    float maximum_absolute = 0.0f;
    for (size_t index = 0; index < actual.size(); index++) {
        maximum_absolute = std::max(maximum_absolute,
            std::fabs(static_cast<float>(actual[index]) -
                      static_cast<float>(expected[index])));
    }
    CHECK(maximum_absolute <= 0.0078125f,
          "sparse attention numerical mismatch");
    const uint32_t identity_ids[] = {0, 1, 2, 3};
    HIP_CHECK(hipMemcpy(device_ids, identity_ids, sizeof(identity_ids),
                        hipMemcpyHostToDevice));
    CHECK(k31_dsa_sparse_attention_bf16(
              device_output, device_scores, device_probabilities,
              device_query, device_cache, device_ids, 4,
              heads, cache_dim, latent_dim, scale, nullptr),
          "explicit identity sparse attention launch");
    std::vector<hip_bfloat16> identity_expected(expected.size());
    HIP_CHECK(hipMemcpy(identity_expected.data(), device_output,
                        identity_expected.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    CHECK(k31_dsa_sparse_attention_bf16(
              device_output, device_scores, device_probabilities,
              device_query, device_cache, nullptr, 4,
              heads, cache_dim, latent_dim, scale, nullptr),
          "implicit identity sparse attention launch");
    HIP_CHECK(hipMemcpy(actual.data(), device_output,
                        actual.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    CHECK(actual == identity_expected, "implicit identity rows differ");

    HIP_CHECK(hipMemset(device_output, 0xff,
                        actual.size() * sizeof(hip_bfloat16)));
    CHECK(k31_dsa_sparse_attention_bf16(
              device_output, nullptr, nullptr, nullptr, nullptr, nullptr,
              0, heads, cache_dim, latent_dim, scale, nullptr),
          "zero-selection launch");
    HIP_CHECK(hipMemcpy(actual.data(), device_output,
                        actual.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    for (const auto value : actual) {
        CHECK(static_cast<float>(value) == 0.0f,
              "zero-selection output was not zero");
    }

    HIP_CHECK(hipMemset(device_ids, 0xff, 2 * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(device_output, 0xff,
                        actual.size() * sizeof(hip_bfloat16)));
    CHECK(k31_dsa_sparse_attention_bf16(
              device_output, device_scores, device_probabilities,
              device_query, device_cache, device_ids, 2,
              heads, cache_dim, latent_dim, scale, nullptr),
          "fully padded sparse attention launch");
    HIP_CHECK(hipMemcpy(actual.data(), device_output,
                        actual.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    for (const auto value : actual) {
        CHECK(static_cast<float>(value) == 0.0f,
              "fully padded output was not zero");
    }

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_probabilities));
    HIP_CHECK(hipFree(device_scores));
    HIP_CHECK(hipFree(device_ids));
    HIP_CHECK(hipFree(device_cache));
    HIP_CHECK(hipFree(device_query));
    std::printf("sparse attention max_abs=%g\n", maximum_absolute);
    return 0;
}

static int test_decode_prefill_equivalence(void) {
    constexpr uint32_t tokens = 4;
    constexpr uint32_t index_heads = 2;
    constexpr uint32_t index_dim = 4;
    constexpr uint32_t heads = 2;
    constexpr uint32_t cache_dim = 8;
    constexpr uint32_t latent_dim = 4;
    constexpr uint32_t top_k = 3;
    std::vector<hip_bfloat16> index_queries(tokens * index_heads * index_dim);
    std::vector<hip_bfloat16> index_keys(tokens * index_dim);
    std::vector<hip_bfloat16> head_weights(tokens * index_heads);
    std::vector<hip_bfloat16> absorbed(tokens * heads * cache_dim);
    std::vector<hip_bfloat16> cache(tokens * cache_dim);
    for (auto &value : index_queries) value = random_bf16(0.3f);
    for (auto &value : index_keys) value = random_bf16(0.3f);
    for (auto &value : head_weights) value = random_bf16(0.7f);
    for (auto &value : absorbed) value = random_bf16(0.2f);
    for (auto &value : cache) value = random_bf16(0.2f);

    hip_bfloat16 *device_queries = nullptr;
    hip_bfloat16 *device_keys = nullptr;
    hip_bfloat16 *device_weights = nullptr;
    hip_bfloat16 *device_absorbed = nullptr;
    hip_bfloat16 *device_cache = nullptr;
    float *device_all_scores = nullptr;
    float *device_decode_scores = nullptr;
    uint32_t *device_ids = nullptr;
    float *device_attention_scores = nullptr;
    hip_bfloat16 *device_probabilities = nullptr;
    hip_bfloat16 *device_prefill = nullptr;
    hip_bfloat16 *device_decode = nullptr;
    void *topk_workspace = nullptr;
    HIP_CHECK(hipMalloc(&device_queries,
                        index_queries.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_keys, index_keys.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_weights,
                        head_weights.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_absorbed, absorbed.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_cache, cache.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_all_scores, tokens * tokens * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_decode_scores, tokens * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_ids, top_k * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&device_attention_scores, heads * top_k * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_probabilities,
                        heads * top_k * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_prefill,
                        tokens * heads * latent_dim * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&device_decode,
                        tokens * heads * latent_dim * sizeof(hip_bfloat16)));
    size_t workspace_bytes = 0;
    CHECK(k31_dsa_topk_workspace_bytes(tokens, nullptr, &workspace_bytes),
          "decode/prefill top-k workspace query");
    HIP_CHECK(hipMalloc(&topk_workspace, workspace_bytes));
    HIP_CHECK(hipMemcpy(device_queries, index_queries.data(),
                        index_queries.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_keys, index_keys.data(),
                        index_keys.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_weights, head_weights.data(),
                        head_weights.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_absorbed, absorbed.data(),
                        absorbed.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_cache, cache.data(),
                        cache.size() * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    CHECK(k31_dsa_index_scores_f32(
              device_all_scores, device_queries, device_keys, device_weights,
              nullptr, nullptr, tokens, tokens, index_heads, index_dim, nullptr),
          "batched prefill indexer scores");

    for (uint32_t token = 0; token < tokens; token++) {
        const uint32_t visible = token + 1u;
        CHECK(k31_dsa_topk_f32(device_ids, device_all_scores + token * tokens,
                               visible, top_k, topk_workspace,
                               workspace_bytes, nullptr),
              "prefill top-k");
        CHECK(k31_dsa_sparse_attention_bf16(
                  device_prefill + (uint64_t)token * heads * latent_dim,
                  device_attention_scores, device_probabilities,
                  device_absorbed + (uint64_t)token * heads * cache_dim,
                  device_cache, device_ids, visible < top_k ? visible : top_k,
                  heads, cache_dim, latent_dim, 0.5f, nullptr),
              "prefill sparse attention");
        CHECK(k31_dsa_index_scores_f32(
                  device_decode_scores,
                  device_queries + (uint64_t)token * index_heads * index_dim,
                  device_keys,
                  device_weights + (uint64_t)token * index_heads,
                  nullptr, nullptr, 1, visible, index_heads, index_dim, nullptr),
              "decode indexer scores");
        std::vector<uint32_t> prefill_ids(top_k), decode_ids(top_k);
        std::vector<float> decode_scores(visible);
        HIP_CHECK(hipMemcpy(prefill_ids.data(), device_ids,
                            top_k * sizeof(uint32_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(decode_scores.data(), device_decode_scores,
                            visible * sizeof(float), hipMemcpyDeviceToHost));
        CHECK(k31_dsa_topk_f32(device_ids, device_decode_scores,
                               visible, top_k, topk_workspace,
                               workspace_bytes, nullptr),
              "decode top-k");
        HIP_CHECK(hipMemcpy(decode_ids.data(), device_ids,
                            top_k * sizeof(uint32_t), hipMemcpyDeviceToHost));
        CHECK(prefill_ids == decode_ids, "decode/prefill selected IDs differ");
        CHECK(k31_dsa_sparse_attention_bf16(
                  device_decode + (uint64_t)token * heads * latent_dim,
                  device_attention_scores, device_probabilities,
                  device_absorbed + (uint64_t)token * heads * cache_dim,
                  device_cache, device_ids, visible < top_k ? visible : top_k,
                  heads, cache_dim, latent_dim, 0.5f, nullptr),
              "decode sparse attention");
    }
    std::vector<hip_bfloat16> prefill(tokens * heads * latent_dim);
    std::vector<hip_bfloat16> decode(prefill.size());
    HIP_CHECK(hipMemcpy(prefill.data(), device_prefill,
                        prefill.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(decode.data(), device_decode,
                        decode.size() * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    for (size_t index = 0; index < prefill.size(); index++) {
        CHECK(static_cast<float>(prefill[index]) ==
              static_cast<float>(decode[index]),
              "decode/prefill sparse output differs");
    }

    HIP_CHECK(hipFree(topk_workspace));
    HIP_CHECK(hipFree(device_decode));
    HIP_CHECK(hipFree(device_prefill));
    HIP_CHECK(hipFree(device_probabilities));
    HIP_CHECK(hipFree(device_attention_scores));
    HIP_CHECK(hipFree(device_ids));
    HIP_CHECK(hipFree(device_decode_scores));
    HIP_CHECK(hipFree(device_all_scores));
    HIP_CHECK(hipFree(device_cache));
    HIP_CHECK(hipFree(device_absorbed));
    HIP_CHECK(hipFree(device_weights));
    HIP_CHECK(hipFree(device_keys));
    HIP_CHECK(hipFree(device_queries));
    return 0;
}

int main(void) {
    if (test_topk() != 0 || test_indexer() != 0 ||
        test_sparse_attention() != 0 ||
        test_decode_prefill_equivalence() != 0) return 1;
    std::puts("PASS: K3.1 DSA top-k and sparse-attention ROCm primitives");
    return 0;
}
