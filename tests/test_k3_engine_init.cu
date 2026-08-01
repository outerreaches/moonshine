#include "k3_engine.h"

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

static const uint64_t K3_EXPECTED_STATIC_BYTES =
    UINT64_C(59345729536);
static const uint64_t K3_EXPECTED_CACHE_BYTES =
    UINT64_C(51659145216);
static const uint64_t K3_FIXED_STATE_BYTES =
    UINT64_C(756547584);
static const uint64_t K3_CONTEXT_STATE_BYTES_PER_TOKEN =
    UINT64_C(28224);
static const uint64_t K3_EXPECTED_STAGING_BYTES =
    UINT64_C(280821760);

static uint32_t rng_state = UINT32_C(0x4b334530);

static float random_input(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & UINT32_C(0x00ffffff)) /
            (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * 0.125f;
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const uint8_t *values = (const uint8_t *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        hash ^= values[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool expected_hash(const char *label,
                          uint64_t got,
                          uint64_t expected) {
    if (got == expected) return true;
    fprintf(
        stderr,
        "FAIL: %s hash got 0x%016llx, expected 0x%016llx\n",
        label,
        (unsigned long long)got,
        (unsigned long long)expected);
    return false;
}

int main(int argc, char **argv) {
    if (argc > 3) {
        fprintf(stderr, "usage: %s [MODEL_ROOT [CONTEXT]]\n", argv[0]);
        return 2;
    }
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    uint32_t context = 8192u;
    if (argc > 2) {
        char *end = NULL;
        const unsigned long parsed = strtoul(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' ||
            parsed < 64ul || parsed > 1048576ul) {
            fprintf(stderr, "invalid context: %s\n", argv[2]);
            return 2;
        }
        context = (uint32_t)parsed;
    }
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 engine init: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 engine init on %s (%s), context=%u\n",
           properties.name, properties.gcnArchName, context);

    char error[512];
    k3_engine *engine = NULL;
    k3_engine_stats stats;
    CHECK(k3_engine_create(
              &engine, root, context, 32u, 16u, true,
              &stats, error, sizeof(error)),
          error);
    CHECK(stats.static_store.resident_bytes ==
              K3_EXPECTED_STATIC_BYTES,
          "engine static tier byte ledger");
    CHECK(stats.cache_bytes == K3_EXPECTED_CACHE_BYTES &&
              stats.cache_slots == 2944u,
          "engine cache byte ledger");
    const uint64_t expected_state_bytes =
        K3_FIXED_STATE_BYTES +
        (uint64_t)context * K3_CONTEXT_STATE_BYTES_PER_TOKEN;
    CHECK(stats.state_bytes == expected_state_bytes,
          "engine attention-state byte ledger");
    CHECK(stats.staging_bytes == K3_EXPECTED_STAGING_BYTES &&
              stats.staging_slots == 16u,
          "engine staging byte ledger");
    CHECK(k3_engine_find_weight(
              engine,
              "language_model.model.layers.0."
              "self_attn.q_proj.weight") != NULL &&
          k3_engine_find_weight(
              engine,
              "language_model.lm_head.weight") != NULL,
          "engine static-weight lookup");
    CHECK(k3_engine_cache_slot(engine, 0u) != NULL &&
          k3_engine_cache_slot(engine, 2943u) != NULL &&
          k3_engine_cache_slot(engine, 2944u) == NULL,
          "engine cache-slot bounds");
    CHECK(k3_engine_staging_host(engine, 0u) != NULL &&
          k3_engine_staging_device(engine, 15u) != NULL &&
          k3_engine_staging_host(engine, 16u) == NULL,
          "engine staging-slot bounds");

    const uint32_t hidden = 7168u;
    const size_t hidden_bytes =
        (size_t)hidden * sizeof(hip_bfloat16);
    hip_bfloat16 *input =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *output =
        (hip_bfloat16 *)malloc(hidden_bytes);
    CHECK(input && output, "engine layer-0 host allocation");
    for (uint32_t i = 0; i < hidden; i++) {
        input[i] = hip_bfloat16(random_input());
    }
    void *d_input = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_input, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_output, hidden_bytes));
    HIP_CHECK(hipMemcpy(
        d_input, input, hidden_bytes,
        hipMemcpyHostToDevice));
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    CHECK(k3_engine_decode_layer0(
              engine, d_input, d_output,
              error, sizeof(error)),
          error);
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float layer0_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(
        &layer0_ms, start, stop));
    HIP_CHECK(hipMemcpy(
        output, d_output, hidden_bytes,
        hipMemcpyDeviceToHost));
    bool nonzero = false;
    for (uint32_t i = 0; i < hidden; i++) {
        const float value = (float)output[i];
        CHECK(isfinite(value),
              "engine layer-0 produced non-finite output");
        if (value != 0.0f) nonzero = true;
    }
    CHECK(nonzero, "engine layer-0 output is zero");
    const uint64_t layer0_hash =
        fnv1a64(output, hidden_bytes);
    CHECK(expected_hash(
              "engine layer-0 output", layer0_hash,
              UINT64_C(0x490bddfe54cbae44)),
          "engine layer-0 output hash changed");

    HIP_CHECK(hipEventRecord(start, NULL));
    CHECK(k3_engine_decode_layer1(
              engine, d_output, d_input,
              error, sizeof(error)),
          error);
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float layer1_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(
        &layer1_ms, start, stop));
    HIP_CHECK(hipMemcpy(
        output, d_input, hidden_bytes,
        hipMemcpyDeviceToHost));
    nonzero = false;
    for (uint32_t i = 0; i < hidden; i++) {
        const float value = (float)output[i];
        CHECK(isfinite(value),
              "engine layer-1 produced non-finite output");
        if (value != 0.0f) nonzero = true;
    }
    CHECK(nonzero, "engine layer-1 output is zero");
    const uint64_t layer1_hash =
        fnv1a64(output, hidden_bytes);
    /*
     * These routed hashes and the greedy score below are scoped to the
     * accepted group-vectorized MXFP4 reduction introduced by native commit
     * f68aa08. Its scalar parent 9a237c2 has a different, valid FP32 addition
     * order; see docs/qualification-128k.md.
     */
    bool routed_hashes_ok = expected_hash(
        "engine layer-1 output", layer1_hash,
        UINT64_C(0x4a33492ac1238cca));

    float next_layer_ms[91] = { 0.0f };
    uint64_t next_layer_hash[91] = { 0u };
    void *next_input = d_input;
    void *next_output = d_output;
    for (uint32_t layer = 2u; layer <= 92u; layer++) {
        HIP_CHECK(hipEventRecord(start, NULL));
        CHECK(k3_engine_decode_next_layer(
                  engine, next_input, next_output,
                  error, sizeof(error)),
              error);
        HIP_CHECK(hipEventRecord(stop, NULL));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(
            &next_layer_ms[layer - 2u], start, stop));
        HIP_CHECK(hipMemcpy(
            output, next_output, hidden_bytes,
            hipMemcpyDeviceToHost));
        nonzero = false;
        for (uint32_t i = 0; i < hidden; i++) {
            const float value = (float)output[i];
            CHECK(isfinite(value),
                  "engine routed layer produced non-finite output");
            if (value != 0.0f) nonzero = true;
        }
        CHECK(nonzero, "engine routed-layer output is zero");
        next_layer_hash[layer - 2u] =
            fnv1a64(output, hidden_bytes);
        void *swap = next_input;
        next_input = next_output;
        next_output = swap;
    }
    routed_hashes_ok &= expected_hash(
        "engine layer-2 output", next_layer_hash[0],
        UINT64_C(0x404ae49a560abd6d));
    routed_hashes_ok &= expected_hash(
        "engine layer-3 output", next_layer_hash[1],
        UINT64_C(0x20c3ea370fe7830a));
    routed_hashes_ok &= expected_hash(
        "engine layer-12 output", next_layer_hash[10],
        UINT64_C(0x272508823475f6da));
    routed_hashes_ok &= expected_hash(
        "engine layer-92 output", next_layer_hash[90],
        UINT64_C(0x8a266bfc106e5d51));
    CHECK(routed_hashes_ok,
          "engine routed-layer output hash changed");
    double remaining_ms = 0.0;
    double kda_routed_ms = 0.0;
    double mla_routed_ms = 0.0;
    uint32_t kda_routed_layers = 0u;
    uint32_t mla_routed_layers = 0u;
    for (uint32_t layer = 2u; layer <= 92u; layer++) {
        const float elapsed = next_layer_ms[layer - 2u];
        remaining_ms += elapsed;
        if (layer % 4u == 3u || layer == 92u) {
            mla_routed_ms += elapsed;
            mla_routed_layers++;
        } else {
            kda_routed_ms += elapsed;
            kda_routed_layers++;
        }
    }
    uint32_t greedy_token = 0u;
    float greedy_value = 0.0f;
    HIP_CHECK(hipEventRecord(start, NULL));
    CHECK(k3_engine_decode_greedy(
              engine, next_input, &greedy_token, &greedy_value,
              error, sizeof(error)),
          error);
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float output_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(
        &output_ms, start, stop));
    CHECK(greedy_token < 163840u && isfinite(greedy_value),
          "engine greedy output is invalid");
    if (greedy_token != 220u || greedy_value != 6.875f) {
        fprintf(stderr,
                "FAIL: engine first greedy output got token %u "
                "value %.8f, expected token 220 value 6.87500000\n",
                greedy_token, greedy_value);
        CHECK(false, "engine first greedy output changed");
    }
    k3_engine_cache_stats first_cache_stats;
    k3_engine_get_cache_stats(engine, &first_cache_stats);
    CHECK(first_cache_stats.batches == 92u &&
          first_cache_stats.accesses == 92u * 16u &&
          first_cache_stats.hits == 0u &&
          first_cache_stats.misses == 92u * 16u &&
          first_cache_stats.admissions == 92u * 16u,
          "engine cold-token cache telemetry mismatch");

    CHECK(k3_engine_load_embedding(
              engine, 42u, next_output,
              error, sizeof(error)),
          error);
    HIP_CHECK(hipMemcpy(
        output, next_output, hidden_bytes,
        hipMemcpyDeviceToHost));
    const uint64_t embedding_hash =
        fnv1a64(output, hidden_bytes);
    CHECK(embedding_hash == UINT64_C(0x1c863517bc2e2a4a),
          "engine streamed embedding hash changed");

    HIP_CHECK(hipEventRecord(start, NULL));
    CHECK(k3_engine_decode_layer0(
              engine, d_input, d_output,
              error, sizeof(error)),
          error);
    CHECK(k3_engine_decode_layer1(
              engine, d_output, d_input,
              error, sizeof(error)),
          error);
    void *second_input = d_input;
    void *second_output = d_output;
    for (uint32_t layer = 2u; layer <= 92u; layer++) {
        CHECK(k3_engine_decode_next_layer(
                  engine, second_input, second_output,
                  error, sizeof(error)),
              error);
        void *swap = second_input;
        second_input = second_output;
        second_output = swap;
    }
    uint32_t second_token = 0u;
    float second_value = 0.0f;
    CHECK(k3_engine_decode_greedy(
              engine, second_input, &second_token, &second_value,
              error, sizeof(error)),
          error);
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float second_token_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(
        &second_token_ms, start, stop));
    CHECK(second_token < 163840u && isfinite(second_value),
          "engine second greedy output is invalid");
    k3_engine_cache_stats second_cache_stats;
    k3_engine_get_cache_stats(engine, &second_cache_stats);
    const uint64_t second_accesses =
        second_cache_stats.accesses - first_cache_stats.accesses;
    const uint64_t second_hits =
        second_cache_stats.hits - first_cache_stats.hits;
    CHECK(second_accesses == 92u * 16u,
          "engine second-token cache telemetry mismatch");
    float continuation_ms[3] = { 0.0f };
    uint32_t continuation_token[3] = { 0u };
    float continuation_value[3] = { 0.0f };
    uint64_t continuation_hits[3] = { 0u };
    uint64_t previous_hits = second_cache_stats.hits;
    uint64_t previous_accesses = second_cache_stats.accesses;
    uint32_t continuation_input_token = second_token;
    for (uint32_t step = 0u; step < 3u; step++) {
        CHECK(k3_engine_load_embedding(
                  engine, continuation_input_token, d_input,
                  error, sizeof(error)),
              error);
        HIP_CHECK(hipEventRecord(start, NULL));
        CHECK(k3_engine_decode_layer0(
                  engine, d_input, d_output,
                  error, sizeof(error)),
              error);
        CHECK(k3_engine_decode_layer1(
                  engine, d_output, d_input,
                  error, sizeof(error)),
              error);
        void *continuation_input = d_input;
        void *continuation_output = d_output;
        for (uint32_t layer = 2u; layer <= 92u; layer++) {
            CHECK(k3_engine_decode_next_layer(
                      engine, continuation_input,
                      continuation_output,
                      error, sizeof(error)),
                  error);
            void *swap = continuation_input;
            continuation_input = continuation_output;
            continuation_output = swap;
        }
        CHECK(k3_engine_decode_greedy(
                  engine, continuation_input,
                  &continuation_token[step],
                  &continuation_value[step],
                  error, sizeof(error)),
              error);
        HIP_CHECK(hipEventRecord(stop, NULL));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(
            &continuation_ms[step], start, stop));
        k3_engine_cache_stats continuation_stats;
        k3_engine_get_cache_stats(
            engine, &continuation_stats);
        CHECK(continuation_stats.accesses - previous_accesses ==
                  92u * 16u,
              "engine continuation cache telemetry mismatch");
        continuation_hits[step] =
            continuation_stats.hits - previous_hits;
        previous_hits = continuation_stats.hits;
        previous_accesses = continuation_stats.accesses;
        continuation_input_token = continuation_token[step];
    }

    printf("  static %.3f GiB; cache %.3f GiB; "
           "state %.3f GiB; staging %.3f GiB\n",
           stats.static_store.resident_bytes / 1073741824.0,
           stats.cache_bytes / 1073741824.0,
           stats.state_bytes / 1073741824.0,
           stats.staging_bytes / 1073741824.0);
    printf("  startup %.3f s "
           "(static read %.3f s, upload/Q8 %.3f s, "
           "MLA pack %.3f s)\n",
           stats.startup_seconds,
           stats.static_store.read_seconds,
           stats.static_store.upload_quantize_seconds,
           stats.mla_pack_seconds);
    printf("  complete Q8 layer 0: %.3f ms, "
           "hash 0x%016llx\n",
           layer0_ms,
           (unsigned long long)layer0_hash);
    printf("  complete Q8 streamed layer 1: %.3f ms, "
           "hash 0x%016llx\n",
           layer1_ms,
           (unsigned long long)layer1_hash);
    printf("  complete Q8 streamed layer 2 KDA: %.3f ms, "
           "hash 0x%016llx\n",
           next_layer_ms[0],
           (unsigned long long)next_layer_hash[0]);
    printf("  complete Q8 streamed layer 3 MLA: %.3f ms, "
           "hash 0x%016llx\n",
           next_layer_ms[1],
           (unsigned long long)next_layer_hash[1]);
    printf("  layer 12 boundary: %.3f ms, "
           "hash 0x%016llx\n",
           next_layer_ms[10],
           (unsigned long long)next_layer_hash[10]);
    printf("  layer 92 final MLA: %.3f ms, "
           "hash 0x%016llx\n",
           next_layer_ms[90],
           (unsigned long long)next_layer_hash[90]);
    printf("  cold routed layers 2..92: %.3f s "
           "(KDA %.3f ms/layer, MLA %.3f ms/layer)\n",
           remaining_ms / 1000.0,
           kda_routed_ms / kda_routed_layers,
           mla_routed_ms / mla_routed_layers);
    printf("  final AttnRes + norm + head: %.3f ms, "
           "greedy token %u value %.6f\n",
           output_ms, greedy_token, greedy_value);
    printf("  streamed embedding token 42: "
           "hash 0x%016llx\n",
           (unsigned long long)embedding_hash);
    printf("  second token: %.3f s, greedy token %u "
           "value %.6f, cache hits %llu/%llu (%.2f%%)\n",
           second_token_ms / 1000.0,
           second_token, second_value,
           (unsigned long long)second_hits,
           (unsigned long long)second_accesses,
           100.0 * (double)second_hits /
               (double)second_accesses);
    for (uint32_t step = 0u; step < 3u; step++) {
        printf("  continuation token %u: %.3f s, "
               "greedy token %u value %.6f, "
               "cache hits %llu/%u (%.2f%%)\n",
               step + 3u, continuation_ms[step] / 1000.0,
               continuation_token[step],
               continuation_value[step],
               (unsigned long long)continuation_hits[step],
               92u * 16u,
               100.0 * (double)continuation_hits[step] /
                   (double)(92u * 16u));
    }
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_input));
    free(output);
    free(input);
    k3_engine_destroy(engine);
    printf("K3 engine init: PASS\n");
    return 0;
}
