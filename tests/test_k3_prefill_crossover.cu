#include "k3_engine.h"

#include <hip/hip_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                       \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                      \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static double elapsed_seconds(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static bool digest_equal(const k3_engine_state_digest *left,
                         const k3_engine_state_digest *right) {
    return left->token_position == right->token_position &&
           left->kda_state_hash == right->kda_state_hash &&
           left->kda_conv_hash == right->kda_conv_hash &&
           left->mla_cache_hash == right->mla_cache_hash &&
           left->attn_res_hash == right->attn_res_hash;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s MODEL_ROOT TOKEN_COUNT [TOKEN_COUNT ...]\n",
                argv[0]);
        return 2;
    }
    uint32_t max_tokens = 0u;
    for (int i = 2; i < argc; i++) {
        char *end = NULL;
        const unsigned long parsed = strtoul(argv[i], &end, 10);
        if (end == NULL || *end != '\0' ||
            parsed < 2ul || parsed > 256ul) {
            fprintf(stderr, "invalid token count: %s\n", argv[i]);
            return 2;
        }
        if (parsed > max_tokens) max_tokens = (uint32_t)parsed;
    }

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 prefill crossover: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));

    static const uint32_t prompt[] = {
        163587u, 2778u, 6244u, 878u, 2482u, 1u,
        163589u, 71079u, 40493u, 13u,
        163588u, 2778u, 163589u, 163586u,
        163587u, 2778u, 6244u, 878u, 69702u, 1u,
        163589u, 163587u, 12092u, 163589u,
    };
    const uint32_t prompt_count =
        (uint32_t)(sizeof(prompt) / sizeof(prompt[0]));
    uint32_t *tokens = (uint32_t *)malloc(
        (size_t)max_tokens * sizeof(*tokens));
    CHECK(tokens != NULL, "allocating crossover tokens failed");
    for (uint32_t i = 0u; i < max_tokens; i++) {
        tokens[i] = prompt[i % prompt_count];
    }

    char error[512] = { 0 };
    k3_engine *engine = NULL;
    k3_engine_stats startup;
    CHECK(k3_engine_create(
              &engine, argv[1], 8192u, 32u, 16u, true,
              &startup, error, sizeof(error)),
          error);
    printf("K3 selected-prefill crossover (startup %.3f s)\n",
           startup.startup_seconds);
    printf("tokens sequential_s range_s speedup range_GiB unique_min/avg/max\n");
    fflush(stdout);

    for (int argument = 2; argument < argc; argument++) {
        const uint32_t token_count =
            (uint32_t)strtoul(argv[argument], NULL, 10);

        CHECK(k3_engine_reset_state(
                  engine, false, error, sizeof(error)),
              error);
        uint32_t predicted = 0u;
        float value = 0.0f;
        for (uint32_t i = 0u; i < prompt_count; i++) {
            CHECK(k3_engine_forward_token(
                      engine, prompt[i], &predicted, &value,
                      error, sizeof(error)),
                  error);
        }

        CHECK(k3_engine_reset_state(
                  engine, false, error, sizeof(error)),
              error);
        struct timespec sequential_start;
        struct timespec sequential_end;
        clock_gettime(CLOCK_MONOTONIC, &sequential_start);
        for (uint32_t i = 0u; i < token_count; i++) {
            CHECK(k3_engine_forward_token(
                      engine, tokens[i], &predicted, &value,
                      error, sizeof(error)),
                  error);
        }
        clock_gettime(CLOCK_MONOTONIC, &sequential_end);
        const uint32_t sequential_next = predicted;
        const float sequential_value = value;
        k3_engine_state_digest sequential_digest;
        CHECK(k3_engine_get_state_digest(
                  engine, &sequential_digest,
                  error, sizeof(error)),
              error);

        CHECK(k3_engine_reset_state(
                  engine, false, error, sizeof(error)),
              error);
        k3_engine_prefill_stats range_stats;
        uint32_t range_next = 0u;
        float range_value = 0.0f;
        CHECK(k3_engine_forward_range(
                  engine, tokens, token_count,
                  &range_next, &range_value, &range_stats,
                  error, sizeof(error)),
              error);
        k3_engine_state_digest range_digest;
        CHECK(k3_engine_get_state_digest(
                  engine, &range_digest,
                  error, sizeof(error)),
              error);

        if (token_count >= 4u) {
            CHECK(k3_engine_reset_state(
                      engine, false, error, sizeof(error)),
                  error);
            const uint32_t first_chunk = token_count - 2u;
            uint32_t split_next = 0u;
            float split_value = 0.0f;
            k3_engine_prefill_stats split_stats;
            CHECK(k3_engine_forward_range(
                      engine, tokens, first_chunk,
                      &split_next, &split_value, &split_stats,
                      error, sizeof(error)),
                  error);
            CHECK(k3_engine_forward_range(
                      engine, tokens + first_chunk, 2u,
                      &split_next, &split_value, &split_stats,
                      error, sizeof(error)),
                  error);
            k3_engine_state_digest split_digest;
            CHECK(k3_engine_get_state_digest(
                      engine, &split_digest,
                      error, sizeof(error)),
                  error);
            CHECK(split_next == range_next,
                  "split-range greedy token mismatch");
            CHECK(memcmp(
                      &split_value, &range_value,
                      sizeof(split_value)) == 0,
                  "split-range selected value mismatch");
            CHECK(digest_equal(&split_digest, &range_digest),
                  "split-range causal-state digest mismatch");
        }

        CHECK(range_next == sequential_next,
              "crossover greedy token mismatch");
        CHECK(memcmp(
                  &range_value, &sequential_value,
                  sizeof(range_value)) == 0,
              "crossover selected value mismatch");
        CHECK(digest_equal(&range_digest, &sequential_digest),
              "crossover causal-state digest mismatch");
        CHECK(range_stats.expert_read_requests ==
                  range_stats.unique_experts_across_layers &&
              range_stats.routed_layer_sweeps == 92u &&
              range_stats.selected_expert_routes ==
                  (uint64_t)token_count * 16u * 92u,
              "crossover selected-read ledger mismatch");

        const double sequential_seconds =
            elapsed_seconds(sequential_start, sequential_end);
        const double average_unique =
            (double)range_stats.unique_experts_across_layers /
            range_stats.routed_layer_sweeps;
        printf("%u %.3f %.3f %.3f %.3f %u/%.1f/%u\n",
               token_count,
               sequential_seconds,
               range_stats.wall_seconds,
               sequential_seconds / range_stats.wall_seconds,
               (double)range_stats.routed_physical_read_bytes /
                   (1024.0 * 1024.0 * 1024.0),
               range_stats.min_unique_experts_per_layer,
               average_unique,
               range_stats.max_unique_experts_per_layer);
        fflush(stdout);
    }

    k3_engine_destroy(engine);
    free(tokens);
    printf("K3 selected-prefill crossover: PASS\n");
    return 0;
}
