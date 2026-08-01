#include "k3_engine.h"

#include <hip/hip_runtime.h>

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

static double elapsed_seconds(struct timespec start,
                              struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    bool q8_projections = true;
    if (argc > 2) {
        if (strcmp(argv[2], "q8") == 0) {
            q8_projections = true;
        } else if (strcmp(argv[2], "bf16") == 0) {
            q8_projections = false;
        } else {
            fprintf(
                stderr,
                "usage: %s [MODEL_ROOT [q8|bf16 "
                "[EXPERTS_PER_LAYER [CONTEXT]]]]\n",
                argv[0]);
            return 2;
        }
    }
    unsigned long parsed_experts =
        q8_projections ? 32ul : 8ul;
    if (argc > 3) {
        char *end = NULL;
        parsed_experts = strtoul(argv[3], &end, 10);
        if (!end || *end != '\0' ||
            parsed_experts == 0ul || parsed_experts > 384ul) {
            fprintf(stderr, "invalid experts/layer: %s\n", argv[3]);
            return 2;
        }
    }
    const uint16_t experts_per_layer =
        (uint16_t)parsed_experts;
    uint32_t context = 8192u;
    if (argc > 4) {
        char *end = NULL;
        const unsigned long parsed = strtoul(argv[4], &end, 10);
        if (!end || end == argv[4] || *end != '\0' ||
            parsed < 64ul || parsed > 1048576ul) {
            fprintf(stderr, "invalid context: %s\n", argv[4]);
            return 2;
        }
        context = (uint32_t)parsed;
    }
    if (argc > 5) {
        fprintf(
            stderr,
            "usage: %s [MODEL_ROOT [q8|bf16 "
            "[EXPERTS_PER_LAYER [CONTEXT]]]]\n",
            argv[0]);
        return 2;
    }
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 hello: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 hello on %s (%s); context=%u; static=%s; "
           "experts/layer=%u\n",
           properties.name, properties.gcnArchName,
           context,
           q8_projections ? "q8" : "bf16",
           experts_per_layer);

    char error[512];
    k3_engine *engine = NULL;
    k3_engine_stats stats;
    CHECK(k3_engine_create(
              &engine, root, context, experts_per_layer, 16u,
              q8_projections,
              &stats, error, sizeof(error)),
          error);

    /*
     * Official non-thinking XTML rendering for:
     *   user: Say hello.
     * followed by an open assistant response. These IDs are generated from
     * the checkpoint's own tokenizer_config.json, tiktoken.model, and
     * encoding_k3.py.
     */
    static const uint32_t prompt[] = {
        163587u, 2778u, 6244u, 878u, 2482u, 1u,
        163589u, 71079u, 40493u, 13u,
        163588u, 2778u, 163589u, 163586u,
        163587u, 2778u, 6244u, 878u, 69702u, 1u,
        163589u, 163587u, 12092u, 163589u,
    };
    const uint32_t prompt_count =
        (uint32_t)(sizeof(prompt) / sizeof(prompt[0]));
    uint32_t predicted = 0u;
    float predicted_value = 0.0f;
    struct timespec prompt_start;
    struct timespec prompt_end;
    clock_gettime(CLOCK_MONOTONIC, &prompt_start);
    for (uint32_t i = 0; i < prompt_count; i++) {
        CHECK(k3_engine_forward_token(
                  engine, prompt[i], &predicted, &predicted_value,
                  error, sizeof(error)),
              error);
        CHECK(predicted < 163840u && isfinite(predicted_value),
              "invalid prompt-forward output");
    }
    clock_gettime(CLOCK_MONOTONIC, &prompt_end);

    enum { MAX_GENERATED = 20 };
    uint32_t generated[MAX_GENERATED];
    float values[MAX_GENERATED];
    uint32_t generated_count = 0u;
    struct timespec decode_start;
    struct timespec decode_end;
    clock_gettime(CLOCK_MONOTONIC, &decode_start);
    generated[generated_count] = predicted;
    values[generated_count++] = predicted_value;
    while (generated_count < MAX_GENERATED &&
           predicted != 163586u) {
        const uint32_t input = predicted;
        CHECK(k3_engine_forward_token(
                  engine, input, &predicted, &predicted_value,
                  error, sizeof(error)),
              error);
        CHECK(predicted < 163840u && isfinite(predicted_value),
              "invalid generated-token output");
        generated[generated_count] = predicted;
        values[generated_count++] = predicted_value;
    }
    clock_gettime(CLOCK_MONOTONIC, &decode_end);
    static const uint32_t expected_generated[] = {
        19180u, 0u, 130732u, 233u, 3653u, 691u,
        374u, 1833u, 398u, 4245u, 30u,
        163588u, 12092u, 163589u,
        163588u, 2778u, 163589u, 163586u,
    };
    if (q8_projections) {
        CHECK(
            generated_count ==
                sizeof(expected_generated) /
                    sizeof(expected_generated[0]) &&
            memcmp(
                generated, expected_generated,
                sizeof(expected_generated)) == 0,
            "configured-8K Q8 hello token sequence changed");
    }

    k3_engine_cache_stats cache;
    k3_engine_get_cache_stats(engine, &cache);
    const double prompt_seconds =
        elapsed_seconds(prompt_start, prompt_end);
    const double decode_seconds =
        elapsed_seconds(decode_start, decode_end);
    printf("  configured context: %u; prompt tokens: %u\n",
           context, prompt_count);
    printf("  static mode: %s; experts/layer: %u; "
           "resident static: %.3f GiB; cache: %.3f GiB\n",
           q8_projections ? "q8" : "bf16",
           experts_per_layer,
           (double)stats.static_store.resident_bytes /
               (1024.0 * 1024.0 * 1024.0),
           (double)stats.cache_bytes /
               (1024.0 * 1024.0 * 1024.0));
    printf("  startup %.3f s; prompt %.3f s (%.3f tok/s)\n",
           stats.startup_seconds, prompt_seconds,
           prompt_count / prompt_seconds);
    printf("  generated IDs:");
    for (uint32_t i = 0; i < generated_count; i++) {
        printf(" %u", generated[i]);
    }
    printf("\n  selected values:");
    for (uint32_t i = 0; i < generated_count; i++) {
        printf(" %.6f", values[i]);
    }
    printf("\n");
    const uint32_t decode_steps =
        generated_count > 0u ? generated_count - 1u : 0u;
    printf("  decode %.3f s for %u post-TTFT steps "
           "(%.3f tok/s)\n",
           decode_seconds, decode_steps,
           decode_steps / decode_seconds);
    printf("  cumulative expert-cache hits: %llu/%llu (%.2f%%)\n",
           (unsigned long long)cache.hits,
           (unsigned long long)cache.accesses,
           cache.accesses ?
               100.0 * (double)cache.hits /
                   (double)cache.accesses :
               0.0);
    k3_engine_destroy(engine);
    printf("K3 hello: PASS\n");
    return 0;
}
