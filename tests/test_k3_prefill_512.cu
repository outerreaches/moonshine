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

static const uint64_t K3_ROUTED_PHYSICAL_BYTES_CEILING =
    UINT64_C(1446793422960);

typedef struct {
    struct timespec start;
} scale_progress;

static void report_progress(uint32_t completed,
                            uint32_t total,
                            void *user_data) {
    if (completed != 1u && completed != total && completed % 8u != 0u) {
        return;
    }
    const scale_progress *progress = (const scale_progress *)user_data;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    const double elapsed =
        (double)(now.tv_sec - progress->start.tv_sec) +
        (double)(now.tv_nsec - progress->start.tv_nsec) / 1e9;
    printf("  progress: layer %u/%u, elapsed %.1f s\n",
           completed, total, elapsed);
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    unsigned long parsed_tokens = 512ul;
    if (argc > 2) {
        char *end = NULL;
        parsed_tokens = strtoul(argv[2], &end, 10);
        if (!end || *end != '\0' ||
            parsed_tokens < 2ul || parsed_tokens > 32768ul) {
            fprintf(stderr, "invalid token count: %s\n", argv[2]);
            return 2;
        }
    }
    unsigned long parsed_context =
        parsed_tokens > 8192ul ? parsed_tokens : 8192ul;
    bool kda_blas = false;
    if (argc > 3) {
        if (strcmp(argv[3], "kda-blas") == 0) {
            kda_blas = true;
        } else {
            char *end = NULL;
            parsed_context = strtoul(argv[3], &end, 10);
            if (!end || *end != '\0' ||
                parsed_context < parsed_tokens ||
                parsed_context > 131072ul) {
                fprintf(stderr, "invalid context: %s\n", argv[3]);
                return 2;
            }
        }
    }
    if (argc > 4 && strcmp(argv[4], "kda-blas") == 0) {
        kda_blas = true;
    }
    if (argc > 5 ||
        (argc > 4 && strcmp(argv[4], "kda-blas") != 0)) {
        fprintf(stderr,
                "usage: %s [MODEL_ROOT [TOKENS [CONTEXT [kda-blas]]]]\n"
                "       %s [MODEL_ROOT [TOKENS [kda-blas]]]\n",
                argv[0],
                argv[0]);
        return 2;
    }
    const k3_prefill_projection_backend backend =
        kda_blas ?
            K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT :
            K3_PREFILL_PROJECTION_DEFAULT;
    const uint32_t token_count = (uint32_t)parsed_tokens;
    const uint32_t context = (uint32_t)parsed_context;
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 prefill 512: SKIP (no ROCm device)\n");
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
    uint32_t *tokens =
        (uint32_t *)malloc(
            (size_t)token_count * sizeof(uint32_t));
    CHECK(tokens != NULL, "scale-token host allocation");
    for (uint32_t index = 0u; index < token_count; index++) {
        tokens[index] =
            prompt[index %
                (sizeof(prompt) / sizeof(prompt[0]))];
    }

    char error[512];
    k3_engine *engine = NULL;
    k3_engine_stats startup;
    CHECK(k3_engine_create(
              &engine, root, context, 32u, 16u, true,
              &startup, error, sizeof(error)),
          error);
    k3_prefill_plan plan;
    CHECK(k3_engine_plan_prefill_with_projection_backend(
              engine, token_count,
              K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE,
              backend,
              &plan, error, sizeof(error)),
          error);
    CHECK(plan.batch_workspace_bytes ==
              (uint64_t)token_count * UINT64_C(1116360) +
                  (kda_blas ? UINT64_C(176160768) : 0u) &&
          plan.borrowed_cache_bytes ==
              plan.batch_workspace_bytes &&
          plan.new_device_bytes == 0u,
          "scale-token cold-workspace ledger");

    uint32_t next = 0u;
    float value = 0.0f;
    k3_engine_prefill_stats measured;
    bool forwarded = false;
    if (kda_blas) {
        forwarded = k3_engine_forward_range_with_projection_backend(
            engine, tokens, token_count, backend,
            &next, &value, &measured,
            error, sizeof(error));
    } else {
        scale_progress progress;
        clock_gettime(CLOCK_MONOTONIC, &progress.start);
        forwarded = k3_engine_forward_range_with_progress(
            engine, tokens, token_count,
            &next, &value, &measured,
            report_progress, &progress,
            error, sizeof(error));
    }
    CHECK(forwarded, error);
    CHECK(next < 163840u && isfinite(value),
          "scale-token output is invalid");
    if (token_count == 512u) {
        CHECK(next == 40493u &&
                  value ==
                      (kda_blas ? 24.75f : 24.875f),
              "512-token output changed");
    } else if (token_count == 8192u) {
        CHECK(next == 40493u &&
                  value ==
                      (kda_blas ? 27.75f : 28.125f),
              "8192-token output changed");
    } else if (token_count == 16384u && !kda_blas) {
        CHECK(next == 6244u && value == 26.875f,
              "16384-token default output changed");
    } else if (token_count == 32768u && !kda_blas) {
        CHECK(next == 40493u && value == 28.25f,
              "32768-token default output changed");
    }
    CHECK(measured.routed_layer_sweeps == 92u &&
          measured.expert_read_requests ==
              measured.unique_experts_across_layers &&
          measured.expert_read_requests > 0u &&
          measured.expert_read_requests <= 82432u &&
          measured.selected_expert_routes ==
              (uint64_t)92u * token_count * 16u &&
          measured.routed_physical_read_bytes > 0u &&
          measured.routed_physical_read_bytes <=
              K3_ROUTED_PHYSICAL_BYTES_CEILING &&
          (token_count > 512u ||
           measured.routed_physical_read_bytes <
               K3_ROUTED_PHYSICAL_BYTES_CEILING),
          "scale-token runtime I/O ledger");

    printf("K3 prefill scale: %u/%u tokens/context: PASS\n",
           token_count, context);
    printf("  next=%u value=%.8g; wall=%.3f s; %.3f tok/s\n",
           next, value, measured.wall_seconds,
           token_count / measured.wall_seconds);
    printf("  routed reads=%" PRIu64
           " bytes; sweeps=%u; requests=%u; "
           "unique=%u/%.1f/%u per layer\n",
           measured.routed_physical_read_bytes,
           measured.routed_layer_sweeps,
           measured.expert_read_requests,
           measured.min_unique_experts_per_layer,
           (double)measured.unique_experts_across_layers /
               measured.routed_layer_sweeps,
           measured.max_unique_experts_per_layer);
    printf("  workspace/cache loan=%.3f GiB; startup=%.3f s\n",
           plan.batch_workspace_bytes / 1073741824.0,
           startup.startup_seconds);
    const double phase_seconds =
        measured.layer0_seconds +
        measured.attention_seconds +
        measured.router_seconds +
        measured.routed_stream_seconds +
        measured.moe_tail_seconds +
        measured.output_seconds;
    printf("  phases: layer0=%.3f attention=%.3f "
           "(kda=%.3f mla=%.3f) router=%.3f "
           "stream=%.3f tail=%.3f output=%.3f s\n",
           measured.layer0_seconds,
           measured.attention_seconds,
           measured.kda_attention_seconds,
           measured.mla_attention_seconds,
           measured.router_seconds,
           measured.routed_stream_seconds,
           measured.moe_tail_seconds,
           measured.output_seconds);
    printf("  phase coverage=%.3f/%.3f s (%.2f%%)\n",
           phase_seconds, measured.wall_seconds,
           100.0 * phase_seconds / measured.wall_seconds);
    const double kda_detail =
        measured.kda_projection_seconds +
        measured.kda_convolution_seconds +
        measured.kda_recurrent_seconds +
        measured.kda_gate_norm_seconds +
        measured.kda_output_projection_seconds;
    const double stream_detail =
        measured.routed_read_wait_seconds +
        measured.routed_submit_seconds +
        measured.routed_index_seconds +
        measured.routed_expert_pipeline_seconds;
    printf("  KDA detail: projection=%.3f conv=%.3f "
           "recurrent=%.3f gate/norm=%.3f output=%.3f "
           "accounted=%.3f other=%.3f s\n",
           measured.kda_projection_seconds,
           measured.kda_convolution_seconds,
           measured.kda_recurrent_seconds,
           measured.kda_gate_norm_seconds,
           measured.kda_output_projection_seconds,
           kda_detail,
           measured.kda_attention_seconds - kda_detail);
    if (kda_blas) {
        printf("  candidate kernels: dequant=%.3f "
               "hipBLAS=%.3f s\n",
               measured.kda_dequantize_seconds,
               measured.kda_blas_seconds);
    }
    printf("  stream detail: read-wait=%.3f submit=%.3f "
           "index=%.3f expert-pipeline=%.3f "
           "accounted=%.3f other=%.3f s\n",
           measured.routed_read_wait_seconds,
           measured.routed_submit_seconds,
           measured.routed_index_seconds,
           measured.routed_expert_pipeline_seconds,
           stream_detail,
           measured.routed_stream_seconds - stream_detail);
    k3_engine_destroy(engine);
    free(tokens);
    return 0;
}
