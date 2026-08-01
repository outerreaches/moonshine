#include "k3_engine.h"

#include <hip/hip_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    const bool range_only =
        argc > 2 && strcmp(argv[2], "range-only") == 0;
    const bool kda_blas =
        argc > 3 && strcmp(argv[3], "kda-blas") == 0;
    if (argc > 4 || (argc > 2 && !range_only) ||
        (argc > 3 && !kda_blas)) {
        fprintf(stderr,
                "usage: %s [MODEL_ROOT "
                "[range-only [kda-blas]]]\n",
                argv[0]);
        return 2;
    }
    const k3_prefill_projection_backend backend =
        kda_blas ?
            K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT :
            K3_PREFILL_PROJECTION_DEFAULT;
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 prefill chunk: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    static const uint32_t tokens[2] = {
        163587u, 2778u,
    };
    char error[512];

    uint32_t sequential_next = 327u;
    float sequential_value = 15.375f;
    k3_engine_state_digest sequential_digest = {
        UINT64_C(0xc91b2d0c9fc32823),
        UINT64_C(0x52d3f2ec7a8e2a5b),
        UINT64_C(0xe66b31cf80821bed),
        UINT64_C(0x6cc9b91dcfd28df0),
        2u,
    };
    if (!range_only) {
        k3_engine *sequential = NULL;
        k3_engine_stats sequential_startup;
        CHECK(k3_engine_create(
                  &sequential, root, 8192u, 32u, 16u, true,
                  &sequential_startup, error, sizeof(error)),
              error);
        for (uint32_t token = 0u; token < 2u; token++) {
            CHECK(k3_engine_forward_token(
                      sequential, tokens[token],
                      &sequential_next, &sequential_value,
                      error, sizeof(error)),
                  error);
        }
        CHECK(k3_engine_get_state_digest(
                  sequential, &sequential_digest,
                  error, sizeof(error)),
              error);
        k3_engine_destroy(sequential);
        HIP_CHECK(hipDeviceSynchronize());
    }
    printf("  sequential oracle%s: next=%u value=%.8g "
           "kda=%016" PRIx64 " conv=%016" PRIx64
           " mla=%016" PRIx64 " attnres=%016" PRIx64 "\n",
           range_only ? " (locked)" : "",
           sequential_next, sequential_value,
           sequential_digest.kda_state_hash,
           sequential_digest.kda_conv_hash,
           sequential_digest.mla_cache_hash,
           sequential_digest.attn_res_hash);
    fflush(stdout);

    k3_engine *range = NULL;
    k3_engine_stats range_startup;
    CHECK(k3_engine_create(
              &range, root, 8192u, 32u, 16u, true,
              &range_startup, error, sizeof(error)),
          error);
    k3_prefill_plan plan;
    CHECK(k3_engine_plan_prefill_with_projection_backend(
              range, 2u, K3_PREFILL_CACHE_RETAIN, backend,
              &plan, error, sizeof(error)),
          error);
    CHECK(plan.routed_layer_sweeps == 92u &&
          plan.expert_read_requests == 82432u &&
          plan.routed_store_physical_read_bytes ==
              K3_ROUTED_PHYSICAL_BYTES_CEILING,
          "range preflight I/O ledger");

    uint32_t range_next = 0u;
    float range_value = 0.0f;
    k3_engine_prefill_stats measured;
    CHECK(k3_engine_forward_range_with_projection_backend(
              range, tokens, 2u, backend,
              &range_next, &range_value, &measured,
              error, sizeof(error)),
          error);
    k3_engine_state_digest range_digest;
    CHECK(k3_engine_get_state_digest(
              range, &range_digest,
              error, sizeof(error)),
          error);

    CHECK(range_next == sequential_next,
          "range/sequential greedy token mismatch");
    CHECK(isfinite(range_value),
          "range selected logit is not finite");
    const bool exact_value = memcmp(
        &range_value, &sequential_value,
        sizeof(range_value)) == 0;
    const bool exact_digest =
        range_digest.token_position ==
            sequential_digest.token_position &&
        range_digest.kda_state_hash ==
            sequential_digest.kda_state_hash &&
        range_digest.kda_conv_hash ==
            sequential_digest.kda_conv_hash &&
        range_digest.mla_cache_hash ==
            sequential_digest.mla_cache_hash &&
        range_digest.attn_res_hash ==
            sequential_digest.attn_res_hash;
    if (!kda_blas) {
        CHECK(exact_value,
              "range/sequential selected logit mismatch");
        CHECK(exact_digest,
              "range/sequential causal-state digest mismatch");
    } else {
        CHECK(range_value == 15.3125f,
              "experimental range selected logit changed");
        CHECK(range_digest.token_position == 2u &&
              range_digest.kda_state_hash ==
                  UINT64_C(0xe28c1c41862a80d7) &&
              range_digest.kda_conv_hash ==
                  UINT64_C(0xf7fc7d915bef168d) &&
              range_digest.mla_cache_hash ==
                  UINT64_C(0x8da9b9384d2066b7) &&
              range_digest.attn_res_hash ==
                  UINT64_C(0x46a8ec580f10a41c),
              "experimental range causal-state digest changed");
    }
    CHECK(measured.routed_layer_sweeps == 92u &&
          measured.expert_read_requests ==
              measured.unique_experts_across_layers &&
          measured.expert_read_requests >= 92u * 16u &&
          measured.expert_read_requests <= 92u * 32u &&
          measured.selected_expert_routes == 92u * 2u * 16u &&
          measured.routed_physical_read_bytes > 0u &&
          measured.routed_physical_read_bytes <
              K3_ROUTED_PHYSICAL_BYTES_CEILING,
          "range runtime I/O ledger");

    printf("K3 prefill chunk: 2-token layer-major %s: PASS\n",
           kda_blas ? "KDA dequant/BLAS comparison" : "oracle");
    printf("  next token=%u value=%.8g\n",
           range_next, range_value);
    if (kda_blas) {
        printf("  comparison: value_exact=%s delta=%.8g "
               "state_exact=%s\n",
               exact_value ? "yes" : "no",
               range_value - sequential_value,
               exact_digest ? "yes" : "no");
        printf("  candidate kernels: dequant=%.3f "
               "hipBLAS=%.3f s\n",
               measured.kda_dequantize_seconds,
               measured.kda_blas_seconds);
    }
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
    printf("  wall=%.3f s (%.4f tok/s); "
           "workspace=%.3f MiB\n",
           measured.wall_seconds,
           2.0 / measured.wall_seconds,
           plan.batch_workspace_bytes / 1048576.0);
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
    printf("  stream detail: read-wait=%.3f submit=%.3f "
           "index=%.3f expert-pipeline=%.3f "
           "accounted=%.3f other=%.3f s\n",
           measured.routed_read_wait_seconds,
           measured.routed_submit_seconds,
           measured.routed_index_seconds,
           measured.routed_expert_pipeline_seconds,
           stream_detail,
           measured.routed_stream_seconds - stream_detail);
    printf("  state hashes: kda=%016" PRIx64
           " conv=%016" PRIx64 " mla=%016" PRIx64
           " attnres=%016" PRIx64 "\n",
           range_digest.kda_state_hash,
           range_digest.kda_conv_hash,
           range_digest.mla_cache_hash,
           range_digest.attn_res_hash);
    k3_engine_destroy(range);
    printf("K3 prefill chunk: PASS\n");
    return 0;
}
