#include "k3_prefill.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static const uint64_t K3_EXPERT_BYTES = UINT64_C(17547264);
static const uint64_t K3_LAYER_BYTES = UINT64_C(15722348544);
static const uint64_t K3_ROUTED_BYTES = UINT64_C(1446456066048);
static const uint64_t K3_REQUESTED_READ_BYTES =
    UINT64_C(1446793707520);
static const uint64_t K3_PHYSICAL_READ_BYTES =
    UINT64_C(1446793422960);
static const uint32_t K3_PHYSICAL_LAYER_SEGMENTS = 92u;
static const uint64_t K3_CACHE_32_BYTES = UINT64_C(51659145216);
static const uint64_t K3_ONE_TOKEN_WORKSPACE = UINT64_C(1116360);
static const uint64_t K3_KDA_DEQUANT_WORKSPACE =
    UINT64_C(176160768);

static int check_chunk(
        const k3_st_model *model,
        uint32_t chunk,
        k3_prefill_cache_lease lease,
        uint64_t expected_physical,
        uint32_t expected_segments) {
    char error[512];
    k3_prefill_plan plan;
    const uint32_t context = chunk > 8192u ? chunk : 8192u;
    CHECK(k3_prefill_plan_build(
              model, chunk, context, 32u, 16u, lease,
              &plan, error, sizeof(error)),
          error);
    CHECK(plan.chunk_tokens == chunk,
          "chunk token count");
    CHECK(plan.routed_layers == 92u &&
          plan.routed_layer_sweeps == 92u,
          "exactly one sweep per routed layer");
    CHECK(plan.expert_read_requests == 92u * 896u,
          "one physical request per expert");
    CHECK(plan.shard_segments == expected_segments,
          "physical shard-segment count");
    CHECK(plan.routed_layer_payload_bytes == K3_LAYER_BYTES,
          "routed-layer payload bytes");
    CHECK(plan.routed_store_payload_bytes == K3_ROUTED_BYTES,
          "routed-store payload bytes");
    CHECK(plan.routed_store_physical_read_bytes ==
              expected_physical,
          "routed-store physical read bytes");
    CHECK(plan.routed_store_requested_read_bytes ==
              K3_REQUESTED_READ_BYTES,
          "routed-store requested read bytes");
    CHECK(plan.decode_cache_bytes == K3_CACHE_32_BYTES,
          "Q8/32 decode-cache bytes");
    CHECK(plan.batch_workspace_bytes ==
              K3_ONE_TOKEN_WORKSPACE * chunk,
          "batch workspace scales exactly by chunk");
    CHECK(plan.auxiliary_workspace_bytes == 0u,
          "default plan has no auxiliary workspace");
    CHECK(plan.incremental_workspace_bytes ==
              K3_ONE_TOKEN_WORKSPACE * (chunk - 1u),
          "incremental workspace excludes existing token row");
    CHECK(plan.mla_append_bytes ==
              (uint64_t)chunk * 24u * 576u * sizeof(uint16_t),
          "MLA append ledger");
    CHECK(plan.staging_window_bytes ==
              UINT64_C(2) * UINT64_C(17551360) &&
          plan.registered_staging_bytes ==
              UINT64_C(16) * UINT64_C(17551360),
          "fixed staging window ledger");
    CHECK(plan.new_layer_buffer_bytes == 0u,
          "no hidden layer-buffer allocation");

    if (lease == K3_PREFILL_CACHE_RETAIN) {
        CHECK(plan.retained_cache_bytes ==
                  K3_CACHE_32_BYTES &&
              plan.borrowed_cache_bytes == 0u &&
              plan.new_device_bytes ==
                  plan.batch_workspace_bytes,
              "warm cache retained exactly");
    } else if (lease ==
               K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE) {
        CHECK(plan.borrowed_cache_bytes ==
                  plan.batch_workspace_bytes &&
              plan.retained_cache_bytes ==
                  K3_CACHE_32_BYTES -
                      plan.batch_workspace_bytes &&
              plan.new_device_bytes == 0u,
              "cold workspace lease is bounded exactly");
    } else {
        const uint64_t expected_borrowed =
            ((plan.batch_workspace_bytes + K3_EXPERT_BYTES - 1u) /
             K3_EXPERT_BYTES) * K3_EXPERT_BYTES;
        CHECK(plan.borrowed_cache_bytes == expected_borrowed &&
              plan.retained_cache_bytes ==
                  K3_CACHE_32_BYTES - expected_borrowed &&
              plan.new_device_bytes == 0u,
              "warm workspace lease is slot aligned");
    }
    printf("  chunk=%u lease=%s workspace=%.3f GiB "
           "MLA-append=%.3f MiB\n",
           chunk,
           lease == K3_PREFILL_CACHE_RETAIN ? "retain" :
               (lease == K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE ?
                    "borrow-cold-workspace" :
                    "borrow-warm-workspace"),
           plan.batch_workspace_bytes / 1073741824.0,
           plan.mla_append_bytes / 1048576.0);
    return 0;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    if (argc > 2) {
        fprintf(stderr, "usage: %s [MODEL_ROOT]\n", argv[0]);
        return 2;
    }
    char error[512];
    k3_st_model model;
    CHECK(k3_st_model_open(
              &model, root, 96u, error, sizeof(error)),
          error);

    k3_prefill_plan oracle;
    CHECK(k3_prefill_plan_build(
              &model, 2u, 8192u, 32u, 16u,
              K3_PREFILL_CACHE_RETAIN,
              &oracle, error, sizeof(error)),
          error);
    CHECK(oracle.routed_store_physical_read_bytes ==
              K3_PHYSICAL_READ_BYTES,
          "locked physical-read oracle");
    /*
     * Each layer is sorted by physical (shard, offset) before this count.
     * Lock the result so execution cannot silently regress to numeric-expert
     * order or add a second pass.
     */
    CHECK(oracle.shard_segments == K3_PHYSICAL_LAYER_SEGMENTS,
          "one contiguous physical segment per routed layer");
    k3_prefill_plan larger_context;
    CHECK(k3_prefill_plan_build(
              &model, 2u, 32768u, 32u, 16u,
              K3_PREFILL_CACHE_RETAIN,
              &larger_context, error, sizeof(error)),
          "32K context plan");
    CHECK(larger_context.batch_workspace_bytes ==
              oracle.batch_workspace_bytes &&
          larger_context.context_remaining == 32768u,
          "32K context preserves prompt workspace accounting");
    CHECK(k3_prefill_plan_build(
              &model, 2u, 131072u, 32u, 16u,
              K3_PREFILL_CACHE_RETAIN,
              &larger_context, error, sizeof(error)),
          "128K context plan");
    CHECK(larger_context.batch_workspace_bytes ==
              oracle.batch_workspace_bytes &&
          larger_context.context_remaining == 131072u,
          "128K context preserves prompt workspace accounting");
    CHECK(!k3_prefill_plan_build(
              &model, 2u, 1048577u, 32u, 16u,
              K3_PREFILL_CACHE_RETAIN,
              &larger_context, error, sizeof(error)),
          "context beyond the model limit was accepted");

    static const uint32_t chunks[] = {
        2u, 8u, 512u, 8192u, 16384u, 32768u,
    };
    for (size_t index = 0;
         index < sizeof(chunks) / sizeof(chunks[0]); index++) {
        if (check_chunk(
                &model, chunks[index],
                K3_PREFILL_CACHE_RETAIN,
                K3_PHYSICAL_READ_BYTES,
                K3_PHYSICAL_LAYER_SEGMENTS) != 0) {
            k3_st_model_close(&model);
            return 1;
        }
    }
    if (check_chunk(
            &model, 8u,
            K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE,
            K3_PHYSICAL_READ_BYTES,
            K3_PHYSICAL_LAYER_SEGMENTS) != 0) {
        k3_st_model_close(&model);
        return 1;
    }
    if (check_chunk(
            &model, 3905u,
            K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE,
            K3_PHYSICAL_READ_BYTES,
            K3_PHYSICAL_LAYER_SEGMENTS) != 0) {
        k3_st_model_close(&model);
        return 1;
    }
    static const uint32_t filled_chunks[] = {16384u, 32768u};
    for (size_t index = 0u;
         index < sizeof(filled_chunks) / sizeof(filled_chunks[0]);
         index++) {
        if (check_chunk(
                &model, filled_chunks[index],
                K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE,
                K3_PHYSICAL_READ_BYTES,
                K3_PHYSICAL_LAYER_SEGMENTS) != 0) {
            k3_st_model_close(&model);
            return 1;
        }
    }

    k3_prefill_plan candidate;
    CHECK(k3_prefill_plan_build_with_aux_workspace(
              &model, 8192u, 8192u, 32u, 16u,
              K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE,
              K3_KDA_DEQUANT_WORKSPACE,
              &candidate, error, sizeof(error)),
          error);
    CHECK(candidate.auxiliary_workspace_bytes ==
              K3_KDA_DEQUANT_WORKSPACE &&
          candidate.batch_workspace_bytes ==
              K3_ONE_TOKEN_WORKSPACE * 8192u +
                  K3_KDA_DEQUANT_WORKSPACE &&
          candidate.incremental_workspace_bytes ==
              K3_ONE_TOKEN_WORKSPACE * 8191u &&
          candidate.borrowed_cache_bytes ==
              candidate.batch_workspace_bytes &&
          candidate.new_device_bytes == 0u,
          "8K auxiliary cold-workspace ledger");
    CHECK(k3_prefill_plan_build_with_aux_workspace(
              &model, 512u, 8192u, 32u, 16u,
              K3_PREFILL_CACHE_RETAIN,
              K3_KDA_DEQUANT_WORKSPACE,
              &candidate, error, sizeof(error)),
          error);
    CHECK(candidate.batch_workspace_bytes ==
              K3_ONE_TOKEN_WORKSPACE * 512u +
                  K3_KDA_DEQUANT_WORKSPACE &&
          candidate.retained_cache_bytes ==
              K3_CACHE_32_BYTES &&
          candidate.new_device_bytes ==
              candidate.batch_workspace_bytes,
          "512-token auxiliary retained-cache ledger");

    CHECK(!k3_prefill_plan_build(
              &model, 65536u, 131072u, 30u, 16u,
              K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE,
              &candidate, error, sizeof(error)),
          "oversized warm reset workspace was accepted");
    CHECK(strstr(error, "cannot lend") != NULL,
          "oversized warm reset rejection lost its cache-capacity reason");

    CHECK(K3_LAYER_BYTES == K3_EXPERT_BYTES * 896u,
          "layer/expert byte oracle");
    printf("K3 prefill plan: PASS\n");
    printf("  routed payload: %.3f GiB in 92 sweeps\n",
           oracle.routed_store_payload_bytes / 1073741824.0);
    printf("  physical reads: %" PRIu64
           " bytes (%" PRIu64 " requested) in %u requests "
           "across %u shard segments\n",
           K3_PHYSICAL_READ_BYTES,
           K3_REQUESTED_READ_BYTES,
           oracle.expert_read_requests,
           K3_PHYSICAL_LAYER_SEGMENTS);
    printf("  2-token cold cache loan: %.3f MiB of %.3f GiB\n",
           oracle.batch_workspace_bytes / 1048576.0,
           oracle.decode_cache_bytes / 1073741824.0);
    k3_st_model_close(&model);
    return 0;
}
