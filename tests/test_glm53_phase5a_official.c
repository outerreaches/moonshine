#include "../glm53_expert_stream.h"
#include "../glm53_architecture.h"
#include "../glm53_engine_plan.h"
#include "../glm53_manifest.h"
#include "../glm53_weights.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OFFICIAL_EXPERT_COUNT ((size_t)GLM53_EXPERT_LAYER_COUNT * \
                               (size_t)GLM53_EXPERTS_PER_LAYER)

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    goto done; \
} } while (0)

static int add_u64(uint64_t *sum, uint64_t value) {
    if (*sum > UINT64_MAX - value) return 0;
    *sum += value;
    return 1;
}

static int ledger_is(const glm53_weight_ledger *ledger,
                     uint64_t f8, uint64_t f32, uint64_t bf16,
                     uint64_t total, size_t count) {
    return ledger->f8_bytes == f8 && ledger->f32_bytes == f32 &&
           ledger->bf16_bytes == bf16 && ledger->total_bytes == total &&
           ledger->tensor_count == count;
}

int main(int argc, char **argv) {
    glm53_manifest manifest;
    k3_st_model all, main_model;
    glm53_architecture_report architecture;
    glm53_weight_plan weights;
    glm53_engine_plan_config config;
    glm53_engine_static_ledger static_ledger;
    glm53_engine_plan_report engine;
    glm53_expert_stream_stage stage;
    glm53_expert_stream_ledger stage_ledger;
    char error[512];
    size_t i, main_count = 0u;
    uint64_t logical_requests = 0u, logical_bytes = 0u;
    uint64_t physical_requests = 0u, physical_bytes = 0u;
    uint64_t min_cover = UINT64_MAX, max_cover = 0u;
    int result = 1;

    memset(&manifest, 0, sizeof(manifest));
    memset(&all, 0, sizeof(all));
    memset(&main_model, 0, sizeof(main_model));
    memset(&architecture, 0, sizeof(architecture));
    memset(&weights, 0, sizeof(weights));
    memset(&engine, 0, sizeof(engine));
    if (argc != 2) {
        fprintf(stderr, "usage: %s OFFICIAL_ROOT\n", argv[0]);
        return 2;
    }

    /* These APIs read the pinned JSON and all 62 SafeTensors headers only.
     * No tensor payload is mapped or read by this metadata integration test. */
    CHECK(glm53_manifest_load(&manifest, argv[1], error, sizeof(error)));
    CHECK(k3_st_model_open_5digit_total(&all, argv[1], GLM53_SHARD_COUNT,
                                        error, sizeof(error)));
    CHECK(glm53_manifest_reconcile(&manifest, &all, error, sizeof(error)));
    CHECK(glm53_architecture_validate(&all, &architecture,
                                      error, sizeof(error)));
    CHECK(architecture.main_count == GLM53_MAIN_TENSOR_COUNT);
    CHECK(architecture.mtp_count == GLM53_MTP_TENSOR_COUNT);
    CHECK(architecture.vision_count == GLM53_VISION_TENSOR_COUNT);

    /* A shallow exact main-text view borrows every name and shard from all. */
    main_model.tensors = (k3_st_tensor *)calloc(
        GLM53_MAIN_TENSOR_COUNT, sizeof(*main_model.tensors));
    CHECK(main_model.tensors != NULL);
    main_model.shards = all.shards;
    main_model.shard_count = all.shard_count;
    main_model.routed_span = all.routed_span;
    main_model.routed_span_context = all.routed_span_context;
    for (i = 0u; i < all.tensor_count; ++i) {
        if (glm53_architecture_validate_main_tensor(&all.tensors[i], NULL)) {
            CHECK(main_count < GLM53_MAIN_TENSOR_COUNT);
            main_model.tensors[main_count++] = all.tensors[i];
        }
    }
    CHECK(main_count == GLM53_MAIN_TENSOR_COUNT);
    main_model.tensor_count = main_count;
    main_model.tensor_capacity = main_count;
    CHECK(glm53_architecture_validate_main(&main_model, &architecture,
                                           error, sizeof(error)));
    CHECK(glm53_weight_plan_build_manifest(&weights, &manifest, &main_model,
                                            error, sizeof(error)));

    CHECK(ledger_is(&weights.resident_static,
                    UINT64_C(2617245696), UINT64_C(1819896),
                    UINT64_C(12606927872), GLM53_WEIGHT_RESIDENT_STATIC_BYTES,
                    GLM53_WEIGHT_RESIDENT_STATIC_COUNT));
    CHECK(ledger_is(&weights.streamed_routed_experts,
                    GLM53_WEIGHT_STREAMED_BYTES, 0u, 0u,
                    GLM53_WEIGHT_STREAMED_BYTES,
                    GLM53_WEIGHT_STREAMED_COUNT));
    CHECK(ledger_is(&weights.resident_routed_scales,
                    0u, GLM53_WEIGHT_ROUTED_SCALE_BYTES, 0u,
                    GLM53_WEIGHT_ROUTED_SCALE_BYTES,
                    GLM53_WEIGHT_ROUTED_SCALE_COUNT));
    CHECK(ledger_is(&weights.total,
                    GLM53_WEIGHT_MAIN_F8_BYTES, GLM53_WEIGHT_MAIN_F32_BYTES,
                    GLM53_WEIGHT_MAIN_BF16_BYTES, GLM53_WEIGHT_MAIN_BYTES,
                    GLM53_MAIN_TENSOR_COUNT));
    CHECK(weights.routed_experts.expert_count == OFFICIAL_EXPERT_COUNT);
    CHECK(weights.routed_experts.logical_bytes == UINT64_C(304480124928));
    CHECK(weights.routed_experts.physical_bytes == UINT64_C(304578072576));

    glm53_engine_plan_config_init(&config);
    CHECK(glm53_weight_plan_engine_resident_bytes(
              &weights, &static_ledger.resident_bytes));
    CHECK(static_ledger.resident_bytes == UINT64_C(15300311288));
    /* Conservative load-time scratch: retain one full mapped staging slot. */
    static_ledger.peak_transient_bytes = GLM53_ENGINE_PLAN_MAPPED_STAGING_BYTES;
    CHECK(glm53_engine_plan_build(&config, &static_ledger,
              GLM53_ENGINE_PLAN_HARD_CEILING_BYTES - UINT64_C(1), &engine) ==
          GLM53_ENGINE_PLAN_OK);
    CHECK(config.context_tokens == 512u && config.batch_size == 1u);
    CHECK(config.expert_cache_slots_per_layer == 0u);
    CHECK(config.mapped_staging_slots == 1u);
    CHECK(engine.context_tokens == 512u && engine.batch_size == 1u);
    CHECK(engine.expert_cache_slots_per_layer == 0u);
    CHECK(engine.mapped_staging_slots == 1u);
    CHECK(engine.expert_resident_bytes == 0u);
    CHECK(engine.static_resident_bytes == UINT64_C(15300311288));
    CHECK(engine.static_peak_transient_bytes == UINT64_C(33562624));
    CHECK(engine.kda_state_bytes == UINT64_C(142606336));
    CHECK(engine.kda_conv_bytes == UINT64_C(13369344));
    CHECK(engine.dsa_compact_bytes == UINT64_C(8662016));
    CHECK(engine.mapped_staging_bytes == UINT64_C(33562624));
    CHECK(engine.compute_workspace_bytes == UINT64_C(268435456));
    CHECK(engine.library_workspace_bytes == UINT64_C(67108864));
    CHECK(engine.allocator_guard_bytes == UINT64_C(268435456));
    CHECK(engine.runtime_guard_bytes == UINT64_C(1073741824));
    CHECK(engine.host_guard_bytes == UINT64_C(4294967296));
    CHECK(engine.runtime_bytes_before_guards == UINT64_C(15834055928));
    CHECK(engine.load_bytes_before_guards == UINT64_C(15333873912));
    CHECK(engine.peak_bytes_before_guards == UINT64_C(15834055928));
    CHECK(engine.admitted_bytes == UINT64_C(21471200504));
    CHECK(engine.abort_reserve_bytes == UINT64_C(111672785671));

    for (i = 0u; i < weights.routed_experts.expert_count; ++i) {
        const glm53_expert_plan *expert = &weights.routed_experts.experts[i];
        const glm53_expert_extent *weight = &expert->logical[0];
        const glm53_expert_extent *scale = &expert->logical[1];
        uint64_t request_end, weight_end, scale_end;

        {
            glm53_expert_stream_status stream_status =
                glm53_expert_stream_stage_build(&stage, expert, &all,
                                                  error, sizeof(error));
            if (stream_status != GLM53_EXPERT_STREAM_OK)
                fprintf(stderr, "expert %zu: %s\n", i, error);
            CHECK(stream_status == GLM53_EXPERT_STREAM_OK);
        }
        CHECK(glm53_expert_stream_ledger_from_stage(&stage_ledger, &stage) ==
              GLM53_EXPERT_STREAM_OK);
        CHECK(stage.request_count == 1u && stage.copy_count == 1u);
        CHECK(stage.logical_bytes == GLM53_EXPERT_WEIGHT_BYTES);
        CHECK(stage.copy.slot_offset == 0u);
        CHECK(stage.copy.byte_count == GLM53_EXPERT_WEIGHT_BYTES);
        CHECK(stage.request.shard == weight->shard);
        if (stage.request.direct_io) {
            CHECK(stage.request.file_offset % GLM53_EXPERT_IO_ALIGNMENT == 0u);
            CHECK(stage.request.byte_count % GLM53_EXPERT_IO_ALIGNMENT == 0u);
        } else {
            CHECK(stage.request.file_offset == weight->offset);
            CHECK(stage.request.byte_count == weight->length);
        }
        CHECK(stage.request.byte_count >= GLM53_EXPERT_WEIGHT_BYTES);
        CHECK(stage.request.byte_count <= GLM53_EXPERT_STREAM_MAX_COVER);
        CHECK(add_u64(&logical_requests, stage_ledger.copy_count));
        CHECK(add_u64(&logical_bytes, stage_ledger.copied_bytes));
        CHECK(add_u64(&physical_requests, stage_ledger.request_count));
        CHECK(add_u64(&physical_bytes, stage_ledger.requested_bytes));

        CHECK(stage.request.file_offset <= weight->offset);
        CHECK(stage.copy.request_offset ==
              weight->offset - stage.request.file_offset);
        request_end = stage.request.file_offset;
        CHECK(add_u64(&request_end, stage.request.byte_count));
        weight_end = weight->offset;
        CHECK(add_u64(&weight_end, weight->length));
        CHECK(stage.copy.request_offset + stage.copy.byte_count <=
              stage.request.byte_count);
        CHECK(request_end >= weight_end);
        scale_end = scale->offset;
        CHECK(add_u64(&scale_end, scale->length));
        CHECK(scale->shard != stage.request.shard ||
              request_end <= scale->offset || stage.request.file_offset >= scale_end);
        CHECK(stage.request.shard < all.shard_count);
        CHECK(request_end <= all.shards[stage.request.shard].file_bytes);
        if (stage.request.byte_count < min_cover)
            min_cover = stage.request.byte_count;
        if (stage.request.byte_count > max_cover)
            max_cover = stage.request.byte_count;
    }
    CHECK(logical_requests == OFFICIAL_EXPERT_COUNT);
    CHECK(logical_bytes == GLM53_WEIGHT_STREAMED_BYTES);
    CHECK(physical_requests == OFFICIAL_EXPERT_COUNT);
    CHECK(physical_bytes == UINT64_C(304455114752));
    CHECK(min_cover == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(max_cover == GLM53_EXPERT_STREAM_MAX_COVER);

    printf("PASS phase5a official metadata: experts=%u logical_requests=%" PRIu64
           " logical_bytes=%" PRIu64 " physical_requests=%" PRIu64
           " physical_bytes=%" PRIu64 " cover=[%" PRIu64 ",%" PRIu64 "]"
           " admitted=%" PRIu64 " expert_cache=%u\n",
           (unsigned)OFFICIAL_EXPERT_COUNT, logical_requests, logical_bytes,
           physical_requests, physical_bytes, min_cover, max_cover,
           engine.admitted_bytes, engine.expert_cache_slots_per_layer);
    result = 0;

done:
    glm53_weight_plan_free(&weights);
    free(main_model.tensors);
    /* main_model is a borrowed view; only all owns names and shard storage. */
    k3_st_model_close(&all);
    glm53_manifest_free(&manifest);
    return result;
}
