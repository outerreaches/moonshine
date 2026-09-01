#include "../glm53_static_layout.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)
#define OK(c) CHECK((c) == GLM53_STATIC_LAYOUT_OK)

typedef struct {
    glm53_weight_plan plan;
    k3_st_tensor tensors[7];
    glm53_expert_plan expert;
} fixture;

static k3_st_tensor tensor(char *name, k3_st_dtype dtype, uint64_t bytes,
                           uint16_t shard, uint64_t physical) {
    k3_st_tensor t;
    memset(&t, 0, sizeof(t));
    t.name = name;
    t.dtype = dtype;
    t.ndim = 1u;
    t.shape[0] = bytes / (dtype == K3_ST_DTYPE_F32 ? 4u :
                           dtype == K3_ST_DTYPE_BF16 ? 2u : 1u);
    t.byte_length = bytes;
    t.shard = shard;
    t.physical_offset = physical;
    return t;
}

static void init_fixture(fixture *f) {
    memset(f, 0, sizeof(*f));
    f->tensors[0] = tensor("zeta", K3_ST_DTYPE_BF16, 4u, 7u, 700u);
    f->tensors[1] = tensor("alpha", K3_ST_DTYPE_F8_E4M3, 257u, 2u, 200u);
    f->tensors[2] = tensor("middle", K3_ST_DTYPE_F32, 256u, 4u, 400u);
    f->tensors[3] = tensor("beta.scale", K3_ST_DTYPE_F32, 4u, 8u, 800u);
    f->tensors[4] = tensor("yield.scale", K3_ST_DTYPE_F32, 252u, 9u, 900u);
    f->tensors[5] = tensor("stream.down", K3_ST_DTYPE_F8_E4M3, 999u, 1u, 10u);
    f->tensors[6] = tensor("stream.up", K3_ST_DTYPE_F8_E4M3, 888u, 1u, 20u);
    f->plan.built = true;
    f->plan.globals[0] = &f->tensors[0];
    f->plan.layers[2].roles[GLM53_ROLE_INPUT_NORM] = &f->tensors[1];
    f->plan.layers[1].roles[GLM53_ROLE_POST_ATTN_NORM] = &f->tensors[2];
    f->plan.resident_static.tensor_count = 3u;
    f->plan.resident_static.f8_bytes = 257u;
    f->plan.resident_static.f32_bytes = 256u;
    f->plan.resident_static.bf16_bytes = 4u;
    f->plan.resident_static.total_bytes = 517u;
    f->expert.tensors[GLM53_EXPERT_DOWN_WEIGHT] = &f->tensors[5];
    f->expert.tensors[GLM53_EXPERT_DOWN_SCALE] = &f->tensors[3];
    f->expert.tensors[GLM53_EXPERT_UP_WEIGHT] = &f->tensors[6];
    f->expert.tensors[GLM53_EXPERT_UP_SCALE] = &f->tensors[4];
    f->plan.routed_experts.experts = &f->expert;
    f->plan.routed_experts.expert_count = 1u;
    f->plan.resident_routed_scales.tensor_count = 2u;
    f->plan.resident_routed_scales.f32_bytes = 256u;
    f->plan.resident_routed_scales.total_bytes = 256u;
    f->plan.streamed_routed_experts.tensor_count = 2u;
    f->plan.streamed_routed_experts.f8_bytes = 1887u;
    f->plan.streamed_routed_experts.total_bytes = 1887u;
}

static bool test_order_offsets_and_lookup(void) {
    fixture f;
    glm53_static_layout layout;
    const char *want[] = { "alpha", "beta.scale", "middle", "yield.scale", "zeta" };
    const uint64_t offsets[] = { 0u, 512u, 768u, 1024u, 1280u };
    size_t i;
    char error[128];
    init_fixture(&f);
    memset(&layout, 0, sizeof(layout));
    OK(glm53_static_layout_build(&layout, &f.plan, error, sizeof(error)));
    CHECK(error[0] == '\0' && layout.built);
    CHECK(layout.entry_count == 5u && layout.tensor_count == 5u);
    CHECK(layout.logical_bytes == 773u);
    CHECK(layout.padded_bytes == 1536u);
    CHECK(layout.max_tensor_bytes == 257u);
    for (i = 0u; i < layout.entry_count; ++i) {
        const glm53_static_layout_entry *e = &layout.entries[i];
        CHECK(strcmp(e->tensor->name, want[i]) == 0);
        CHECK(e->device_offset == offsets[i]);
        CHECK(e->device_offset % GLM53_STATIC_LAYOUT_ALIGNMENT == 0u);
        CHECK(e->source_shard == e->tensor->shard);
        CHECK(e->source_physical_offset == e->tensor->physical_offset);
        CHECK(e->logical_bytes == e->tensor->byte_length);
        CHECK(e->dtype == e->tensor->dtype);
        CHECK(glm53_static_layout_find_name(&layout, want[i]) == e);
        CHECK(glm53_static_layout_find_tensor(&layout, e->tensor) == e);
    }
    CHECK(glm53_static_layout_find_name(&layout, "stream.down") == NULL);
    CHECK(glm53_static_layout_find_name(&layout, "absent") == NULL);
    CHECK(glm53_static_layout_find_tensor(&layout, &f.tensors[5]) == NULL);
    glm53_static_layout_free(&layout);
    CHECK(!layout.built && layout.entries == NULL && layout.entry_count == 0u);
    return true;
}

static bool failed_unchanged(glm53_weight_plan *plan,
                             glm53_static_layout_status wanted) {
    glm53_static_layout layout, before;
    char error[96];
    memset(&layout, 0xa5, sizeof(layout));
    before = layout;
    memset(error, 0, sizeof(error));
    CHECK(glm53_static_layout_build(&layout, plan, error, sizeof(error)) == wanted);
    CHECK(memcmp(&layout, &before, sizeof(layout)) == 0);
    CHECK(error[0] != '\0');
    return true;
}

static bool test_rejections_transactional(void) {
    fixture f;
    init_fixture(&f);
    f.plan.built = false;
    CHECK(failed_unchanged(&f.plan, GLM53_STATIC_LAYOUT_UNBUILT_PLAN));

    init_fixture(&f);
    f.tensors[2].name = f.tensors[1].name;
    CHECK(failed_unchanged(&f.plan, GLM53_STATIC_LAYOUT_DUPLICATE_TENSOR));

    init_fixture(&f);
    f.plan.resident_static.tensor_count = 2u;
    CHECK(failed_unchanged(&f.plan, GLM53_STATIC_LAYOUT_BAD_PLAN));

    init_fixture(&f);
    f.tensors[3].dtype = K3_ST_DTYPE_F8_E4M3;
    f.plan.resident_routed_scales.f32_bytes = 255u;
    f.plan.resident_routed_scales.f8_bytes = 1u;
    CHECK(failed_unchanged(&f.plan, GLM53_STATIC_LAYOUT_BAD_PLAN));

    init_fixture(&f);
    f.plan.routed_experts.experts = NULL;
    CHECK(failed_unchanged(&f.plan, GLM53_STATIC_LAYOUT_BAD_PLAN));

    init_fixture(&f);
    f.plan.resident_static.f8_bytes -= 1u;
    f.plan.resident_static.bf16_bytes += 1u;
    CHECK(failed_unchanged(&f.plan, GLM53_STATIC_LAYOUT_BAD_PLAN));

    init_fixture(&f);
    f.tensors[1].shape[0] += 1u;
    CHECK(failed_unchanged(&f.plan, GLM53_STATIC_LAYOUT_BAD_PLAN));

    init_fixture(&f);
    f.tensors[1].physical_offset = UINT64_MAX - 128u;
    CHECK(failed_unchanged(&f.plan, GLM53_STATIC_LAYOUT_BAD_PLAN));
    return true;
}

static bool test_checked_alignment_overflow(void) {
    glm53_weight_plan plan;
    glm53_static_layout layout, before;
    k3_st_tensor huge = tensor("huge", K3_ST_DTYPE_F8_E4M3,
                               UINT64_MAX - UINT64_C(127), 0u, 0u);
    char error[64];
    memset(&plan, 0, sizeof(plan));
    plan.built = true;
    plan.globals[0] = &huge;
    plan.resident_static.tensor_count = 1u;
    plan.resident_static.f8_bytes = huge.byte_length;
    plan.resident_static.total_bytes = huge.byte_length;
    memset(&layout, 0x3c, sizeof(layout));
    before = layout;
    CHECK(glm53_static_layout_build(&layout, &plan, error, sizeof(error)) ==
          GLM53_STATIC_LAYOUT_OVERFLOW);
    CHECK(memcmp(&layout, &before, sizeof(layout)) == 0);
    return true;
}

#include "../glm53_architecture.h"
#include "../glm53_manifest.h"
static bool optional_official(void) {
    const char *root = getenv("GLM53_OFFICIAL_ROOT");
    k3_st_model all, main;
    glm53_manifest manifest;
    glm53_weight_plan plan;
    glm53_static_layout layout;
    char error[512];
    size_t i, count = 0u;
    if (!root || !root[0]) return true;
    memset(&all, 0, sizeof(all)); memset(&main, 0, sizeof(main));
    memset(&manifest, 0, sizeof(manifest)); memset(&plan, 0, sizeof(plan));
    memset(&layout, 0, sizeof(layout));
    CHECK(glm53_manifest_load(&manifest, root, error, sizeof(error)));
    CHECK(k3_st_model_open_5digit_total(&all, root, GLM53_SHARD_COUNT,
                                        error, sizeof(error)));
    main.tensors = (k3_st_tensor *)calloc(GLM53_MAIN_TENSOR_COUNT,
                                          sizeof(*main.tensors));
    CHECK(main.tensors != NULL);
    main.shards = all.shards; main.shard_count = all.shard_count;
    main.routed_span = all.routed_span;
    main.routed_span_context = all.routed_span_context;
    for (i = 0u; i < all.tensor_count; ++i)
        if (glm53_architecture_validate_main_tensor(&all.tensors[i], NULL))
            main.tensors[count++] = all.tensors[i];
    main.tensor_count = count; main.tensor_capacity = count;
    CHECK(count == GLM53_MAIN_TENSOR_COUNT);
    CHECK(glm53_weight_plan_build_manifest(&plan, &manifest, &main,
                                            error, sizeof(error)));
    OK(glm53_static_layout_build(&layout, &plan, error, sizeof(error)));
    CHECK(layout.logical_bytes == UINT64_C(15300311288));
    CHECK(layout.padded_bytes == UINT64_C(15300353024));
    CHECK(layout.tensor_count == 37713u);
    CHECK(layout.tensor_count == 1425u + 36288u);
    glm53_static_layout_free(&layout);
    glm53_weight_plan_free(&plan);
    free(main.tensors);
    k3_st_model_close(&all);
    glm53_manifest_free(&manifest);
    return true;
}

int main(void) {
    CHECK(test_order_offsets_and_lookup());
    CHECK(test_rejections_transactional());
    CHECK(test_checked_alignment_overflow());
    CHECK(optional_official());
    puts("glm53 static layout: all tests passed");
    return 0;
}
