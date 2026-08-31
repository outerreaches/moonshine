#include "../glm53_expert_plan.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)
#define OK(c) CHECK((c) == GLM53_EXPERT_PLAN_OK)

typedef struct {
    k3_st_shard shard;
    k3_st_tensor tensors[6];
    char names[6][128];
    const k3_st_tensor *order[6];
    k3_st_model model;
} fixture;

static void init_fixture(fixture *f, uint16_t layer, uint16_t expert,
                         uint64_t weight_at, uint64_t scale_at) {
    static const char *projection[6] = {
        "down_proj", "down_proj", "gate_proj", "gate_proj",
        "up_proj", "up_proj"
    };
    size_t i;
    memset(f, 0, sizeof(*f));
    f->shard.data_offset = 4096u;
    f->shard.file_bytes = UINT64_C(100000000);
    for (i = 0u; i < 6u; ++i) {
        const bool scale = (i & 1u) != 0u;
        k3_st_tensor *t = &f->tensors[i];
        (void)snprintf(f->names[i], sizeof(f->names[i]),
            "model.language_model.layers.%u.mlp.experts.%u.%s.%s",
            (unsigned)layer, (unsigned)expert, projection[i],
            scale ? "weight_scale_inv" : "weight");
        t->name = f->names[i];
        t->physical_offset = scale ? scale_at : weight_at;
        t->byte_length = scale ? 2048u : 8388608u;
        t->shape[0] = scale ? (i < 2u ? 32u : 16u) :
                              (i < 2u ? 4096u : 2048u);
        t->shape[1] = scale ? (i < 2u ? 16u : 32u) :
                              (i < 2u ? 2048u : 4096u);
        t->shard = 0u;
        t->ndim = 2u;
        t->dtype = scale ? K3_ST_DTYPE_F32 : K3_ST_DTYPE_F8_E4M3;
        f->order[i] = t;
        if (scale) scale_at += 2048u; else weight_at += 8388608u;
    }
    f->model.shards = &f->shard;
    f->model.shard_count = 1u;
    f->model.tensors = f->tensors;
    f->model.tensor_count = 6u;
    f->model.tensor_capacity = 6u;
}

static bool test_positive_order_and_page_phase(void) {
    fixture f;
    glm53_expert_plan p;
    char error[256];
    init_fixture(&f, 3u, 0u, 4096u, UINT64_C(41943040));
    OK(glm53_expert_plan_build(&p, &f.model, 3u, 0u, f.order, 6u,
                               error, sizeof(error)));
    CHECK(p.logical_bytes == GLM53_EXPERT_LOGICAL_BYTES);
    CHECK(p.logical[0].length == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(p.logical[1].length == GLM53_EXPERT_SCALE_BYTES);
    CHECK(p.physical_count == 2u);
    /* Page-aligned 6144 logical bytes occupy two pages. */
    CHECK(p.physical[1].length == 8192u);

    { /* Tensor order is irrelevant. */
        const k3_st_tensor *reverse[6] = {
            f.order[5], f.order[4], f.order[3], f.order[2], f.order[1], f.order[0]
        };
        glm53_expert_plan q;
        OK(glm53_expert_plan_build(&q, &f.model, 3u, 0u, reverse, 6u,
                                   error, sizeof(error)));
        CHECK(q.logical[0].offset == p.logical[0].offset);
        CHECK(q.physical_bytes == p.physical_bytes);
    }

    /* A phase beyond 2048 crosses three 4K pages. */
    init_fixture(&f, 44u, 287u, 4096u, UINT64_C(41946040));
    OK(glm53_expert_plan_build(&p, &f.model, 44u, 287u, f.order, 6u,
                               error, sizeof(error)));
    CHECK(p.logical[1].offset == UINT64_C(41946040));
    CHECK((p.logical[1].offset & 4095u) > 2048u);
    CHECK(p.physical[1].length == 12288u);
    return true;
}

static bool test_physical_coalescing(void) {
    fixture f;
    glm53_expert_plan p;
    char error[128];
    const uint64_t weight_at = 8192u;
    init_fixture(&f, 10u, 7u, weight_at,
                 weight_at + GLM53_EXPERT_WEIGHT_BYTES);
    OK(glm53_expert_plan_build(&p, &f.model, 10u, 7u, f.order, 6u,
                               error, sizeof(error)));
    CHECK(p.physical_count == 1u);
    CHECK(p.physical[0].offset == weight_at);
    CHECK(p.physical[0].length == GLM53_EXPERT_WEIGHT_BYTES + 8192u);
    CHECK(p.physical_bytes == p.physical[0].length);
    return true;
}

static bool expect_failure(fixture *f, glm53_expert_plan_status wanted) {
    glm53_expert_plan p;
    char error[128];
    const glm53_expert_plan_status got = glm53_expert_plan_build(
        &p, &f->model, 3u, 1u, f->order, 6u, error, sizeof(error));
    CHECK(got == wanted);
    CHECK(error[0] != '\0');
    return true;
}

static bool test_corruptions(void) {
    fixture f;
    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.order[5] = f.order[4];
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_DUPLICATE_ROLE));

    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.tensors[0].dtype = K3_ST_DTYPE_F32;
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_BAD_TENSOR));
    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.tensors[0].shape[0] = 1u;
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_BAD_TENSOR));
    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.tensors[0].byte_length = 0u;
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_BAD_TENSOR));

    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    {
        k3_st_shard shards[2] = { f.shard, f.shard };
        f.tensors[2].shard = 1u;
        f.model.shards = shards;
        f.model.shard_count = 2u;
        CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_CROSS_SHARD));
    }

    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.tensors[2].physical_offset += 1u;
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_NONCONTIGUOUS));
    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.tensors[1].physical_offset += 1u;
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_NONCONTIGUOUS));

    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.tensors[0].physical_offset = 4095u;
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_OUT_OF_BOUNDS));
    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.shard.file_bytes = UINT64_C(40006144); /* exact EOF fallback */
    {
        glm53_expert_plan p;
        char error[128];
        OK(glm53_expert_plan_build(&p, &f.model, 3u, 1u, f.order, 6u,
                                   error, sizeof(error)));
        CHECK(p.physical[1].offset == UINT64_C(40000000));
        CHECK(p.physical[1].length == GLM53_EXPERT_SCALE_BYTES);
    }
    f.shard.file_bytes = UINT64_C(40006143);
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_OUT_OF_BOUNDS));
    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    f.tensors[5].physical_offset = UINT64_MAX - 1000u;
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_OVERFLOW));

    init_fixture(&f, 3u, 1u, 4096u, UINT64_C(40000000));
    (void)snprintf(f.names[0], sizeof(f.names[0]),
        "model.language_model.layers.3.mlp.experts.1.bad.weight");
    CHECK(expect_failure(&f, GLM53_EXPERT_PLAN_MALFORMED_NAME));
    return true;
}

static void free_full(k3_st_model *model) {
    size_t i;
    for (i = 0u; i < model->tensor_count; ++i) free(model->tensors[i].name);
    free(model->tensors);
    free(model->shards);
    memset(model, 0, sizeof(*model));
}

static bool make_full(k3_st_model *model) {
    const size_t count = (size_t)GLM53_EXPERT_COUNT * 6u;
    size_t at = 0u;
    unsigned layer, expert, role;
    static const char *projection[6] = {
        "down_proj", "down_proj", "gate_proj", "gate_proj",
        "up_proj", "up_proj"
    };
    memset(model, 0, sizeof(*model));
    model->shards = (k3_st_shard *)calloc(1u, sizeof(*model->shards));
    model->tensors = (k3_st_tensor *)calloc(count, sizeof(*model->tensors));
    if (!model->shards || !model->tensors) { free_full(model); return false; }
    model->shard_count = 1u; model->tensor_count = count;
    model->tensor_capacity = count;
    model->shards[0].data_offset = 4096u;
    model->shards[0].file_bytes = UINT64_C(100000000);
    for (layer = 3u; layer <= 44u; ++layer) {
        for (expert = 0u; expert < 288u; ++expert) {
            uint64_t wo = 4096u, so = UINT64_C(40000000);
            for (role = 0u; role < 6u; ++role) {
                const bool scale = (role & 1u) != 0u;
                k3_st_tensor *t = &model->tensors[at++];
                t->name = (char *)malloc(128u);
                if (!t->name) { free_full(model); return false; }
                (void)snprintf(t->name, 128u,
                    "model.language_model.layers.%u.mlp.experts.%u.%s.%s",
                    layer, expert, projection[role],
                    scale ? "weight_scale_inv" : "weight");
                t->physical_offset = scale ? so : wo;
                t->byte_length = scale ? 2048u : 8388608u;
                t->shape[0] = scale ? (role < 2u ? 32u : 16u) :
                                      (role < 2u ? 4096u : 2048u);
                t->shape[1] = scale ? (role < 2u ? 16u : 32u) :
                                      (role < 2u ? 2048u : 4096u);
                t->ndim = 2u; t->shard = 0u;
                t->dtype = scale ? K3_ST_DTYPE_F32 : K3_ST_DTYPE_F8_E4M3;
                if (scale) so += 2048u; else wo += 8388608u;
            }
        }
    }
    return true;
}

static bool test_full_model_and_namespace_rejection(void) {
    k3_st_model model;
    glm53_expert_model_plan plan = { 0 };
    char error[256];
    CHECK(make_full(&model));
    /* Reverse directory order to prove the full builder does not rely on it. */
    { size_t i; for (i = 0u; i < model.tensor_count / 2u; ++i) {
        k3_st_tensor x = model.tensors[i];
        model.tensors[i] = model.tensors[model.tensor_count - 1u - i];
        model.tensors[model.tensor_count - 1u - i] = x;
    }}
    OK(glm53_expert_model_plan_build(&plan, &model, error, sizeof(error)));
    CHECK(plan.expert_count == GLM53_EXPERT_COUNT);
    CHECK(plan.experts[0].layer == 3u && plan.experts[0].expert == 0u);
    CHECK(plan.experts[GLM53_EXPERT_COUNT - 1u].layer == 44u);
    CHECK(plan.logical_bytes == GLM53_EXPERT_LOGICAL_BYTES *
                                (uint64_t)GLM53_EXPERT_COUNT);
    glm53_expert_model_plan_free(&plan);

    free(model.tensors[0].name);
    model.tensors[0].name = (char *)malloc(96u); CHECK(model.tensors[0].name);
    (void)snprintf(model.tensors[0].name, 96u,
        "model.language_model.layers.45.mlp.experts.0.down_proj.weight");
    CHECK(glm53_expert_model_plan_build(&plan, &model, error, sizeof(error)) ==
          GLM53_EXPERT_PLAN_INCOMPLETE_MODEL);
    free(model.tensors[0].name);
    model.tensors[0].name = (char *)malloc(96u); CHECK(model.tensors[0].name);
    (void)snprintf(model.tensors[0].name, 96u,
        "model.language_model.layers.44.mlp.shared_experts.down_proj.weight");
    CHECK(glm53_expert_model_plan_build(&plan, &model, error, sizeof(error)) ==
          GLM53_EXPERT_PLAN_INCOMPLETE_MODEL);
    free_full(&model);

    /* A malformed additive shared namespace must fail even when all main
     * routed tensors remain complete. */
    CHECK(make_full(&model));
    {
        const size_t old_count = model.tensor_count;
        k3_st_tensor *grown = (k3_st_tensor *)realloc(
            model.tensors, (old_count + 1u) * sizeof(*model.tensors));
        CHECK(grown != NULL);
        model.tensors = grown;
        memset(&model.tensors[old_count], 0, sizeof(model.tensors[old_count]));
        model.tensors[old_count].name = (char *)malloc(96u);
        CHECK(model.tensors[old_count].name != NULL);
        (void)snprintf(model.tensors[old_count].name, 96u,
            "model.language_model.layers.44.mlp.shared_experts.0.bad");
        model.tensor_count = old_count + 1u;
        model.tensor_capacity = model.tensor_count;
        CHECK(glm53_expert_model_plan_build(
                  &plan, &model, error, sizeof(error)) ==
              GLM53_EXPERT_PLAN_MALFORMED_NAME);
    }
    free_full(&model);
    return true;
}

int main(void) {
    CHECK(test_positive_order_and_page_phase());
    CHECK(test_physical_coalescing());
    CHECK(test_corruptions());
    CHECK(test_full_model_and_namespace_rejection());
    puts("glm53 expert plan: all tests passed");
    return 0;
}
