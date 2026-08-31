#include "glm53_expert_plan.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint16_t layer;
    uint16_t expert;
    uint8_t role;
    const k3_st_tensor *tensor;
} tensor_ref;

static void set_error(char *error, size_t size, const char *format, ...) {
    va_list ap;
    if (!error || size == 0u) return;
    va_start(ap, format);
    (void)vsnprintf(error, size, format, ap);
    va_end(ap);
}

static glm53_expert_plan_status reject(char *error, size_t size,
                                        glm53_expert_plan_status status,
                                        const char *message) {
    set_error(error, size, "%s", message);
    return status;
}

static bool add_u64(uint64_t a, uint64_t b, uint64_t *result) {
    if (a > UINT64_MAX - b) return false;
    *result = a + b;
    return true;
}

/* Returns 1 for an exact routed name, 0 for an unrelated name, and -1 for a
 * name which enters a routed/shared-expert namespace but is not admissible. */
static int parse_name(const char *name, uint16_t *layer, uint16_t *expert,
                      uint8_t *role) {
    static const char prefix[] = "model.language_model.layers.";
    static const char expert_component[] = ".mlp.experts.";
    const char *p;
    unsigned long layer_value = 0ul;
    unsigned long expert_value = 0ul;
    size_t digits = 0u;
    const char *projection;
    size_t projection_size;
    bool scale;

    if (!name) return -1;
    if (strncmp(name, prefix, sizeof(prefix) - 1u) != 0) {
        if (strncmp(name, "model.language_model.", 21u) == 0 &&
            (strstr(name, ".experts.") || strstr(name, ".shared_experts.")))
            return -1;
        return 0;
    }
    p = name + sizeof(prefix) - 1u;
    while (p[digits] >= '0' && p[digits] <= '9') {
        if (layer_value > 6553ul) return -1;
        layer_value = layer_value * 10ul + (unsigned long)(p[digits] - '0');
        ++digits;
    }
    if (digits == 0u || (digits > 1u && p[0] == '0')) {
        if (strstr(p, ".mlp.experts") || strstr(p, ".mlp.shared_experts"))
            return -1;
        return 0;
    }
    if (strncmp(p + digits, ".mlp.shared_experts.", 20u) == 0) {
        const char *suffix = p + digits + 20u;
        const bool valid_suffix =
            strcmp(suffix, "down_proj.weight") == 0 ||
            strcmp(suffix, "down_proj.weight_scale_inv") == 0 ||
            strcmp(suffix, "gate_proj.weight") == 0 ||
            strcmp(suffix, "gate_proj.weight_scale_inv") == 0 ||
            strcmp(suffix, "up_proj.weight") == 0 ||
            strcmp(suffix, "up_proj.weight_scale_inv") == 0;
        /* Exact official shared experts are valid but not routed slots. */
        return valid_suffix && layer_value >= GLM53_EXPERT_FIRST_LAYER &&
               layer_value <= 45ul ? 0 : -1;
    }
    if (strncmp(p + digits, expert_component,
                sizeof(expert_component) - 1u) != 0) {
        if (strstr(p, ".mlp.experts") || strstr(p, ".mlp.shared_experts"))
            return -1;
        return 0;
    }
    p += digits + sizeof(expert_component) - 1u;
    digits = 0u;
    while (p[digits] >= '0' && p[digits] <= '9') {
        if (expert_value > 6553ul) return -1;
        expert_value = expert_value * 10ul + (unsigned long)(p[digits] - '0');
        ++digits;
    }
    if (digits == 0u || (digits > 1u && p[0] == '0') || p[digits] != '.')
        return -1;
    p += digits + 1u;
    if (strcmp(p, "down_proj.weight") == 0) {
        projection = "down"; projection_size = 4u; scale = false;
    } else if (strcmp(p, "down_proj.weight_scale_inv") == 0) {
        projection = "down"; projection_size = 4u; scale = true;
    } else if (strcmp(p, "gate_proj.weight") == 0) {
        projection = "gate"; projection_size = 4u; scale = false;
    } else if (strcmp(p, "gate_proj.weight_scale_inv") == 0) {
        projection = "gate"; projection_size = 4u; scale = true;
    } else if (strcmp(p, "up_proj.weight") == 0) {
        projection = "up"; projection_size = 2u; scale = false;
    } else if (strcmp(p, "up_proj.weight_scale_inv") == 0) {
        projection = "up"; projection_size = 2u; scale = true;
    } else {
        return -1;
    }
    if (expert_value >= GLM53_EXPERTS_PER_LAYER) return -1;
    if (layer_value == 45ul) {
        /* The released MTP layer is valid but excluded from the main plan. */
        return 0;
    }
    if (layer_value < GLM53_EXPERT_FIRST_LAYER ||
        layer_value > GLM53_EXPERT_LAST_LAYER) return -1;
    *layer = (uint16_t)layer_value;
    *expert = (uint16_t)expert_value;
    if (projection_size == 4u && projection[0] == 'd')
        *role = (uint8_t)(scale ? GLM53_EXPERT_DOWN_SCALE :
                                  GLM53_EXPERT_DOWN_WEIGHT);
    else if (projection_size == 4u)
        *role = (uint8_t)(scale ? GLM53_EXPERT_GATE_SCALE :
                                  GLM53_EXPERT_GATE_WEIGHT);
    else
        *role = (uint8_t)(scale ? GLM53_EXPERT_UP_SCALE :
                                  GLM53_EXPERT_UP_WEIGHT);
    return 1;
}

static bool tensor_geometry(const k3_st_tensor *tensor, uint8_t role) {
    const bool down = role == GLM53_EXPERT_DOWN_WEIGHT ||
                      role == GLM53_EXPERT_DOWN_SCALE;
    const bool scale = role == GLM53_EXPERT_DOWN_SCALE ||
                       role == GLM53_EXPERT_GATE_SCALE ||
                       role == GLM53_EXPERT_UP_SCALE;
    const uint64_t rows = scale ? (down ? 32u : 16u) :
                                  (down ? 4096u : 2048u);
    const uint64_t cols = scale ? (down ? 16u : 32u) :
                                  (down ? 2048u : 4096u);
    const uint64_t bytes = scale ? UINT64_C(2048) : UINT64_C(8388608);
    return tensor->ndim == 2u && tensor->shape[0] == rows &&
           tensor->shape[1] == cols && tensor->byte_length == bytes &&
           tensor->dtype == (scale ? K3_ST_DTYPE_F32 : K3_ST_DTYPE_F8_E4M3);
}

static void sort_three(const k3_st_tensor **items) {
    size_t i;
    for (i = 1u; i < 3u; ++i) {
        const k3_st_tensor *value = items[i];
        size_t j = i;
        while (j > 0u && items[j - 1u]->physical_offset > value->physical_offset) {
            items[j] = items[j - 1u];
            --j;
        }
        items[j] = value;
    }
}

static glm53_expert_plan_status make_logical_extent(
        glm53_expert_extent *extent, const k3_st_tensor *const source[3],
        uint64_t wanted, char *error, size_t error_size) {
    const k3_st_tensor *sorted[3] = { source[0], source[1], source[2] };
    uint64_t end;
    size_t i;
    sort_three(sorted);
    if (!add_u64(sorted[0]->physical_offset, sorted[0]->byte_length, &end))
        return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                      "tensor extent overflows");
    for (i = 1u; i < 3u; ++i) {
        if (sorted[i]->physical_offset != end)
            return reject(error, error_size, GLM53_EXPERT_PLAN_NONCONTIGUOUS,
                          "logical extent is not contiguous");
        if (!add_u64(end, sorted[i]->byte_length, &end))
            return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                          "tensor extent overflows");
    }
    if (end - sorted[0]->physical_offset != wanted)
        return reject(error, error_size, GLM53_EXPERT_PLAN_BAD_TENSOR,
                      "logical extent has wrong length");
    extent->shard = sorted[0]->shard;
    extent->offset = sorted[0]->physical_offset;
    extent->length = wanted;
    return GLM53_EXPERT_PLAN_OK;
}

static glm53_expert_plan_status aligned_extent(
        const glm53_expert_extent *logical, glm53_expert_extent *physical,
        char *error, size_t error_size) {
    uint64_t end;
    const uint64_t mask = GLM53_EXPERT_IO_ALIGNMENT - 1u;
    if (!add_u64(logical->offset, logical->length, &end) ||
        end > UINT64_MAX - mask)
        return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                      "aligned extent overflows");
    physical->shard = logical->shard;
    physical->offset = logical->offset & ~mask;
    end = (end + mask) & ~mask;
    physical->length = end - physical->offset;
    if (physical->length == 0u)
        return reject(error, error_size, GLM53_EXPERT_PLAN_BAD_TENSOR,
                      "zero physical extent");
    return GLM53_EXPERT_PLAN_OK;
}

glm53_expert_plan_status glm53_expert_plan_build(
    glm53_expert_plan *plan, const k3_st_model *model,
    uint16_t layer, uint16_t expert, const k3_st_tensor *const *tensors,
    size_t tensor_count, char *error, size_t error_size) {
    glm53_expert_plan temp;
    const k3_st_tensor *weights[3];
    const k3_st_tensor *scales[3];
    size_t i;
    glm53_expert_plan_status status;
    uint64_t end;

    if (error && error_size) error[0] = '\0';
    if (!plan || !model || !model->shards || !tensors ||
        layer < GLM53_EXPERT_FIRST_LAYER || layer > GLM53_EXPERT_LAST_LAYER ||
        expert >= GLM53_EXPERTS_PER_LAYER)
        return reject(error, error_size, GLM53_EXPERT_PLAN_INVALID_ARGUMENT,
                      "invalid expert-plan arguments");
    if (tensor_count != GLM53_EXPERT_TENSOR_COUNT)
        return reject(error, error_size, GLM53_EXPERT_PLAN_INCOMPLETE_MODEL,
                      "expert must contain exactly six tensors");
    memset(&temp, 0, sizeof(temp));
    temp.layer = layer;
    temp.expert = expert;
    for (i = 0u; i < tensor_count; ++i) {
        uint16_t got_layer, got_expert;
        uint8_t role;
        int parsed;
        const k3_st_tensor *tensor = tensors[i];
        if (!tensor || !tensor->name)
            return reject(error, error_size, GLM53_EXPERT_PLAN_BAD_TENSOR,
                          "null tensor metadata");
        parsed = parse_name(tensor->name, &got_layer, &got_expert, &role);
        if (parsed != 1 || got_layer != layer || got_expert != expert)
            return reject(error, error_size, GLM53_EXPERT_PLAN_MALFORMED_NAME,
                          "tensor name does not identify requested expert");
        if (temp.tensors[role])
            return reject(error, error_size, GLM53_EXPERT_PLAN_DUPLICATE_ROLE,
                          "duplicate expert tensor role");
        if (!tensor_geometry(tensor, role) || tensor->byte_length == 0u)
            return reject(error, error_size, GLM53_EXPERT_PLAN_BAD_TENSOR,
                          "expert tensor dtype, shape, or size is wrong");
        if ((size_t)tensor->shard >= model->shard_count)
            return reject(error, error_size, GLM53_EXPERT_PLAN_OUT_OF_BOUNDS,
                          "tensor shard is out of bounds");
        if (!add_u64(tensor->physical_offset, tensor->byte_length, &end))
            return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                          "tensor extent overflows");
        if (tensor->physical_offset < model->shards[tensor->shard].data_offset ||
            end > model->shards[tensor->shard].file_bytes)
            return reject(error, error_size, GLM53_EXPERT_PLAN_OUT_OF_BOUNDS,
                          "tensor extent is outside shard data");
        temp.tensors[role] = tensor;
    }
    temp.shard = temp.tensors[0]->shard;
    for (i = 1u; i < GLM53_EXPERT_TENSOR_COUNT; ++i) {
        if (!temp.tensors[i])
            return reject(error, error_size, GLM53_EXPERT_PLAN_INCOMPLETE_MODEL,
                          "expert tensor role is missing");
        if (temp.tensors[i]->shard != temp.shard)
            return reject(error, error_size, GLM53_EXPERT_PLAN_CROSS_SHARD,
                          "expert tensors cross shards");
    }
    weights[0] = temp.tensors[GLM53_EXPERT_DOWN_WEIGHT];
    weights[1] = temp.tensors[GLM53_EXPERT_GATE_WEIGHT];
    weights[2] = temp.tensors[GLM53_EXPERT_UP_WEIGHT];
    scales[0] = temp.tensors[GLM53_EXPERT_DOWN_SCALE];
    scales[1] = temp.tensors[GLM53_EXPERT_GATE_SCALE];
    scales[2] = temp.tensors[GLM53_EXPERT_UP_SCALE];
    status = make_logical_extent(&temp.logical[0], weights,
                                 GLM53_EXPERT_WEIGHT_BYTES, error, error_size);
    if (status != GLM53_EXPERT_PLAN_OK) return status;
    status = make_logical_extent(&temp.logical[1], scales,
                                 GLM53_EXPERT_SCALE_BYTES, error, error_size);
    if (status != GLM53_EXPERT_PLAN_OK) return status;
    {
        uint64_t weight_end;
        uint64_t scale_end;
        if (!add_u64(temp.logical[0].offset, temp.logical[0].length,
                     &weight_end) ||
            !add_u64(temp.logical[1].offset, temp.logical[1].length,
                     &scale_end))
            return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                          "logical extent overflows");
        if (temp.logical[0].offset < scale_end &&
            temp.logical[1].offset < weight_end)
            return reject(error, error_size, GLM53_EXPERT_PLAN_BAD_TENSOR,
                          "weight and scale extents overlap");
    }
    for (i = 0u; i < 2u; ++i) {
        status = aligned_extent(&temp.logical[i], &temp.physical[i],
                                error, error_size);
        if (status != GLM53_EXPERT_PLAN_OK) return status;
        if (!add_u64(temp.physical[i].offset, temp.physical[i].length, &end))
            return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                          "aligned read overflows");
        if (end > model->shards[temp.shard].file_bytes) {
            /* Match k3_st_read_span*: use an exact buffered read at EOF. */
            temp.physical[i] = temp.logical[i];
            if (!add_u64(temp.physical[i].offset, temp.physical[i].length,
                         &end) ||
                end > model->shards[temp.shard].file_bytes)
                return reject(error, error_size,
                              GLM53_EXPERT_PLAN_OUT_OF_BOUNDS,
                              "read exceeds shard file");
        }
    }
    if (temp.physical[1].offset < temp.physical[0].offset) {
        glm53_expert_extent swap = temp.physical[0];
        temp.physical[0] = temp.physical[1]; temp.physical[1] = swap;
    }
    temp.physical_count = 2u;
    if (!add_u64(temp.physical[0].offset, temp.physical[0].length, &end))
        return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                      "physical ledger overflows");
    if (temp.physical[1].offset <= end) {
        uint64_t second_end;
        if (!add_u64(temp.physical[1].offset, temp.physical[1].length,
                     &second_end))
            return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                          "physical ledger overflows");
        if (second_end > end) end = second_end;
        temp.physical[0].length = end - temp.physical[0].offset;
        memset(&temp.physical[1], 0, sizeof(temp.physical[1]));
        temp.physical_count = 1u;
    }
    temp.logical_bytes = GLM53_EXPERT_LOGICAL_BYTES;
    temp.physical_bytes = temp.physical[0].length;
    if (temp.physical_count == 2u &&
        !add_u64(temp.physical_bytes, temp.physical[1].length,
                 &temp.physical_bytes))
        return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                      "physical total overflows");
    *plan = temp;
    return GLM53_EXPERT_PLAN_OK;
}

static int compare_refs(const void *left, const void *right) {
    const tensor_ref *a = (const tensor_ref *)left;
    const tensor_ref *b = (const tensor_ref *)right;
    if (a->layer != b->layer) return a->layer < b->layer ? -1 : 1;
    if (a->expert != b->expert) return a->expert < b->expert ? -1 : 1;
    if (a->role != b->role) return a->role < b->role ? -1 : 1;
    return 0;
}

void glm53_expert_model_plan_free(glm53_expert_model_plan *plan) {
    if (!plan) return;
    free(plan->experts);
    memset(plan, 0, sizeof(*plan));
}

glm53_expert_plan_status glm53_expert_model_plan_build(
    glm53_expert_model_plan *plan, const k3_st_model *model,
    char *error, size_t error_size) {
    glm53_expert_model_plan temp;
    tensor_ref *refs = NULL;
    size_t ref_count = 0u;
    size_t i;
    glm53_expert_plan_status status = GLM53_EXPERT_PLAN_OK;

    if (error && error_size) error[0] = '\0';
    if (!plan || !model || (model->tensor_count && !model->tensors) ||
        !model->shards)
        return reject(error, error_size, GLM53_EXPERT_PLAN_INVALID_ARGUMENT,
                      "invalid model-plan arguments");
    memset(&temp, 0, sizeof(temp));
    if (model->tensor_count > SIZE_MAX / sizeof(*refs))
        return reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                      "tensor directory allocation overflows");
    refs = (tensor_ref *)malloc((model->tensor_count ? model->tensor_count : 1u) *
                                sizeof(*refs));
    if (!refs)
        return reject(error, error_size, GLM53_EXPERT_PLAN_ALLOCATION_FAILED,
                      "cannot allocate routed tensor index");
    for (i = 0u; i < model->tensor_count; ++i) {
        uint16_t layer, expert;
        uint8_t role;
        const int parsed = parse_name(model->tensors[i].name,
                                      &layer, &expert, &role);
        if (parsed < 0) {
            status = reject(error, error_size, GLM53_EXPERT_PLAN_MALFORMED_NAME,
                            "malformed or excluded routed expert name");
            goto done;
        }
        if (parsed > 0) {
            refs[ref_count].layer = layer;
            refs[ref_count].expert = expert;
            refs[ref_count].role = role;
            refs[ref_count].tensor = &model->tensors[i];
            ++ref_count;
        }
    }
    if (ref_count != (size_t)GLM53_EXPERT_COUNT * GLM53_EXPERT_TENSOR_COUNT) {
        status = reject(error, error_size, GLM53_EXPERT_PLAN_INCOMPLETE_MODEL,
                        "model does not contain exactly 12096 routed experts");
        goto done;
    }
    qsort(refs, ref_count, sizeof(*refs), compare_refs);
    temp.experts = (glm53_expert_plan *)calloc(GLM53_EXPERT_COUNT,
                                               sizeof(*temp.experts));
    if (!temp.experts) {
        status = reject(error, error_size, GLM53_EXPERT_PLAN_ALLOCATION_FAILED,
                        "cannot allocate expert plans");
        goto done;
    }
    temp.expert_count = GLM53_EXPERT_COUNT;
    for (i = 0u; i < GLM53_EXPERT_COUNT; ++i) {
        const uint16_t wanted_layer = (uint16_t)(GLM53_EXPERT_FIRST_LAYER +
                                  i / GLM53_EXPERTS_PER_LAYER);
        const uint16_t wanted_expert = (uint16_t)(i % GLM53_EXPERTS_PER_LAYER);
        const k3_st_tensor *six[GLM53_EXPERT_TENSOR_COUNT];
        size_t j;
        for (j = 0u; j < GLM53_EXPERT_TENSOR_COUNT; ++j) {
            const tensor_ref *ref = &refs[i * GLM53_EXPERT_TENSOR_COUNT + j];
            if (ref->layer != wanted_layer || ref->expert != wanted_expert ||
                ref->role != j) {
                status = reject(error, error_size,
                    ref->layer == wanted_layer && ref->expert == wanted_expert ?
                    GLM53_EXPERT_PLAN_DUPLICATE_ROLE :
                    GLM53_EXPERT_PLAN_INCOMPLETE_MODEL,
                    "missing or duplicate routed tensor role");
                goto done;
            }
            six[j] = ref->tensor;
        }
        status = glm53_expert_plan_build(&temp.experts[i], model,
                                         wanted_layer, wanted_expert, six,
                                         GLM53_EXPERT_TENSOR_COUNT,
                                         error, error_size);
        if (status != GLM53_EXPERT_PLAN_OK) goto done;
        if (!add_u64(temp.logical_bytes, temp.experts[i].logical_bytes,
                     &temp.logical_bytes) ||
            !add_u64(temp.physical_bytes, temp.experts[i].physical_bytes,
                     &temp.physical_bytes)) {
            status = reject(error, error_size, GLM53_EXPERT_PLAN_OVERFLOW,
                            "model plan totals overflow");
            goto done;
        }
    }
    *plan = temp;
    memset(&temp, 0, sizeof(temp));
done:
    free(refs);
    glm53_expert_model_plan_free(&temp);
    return status;
}
