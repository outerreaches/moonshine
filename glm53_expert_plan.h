#ifndef GLM53_EXPERT_PLAN_H
#define GLM53_EXPERT_PLAN_H

#include "k3_safetensors.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GLM53_EXPERT_FIRST_LAYER = 3,
    GLM53_EXPERT_LAST_LAYER = 44,
    GLM53_EXPERT_LAYER_COUNT = 42,
    GLM53_EXPERTS_PER_LAYER = 288,
    GLM53_EXPERT_COUNT = 12096,
    GLM53_EXPERT_TENSOR_COUNT = 6,
    GLM53_EXPERT_LOGICAL_EXTENTS = 2,
    GLM53_EXPERT_MAX_PHYSICAL_EXTENTS = 2,
    GLM53_EXPERT_IO_ALIGNMENT = 4096
};

#define GLM53_EXPERT_WEIGHT_BYTES UINT64_C(25165824)
#define GLM53_EXPERT_SCALE_BYTES UINT64_C(6144)
#define GLM53_EXPERT_LOGICAL_BYTES UINT64_C(25171968)

typedef enum {
    GLM53_EXPERT_PLAN_OK = 0,
    GLM53_EXPERT_PLAN_INVALID_ARGUMENT,
    GLM53_EXPERT_PLAN_ALLOCATION_FAILED,
    GLM53_EXPERT_PLAN_MALFORMED_NAME,
    GLM53_EXPERT_PLAN_INCOMPLETE_MODEL,
    GLM53_EXPERT_PLAN_DUPLICATE_ROLE,
    GLM53_EXPERT_PLAN_BAD_TENSOR,
    GLM53_EXPERT_PLAN_CROSS_SHARD,
    GLM53_EXPERT_PLAN_NONCONTIGUOUS,
    GLM53_EXPERT_PLAN_OUT_OF_BOUNDS,
    GLM53_EXPERT_PLAN_OVERFLOW
} glm53_expert_plan_status;

typedef enum {
    GLM53_EXPERT_DOWN_WEIGHT = 0,
    GLM53_EXPERT_DOWN_SCALE,
    GLM53_EXPERT_GATE_WEIGHT,
    GLM53_EXPERT_GATE_SCALE,
    GLM53_EXPERT_UP_WEIGHT,
    GLM53_EXPERT_UP_SCALE
} glm53_expert_tensor_role;

typedef struct {
    uint16_t shard;
    uint64_t offset;
    uint64_t length;
} glm53_expert_extent;

typedef struct {
    uint16_t layer;
    uint16_t expert;
    uint16_t shard;
    const k3_st_tensor *tensors[GLM53_EXPERT_TENSOR_COUNT];
    /* weights first, scales second */
    glm53_expert_extent logical[GLM53_EXPERT_LOGICAL_EXTENTS];
    glm53_expert_extent physical[GLM53_EXPERT_MAX_PHYSICAL_EXTENTS];
    size_t physical_count;
    uint64_t logical_bytes;
    uint64_t physical_bytes;
} glm53_expert_plan;

typedef struct {
    glm53_expert_plan *experts; /* layer-major, then expert-major */
    size_t expert_count;
    uint64_t logical_bytes;
    uint64_t physical_bytes; /* sum of each expert's physical ledger */
} glm53_expert_model_plan;

/* Build one expert from exactly its six tensors. Input order is irrelevant. */
glm53_expert_plan_status glm53_expert_plan_build(
    glm53_expert_plan *plan,
    const k3_st_model *model,
    uint16_t layer,
    uint16_t expert,
    const k3_st_tensor *const *tensors,
    size_t tensor_count,
    char *error,
    size_t error_size);

/* Scan model metadata, reject malformed routed namespaces, and build all
 * 42*288 experts. This is O(model->tensor_count log model->tensor_count). */
glm53_expert_plan_status glm53_expert_model_plan_build(
    glm53_expert_model_plan *plan,
    const k3_st_model *model,
    char *error,
    size_t error_size);

void glm53_expert_model_plan_free(glm53_expert_model_plan *plan);

#ifdef __cplusplus
}
#endif

#endif
