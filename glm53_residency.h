#ifndef GLM53_RESIDENCY_H
#define GLM53_RESIDENCY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GLM53_RESIDENCY_FIRST_LAYER = 3,
    GLM53_RESIDENCY_LAYER_COUNT = 42,
    GLM53_RESIDENCY_LAST_LAYER = 44,
    GLM53_RESIDENCY_EXPERTS_PER_LAYER = 288,
    GLM53_RESIDENCY_DEFAULT_SLOTS_PER_LAYER = 48,
    GLM53_RESIDENCY_DEFAULT_SLOT_COUNT = 2016,
    GLM53_RESIDENCY_NO_SLOT = UINT16_MAX
};

typedef enum {
    GLM53_RESIDENCY_OK = 0,
    GLM53_RESIDENCY_INVALID_ARGUMENT,
    GLM53_RESIDENCY_INVALID_LAYER,
    GLM53_RESIDENCY_INVALID_EXPERT,
    GLM53_RESIDENCY_DUPLICATE_EXPERT,
    GLM53_RESIDENCY_TOO_MANY_ROUTES,
    GLM53_RESIDENCY_PLAN_PENDING,
    GLM53_RESIDENCY_NO_PENDING_PLAN,
    GLM53_RESIDENCY_OVERFLOW
} glm53_residency_status;

typedef struct {
    bool hit;
    uint16_t source_slot;
    uint16_t destination_slot;
} glm53_residency_access;

/* Internal metadata is public only so callers can own it without allocation. */
typedef struct {
    uint16_t expert_id;
    uint16_t slot;
} glm53_residency_entry;

typedef struct {
    uint16_t slots_per_layer;
    uint16_t count[GLM53_RESIDENCY_LAYER_COUNT];
    glm53_residency_entry
        entries[GLM53_RESIDENCY_LAYER_COUNT]
               [GLM53_RESIDENCY_EXPERTS_PER_LAYER];
    uint16_t pending_count[GLM53_RESIDENCY_LAYER_COUNT];
    glm53_residency_entry
        pending_entries[GLM53_RESIDENCY_LAYER_COUNT]
                       [GLM53_RESIDENCY_EXPERTS_PER_LAYER];
    bool pending[GLM53_RESIDENCY_LAYER_COUNT];
} glm53_residency;

/* slots_per_layer == 0 selects GLM53_RESIDENCY_DEFAULT_SLOTS_PER_LAYER. */
glm53_residency_status glm53_residency_init(
    glm53_residency *residency, uint16_t slots_per_layer);

/*
 * Plan a routed batch for model_layer in [3, 44]. The operation does not
 * change live mappings or LRU order. source_slot remains valid until commit.
 * Miss destinations and all slot values are local to the selected layer.
 * Route order is LRU touch order: the final route becomes most-recently used.
 *
 * The transaction covers metadata only. Stage and validate miss payloads in
 * separate storage. Do not overwrite a destination victim before the caller
 * has crossed an irrevocable commit boundary; abort cannot restore slot bytes.
 */
glm53_residency_status glm53_residency_plan(
    glm53_residency *residency,
    uint16_t model_layer,
    const uint16_t *expert_ids,
    uint16_t route_count,
    glm53_residency_access *accesses);

glm53_residency_status glm53_residency_commit(
    glm53_residency *residency, uint16_t model_layer);

glm53_residency_status glm53_residency_abort(
    glm53_residency *residency, uint16_t model_layer);

/* Copy the committed LRU, oldest first. No pending metadata is exposed. */
glm53_residency_status glm53_residency_snapshot(
    const glm53_residency *residency,
    uint16_t model_layer,
    uint16_t *expert_ids,
    uint16_t expert_capacity,
    uint16_t *expert_count);

uint32_t glm53_residency_slot_count(const glm53_residency *residency);

/* Checked result for 42 * slots_per_layer * bytes_per_expert. Zero slots means
 * the default. On failure, *bytes_out is set to zero when bytes_out is valid. */
glm53_residency_status glm53_residency_logical_capacity(
    uint16_t slots_per_layer,
    uint64_t bytes_per_expert,
    uint64_t *bytes_out);

#ifdef __cplusplus
}
#endif

#endif
