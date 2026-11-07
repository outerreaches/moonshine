#ifndef GLM53_STATIC_BINDINGS_H
#define GLM53_STATIC_BINDINGS_H

#include "glm53_static_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_STATIC_BINDING_COUNT \
    (GLM53_WEIGHT_RESIDENT_STATIC_COUNT + GLM53_WEIGHT_ROUTED_SCALE_COUNT)

typedef enum {
    GLM53_STATIC_BINDINGS_OK = 0,
    GLM53_STATIC_BINDINGS_INVALID_ARGUMENT,
    GLM53_STATIC_BINDINGS_UNBUILT_INPUT,
    GLM53_STATIC_BINDINGS_BAD_PLAN,
    GLM53_STATIC_BINDINGS_BAD_LAYOUT,
    GLM53_STATIC_BINDINGS_BAD_STORE,
    GLM53_STATIC_BINDINGS_MISSING_TENSOR,
    GLM53_STATIC_BINDINGS_DUPLICATE_TENSOR,
    GLM53_STATIC_BINDINGS_ALLOCATION_FAILED
} glm53_static_bindings_status;

typedef enum {
    GLM53_ROUTED_SCALE_DOWN = 0,
    GLM53_ROUTED_SCALE_GATE,
    GLM53_ROUTED_SCALE_UP,
    GLM53_ROUTED_SCALE_ROLE_COUNT
} glm53_routed_scale_role;

typedef enum {
    GLM53_STATIC_BINDING_GLOBAL = 0,
    GLM53_STATIC_BINDING_LAYER,
    GLM53_STATIC_BINDING_ROUTED_SCALE
} glm53_static_binding_kind;

/* The runtime entry and device allocation are borrowed from the store.  All
 * schema and identity fields are values, and therefore do not refer back to
 * the weight plan or static layout. */
typedef struct {
    const glm53_static_runtime_entry *runtime;
    void *device;
    glm53_static_binding_kind kind;
    k3_st_dtype dtype;
    uint64_t shape[K3_ST_MAX_DIMS];
    uint64_t logical_bytes;
    uint16_t layer;
    uint16_t expert;
    uint16_t role;
    uint8_t ndim;
} glm53_static_binding;

typedef struct {
    glm53_static_binding globals[GLM53_WEIGHT_GLOBAL_COUNT];
    glm53_static_binding layers[GLM53_WEIGHT_LAYER_COUNT]
                                  [GLM53_WEIGHT_LAYER_ROLE_COUNT];
    glm53_static_binding *routed_scales; /* layer-major, expert, down/gate/up */
    size_t resident_static_count;
    size_t routed_scale_count;
    size_t binding_count;
    bool built;
} glm53_static_bindings;

/* Publish all official runtime bindings atomically.  out must be zero-initialized
 * and unbuilt; rebuilding an existing result is rejected.  On failure, *out is
 * byte-for-byte unchanged.  The store must outlive the successful result. */
glm53_static_bindings_status glm53_static_bindings_build(
    glm53_static_bindings *out,
    const glm53_weight_plan *plan,
    const glm53_static_layout *layout,
    const glm53_static_store *store,
    char *error, size_t error_size);

/* CPU-only test seam.  It applies the same validation as build(), using a
 * sorted or unsorted array of mock runtime records and a nominal device
 * allocation.  No address is dereferenced. */
glm53_static_bindings_status glm53_static_bindings_build_mock(
    glm53_static_bindings *out,
    const glm53_weight_plan *plan,
    const glm53_static_layout *layout,
    const glm53_static_runtime_entry *entries,
    size_t entry_count,
    void *device_base,
    uint64_t device_bytes,
    char *error, size_t error_size);

const glm53_static_binding *glm53_static_bindings_global(
    const glm53_static_bindings *bindings, glm53_global_role role);
const glm53_static_binding *glm53_static_bindings_layer(
    const glm53_static_bindings *bindings, uint32_t layer,
    glm53_weight_layer_role role);
const glm53_static_binding *glm53_static_bindings_expert_scale(
    const glm53_static_bindings *bindings, uint32_t layer, uint32_t expert,
    glm53_routed_scale_role role);
/* Convenience adapter for GLM53_EXPERT_{DOWN,GATE,UP}_SCALE. */
const glm53_static_binding *glm53_static_bindings_expert_tensor_scale(
    const glm53_static_bindings *bindings, uint32_t layer, uint32_t expert,
    glm53_expert_tensor_role role);

void glm53_static_bindings_free(glm53_static_bindings *bindings);
const char *glm53_static_bindings_status_string(glm53_static_bindings_status status);

#ifdef __cplusplus
}
#endif
#endif
