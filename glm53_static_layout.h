#ifndef GLM53_STATIC_LAYOUT_H
#define GLM53_STATIC_LAYOUT_H

#include "glm53_weights.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_STATIC_LAYOUT_ALIGNMENT UINT64_C(256)

typedef enum {
    GLM53_STATIC_LAYOUT_OK = 0,
    GLM53_STATIC_LAYOUT_INVALID_ARGUMENT,
    GLM53_STATIC_LAYOUT_UNBUILT_PLAN,
    GLM53_STATIC_LAYOUT_BAD_PLAN,
    GLM53_STATIC_LAYOUT_DUPLICATE_TENSOR,
    GLM53_STATIC_LAYOUT_ALLOCATION_FAILED,
    GLM53_STATIC_LAYOUT_OVERFLOW
} glm53_static_layout_status;

typedef struct {
    const k3_st_tensor *tensor;
    uint64_t device_offset;
    uint16_t source_shard;
    uint64_t source_physical_offset;
    uint64_t logical_bytes;
    k3_st_dtype dtype;
} glm53_static_layout_entry;

typedef struct {
    glm53_static_layout_entry *entries;
    size_t entry_count;
    size_t tensor_count; /* Equal to entry_count; retained as a ledger name. */
    uint64_t logical_bytes;
    uint64_t padded_bytes;
    uint64_t max_tensor_bytes;
    bool built;
} glm53_static_layout;

/* Build a payload-free device map from the two resident weight-plan classes.
 * Entries are sorted by tensor name and every device offset is 256-byte
 * aligned. Streamed routed FP8 values are never added. The weight plan and
 * its backing tensor metadata must outlive the layout. On failure, layout is
 * unchanged. Free a successful layout before building another into it. */
glm53_static_layout_status glm53_static_layout_build(
    glm53_static_layout *layout,
    const glm53_weight_plan *weights,
    char *error,
    size_t error_size);

const glm53_static_layout_entry *glm53_static_layout_find_name(
    const glm53_static_layout *layout, const char *name);
const glm53_static_layout_entry *glm53_static_layout_find_tensor(
    const glm53_static_layout *layout, const k3_st_tensor *tensor);

/* Explicit by-name aliases. */
const glm53_static_layout_entry *glm53_static_layout_find_by_name(
    const glm53_static_layout *layout, const char *name);
const glm53_static_layout_entry *glm53_static_layout_find_by_tensor(
    const glm53_static_layout *layout, const k3_st_tensor *tensor);

const char *glm53_static_layout_status_string(glm53_static_layout_status status);
void glm53_static_layout_free(glm53_static_layout *layout);

#ifdef __cplusplus
}
#endif
#endif
