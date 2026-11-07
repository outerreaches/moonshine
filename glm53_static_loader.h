#ifndef GLM53_STATIC_LOADER_H
#define GLM53_STATIC_LOADER_H

#include "glm53_static_layout.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_STATIC_LOADER_CHUNK_BYTES (UINT64_C(32) * UINT64_C(1024) * UINT64_C(1024))
#define GLM53_STATIC_LOADER_IO_BYTES (GLM53_STATIC_LOADER_CHUNK_BYTES + UINT64_C(8192))
#define GLM53_STATIC_LOADER_IO_ALIGNMENT UINT64_C(4096)

typedef bool (*glm53_static_loader_cancel_fn)(void *context);

typedef enum {
    GLM53_STATIC_LOADER_OK = 0,
    GLM53_STATIC_LOADER_INVALID_ARGUMENT,
    GLM53_STATIC_LOADER_BAD_LAYOUT,
    GLM53_STATIC_LOADER_ALLOCATION_FAILED,
    GLM53_STATIC_LOADER_READ_FAILED,
    GLM53_STATIC_LOADER_HIP_FAILED,
    GLM53_STATIC_LOADER_CANCELLED,
    GLM53_STATIC_LOADER_OVERFLOW,
    GLM53_STATIC_LOADER_LEDGER_MISMATCH
} glm53_static_loader_status;

typedef struct {
    char *name;                 /* Owned by the store. */
    k3_st_dtype dtype;
    uint64_t device_offset;
    uint64_t logical_bytes;
    uint64_t crc64_ecma;
} glm53_static_runtime_entry;

typedef struct {
    uint64_t logical_read_bytes;
    uint64_t logical_submitted_bytes;
    uint64_t logical_completed_bytes;
    uint64_t physical_read_bytes;
    uint64_t direct_logical_bytes;
    uint64_t direct_physical_bytes;
    uint64_t buffered_logical_bytes;
    uint64_t buffered_physical_bytes;
    uint64_t device_allocation_bytes;
    uint64_t aggregate_crc64_ecma;
    uint64_t read_requests;
    uint64_t submitted_requests;
    uint64_t completed_requests;
    uint64_t direct_requests;
    uint64_t buffered_requests;
    size_t entry_count;
} glm53_static_loader_stats;

typedef struct glm53_static_store glm53_static_store;

/*
 * Load a built static layout into one contiguous device allocation.  The
 * model and layout are borrowed only for the duration of this call.  On
 * failure *out is unchanged.  `stats`, when non-NULL, receives the completed
 * (or partial failure/cancellation) ledgers.
 *
 * Per-entry CRC64 uses CRC64-ECMA-182 (poly 0x42f0e1eba9ea3693, init/xor 0).
 * The aggregate applies the same CRC, in entry order, to this unambiguous
 * stream: name length (LE64), name bytes, dtype (LE32), device offset (LE64),
 * logical length (LE64), then the logical payload bytes.
 */
glm53_static_loader_status glm53_static_loader_load(
    glm53_static_store **out,
    const k3_st_model *model,
    const glm53_static_layout *layout,
    glm53_static_loader_cancel_fn cancelled,
    void *cancel_context,
    glm53_static_loader_stats *stats,
    char *error,
    size_t error_size);

const glm53_static_runtime_entry *glm53_static_store_find(
    const glm53_static_store *store, const char *name);
void *glm53_static_store_device_pointer(
    const glm53_static_store *store, const glm53_static_runtime_entry *entry);
void *glm53_static_store_device_base(const glm53_static_store *store);
size_t glm53_static_store_entry_count(const glm53_static_store *store);
uint64_t glm53_static_store_device_bytes(const glm53_static_store *store);
const glm53_static_loader_stats *glm53_static_store_stats(
    const glm53_static_store *store);
void glm53_static_store_destroy(glm53_static_store *store);
const char *glm53_static_loader_status_string(glm53_static_loader_status status);

#ifdef __cplusplus
}
#endif
#endif
