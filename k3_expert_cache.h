#ifndef K3_EXPERT_CACHE_H
#define K3_EXPERT_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define K3_EXPERT_CACHE_NO_SLOT UINT32_MAX

typedef struct k3_expert_cache k3_expert_cache;

typedef struct {
    bool hit;
    bool admit;
    uint32_t source_slot;
    uint32_t destination_slot;
} k3_expert_cache_access;

typedef struct {
    uint64_t batches;
    uint64_t accesses;
    uint64_t hits;
    uint64_t misses;
    uint64_t admissions;
    uint64_t evictions;
} k3_expert_cache_stats;

/*
 * Slots are caller-owned. A returned slot is a global index in
 * [0, layer_count * slots_per_layer), allowing the runtime to keep storage
 * policy separate from HIP allocation policy.
 */
bool k3_expert_cache_create(k3_expert_cache **out,
                            uint16_t layer_count,
                            uint16_t slots_per_layer,
                            char *error,
                            size_t error_size);

/*
 * Plan one token's routed-expert batch without mutating the live cache.
 *
 * Hits may be read from source_slot until commit. Misses retained by the
 * token-batch LRU have admit=true and must be copied into destination_slot
 * after all current cache readers have finished. This two-phase contract
 * prevents an admission from overwriting a hit still consumed by ROCm.
 * Exactly one plan may be pending for a layer.
 */
bool k3_expert_cache_plan(k3_expert_cache *cache,
                          uint16_t layer,
                          const uint16_t *expert_ids,
                          uint16_t expert_count,
                          k3_expert_cache_access *accesses,
                          char *error,
                          size_t error_size);

bool k3_expert_cache_commit(k3_expert_cache *cache,
                            uint16_t layer,
                            char *error,
                            size_t error_size);

void k3_expert_cache_abort(k3_expert_cache *cache, uint16_t layer);

uint32_t k3_expert_cache_slot_count(const k3_expert_cache *cache);
uint64_t k3_expert_cache_storage_bytes(const k3_expert_cache *cache,
                                       uint64_t bytes_per_expert);
void k3_expert_cache_get_stats(const k3_expert_cache *cache,
                               k3_expert_cache_stats *stats);

/*
 * Forget every resident mapping and zero telemetry without touching the
 * caller-owned slot storage. No layer may have a pending plan.
 */
bool k3_expert_cache_reset(k3_expert_cache *cache,
                           char *error,
                           size_t error_size);

void k3_expert_cache_destroy(k3_expert_cache *cache);

#ifdef __cplusplus
}
#endif

#endif
