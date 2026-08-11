#include "k3_expert_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K3_CACHE_EMPTY UINT16_MAX

typedef struct {
    uint16_t expert;
    uint16_t slot;
} k3_cache_entry;

typedef struct {
    bool active;
    uint16_t count;
    uint16_t accesses;
    uint16_t hits;
    uint16_t admissions;
    uint16_t evictions;
} k3_cache_pending;

struct k3_expert_cache {
    uint16_t layer_count;
    uint16_t capacity;
    uint32_t slot_count;
    uint16_t *counts;
    k3_cache_entry *entries;
    k3_cache_entry *pending_entries;
    k3_cache_pending *pending;
    k3_expert_cache_stats stats;
};

static void k3_cache_error(char *error, size_t error_size,
                           const char *message) {
    if (error && error_size) {
        snprintf(error, error_size, "%s", message);
    }
}

static k3_cache_entry *k3_cache_layer(k3_cache_entry *entries,
                                      const k3_expert_cache *cache,
                                      uint16_t layer) {
    return entries + (size_t)layer * cache->capacity;
}

static int k3_cache_find(const k3_cache_entry *entries,
                         uint16_t count,
                         uint16_t expert) {
    for (uint16_t i = 0; i < count; i++) {
        if (entries[i].expert == expert) return (int)i;
    }
    return -1;
}

static uint32_t k3_cache_global_slot(const k3_expert_cache *cache,
                                     uint16_t layer,
                                     uint16_t slot) {
    return (uint32_t)layer * cache->capacity + slot;
}

void k3_expert_cache_destroy(k3_expert_cache *cache) {
    if (!cache) return;
    free(cache->pending);
    free(cache->pending_entries);
    free(cache->entries);
    free(cache->counts);
    free(cache);
}

bool k3_expert_cache_create(k3_expert_cache **out,
                            uint16_t layer_count,
                            uint16_t slots_per_layer,
                            char *error,
                            size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (out) *out = NULL;
    if (!out || layer_count == 0 || slots_per_layer == 0) {
        k3_cache_error(error, error_size,
                       "invalid expert-cache dimensions");
        return false;
    }
    uint32_t slot_count = (uint32_t)layer_count * slots_per_layer;

    k3_expert_cache *cache =
        (k3_expert_cache *)calloc(1, sizeof(*cache));
    if (!cache) {
        k3_cache_error(error, error_size,
                       "expert-cache allocation failed");
        return false;
    }
    cache->layer_count = layer_count;
    cache->capacity = slots_per_layer;
    cache->slot_count = slot_count;
    cache->counts = (uint16_t *)calloc(layer_count, sizeof(*cache->counts));
    cache->entries =
        (k3_cache_entry *)malloc((size_t)slot_count *
                                 sizeof(*cache->entries));
    cache->pending_entries =
        (k3_cache_entry *)malloc((size_t)slot_count *
                                 sizeof(*cache->pending_entries));
    cache->pending =
        (k3_cache_pending *)calloc(layer_count, sizeof(*cache->pending));
    if (!cache->counts || !cache->entries || !cache->pending_entries ||
        !cache->pending) {
        k3_cache_error(error, error_size,
                       "expert-cache table allocation failed");
        k3_expert_cache_destroy(cache);
        return false;
    }
    for (uint32_t i = 0; i < slot_count; i++) {
        cache->entries[i].expert = K3_CACHE_EMPTY;
        cache->entries[i].slot = (uint16_t)(i % slots_per_layer);
        cache->pending_entries[i] = cache->entries[i];
    }
    *out = cache;
    return true;
}

bool k3_expert_cache_plan(k3_expert_cache *cache,
                          uint16_t layer,
                          const uint16_t *expert_ids,
                          uint16_t expert_count,
                          k3_expert_cache_access *accesses,
                          char *error,
                          size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!cache || layer >= cache->layer_count || !expert_ids ||
        expert_count == 0 || !accesses) {
        k3_cache_error(error, error_size,
                       "invalid expert-cache plan arguments");
        return false;
    }
    if (cache->pending[layer].active) {
        k3_cache_error(error, error_size,
                       "expert-cache layer already has a pending plan");
        return false;
    }
    for (uint16_t i = 0; i < expert_count; i++) {
        for (uint16_t j = 0; j < i; j++) {
            if (expert_ids[i] == expert_ids[j]) {
                k3_cache_error(error, error_size,
                               "expert-cache batch contains duplicate IDs");
                return false;
            }
        }
    }

    const k3_cache_entry *current =
        k3_cache_layer(cache->entries, cache, layer);
    k3_cache_entry *next =
        k3_cache_layer(cache->pending_entries, cache, layer);
    uint16_t current_count = cache->counts[layer];
    uint16_t next_count = current_count;
    memcpy(next, current, (size_t)current_count * sizeof(*next));

    uint16_t hit_count = 0;
    for (uint16_t rank = 0; rank < expert_count; rank++) {
        accesses[rank].hit = false;
        accesses[rank].admit = false;
        accesses[rank].source_slot = K3_EXPERT_CACHE_NO_SLOT;
        accesses[rank].destination_slot = K3_EXPERT_CACHE_NO_SLOT;
        int current_index =
            k3_cache_find(current, current_count, expert_ids[rank]);
        if (current_index >= 0) {
            accesses[rank].hit = true;
            accesses[rank].source_slot = k3_cache_global_slot(
                cache, layer, current[current_index].slot);
            hit_count++;
        }

        int next_index = k3_cache_find(next, next_count, expert_ids[rank]);
        k3_cache_entry touched = {
            expert_ids[rank],
            next_index >= 0 ? next[next_index].slot : K3_CACHE_EMPTY,
        };
        if (next_index >= 0) {
            memmove(&next[next_index], &next[next_index + 1],
                    (size_t)(next_count - (uint16_t)next_index - 1u) *
                        sizeof(*next));
            next_count--;
        } else if (next_count == cache->capacity) {
            memmove(&next[0], &next[1],
                    (size_t)(next_count - 1u) * sizeof(*next));
            next_count--;
        }
        next[next_count++] = touched;
    }

    bool used[cache->capacity];
    memset(used, 0, sizeof(used));
    for (uint16_t i = 0; i < next_count; i++) {
        if (next[i].slot != K3_CACHE_EMPTY) {
            used[next[i].slot] = true;
        }
    }
    /*
     * A hit can fall out of the intermediate LRU order and re-enter later in
     * the same top-k batch. Preserve its old slot when it is still free;
     * otherwise commit would point at a slot whose bytes were never admitted.
     */
    for (uint16_t i = 0; i < next_count; i++) {
        if (next[i].slot != K3_CACHE_EMPTY) continue;
        int old_index =
            k3_cache_find(current, current_count, next[i].expert);
        if (old_index >= 0 && !used[current[old_index].slot]) {
            next[i].slot = current[old_index].slot;
            used[next[i].slot] = true;
        }
    }
    for (uint16_t i = 0; i < next_count; i++) {
        if (next[i].slot != K3_CACHE_EMPTY) continue;
        uint16_t slot = 0;
        while (slot < cache->capacity && used[slot]) slot++;
        if (slot == cache->capacity) {
            k3_cache_error(error, error_size,
                           "expert-cache plan has no free slot");
            return false;
        }
        next[i].slot = slot;
        used[slot] = true;
    }

    uint16_t admission_count = 0;
    for (uint16_t rank = 0; rank < expert_count; rank++) {
        if (accesses[rank].hit) continue;
        int next_index =
            k3_cache_find(next, next_count, expert_ids[rank]);
        if (next_index >= 0) {
            accesses[rank].admit = true;
            accesses[rank].destination_slot = k3_cache_global_slot(
                cache, layer, next[next_index].slot);
            admission_count++;
        }
    }
    uint16_t eviction_count = 0;
    for (uint16_t i = 0; i < current_count; i++) {
        if (k3_cache_find(next, next_count, current[i].expert) < 0) {
            eviction_count++;
        }
    }

    cache->pending[layer].active = true;
    cache->pending[layer].count = next_count;
    cache->pending[layer].accesses = expert_count;
    cache->pending[layer].hits = hit_count;
    cache->pending[layer].admissions = admission_count;
    cache->pending[layer].evictions = eviction_count;
    return true;
}

bool k3_expert_cache_commit(k3_expert_cache *cache,
                            uint16_t layer,
                            char *error,
                            size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!cache || layer >= cache->layer_count ||
        !cache->pending[layer].active) {
        k3_cache_error(error, error_size,
                       "expert-cache layer has no pending plan");
        return false;
    }
    k3_cache_pending *pending = &cache->pending[layer];
    k3_cache_entry *current =
        k3_cache_layer(cache->entries, cache, layer);
    const k3_cache_entry *next =
        k3_cache_layer(cache->pending_entries, cache, layer);
    memcpy(current, next, (size_t)pending->count * sizeof(*current));
    for (uint16_t i = pending->count; i < cache->capacity; i++) {
        current[i].expert = K3_CACHE_EMPTY;
        current[i].slot = i;
    }
    cache->counts[layer] = pending->count;
    cache->stats.batches++;
    cache->stats.accesses += pending->accesses;
    cache->stats.hits += pending->hits;
    cache->stats.misses += pending->accesses - pending->hits;
    cache->stats.admissions += pending->admissions;
    cache->stats.evictions += pending->evictions;
    memset(pending, 0, sizeof(*pending));
    return true;
}

void k3_expert_cache_abort(k3_expert_cache *cache, uint16_t layer) {
    if (!cache || layer >= cache->layer_count) return;
    memset(&cache->pending[layer], 0, sizeof(cache->pending[layer]));
}

uint32_t k3_expert_cache_slot_count(const k3_expert_cache *cache) {
    return cache ? cache->slot_count : 0;
}

uint64_t k3_expert_cache_storage_bytes(const k3_expert_cache *cache,
                                       uint64_t bytes_per_expert) {
    if (!cache || (bytes_per_expert != 0 &&
                   cache->slot_count > UINT64_MAX / bytes_per_expert)) {
        return UINT64_MAX;
    }
    return (uint64_t)cache->slot_count * bytes_per_expert;
}

void k3_expert_cache_get_stats(const k3_expert_cache *cache,
                               k3_expert_cache_stats *stats) {
    if (!stats) return;
    if (cache) {
        *stats = cache->stats;
    } else {
        memset(stats, 0, sizeof(*stats));
    }
}
bool k3_expert_cache_snapshot_layer(
        const k3_expert_cache *cache,
        uint16_t layer,
        uint16_t *expert_ids,
        uint16_t expert_capacity,
        uint16_t *expert_count,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (expert_count) *expert_count = 0u;
    if (!cache || layer >= cache->layer_count || !expert_ids ||
        !expert_count || cache->pending[layer].active ||
        expert_capacity < cache->counts[layer]) {
        k3_cache_error(error, error_size,
                       "invalid expert-cache snapshot request");
        return false;
    }
    const k3_cache_entry *entries =
        k3_cache_layer(cache->entries, cache, layer);
    for (uint16_t i = 0u; i < cache->counts[layer]; i++) {
        expert_ids[i] = entries[i].expert;
    }
    *expert_count = cache->counts[layer];
    return true;
}


bool k3_expert_cache_reset(k3_expert_cache *cache,
                           char *error,
                           size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!cache) {
        k3_cache_error(error, error_size,
                       "invalid expert-cache reset");
        return false;
    }
    for (uint16_t layer = 0; layer < cache->layer_count; layer++) {
        if (cache->pending[layer].active) {
            k3_cache_error(error, error_size,
                           "cannot reset expert cache with a pending plan");
            return false;
        }
    }
    memset(cache->counts, 0,
           (size_t)cache->layer_count * sizeof(*cache->counts));
    memset(cache->pending, 0,
           (size_t)cache->layer_count * sizeof(*cache->pending));
    memset(&cache->stats, 0, sizeof(cache->stats));
    for (uint32_t i = 0; i < cache->slot_count; i++) {
        cache->entries[i].expert = K3_CACHE_EMPTY;
        cache->entries[i].slot =
            (uint16_t)(i % cache->capacity);
        cache->pending_entries[i] = cache->entries[i];
    }
    return true;
}

bool k3_expert_cache_invalidate_slots(
        k3_expert_cache *cache,
        uint32_t first_slot,
        uint32_t slot_count,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!cache || slot_count == 0u ||
        first_slot > cache->slot_count ||
        slot_count > cache->slot_count - first_slot) {
        k3_cache_error(error, error_size,
                       "invalid expert-cache invalidation range");
        return false;
    }
    for (uint16_t layer = 0u;
         layer < cache->layer_count; layer++) {
        if (cache->pending[layer].active) {
            k3_cache_error(
                error, error_size,
                "cannot invalidate expert cache with a pending plan");
            return false;
        }
    }
    const uint32_t end_slot = first_slot + slot_count;
    for (uint16_t layer = 0u;
         layer < cache->layer_count; layer++) {
        k3_cache_entry *entries =
            k3_cache_layer(cache->entries, cache, layer);
        uint16_t count = cache->counts[layer];
        uint16_t index = 0u;
        while (index < count) {
            const uint32_t global = k3_cache_global_slot(
                cache, layer, entries[index].slot);
            if (global < first_slot || global >= end_slot) {
                index++;
                continue;
            }
            memmove(
                &entries[index], &entries[index + 1u],
                (size_t)(count - index - 1u) * sizeof(*entries));
            count--;
            cache->stats.evictions++;
        }
        cache->counts[layer] = count;
        for (uint16_t i = count; i < cache->capacity; i++) {
            entries[i].expert = K3_CACHE_EMPTY;
            entries[i].slot = i;
        }
        memcpy(
            k3_cache_layer(cache->pending_entries, cache, layer),
            entries, (size_t)cache->capacity * sizeof(*entries));
    }
    return true;
}
