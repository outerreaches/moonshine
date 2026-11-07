#include "glm53_residency.h"

#include <limits.h>
#include <string.h>

static bool glm53_layer_index(uint16_t model_layer, uint16_t *index) {
    if (model_layer < GLM53_RESIDENCY_FIRST_LAYER ||
        model_layer > GLM53_RESIDENCY_LAST_LAYER) {
        return false;
    }
    *index = (uint16_t)(model_layer - GLM53_RESIDENCY_FIRST_LAYER);
    return true;
}

static int glm53_find(const glm53_residency_entry *entries,
                      uint16_t count,
                      uint16_t expert_id) {
    uint16_t i;
    for (i = 0; i < count; ++i) {
        if (entries[i].expert_id == expert_id) return (int)i;
    }
    return -1;
}

static bool glm53_entries_valid(const glm53_residency_entry *entries,
                                uint16_t count,
                                uint16_t slots_per_layer) {
    bool experts[GLM53_RESIDENCY_EXPERTS_PER_LAYER] = { false };
    bool slots[GLM53_RESIDENCY_EXPERTS_PER_LAYER] = { false };
    uint16_t i;
    if (count > slots_per_layer ||
        slots_per_layer > GLM53_RESIDENCY_EXPERTS_PER_LAYER) {
        return false;
    }
    for (i = 0; i < count; ++i) {
        if (entries[i].expert_id >= GLM53_RESIDENCY_EXPERTS_PER_LAYER ||
            entries[i].slot >= slots_per_layer ||
            experts[entries[i].expert_id] || slots[entries[i].slot]) {
            return false;
        }
        experts[entries[i].expert_id] = true;
        slots[entries[i].slot] = true;
    }
    return true;
}

glm53_residency_status glm53_residency_init(
    glm53_residency *residency, uint16_t slots_per_layer) {
    if (!residency) return GLM53_RESIDENCY_INVALID_ARGUMENT;
    if (slots_per_layer == 0) {
        slots_per_layer = GLM53_RESIDENCY_DEFAULT_SLOTS_PER_LAYER;
    }
    if (slots_per_layer > GLM53_RESIDENCY_EXPERTS_PER_LAYER) {
        return GLM53_RESIDENCY_INVALID_ARGUMENT;
    }
    memset(residency, 0, sizeof(*residency));
    residency->slots_per_layer = slots_per_layer;
    return GLM53_RESIDENCY_OK;
}

glm53_residency_status glm53_residency_plan(
    glm53_residency *residency,
    uint16_t model_layer,
    const uint16_t *expert_ids,
    uint16_t route_count,
    glm53_residency_access *accesses) {
    uint16_t layer;
    uint16_t i;
    uint16_t j;
    uint16_t hit_count = 0;
    uint16_t survivor_count = 0;
    uint16_t drop_count;
    uint16_t next_count = 0;
    bool route_hit[GLM53_RESIDENCY_EXPERTS_PER_LAYER];
    bool used[GLM53_RESIDENCY_EXPERTS_PER_LAYER];
    glm53_residency_entry next[GLM53_RESIDENCY_EXPERTS_PER_LAYER];
    glm53_residency_access result[GLM53_RESIDENCY_EXPERTS_PER_LAYER];
    const glm53_residency_entry *current;
    uint16_t current_count;

    if (!residency || !expert_ids || !accesses || route_count == 0 ||
        residency->slots_per_layer == 0 ||
        residency->slots_per_layer > GLM53_RESIDENCY_EXPERTS_PER_LAYER) {
        return GLM53_RESIDENCY_INVALID_ARGUMENT;
    }
    if (!glm53_layer_index(model_layer, &layer)) {
        return GLM53_RESIDENCY_INVALID_LAYER;
    }
    if (route_count > residency->slots_per_layer) {
        return GLM53_RESIDENCY_TOO_MANY_ROUTES;
    }
    if (residency->pending[layer]) return GLM53_RESIDENCY_PLAN_PENDING;

    for (i = 0; i < route_count; ++i) {
        if (expert_ids[i] >= GLM53_RESIDENCY_EXPERTS_PER_LAYER) {
            return GLM53_RESIDENCY_INVALID_EXPERT;
        }
        for (j = 0; j < i; ++j) {
            if (expert_ids[i] == expert_ids[j]) {
                return GLM53_RESIDENCY_DUPLICATE_EXPERT;
            }
        }
    }

    current = residency->entries[layer];
    current_count = residency->count[layer];
    if (!glm53_entries_valid(current, current_count,
                             residency->slots_per_layer)) {
        return GLM53_RESIDENCY_INVALID_ARGUMENT;
    }
    memset(used, 0, sizeof(used));

    /* Identify every hit before choosing victims. Thus a route hit is never
     * the "oldest untouched" victim of a miss in the same plan. */
    for (i = 0; i < route_count; ++i) {
        int old_index = glm53_find(current, current_count, expert_ids[i]);
        route_hit[i] = old_index >= 0;
        if (route_hit[i]) ++hit_count;
    }
    survivor_count = (uint16_t)(current_count - hit_count);
    drop_count = survivor_count + route_count > residency->slots_per_layer
                     ? (uint16_t)(survivor_count + route_count -
                                  residency->slots_per_layer)
                     : 0;

    /* Keep untouched residents in their old relative LRU order, dropping the
     * oldest ones first. */
    for (i = 0; i < current_count; ++i) {
        bool touched = false;
        for (j = 0; j < route_count; ++j) {
            if (current[i].expert_id == expert_ids[j]) {
                touched = true;
                break;
            }
        }
        if (touched) continue;
        if (drop_count != 0) {
            --drop_count;
            continue;
        }
        if (current[i].slot >= residency->slots_per_layer ||
            used[current[i].slot]) {
            return GLM53_RESIDENCY_INVALID_ARGUMENT;
        }
        next[next_count++] = current[i];
        used[current[i].slot] = true;
    }

    /* Append all routes in touch order. Hits retain their committed slots. */
    for (i = 0; i < route_count; ++i) {
        int old_index = glm53_find(current, current_count, expert_ids[i]);
        next[next_count].expert_id = expert_ids[i];
        next[next_count].slot = GLM53_RESIDENCY_NO_SLOT;
        result[i].hit = route_hit[i];
        result[i].source_slot = GLM53_RESIDENCY_NO_SLOT;
        result[i].destination_slot = GLM53_RESIDENCY_NO_SLOT;
        if (old_index >= 0) {
            uint16_t slot = current[old_index].slot;
            if (slot >= residency->slots_per_layer || used[slot]) {
                return GLM53_RESIDENCY_INVALID_ARGUMENT;
            }
            next[next_count].slot = slot;
            used[slot] = true;
            result[i].source_slot = slot;
        }
        ++next_count;
    }

    /* Give each miss a distinct free layer-local slot. Slots belonging to an
     * evicted entry can be overwritten only after the caller commits. */
    for (i = 0; i < route_count; ++i) {
        uint16_t slot;
        uint16_t next_index;
        if (route_hit[i]) continue;
        for (slot = 0; slot < residency->slots_per_layer; ++slot) {
            if (!used[slot]) break;
        }
        if (slot == residency->slots_per_layer) {
            return GLM53_RESIDENCY_INVALID_ARGUMENT;
        }
        used[slot] = true;
        next_index = (uint16_t)(next_count - route_count + i);
        next[next_index].slot = slot;
        result[i].destination_slot = slot;
    }

    memcpy(residency->pending_entries[layer], next,
           (size_t)next_count * sizeof(next[0]));
    residency->pending_count[layer] = next_count;
    residency->pending[layer] = true;
    memcpy(accesses, result, (size_t)route_count * sizeof(result[0]));
    return GLM53_RESIDENCY_OK;
}

glm53_residency_status glm53_residency_commit(
    glm53_residency *residency, uint16_t model_layer) {
    uint16_t layer;
    uint16_t count;
    if (!residency) return GLM53_RESIDENCY_INVALID_ARGUMENT;
    if (!glm53_layer_index(model_layer, &layer)) {
        return GLM53_RESIDENCY_INVALID_LAYER;
    }
    if (!residency->pending[layer]) {
        return GLM53_RESIDENCY_NO_PENDING_PLAN;
    }
    count = residency->pending_count[layer];
    if (!glm53_entries_valid(residency->pending_entries[layer], count,
                             residency->slots_per_layer)) {
        return GLM53_RESIDENCY_INVALID_ARGUMENT;
    }
    memcpy(residency->entries[layer], residency->pending_entries[layer],
           (size_t)count * sizeof(residency->entries[layer][0]));
    residency->count[layer] = count;
    residency->pending[layer] = false;
    residency->pending_count[layer] = 0;
    return GLM53_RESIDENCY_OK;
}

glm53_residency_status glm53_residency_abort(
    glm53_residency *residency, uint16_t model_layer) {
    uint16_t layer;
    if (!residency) return GLM53_RESIDENCY_INVALID_ARGUMENT;
    if (!glm53_layer_index(model_layer, &layer)) {
        return GLM53_RESIDENCY_INVALID_LAYER;
    }
    if (!residency->pending[layer]) {
        return GLM53_RESIDENCY_NO_PENDING_PLAN;
    }
    residency->pending[layer] = false;
    residency->pending_count[layer] = 0;
    return GLM53_RESIDENCY_OK;
}

glm53_residency_status glm53_residency_snapshot(
    const glm53_residency *residency,
    uint16_t model_layer,
    uint16_t *expert_ids,
    uint16_t expert_capacity,
    uint16_t *expert_count) {
    uint16_t layer;
    uint16_t i;
    uint16_t count;
    if (expert_count) *expert_count = 0;
    if (!residency || !expert_count) {
        return GLM53_RESIDENCY_INVALID_ARGUMENT;
    }
    if (!glm53_layer_index(model_layer, &layer)) {
        return GLM53_RESIDENCY_INVALID_LAYER;
    }
    count = residency->count[layer];
    if (!glm53_entries_valid(residency->entries[layer], count,
                             residency->slots_per_layer) ||
        expert_capacity < count || (count != 0 && !expert_ids)) {
        return GLM53_RESIDENCY_INVALID_ARGUMENT;
    }
    for (i = 0; i < count; ++i) {
        expert_ids[i] = residency->entries[layer][i].expert_id;
    }
    *expert_count = count;
    return GLM53_RESIDENCY_OK;
}

uint32_t glm53_residency_slot_count(const glm53_residency *residency) {
    if (!residency || residency->slots_per_layer == 0 ||
        residency->slots_per_layer > GLM53_RESIDENCY_EXPERTS_PER_LAYER) {
        return 0;
    }
    return (uint32_t)GLM53_RESIDENCY_LAYER_COUNT *
           residency->slots_per_layer;
}

glm53_residency_status glm53_residency_logical_capacity(
    uint16_t slots_per_layer,
    uint64_t bytes_per_expert,
    uint64_t *bytes_out) {
    uint64_t slots;
    if (!bytes_out) return GLM53_RESIDENCY_INVALID_ARGUMENT;
    *bytes_out = 0;
    if (slots_per_layer == 0) {
        slots_per_layer = GLM53_RESIDENCY_DEFAULT_SLOTS_PER_LAYER;
    }
    if (slots_per_layer > GLM53_RESIDENCY_EXPERTS_PER_LAYER) {
        return GLM53_RESIDENCY_INVALID_ARGUMENT;
    }
    slots = (uint64_t)GLM53_RESIDENCY_LAYER_COUNT * slots_per_layer;
    if (bytes_per_expert != 0 && slots > UINT64_MAX / bytes_per_expert) {
        return GLM53_RESIDENCY_OVERFLOW;
    }
    *bytes_out = slots * bytes_per_expert;
    return GLM53_RESIDENCY_OK;
}
