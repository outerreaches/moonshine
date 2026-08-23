#include "k3_prefill_route_index.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct k3_prefill_route_index {
    uint32_t expert_count;
    uint32_t token_count;
    uint32_t top_k;
    uint32_t route_count;
    uint32_t selected_count;
    uint32_t route_capacity;
    uint32_t *counts;
    uint32_t *offsets;
    uint32_t *next_offsets;
    uint32_t *cursors;
    uint32_t *seen;
    uint32_t generation;
    uint32_t *tokens;
    uint32_t *outputs;
    bool built;
};

static uint32_t next_generation(k3_prefill_route_index *index) {
    index->generation++;
    if (index->generation == 0u) {
        memset(index->seen, 0,
               (size_t)index->expert_count * sizeof(*index->seen));
        index->generation = 1u;
    }
    return index->generation;
}

static bool reserve_routes(
        k3_prefill_route_index *index,
        uint32_t route_count) {
    if (route_count <= index->route_capacity) return true;
    uint32_t capacity = index->route_capacity == 0u ? 64u :
        index->route_capacity;
    while (capacity < route_count) {
        if (capacity > UINT32_MAX / 2u) {
            capacity = route_count;
            break;
        }
        capacity *= 2u;
    }
    uint32_t *tokens = (uint32_t *)malloc(
        (size_t)capacity * sizeof(*tokens));
    uint32_t *outputs = (uint32_t *)malloc(
        (size_t)capacity * sizeof(*outputs));
    if (tokens == NULL || outputs == NULL) {
        free(outputs);
        free(tokens);
        return false;
    }
    free(index->outputs);
    free(index->tokens);
    index->tokens = tokens;
    index->outputs = outputs;
    index->route_capacity = capacity;
    return true;
}

bool k3_prefill_route_index_create(
        k3_prefill_route_index **out,
        uint32_t expert_count) {
    if (out != NULL) *out = NULL;
    if (out == NULL || expert_count == 0u ||
        expert_count == UINT32_MAX) {
        return false;
    }
    k3_prefill_route_index *index =
        (k3_prefill_route_index *)calloc(1u, sizeof(*index));
    if (index == NULL) return false;
    index->expert_count = expert_count;
    index->counts = (uint32_t *)calloc(
        expert_count, sizeof(*index->counts));
    index->offsets = (uint32_t *)calloc(
        (size_t)expert_count + 1u, sizeof(*index->offsets));
    index->next_offsets = (uint32_t *)calloc(
        (size_t)expert_count + 1u, sizeof(*index->next_offsets));
    index->cursors = (uint32_t *)calloc(
        expert_count, sizeof(*index->cursors));
    index->seen = (uint32_t *)calloc(
        expert_count, sizeof(*index->seen));
    if (index->counts == NULL || index->offsets == NULL ||
        index->next_offsets == NULL || index->cursors == NULL ||
        index->seen == NULL) {
        k3_prefill_route_index_destroy(index);
        return false;
    }
    *out = index;
    return true;
}

void k3_prefill_route_index_destroy(k3_prefill_route_index *index) {
    if (index == NULL) return;
    free(index->outputs);
    free(index->tokens);
    free(index->cursors);
    free(index->seen);
    free(index->next_offsets);
    free(index->offsets);
    free(index->counts);
    free(index);
}

bool k3_prefill_route_index_build(
        k3_prefill_route_index *index,
        const uint32_t *route_ids,
        uint32_t token_count,
        uint32_t top_k) {
    if (index == NULL || route_ids == NULL ||
        token_count == 0u || top_k == 0u) {
        return false;
    }
    const uint64_t route_count64 =
        (uint64_t)token_count * top_k;
    if (route_count64 > UINT32_MAX ||
        route_count64 > SIZE_MAX / sizeof(*route_ids)) {
        return false;
    }
    const uint32_t route_count = (uint32_t)route_count64;
    for (uint32_t token = 0u; token < token_count; token++) {
        const uint64_t base = (uint64_t)token * top_k;
        const uint32_t generation = next_generation(index);
        for (uint32_t rank = 0u; rank < top_k; rank++) {
            const uint32_t expert = route_ids[base + rank];
            if (expert >= index->expert_count ||
                index->seen[expert] == generation) {
                return false;
            }
            index->seen[expert] = generation;
        }
    }
    if (!reserve_routes(index, route_count)) return false;

    memset(index->counts, 0,
           (size_t)index->expert_count * sizeof(*index->counts));
    for (uint32_t route = 0u; route < route_count; route++) {
        index->counts[route_ids[route]]++;
    }
    uint32_t offset = 0u;
    uint32_t selected_count = 0u;
    for (uint32_t expert = 0u;
         expert < index->expert_count; expert++) {
        index->next_offsets[expert] = offset;
        index->cursors[expert] = offset;
        if (index->counts[expert] != 0u) selected_count++;
        offset += index->counts[expert];
    }
    index->next_offsets[index->expert_count] = offset;
    if (offset != route_count) return false;

    for (uint32_t token = 0u; token < token_count; token++) {
        const uint64_t base = (uint64_t)token * top_k;
        for (uint32_t rank = 0u; rank < top_k; rank++) {
            const uint32_t route = (uint32_t)(base + rank);
            const uint32_t expert = route_ids[route];
            const uint32_t destination = index->cursors[expert]++;
            index->tokens[destination] = token;
            index->outputs[destination] = route;
        }
    }
    memcpy(index->offsets, index->next_offsets,
           ((size_t)index->expert_count + 1u) * sizeof(*index->offsets));
    index->token_count = token_count;
    index->top_k = top_k;
    index->route_count = route_count;
    index->selected_count = selected_count;
    index->built = true;
    return true;
}

uint32_t k3_prefill_route_index_expert_count(
        const k3_prefill_route_index *index) {
    return index != NULL ? index->expert_count : 0u;
}

uint32_t k3_prefill_route_index_token_count(
        const k3_prefill_route_index *index) {
    return index != NULL && index->built ? index->token_count : 0u;
}

uint32_t k3_prefill_route_index_top_k(
        const k3_prefill_route_index *index) {
    return index != NULL && index->built ? index->top_k : 0u;
}

uint32_t k3_prefill_route_index_route_count(
        const k3_prefill_route_index *index) {
    return index != NULL && index->built ? index->route_count : 0u;
}

uint32_t k3_prefill_route_index_selected_count(
        const k3_prefill_route_index *index) {
    return index != NULL && index->built ? index->selected_count : 0u;
}

bool k3_prefill_route_index_slice(
        const k3_prefill_route_index *index,
        uint32_t expert,
        k3_prefill_route_slice *slice) {
    if (slice != NULL) memset(slice, 0, sizeof(*slice));
    if (index == NULL || !index->built ||
        expert >= index->expert_count || slice == NULL) {
        return false;
    }
    const uint32_t begin = index->offsets[expert];
    const uint32_t end = index->offsets[expert + 1u];
    slice->tokens = index->tokens + begin;
    slice->outputs = index->outputs + begin;
    slice->count = end - begin;
    return true;
}
