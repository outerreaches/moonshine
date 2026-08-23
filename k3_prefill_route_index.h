#ifndef K3_PREFILL_ROUTE_INDEX_H
#define K3_PREFILL_ROUTE_INDEX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k3_prefill_route_index k3_prefill_route_index;

typedef struct {
    const uint32_t *tokens;
    const uint32_t *outputs;
    uint32_t count;
} k3_prefill_route_slice;

bool k3_prefill_route_index_create(
    k3_prefill_route_index **out,
    uint32_t expert_count);
void k3_prefill_route_index_destroy(k3_prefill_route_index *index);

/*
 * Build stable expert-to-row slices in O(token_count * top_k + expert_count).
 * ROUTE_IDS is token-major and rank-minor. Duplicate experts within one token
 * are rejected because one expert may contribute at most one selected row per
 * token. A failed build leaves the previous valid index intact.
 */
bool k3_prefill_route_index_build(
    k3_prefill_route_index *index,
    const uint32_t *route_ids,
    uint32_t token_count,
    uint32_t top_k);

uint32_t k3_prefill_route_index_expert_count(
    const k3_prefill_route_index *index);
uint32_t k3_prefill_route_index_token_count(
    const k3_prefill_route_index *index);
uint32_t k3_prefill_route_index_top_k(
    const k3_prefill_route_index *index);
uint32_t k3_prefill_route_index_route_count(
    const k3_prefill_route_index *index);
uint32_t k3_prefill_route_index_selected_count(
    const k3_prefill_route_index *index);

/* Borrowed slice, valid until the next successful build or destruction. */
bool k3_prefill_route_index_slice(
    const k3_prefill_route_index *index,
    uint32_t expert,
    k3_prefill_route_slice *slice);

#ifdef __cplusplus
}
#endif

#endif
