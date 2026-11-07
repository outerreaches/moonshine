#include "k31_dsa_memory.h"

#include <string.h>

static bool add_u64(uint64_t *total, uint64_t value) {
    if (*total > UINT64_MAX - value) return false;
    *total += value;
    return true;
}

static bool multiply_u64(uint64_t left, uint64_t right, uint64_t *product) {
    if (left != 0u && right > UINT64_MAX / left) return false;
    *product = left * right;
    return true;
}

bool k31_dsa_memory_plan_build(
        uint32_t context,
        uint32_t dsa_layers,
        uint32_t mla_elements_per_token,
        uint32_t index_elements_per_token,
        uint32_t index_heads,
        uint32_t attention_heads,
        uint32_t latent_elements,
        uint32_t top_k,
        k31_dsa_memory_plan *plan) {
    if (!plan || context == 0u || dsa_layers == 0u ||
        mla_elements_per_token == 0u || index_elements_per_token == 0u ||
        index_heads == 0u || attention_heads == 0u || latent_elements == 0u ||
        latent_elements > mla_elements_per_token || top_k == 0u) {
        return false;
    }
    k31_dsa_memory_plan candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.context = context;
    candidate.dsa_layers = dsa_layers;
    candidate.mla_elements_per_token = mla_elements_per_token;
    candidate.index_elements_per_token = index_elements_per_token;
    candidate.attention_heads = attention_heads;
    candidate.latent_elements = latent_elements;
    candidate.top_k = top_k;

    if (!multiply_u64(context, (uint64_t)mla_elements_per_token * sizeof(uint16_t),
                      &candidate.mla_layer_bytes) ||
        !multiply_u64(context, (uint64_t)index_elements_per_token * sizeof(uint16_t),
                      &candidate.index_layer_bytes) ||
        !multiply_u64(candidate.mla_layer_bytes, dsa_layers,
                      &candidate.mla_cache_bytes) ||
        !multiply_u64(candidate.index_layer_bytes, dsa_layers,
                      &candidate.index_cache_bytes)) {
        return false;
    }

    uint64_t term = 0u;
    if (!multiply_u64(context, sizeof(float),
                      &candidate.attention_workspace_bytes) ||
        !multiply_u64(top_k, sizeof(uint32_t), &term) ||
        !add_u64(&candidate.attention_workspace_bytes, term) ||
        !multiply_u64((uint64_t)attention_heads * top_k,
                      sizeof(float) + sizeof(uint16_t), &term) ||
        !add_u64(&candidate.attention_workspace_bytes, term) ||
        !multiply_u64((uint64_t)attention_heads * latent_elements,
                      sizeof(uint16_t), &term) ||
        !add_u64(&candidate.attention_workspace_bytes, term) ||
        !multiply_u64((uint64_t)top_k * mla_elements_per_token,
                      sizeof(uint16_t), &candidate.gathered_cache_bytes) ||
        !add_u64(&candidate.attention_workspace_bytes,
                 candidate.gathered_cache_bytes) ||
        !multiply_u64((uint64_t)context * index_heads, sizeof(float),
                      &candidate.index_dot_bytes) ||
        !add_u64(&candidate.attention_workspace_bytes,
                 candidate.index_dot_bytes)) {
        return false;
    }
    *plan = candidate;
    return true;
}

bool k31_dsa_cache_offsets(
        const k31_dsa_memory_plan *plan,
        uint32_t dsa_layer,
        uint32_t token_position,
        uint64_t *mla_offset,
        uint64_t *index_offset) {
    if (!plan || !mla_offset || !index_offset ||
        dsa_layer >= plan->dsa_layers || token_position >= plan->context) {
        return false;
    }
    uint64_t mla_row = 0u;
    uint64_t index_row = 0u;
    if (!multiply_u64(token_position,
                      (uint64_t)plan->mla_elements_per_token * sizeof(uint16_t),
                      &mla_row) ||
        !multiply_u64(token_position,
                      (uint64_t)plan->index_elements_per_token * sizeof(uint16_t),
                      &index_row)) {
        return false;
    }
    *mla_offset = (uint64_t)dsa_layer * plan->mla_layer_bytes + mla_row;
    *index_offset = (uint64_t)dsa_layer * plan->index_layer_bytes + index_row;
    return true;
}

bool k31_dsa_occupied_state_bytes(
        const k31_dsa_memory_plan *plan,
        uint32_t token_position,
        uint64_t *bytes) {
    if (!plan || !bytes || token_position > plan->context) return false;
    uint64_t per_position = 0u;
    const uint64_t combined_elements =
        (uint64_t)plan->mla_elements_per_token +
        plan->index_elements_per_token;
    if (!multiply_u64(plan->dsa_layers,
                      combined_elements * sizeof(uint16_t),
                      &per_position)) {
        return false;
    }
    return multiply_u64(token_position, per_position, bytes);
}
