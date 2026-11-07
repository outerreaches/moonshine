#ifndef K31_DSA_MEMORY_H
#define K31_DSA_MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t context;
    uint32_t dsa_layers;
    uint32_t mla_elements_per_token;
    uint32_t index_elements_per_token;
    uint32_t attention_heads;
    uint32_t latent_elements;
    uint32_t top_k;
    uint64_t mla_layer_bytes;
    uint64_t index_layer_bytes;
    uint64_t mla_cache_bytes;
    uint64_t index_cache_bytes;
    uint64_t gathered_cache_bytes;
    uint64_t index_dot_bytes;
    uint64_t attention_workspace_bytes;
} k31_dsa_memory_plan;

/* All cache elements are BF16. Attention scores are F32 and probabilities BF16. */
bool k31_dsa_memory_plan_build(
        uint32_t context,
        uint32_t dsa_layers,
        uint32_t mla_elements_per_token,
        uint32_t index_elements_per_token,
        uint32_t index_heads,
        uint32_t attention_heads,
        uint32_t latent_elements,
        uint32_t top_k,
        k31_dsa_memory_plan *plan);

/* Byte offsets from the start of the respective persistent cache allocation. */
bool k31_dsa_cache_offsets(
        const k31_dsa_memory_plan *plan,
        uint32_t dsa_layer,
        uint32_t token_position,
        uint64_t *mla_offset,
        uint64_t *index_offset);

/* Occupied MLA + indexer semantic state, excluding derived top-k indices. */
bool k31_dsa_occupied_state_bytes(
        const k31_dsa_memory_plan *plan,
        uint32_t token_position,
        uint64_t *bytes);

#ifdef __cplusplus
}
#endif

#endif
