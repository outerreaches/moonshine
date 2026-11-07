#ifndef K31_DSA_ROCM_H
#define K31_DSA_ROCM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* BF16 LayerNorm with F32 mean/variance and BF16 gamma/beta. */
bool k31_dsa_layer_norm_bf16(
        void       *output,
        const void *input,
        const void *gamma,
        const void *beta,
        uint32_t    vector_count,
        uint32_t    head_dim,
        float       epsilon,
        void       *stream);

/*
 * BF16 queries [query_count,index_heads,index_head_dim], shared BF16 keys
 * [key_count,index_head_dim], and BF16 head weights [query_count,index_heads].
 * Optional F32 scales have the same query/head and key shapes. Output is F32
 * [query_count,key_count], with per-head ReLU and 1/sqrt(heads*dim) scaling.
 */
bool k31_dsa_index_scores_f32(
        void       *scores,
        const void *queries,
        const void *keys,
        const void *head_weights,
        const void *query_scales,
        const void *key_scales,
        uint32_t    query_count,
        uint32_t    key_count,
        uint32_t    index_heads,
        uint32_t    index_head_dim,
        void       *stream);

/* Reduce F32 [index_heads,key_count] dots after the BF16 BLAS GEMM. */
bool k31_dsa_index_reduce_f32(
        void       *scores,
        const void *head_dot_products,
        const void *head_weights,
        const void *query_scales,
        const void *key_scales,
        uint32_t    key_count,
        uint32_t    index_heads,
        uint32_t    index_head_dim,
        void       *stream);

/*
 * Return the aligned workspace required by k31_dsa_topk_f32. The input score
 * vector is not modified. Selection is score-descending with lower token IDs
 * breaking ties, matching the initial ascending-ID radix payload.
 */
bool k31_dsa_topk_workspace_bytes(
        uint32_t token_count,
        void    *stream,
        size_t  *workspace_bytes);

/* Write min(token_count, top_k) selected token IDs and pad the rest with -1. */
bool k31_dsa_topk_f32(
        void       *selected_ids,
        const void *scores,
        uint32_t    token_count,
        uint32_t    top_k,
        void       *workspace,
        size_t      workspace_bytes,
        void       *stream);

/*
 * Compressed-cache attention over selected rows. absorbed_q is BF16
 * [attention_heads, cache_dim], cache is BF16 [token_capacity, cache_dim],
 * selected_ids is uint32[selected_count], and latent_output is BF16
 * [attention_heads, latent_dim]. score_workspace is F32 and
 * probability_workspace is BF16, both [attention_heads, selected_count].
 * A null selected_ids means identity rows [0, selected_count).
 * Zero selected rows produce zero output.
 */
bool k31_dsa_sparse_attention_bf16(
        void       *latent_output,
        void       *score_workspace,
        void       *probability_workspace,
        const void *absorbed_q,
        const void *cache,
        const void *selected_ids,
        uint32_t    selected_count,
        uint32_t    attention_heads,
        uint32_t    cache_dim,
        uint32_t    latent_dim,
        float       attention_scale,
        void       *stream);

#ifdef __cplusplus
}
#endif

#endif
