#ifndef K31_DSA_REFERENCE_H
#define K31_DSA_REFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Deterministic, dequantized-float CPU oracle for Kimi K3.1 Gated DSA.
 * Arrays are row-major. Functions return false for invalid geometry/input;
 * output contents are unspecified after failure. No function allocates.
 *
 * Official algorithm sources:
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/test/kernels/paged_mqa.py
 * https://github.com/vllm-project/vllm/blob/main/csrc/cpu/sgl-kernels/paged_mqa_logits.cpp
 */

/* output = input * weight^T; weight is [output_features, input_features]. */
bool k31_dsa_linear(const float *input, size_t rows,
                    const float *weight, size_t input_features,
                    size_t output_features, float *output);

/*
 * Apply the released indexer's key LayerNorm (not RMSNorm). The pinned K3.1
 * config has index_rope_dim=0, so any nonzero value is rejected and neither
 * queries nor keys receive rotary embedding. gamma/beta have head_dim items.
 * raw_keys and normalized_keys may be identical.
 */
bool k31_dsa_normalize_keys(const float *raw_keys, size_t num_tokens,
                            size_t head_dim, size_t index_rope_dim,
                            const float *gamma, const float *beta,
                            double epsilon, float *normalized_keys);

/*
 * queries:       [num_queries, num_heads, head_dim]
 * keys:          [num_keys, head_dim] (one shared index key per token)
 * head_weights:  [num_queries, num_heads], before sigmoid-free score scaling
 * query_scales:  optional [num_queries, num_heads], otherwise 1
 * key_scales:    optional [num_keys], otherwise 1
 * scores:        [num_queries, num_keys]
 *
 * For head h and key token t:
 *   dot = max(0, dot(queries[q,h], keys[t]))
 *   scores[q,t] = key_scales[t]
 *                 * sum_h(head_weights[q,h] * query_scales[q,h] * dot_h)
 *                 / sqrt(num_heads * head_dim)
 * Accumulation is double precision in ascending dimension/head order. For FP8
 * HIP inputs, pass dequantized values and their UE8M0 scales; query scales are
 * per head and key scales are per token. For group-scaled MXFP4, apply each
 * group scale during dequantization and pass unit key_scales/query_scales.
 * Do not apply sigmoid to head_weights.
 */
bool k31_dsa_scores(const float *queries, size_t num_queries,
                    size_t num_heads, size_t head_dim,
                    const float *keys, size_t num_keys,
                    const float *head_weights, const float *query_scales,
                    const float *key_scales, float *scores);

/*
 * Select from [row_starts[r], row_starts[r] + row_lengths[r]) in each score
 * row. A NULL row_starts means zero for every row. Output entries are absolute
 * score-column indices, sorted by descending score then ascending column.
 * Unused entries are -1; valid -inf scores are also padding, matching the
 * official unfused reference. NaN in a candidate range is rejected.
 *
 * This total order is the deterministic oracle contract, not a claim that all
 * official fused backends preserve order. Official consumers use the selected
 * set and -1 padding; page-table/ragged address transforms are separate.
 */
bool k31_dsa_select_topk(const float *scores, size_t num_rows,
                         size_t score_row_stride, size_t score_columns,
                         const size_t *row_starts, const size_t *row_lengths,
                         size_t topk, int32_t *selected);

/* Contiguous-cache causal special case: row r covers keys [0, position[r]]. */
bool k31_dsa_select_causal(const float *scores, size_t num_rows,
                           size_t score_row_stride, size_t score_columns,
                           const size_t *query_positions, size_t topk,
                           int32_t *selected);

/*
 * Exact released-config layer schedule helper. With index_topk_freq=1 and
 * index_skip_topk_offset=1 this always returns reuse=false. This offset is a
 * zero-based layer scheduling offset, NOT a token/causal offset.
 */
bool k31_dsa_layer_reuses_topk(size_t layer_id, size_t index_topk_freq,
                               size_t index_skip_topk_offset, bool *reuse);

/*
 * Sparse absorbed-MLA attention over selected cache rows.
 * queries:       [num_queries, num_heads, qk_dim] (already W_UK-absorbed)
 * latent_cache:  [cache_rows, qk_dim]
 * selected:      [num_queries, topk], cache row IDs and trailing -1 padding
 * latent_output: [num_queries, num_heads, kv_lora_rank]
 *
 * The first kv_lora_rank cache columns are also the MLA latent values. Softmax
 * is stable and double-precision; softmax_scale is explicit because released
 * MLA scaling is based on its pre-absorption QK geometry, not qk_dim here.
 * Causality is imposed by selection, not repeated here. A row with no selected
 * entries is rejected rather than assigned a fabricated value.
 */
bool k31_dsa_sparse_mla_attention(
    const float *queries, size_t num_queries, size_t num_heads,
    size_t qk_dim, const float *latent_cache, size_t cache_rows,
    size_t kv_lora_rank, const int32_t *selected, size_t topk,
    double softmax_scale, float *latent_output);

/* value_weights is [num_heads, kv_lora_rank, value_dim] (W_UV per head). */
bool k31_dsa_project_mla_values(
    const float *latent, size_t num_queries, size_t num_heads,
    size_t kv_lora_rank, const float *value_weights, size_t value_dim,
    float *values);

/*
 * Released Gated DSA applies values * sigmoid(gate) after W_UV and before
 * o_proj. It does not gate indexer logits. In-place operation is supported.
 */
bool k31_dsa_apply_output_gate(const float *values, size_t num_queries,
                               size_t num_heads, size_t value_dim,
                               const float *gate, float *output);

#ifdef __cplusplus
}
#endif

#endif
