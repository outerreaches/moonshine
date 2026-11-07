#include "k31_dsa_reference.h"

#include <math.h>

static bool nonzero_geometry(size_t rows, size_t columns) {
    return rows != 0u && columns != 0u;
}

bool k31_dsa_linear(const float *input, size_t rows,
                    const float *weight, size_t input_features,
                    size_t output_features, float *output) {
    if (!nonzero_geometry(input_features, output_features)) return false;
    if (rows == 0u) return true;
    if (!input || !weight || !output) return false;

    /* Bias-free wq_b/wk/weights_proj contract:
     * https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L346-L392
     */
    for (size_t row = 0u; row < rows; row++) {
        for (size_t column = 0u; column < output_features; column++) {
            double sum = 0.0;
            for (size_t i = 0u; i < input_features; i++) {
                sum += (double)input[row * input_features + i] *
                       (double)weight[column * input_features + i];
            }
            output[row * output_features + column] = (float)sum;
            if (!isfinite(output[row * output_features + column])) return false;
        }
    }
    return true;
}

bool k31_dsa_normalize_keys(const float *raw_keys, size_t num_tokens,
                            size_t head_dim, size_t index_rope_dim,
                            const float *gamma, const float *beta,
                            double epsilon, float *normalized_keys) {
    /* index_rope_dim=0 and FP32 LayerNorm with the model RMS epsilon:
     * https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/kimi_linear.py#L151-L186
     * https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L300-L425
     */
    if (head_dim == 0u || index_rope_dim != 0u ||
        !isfinite(epsilon) || epsilon < 0.0) {
        return false;
    }
    if (num_tokens == 0u) return true;
    if (!raw_keys || !gamma || !beta || !normalized_keys) return false;

    for (size_t token = 0u; token < num_tokens; token++) {
        const float *raw = raw_keys + token * head_dim;
        float *normalized = normalized_keys + token * head_dim;
        double mean = 0.0;
        for (size_t i = 0u; i < head_dim; i++) {
            mean += (double)raw[i];
        }
        mean /= (double)head_dim;
        double variance = 0.0;
        for (size_t i = 0u; i < head_dim; i++) {
            const double delta = (double)raw[i] - mean;
            variance += delta * delta;
        }
        variance /= (double)head_dim;
        const double inverse_stddev = 1.0 / sqrt(variance + epsilon);
        for (size_t i = 0u; i < head_dim; i++) {
            const double value =
                ((double)raw[i] - mean) * inverse_stddev *
                    (double)gamma[i] +
                (double)beta[i];
            normalized[i] = (float)value;
            if (!isfinite(normalized[i])) return false;
        }
    }
    return true;
}

bool k31_dsa_scores(const float *queries, size_t num_queries,
                    size_t num_heads, size_t head_dim,
                    const float *keys, size_t num_keys,
                    const float *head_weights, const float *query_scales,
                    const float *key_scales, float *scores) {
    if (!nonzero_geometry(num_heads, head_dim)) return false;
    if (num_queries == 0u || num_keys == 0u) return true;
    if (!queries || !keys || !head_weights || !scores) return false;

    /* Official reference: ReLU each head's QK dot, weight and sum heads, then
     * multiply the key's per-token scale. The 1/sqrt(H) head scale and
     * 1/sqrt(D) softmax scale are folded into head weights by the runtime.
     * https://github.com/sgl-project/sglang/blob/main/python/sglang/test/kernels/paged_mqa.py#L24-L67
     * https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L624-L637
     */
    const double head_score_scale = 1.0 / sqrt((double)num_heads *
                                               (double)head_dim);
    for (size_t query = 0u; query < num_queries; query++) {
        for (size_t key = 0u; key < num_keys; key++) {
            double score = 0.0;
            for (size_t head = 0u; head < num_heads; head++) {
                const double query_scale =
                    query_scales
                        ? (double)query_scales[query * num_heads + head]
                        : 1.0;
                if (!isfinite(query_scale) || query_scale <= 0.0) return false;
                double dot = 0.0;
                for (size_t dimension = 0u; dimension < head_dim; dimension++) {
                    dot += (double)queries[(query * num_heads + head) *
                                              head_dim + dimension] *
                           (double)keys[key * head_dim + dimension];
                }
                if (dot < 0.0) dot = 0.0;
                score += (double)head_weights[query * num_heads + head] * dot *
                         query_scale;
            }
            const double key_scale = key_scales ? (double)key_scales[key] : 1.0;
            if (!isfinite(key_scale) || key_scale <= 0.0) return false;
            score *= key_scale * head_score_scale;
            scores[query * num_keys + key] = (float)score;
            if (!isfinite(scores[query * num_keys + key])) return false;
        }
    }
    return true;
}

static bool score_precedes(float score, size_t column,
                           float other_score, size_t other_column) {
    return score > other_score ||
           (score == other_score && column < other_column);
}

static bool select_topk_row(const float *scores, size_t start, size_t length,
                            size_t topk, int32_t *selected) {
    for (size_t slot = 0u; slot < topk; slot++) selected[slot] = -1;
    if (start > SIZE_MAX - length || length > (size_t)INT32_MAX ||
        start + length > (size_t)INT32_MAX) {
        return false;
    }

    size_t occupied = 0u;
    for (size_t column = start; column < start + length; column++) {
        const float score = scores[column];
        if (isnan(score)) return false;
        if (score == -INFINITY) continue;

        size_t insertion = 0u;
        while (insertion < occupied &&
               !score_precedes(score, column, scores[selected[insertion]],
                               (size_t)selected[insertion])) {
            insertion++;
        }
        if (insertion >= topk) continue;
        if (occupied < topk) occupied++;
        for (size_t slot = occupied - 1u; slot > insertion; slot--) {
            selected[slot] = selected[slot - 1u];
        }
        selected[insertion] = (int32_t)column;
    }
    return true;
}

bool k31_dsa_select_topk(const float *scores, size_t num_rows,
                         size_t score_row_stride, size_t score_columns,
                         const size_t *row_starts, const size_t *row_lengths,
                         size_t topk, int32_t *selected) {
    if (score_row_stride < score_columns || score_columns > (size_t)INT32_MAX) {
        return false;
    }
    if (num_rows == 0u) return true;
    if (topk == 0u) return true;
    if (!scores || !row_lengths || !selected) return false;

    /* Official unfused selection uses a per-row start/length window and -1
     * padding. Fused backends may then map positions to physical cache slots.
     * https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py#L268-L305
     */
    for (size_t row = 0u; row < num_rows; row++) {
        const size_t start = row_starts ? row_starts[row] : 0u;
        if (start > score_columns || row_lengths[row] > score_columns - start) {
            return false;
        }
        if (!select_topk_row(scores + row * score_row_stride, start,
                             row_lengths[row], topk,
                             selected + row * topk)) {
            return false;
        }
    }
    return true;
}

bool k31_dsa_select_causal(const float *scores, size_t num_rows,
                           size_t score_row_stride, size_t score_columns,
                           const size_t *query_positions, size_t topk,
                           int32_t *selected) {
    if (score_row_stride < score_columns || score_columns > (size_t)INT32_MAX) {
        return false;
    }
    if (num_rows == 0u || topk == 0u) return true;
    if (!scores || !query_positions || !selected) return false;

    /* Inclusive self-attention: query position p sees exactly [0,p]. */
    for (size_t row = 0u; row < num_rows; row++) {
        if (query_positions[row] >= score_columns) return false;
        if (!select_topk_row(scores + row * score_row_stride, 0u,
                             query_positions[row] + 1u, topk,
                             selected + row * topk)) {
            return false;
        }
    }
    return true;
}

bool k31_dsa_layer_reuses_topk(size_t layer_id, size_t index_topk_freq,
                               size_t index_skip_topk_offset, bool *reuse) {
    if (index_topk_freq == 0u || index_skip_topk_offset == 0u || !reuse) {
        return false;
    }
    /* https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py#L518-L551
     * max(layer_id - offset + 1, 0) without unsigned underflow.
     */
    const size_t producer_phase =
        layer_id + 1u > index_skip_topk_offset
            ? layer_id + 1u - index_skip_topk_offset
            : 0u;
    *reuse = producer_phase % index_topk_freq != 0u;
    return true;
}

bool k31_dsa_sparse_mla_attention(
    const float *queries, size_t num_queries, size_t num_heads,
    size_t qk_dim, const float *latent_cache, size_t cache_rows,
    size_t kv_lora_rank, const int32_t *selected, size_t topk,
    double softmax_scale, float *latent_output) {
    if (!nonzero_geometry(num_heads, qk_dim) || kv_lora_rank == 0u ||
        kv_lora_rank > qk_dim || !isfinite(softmax_scale) ||
        softmax_scale <= 0.0) {
        return false;
    }
    if (num_queries == 0u) return true;
    if (topk == 0u || !queries || !latent_cache || !selected ||
        !latent_output) {
        return false;
    }

    /* Absorbed MLA computes softmax(Q * latent_K^T) * latent_V for only the
     * selected rows; latent_V is the cache's first kv_lora_rank columns.
     * https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py#L95-L115
     * https://github.com/sgl-project/sglang/main/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L794-L884
     */
    for (size_t query = 0u; query < num_queries; query++) {
        const int32_t *row = selected + query * topk;
        size_t count = 0u;
        bool padding = false;
        for (size_t slot = 0u; slot < topk; slot++) {
            if (row[slot] == -1) {
                padding = true;
                continue;
            }
            if (padding || row[slot] < 0 || (size_t)row[slot] >= cache_rows) {
                return false;
            }
            count++;
        }
        if (count == 0u) return false;

        for (size_t head = 0u; head < num_heads; head++) {
            const float *q = queries + (query * num_heads + head) * qk_dim;
            double maximum = -INFINITY;
            for (size_t slot = 0u; slot < count; slot++) {
                const float *key = latent_cache + (size_t)row[slot] * qk_dim;
                double dot = 0.0;
                for (size_t dimension = 0u; dimension < qk_dim; dimension++) {
                    dot += (double)q[dimension] * (double)key[dimension];
                }
                dot *= softmax_scale;
                if (!isfinite(dot)) return false;
                if (dot > maximum) maximum = dot;
            }

            double denominator = 0.0;
            for (size_t slot = 0u; slot < count; slot++) {
                const float *key = latent_cache + (size_t)row[slot] * qk_dim;
                double dot = 0.0;
                for (size_t dimension = 0u; dimension < qk_dim; dimension++) {
                    dot += (double)q[dimension] * (double)key[dimension];
                }
                denominator += exp(dot * softmax_scale - maximum);
            }

            for (size_t value = 0u; value < kv_lora_rank; value++) {
                double numerator = 0.0;
                for (size_t slot = 0u; slot < count; slot++) {
                    const float *key =
                        latent_cache + (size_t)row[slot] * qk_dim;
                    double dot = 0.0;
                    for (size_t dimension = 0u; dimension < qk_dim; dimension++) {
                        dot += (double)q[dimension] * (double)key[dimension];
                    }
                    numerator += exp(dot * softmax_scale - maximum) *
                                 (double)key[value];
                }
                const size_t output_index =
                    (query * num_heads + head) * kv_lora_rank + value;
                latent_output[output_index] = (float)(numerator / denominator);
                if (!isfinite(latent_output[output_index])) return false;
            }
        }
    }
    return true;
}

bool k31_dsa_project_mla_values(
    const float *latent, size_t num_queries, size_t num_heads,
    size_t kv_lora_rank, const float *value_weights, size_t value_dim,
    float *values) {
    if (!nonzero_geometry(num_heads, kv_lora_rank) || value_dim == 0u) {
        return false;
    }
    if (num_queries == 0u) return true;
    if (!latent || !value_weights || !values) return false;

    /* o[q,h,v] = sum_l latent[q,h,l] * W_UV[h,l,v]. This expansion precedes
     * the K3 output gate and o_proj.
     * https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L971-L1058
     */
    for (size_t query = 0u; query < num_queries; query++) {
        for (size_t head = 0u; head < num_heads; head++) {
            for (size_t value = 0u; value < value_dim; value++) {
                double sum = 0.0;
                for (size_t l = 0u; l < kv_lora_rank; l++) {
                    sum += (double)latent[(query * num_heads + head) *
                                             kv_lora_rank + l] *
                           (double)value_weights[(head * kv_lora_rank + l) *
                                                    value_dim + value];
                }
                const size_t output_index =
                    (query * num_heads + head) * value_dim + value;
                values[output_index] = (float)sum;
                if (!isfinite(values[output_index])) return false;
            }
        }
    }
    return true;
}

bool k31_dsa_apply_output_gate(const float *values, size_t num_queries,
                               size_t num_heads, size_t value_dim,
                               const float *gate, float *output) {
    if (!nonzero_geometry(num_heads, value_dim)) return false;
    if (num_queries == 0u) return true;
    if (!values || !gate || !output) return false;

    /* K3 applies the sigmoid gate in TP-local space immediately before o_proj:
     * https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L2286-L2327
     */
    const size_t count = num_queries * num_heads * value_dim;
    for (size_t i = 0u; i < count; i++) {
        const double x = (double)gate[i];
        const double sigmoid =
            x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
        output[i] = (float)((double)values[i] * sigmoid);
        if (!isfinite(output[i])) return false;
    }
    return true;
}
