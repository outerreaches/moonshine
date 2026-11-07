#include "mimo26_attention.h"

#include "mimo26_architecture.h"
#include "mimo26_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

float mimo26_attention_scale(void)
{
    /* head_dim ** -0.5, from the QK dimension. Using the V dimension here is
     * the easiest silent error available with asymmetric 192/128 heads. */
    return 1.0f / sqrtf((float)MIMO26_QK_HEAD_DIM);
}

bool mimo26_attention_visible(uint64_t kv_position, uint64_t query_position,
                              size_t window)
{
    if (kv_position > query_position) {
        return false; /* causal */
    }
    if (window == 0u) {
        return true; /* full attention */
    }
    /* kv > q - window, written to avoid unsigned wrap at small q. */
    return query_position - kv_position < (uint64_t)window;
}

mimo26_attention_status mimo26_attention_config_for_layer(
    uint32_t layer, mimo26_attention_config *config)
{
    if (config == NULL) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    memset(config, 0, sizeof *config);

    const mimo26_layer_kind kind = mimo26_architecture_layer_kind(layer);
    if (kind == MIMO26_LAYER_INVALID) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    const bool is_swa = (kind == MIMO26_LAYER_MOE_SWA);

    config->is_swa = is_swa;
    config->kv_heads = is_swa ? MIMO26_SWA_KV_HEADS : MIMO26_GLOBAL_KV_HEADS;
    config->kv_groups = MIMO26_QUERY_HEADS / config->kv_heads;
    config->qkv_width = is_swa ? MIMO26_SWA_QKV_WIDTH : MIMO26_GLOBAL_QKV_WIDTH;
    config->rope_theta = is_swa ? MIMO26_SWA_ROPE_THETA
                                : MIMO26_GLOBAL_ROPE_THETA;
    /* add_swa_attention_sink_bias is true, add_full_attention_sink_bias is
     * false, so only windowed layers carry a sink. */
    config->has_sink = is_swa;
    config->window = is_swa ? MIMO26_SLIDING_WINDOW : 0u;
    return MIMO26_ATTENTION_OK;
}

mimo26_attention_status mimo26_attention_split_qkv(
    const uint16_t *fused, const mimo26_attention_config *config,
    uint16_t *q, uint16_t *k, uint16_t *v)
{
    if (fused == NULL || config == NULL || q == NULL || k == NULL ||
        v == NULL || config->kv_heads == 0u) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    const size_t q_size = MIMO26_QUERY_HEADS * MIMO26_QK_HEAD_DIM;
    const size_t k_size = config->kv_heads * MIMO26_QK_HEAD_DIM;
    const size_t v_size = config->kv_heads * MIMO26_V_HEAD_DIM;
    if (q_size + k_size + v_size != config->qkv_width) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }

    memcpy(q, fused, q_size * sizeof *q);
    memcpy(k, fused + q_size, k_size * sizeof *k);
    /* V is scaled before it reaches a cache, so cached V is pre-scaled. */
    for (size_t i = 0; i < v_size; i++) {
        const float value = mimo26_bf16_to_f32(fused[q_size + k_size + i]);
        if (!isfinite(value)) {
            return MIMO26_ATTENTION_NONFINITE_VALUE;
        }
        v[i] = mimo26_f32_to_bf16(value * MIMO26_VALUE_SCALE);
    }
    return MIMO26_ATTENTION_OK;
}

mimo26_attention_status mimo26_rope_table(uint16_t cos_table[MIMO26_ROPE_DIM],
                                          uint16_t sin_table[MIMO26_ROPE_DIM],
                                          uint64_t position, float theta)
{
    if (cos_table == NULL || sin_table == NULL || !(theta > 1.0f)) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    for (size_t j = 0; j < MIMO26_ROPE_PAIRS; j++) {
        /*
         * Deliberately F32 end to end, matching the reference's
         * `1.0 / (base ** (arange(0, dim, 2).float() / dim))` and its F32
         * position product. Computing this in double is *more* accurate and
         * therefore wrong for parity: the f32 inv_freq carries up to 6.4e-8
         * relative error, which the position multiplies. By position 2^20 that
         * reaches 2.5e-2 rad and 8.1e-3 in cos -- twice one BF16 ulp. See
         * docs/mimo26-precision-contract.md.
         */
        const float exponent = (float)(2u * j) / (float)MIMO26_ROPE_DIM;
        const float inv_freq = 1.0f / powf(theta, exponent);
        const float angle = (float)position * inv_freq;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        if (!isfinite(cosine) || !isfinite(sine)) {
            return MIMO26_ATTENTION_NONFINITE_VALUE;
        }
        /* cat((freqs, freqs)) duplicates the 32 angles across 64 entries. */
        const uint16_t cos_bf16 = mimo26_f32_to_bf16(cosine);
        const uint16_t sin_bf16 = mimo26_f32_to_bf16(sine);
        cos_table[j] = cos_bf16;
        cos_table[j + MIMO26_ROPE_PAIRS] = cos_bf16;
        sin_table[j] = sin_bf16;
        sin_table[j + MIMO26_ROPE_PAIRS] = sin_bf16;
    }
    return MIMO26_ATTENTION_OK;
}

mimo26_attention_status mimo26_rope_apply(
    uint16_t head[MIMO26_QK_HEAD_DIM],
    const uint16_t cos_table[MIMO26_ROPE_DIM],
    const uint16_t sin_table[MIMO26_ROPE_DIM])
{
    if (head == NULL || cos_table == NULL || sin_table == NULL) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    uint16_t rotated[MIMO26_ROPE_DIM];
    for (size_t j = 0; j < MIMO26_ROPE_PAIRS; j++) {
        const float low = mimo26_bf16_to_f32(head[j]);
        const float high = mimo26_bf16_to_f32(head[j + MIMO26_ROPE_PAIRS]);
        const float cosine = mimo26_bf16_to_f32(cos_table[j]);
        const float sine = mimo26_bf16_to_f32(sin_table[j]);
        if (!isfinite(low) || !isfinite(high)) {
            return MIMO26_ATTENTION_NONFINITE_VALUE;
        }
        /*
         * rotate_half gives (-x2, x1) against the duplicated cos/sin halves.
         * The reference evaluates (q * cos) + (rotate_half(q) * sin) with every
         * operand in BF16, so each product rounds and then the sum rounds --
         * three roundings per coordinate. Accumulating in F32 and rounding once
         * is more accurate and breaks parity.
         */
        const float low_cos = mimo26_bf16_to_f32(mimo26_f32_to_bf16(low * cosine));
        const float high_sin = mimo26_bf16_to_f32(mimo26_f32_to_bf16(-high * sine));
        const float high_cos = mimo26_bf16_to_f32(mimo26_f32_to_bf16(high * cosine));
        const float low_sin = mimo26_bf16_to_f32(mimo26_f32_to_bf16(low * sine));
        rotated[j] = mimo26_f32_to_bf16(low_cos + high_sin);
        rotated[j + MIMO26_ROPE_PAIRS] = mimo26_f32_to_bf16(high_cos + low_sin);
    }
    for (size_t j = 0; j < MIMO26_ROPE_DIM; j++) {
        head[j] = rotated[j];
    }
    /* Coordinates 64..191 are the nope half and stay as they are. */
    return MIMO26_ATTENTION_OK;
}

mimo26_attention_status mimo26_attention_decode(
    uint16_t *out, const uint16_t *query, const uint16_t *keys,
    const uint16_t *values, const uint16_t *current_keys,
    const uint16_t *current_values, const uint16_t *sink_bias,
    const mimo26_attention_config *config, size_t history,
    uint64_t first_position, uint64_t query_position)
{
    const bool have_current = (current_keys != NULL && current_values != NULL);
    if (out == NULL || query == NULL || config == NULL ||
        config->kv_heads == 0u) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    if ((current_keys == NULL) != (current_values == NULL)) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    if (history > 0u && (keys == NULL || values == NULL)) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    if (history == 0u && !have_current) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    if (config->has_sink && sink_bias == NULL) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    if (config->kv_groups == 0u ||
        config->kv_groups * config->kv_heads != MIMO26_QUERY_HEADS) {
        return MIMO26_ATTENTION_INVALID_ARGUMENT;
    }
    /*
     * The committed history covers first_position .. first_position+history-1
     * and must not reach past the query. When the current token is supplied
     * separately the history stops one short of it, so the bound tightens.
     */
    if (history > 0u) {
        if (first_position > query_position) {
            return MIMO26_ATTENTION_INVALID_ARGUMENT;
        }
        const uint64_t span = query_position - first_position + 1u;
        if (span < (uint64_t)history) {
            return MIMO26_ATTENTION_INVALID_ARGUMENT;
        }
        if (have_current && span - 1u < (uint64_t)history) {
            return MIMO26_ATTENTION_INVALID_ARGUMENT;
        }
    }
    /* One virtual slot for the uncommitted token, one for the sink. */
    const size_t slot_total = history + (have_current ? 1u : 0u);

    const float scale = mimo26_attention_scale();
    const size_t kv_heads = config->kv_heads;

    /* Scores for one query head, plus one slot for the sink logit. */
    float *scores = NULL;
    float stack_scores[MIMO26_SLIDING_WINDOW + 2u];
    float *heap_scores = NULL;
    if (slot_total + 1u <= sizeof stack_scores / sizeof stack_scores[0]) {
        scores = stack_scores;
    } else {
        heap_scores = (float *)calloc(slot_total + 1u, sizeof *heap_scores);
        if (heap_scores == NULL) {
            return MIMO26_ATTENTION_INVALID_ARGUMENT;
        }
        scores = heap_scores;
    }

    mimo26_attention_status status = MIMO26_ATTENTION_OK;
    for (size_t h = 0; h < MIMO26_QUERY_HEADS && status == MIMO26_ATTENTION_OK;
         h++) {
        const size_t kv_head = h / config->kv_groups;
        const uint16_t *q_head = query + h * MIMO26_QK_HEAD_DIM;

        size_t visible = 0;
        float maximum = -INFINITY;
        for (size_t t = 0; t < slot_total; t++) {
            const bool is_current = (have_current && t == history);
            const uint64_t kv_position =
                is_current ? query_position : first_position + (uint64_t)t;
            if (!mimo26_attention_visible(kv_position, query_position,
                                          config->window)) {
                scores[t] = -INFINITY;
                continue;
            }
            const uint16_t *k_head =
                is_current
                    ? current_keys + kv_head * MIMO26_QK_HEAD_DIM
                    : keys + (t * kv_heads + kv_head) * MIMO26_QK_HEAD_DIM;
            float dot = 0.0f;
            for (size_t d = 0; d < MIMO26_QK_HEAD_DIM; d++) {
                dot += mimo26_bf16_to_f32(q_head[d]) *
                       mimo26_bf16_to_f32(k_head[d]);
            }
            if (!isfinite(dot)) {
                status = MIMO26_ATTENTION_NONFINITE_VALUE;
                break;
            }
            /* The reference's matmul output is BF16, and the scaling stays in
             * that dtype, so round here rather than carrying F32 forward. */
            const float scaled =
                mimo26_bf16_to_f32(mimo26_f32_to_bf16(dot * scale));
            scores[t] = scaled;
            if (scaled > maximum) {
                maximum = scaled;
            }
            visible++;
        }
        if (status != MIMO26_ATTENTION_OK) {
            break;
        }
        if (visible == 0u) {
            /* No visible key: without a sink there is nothing to normalize. */
            if (!config->has_sink) {
                status = MIMO26_ATTENTION_INVALID_ARGUMENT;
                break;
            }
        }

        size_t slots = slot_total;
        if (config->has_sink) {
            const float sink = mimo26_bf16_to_f32(sink_bias[h]);
            if (!isfinite(sink)) {
                status = MIMO26_ATTENTION_NONFINITE_VALUE;
                break;
            }
            scores[slot_total] = sink;
            if (sink > maximum) {
                maximum = sink;
            }
            slots = slot_total + 1u;
        }

        /* Explicit max subtraction, as the reference does, before softmax. */
        double total = 0.0;
        for (size_t t = 0; t < slots; t++) {
            if (scores[t] == -INFINITY) {
                scores[t] = 0.0f;
                continue;
            }
            const float shifted =
                mimo26_bf16_to_f32(mimo26_f32_to_bf16(scores[t] - maximum));
            const float weight = expf(shifted);
            scores[t] = weight;
            total += (double)weight;
        }
        if (!(total > 0.0)) {
            status = MIMO26_ATTENTION_NONFINITE_VALUE;
            break;
        }

        /* Probabilities are rounded to BF16 before weighting the values, and
         * the sink's share is then dropped -- so the surviving probabilities
         * deliberately sum to less than one. */
        float accumulator[MIMO26_V_HEAD_DIM];
        memset(accumulator, 0, sizeof accumulator);
        for (size_t t = 0; t < slot_total; t++) {
            if (scores[t] == 0.0f) {
                continue;
            }
            const float probability =
                mimo26_bf16_to_f32(mimo26_f32_to_bf16(
                    (float)((double)scores[t] / total)));
            if (probability == 0.0f) {
                continue;
            }
            const uint16_t *v_head =
                (have_current && t == history)
                    ? current_values + kv_head * MIMO26_V_HEAD_DIM
                    : values + (t * kv_heads + kv_head) * MIMO26_V_HEAD_DIM;
            for (size_t d = 0; d < MIMO26_V_HEAD_DIM; d++) {
                accumulator[d] += probability * mimo26_bf16_to_f32(v_head[d]);
            }
        }
        uint16_t *out_head = out + h * MIMO26_V_HEAD_DIM;
        for (size_t d = 0; d < MIMO26_V_HEAD_DIM; d++) {
            if (!isfinite(accumulator[d])) {
                status = MIMO26_ATTENTION_NONFINITE_VALUE;
                break;
            }
            out_head[d] = mimo26_f32_to_bf16(accumulator[d]);
        }
    }

    free(heap_scores);
    return status;
}
