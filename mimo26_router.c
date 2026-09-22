#include "mimo26_router.h"

#include <math.h>
#include <string.h>

static float sigmoid_f32(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

mimo26_router_status mimo26_router_check_grouping(size_t groups,
                                                  size_t topk_groups)
{
    if (groups == 0u || topk_groups == 0u) {
        return MIMO26_ROUTER_INVALID_ARGUMENT;
    }
    /*
     * With one group covering every expert, the reference's group top-2 sum,
     * group top-k mask and masked_fill collapse to selecting over all
     * experts. Any other grouping changes which experts are reachable, so it
     * needs its own implementation and its own tests rather than this path.
     */
    if (groups != MIMO26_ROUTER_GROUPS ||
        topk_groups != MIMO26_ROUTER_TOPK_GROUPS) {
        return MIMO26_ROUTER_UNSUPPORTED_GROUPING;
    }
    return MIMO26_ROUTER_OK;
}

mimo26_router_status mimo26_router_logits_f32(
    float *logits, const float *hidden, const float *weight,
    size_t experts, size_t hidden_size)
{
    if (logits == NULL || hidden == NULL || weight == NULL ||
        experts == 0u || hidden_size == 0u) {
        return MIMO26_ROUTER_INVALID_ARGUMENT;
    }
    if (experts > SIZE_MAX / hidden_size) {
        return MIMO26_ROUTER_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < hidden_size; i++) {
        if (!isfinite(hidden[i])) {
            return MIMO26_ROUTER_NONFINITE_VALUE;
        }
    }
    for (size_t e = 0; e < experts; e++) {
        const float *row = weight + e * hidden_size;
        float sum = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            if (!isfinite(row[i])) {
                return MIMO26_ROUTER_NONFINITE_VALUE;
            }
            sum += row[i] * hidden[i];
        }
        if (!isfinite(sum)) {
            return MIMO26_ROUTER_NONFINITE_VALUE;
        }
        logits[e] = sum;
    }
    return MIMO26_ROUTER_OK;
}

mimo26_router_status mimo26_router_select_f32(
    uint32_t *indices, float *weights, const float *logits,
    const float *bias, size_t experts, size_t k, float scale)
{
    if (indices == NULL || weights == NULL || logits == NULL ||
        bias == NULL || experts == 0u || k == 0u || k > experts ||
        experts > MIMO26_ROUTER_EXPERTS) {
        return MIMO26_ROUTER_INVALID_ARGUMENT;
    }
    if (!isfinite(scale)) {
        return MIMO26_ROUTER_NONFINITE_VALUE;
    }
    /* Validate before writing, so a rejected call leaves outputs untouched. */
    for (size_t e = 0; e < experts; e++) {
        const float raw = sigmoid_f32(logits[e]);
        const float choice = raw + bias[e];
        if (!isfinite(logits[e]) || !isfinite(bias[e]) || !isfinite(raw) ||
            !isfinite(choice)) {
            return MIMO26_ROUTER_NONFINITE_VALUE;
        }
    }

    /*
     * Selection pass: highest sigmoid+bias wins, ties to the lower expert id.
     * Selection order is not retained -- only the chosen set matters, because
     * the declared accumulation order is ascending expert id.
     */
    bool taken[MIMO26_ROUTER_EXPERTS];
    memset(taken, 0, sizeof taken);

    for (size_t j = 0; j < k; j++) {
        size_t best = experts;
        float best_score = 0.0f;
        for (size_t e = 0; e < experts; e++) {
            if (taken[e]) {
                continue;
            }
            const float score = sigmoid_f32(logits[e]) + bias[e];
            if (best == experts || score > best_score) {
                best = e;
                best_score = score;
            }
            /* Equal scores keep the lower id, which the scan reaches first. */
        }
        if (best == experts) {
            return MIMO26_ROUTER_INVALID_ARGUMENT;
        }
        taken[best] = true;
    }

    /* Emit in ascending expert id by walking the membership flags. */
    size_t written = 0;
    for (size_t e = 0; e < experts && written < k; e++) {
        if (taken[e]) {
            indices[written++] = (uint32_t)e;
        }
    }

    /* Normalize over the uncorrected sigmoid scores, accumulating in the
     * same declared ascending order so the denominator is reproducible. */
    float sum = 0.0f;
    for (size_t j = 0; j < k; j++) {
        weights[j] = sigmoid_f32(logits[indices[j]]);
        sum += weights[j];
    }
    sum += MIMO26_ROUTER_NORM_EPSILON;
    for (size_t j = 0; j < k; j++) {
        weights[j] = (weights[j] / sum) * scale;
    }
    return MIMO26_ROUTER_OK;
}

mimo26_router_status mimo26_router_top8_256_f32(
    uint32_t indices[MIMO26_ROUTER_TOP_K],
    float weights[MIMO26_ROUTER_TOP_K],
    const float logits[MIMO26_ROUTER_EXPERTS],
    const float bias[MIMO26_ROUTER_EXPERTS])
{
    return mimo26_router_select_f32(indices, weights, logits, bias,
                                    MIMO26_ROUTER_EXPERTS,
                                    MIMO26_ROUTER_TOP_K,
                                    MIMO26_ROUTER_SCALE);
}
