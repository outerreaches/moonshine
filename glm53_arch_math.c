#include "glm53_arch_math.h"

#include <math.h>
#include <string.h>

static float sigmoid_f32(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

static float swiglu_one(float gate, float up, float limit) {
    float s;
    if (gate > limit) gate = limit;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;
    s = sigmoid_f32(gate);
    return (gate * s) * up;
}

glm53_arch_math_status glm53_limited_swiglu_f32(
    float *out, const float *gate, const float *up, size_t count, float limit) {
    size_t i;
    if (out == NULL || gate == NULL || up == NULL || count == 0u ||
        limit < 0.0f) {
        return GLM53_ARCH_MATH_INVALID_ARGUMENT;
    }
    if (!isfinite(limit)) return GLM53_ARCH_MATH_NONFINITE_VALUE;
    for (i = 0u; i < count; ++i) {
        if (!isfinite(gate[i]) || !isfinite(up[i]) ||
            !isfinite(swiglu_one(gate[i], up[i], limit))) {
            return GLM53_ARCH_MATH_NONFINITE_VALUE;
        }
    }
    for (i = 0u; i < count; ++i) out[i] = swiglu_one(gate[i], up[i], limit);
    return GLM53_ARCH_MATH_OK;
}

glm53_arch_math_status glm53_router_topk_f32(
    size_t *indices, float *weights, const float *logits, const float *bias,
    size_t num_experts, size_t k, float scale) {
    size_t e, j;
    float sum = 0.0f;
    if (indices == NULL || weights == NULL || logits == NULL || bias == NULL ||
        num_experts == 0u || k == 0u || k > num_experts) {
        return GLM53_ARCH_MATH_INVALID_ARGUMENT;
    }
    if (!isfinite(scale)) return GLM53_ARCH_MATH_NONFINITE_VALUE;
    /* Validate everything that can fail before using output as selection state. */
    for (e = 0u; e < num_experts; ++e) {
        const float raw = sigmoid_f32(logits[e]);
        const float choice = raw + bias[e];
        if (!isfinite(logits[e]) || !isfinite(bias[e]) ||
            !isfinite(raw) || !isfinite(choice)) {
            return GLM53_ARCH_MATH_NONFINITE_VALUE;
        }
    }
    for (j = 0u; j < k; ++j) {
        size_t best = num_experts;
        float best_score = 0.0f;
        for (e = 0u; e < num_experts; ++e) {
            int used = 0;
            size_t q;
            const float score = sigmoid_f32(logits[e]) + bias[e];
            for (q = 0u; q < j; ++q) {
                if (indices[q] == e) { used = 1; break; }
            }
            if (!used && (best == num_experts || score > best_score ||
                          (score == best_score && e < best))) {
                best = e;
                best_score = score;
            }
        }
        indices[j] = best;
        weights[j] = sigmoid_f32(logits[best]);
        sum += weights[j];
    }
    sum += 1.0e-20f;
    for (j = 0u; j < k; ++j) weights[j] = (weights[j] / sum) * scale;
    return GLM53_ARCH_MATH_OK;
}

glm53_arch_math_status glm53_router_top8_288_f32(
    size_t indices[GLM53_ROUTER_TOP_K],
    float weights[GLM53_ROUTER_TOP_K],
    const float logits[GLM53_ROUTER_EXPERTS],
    const float bias[GLM53_ROUTER_EXPERTS], float scale) {
    return glm53_router_topk_f32(indices, weights, logits, bias,
                                 GLM53_ROUTER_EXPERTS,
                                 GLM53_ROUTER_TOP_K, scale);
}

glm53_arch_math_status glm53_mhc_weights4_f32(
    float pre[GLM53_MHC_H], float post[GLM53_MHC_H],
    float comb[GLM53_MHC_H * GLM53_MHC_H],
    const float mix[GLM53_MHC_MIX], const float base[GLM53_MHC_MIX],
    const float scale[3], float eps) {
    float p[4], q[4], c[16];
    size_t i, src, dst, iteration;
    if (pre == NULL || post == NULL || comb == NULL || mix == NULL ||
        base == NULL || scale == NULL || eps < 0.0f) {
        return GLM53_ARCH_MATH_INVALID_ARGUMENT;
    }
    if (!isfinite(eps)) return GLM53_ARCH_MATH_NONFINITE_VALUE;
    for (i = 0u; i < 24u; ++i) {
        if (!isfinite(mix[i]) || !isfinite(base[i]))
            return GLM53_ARCH_MATH_NONFINITE_VALUE;
    }
    for (i = 0u; i < 3u; ++i) if (!isfinite(scale[i]))
        return GLM53_ARCH_MATH_NONFINITE_VALUE;
    for (i = 0u; i < 4u; ++i) {
        const float a = mix[i] * scale[0] + base[i];
        const float b = mix[4u + i] * scale[1] + base[4u + i];
        p[i] = sigmoid_f32(a) + eps;
        q[i] = 2.0f * sigmoid_f32(b);
        if (!isfinite(a) || !isfinite(b) || !isfinite(p[i]) || !isfinite(q[i]))
            return GLM53_ARCH_MATH_NONFINITE_VALUE;
    }
    for (src = 0u; src < 4u; ++src) {
        float max_logit = -INFINITY;
        float denom = 0.0f;
        for (dst = 0u; dst < 4u; ++dst) {
            const size_t index = 8u + src * 4u + dst;
            c[src * 4u + dst] = mix[index] * scale[2] + base[index];
            if (!isfinite(c[src * 4u + dst])) return GLM53_ARCH_MATH_NONFINITE_VALUE;
            if (c[src * 4u + dst] > max_logit) max_logit = c[src * 4u + dst];
        }
        for (dst = 0u; dst < 4u; ++dst) {
            c[src * 4u + dst] = expf(c[src * 4u + dst] - max_logit);
            denom += c[src * 4u + dst];
        }
        for (dst = 0u; dst < 4u; ++dst)
            c[src * 4u + dst] = c[src * 4u + dst] / denom + eps;
    }
    /* HF performs the initial column normalization before the remaining 19 pairs. */
    for (dst = 0u; dst < 4u; ++dst) {
        float denom = eps;
        for (src = 0u; src < 4u; ++src) denom += c[src * 4u + dst];
        if (!isfinite(denom) || denom == 0.0f) return GLM53_ARCH_MATH_NONFINITE_VALUE;
        for (src = 0u; src < 4u; ++src) c[src * 4u + dst] /= denom;
    }
    for (iteration = 1u; iteration < 20u; ++iteration) {
        for (src = 0u; src < 4u; ++src) {
            float denom = eps;
            for (dst = 0u; dst < 4u; ++dst) denom += c[src * 4u + dst];
            if (!isfinite(denom) || denom == 0.0f) return GLM53_ARCH_MATH_NONFINITE_VALUE;
            for (dst = 0u; dst < 4u; ++dst) c[src * 4u + dst] /= denom;
        }
        for (dst = 0u; dst < 4u; ++dst) {
            float denom = eps;
            for (src = 0u; src < 4u; ++src) denom += c[src * 4u + dst];
            if (!isfinite(denom) || denom == 0.0f) return GLM53_ARCH_MATH_NONFINITE_VALUE;
            for (src = 0u; src < 4u; ++src) c[src * 4u + dst] /= denom;
        }
    }
    memcpy(pre, p, sizeof(p));
    memcpy(post, q, sizeof(q));
    memcpy(comb, c, sizeof(c));
    return GLM53_ARCH_MATH_OK;
}

glm53_arch_math_status glm53_mhc_collapse4_f32(
    float *collapsed, const float *streams, size_t width, const float pre[4]) {
    size_t d, src;
    if (collapsed == NULL || streams == NULL || pre == NULL || width == 0u)
        return GLM53_ARCH_MATH_INVALID_ARGUMENT;
    for (src = 0u; src < 4u; ++src) if (!isfinite(pre[src]))
        return GLM53_ARCH_MATH_NONFINITE_VALUE;
    for (d = 0u; d < width; ++d) {
        float sum = 0.0f;
        for (src = 0u; src < 4u; ++src) {
            if (!isfinite(streams[src * width + d])) return GLM53_ARCH_MATH_NONFINITE_VALUE;
            sum += pre[src] * streams[src * width + d];
        }
        if (!isfinite(sum)) return GLM53_ARCH_MATH_NONFINITE_VALUE;
    }
    for (d = 0u; d < width; ++d) {
        float sum = 0.0f;
        for (src = 0u; src < 4u; ++src) sum += pre[src] * streams[src * width + d];
        collapsed[d] = sum;
    }
    return GLM53_ARCH_MATH_OK;
}

glm53_arch_math_status glm53_mhc_post_apply4_f32(
    float *out, const float *residual, const float *branch, size_t width,
    const float post[4], const float comb[16]) {
    size_t d, src, dst;
    if (out == NULL || residual == NULL || branch == NULL || post == NULL ||
        comb == NULL || width == 0u) return GLM53_ARCH_MATH_INVALID_ARGUMENT;
    for (dst = 0u; dst < 4u; ++dst) if (!isfinite(post[dst]))
        return GLM53_ARCH_MATH_NONFINITE_VALUE;
    for (src = 0u; src < 16u; ++src) if (!isfinite(comb[src]))
        return GLM53_ARCH_MATH_NONFINITE_VALUE;
    for (d = 0u; d < width; ++d) {
        if (!isfinite(branch[d])) return GLM53_ARCH_MATH_NONFINITE_VALUE;
        for (src = 0u; src < 4u; ++src)
            if (!isfinite(residual[src * width + d])) return GLM53_ARCH_MATH_NONFINITE_VALUE;
        for (dst = 0u; dst < 4u; ++dst) {
            float sum = post[dst] * branch[d];
            for (src = 0u; src < 4u; ++src)
                sum += comb[src * 4u + dst] * residual[src * width + d];
            if (!isfinite(sum)) return GLM53_ARCH_MATH_NONFINITE_VALUE;
        }
    }
    /* Four temporaries make exact in-place out==residual safe. */
    for (d = 0u; d < width; ++d) {
        float v[4];
        for (dst = 0u; dst < 4u; ++dst) {
            float sum = post[dst] * branch[d];
            for (src = 0u; src < 4u; ++src)
                sum += comb[src * 4u + dst] * residual[src * width + d];
            v[dst] = sum;
        }
        for (dst = 0u; dst < 4u; ++dst) out[dst * width + d] = v[dst];
    }
    return GLM53_ARCH_MATH_OK;
}
