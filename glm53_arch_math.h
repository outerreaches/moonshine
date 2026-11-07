#ifndef GLM53_ARCH_MATH_H
#define GLM53_ARCH_MATH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CPU binary32 oracles for the GLM-5.3-Flash equations supported by
 * Hugging Face Transformers commit eb4d9e2a64a013bec12289288b85d0b1210ba0aa. */
#define GLM53_MHC_H 4u
#define GLM53_MHC_MIX 24u
#define GLM53_ROUTER_EXPERTS 288u
#define GLM53_ROUTER_TOP_K 8u

typedef enum glm53_arch_math_status {
    GLM53_ARCH_MATH_OK = 0,
    GLM53_ARCH_MATH_INVALID_ARGUMENT,
    GLM53_ARCH_MATH_NONFINITE_VALUE
} glm53_arch_math_status;

/* Elementwise limited SwiGLU. gate is capped above (not below), up is clamped
 * to [-limit,+limit], and out = silu(gate)*up. Buffers must not overlap. */
glm53_arch_math_status glm53_limited_swiglu_f32(
    float *out, const float *gate, const float *up, size_t count, float limit);

/* Select k experts by sigmoid(logits[e]) + bias[e], but return normalized raw
 * sigmoid(logits[e]) weights, multiplied by scale. Selection ties are resolved
 * by the lower expert index. Each expert can occur at most once, so duplicate
 * output indices are impossible. Returned entries are in descending selection
 * order. Inputs and outputs must not overlap. */
glm53_arch_math_status glm53_router_topk_f32(
    size_t *indices, float *weights, const float *logits, const float *bias,
    size_t num_experts, size_t k, float scale);

/* Model-shape convenience wrapper: top 8 of 288 experts. */
glm53_arch_math_status glm53_router_top8_288_f32(
    size_t indices[GLM53_ROUTER_TOP_K],
    float weights[GLM53_ROUTER_TOP_K],
    const float logits[GLM53_ROUTER_EXPERTS],
    const float bias[GLM53_ROUTER_EXPERTS], float scale);

/* Convert the 24 already-projected mHC logits and learned base/scale values
 * into pre[4], post[4], and C[4][4]. C is row-major C[src,dst]. This follows
 * the HF order exactly: sigmoid transforms; row softmax + eps; one column
 * normalization; then 19 row-and-column normalization pairs (20 iterations).
 * Inputs and outputs must not overlap. */
glm53_arch_math_status glm53_mhc_weights4_f32(
    float pre[GLM53_MHC_H], float post[GLM53_MHC_H],
    float comb[GLM53_MHC_H * GLM53_MHC_H],
    const float mix[GLM53_MHC_MIX], const float base[GLM53_MHC_MIX],
    const float scale[3], float eps);

/* Collapse streams[src,feature] into one vector using pre[src]. */
glm53_arch_math_status glm53_mhc_collapse4_f32(
    float *collapsed, const float *streams, size_t width,
    const float pre[GLM53_MHC_H]);

/* Apply HF expansion: out[dst,d] = post[dst]*branch[d] +
 * sum_src C[src,dst]*residual[src,d]. out may equal residual exactly; other
 * overlap is not supported. */
glm53_arch_math_status glm53_mhc_post_apply4_f32(
    float *out, const float *residual, const float *branch, size_t width,
    const float post[GLM53_MHC_H],
    const float comb[GLM53_MHC_H * GLM53_MHC_H]);

#ifdef __cplusplus
}
#endif
#endif
