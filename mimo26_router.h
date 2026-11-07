#ifndef MIMO26_ROUTER_H
#define MIMO26_ROUTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIMO26_ROUTER_EXPERTS 256u
#define MIMO26_ROUTER_TOP_K 8u

/* config n_group and topk_group are both 1, which makes the reference's
 * noaux_tc group mask a no-op. Asserted rather than assumed. */
#define MIMO26_ROUTER_GROUPS 1u
#define MIMO26_ROUTER_TOPK_GROUPS 1u

/* routed_scaling_factor is absent from config, so the reference default 1.0
 * applies. Declared here so it cannot be silently reinterpreted. */
#define MIMO26_ROUTER_SCALE 1.0f

/* The reference divides by (sum + 1e-20). Part of the contract, not slack. */
#define MIMO26_ROUTER_NORM_EPSILON 1.0e-20f

typedef enum {
    MIMO26_ROUTER_OK = 0,
    MIMO26_ROUTER_INVALID_ARGUMENT,
    MIMO26_ROUTER_NONFINITE_VALUE,
    MIMO26_ROUTER_UNSUPPORTED_GROUPING
} mimo26_router_status;

/*
 * Project a hidden state to expert logits.
 *
 * config sets moe_router_dtype to bfloat16, but the reference forward runs
 * F.linear(x.float(), w.float()) unconditionally. The executed arithmetic
 * wins, so this is F32 throughout: weight is [experts, hidden] row-major and
 * BF16 on disk, so the caller converts it once at load time.
 */
mimo26_router_status mimo26_router_logits_f32(
    float *logits, const float *hidden, const float *weight,
    size_t experts, size_t hidden_size);

/*
 * Select the top-k experts and produce their mixing weights.
 *
 * Selection uses sigmoid(logit) + e_score_correction_bias; the returned
 * weights are the *uncorrected* sigmoid values at the selected indices,
 * normalized by their sum plus the epsilon above and scaled.
 *
 * Indices are returned in **ascending expert id**, and the normalization sum
 * accumulates in that same order. The reference calls topk(sorted=False),
 * leaving both unspecified, so an order is declared here to make the result
 * reproducible. Selection ties resolve to the lower expert id.
 */
mimo26_router_status mimo26_router_select_f32(
    uint32_t *indices, float *weights, const float *logits,
    const float *bias, size_t experts, size_t k, float scale);

/* Convenience wrapper at the model's shape: top 8 of 256. */
mimo26_router_status mimo26_router_top8_256_f32(
    uint32_t indices[MIMO26_ROUTER_TOP_K],
    float weights[MIMO26_ROUTER_TOP_K],
    const float logits[MIMO26_ROUTER_EXPERTS],
    const float bias[MIMO26_ROUTER_EXPERTS]);

/* Reject a checkpoint whose grouping would make the no-op mask meaningful. */
mimo26_router_status mimo26_router_check_grouping(size_t groups,
                                                  size_t topk_groups);

#ifdef __cplusplus
}
#endif

#endif
