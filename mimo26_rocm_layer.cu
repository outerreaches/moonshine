#include "mimo26_rocm_layer.h"

#include "k3_rocm_ops.h"

#include <hip/hip_runtime.h>

#include <stdlib.h>
#include <string.h>

#define ROUTER_SCALE 1.0f
#define RMS_EPSILON 1e-6f
#define QUERY_HEADS 64u
#define QK_DIM 192u
#define V_DIM 128u
#define DENSE_INTERMEDIATE 16384u
#define EXPERT_INTERMEDIATE 2048u

/* mimo26_attention_scale(), inlined so this file does not pull in the CPU
 * attention module just for a constant: 1/sqrt(192), from the QK dimension
 * rather than the V dimension. */
static float attention_scale(void)
{
    return 1.0f / sqrtf((float)QK_DIM);
}

static mimo26_rocm_layer_status run_mlp_dense(
    const mimo26_rocm_layer_weights *w, mimo26_rocm_layer_scratch *scratch,
    hipStream_t stream)
{
    if (!k3_rocm_bf16_gemv_bf16(scratch->mlp_gate, w->dense_gate,
                                scratch->normed, DENSE_INTERMEDIATE,
                                MIMO26_ROCM_HIDDEN, stream) ||
        !k3_rocm_bf16_gemv_bf16(scratch->mlp_up, w->dense_up, scratch->normed,
                                DENSE_INTERMEDIATE, MIMO26_ROCM_HIDDEN,
                                stream) ||
        !mimo26_rocm_silu_product_bf16(scratch->mlp_active, scratch->mlp_gate,
                                       scratch->mlp_up, DENSE_INTERMEDIATE,
                                       stream) ||
        !k3_rocm_bf16_gemv_bf16(scratch->projected, w->dense_down,
                                scratch->mlp_active, MIMO26_ROCM_HIDDEN,
                                DENSE_INTERMEDIATE, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    return MIMO26_ROCM_LAYER_OK;
}

static mimo26_rocm_layer_status run_mlp_moe(
    const mimo26_rocm_layer *layer, mimo26_rocm_layer_scratch *scratch,
    uint32_t *route, hipStream_t stream)
{
    const mimo26_rocm_layer_weights *w = layer->weights;

    /*
     * Router logits are F32: the reference runs F.linear(x.float(),
     * w.float()) regardless of moe_router_dtype, so rounding to BF16 here
     * would round too early. MiMo's own kernel rather than K3's, because
     * this sum cancels heavily and its association has to match the CPU's --
     * see mimo26_rocm_router_logits_f32.
     */
    if (!mimo26_rocm_router_logits_f32(scratch->router_logits, w->gate_weight,
                                       scratch->normed, MIMO26_ROCM_EXPERTS,
                                       MIMO26_ROCM_HIDDEN, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_router_topk(scratch->router_ids, scratch->router_weights,
                                 scratch->router_logits, w->gate_bias, 1u,
                                 MIMO26_ROCM_EXPERTS, MIMO26_ROCM_TOP_K,
                                 ROUTER_SCALE, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }

    /*
     * Read the selection back. This is the one synchronization per MoE layer
     * per token, and it is what buys a bounded cache the chance to plan: the
     * host learns the whole working set before any expert is touched.
     */
    if (hipMemcpyAsync(scratch->host_ids, scratch->router_ids,
                       sizeof scratch->host_ids, hipMemcpyDeviceToHost,
                       stream) != hipSuccess ||
        hipMemcpyAsync(scratch->host_weights, scratch->router_weights,
                       sizeof scratch->host_weights, hipMemcpyDeviceToHost,
                       stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        return MIMO26_ROCM_LAYER_SYNC_FAILED;
    }
    if (route != NULL) {
        for (uint32_t k = 0; k < MIMO26_ROCM_TOP_K; k++) {
            route[k] = scratch->host_ids[k];
        }
    }

    if (layer->prepare != NULL &&
        !layer->prepare(layer->provider_context, w->layer, scratch->host_ids,
                        MIMO26_ROCM_TOP_K)) {
        return MIMO26_ROCM_LAYER_EXPERT_UNAVAILABLE;
    }

    if (!mimo26_rocm_zero_f32(scratch->accumulator, MIMO26_ROCM_HIDDEN,
                              stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    /*
     * Ascending expert id, matching the order the router declares and the
     * order the CPU accumulates in. F32 addition is not associative, so
     * visiting the experts in a different order would give a different
     * answer -- defensibly, but not the same one.
     */
    for (uint32_t k = 0; k < MIMO26_ROCM_TOP_K; k++) {
        mimo26_rocm_expert expert;
        expert.gate_packed = NULL;
        expert.gate_scales = NULL;
        expert.up_packed = NULL;
        expert.up_scales = NULL;
        expert.down_packed = NULL;
        expert.down_scales = NULL;
        if (!layer->provider(layer->provider_context, w->layer,
                             scratch->host_ids[k], &expert) ||
            expert.gate_packed == NULL || expert.gate_scales == NULL ||
            expert.up_packed == NULL || expert.up_scales == NULL ||
            expert.down_packed == NULL || expert.down_scales == NULL) {
            return MIMO26_ROCM_LAYER_EXPERT_UNAVAILABLE;
        }
        if (!k3_rocm_mxfp4_gemv_bf16(scratch->mlp_gate, expert.gate_packed,
                                     expert.gate_scales, scratch->normed,
                                     EXPERT_INTERMEDIATE, MIMO26_ROCM_HIDDEN,
                                     stream) ||
            !k3_rocm_mxfp4_gemv_bf16(scratch->mlp_up, expert.up_packed,
                                     expert.up_scales, scratch->normed,
                                     EXPERT_INTERMEDIATE, MIMO26_ROCM_HIDDEN,
                                     stream) ||
            !mimo26_rocm_silu_product_bf16(scratch->mlp_active,
                                           scratch->mlp_gate, scratch->mlp_up,
                                           EXPERT_INTERMEDIATE, stream) ||
            !k3_rocm_mxfp4_gemv_bf16(scratch->expert_out, expert.down_packed,
                                     expert.down_scales, scratch->mlp_active,
                                     MIMO26_ROCM_HIDDEN, EXPERT_INTERMEDIATE,
                                     stream) ||
            !mimo26_rocm_expert_accumulate_f32(
                scratch->accumulator, scratch->expert_out,
                scratch->host_weights[k], MIMO26_ROCM_HIDDEN, stream)) {
            return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
        }
    }
    /* One rounding at the end, not one per expert. */
    if (!mimo26_rocm_expert_finalize_bf16(scratch->projected,
                                          scratch->accumulator,
                                          MIMO26_ROCM_HIDDEN, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    return MIMO26_ROCM_LAYER_OK;
}

extern "C"
mimo26_rocm_layer_status mimo26_rocm_layer_decode(
    const mimo26_rocm_layer *layer, mimo26_rocm_layer_scratch *scratch,
    void *hidden, const void *keys, const void *values,
    const void *cos_table, const void *sin_table, uint64_t history,
    uint64_t first_position, uint64_t position, uint32_t *route,
    void *stream_handle)
{
    if (layer == NULL || layer->weights == NULL || scratch == NULL ||
        hidden == NULL || cos_table == NULL || sin_table == NULL) {
        return MIMO26_ROCM_LAYER_INVALID_ARGUMENT;
    }
    const mimo26_rocm_layer_weights *w = layer->weights;
    if (w->is_moe && layer->provider == NULL) {
        return MIMO26_ROCM_LAYER_INVALID_ARGUMENT;
    }
    if (history > scratch->attention_capacity ||
        scratch->attention_scratch_floats <
            mimo26_rocm_attention_scratch_floats(history)) {
        return MIMO26_ROCM_LAYER_INVALID_ARGUMENT;
    }
    hipStream_t stream = (hipStream_t)stream_handle;

    /* --- attention --- */
    if (!mimo26_rocm_rmsnorm_bf16(scratch->normed, hidden,
                                  w->input_layernorm, 1u, MIMO26_ROCM_HIDDEN,
                                  RMS_EPSILON, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!k3_rocm_bf16_gemv_bf16(scratch->fused, w->qkv_proj, scratch->normed,
                                w->qkv_width, MIMO26_ROCM_HIDDEN, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_split_qkv(scratch->query, scratch->key, scratch->value,
                               scratch->fused, w->kv_heads, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_rope_apply(scratch->query, cos_table, sin_table,
                                QUERY_HEADS, stream) ||
        !mimo26_rocm_rope_apply(scratch->key, cos_table, sin_table,
                                w->kv_heads, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    /* The staged key and value stay in scratch and reach attention through
     * the uncommitted-token path, so committed history is untouched until
     * every layer has run. */
    if (!mimo26_rocm_attention_decode(
            scratch->attention, scratch->query, history ? keys : NULL,
            history ? values : NULL, scratch->key, scratch->value,
            w->is_swa ? w->sink_bias : NULL, scratch->attention_scratch,
            w->kv_heads, w->kv_groups, w->window, history, first_position,
            position, attention_scale(), stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!k3_rocm_bf16_gemv_bf16(scratch->projected, w->o_proj,
                                scratch->attention, MIMO26_ROCM_HIDDEN,
                                QUERY_HEADS * V_DIM, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_residual_add_bf16(hidden, hidden, scratch->projected,
                                       MIMO26_ROCM_HIDDEN, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }

    /* --- MLP --- */
    if (!mimo26_rocm_rmsnorm_bf16(scratch->normed, hidden,
                                  w->post_attention_layernorm, 1u,
                                  MIMO26_ROCM_HIDDEN, RMS_EPSILON, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    const mimo26_rocm_layer_status status =
        w->is_moe ? run_mlp_moe(layer, scratch, route, stream)
                  : run_mlp_dense(w, scratch, stream);
    if (status != MIMO26_ROCM_LAYER_OK) {
        return status;
    }
    if (!mimo26_rocm_residual_add_bf16(hidden, hidden, scratch->projected,
                                       MIMO26_ROCM_HIDDEN, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    return MIMO26_ROCM_LAYER_OK;
}

/* ---- layer-major prefill ---- */

/*
 * The MoE half for a chunk.
 *
 * The default applies experts per token. Explicit expert-major experiments
 * group token rows, but retain the same per-row projection arithmetic.
 */
/*
 * Experimental expert-major execution. Enable only with
 * MIMO26_EXPERT_MAJOR=1, and qualify the exact binary/profile before use.
 *
 * The first implementation used tiled MXFP4 GEMM and was NOT equivalent.
 * A real-input projection guard found a down-projection mismatch in layer 3
 * of the FIRST 128-token chunk: one BF16 element was 0x35f7 versus GEMV's
 * 0x35f6. The old synthetic width gate covered only gate/up and missed it.
 * Final-logit failures at 224+ tokens did not establish an eviction bug or
 * prove the preceding tokens' hidden/KV state correct. All old grouped
 * speedups are withdrawn; see the vault's 2026-09-24 review evidence.
 *
 * This path instead launches the UNCHANGED GEMV kernel over a 2-D grid.
 * Each vector has its own block and preserves baseline reduction/rounding.
 * It amortizes launches, but does not implement explicit cross-vector
 * weight reuse: route sharing (7.56x on one captured prompt) is an opportunity,
 * not measured bandwidth reduction or a speedup for this implementation.
 *
 * Outputs are stashed by (token, rank), then accumulated in original rank
 * order, preserving the F32 sum regardless of group execution order.
 */
/*
 * No environment read here. This used to resolve MIMO26_EXPERT_MAJOR itself,
 * which meant the layer could run a mode the server had never agreed to and
 * /health could not see. It also treated every value except a literal "0" as
 * true, so MIMO26_EXPERT_MAJOR=off turned grouping ON. The override is now
 * resolved once, strictly, before the worker exists, and arrives here as
 * layer->expert_major like any other part of the profile.
 */

static mimo26_rocm_layer_status run_mlp_moe_expert_major(
    const mimo26_rocm_layer *layer, mimo26_rocm_layer_scratch *scratch,
    uint32_t count, const uint32_t *ids, const float *weights,
    hipStream_t stream)
{
    const mimo26_rocm_layer_weights *w = layer->weights;
    const size_t selections = (size_t)count * MIMO26_ROCM_TOP_K;

    /* Counting sort of (token, rank) pairs by expert id. */
    uint32_t offset[MIMO26_ROCM_EXPERTS + 1u];
    memset(offset, 0, sizeof offset);
    for (size_t i = 0; i < selections; i++) {
        if (ids[i] >= MIMO26_ROCM_EXPERTS) {
            return MIMO26_ROCM_LAYER_EXPERT_UNAVAILABLE;
        }
        offset[ids[i] + 1u]++;
    }
    for (uint32_t e = 0; e < MIMO26_ROCM_EXPERTS; e++) {
        offset[e + 1u] += offset[e];
    }
    uint32_t *host_ids = (uint32_t *)malloc(2u * selections * sizeof *host_ids);
    if (host_ids == NULL) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    uint32_t *gather = host_ids;               /* row of scratch->normed */
    uint32_t *scatter = host_ids + selections; /* row of scratch->expert_stash */
    {
        uint32_t cursor[MIMO26_ROCM_EXPERTS];
        memcpy(cursor, offset, sizeof cursor);
        for (size_t i = 0; i < selections; i++) {
            const uint32_t slot = cursor[ids[i]]++;
            gather[slot] = (uint32_t)(i / MIMO26_ROCM_TOP_K);
            scatter[slot] = (uint32_t)i;
        }
    }
    const bool uploaded =
        hipMemcpyAsync(scratch->expert_row_ids, host_ids,
                       2u * selections * sizeof *host_ids,
                       hipMemcpyHostToDevice, stream) == hipSuccess;
    free(host_ids);
    if (!uploaded) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    const uint32_t *device_gather = scratch->expert_row_ids;
    const uint32_t *device_scatter = scratch->expert_row_ids + selections;

    /*
     * Visit groups least-shared first.
     *
     * Expert-major gives every expert exactly one access per chunk, which
     * erases the access-frequency signal LRU depends on: in the per-token
     * path a heavily-shared expert is touched by many tokens and stays hot,
     * whereas here a 30-token expert and a 1-token expert look identical to
     * the policy. Ascending id order then evicts almost everything between
     * chunks, which cost 4x more SSD reads than the per-token path.
     *
     * Ordering by group size restores the signal through recency instead:
     * the most-shared experts are touched last, so they are the ones LRU
     * keeps for the next chunk, which is also the ones most likely to be
     * wanted. Execution order does not affect the result -- outputs are
     * stashed per (token, rank) and accumulated separately.
     */
    uint32_t order[MIMO26_ROCM_EXPERTS];
    uint32_t order_count = 0u;
    for (uint32_t e = 0; e < MIMO26_ROCM_EXPERTS; e++) {
        if (offset[e + 1u] > offset[e]) {
            order[order_count++] = e;
        }
    }
    for (uint32_t i = 1u; i < order_count; i++) {
        const uint32_t key = order[i];
        const uint32_t key_members = offset[key + 1u] - offset[key];
        uint32_t j = i;
        while (j > 0u &&
               (offset[order[j - 1u] + 1u] - offset[order[j - 1u]]) >
                   key_members) {
            order[j] = order[j - 1u];
            j--;
        }
        order[j] = key;
    }

    for (uint32_t index = 0; index < order_count; index++) {
        const uint32_t e = order[index];
        const uint32_t members = offset[e + 1u] - offset[e];
        /*
         * One expert admitted at a time. A chunk can select more distinct
         * experts than the cache has slots, so admitting the whole chunk up
         * front would thrash; admitting immediately before its group runs
         * also means each expert is read at most once per chunk.
         */
        const bool prepared = layer->prepare_group_future != NULL
            ? layer->prepare_group_future(layer->provider_context, w->layer, e,
                                          order + index + 1u, order_count - index - 1u)
            : layer->prepare == NULL || layer->prepare(layer->provider_context, w->layer, &e, 1u);
        if (!prepared) {
            return MIMO26_ROCM_LAYER_EXPERT_UNAVAILABLE;
        }
        mimo26_rocm_expert expert;
        memset(&expert, 0, sizeof expert);
        if (!layer->provider(layer->provider_context, w->layer, e, &expert) ||
            expert.gate_packed == NULL || expert.down_scales == NULL) {
            return MIMO26_ROCM_LAYER_EXPERT_UNAVAILABLE;
        }
        /*
         * gemv_rows amortizes launches but re-reads the expert's weights once
         * per vector block; the tiled form reuses them across a 16-vector tile.
         * Selected by the profile, resolved in the worker -- never read from the
         * environment here, for the reason above.
         */
        const auto project = layer->expert_weight_reuse ? k3_rocm_mxfp4_gemm_bf16
                                                        : k3_rocm_mxfp4_gemv_rows_bf16;
        if (!k3_rocm_gather_rows_bf16(scratch->expert_gathered, scratch->normed,
                                      device_gather + offset[e], members,
                                      MIMO26_ROCM_HIDDEN, stream) ||
            !project(scratch->mlp_gate, expert.gate_packed,
                                     expert.gate_scales,
                                     scratch->expert_gathered, members,
                                     EXPERT_INTERMEDIATE, MIMO26_ROCM_HIDDEN,
                                     stream) ||
            !project(scratch->mlp_up, expert.up_packed,
                                     expert.up_scales,
                                     scratch->expert_gathered, members,
                                     EXPERT_INTERMEDIATE, MIMO26_ROCM_HIDDEN,
                                     stream) ||
            !mimo26_rocm_silu_product_bf16(
                scratch->mlp_active, scratch->mlp_gate, scratch->mlp_up,
                (uint64_t)members * EXPERT_INTERMEDIATE, stream) ||
            !project(scratch->expert_out, expert.down_packed,
                                     expert.down_scales, scratch->mlp_active,
                                     members, MIMO26_ROCM_HIDDEN,
                                     EXPERT_INTERMEDIATE, stream) ||
            !k3_rocm_scatter_rows_bf16(scratch->expert_stash,
                                       scratch->expert_out,
                                       device_scatter + offset[e], members,
                                       MIMO26_ROCM_HIDDEN, stream)) {
            return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
        }
    }

    if (!mimo26_rocm_zero_f32(scratch->accumulator,
                              (uint64_t)count * MIMO26_ROCM_HIDDEN, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    /* Original (token, rank) order, so the F32 sum matches per-token exactly. */
    for (uint32_t token = 0; token < count; token++) {
        float *accumulator_row =
            scratch->accumulator + (size_t)token * MIMO26_ROCM_HIDDEN;
        for (uint32_t k = 0; k < MIMO26_ROCM_TOP_K; k++) {
            const size_t index = (size_t)token * MIMO26_ROCM_TOP_K + k;
            const uint16_t *stashed = (const uint16_t *)scratch->expert_stash +
                                      index * MIMO26_ROCM_HIDDEN;
            if (!mimo26_rocm_expert_accumulate_f32(accumulator_row, stashed,
                                                   weights[index],
                                                   MIMO26_ROCM_HIDDEN,
                                                   stream)) {
                return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
            }
        }
    }
    return MIMO26_ROCM_LAYER_OK;
}

static mimo26_rocm_layer_status run_mlp_moe_batch(
    const mimo26_rocm_layer *layer, mimo26_rocm_layer_scratch *scratch,
    uint32_t count, uint32_t *routes, hipStream_t stream)
{
    const mimo26_rocm_layer_weights *w = layer->weights;

    if (!mimo26_rocm_ordered_gemv_f32_batch(scratch->router_logits,
                                            w->gate_weight, scratch->normed,
                                            MIMO26_ROCM_EXPERTS,
                                            MIMO26_ROCM_HIDDEN, count,
                                            stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_router_topk(scratch->router_ids, scratch->router_weights,
                                 scratch->router_logits, w->gate_bias, count,
                                 MIMO26_ROCM_EXPERTS, MIMO26_ROCM_TOP_K,
                                 ROUTER_SCALE, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }

    /*
     * One synchronization for the whole chunk rather than one per token.
     * That is a real part of the win: the decode path pays a device-host
     * round trip per token per MoE layer, 47 of them, and here it is 47 per
     * chunk.
     */
    const size_t selection_bytes = (size_t)count * MIMO26_ROCM_TOP_K;
    uint32_t *ids = (uint32_t *)malloc(selection_bytes * sizeof *ids);
    float *weights = (float *)malloc(selection_bytes * sizeof *weights);
    if (ids == NULL || weights == NULL) {
        free(ids);
        free(weights);
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (hipMemcpyAsync(ids, scratch->router_ids,
                       selection_bytes * sizeof *ids, hipMemcpyDeviceToHost,
                       stream) != hipSuccess ||
        hipMemcpyAsync(weights, scratch->router_weights,
                       selection_bytes * sizeof *weights,
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        free(ids);
        free(weights);
        return MIMO26_ROCM_LAYER_SYNC_FAILED;
    }
    if (routes != NULL) {
        memcpy(routes, ids, selection_bytes * sizeof *ids);
    }

    /*
     * Explicit experimental opt-in; never a production default. The scratch check
     * is not belt-and-braces: a decode-only scratch has no stash to hold the
     * per-rank outputs, and silently falling back is better than overrunning.
     */
    if (layer->expert_major && count > 1u &&
        scratch->expert_stash != NULL && scratch->expert_gathered != NULL &&
        scratch->expert_row_ids != NULL) {
        const mimo26_rocm_layer_status grouped =
            run_mlp_moe_expert_major(layer, scratch, count, ids, weights,
                                     stream);
        free(ids);
        free(weights);
        if (grouped != MIMO26_ROCM_LAYER_OK) {
            return grouped;
        }
        return mimo26_rocm_expert_finalize_bf16(
                   scratch->projected, scratch->accumulator,
                   (uint64_t)count * MIMO26_ROCM_HIDDEN, stream)
                   ? MIMO26_ROCM_LAYER_OK
                   : MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }

    mimo26_rocm_layer_status status = MIMO26_ROCM_LAYER_OK;
    if (!mimo26_rocm_zero_f32(scratch->accumulator,
                              (uint64_t)count * MIMO26_ROCM_HIDDEN,
                              stream)) {
        status = MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    for (uint32_t token = 0; token < count && status == MIMO26_ROCM_LAYER_OK;
         token++) {
        const uint32_t *selection = ids + (size_t)token * MIMO26_ROCM_TOP_K;
        const bool prepared = layer->prepare_future != NULL
            ? layer->prepare_future(layer->provider_context, w->layer, selection,
                                    MIMO26_ROCM_TOP_K,
                                    ids + (size_t)(token + 1) * MIMO26_ROCM_TOP_K,
                                    count - token - 1)
            : (layer->prepare == NULL || layer->prepare(
                   layer->provider_context, w->layer, selection, MIMO26_ROCM_TOP_K));
        if (!prepared) {
            status = MIMO26_ROCM_LAYER_EXPERT_UNAVAILABLE;
            break;
        }
        uint16_t *normed_row =
            (uint16_t *)scratch->normed + (size_t)token * MIMO26_ROCM_HIDDEN;
        float *accumulator_row =
            scratch->accumulator + (size_t)token * MIMO26_ROCM_HIDDEN;
        for (uint32_t k = 0; k < MIMO26_ROCM_TOP_K; k++) {
            mimo26_rocm_expert expert;
            memset(&expert, 0, sizeof expert);
            if (!layer->provider(layer->provider_context, w->layer,
                                 selection[k], &expert) ||
                expert.gate_packed == NULL || expert.down_scales == NULL) {
                status = MIMO26_ROCM_LAYER_EXPERT_UNAVAILABLE;
                break;
            }
            if (!k3_rocm_mxfp4_gemv_bf16(scratch->mlp_gate, expert.gate_packed,
                                         expert.gate_scales, normed_row,
                                         EXPERT_INTERMEDIATE,
                                         MIMO26_ROCM_HIDDEN, stream) ||
                !k3_rocm_mxfp4_gemv_bf16(scratch->mlp_up, expert.up_packed,
                                         expert.up_scales, normed_row,
                                         EXPERT_INTERMEDIATE,
                                         MIMO26_ROCM_HIDDEN, stream) ||
                !mimo26_rocm_silu_product_bf16(scratch->mlp_active,
                                               scratch->mlp_gate,
                                               scratch->mlp_up,
                                               EXPERT_INTERMEDIATE, stream) ||
                !k3_rocm_mxfp4_gemv_bf16(scratch->expert_out,
                                         expert.down_packed,
                                         expert.down_scales,
                                         scratch->mlp_active,
                                         MIMO26_ROCM_HIDDEN,
                                         EXPERT_INTERMEDIATE, stream) ||
                !mimo26_rocm_expert_accumulate_f32(
                    accumulator_row, scratch->expert_out,
                    weights[(size_t)token * MIMO26_ROCM_TOP_K + k],
                    MIMO26_ROCM_HIDDEN, stream)) {
                status = MIMO26_ROCM_LAYER_LAUNCH_FAILED;
                break;
            }
        }
    }
    free(ids);
    free(weights);
    if (status != MIMO26_ROCM_LAYER_OK) {
        return status;
    }
    if (!mimo26_rocm_expert_finalize_bf16(
            scratch->projected, scratch->accumulator,
            (uint64_t)count * MIMO26_ROCM_HIDDEN, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    return MIMO26_ROCM_LAYER_OK;
}

extern "C"
mimo26_rocm_layer_status mimo26_rocm_layer_prefill(
    const mimo26_rocm_layer *layer, mimo26_rocm_layer_scratch *scratch,
    void *hidden, void *keys, void *values, const void *cos_tables,
    const void *sin_tables, uint64_t history, uint64_t first_position,
    uint64_t first_token_position, uint32_t count, uint32_t *routes,
    void *stream_handle)
{
    if (layer == NULL || layer->weights == NULL || scratch == NULL ||
        hidden == NULL || keys == NULL || values == NULL ||
        cos_tables == NULL || sin_tables == NULL || count == 0u) {
        return MIMO26_ROCM_LAYER_INVALID_ARGUMENT;
    }
    /* Refuse rather than overrun scratch sized for a single token. The
     * attention scratch is checked for one query's row only -- wider batches
     * are split to fit rather than refused. */
    if (count > scratch->batch_capacity ||
        history + count > scratch->attention_capacity ||
        scratch->attention_scratch_floats <
            mimo26_rocm_attention_scratch_floats(history + count)) {
        return MIMO26_ROCM_LAYER_INVALID_ARGUMENT;
    }
    const mimo26_rocm_layer_weights *w = layer->weights;
    if (w->is_moe && layer->provider == NULL) {
        return MIMO26_ROCM_LAYER_INVALID_ARGUMENT;
    }
    hipStream_t stream = (hipStream_t)stream_handle;

    /* --- attention --- */
    if (!mimo26_rocm_rmsnorm_bf16(scratch->normed, hidden,
                                  w->input_layernorm, count,
                                  MIMO26_ROCM_HIDDEN, RMS_EPSILON, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    /* The GEMM keeps the GEMV's per-row reduction order, so batching the
     * projection does not change any row's value. */
    if (!k3_rocm_bf16_gemm_bf16(scratch->fused, w->qkv_proj, scratch->normed,
                                count, w->qkv_width, MIMO26_ROCM_HIDDEN,
                                stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_split_qkv_batch(scratch->query, scratch->key,
                                     scratch->value, scratch->fused,
                                     w->kv_heads, w->qkv_width, count,
                                     stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_rope_apply_batch(scratch->query, cos_tables, sin_tables,
                                      QUERY_HEADS, count, stream) ||
        !mimo26_rocm_rope_apply_batch(scratch->key, cos_tables, sin_tables,
                                      w->kv_heads, count, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    /*
     * The chunk's keys and values join the history before attention runs.
     * Attention then masks causally on absolute position, so token b sees
     * exactly the prior history plus chunk entries 0..b -- the same view it
     * would have had decoding alone.
     */
    if (!mimo26_rocm_append_kv(keys, values, scratch->key, scratch->value,
                               w->kv_heads, history, count, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_attention_prefill(
            scratch->attention, scratch->query, keys, values,
            w->is_swa ? w->sink_bias : NULL, scratch->attention_scratch,
            scratch->attention_scratch_floats,
            w->kv_heads, w->kv_groups, w->window, history + count,
            first_position, first_token_position, count, attention_scale(),
            stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!k3_rocm_bf16_gemm_bf16(scratch->projected, w->o_proj,
                                scratch->attention, count,
                                MIMO26_ROCM_HIDDEN, QUERY_HEADS * V_DIM,
                                stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (!mimo26_rocm_residual_add_bf16(hidden, hidden, scratch->projected,
                                       (uint64_t)count * MIMO26_ROCM_HIDDEN,
                                       stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }

    /* --- MLP --- */
    if (!mimo26_rocm_rmsnorm_bf16(scratch->normed, hidden,
                                  w->post_attention_layernorm, count,
                                  MIMO26_ROCM_HIDDEN, RMS_EPSILON, stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    mimo26_rocm_layer_status status;
    if (w->is_moe) {
        status = run_mlp_moe_batch(layer, scratch, count, routes, stream);
    } else {
        /* The dense layer is a straight batch: three GEMMs and a product. */
        status = (k3_rocm_bf16_gemm_bf16(scratch->mlp_gate, w->dense_gate,
                                         scratch->normed, count,
                                         DENSE_INTERMEDIATE,
                                         MIMO26_ROCM_HIDDEN, stream) &&
                  k3_rocm_bf16_gemm_bf16(scratch->mlp_up, w->dense_up,
                                         scratch->normed, count,
                                         DENSE_INTERMEDIATE,
                                         MIMO26_ROCM_HIDDEN, stream) &&
                  mimo26_rocm_silu_product_bf16(
                      scratch->mlp_active, scratch->mlp_gate, scratch->mlp_up,
                      (uint64_t)count * DENSE_INTERMEDIATE, stream) &&
                  k3_rocm_bf16_gemm_bf16(scratch->projected, w->dense_down,
                                         scratch->mlp_active, count,
                                         MIMO26_ROCM_HIDDEN,
                                         DENSE_INTERMEDIATE, stream))
                     ? MIMO26_ROCM_LAYER_OK
                     : MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    if (status != MIMO26_ROCM_LAYER_OK) {
        return status;
    }
    if (!mimo26_rocm_residual_add_bf16(hidden, hidden, scratch->projected,
                                       (uint64_t)count * MIMO26_ROCM_HIDDEN,
                                       stream)) {
        return MIMO26_ROCM_LAYER_LAUNCH_FAILED;
    }
    return MIMO26_ROCM_LAYER_OK;
}
