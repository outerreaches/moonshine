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
    if (history > scratch->attention_capacity) {
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
 * Experts are still applied per token rather than grouped expert-major.
 * That is the next optimization and it is deliberately not bundled with
 * this change: going layer-major already removes the BF16 re-reads, which
 * are the larger term, and doing both at once would leave a regression with
 * two candidate causes. The cache still amortizes across the chunk, since a
 * second token wanting the same expert finds it resident.
 */
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
    /* Refuse rather than overrun scratch sized for a single token. */
    if (count > scratch->batch_capacity ||
        history + count > scratch->attention_capacity) {
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
