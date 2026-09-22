#include "mimo26_layer.h"

#include "mimo26_architecture.h"
#include "mimo26_ops.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIMO26_DENSE_INTERMEDIATE 16384u
#define MIMO26_MOE_INTERMEDIATE 2048u
#define MIMO26_ATTENTION_OUT (MIMO26_QUERY_HEADS * MIMO26_V_HEAD_DIM)

struct mimo26_layer_scratch {
    uint16_t *residual;      /* [4096] */
    uint16_t *normed;        /* [4096] */
    uint16_t *fused;         /* [max qkv width] */
    uint16_t *query;         /* [64][192] */
    uint16_t *keys;          /* [8][192] */
    uint16_t *values;        /* [8][128] */
    uint16_t *heads;         /* [64][128] */
    uint16_t *projected;     /* [4096] */
    uint16_t *gate;          /* [16384] */
    uint16_t *up;            /* [16384] */
    uint16_t *activated;     /* [16384] */
    uint16_t *expert_out;    /* [4096] */
    float    *accumulator;   /* [4096] */
    float    *router_logits; /* [256] */
    uint16_t  cos_table[MIMO26_ROPE_DIM];
    uint16_t  sin_table[MIMO26_ROPE_DIM];
    size_t    bytes;
};

static mimo26_layer_status fail(char *error, size_t size,
                                mimo26_layer_status status,
                                const char *format, ...)
{
    if (error != NULL && size > 0) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return status;
}

void mimo26_layer_scratch_destroy(mimo26_layer_scratch *scratch)
{
    if (scratch == NULL) {
        return;
    }
    free(scratch->residual);
    free(scratch->normed);
    free(scratch->fused);
    free(scratch->query);
    free(scratch->keys);
    free(scratch->values);
    free(scratch->heads);
    free(scratch->projected);
    free(scratch->gate);
    free(scratch->up);
    free(scratch->activated);
    free(scratch->expert_out);
    free(scratch->accumulator);
    free(scratch->router_logits);
    free(scratch);
}

mimo26_layer_status mimo26_layer_scratch_create(mimo26_layer_scratch **out)
{
    if (out == NULL) {
        return MIMO26_LAYER_INVALID_ARGUMENT;
    }
    *out = NULL;
    mimo26_layer_scratch *scratch = calloc(1u, sizeof *scratch);
    if (scratch == NULL) {
        return MIMO26_LAYER_OUT_OF_MEMORY;
    }
    scratch->bytes = sizeof *scratch;

    /* Sized for the larger of the two layer kinds throughout, so one scratch
     * serves every layer. */
    struct { void **slot; size_t bytes; } allocations[] = {
        {(void **)&scratch->residual, MIMO26_HIDDEN_SIZE * sizeof(uint16_t)},
        {(void **)&scratch->normed, MIMO26_HIDDEN_SIZE * sizeof(uint16_t)},
        {(void **)&scratch->fused, MIMO26_SWA_QKV_WIDTH * sizeof(uint16_t)},
        {(void **)&scratch->query,
         MIMO26_QUERY_HEADS * MIMO26_QK_HEAD_DIM * sizeof(uint16_t)},
        {(void **)&scratch->keys,
         MIMO26_SWA_KV_HEADS * MIMO26_QK_HEAD_DIM * sizeof(uint16_t)},
        {(void **)&scratch->values,
         MIMO26_SWA_KV_HEADS * MIMO26_V_HEAD_DIM * sizeof(uint16_t)},
        {(void **)&scratch->heads, MIMO26_ATTENTION_OUT * sizeof(uint16_t)},
        {(void **)&scratch->projected, MIMO26_HIDDEN_SIZE * sizeof(uint16_t)},
        {(void **)&scratch->gate, MIMO26_DENSE_INTERMEDIATE * sizeof(uint16_t)},
        {(void **)&scratch->up, MIMO26_DENSE_INTERMEDIATE * sizeof(uint16_t)},
        {(void **)&scratch->activated,
         MIMO26_DENSE_INTERMEDIATE * sizeof(uint16_t)},
        {(void **)&scratch->expert_out, MIMO26_HIDDEN_SIZE * sizeof(uint16_t)},
        {(void **)&scratch->accumulator, MIMO26_HIDDEN_SIZE * sizeof(float)},
        {(void **)&scratch->router_logits,
         MIMO26_ROUTER_EXPERTS * sizeof(float)},
    };
    for (size_t i = 0; i < sizeof allocations / sizeof allocations[0]; i++) {
        *allocations[i].slot = calloc(1u, allocations[i].bytes);
        if (*allocations[i].slot == NULL) {
            mimo26_layer_scratch_destroy(scratch);
            return MIMO26_LAYER_OUT_OF_MEMORY;
        }
        scratch->bytes += allocations[i].bytes;
    }
    *out = scratch;
    return MIMO26_LAYER_OK;
}

size_t mimo26_layer_scratch_bytes(const mimo26_layer_scratch *scratch)
{
    return scratch != NULL ? scratch->bytes : 0u;
}

static mimo26_layer_status run_dense_mlp(const mimo26_layer_weights *weights,
                                         mimo26_layer_scratch *scratch,
                                         const uint16_t *input,
                                         uint16_t *output, char *error,
                                         size_t error_size)
{
    if (mimo26_matmul_bf16(scratch->gate, weights->dense_gate, input,
                           MIMO26_DENSE_INTERMEDIATE, MIMO26_HIDDEN_SIZE) !=
            MIMO26_OPS_OK ||
        mimo26_matmul_bf16(scratch->up, weights->dense_up, input,
                           MIMO26_DENSE_INTERMEDIATE, MIMO26_HIDDEN_SIZE) !=
            MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "dense gate/up projection failed");
    }
    if (mimo26_silu_product_bf16(scratch->activated, scratch->gate, scratch->up,
                                 MIMO26_DENSE_INTERMEDIATE) != MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "dense activation failed");
    }
    if (mimo26_matmul_bf16(output, weights->dense_down, scratch->activated,
                           MIMO26_HIDDEN_SIZE, MIMO26_DENSE_INTERMEDIATE) !=
        MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "dense down projection failed");
    }
    return MIMO26_LAYER_OK;
}

static mimo26_layer_status run_moe(const mimo26_layer *layer,
                                   mimo26_layer_scratch *scratch,
                                   const uint16_t *input, uint16_t *output,
                                   mimo26_layer_route *route, char *error,
                                   size_t error_size)
{
    const mimo26_layer_weights *weights = layer->weights;
    if (layer->provider == NULL) {
        return fail(error, error_size, MIMO26_LAYER_INVALID_ARGUMENT,
                    "layer %u is MoE but no expert provider was given",
                    weights->layer);
    }
    if (mimo26_router_check_grouping(MIMO26_ROUTER_GROUPS,
                                     MIMO26_ROUTER_TOPK_GROUPS) !=
        MIMO26_ROUTER_OK) {
        return fail(error, error_size, MIMO26_LAYER_INVALID_ARGUMENT,
                    "router grouping is unsupported");
    }

    /* The router projects in F32 from a BF16 weight matrix, so convert the
     * row on the fly rather than keeping an F32 copy of the gate. */
    for (size_t e = 0; e < MIMO26_ROUTER_EXPERTS; e++) {
        const uint16_t *row = weights->gate_weight + e * MIMO26_HIDDEN_SIZE;
        float sum = 0.0f;
        for (size_t i = 0; i < MIMO26_HIDDEN_SIZE; i++) {
            sum += mimo26_bf16_to_f32(row[i]) * mimo26_bf16_to_f32(input[i]);
        }
        scratch->router_logits[e] = sum;
    }

    uint32_t experts[MIMO26_ROUTER_TOP_K];
    float mixing[MIMO26_ROUTER_TOP_K];
    if (mimo26_router_select_f32(experts, mixing, scratch->router_logits,
                                 weights->gate_bias, MIMO26_ROUTER_EXPERTS,
                                 MIMO26_ROUTER_TOP_K, MIMO26_ROUTER_SCALE) !=
        MIMO26_ROUTER_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "router selection failed");
    }
    if (route != NULL) {
        memcpy(route->experts, experts, sizeof experts);
        memcpy(route->weights, mixing, sizeof mixing);
        route->routed = true;
    }

    memset(scratch->accumulator, 0,
           MIMO26_HIDDEN_SIZE * sizeof *scratch->accumulator);
    /* Ascending expert id, matching the declared accumulation order. */
    for (size_t j = 0; j < MIMO26_ROUTER_TOP_K; j++) {
        const mimo26_expert_weights *expert = NULL;
        const mimo26_layer_status status = layer->provider(
            layer->provider_context, weights->layer, experts[j], &expert,
            error, error_size);
        if (status != MIMO26_LAYER_OK) {
            return status;
        }
        if (expert == NULL) {
            return fail(error, error_size, MIMO26_LAYER_EXPERT_UNAVAILABLE,
                        "provider returned no weights for expert %u",
                        experts[j]);
        }
        if (mimo26_matmul_bf16(scratch->gate, expert->gate, input,
                               MIMO26_MOE_INTERMEDIATE, MIMO26_HIDDEN_SIZE) !=
                MIMO26_OPS_OK ||
            mimo26_matmul_bf16(scratch->up, expert->up, input,
                               MIMO26_MOE_INTERMEDIATE, MIMO26_HIDDEN_SIZE) !=
                MIMO26_OPS_OK) {
            return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                        "expert %u gate/up failed", experts[j]);
        }
        if (mimo26_silu_product_bf16(scratch->activated, scratch->gate,
                                     scratch->up, MIMO26_MOE_INTERMEDIATE) !=
            MIMO26_OPS_OK) {
            return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                        "expert %u activation failed", experts[j]);
        }
        if (mimo26_matmul_bf16(scratch->expert_out, expert->down,
                               scratch->activated, MIMO26_HIDDEN_SIZE,
                               MIMO26_MOE_INTERMEDIATE) != MIMO26_OPS_OK) {
            return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                        "expert %u down projection failed", experts[j]);
        }
        /* F32 accumulation, no per-expert rounding. */
        if (mimo26_expert_accumulate_f32(scratch->accumulator,
                                         scratch->expert_out, mixing[j],
                                         MIMO26_HIDDEN_SIZE) !=
            MIMO26_OPS_OK) {
            return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                        "expert %u accumulation failed", experts[j]);
        }
    }
    /* One BF16 rounding after the whole weighted sum. */
    if (mimo26_expert_finalize_bf16(output, scratch->accumulator,
                                    MIMO26_HIDDEN_SIZE) != MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "expert finalize failed");
    }
    return MIMO26_LAYER_OK;
}

mimo26_layer_status mimo26_layer_decode(const mimo26_layer *layer,
                                        mimo26_layer_scratch *scratch,
                                        uint16_t *hidden, mimo26_kv_cache *kv,
                                        uint64_t position,
                                        mimo26_layer_route *route,
                                        char *error, size_t error_size)
{
    if (layer == NULL || layer->weights == NULL || scratch == NULL ||
        hidden == NULL || kv == NULL) {
        return fail(error, error_size, MIMO26_LAYER_INVALID_ARGUMENT,
                    "invalid layer decode arguments");
    }
    const mimo26_layer_weights *weights = layer->weights;
    const mimo26_attention_config *attention = &weights->attention;
    if (route != NULL) {
        memset(route, 0, sizeof *route);
    }

    memcpy(scratch->residual, hidden, MIMO26_HIDDEN_SIZE * sizeof *hidden);

    if (mimo26_rmsnorm_bf16(scratch->normed, hidden, weights->input_layernorm,
                            MIMO26_HIDDEN_SIZE, MIMO26_LAYERNORM_EPSILON) !=
        MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "input layernorm failed");
    }
    if (mimo26_matmul_bf16(scratch->fused, weights->qkv_proj, scratch->normed,
                           attention->qkv_width, MIMO26_HIDDEN_SIZE) !=
        MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "qkv projection failed");
    }
    if (mimo26_attention_split_qkv(scratch->fused, attention, scratch->query,
                                   scratch->keys, scratch->values) !=
        MIMO26_ATTENTION_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "qkv split failed");
    }

    if (mimo26_rope_table(scratch->cos_table, scratch->sin_table, position,
                          attention->rope_theta) != MIMO26_ATTENTION_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "rope table failed at position %llu",
                    (unsigned long long)position);
    }
    for (size_t h = 0; h < MIMO26_QUERY_HEADS; h++) {
        if (mimo26_rope_apply(scratch->query + h * MIMO26_QK_HEAD_DIM,
                              scratch->cos_table, scratch->sin_table) !=
            MIMO26_ATTENTION_OK) {
            return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                        "rope failed on query head %zu", h);
        }
    }
    for (size_t h = 0; h < attention->kv_heads; h++) {
        if (mimo26_rope_apply(scratch->keys + h * MIMO26_QK_HEAD_DIM,
                              scratch->cos_table, scratch->sin_table) !=
            MIMO26_ATTENTION_OK) {
            return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                        "rope failed on key head %zu", h);
        }
    }

    /* Stage, do not commit: the caller commits after every layer succeeds. */
    if (mimo26_kv_stage(kv, weights->layer, scratch->keys, scratch->values) !=
        MIMO26_KV_OK) {
        return fail(error, error_size, MIMO26_LAYER_KV_FAILED,
                    "staging layer %u keys failed", weights->layer);
    }

    const uint16_t *history_keys = NULL;
    const uint16_t *history_values = NULL;
    size_t history = 0;
    uint64_t first_position = 0;
    if (mimo26_kv_view(kv, weights->layer, &history_keys, &history_values,
                       &history, &first_position) != MIMO26_KV_OK) {
        return fail(error, error_size, MIMO26_LAYER_KV_FAILED,
                    "reading layer %u history failed", weights->layer);
    }
    if (mimo26_attention_decode(scratch->heads, scratch->query, history_keys,
                                history_values, scratch->keys, scratch->values,
                                weights->sink_bias, attention, history,
                                first_position, position) !=
        MIMO26_ATTENTION_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "attention failed at position %llu",
                    (unsigned long long)position);
    }
    if (mimo26_matmul_bf16(scratch->projected, weights->o_proj, scratch->heads,
                           MIMO26_HIDDEN_SIZE, MIMO26_ATTENTION_OUT) !=
        MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "output projection failed");
    }
    if (mimo26_residual_add_bf16(hidden, scratch->residual, scratch->projected,
                                 MIMO26_HIDDEN_SIZE) != MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "attention residual failed");
    }

    memcpy(scratch->residual, hidden, MIMO26_HIDDEN_SIZE * sizeof *hidden);
    if (mimo26_rmsnorm_bf16(scratch->normed, hidden,
                            weights->post_attention_layernorm,
                            MIMO26_HIDDEN_SIZE, MIMO26_LAYERNORM_EPSILON) !=
        MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "post attention layernorm failed");
    }

    mimo26_layer_status status;
    if (weights->is_moe) {
        status = run_moe(layer, scratch, scratch->normed, scratch->projected,
                         route, error, error_size);
    } else {
        status = run_dense_mlp(weights, scratch, scratch->normed,
                               scratch->projected, error, error_size);
    }
    if (status != MIMO26_LAYER_OK) {
        return status;
    }
    if (mimo26_residual_add_bf16(hidden, scratch->residual, scratch->projected,
                                 MIMO26_HIDDEN_SIZE) != MIMO26_OPS_OK) {
        return fail(error, error_size, MIMO26_LAYER_ARITHMETIC_FAILED,
                    "mlp residual failed");
    }
    return MIMO26_LAYER_OK;
}
