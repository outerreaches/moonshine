#ifndef MIMO26_LAYER_H
#define MIMO26_LAYER_H

#include "mimo26_kv.h"
#include "mimo26_router.h"
#include "mimo26_weights.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIMO26_LAYER_OK = 0,
    MIMO26_LAYER_INVALID_ARGUMENT,
    MIMO26_LAYER_OUT_OF_MEMORY,
    MIMO26_LAYER_ARITHMETIC_FAILED,
    MIMO26_LAYER_EXPERT_UNAVAILABLE,
    MIMO26_LAYER_KV_FAILED
} mimo26_layer_status;

/*
 * Supplies one routed expert's dequantized weights.
 *
 * The layer deliberately does not own residency: a caller may keep every
 * expert in memory, stream from the checkpoint, or serve from a GPU-side
 * cache. Returning MIMO26_LAYER_EXPERT_UNAVAILABLE aborts the step without
 * touching committed KV, which is what makes a mid-layer streaming failure
 * recoverable.
 */
typedef mimo26_layer_status (*mimo26_expert_provider)(
    void *context, uint32_t layer, uint32_t expert,
    const mimo26_expert_weights **weights, char *error, size_t error_size);

typedef struct {
    const mimo26_layer_weights *weights;
    mimo26_expert_provider      provider;      /* required on MoE layers */
    void                       *provider_context;
} mimo26_layer;

/* Scratch buffers, allocated once and reused across steps and layers. */
typedef struct mimo26_layer_scratch mimo26_layer_scratch;

mimo26_layer_status mimo26_layer_scratch_create(mimo26_layer_scratch **scratch);
void mimo26_layer_scratch_destroy(mimo26_layer_scratch *scratch);
size_t mimo26_layer_scratch_bytes(const mimo26_layer_scratch *scratch);

/* Routing decision for one token, for diagnostics and regression fixtures. */
typedef struct {
    uint32_t experts[MIMO26_ROUTER_TOP_K];
    float    weights[MIMO26_ROUTER_TOP_K];
    bool     routed;   /* false on the dense layer */
} mimo26_layer_route;

/*
 * One decode step through one layer, following the reference's forward:
 *
 *   residual = hidden
 *   hidden   = attention(input_layernorm(hidden))            + residual
 *   residual = hidden
 *   hidden   = mlp(post_attention_layernorm(hidden))          + residual
 *
 * hidden is [4096] BF16, updated in place. The step's own key and value are
 * staged into kv but not committed: the caller commits once every layer has
 * run, so a failure part-way leaves committed history untouched. Attention
 * sees the staged token through the uncommitted-token path rather than a copy
 * of the history.
 *
 * route is optional and receives the selected experts in ascending id order.
 */
mimo26_layer_status mimo26_layer_decode(const mimo26_layer *layer,
                                        mimo26_layer_scratch *scratch,
                                        uint16_t *hidden, mimo26_kv_cache *kv,
                                        uint64_t position,
                                        mimo26_layer_route *route,
                                        char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
