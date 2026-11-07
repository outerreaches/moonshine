#ifndef MIMO26_WEIGHTS_H
#define MIMO26_WEIGHTS_H

#include "k3_safetensors.h"
#include "mimo26_attention.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIMO26_WEIGHTS_OK = 0,
    MIMO26_WEIGHTS_INVALID_ARGUMENT,
    MIMO26_WEIGHTS_OUT_OF_MEMORY,
    MIMO26_WEIGHTS_MISSING_TENSOR,
    MIMO26_WEIGHTS_UNEXPECTED_LAYOUT,
    MIMO26_WEIGHTS_READ_FAILED,
    MIMO26_WEIGHTS_NONFINITE_VALUE
} mimo26_weights_status;

/*
 * One text layer's non-expert weights, dequantized to BF16.
 *
 * Quantized tensors are decoded in F32 and rounded once to BF16, matching the
 * reference's `dequantize(...).to(bfloat16)`. Storing BF16 rather than F32
 * halves resident size and is the dtype every consumer wants anyway.
 */
typedef struct {
    uint32_t                layer;
    mimo26_attention_config attention;
    bool                    is_moe;

    uint16_t *input_layernorm;           /* [4096] */
    uint16_t *post_attention_layernorm;  /* [4096] */
    uint16_t *qkv_proj;                  /* [qkv_width][4096], from F8_E4M3 */
    uint16_t *o_proj;                    /* [4096][8192], native BF16 */
    uint16_t *sink_bias;                 /* [64] on windowed layers, else NULL */

    uint16_t *gate_weight;               /* [256][4096] on MoE layers */
    float    *gate_bias;                 /* [256] F32 on MoE layers */

    /* Layer 0 only: the dense MLP, intermediate size 16384. */
    uint16_t *dense_gate;                /* [16384][4096] */
    uint16_t *dense_up;                  /* [16384][4096] */
    uint16_t *dense_down;                /* [4096][16384] */

    size_t bytes;                        /* resident footprint */
} mimo26_layer_weights;

/* One routed expert, dequantized from packed MXFP4 to BF16. */
typedef struct {
    uint32_t  layer;
    uint32_t  expert;
    uint16_t *gate;  /* [2048][4096] */
    uint16_t *up;    /* [2048][4096] */
    uint16_t *down;  /* [4096][2048] */
    size_t    bytes;
} mimo26_expert_weights;

/*
 * Load and dequantize. The model must come from mimo26_manifest_open_model,
 * so shard indices match the manifest. Every tensor's dtype and shape is
 * checked against the architecture contract before it is decoded; an
 * unexpected layout fails rather than being reinterpreted.
 */
mimo26_weights_status mimo26_layer_weights_load(
    mimo26_layer_weights *weights, const k3_st_model *model, uint32_t layer,
    char *error, size_t error_size);
void mimo26_layer_weights_free(mimo26_layer_weights *weights);

mimo26_weights_status mimo26_expert_weights_load(
    mimo26_expert_weights *weights, const k3_st_model *model, uint32_t layer,
    uint32_t expert, char *error, size_t error_size);
void mimo26_expert_weights_free(mimo26_expert_weights *weights);

/* Static text tensors shared by every layer. */
typedef struct {
    uint16_t *embed_tokens; /* [152576][4096] */
    uint16_t *norm;         /* [4096] */
    uint16_t *lm_head;      /* [152576][4096] */
    size_t    bytes;
} mimo26_static_weights;

mimo26_weights_status mimo26_static_weights_load(
    mimo26_static_weights *weights, const k3_st_model *model, char *error,
    size_t error_size);
void mimo26_static_weights_free(mimo26_static_weights *weights);

/*
 * Dequantizers, exposed so tests can exercise them against the checkpoint
 * without loading a whole layer.
 *
 * FP8: `codes` is [rows][cols] E4M3, `scales` is F32 with row stride
 * `scale_stride` and at least ceil(rows/128) x ceil(cols/128) live entries.
 * Surplus scale rows are permitted and ignored, which the global QKV grid
 * requires. MXFP4: `packed` is [rows][cols/2], `scales` is E8M0 per 32-element
 * block.
 */
mimo26_weights_status mimo26_dequantize_fp8_block(
    uint16_t *out, const uint8_t *codes, const float *scales,
    size_t scale_stride, size_t rows, size_t cols, char *error,
    size_t error_size);

mimo26_weights_status mimo26_dequantize_mxfp4(
    uint16_t *out, const uint8_t *packed, const uint8_t *scales, size_t rows,
    size_t cols, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
