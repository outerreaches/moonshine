#ifndef MIMO26_ATTENTION_H
#define MIMO26_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIMO26_QUERY_HEADS 64u
#define MIMO26_QK_HEAD_DIM 192u
#define MIMO26_V_HEAD_DIM 128u
#define MIMO26_GLOBAL_KV_HEADS 4u
#define MIMO26_SWA_KV_HEADS 8u

/* int(192 * 0.334); the first 64 coordinates rotate, the other 128 do not. */
#define MIMO26_ROPE_DIM 64u
#define MIMO26_ROPE_PAIRS (MIMO26_ROPE_DIM / 2u)

#define MIMO26_GLOBAL_ROPE_THETA 1.0e7f
#define MIMO26_SWA_ROPE_THETA 1.0e4f

/* Literally 0.707, not 1/sqrt(2). Applied to V before the cache write. */
#define MIMO26_VALUE_SCALE 0.707f

/* q - 128 < kv <= q, so the window includes the current token and 128 ring
 * slots suffice. Eviction begins at position 128. */
#define MIMO26_SLIDING_WINDOW 128u

/* Fused QKV widths: 64*192 + kv*192 + kv*128. */
#define MIMO26_GLOBAL_QKV_WIDTH 13568u
#define MIMO26_SWA_QKV_WIDTH 14848u

typedef enum {
    MIMO26_ATTENTION_OK = 0,
    MIMO26_ATTENTION_INVALID_ARGUMENT,
    MIMO26_ATTENTION_NONFINITE_VALUE
} mimo26_attention_status;

typedef struct {
    bool   is_swa;
    size_t kv_heads;
    size_t kv_groups;   /* query heads per KV head */
    size_t qkv_width;
    float  rope_theta;
    bool   has_sink;    /* SWA layers only */
    size_t window;      /* 0 means unbounded (full attention) */
} mimo26_attention_config;

/* Derive the attention shape for a text layer from the architecture's
 * explicit layer pattern. Layer 0 is dense but its attention is global. */
mimo26_attention_status mimo26_attention_config_for_layer(
    uint32_t layer, mimo26_attention_config *config);

/*
 * Split a fused QKV projection output into per-head Q, K and V.
 *
 * Layout is [q | k | v] with q = 64*192, k = kv_heads*192, v = kv_heads*128.
 * V is multiplied by MIMO26_VALUE_SCALE here, matching the reference, which
 * scales before the cache write -- so a cache holds pre-scaled V.
 *
 * Buffers are BF16 as uint16_t: fused is [qkv_width], q is
 * [64][192], k is [kv_heads][192], v is [kv_heads][128].
 */
mimo26_attention_status mimo26_attention_split_qkv(
    const uint16_t *fused, const mimo26_attention_config *config,
    uint16_t *q, uint16_t *k, uint16_t *v);

/*
 * Build the RoPE table for one absolute position.
 *
 * inv_freq[j] = 1 / theta^(2j/64) for j < 32, freqs[j] = position *
 * inv_freq[j], and the table duplicates those 32 values across 64 entries
 * exactly as the reference's cat((freqs, freqs)) does. Values are computed in
 * F32 and then rounded to BF16, because the reference casts cos/sin to the
 * activation dtype before use -- the rotation itself runs in BF16.
 */
mimo26_attention_status mimo26_rope_table(uint16_t cos_table[MIMO26_ROPE_DIM],
                                          uint16_t sin_table[MIMO26_ROPE_DIM],
                                          uint64_t position, float theta);

/*
 * Rotate the first 64 coordinates of one head vector in place, leaving the
 * remaining 128 untouched. Split-half (NeoX) convention:
 *   out[j]    = x[j]*cos[j]    - x[j+32]*sin[j]
 *   out[j+32] = x[j+32]*cos[j] + x[j]*sin[j]
 */
mimo26_attention_status mimo26_rope_apply(
    uint16_t head[MIMO26_QK_HEAD_DIM],
    const uint16_t cos_table[MIMO26_ROPE_DIM],
    const uint16_t sin_table[MIMO26_ROPE_DIM]);

/*
 * One decode step against a key/value history.
 *
 * keys is [history][kv_heads][192] and values is [history][kv_heads][128],
 * both BF16 and both in ascending absolute-position order starting at
 * first_position. query is [64][192], already rotated. out is [64][128].
 *
 * Masking uses absolute positions: kv is visible when
 * kv_position <= query_position and, for a windowed layer,
 * kv_position > query_position - window.
 *
 * sink_bias is [64] per-head BF16 and required when config->has_sink. Its
 * logit participates in the softmax and its probability is then discarded,
 * so the surviving probabilities deliberately sum to less than one.
 *
 * Cast points follow the reference: scores are rounded to BF16 after scaling,
 * the row max is subtracted in BF16, the softmax runs in F32, probabilities
 * are rounded to BF16, and the value-weighted sum is accumulated in F32 and
 * rounded once into out.
 */
mimo26_attention_status mimo26_attention_decode(
    uint16_t *out, const uint16_t *query, const uint16_t *keys,
    const uint16_t *values, const uint16_t *sink_bias,
    const mimo26_attention_config *config, size_t history,
    uint64_t first_position, uint64_t query_position);

/* Softmax scale, from the QK dimension rather than the V dimension. */
float mimo26_attention_scale(void);

/* Visibility predicate, exposed so tests can assert it independently. */
bool mimo26_attention_visible(uint64_t kv_position, uint64_t query_position,
                              size_t window);

#ifdef __cplusplus
}
#endif

#endif
