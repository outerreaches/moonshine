#ifndef MIMO26_ROCM_LAYER_H
#define MIMO26_ROCM_LAYER_H

/*
 * One MiMo text layer on the GPU.
 *
 * Composes the primitives gated by G1-G3 into the reference's forward:
 *
 *   residual = hidden
 *   hidden   = attention(input_layernorm(hidden))    + residual
 *   residual = hidden
 *   hidden   = mlp(post_attention_layernorm(hidden)) + residual
 *
 * Weights are device pointers the caller owns and keeps resident. Residency
 * and eviction are the worker's business, not this layer's, so nothing here
 * allocates or frees.
 *
 * Experts arrive through a provider callback in their PACKED MXFP4 form,
 * never dequantized. That is measured, not stylistic: packed is 12.75 MiB
 * against 48 MiB expanded, and tools/mimo26_gpu_bench puts the same GTT at
 * 26.7 tok/s packed against 7.1 expanded, with 77% of all expert identities
 * resident rather than 21%.
 *
 * The provider is called from the host after the router has run on device
 * and its selection has been read back, which costs one synchronization per
 * MoE layer per token. That is the same plan-then-commit shape the CPU
 * worker uses, and it is what lets a bounded cache decide what to admit
 * before any expert is touched.
 */

#include "mimo26_rocm_ops.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIMO26_ROCM_LAYER_OK = 0,
    MIMO26_ROCM_LAYER_INVALID_ARGUMENT,
    MIMO26_ROCM_LAYER_LAUNCH_FAILED,
    MIMO26_ROCM_LAYER_EXPERT_UNAVAILABLE,
    MIMO26_ROCM_LAYER_SYNC_FAILED
} mimo26_rocm_layer_status;

/* Device pointers to one expert's three packed projections and their E8M0
 * scales. Every field must be set for the provider to count as satisfied. */
typedef struct {
    const void *gate_packed;
    const void *gate_scales;
    const void *up_packed;
    const void *up_scales;
    const void *down_packed;
    const void *down_scales;
} mimo26_rocm_expert;

/*
 * Make one expert resident and report where it landed. Returning false is a
 * hard failure for the step; the layer does not fall back to a different
 * expert, because a silently different expert is a silently different model.
 */
typedef bool (*mimo26_rocm_expert_provider)(void *context, uint32_t layer,
                                            uint32_t expert,
                                            mimo26_rocm_expert *out);

/*
 * Optional: announce the whole selection before any of it is requested, so a
 * bounded cache can plan admissions and evictions knowing the full working
 * set. Without it a cache can evict an expert this same step will need.
 */
typedef bool (*mimo26_rocm_expert_prepare)(void *context, uint32_t layer,
                                           const uint32_t *experts,
                                           size_t count);

/* Optional prefill-only callback. Future routes are valid only during this
 * synchronous call: [future_tokens][TOP_K], from this same chunk, excluding
 * the current token. Never retain these pointers or use them asynchronously. */
typedef bool (*mimo26_rocm_expert_prepare_future)(
    void *context, uint32_t layer, const uint32_t *experts, size_t count,
    const uint32_t *future, size_t future_tokens);

// Experimental grouped-prefill lookahead. remaining is the unique, ordered
// list of groups still to execute in THIS chunk, not token/top-k tuples.
// Synchronous callback only; no host pointer may be retained.
typedef bool (*mimo26_rocm_expert_prepare_group_future)(
    void *context, uint32_t layer, uint32_t expert,
    const uint32_t *remaining, size_t remaining_count);

typedef struct {
    uint32_t layer;
    bool     is_swa;
    bool     is_moe;
    uint32_t kv_heads;
    uint32_t kv_groups;
    uint32_t qkv_width;
    uint32_t window;          /* 0 for full attention */

    /* Device pointers, all BF16 unless noted. */
    const void *input_layernorm;            /* [4096] */
    const void *post_attention_layernorm;   /* [4096] */
    const void *qkv_proj;                   /* [qkv_width][4096] */
    const void *o_proj;                     /* [4096][8192] */
    const void *sink_bias;                  /* [64], windowed layers only */
    const void *gate_weight;                /* [256][4096], MoE layers */
    const float *gate_bias;                 /* [256] F32, MoE layers */
    const void *dense_gate;                 /* [16384][4096], layer 0 only */
    const void *dense_up;                   /* [16384][4096] */
    const void *dense_down;                 /* [4096][16384] */
} mimo26_rocm_layer_weights;

/*
 * Device scratch, allocated once by the worker and reused every step. Sized
 * for the widest layer, so one allocation serves all 48.
 */
typedef struct {
    void  *normed;        /* [4096] bf16 */
    void  *fused;         /* [max qkv_width] bf16 */
    void  *query;         /* [64][192] bf16 */
    void  *key;           /* [kv_heads][192] bf16 */
    void  *value;         /* [kv_heads][128] bf16 */
    void  *attention;     /* [64][128] bf16 */
    void  *projected;     /* [4096] bf16 */
    void  *mlp_gate;      /* [16384] bf16, dense layer needs the full width */
    void  *mlp_up;        /* [16384] bf16 */
    void  *mlp_active;    /* [16384] bf16 */
    void  *expert_out;    /* [batch][4096] bf16; [4096] when decode-only */
    /*
     * Expert-major prefill scratch. Per-token execution re-reads a selected
     * expert's 12.75 MiB from GTT once for every token that picks it;
     * grouping a chunk's tokens permits reuse. The exact grouped-GEMV path
     * only amortizes launches; it does NOT load weights once per group.
     * One route sample suggests 7.56x ideal sharing, not measured traffic
     * reduction (and the former weight-reuse kernel failed equality).
     *
     * gathered holds one expert's token rows contiguously for the batched
     * projection. stash holds every (token, rank) expert output, so the final F32
     * accumulation still runs in the original top-k rank order rather than in
     * whatever order the experts happened to be executed. All three are NULL
     * on a decode-only scratch.
     */
    void     *expert_gathered;  /* [batch][4096] bf16 */
    void     *expert_stash;     /* [batch][8][4096] bf16 */
    uint32_t *expert_row_ids;   /* [2 * batch * 8] u32: gather ids, then scatter */
    float *accumulator;   /* [4096] f32 */
    float *router_logits; /* [256] f32 */
    float *router_weights;/* [8] f32 */
    uint32_t *router_ids; /* [8] u32 */
    float *attention_scratch;     /* mimo26_rocm_attention_scratch_floats() */
    uint64_t attention_capacity;  /* deepest history the context allows */
    /*
     * Floats actually allocated. Kept apart from attention_capacity because a
     * full-width chunk at the deepest history wants 8 GiB at a 262144 context,
     * which does not fit beside the expert cache. The prefill path attends in
     * sub-batches that fit this buffer instead, so the allocation is a tuning
     * decision rather than a context limit.
     */
    uint64_t attention_scratch_floats;
    /*
     * Prefill widths. Zero means decode-only: the prefill entry point
     * refuses rather than overrunning buffers sized for one token, which is
     * the failure a shared scratch struct invites.
     */
    uint32_t batch_capacity;

    /* Host-visible staging for the router read-back. */
    uint32_t host_ids[MIMO26_ROCM_TOP_K];
    float    host_weights[MIMO26_ROCM_TOP_K];
} mimo26_rocm_layer_scratch;

typedef struct {
    const mimo26_rocm_layer_weights *weights;
    mimo26_rocm_expert_prepare       prepare;   /* optional */
    mimo26_rocm_expert_provider      provider;  /* required on MoE layers */
    void                            *provider_context;
    mimo26_rocm_expert_prepare_future prepare_future; /* optional; zero-init */
    mimo26_rocm_expert_prepare_group_future prepare_group_future; /* optional; zero-init */
    /*
     * Group this chunk's tokens by selected expert. Set by the worker from
     * its config, so the profile is explicit and visible on /health rather
     * than depending on the environment. MIMO26_EXPERT_MAJOR=0 or =1 still
     * overrides it for A/B work.
     */
    bool expert_major;
    /*
     * Run the expert projections through the weight-reuse tiled GEMM
     * (k3_rocm_mxfp4_gemm_bf16) instead of k3_rocm_mxfp4_gemv_rows_bf16, which
     * re-reads the expert's weights once per vector block.
     *
     * NOT bit-exact to gemv_rows: they differ on ~1 in 30,000 outputs of a real
     * down projection. A float64 oracle says the tile is the one that is RIGHT
     * where they differ -- 94 disagreements over 3.08M outputs at 17 widths,
     * all 94 nearer the true value, none against -- and it is ~2.1x faster at
     * production group sizes, 2.8x at width 128. See
     * [[MiMo MXFP4 Kernel Oracle 2026-09-26]].
     *
     * Off by default all the same, because turning it on re-bases every
     * full-vector equality fixture in the lane. Set by the worker from its
     * config so the mode is visible on /health, never read from the
     * environment here.
     */
    bool expert_weight_reuse;
} mimo26_rocm_layer;

/*
 * One decode step. hidden is [4096] BF16 on device, updated in place.
 *
 * keys is [history][kv_heads][192] and values [history][kv_heads][128], both
 * device-resident and both excluding this step's own token, which is passed
 * to attention through the uncommitted path. The caller commits the staged
 * key and value -- left in scratch->key and scratch->value -- only once every
 * layer has run, so a failure part-way leaves committed history untouched.
 *
 * cos_table and sin_table are [64] BF16 on device for this absolute position,
 * built on the host by mimo26_rope_table.
 *
 * route, when given, receives the selected experts in ascending id order.
 */
/*
 * One layer over a whole chunk of tokens, layer-major.
 *
 * This is the path prefill should have been using from the start. The decode
 * entry point below walks one token through all 48 layers, so every token
 * re-reads the 10.72 GB of BF16 projections that a chunk reads once. Going
 * layer-major turns that into a per-chunk cost, and bounds the expert reads
 * per layer by the router union over the chunk rather than 8 per token.
 *
 * hidden is [count][4096] and is updated in place. keys and values hold the
 * prior history and are EXTENDED with this chunk's entries at `history`
 * before attention runs, so attention sees exactly what it would see if the
 * tokens had been decoded one at a time -- which is gated as such.
 *
 * cos_tables and sin_tables are [count][64] BF16, one pair per token,
 * built on the host by mimo26_rope_table.
 *
 * routes, when given, receives [count][8] selected experts.
 */
mimo26_rocm_layer_status mimo26_rocm_layer_prefill(
    const mimo26_rocm_layer *layer, mimo26_rocm_layer_scratch *scratch,
    void *hidden, void *keys, void *values, const void *cos_tables,
    const void *sin_tables, uint64_t history, uint64_t first_position,
    uint64_t first_token_position, uint32_t count, uint32_t *routes,
    void *stream);

mimo26_rocm_layer_status mimo26_rocm_layer_decode(
    const mimo26_rocm_layer *layer, mimo26_rocm_layer_scratch *scratch,
    void *hidden, const void *keys, const void *values,
    const void *cos_table, const void *sin_table, uint64_t history,
    uint64_t first_position, uint64_t position, uint32_t *route,
    void *stream);

#ifdef __cplusplus
}
#endif

#endif
