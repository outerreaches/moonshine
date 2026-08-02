#ifndef K3_ENGINE_H
#define K3_ENGINE_H

#include "k3_prefill.h"
#include "k3_static_store.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k3_engine k3_engine;

typedef struct {
    k3_static_store_stats static_store;
    uint64_t cache_bytes;
    uint64_t state_bytes;
    uint64_t staging_bytes;
    uint32_t cache_slots;
    uint32_t staging_slots;
    double   startup_seconds;
    double   mla_pack_seconds;
} k3_engine_stats;

typedef struct {
    uint64_t batches;
    uint64_t accesses;
    uint64_t hits;
    uint64_t misses;
    uint64_t admissions;
    uint64_t evictions;
} k3_engine_cache_stats;

typedef struct {
    uint64_t routed_physical_read_bytes;
    uint64_t embedding_physical_read_bytes;
    uint64_t warm_cache_workspace_bytes;
    uint32_t routed_layer_sweeps;
    uint32_t expert_read_requests;
    uint64_t selected_expert_routes;
    uint64_t unique_experts_across_layers;
    uint32_t min_unique_experts_per_layer;
    uint32_t max_unique_experts_per_layer;
    double   layer0_seconds;
    double   attention_seconds;
    double   kda_attention_seconds;
    double   mla_attention_seconds;
    double   kda_projection_seconds;
    double   kda_convolution_seconds;
    double   kda_recurrent_seconds;
    double   kda_gate_norm_seconds;
    double   kda_output_projection_seconds;
    double   kda_dequantize_seconds;
    double   kda_blas_seconds;
    double   router_seconds;
    double   routed_stream_seconds;
    double   routed_read_wait_seconds;
    double   routed_submit_seconds;
    double   routed_index_seconds;
    double   routed_expert_pipeline_seconds;
    double   moe_tail_seconds;
    double   output_seconds;
    double   wall_seconds;
} k3_engine_prefill_stats;

typedef struct {
    uint64_t kda_state_hash;
    uint64_t kda_conv_hash;
    uint64_t mla_cache_hash;
    uint64_t attn_res_hash;
    uint32_t token_position;
} k3_engine_state_digest;

typedef struct {
    uint32_t format_version;
    uint32_t token_position;
    uint64_t model_layout_crc64;
    uint64_t payload_bytes;
    uint64_t file_bytes;
    uint64_t payload_crc64;
    bool     q8_projections;
    double   wall_seconds;
} k3_engine_state_file_info;

typedef enum {
    K3_PREFILL_PROJECTION_DEFAULT = 0,
    K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT = 1,
} k3_prefill_projection_backend;

/*
 * Create the single-node K3 decode residency. The model advertises up to
 * 1,048,576 positions; MLA cache and attention workspace scale with context:
 * approximately 27.6 KiB and 576 bytes per configured token, respectively.
 * The accepted Q8/32 host is configured-capacity qualified through 131,072
 * positions; filled-context latency and quality are separate qualifications.
 *   - source-precision or streamed/Q8 permanent static weights;
 *   - device-resident per-layer expert cache;
 *   - mapped fixed-registered O_DIRECT staging slots;
 *   - KDA recurrence/conv state and MLA cache/packed-key state.
 *
 * Before payload allocation, Linux hosts must pass a CMA-aware load/runtime
 * residency check. Unvalidated BF16 residency carries an additional guard.
 */
bool k3_engine_create(k3_engine       **out,
                      const char       *model_root,
                      uint32_t          context,
                      uint16_t          experts_per_layer,
                      uint16_t          staging_slots,
                      bool              q8_projections,
                      k3_engine_stats  *stats,
                      char             *error,
                      size_t            error_size);

const k3_static_weight *k3_engine_find_weight(
    const k3_engine *engine,
    const char      *name);

void *k3_engine_cache_slot(k3_engine *engine, uint32_t global_slot);
void *k3_engine_staging_host(k3_engine *engine, uint16_t slot);
void *k3_engine_staging_device(k3_engine *engine, uint16_t slot);

/*
 * Execute the exact layer-0 residual schedule from a device BF16 hidden vector:
 * save AttnRes block 0, input RMSNorm, KDA, MLP AttnRes, post RMSNorm, dense
 * SiTU MLP, and the final residual add.
 */
bool k3_engine_decode_layer0(k3_engine  *engine,
                             const void *input,
                             void       *output,
                             char       *error,
                             size_t      error_size);

/*
 * Execute the first routed layer. This composes pre-attention AttnRes,
 * persistent KDA state, QD2 fixed-buffer MXFP4 expert reads, per-layer device
 * cache admission, the latent/shared MoE tails, and the final residual add.
 * Layer 0 must have completed first.
 */
bool k3_engine_decode_layer1(k3_engine  *engine,
                             const void *input,
                             void       *output,
                             char       *error,
                             size_t      error_size);

/*
 * Execute the next routed layer selected by engine state. After layer 1 this
 * advances in order through layers 2..92, dispatching KDA or MLA attention and
 * applying 12-layer AttnRes block saves at the exact checkpoint boundaries.
 */
bool k3_engine_decode_next_layer(k3_engine  *engine,
                                 const void *input,
                                 void       *output,
                                 char       *error,
                                 size_t      error_size);

/* Stream one BF16 embedding row into a caller-owned device hidden vector. */
bool k3_engine_load_embedding(k3_engine *engine,
                              uint32_t   token_id,
                              void      *output,
                              char      *error,
                              size_t     error_size);

/*
 * Apply final output AttnRes and RMSNorm, execute the resident BF16 language
 * head, and return mapped-host greedy argmax. This completes one token and
 * resets the depth cursor while preserving KDA/MLA and expert-cache state.
 */
bool k3_engine_decode_greedy(k3_engine  *engine,
                             const void *input,
                             uint32_t   *token_id,
                             float      *token_value,
                             char       *error,
                             size_t      error_size);

void k3_engine_get_cache_stats(
    const k3_engine       *engine,
    k3_engine_cache_stats *stats);

/*
 * Complete one autoregressive token step from an input token ID. The engine
 * streams its embedding, traverses all 93 layers in order, applies the output
 * head, returns greedy next-token ID/value, and retains recurrent/cache state.
 */
bool k3_engine_forward_token(k3_engine *engine,
                             uint32_t   input_token,
                             uint32_t  *next_token,
                             float     *token_value,
                             char      *error,
                             size_t     error_size);

/*
 * Derive the exact payload-free plan at the engine's current context
 * position. Selecting the cold-workspace lease does not mutate cache ownership;
 * the caller still has to prove the cache is cold before execution.
 */
bool k3_engine_plan_prefill(
        const k3_engine       *engine,
        uint32_t               token_count,
        k3_prefill_cache_lease lease,
        k3_prefill_plan       *plan,
        char                  *error,
        size_t                 error_size);

/*
 * Diagnostic-only projection backend selection. The dequantize/BLAS
 * experiment applies only to routed KDA projections and requires Q8 static
 * residency; layer 0, MLA, and decode retain the release path.
 */
bool k3_engine_plan_prefill_with_projection_backend(
        const k3_engine              *engine,
        uint32_t                      token_count,
        k3_prefill_cache_lease        lease,
        k3_prefill_projection_backend backend,
        k3_prefill_plan              *plan,
        char                         *error,
        size_t                        error_size);

/*
 * Prove that a position-zero range prefill can use cache-backed workspace
 * before the caller destroys the current causal state. Clearing the expert
 * cache plans an exact cold loan; retaining it plans a slot-aligned warm tail.
 * This is payload-free, does not mutate the engine, and never relies on a
 * later separate allocation passing the host-memory guard.
 */
bool k3_engine_plan_reset_prefill(
        const k3_engine              *engine,
        uint32_t                      token_count,
        bool                          clear_expert_cache,
        k3_prefill_projection_backend backend,
        k3_prefill_plan              *plan,
        char                         *error,
        size_t                        error_size);

/*
 * Consume a causal token range in layer-major order, returning the greedy
 * next token for the last input row. The initial implementation retains the
 * complete decode cache and streams every routed layer once in physical
 * expert order. k3_engine_forward_token remains the decode regression path.
 */
bool k3_engine_forward_range(
        k3_engine               *engine,
        const uint32_t          *input_tokens,
        uint32_t                 token_count,
        uint32_t                *next_token,
        float                   *token_value,
        k3_engine_prefill_stats *stats,
        char                    *error,
        size_t                   error_size);

/*
 * Called after each completed layer during layer-major prefill. The callback
 * runs on the inference thread and must return quickly. It is informational:
 * inference continues even if a consumer can no longer receive progress.
 */
typedef void (*k3_engine_prefill_progress_callback)(
        uint32_t completed_layers,
        uint32_t total_layers,
        void    *user_data);

bool k3_engine_forward_range_with_progress(
        k3_engine                          *engine,
        const uint32_t                     *input_tokens,
        uint32_t                            token_count,
        uint32_t                           *next_token,
        float                              *token_value,
        k3_engine_prefill_stats            *stats,
        k3_engine_prefill_progress_callback callback,
        void                               *callback_data,
        char                               *error,
        size_t                              error_size);

bool k3_engine_forward_range_with_projection_backend(
        k3_engine                      *engine,
        const uint32_t                 *input_tokens,
        uint32_t                        token_count,
        k3_prefill_projection_backend   backend,
        uint32_t                       *next_token,
        float                          *token_value,
        k3_engine_prefill_stats        *stats,
        char                           *error,
        size_t                          error_size);

bool k3_engine_forward_range_with_projection_backend_and_progress(
        k3_engine                          *engine,
        const uint32_t                     *input_tokens,
        uint32_t                            token_count,
        k3_prefill_projection_backend       backend,
        uint32_t                           *next_token,
        float                              *token_value,
        k3_engine_prefill_stats            *stats,
        k3_engine_prefill_progress_callback progress_callback,
        void                               *progress_data,
        char                               *error,
        size_t                              error_size);

/*
 * Return semantic continuation state to the zero-position model state without
 * reloading static weights. When clear_expert_cache is true, forget all routed
 * cache mappings and telemetry while retaining the allocated slot storage.
 * This can recover an engine invalidated by a failed post-mutation import.
 */
bool k3_engine_reset_state(
        k3_engine *engine,
        bool       clear_expert_cache,
        char      *error,
        size_t     error_size);

/* Diagnostic causal-state digest used by sequential/range oracle tests. */
bool k3_engine_get_state_digest(
        const k3_engine         *engine,
        k3_engine_state_digest  *digest,
        char                    *error,
        size_t                   error_size);

/*
 * Atomically export only semantic continuation state: KDA recurrence and
 * convolution, occupied MLA rows, AttnRes blocks, and token position.
 * Derived MLA packed keys and routed-expert cache/LRU state are excluded.
 */
bool k3_engine_export_state(
        const k3_engine          *engine,
        const char               *path,
        k3_engine_state_file_info *info,
        char                     *error,
        size_t                    error_size);

/*
 * Validate format, header CRC, model/static identity, exact file length, and
 * the complete payload CRC before changing device state. A failure after
 * device mutation invalidates the engine until a later import succeeds.
 */
bool k3_engine_import_state(
        k3_engine                *engine,
        const char               *path,
        k3_engine_state_file_info *info,
        char                     *error,
        size_t                    error_size);

void k3_engine_destroy(k3_engine *engine);

#ifdef __cplusplus
}
#endif

#endif
