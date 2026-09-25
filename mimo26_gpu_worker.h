#ifndef MIMO26_GPU_WORKER_H
#define MIMO26_GPU_WORKER_H

/*
 * The 48-layer MiMo text worker on gfx1151.
 *
 * Mirrors mimo26_worker.h deliberately -- same lifecycle, same transactional
 * step, same statistics -- so the two backends can be compared directly and
 * a caller can be written once. What differs is where the weights live and,
 * decisively, what form the experts are kept in.
 *
 * Experts are cached PACKED. tools/mimo26_gpu_bench measures the consequence:
 * 12.75 MiB packed against 48 MiB dequantized is 4.68 GiB of reads per token
 * against 17.62, which on this GTT is 26.7 tok/s against 7.1, and it lets 77%
 * of the 12,032 expert identities stay resident rather than 21%. The MXFP4
 * kernel decodes in registers, and G1 showed it bit-exact against the CPU
 * dequantizer, so nothing is given up for that.
 *
 * Admission is measured, not assumed: the plan requires the footprint to be
 * checked against real availability before any large allocation, and
 * hipMemGetInfo is consulted rather than a nominal device size.
 */

#include "mimo26_kv.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIMO26_GPU_WORKER_OK = 0,
    MIMO26_GPU_WORKER_INVALID_ARGUMENT,
    MIMO26_GPU_WORKER_OUT_OF_MEMORY,
    MIMO26_GPU_WORKER_MEMORY_LIMIT,
    MIMO26_GPU_WORKER_LOAD_FAILED,
    MIMO26_GPU_WORKER_DECODE_FAILED,
    MIMO26_GPU_WORKER_CAPACITY_EXCEEDED
} mimo26_gpu_worker_status;

typedef struct mimo26_gpu_worker mimo26_gpu_worker;

typedef struct {
    /* Positions retained by full-attention layers; windowed layers always
     * keep 128, so this alone sets the context ceiling. */
    size_t   global_kv_capacity;
    /*
     * Resident PACKED experts per MoE layer. 12.75 MiB each, so 64 slots
     * across 47 routed layers is 37.5 GiB -- which fits, where the same
     * cache dequantized would be 141 GiB and would not.
     */
    uint16_t expert_slots_per_layer;
    /* Refuse to start if the planned footprint exceeds this. 0 means use the
     * device's measured free memory instead. */
    uint64_t memory_limit_bytes;
    /*
     * Tokens per layer-major prefill chunk. 0 disables prefill and falls
     * back to feeding the prompt through decode, which is what this worker
     * did before and is kept as a reference path rather than deleted.
     *
     * Larger chunks amortize the BF16 projections further -- they are read
     * once per chunk rather than once per token -- at the cost of scratch
     * that scales with the chunk.
     */
    uint16_t prefill_chunk;
    /*
     * Prefill-only next-use eviction. Default true as of 2026-09-24: it is
     * what makes grouped prefill pay, and the grouped path is slower without
     * it. Arithmetic, decode scheduling and payload format are unchanged.
     */
    bool expert_lookahead;
    /*
     * Group a prefill chunk's tokens by selected expert. Default true as of
     * 2026-09-24. Output is bit-identical to per-token execution -- verified
     * on a mixed code/agentic corpus, 30/30 full vectors at both 128 and 160
     * slots -- and it is worth -5.8% warm prefill at 160 slots, where the
     * cache can hold a chunk's working set. At 128 slots it is neutral
     * (-0.7%, inside noise), so enabling it by default costs nothing there.
     *
     * Requires expert_lookahead: without the remaining-group schedule the
     * cache treats each group as a one-off access and the path is slower.
     */
    bool expert_major;
    /*
     * Ceiling on the attention score scratch, in bytes.
     *
     * The scratch a full chunk wants is chunk * 64 * (history + 2) floats,
     * which scales with the context: 67 MiB at a 2048 context but 8 GiB at
     * 262144 with a 128 chunk. Sized that way it does not fit beside the
     * expert cache, which is what made a 262144 default fail to load with
     * "prefill scratch allocation failed".
     *
     * Capping it instead makes the prefill path attend in sub-batches that
     * fit, which is bit-exact -- a narrower batch sees the same visible slots
     * in the same order. The cost is extra kernel launches at deep history
     * only; short contexts still run a full chunk in one launch.
     *
     * Zero means no cap, i.e. the old full-width sizing.
     */
    uint64_t attention_scratch_bytes;
} mimo26_gpu_worker_config;

void mimo26_gpu_worker_config_defaults(mimo26_gpu_worker_config *config);

/*
 * Floats the attention scratch will actually be allocated for under this
 * config. Exposed so the startup guard predicts the same number the loader
 * allocates rather than a second estimate of it.
 */
uint64_t mimo26_gpu_worker_attention_scratch_floats(
    const mimo26_gpu_worker_config *config);

/*
 * Grouped prefill as the worker will actually run it, after the
 * MIMO26_EXPERT_MAJOR override is resolved. Report this rather than the
 * requested config, or /health can advertise a mode that is not executing.
 */
bool mimo26_gpu_worker_expert_major(const mimo26_gpu_worker *worker);

/*
 * Fold the MIMO26_EXPERT_MAJOR override into a config, strictly. Call before
 * reporting or admitting a profile so that what is printed, what the guard
 * checks and what executes are the same thing. Idempotent; worker creation
 * calls it too, so direct embedders cannot skip it.
 */
mimo26_gpu_worker_status mimo26_gpu_worker_resolve_overrides(
    mimo26_gpu_worker_config *config, char *error, size_t error_size);

/*
 * Healthy and between requests: no execution in flight, no retention fault,
 * no open KV transaction, no outstanding I/O. Continuing a context requires
 * this exactly as resetting one does.
 */
bool mimo26_gpu_worker_idle(const mimo26_gpu_worker *worker);

/*
 * Persist and restore a context prefix, so a prompt already evaluated once can
 * be reloaded instead of re-prefilled. Both require an idle worker.
 *
 * Import replaces committed history and moves the position to match the file.
 * Whatever the outcome, the worker's position is left equal to the KV's own
 * length -- a rejected file leaves both untouched, and the one failure that
 * can reset the KV resets the position with it -- so the two can never
 * disagree about how much history exists.
 */
mimo26_gpu_worker_status mimo26_gpu_worker_export_state(
    const mimo26_gpu_worker *worker, const char *path,
    mimo26_kv_state_info *info, char *error, size_t error_size);

mimo26_gpu_worker_status mimo26_gpu_worker_import_state(
    mimo26_gpu_worker *worker, const char *path,
    mimo26_kv_state_info *info, char *error, size_t error_size);

/* Geometry identity of this worker's KV, for gating a store of checkpoints. */
uint64_t mimo26_gpu_worker_layout_crc64(const mimo26_gpu_worker *worker);

typedef struct {
    uint64_t tokens;
    uint64_t expert_accesses;
    uint64_t expert_hits;
    uint64_t expert_uploads;     /* host-to-device transfers performed */
    uint64_t aborted_steps;
    double   last_decode_seconds;
    double   load_seconds;
} mimo26_gpu_worker_stats;

/*
 * Build a worker over a checkpoint. Static text weights and every layer's
 * non-expert weights are uploaded and held resident; experts are uploaded on
 * demand into a bounded per-layer cache.
 */
mimo26_gpu_worker_status mimo26_gpu_worker_create(
    mimo26_gpu_worker **worker, const char *root,
    const mimo26_gpu_worker_config *config, char *error, size_t error_size);
void mimo26_gpu_worker_destroy(mimo26_gpu_worker *worker);

/* Planned device bytes, available before creation. */
uint64_t mimo26_gpu_worker_planned_bytes(
    const mimo26_gpu_worker_config *config);
uint64_t mimo26_gpu_worker_resident_bytes(const mimo26_gpu_worker *worker);

/*
 * Run one token through all 48 layers.
 *
 * logits is [152576] F32 on the host and receives the accumulation before
 * any BF16 rounding. Ids at or above the tokenizer's vocabulary are set to
 * -infinity, so a caller cannot sample an id the detokenizer cannot map --
 * the embedding is padded to a multiple of 128 and carries 901 such rows.
 *
 * The step is transactional: keys and values for all 48 layers are staged
 * and committed together, so a failure leaves committed history and the
 * position counter untouched and the caller may retry or cancel.
 */
/*
 * Feed a prompt layer-major, in chunks.
 *
 * Equivalent to calling decode once per token -- gated as such at the layer
 * level -- but reads the BF16 projections once per chunk instead of once per
 * token, which is the dominant cost of a prompt. logits receives the
 * distribution after the LAST token, which is all a prompt needs.
 *
 * Transactional per chunk: a failure leaves committed history and the
 * position at the last completed chunk, so a caller may retry the rest.
 */
/*
 * Called after each chunk lands, with the tokens consumed so far.
 *
 * Prefill does not return control per token, so without this a caller
 * arriving mid-prompt would wait for the whole prompt -- which at 8K is
 * fifteen minutes. Returning false asks prefill to stop cleanly at the last
 * committed chunk, which is what a cancelled request needs.
 */
typedef bool (*mimo26_gpu_prefill_progress)(void *context, size_t done,
                                            size_t total);

mimo26_gpu_worker_status mimo26_gpu_worker_prefill(
    mimo26_gpu_worker *worker, const uint32_t *tokens, size_t count,
    float *logits, mimo26_gpu_prefill_progress progress,
    void *progress_context, char *error, size_t error_size);

mimo26_gpu_worker_status mimo26_gpu_worker_decode(mimo26_gpu_worker *worker,
                                                  uint32_t token_id,
                                                  float *logits, char *error,
                                                  size_t error_size);

uint32_t mimo26_gpu_worker_argmax(const float *logits);

/*
 * Rewind committed steps, for a rejected speculative block. Mirrors
 * mimo26_worker_rollback so a caller written against the CPU backend works
 * against this one unchanged.
 */
mimo26_gpu_worker_status mimo26_gpu_worker_rollback(mimo26_gpu_worker *worker,
                                                    size_t count);
uint64_t mimo26_gpu_worker_position(const mimo26_gpu_worker *worker);
void mimo26_gpu_worker_reset(mimo26_gpu_worker *worker);
/* Single-owner, between requests only. Clears logical KV/position/resolved
 * pointers, preserving expert payloads, mappings and cumulative counters.
 * Refuses during execution or after any execution fault; cold reset does not
 * requalify a faulted worker for retention. Recreate it instead. The server
 * uses this only with explicit --retain-experts on; serving qualification
 * is profile-specific, and cold reset remains the server default. */
mimo26_gpu_worker_status mimo26_gpu_worker_reset_context(
    mimo26_gpu_worker *worker, char *error, size_t error_size);
void mimo26_gpu_worker_get_stats(const mimo26_gpu_worker *worker,
                                 mimo26_gpu_worker_stats *stats);

#define MIMO26_GPU_TOKENIZER_VOCAB 151675u

#ifdef __cplusplus
}
#endif

#endif
