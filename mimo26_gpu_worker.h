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
} mimo26_gpu_worker_config;

void mimo26_gpu_worker_config_defaults(mimo26_gpu_worker_config *config);

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
mimo26_gpu_worker_status mimo26_gpu_worker_prefill(mimo26_gpu_worker *worker,
                                                   const uint32_t *tokens,
                                                   size_t count,
                                                   float *logits,
                                                   char *error,
                                                   size_t error_size);

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
void mimo26_gpu_worker_get_stats(const mimo26_gpu_worker *worker,
                                 mimo26_gpu_worker_stats *stats);

#define MIMO26_GPU_TOKENIZER_VOCAB 151675u

#ifdef __cplusplus
}
#endif

#endif
