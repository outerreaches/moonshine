#ifndef MIMO26_WORKER_H
#define MIMO26_WORKER_H

#include "mimo26_architecture.h"
#include "mimo26_layer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIMO26_WORKER_OK = 0,
    MIMO26_WORKER_INVALID_ARGUMENT,
    MIMO26_WORKER_OUT_OF_MEMORY,
    MIMO26_WORKER_MEMORY_LIMIT,
    MIMO26_WORKER_LOAD_FAILED,
    MIMO26_WORKER_DECODE_FAILED,
    MIMO26_WORKER_CAPACITY_EXCEEDED,
    MIMO26_WORKER_BUSY
} mimo26_worker_status;

typedef struct mimo26_worker mimo26_worker;

typedef struct {
    /* Positions retained by full-attention layers. Windowed layers always
     * keep 128, so this alone sets the context ceiling. */
    size_t   global_kv_capacity;
    /* Committed steps that can be rewound; 8 covers the draft block size. */
    size_t   rollback_depth;
    /*
     * Resident experts per MoE layer.
     *
     * This worker caches experts *dequantized* to BF16, which is 48 MiB each
     * against 12.75 MiB packed -- 3.76x more. A GPU path must instead cache
     * the packed bytes and decode in the kernel, which is what the K3 MXFP4
     * kernels already do. The budget in tools/mimo26_budget.py is stated in
     * packed bytes and therefore does NOT describe this worker's residency.
     */
    uint16_t expert_slots_per_layer;
    /* Refuse to start if the planned footprint exceeds this. 0 disables. */
    uint64_t memory_limit_bytes;
} mimo26_worker_config;

void mimo26_worker_config_defaults(mimo26_worker_config *config);

typedef struct {
    uint64_t tokens;
    uint64_t expert_accesses;
    uint64_t expert_hits;
    uint64_t expert_loads;      /* dequantizations performed */
    uint64_t aborted_steps;
    double   last_decode_seconds;
} mimo26_worker_stats;

/*
 * Build a worker over a checkpoint. Every layer's non-expert weights and the
 * static text tensors are dequantized and held resident; experts are loaded
 * on demand into a bounded per-layer cache.
 *
 * The footprint is computed and checked against memory_limit_bytes before any
 * large allocation, so an over-large configuration is refused rather than
 * discovered by the OOM killer.
 */
mimo26_worker_status mimo26_worker_create(mimo26_worker **worker,
                                          const char *root,
                                          const mimo26_worker_config *config,
                                          char *error, size_t error_size);
void mimo26_worker_destroy(mimo26_worker *worker);

/* Planned resident bytes, available before and after creation. */
uint64_t mimo26_worker_planned_bytes(const mimo26_worker_config *config);
uint64_t mimo26_worker_resident_bytes(const mimo26_worker *worker);

/*
 * Run one token through all 48 layers.
 *
 * logits is [152576] F32 and receives the accumulation before any BF16
 * rounding. Ids at or above MIMO26_TOKENIZER_VOCAB decode to no token and are
 * set to -infinity, so a caller cannot sample an id the detokenizer cannot
 * map -- the embedding matrix is padded to a multiple of 128 and carries 901
 * such rows.
 *
 * The step is transactional: keys and values for all 48 layers are staged and
 * committed together. Any failure aborts, leaving committed history and the
 * position counter untouched, so the caller may retry or cancel.
 */
mimo26_worker_status mimo26_worker_decode(mimo26_worker *worker,
                                          uint32_t token_id, float *logits,
                                          char *error, size_t error_size);

/* Argmax over the maskable vocabulary, ignoring the padded tail. */
uint32_t mimo26_worker_argmax(const float *logits);

/*
 * Observe the residual stream after each layer, for localizing a divergence
 * against a reference trace. Called with the hidden state as it stands after
 * layer `layer` completes.
 */
typedef void (*mimo26_worker_trace)(void *context, uint32_t layer,
                                    const uint16_t *hidden, size_t count);
void mimo26_worker_set_trace(mimo26_worker *worker, mimo26_worker_trace trace,
                             void *context);

/* Positions committed so far. */
uint64_t mimo26_worker_position(const mimo26_worker *worker);

/* Drop all history; weights and expert residency are retained. */
void mimo26_worker_reset(mimo26_worker *worker);

/* Rewind committed steps, for a rejected speculative block. */
mimo26_worker_status mimo26_worker_rollback(mimo26_worker *worker,
                                            size_t count);

void mimo26_worker_get_stats(const mimo26_worker *worker,
                             mimo26_worker_stats *stats);

/* Tokens the tokenizer actually defines; rows beyond this are padding. */
#define MIMO26_TOKENIZER_VOCAB 151675u

#ifdef __cplusplus
}
#endif

#endif
