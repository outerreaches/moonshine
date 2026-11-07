#include "mimo26_worker.h"

#include "k3_expert_cache.h"
#include "mimo26_manifest.h"
#include "mimo26_ops.h"
#include "mimo26_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIMO26_MOE_LAYER_FIRST 1u
#define MIMO26_EXPERT_BF16_BYTES \
    ((size_t)(2u * 2048u * 4096u + 4096u * 2048u) * sizeof(uint16_t))

struct mimo26_worker {
    mimo26_manifest        manifest;
    k3_st_model            model;
    mimo26_static_weights  statics;
    mimo26_layer_weights   layers[MIMO26_TEXT_LAYER_COUNT];
    mimo26_layer_scratch  *scratch;
    mimo26_kv_cache       *kv;

    k3_expert_cache       *expert_cache;
    mimo26_expert_weights *slots;      /* [moe_layers * slots_per_layer] */
    uint32_t              *slot_expert; /* expert id resident in each slot */
    bool                  *slot_filled;
    uint16_t               slots_per_layer;

    /* Resolved for the batch currently being served. */
    uint32_t               batch_layer;
    uint32_t               batch_slot[MIMO26_ROUTER_TOP_K];
    uint32_t               batch_expert[MIMO26_ROUTER_TOP_K];
    size_t                 batch_count;

    uint16_t              *hidden;
    uint16_t              *normed;
    mimo26_worker_stats    stats;
    uint64_t               resident_bytes;
    bool                   decoding;
    mimo26_worker_trace    trace;
    void                  *trace_context;
};

static mimo26_worker_status fail(char *error, size_t size,
                                 mimo26_worker_status status,
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

static mimo26_layer_status layer_fail(char *error, size_t size,
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

void mimo26_worker_config_defaults(mimo26_worker_config *config)
{
    if (config == NULL) {
        return;
    }
    memset(config, 0, sizeof *config);
    config->global_kv_capacity = 2048u;
    config->rollback_depth = 8u;
    config->expert_slots_per_layer = 8u;
    config->memory_limit_bytes = 0u;
}

uint64_t mimo26_worker_planned_bytes(const mimo26_worker_config *config)
{
    if (config == NULL) {
        return 0u;
    }
    /* Dequantized static text: every layer's non-expert weights plus the
     * embedding and head. Measured at 11.15 GiB for this checkpoint. */
    const uint64_t statics = UINT64_C(11973)  * UINT64_C(1048576);
    const uint64_t experts = (uint64_t)MIMO26_MOE_LAYER_COUNT *
                             (uint64_t)config->expert_slots_per_layer *
                             (uint64_t)MIMO26_EXPERT_BF16_BYTES;
    /* Global KV: 9 layers x 4 heads x (192+128) x 2 bytes per position. */
    const uint64_t kv = (uint64_t)config->global_kv_capacity * 23040u +
                        UINT64_C(25559040);
    const uint64_t logits = (uint64_t)MIMO26_VOCAB_SIZE * sizeof(float);
    return statics + experts + kv + logits;
}

uint64_t mimo26_worker_resident_bytes(const mimo26_worker *worker)
{
    return worker != NULL ? worker->resident_bytes : 0u;
}

uint64_t mimo26_worker_position(const mimo26_worker *worker)
{
    return worker != NULL ? mimo26_kv_length(worker->kv) : 0u;
}

void mimo26_worker_get_stats(const mimo26_worker *worker,
                             mimo26_worker_stats *stats)
{
    if (worker == NULL || stats == NULL) {
        return;
    }
    *stats = worker->stats;
    if (worker->expert_cache != NULL) {
        k3_expert_cache_stats cache;
        k3_expert_cache_get_stats(worker->expert_cache, &cache);
        stats->expert_accesses = cache.accesses;
        stats->expert_hits = cache.hits;
    }
}

void mimo26_worker_destroy(mimo26_worker *worker)
{
    if (worker == NULL) {
        return;
    }
    if (worker->slots != NULL) {
        const size_t total = (size_t)MIMO26_MOE_LAYER_COUNT *
                             worker->slots_per_layer;
        for (size_t i = 0; i < total; i++) {
            mimo26_expert_weights_free(&worker->slots[i]);
        }
        free(worker->slots);
    }
    free(worker->slot_expert);
    free(worker->slot_filled);
    k3_expert_cache_destroy(worker->expert_cache);
    mimo26_kv_destroy(worker->kv);
    mimo26_layer_scratch_destroy(worker->scratch);
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        mimo26_layer_weights_free(&worker->layers[i]);
    }
    mimo26_static_weights_free(&worker->statics);
    k3_st_model_close(&worker->model);
    mimo26_manifest_free(&worker->manifest);
    free(worker->hidden);
    free(worker->normed);
    free(worker);
}

static mimo26_layer_status prepare_batch(void *context, uint32_t layer,
                                         const uint32_t *experts, size_t count,
                                         char *error, size_t error_size)
{
    mimo26_worker *worker = context;
    if (count == 0u || count > MIMO26_ROUTER_TOP_K) {
        return MIMO26_LAYER_INVALID_ARGUMENT;
    }
    uint16_t ids[MIMO26_ROUTER_TOP_K] = {0};
    k3_expert_cache_access accesses[MIMO26_ROUTER_TOP_K];
    for (size_t i = 0; i < count; i++) {
        ids[i] = (uint16_t)experts[i];
    }
    const uint16_t cache_layer = (uint16_t)(layer - MIMO26_MOE_LAYER_FIRST);
    if (!k3_expert_cache_plan(worker->expert_cache, cache_layer, ids,
                              (uint16_t)count, accesses, error, error_size)) {
        return MIMO26_LAYER_EXPERT_UNAVAILABLE;
    }

    /*
     * Load every admitted miss into its destination slot. A real GPU path
     * would issue these as copies overlapping the hits already being read;
     * here the load is synchronous, but the plan/commit contract is the same
     * and a failure still aborts before anything is published.
     */
    for (size_t i = 0; i < count; i++) {
        worker->batch_expert[i] = experts[i];
        if (accesses[i].hit) {
            worker->batch_slot[i] = accesses[i].source_slot;
            continue;
        }
        if (!accesses[i].admit ||
            accesses[i].destination_slot == K3_EXPERT_CACHE_NO_SLOT) {
            k3_expert_cache_abort(worker->expert_cache, cache_layer);
            return layer_fail(error, error_size,
                              MIMO26_LAYER_EXPERT_UNAVAILABLE,
                              "expert %u was neither hit nor admitted",
                              experts[i]);
        }
        const uint32_t slot = accesses[i].destination_slot;
        mimo26_expert_weights *storage = &worker->slots[slot];
        mimo26_expert_weights fresh;
        memset(&fresh, 0, sizeof fresh);
        if (mimo26_expert_weights_load(&fresh, &worker->model, layer,
                                       experts[i], error, error_size) !=
            MIMO26_WEIGHTS_OK) {
            /* Nothing was published, so the cache must forget the plan. */
            k3_expert_cache_abort(worker->expert_cache, cache_layer);
            return MIMO26_LAYER_EXPERT_UNAVAILABLE;
        }
        mimo26_expert_weights_free(storage);
        *storage = fresh;
        worker->slot_expert[slot] = experts[i];
        worker->slot_filled[slot] = true;
        worker->batch_slot[i] = slot;
        worker->stats.expert_loads++;
    }
    if (!k3_expert_cache_commit(worker->expert_cache, cache_layer, error,
                                error_size)) {
        return MIMO26_LAYER_EXPERT_UNAVAILABLE;
    }
    worker->batch_layer = layer;
    worker->batch_count = count;
    return MIMO26_LAYER_OK;
}

static mimo26_layer_status provide_expert(void *context, uint32_t layer,
                                          uint32_t expert,
                                          const mimo26_expert_weights **out,
                                          char *error, size_t error_size)
{
    mimo26_worker *worker = context;
    if (worker->batch_layer != layer) {
        return layer_fail(error, error_size, MIMO26_LAYER_EXPERT_UNAVAILABLE,
                          "layer %u was not prepared", layer);
    }
    for (size_t i = 0; i < worker->batch_count; i++) {
        if (worker->batch_expert[i] != expert) {
            continue;
        }
        const uint32_t slot = worker->batch_slot[i];
        if (!worker->slot_filled[slot] ||
            worker->slot_expert[slot] != expert) {
            return layer_fail(error, error_size,
                              MIMO26_LAYER_EXPERT_UNAVAILABLE,
                              "slot %u no longer holds expert %u", slot,
                              expert);
        }
        *out = &worker->slots[slot];
        return MIMO26_LAYER_OK;
    }
    return layer_fail(error, error_size, MIMO26_LAYER_EXPERT_UNAVAILABLE,
                      "expert %u was not in the prepared batch", expert);
}

mimo26_worker_status mimo26_worker_create(mimo26_worker **out, const char *root,
                                          const mimo26_worker_config *config,
                                          char *error, size_t error_size)
{
    if (out == NULL || root == NULL || config == NULL ||
        config->global_kv_capacity == 0u ||
        config->expert_slots_per_layer == 0u) {
        return fail(error, error_size, MIMO26_WORKER_INVALID_ARGUMENT,
                    "invalid worker configuration");
    }
    /*
     * One token routes to MIMO26_ROUTER_TOP_K distinct experts per MoE layer
     * and needs them resident together, so a cache smaller than that cannot
     * serve even a single step: the plan phase admits k entries into fewer
     * slots and the commit phase finds one missing. Refuse here rather than
     * let it surface as a failed decode partway through the first token.
     */
    if (config->expert_slots_per_layer < MIMO26_ROUTER_TOP_K) {
        return fail(error, error_size, MIMO26_WORKER_INVALID_ARGUMENT,
                    "%u expert slots per layer cannot hold the %u experts one "
                    "token routes to", (unsigned)config->expert_slots_per_layer,
                    (unsigned)MIMO26_ROUTER_TOP_K);
    }
    *out = NULL;

    const uint64_t planned = mimo26_worker_planned_bytes(config);
    if (config->memory_limit_bytes != 0u &&
        planned > config->memory_limit_bytes) {
        return fail(error, error_size, MIMO26_WORKER_MEMORY_LIMIT,
                    "plan needs %.2f GiB but the limit is %.2f GiB",
                    (double)planned / (double)(1u << 30),
                    (double)config->memory_limit_bytes / (double)(1u << 30));
    }

    mimo26_worker *worker = calloc(1u, sizeof *worker);
    if (worker == NULL) {
        return fail(error, error_size, MIMO26_WORKER_OUT_OF_MEMORY,
                    "out of memory allocating worker");
    }
    worker->slots_per_layer = config->expert_slots_per_layer;
    mimo26_worker_status status = MIMO26_WORKER_OK;

    if (!mimo26_manifest_load(&worker->manifest, root, error, error_size) ||
        !mimo26_manifest_open_model(&worker->manifest, root, &worker->model,
                                    error, error_size)) {
        status = MIMO26_WORKER_LOAD_FAILED;
        goto failed;
    }
    if (mimo26_static_weights_load(&worker->statics, &worker->model, error,
                                   error_size) != MIMO26_WEIGHTS_OK) {
        status = MIMO26_WORKER_LOAD_FAILED;
        goto failed;
    }
    worker->resident_bytes += worker->statics.bytes;
    for (uint32_t layer = 0; layer < MIMO26_TEXT_LAYER_COUNT; layer++) {
        if (mimo26_layer_weights_load(&worker->layers[layer], &worker->model,
                                      layer, error, error_size) !=
            MIMO26_WEIGHTS_OK) {
            status = MIMO26_WORKER_LOAD_FAILED;
            goto failed;
        }
        worker->resident_bytes += worker->layers[layer].bytes;
    }

    if (mimo26_layer_scratch_create(&worker->scratch) != MIMO26_LAYER_OK) {
        status = MIMO26_WORKER_OUT_OF_MEMORY;
        goto failed;
    }
    if (mimo26_kv_create(&worker->kv, config->global_kv_capacity,
                         config->rollback_depth) != MIMO26_KV_OK) {
        status = MIMO26_WORKER_OUT_OF_MEMORY;
        goto failed;
    }
    worker->resident_bytes += mimo26_kv_allocated_bytes(worker->kv);

    if (!k3_expert_cache_create(&worker->expert_cache,
                                (uint16_t)MIMO26_MOE_LAYER_COUNT,
                                worker->slots_per_layer, error, error_size)) {
        status = MIMO26_WORKER_OUT_OF_MEMORY;
        goto failed;
    }
    const size_t total_slots = (size_t)MIMO26_MOE_LAYER_COUNT *
                               worker->slots_per_layer;
    worker->slots = calloc(total_slots, sizeof *worker->slots);
    worker->slot_expert = calloc(total_slots, sizeof *worker->slot_expert);
    worker->slot_filled = calloc(total_slots, sizeof *worker->slot_filled);
    worker->hidden = calloc(MIMO26_HIDDEN_SIZE, sizeof *worker->hidden);
    worker->normed = calloc(MIMO26_HIDDEN_SIZE, sizeof *worker->normed);
    if (worker->slots == NULL || worker->slot_expert == NULL ||
        worker->slot_filled == NULL || worker->hidden == NULL ||
        worker->normed == NULL) {
        status = MIMO26_WORKER_OUT_OF_MEMORY;
        goto failed;
    }
    worker->batch_layer = UINT32_MAX;

    *out = worker;
    return MIMO26_WORKER_OK;

failed:
    mimo26_worker_destroy(worker);
    return status;
}

void mimo26_worker_set_trace(mimo26_worker *worker, mimo26_worker_trace trace,
                             void *context)
{
    if (worker == NULL) {
        return;
    }
    worker->trace = trace;
    worker->trace_context = context;
}

void mimo26_worker_reset(mimo26_worker *worker)
{
    if (worker == NULL) {
        return;
    }
    mimo26_kv_reset(worker->kv);
    worker->batch_layer = UINT32_MAX;
    worker->batch_count = 0u;
    worker->decoding = false;
}

mimo26_worker_status mimo26_worker_rollback(mimo26_worker *worker, size_t count)
{
    if (worker == NULL) {
        return MIMO26_WORKER_INVALID_ARGUMENT;
    }
    if (worker->decoding) {
        return MIMO26_WORKER_BUSY;
    }
    return mimo26_kv_rollback(worker->kv, count) == MIMO26_KV_OK
               ? MIMO26_WORKER_OK
               : MIMO26_WORKER_INVALID_ARGUMENT;
}

uint32_t mimo26_worker_argmax(const float *logits)
{
    if (logits == NULL) {
        return 0u;
    }
    uint32_t best = 0u;
    float best_value = -INFINITY;
    for (uint32_t i = 0; i < MIMO26_TOKENIZER_VOCAB; i++) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    return best;
}

mimo26_worker_status mimo26_worker_decode(mimo26_worker *worker,
                                          uint32_t token_id, float *logits,
                                          char *error, size_t error_size)
{
    if (worker == NULL || logits == NULL) {
        return fail(error, error_size, MIMO26_WORKER_INVALID_ARGUMENT,
                    "invalid decode arguments");
    }
    if (worker->decoding) {
        return fail(error, error_size, MIMO26_WORKER_BUSY,
                    "a decode step is already in progress");
    }
    if (token_id >= MIMO26_TOKENIZER_VOCAB) {
        return fail(error, error_size, MIMO26_WORKER_INVALID_ARGUMENT,
                    "token id %u is beyond the tokenizer vocabulary", token_id);
    }
    const uint64_t position = mimo26_kv_length(worker->kv);

    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);
    worker->decoding = true;

    if (mimo26_kv_begin(worker->kv, position) != MIMO26_KV_OK) {
        worker->decoding = false;
        return fail(error, error_size, MIMO26_WORKER_DECODE_FAILED,
                    "could not open a transaction at position %llu",
                    (unsigned long long)position);
    }

    memcpy(worker->hidden,
           worker->statics.embed_tokens + (size_t)token_id * MIMO26_HIDDEN_SIZE,
           MIMO26_HIDDEN_SIZE * sizeof *worker->hidden);

    for (uint32_t layer = 0; layer < MIMO26_TEXT_LAYER_COUNT; layer++) {
        mimo26_layer context;
        memset(&context, 0, sizeof context);
        context.weights = &worker->layers[layer];
        context.prepare = prepare_batch;
        context.provider = provide_expert;
        context.provider_context = worker;
        if (mimo26_layer_decode(&context, worker->scratch, worker->hidden,
                                worker->kv, position, NULL, error,
                                error_size) != MIMO26_LAYER_OK) {
            /* Abort leaves committed history and the position untouched. */
            mimo26_kv_abort(worker->kv);
            worker->stats.aborted_steps++;
            worker->decoding = false;
            return MIMO26_WORKER_DECODE_FAILED;
        }
        if (worker->trace != NULL) {
            worker->trace(worker->trace_context, layer, worker->hidden,
                          MIMO26_HIDDEN_SIZE);
        }
    }

    if (mimo26_kv_commit(worker->kv) != MIMO26_KV_OK) {
        mimo26_kv_abort(worker->kv);
        worker->stats.aborted_steps++;
        worker->decoding = false;
        return fail(error, error_size, MIMO26_WORKER_DECODE_FAILED,
                    "commit failed at position %llu",
                    (unsigned long long)position);
    }

    if (mimo26_rmsnorm_bf16(worker->normed, worker->hidden,
                            worker->statics.norm, MIMO26_HIDDEN_SIZE,
                            MIMO26_LAYERNORM_EPSILON) != MIMO26_OPS_OK) {
        worker->decoding = false;
        return fail(error, error_size, MIMO26_WORKER_DECODE_FAILED,
                    "final norm failed");
    }
    /* Logits in F32: the accumulation before any BF16 rounding. */
    for (uint32_t row = 0; row < MIMO26_VOCAB_SIZE; row++) {
        const uint16_t *weights =
            worker->statics.lm_head + (size_t)row * MIMO26_HIDDEN_SIZE;
        float sum = 0.0f;
        for (size_t i = 0; i < MIMO26_HIDDEN_SIZE; i++) {
            sum += mimo26_bf16_to_f32(weights[i]) *
                   mimo26_bf16_to_f32(worker->normed[i]);
        }
        logits[row] = sum;
    }
    /* The padded tail decodes to no token, so it must never be sampled. */
    for (uint32_t row = MIMO26_TOKENIZER_VOCAB; row < MIMO26_VOCAB_SIZE;
         row++) {
        logits[row] = -INFINITY;
    }

    struct timespec finished;
    clock_gettime(CLOCK_MONOTONIC, &finished);
    worker->stats.last_decode_seconds =
        (double)(finished.tv_sec - started.tv_sec) +
        (double)(finished.tv_nsec - started.tv_nsec) / 1e9;
    worker->stats.tokens++;
    worker->decoding = false;
    return MIMO26_WORKER_OK;
}
