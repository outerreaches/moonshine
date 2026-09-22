#include "mimo26_gpu_worker.h"

#include "k3_expert_cache.h"
#include "k3_io_uring.h"
#include "k3_rocm_ops.h"
#include "k3_safetensors.h"
#include "mimo26_architecture.h"
#include "mimo26_attention.h"
#include "mimo26_kv.h"
#include "mimo26_manifest.h"
#include "mimo26_ops.h"
#include "mimo26_rocm_layer.h"
#include "mimo26_rocm_ops.h"
#include "mimo26_router.h"
#include "mimo26_weights.h"

#include <hip/hip_runtime.h>

#include <sys/uio.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HIDDEN MIMO26_HIDDEN_SIZE
#define QK MIMO26_QK_HEAD_DIM
#define VD MIMO26_V_HEAD_DIM
#define VOCAB MIMO26_VOCAB_SIZE
#define LAYERS MIMO26_TEXT_LAYER_COUNT
#define PACKED_EXPERT_BYTES 13369344u   /* 12.75 MiB: 3 projections + scales */

/*
 * Every expert has identical shapes, so a slot's storage can be allocated
 * once at startup and reused for whatever identity occupies it. The first
 * version allocated and freed on every admission -- six hipMalloc and six
 * hipFree per expert, about 1,440 allocator calls per token at a realistic
 * miss rate.
 *
 * The six tensors are also CONTIGUOUS on disk, in one shard, in this order,
 * with no padding between them -- verified across 40 experts spanning five
 * layers, and re-verified for every expert at admission. So the slot is one
 * device block laid out exactly as the file is, admission is a single
 * 12.75 MiB read and a single copy, and the six views are just offsets into
 * it. That replaces six O_DIRECT reads and six copies per expert, which
 * matters because profiling puts the read at 82% of admission and admission
 * at ~70% of a token.
 *
 * Note the order is down, gate, up -- alphabetical, not the order the MLP
 * evaluates them in. Assuming the logical order here would transpose
 * projections and produce a model that still runs.
 */
typedef struct {
    const char *projection;
    const char *suffix;
    uint64_t    offset;
    uint64_t    bytes;
} expert_block_entry;

static const expert_block_entry EXPERT_BLOCK[6] = {
    {"down_proj", "weight",              0u, 4194304u},
    {"down_proj", "weight_scale",  4194304u,  262144u},
    {"gate_proj", "weight",        4456448u, 4194304u},
    {"gate_proj", "weight_scale",  8650752u,  262144u},
    {"up_proj",   "weight",        8912896u, 4194304u},
    {"up_proj",   "weight_scale", 13107200u,  262144u},
};

/*
 * Storage only. Residency, LRU and the two-phase plan/commit belong to
 * k3_expert_cache, which the CPU worker already uses.
 *
 * This file used to carry its own copy of that policy and got it wrong: the
 * victim search tested "empty" before "pinned", so a slot claimed earlier in
 * the same batch was handed back as a victim and the first expert silently
 * never landed. k3_expert_cache exists precisely to prevent "an admission
 * from overwriting a hit still consumed by ROCm", is tested, and is shared
 * with the CPU worker -- so both backends now make the same residency
 * decisions and their hit rates are comparable rather than coincidental.
 */
typedef struct {
    mimo26_rocm_expert view;
    void              *block;   /* one PACKED_EXPERT_BYTES allocation */
} expert_slot;

/*
 * Expert reads go through K3's io_uring rather than a synchronous read per
 * tensor, because the profile said to: the read was 82% of admission and
 * admission ~70% of a token, and the reads were being issued one at a time
 * against an NVMe that wants a queue.
 *
 * Staging is hipHostMalloc'd mapped, the same thing K3's engine does. Two
 * things follow. The io_uring registers GPU-visible buffers directly, and
 * the copy out of them is a pinned host-to-device transfer rather than a
 * pageable one.
 *
 * The pool is small and separate from the cache. Registering every cache
 * slot would be the zero-copy version, but 47 layers x tens of slots is
 * thousands of buffers, past what a ring will register, and a registered
 * buffer may carry only one outstanding request.
 */
#define MIMO26_STAGING_SLOTS 8u
/* 12.75 MiB plus a page, since a span's start is not 4096-aligned. */
#define MIMO26_STAGING_BYTES (PACKED_EXPERT_BYTES + 4096u)

typedef struct {
    uint32_t     slot;        /* global cache slot this read lands in */
    uint64_t     aligned_start;
    uint32_t     aligned_bytes;
    uint32_t     offset_in_buffer;
    int          fd;
    uint32_t     expert;
} pending_read;

/* Accumulated inside prepare_batch, read out per token by the profiler. */
static double g_upload_seconds = 0.0;
static double g_read_seconds = 0.0;   /* disk/page-cache portion */
static double g_copy_seconds = 0.0;   /* host-to-device portion */

struct mimo26_gpu_worker {
    mimo26_gpu_worker_config config;
    mimo26_manifest          manifest;
    k3_st_model              model;
    bool                     model_open;

    /* Device-resident weights. */
    void *embed_tokens;
    void *final_norm;
    void *lm_head;
    mimo26_rocm_layer_weights layers[LAYERS];

    k3_expert_cache *cache;            /* residency policy, shared with CPU */
    expert_slot     *slots;            /* slot_count blocks, global indices */
    uint32_t         slot_count;

    mimo26_rocm_layer_scratch scratch;
    void            *hidden;
    void            *normed;
    float           *device_logits;
    mimo26_kv_cache *kv;               /* host-side history, mirrored up */
    void            *device_keys;
    void            *device_values;
    void            *cos_table;
    void            *sin_table;
    uint16_t        *staging_key;
    uint16_t        *staging_value;
    /* Per-chunk key/value readback, [layer][chunk][...] so the whole chunk
     * can be staged into the journal after every layer has run. */
    uint16_t        *chunk_keys;
    uint16_t        *chunk_values;
    uint16_t        *cos_tables;
    uint16_t        *sin_tables;
    void            *device_cos_tables;
    void            *device_sin_tables;
    uint16_t         prefill_chunk;

    /* Where this layer's selection landed, filled by prepare_batch and read
     * by the provider moments later in the same step. */
    uint32_t resolved[MIMO26_ROUTER_TOP_K];
    uint32_t resolved_expert[MIMO26_ROUTER_TOP_K];
    size_t   resolved_count;

    /* io_uring staging pool. */
    k3_io_uring *ring;
    void        *staging_host[MIMO26_STAGING_SLOTS];
    struct iovec staging_iov[MIMO26_STAGING_SLOTS];
    bool         staging_ready;

    uint64_t resident_bytes;
    uint64_t position;
    mimo26_gpu_worker_stats stats;
    char     last_error[512];
};

static mimo26_gpu_worker_status fail(char *error, size_t size,
                                     mimo26_gpu_worker_status status,
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

void mimo26_gpu_worker_config_defaults(mimo26_gpu_worker_config *config)
{
    if (config == NULL) {
        return;
    }
    config->global_kv_capacity = 2048u;
    /*
     * 16 packed slots per routed layer is 9.4 GiB, which leaves room beside
     * the 8.28 GiB of static text on a 124 GiB part without assuming the
     * whole device is free. Raise it once the hit rate is measured on a real
     * workload; the plan asks for exactly that order.
     */
    config->expert_slots_per_layer = 16u;
    config->memory_limit_bytes = 0u;
    /* 32 tokens per chunk: enough to amortize the 10.72 GB of BF16
     * projections roughly 32-fold while keeping the batch scratch modest. */
    config->prefill_chunk = 32u;
}

uint64_t mimo26_gpu_worker_planned_bytes(
    const mimo26_gpu_worker_config *config)
{
    if (config == NULL) {
        return 0u;
    }
    /* Static text: embedding and lm_head are [152576][4096] BF16 each, plus
     * the final norm. Per layer, the non-expert weights. */
    uint64_t total = 2ull * VOCAB * HIDDEN * sizeof(uint16_t);
    total += HIDDEN * sizeof(uint16_t);
    /* Nine global layers carry the wider o_proj plus a 13568-row QKV; the
     * 39 windowed ones carry 14848 rows and a sink. Layer 0 also carries the
     * dense MLP at 3 x 16384 x 4096. */
    total += 9ull * (13568ull + 4096ull * 2ull) * HIDDEN * sizeof(uint16_t);
    total += 39ull * (14848ull + 4096ull * 2ull) * HIDDEN * sizeof(uint16_t);
    total += 3ull * 16384ull * HIDDEN * sizeof(uint16_t);
    total += 47ull * (MIMO26_ROUTER_EXPERTS * HIDDEN * sizeof(uint16_t) +
                      MIMO26_ROUTER_EXPERTS * sizeof(float));
    total += (uint64_t)MIMO26_MOE_LAYER_COUNT *
             config->expert_slots_per_layer * PACKED_EXPERT_BYTES;
    /* KV: nine global layers hold the full capacity, the rest a 128 ring. */
    total += 9ull * config->global_kv_capacity *
             (MIMO26_GLOBAL_KV_HEADS * (QK + VD)) * sizeof(uint16_t);
    total += 39ull * MIMO26_SLIDING_WINDOW *
             (MIMO26_SWA_KV_HEADS * (QK + VD)) * sizeof(uint16_t);
    total += (uint64_t)VOCAB * sizeof(float);
    return total;
}

uint64_t mimo26_gpu_worker_resident_bytes(const mimo26_gpu_worker *worker)
{
    return worker == NULL ? 0u : worker->resident_bytes;
}

/* ---- uploads ---- */

static void *upload(mimo26_gpu_worker *worker, const void *host, size_t bytes)
{
    void *device = NULL;
    if (hipMalloc(&device, bytes) != hipSuccess) {
        return NULL;
    }
    if (hipMemcpy(device, host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(device);
        return NULL;
    }
    worker->resident_bytes += bytes;
    return device;
}

static void *upload_tensor(mimo26_gpu_worker *worker, const char *name,
                           uint64_t *bytes_out)
{
    const k3_st_tensor *tensor = k3_st_find(&worker->model, name);
    if (tensor == NULL) {
        return NULL;
    }
    char error[256];
    k3_st_read read;
    memset(&read, 0, sizeof read);
    if (!k3_st_read_span(&worker->model, tensor->shard,
                         tensor->physical_offset, tensor->byte_length, 4096u,
                         &read, error, sizeof error)) {
        return NULL;
    }
    void *device = upload(worker, read.data, tensor->byte_length);
    k3_st_read_release(&read);
    if (device != NULL && bytes_out != NULL) {
        *bytes_out += tensor->byte_length;
    }
    return device;
}

/* ---- bounded packed-expert cache ---- */

/*
 * Resolve an expert's physical span and require the layout the block
 * assumes: one shard, expected sizes, contiguous, in the order recorded in
 * EXPERT_BLOCK. Checked per expert rather than trusted from a sample -- the
 * same discipline the fused-QKV layout had to be taught. A mismatch is a
 * hard failure, never a silent fallback to a different reading.
 */
static bool resolve_expert_span(mimo26_gpu_worker *worker, uint32_t layer,
                                uint32_t expert, uint16_t *shard_out,
                                uint64_t *start_out)
{
    uint64_t base_offset = 0u;
    uint16_t shard = 0u;
    for (size_t i = 0; i < 6u; i++) {
        char name[320];
        snprintf(name, sizeof name, "model.layers.%u.mlp.experts.%u.%s.%s",
                 layer, expert, EXPERT_BLOCK[i].projection,
                 EXPERT_BLOCK[i].suffix);
        const k3_st_tensor *tensor = k3_st_find(&worker->model, name);
        if (tensor == NULL || tensor->byte_length != EXPERT_BLOCK[i].bytes) {
            return false;
        }
        if (i == 0u) {
            base_offset = tensor->physical_offset;
            shard = tensor->shard;
        } else if (tensor->shard != shard ||
                   tensor->physical_offset !=
                       base_offset + EXPERT_BLOCK[i].offset) {
            return false;
        }
    }
    *shard_out = shard;
    *start_out = base_offset;
    return true;
}

static void slot_views(expert_slot *slot)
{
    uint8_t *base = (uint8_t *)slot->block;
    slot->view.down_packed = base + EXPERT_BLOCK[0].offset;
    slot->view.down_scales = base + EXPERT_BLOCK[1].offset;
    slot->view.gate_packed = base + EXPERT_BLOCK[2].offset;
    slot->view.gate_scales = base + EXPERT_BLOCK[3].offset;
    slot->view.up_packed = base + EXPERT_BLOCK[4].offset;
    slot->view.up_scales = base + EXPERT_BLOCK[5].offset;
}

/*
 * Plan the layer's whole batch, read every miss, then commit.
 *
 * The ordering is the contract k3_expert_cache documents: a planned hit may
 * be read from its source slot until commit, and an admission may only be
 * written to its destination slot, so admitting the eighth expert can never
 * overwrite the first that the same token is about to consume.
 */
static bool prepare_batch(void *context, uint32_t layer,
                          const uint32_t *experts, size_t count)
{
    mimo26_gpu_worker *worker = (mimo26_gpu_worker *)context;
    uint16_t ids[MIMO26_ROUTER_TOP_K];
    k3_expert_cache_access accesses[MIMO26_ROUTER_TOP_K];
    if (count > MIMO26_ROUTER_TOP_K) {
        return false;
    }
    for (size_t k = 0; k < count; k++) {
        ids[k] = (uint16_t)experts[k];
    }

    char error[256];
    if (!k3_expert_cache_plan(worker->cache, (uint16_t)layer, ids,
                              (uint16_t)count, accesses, error,
                              sizeof error)) {
        fprintf(stderr, "mimo26 layer %u: expert plan failed: %s\n", layer,
                error);
        return false;
    }

    pending_read pending[MIMO26_ROUTER_TOP_K];
    size_t pending_count = 0u;
    for (size_t k = 0; k < count; k++) {
        if (!accesses[k].admit) {
            continue;
        }
        uint16_t shard = 0u;
        uint64_t start = 0u;
        if (!resolve_expert_span(worker, layer, experts[k], &shard, &start)) {
            fprintf(stderr, "mimo26 layer %u: expert %u has an unexpected "
                    "on-disk layout\n", layer, experts[k]);
            k3_expert_cache_abort(worker->cache, (uint16_t)layer);
            return false;
        }
        const int fd = worker->model.shards[shard].direct_fd >= 0
                           ? worker->model.shards[shard].direct_fd
                           : worker->model.shards[shard].fd;
        const uint64_t aligned_start = start & ~UINT64_C(4095);
        const uint64_t aligned_end =
            (start + PACKED_EXPERT_BYTES + UINT64_C(4095)) &
            ~UINT64_C(4095);
        if (fd < 0 ||
            aligned_end - aligned_start > MIMO26_STAGING_BYTES ||
            accesses[k].destination_slot >= worker->slot_count ||
            worker->slots[accesses[k].destination_slot].block == NULL) {
            fprintf(stderr, "mimo26 layer %u: expert %u is not readable\n",
                    layer, experts[k]);
            k3_expert_cache_abort(worker->cache, (uint16_t)layer);
            return false;
        }
        pending[pending_count].slot = accesses[k].destination_slot;
        pending[pending_count].aligned_start = aligned_start;
        pending[pending_count].aligned_bytes =
            (uint32_t)(aligned_end - aligned_start);
        pending[pending_count].offset_in_buffer =
            (uint32_t)(start - aligned_start);
        pending[pending_count].fd = fd;
        pending[pending_count].expert = experts[k];
        pending_count++;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t done = 0u;
    while (done < pending_count) {
        const size_t batch =
            (pending_count - done) < MIMO26_STAGING_SLOTS
                ? (pending_count - done) : MIMO26_STAGING_SLOTS;
        k3_io_request requests[MIMO26_STAGING_SLOTS];
        for (size_t i = 0; i < batch; i++) {
            requests[i].fd = pending[done + i].fd;
            requests[i].offset = pending[done + i].aligned_start;
            requests[i].bytes = pending[done + i].aligned_bytes;
            requests[i].buffer_index = (uint16_t)i;
            requests[i].user_data = (uint64_t)i;
        }
        if (!k3_io_uring_submit(worker->ring, requests, (uint16_t)batch,
                                error, sizeof error)) {
            fprintf(stderr, "mimo26 layer %u: io_uring submit failed: %s\n",
                    layer, error);
            k3_expert_cache_abort(worker->cache, (uint16_t)layer);
            return false;
        }
        size_t completed = 0u;
        while (completed < batch) {
            k3_io_completion completions[MIMO26_STAGING_SLOTS];
            uint16_t got = 0u;
            if (!k3_io_uring_wait(worker->ring, completions,
                                  MIMO26_STAGING_SLOTS, &got, error,
                                  sizeof error)) {
                fprintf(stderr, "mimo26 layer %u: io_uring wait failed: %s\n",
                        layer, error);
                k3_expert_cache_abort(worker->cache, (uint16_t)layer);
                return false;
            }
            for (uint16_t c = 0; c < got; c++) {
                const size_t index = (size_t)completions[c].user_data;
                pending_read *entry = &pending[done + index];
                if (completions[c].result < 0 ||
                    (uint32_t)completions[c].result <
                        entry->offset_in_buffer + PACKED_EXPERT_BYTES) {
                    fprintf(stderr, "mimo26 layer %u: short read for expert "
                            "%u (%d bytes)\n", layer, entry->expert,
                            (int)completions[c].result);
                    k3_expert_cache_abort(worker->cache, (uint16_t)layer);
                    return false;
                }
                const uint8_t *source =
                    (const uint8_t *)worker->staging_host[index] +
                    entry->offset_in_buffer;
                if (hipMemcpy(worker->slots[entry->slot].block, source,
                              PACKED_EXPERT_BYTES,
                              hipMemcpyHostToDevice) != hipSuccess) {
                    fprintf(stderr, "mimo26 layer %u: staging copy failed "
                            "for expert %u\n", layer, entry->expert);
                    k3_expert_cache_abort(worker->cache, (uint16_t)layer);
                    return false;
                }
                worker->stats.expert_uploads++;
                completed++;
            }
        }
        done += batch;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double elapsed = (double)(t1.tv_sec - t0.tv_sec) +
                           (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    g_read_seconds += elapsed;
    g_upload_seconds += elapsed;

    /* Every admission has landed, so hits and admissions can be published
     * together. */
    if (!k3_expert_cache_commit(worker->cache, (uint16_t)layer, error,
                                sizeof error)) {
        fprintf(stderr, "mimo26 layer %u: expert commit failed: %s\n", layer,
                error);
        return false;
    }
    /* Remember where each expert landed for the provider's lookup. */
    for (size_t k = 0; k < count; k++) {
        worker->resolved[k] = accesses[k].admit
                                  ? accesses[k].destination_slot
                                  : accesses[k].source_slot;
        worker->resolved_expert[k] = experts[k];
    }
    worker->resolved_count = count;
    return true;
}

static bool provide_expert(void *context, uint32_t layer, uint32_t expert,
                           mimo26_rocm_expert *out)
{
    mimo26_gpu_worker *worker = (mimo26_gpu_worker *)context;
    (void)layer;
    for (size_t k = 0; k < worker->resolved_count; k++) {
        if (worker->resolved_expert[k] == expert) {
            *out = worker->slots[worker->resolved[k]].view;
            return true;
        }
    }
    /* prepare_batch admits the whole selection before anything is read, so
     * reaching here is a logic error in admission rather than a miss. */
    fprintf(stderr, "mimo26 layer %u: expert %u was not admitted\n", layer,
            expert);
    return false;
}

/*
 * Widen the layer scratch to hold `chunk` tokens.
 *
 * The decode path allocates these for one token; prefill needs the same
 * buffers `chunk` times wider. Reallocated once at create rather than per
 * call, and batch_capacity records the width so the layer entry point can
 * refuse a wider request instead of overrunning.
 */
static void scratch_resize(mimo26_gpu_worker *worker, size_t chunk,
                           char *error, size_t error_size)
{
    (void)error;
    (void)error_size;
    mimo26_rocm_layer_scratch *s = &worker->scratch;
    struct { void **slot; size_t bytes; } widened[] = {
        {&s->normed,     chunk * HIDDEN * sizeof(uint16_t)},
        {&s->fused,      chunk * MIMO26_SWA_QKV_WIDTH * sizeof(uint16_t)},
        {&s->query,      chunk * MIMO26_QUERY_HEADS * QK * sizeof(uint16_t)},
        {&s->key,        chunk * MIMO26_SWA_KV_HEADS * QK * sizeof(uint16_t)},
        {&s->value,      chunk * MIMO26_SWA_KV_HEADS * VD * sizeof(uint16_t)},
        {&s->attention,  chunk * MIMO26_QUERY_HEADS * VD * sizeof(uint16_t)},
        {&s->projected,  chunk * HIDDEN * sizeof(uint16_t)},
        {&s->mlp_gate,   chunk * 16384u * sizeof(uint16_t)},
        {&s->mlp_up,     chunk * 16384u * sizeof(uint16_t)},
        {&s->mlp_active, chunk * 16384u * sizeof(uint16_t)},
        {&s->expert_out, HIDDEN * sizeof(uint16_t)},
    };
    for (size_t i = 0; i < sizeof widened / sizeof widened[0]; i++) {
        hipFree(*widened[i].slot);
        if (hipMalloc(widened[i].slot, widened[i].bytes) != hipSuccess) {
            return;
        }
    }
    hipFree(s->accumulator);
    hipFree(s->router_logits);
    hipFree(s->router_weights);
    hipFree(s->router_ids);
    hipFree(s->attention_scratch);
    if (hipMalloc(&s->accumulator, chunk * HIDDEN * sizeof(float)) !=
            hipSuccess ||
        hipMalloc(&s->router_logits,
                  chunk * MIMO26_ROUTER_EXPERTS * sizeof(float)) !=
            hipSuccess ||
        hipMalloc(&s->router_weights,
                  chunk * MIMO26_ROUTER_TOP_K * sizeof(float)) !=
            hipSuccess ||
        hipMalloc(&s->router_ids,
                  chunk * MIMO26_ROUTER_TOP_K * sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(&s->attention_scratch,
                  chunk * mimo26_rocm_attention_scratch_floats(
                              s->attention_capacity) * sizeof(float)) !=
            hipSuccess) {
        return;
    }
    s->batch_capacity = (uint32_t)chunk;
}

/* ---- lifecycle ---- */

void mimo26_gpu_worker_destroy(mimo26_gpu_worker *worker)
{
    if (worker == NULL) {
        return;
    }
    for (uint32_t l = 0; l < LAYERS; l++) {
        mimo26_rocm_layer_weights *w = &worker->layers[l];
        hipFree((void *)w->input_layernorm);
        hipFree((void *)w->post_attention_layernorm);
        hipFree((void *)w->qkv_proj);
        hipFree((void *)w->o_proj);
        hipFree((void *)w->sink_bias);
        hipFree((void *)w->gate_weight);
        hipFree((void *)w->gate_bias);
        hipFree((void *)w->dense_gate);
        hipFree((void *)w->dense_up);
        hipFree((void *)w->dense_down);
    }
    for (uint32_t s = 0; s < worker->slot_count; s++) {
        if (worker->slots != NULL && worker->slots[s].block != NULL) {
            hipFree(worker->slots[s].block);
        }
    }
    free(worker->slots);
    if (worker->cache != NULL) {
        k3_expert_cache_destroy(worker->cache);
    }
    hipFree(worker->embed_tokens);
    hipFree(worker->final_norm);
    hipFree(worker->lm_head);
    hipFree(worker->hidden);
    hipFree(worker->normed);
    hipFree(worker->device_logits);
    hipFree(worker->device_keys);
    hipFree(worker->device_values);
    hipFree(worker->cos_table);
    hipFree(worker->sin_table);
    hipFree(worker->scratch.normed);
    hipFree(worker->scratch.fused);
    hipFree(worker->scratch.query);
    hipFree(worker->scratch.key);
    hipFree(worker->scratch.value);
    hipFree(worker->scratch.attention);
    hipFree(worker->scratch.projected);
    hipFree(worker->scratch.mlp_gate);
    hipFree(worker->scratch.mlp_up);
    hipFree(worker->scratch.mlp_active);
    hipFree(worker->scratch.expert_out);
    hipFree(worker->scratch.accumulator);
    hipFree(worker->scratch.router_logits);
    hipFree(worker->scratch.router_weights);
    hipFree(worker->scratch.router_ids);
    hipFree(worker->scratch.attention_scratch);
    if (worker->ring != NULL) {
        k3_io_uring_destroy(worker->ring);
    }
    for (uint16_t s = 0; s < MIMO26_STAGING_SLOTS; s++) {
        if (worker->staging_host[s] != NULL) {
            hipHostFree(worker->staging_host[s]);
        }
    }
    free(worker->staging_key);
    free(worker->staging_value);
    free(worker->chunk_keys);
    free(worker->chunk_values);
    free(worker->cos_tables);
    free(worker->sin_tables);
    hipFree(worker->device_cos_tables);
    hipFree(worker->device_sin_tables);
    if (worker->kv != NULL) {
        mimo26_kv_destroy(worker->kv);
    }
    if (worker->model_open) {
        k3_st_model_close(&worker->model);
    }
    free(worker);
}

mimo26_gpu_worker_status mimo26_gpu_worker_create(
    mimo26_gpu_worker **out, const char *root,
    const mimo26_gpu_worker_config *config, char *error, size_t error_size)
{
    if (out == NULL || root == NULL || config == NULL ||
        config->global_kv_capacity == 0u ||
        config->expert_slots_per_layer == 0u) {
        return fail(error, error_size, MIMO26_GPU_WORKER_INVALID_ARGUMENT,
                    "invalid worker configuration");
    }
    /* The same floor the CPU worker enforces, and for the same reason: one
     * token routes to MIMO26_ROUTER_TOP_K experts per layer and needs them
     * resident together. */
    if (config->expert_slots_per_layer < MIMO26_ROUTER_TOP_K) {
        return fail(error, error_size, MIMO26_GPU_WORKER_INVALID_ARGUMENT,
                    "%u expert slots per layer cannot hold the %u experts one "
                    "token routes to", (unsigned)config->expert_slots_per_layer,
                    (unsigned)MIMO26_ROUTER_TOP_K);
    }
    *out = NULL;

    const uint64_t planned = mimo26_gpu_worker_planned_bytes(config);
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        return fail(error, error_size, MIMO26_GPU_WORKER_LOAD_FAILED,
                    "could not query device memory");
    }
    /* Admission against measured availability, not a nominal device size. */
    const uint64_t limit = config->memory_limit_bytes != 0u
                               ? config->memory_limit_bytes
                               : (uint64_t)free_bytes;
    if (planned > limit) {
        return fail(error, error_size, MIMO26_GPU_WORKER_MEMORY_LIMIT,
                    "plan needs %.2f GiB but only %.2f GiB is available",
                    (double)planned / 1073741824.0,
                    (double)limit / 1073741824.0);
    }

    mimo26_gpu_worker *worker =
        (mimo26_gpu_worker *)calloc(1, sizeof *worker);
    if (worker == NULL) {
        return fail(error, error_size, MIMO26_GPU_WORKER_OUT_OF_MEMORY,
                    "out of memory");
    }
    worker->config = *config;

    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);

    if (!mimo26_manifest_load(&worker->manifest, root, error, error_size) ||
        !mimo26_manifest_open_model(&worker->manifest, root, &worker->model,
                                    error, error_size)) {
        free(worker);
        return MIMO26_GPU_WORKER_LOAD_FAILED;
    }
    worker->model_open = true;

    #define REQUIRE(expression, message)                                      \
        do {                                                                  \
            if (!(expression)) {                                              \
                mimo26_gpu_worker_destroy(worker);                            \
                return fail(error, error_size, MIMO26_GPU_WORKER_LOAD_FAILED, \
                            "%s", (message));                                 \
            }                                                                 \
        } while (0)

    mimo26_static_weights statics;
    memset(&statics, 0, sizeof statics);
    REQUIRE(mimo26_static_weights_load(&statics, &worker->model, error,
                                       error_size) == MIMO26_WEIGHTS_OK,
            "static weights failed to load");
    worker->embed_tokens =
        upload(worker, statics.embed_tokens,
               (size_t)VOCAB * HIDDEN * sizeof(uint16_t));
    worker->final_norm =
        upload(worker, statics.norm, HIDDEN * sizeof(uint16_t));
    worker->lm_head = upload(worker, statics.lm_head,
                             (size_t)VOCAB * HIDDEN * sizeof(uint16_t));
    mimo26_static_weights_free(&statics);
    REQUIRE(worker->embed_tokens && worker->final_norm && worker->lm_head,
            "static weights failed to upload");

    for (uint32_t l = 0; l < LAYERS; l++) {
        mimo26_layer_weights host;
        memset(&host, 0, sizeof host);
        REQUIRE(mimo26_layer_weights_load(&host, &worker->model, l, error,
                                          error_size) == MIMO26_WEIGHTS_OK,
                "layer weights failed to load");
        mimo26_rocm_layer_weights *w = &worker->layers[l];
        w->layer = l;
        w->is_swa = host.attention.is_swa;
        w->is_moe = host.is_moe;
        w->kv_heads = (uint32_t)host.attention.kv_heads;
        w->kv_groups = (uint32_t)host.attention.kv_groups;
        w->qkv_width = (uint32_t)host.attention.qkv_width;
        w->window = (uint32_t)host.attention.window;
        w->input_layernorm =
            upload(worker, host.input_layernorm, HIDDEN * sizeof(uint16_t));
        w->post_attention_layernorm =
            upload(worker, host.post_attention_layernorm,
                   HIDDEN * sizeof(uint16_t));
        w->qkv_proj = upload(worker, host.qkv_proj,
                             (size_t)host.attention.qkv_width * HIDDEN *
                                 sizeof(uint16_t));
        w->o_proj = upload(worker, host.o_proj,
                           (size_t)HIDDEN * MIMO26_QUERY_HEADS * VD *
                               sizeof(uint16_t));
        if (host.sink_bias != NULL) {
            w->sink_bias = upload(worker, host.sink_bias,
                                  MIMO26_QUERY_HEADS * sizeof(uint16_t));
        }
        if (host.is_moe) {
            w->gate_weight =
                upload(worker, host.gate_weight,
                       (size_t)MIMO26_ROUTER_EXPERTS * HIDDEN *
                           sizeof(uint16_t));
            w->gate_bias = (const float *)upload(
                worker, host.gate_bias,
                MIMO26_ROUTER_EXPERTS * sizeof(float));
        } else {
            w->dense_gate = upload(worker, host.dense_gate,
                                   (size_t)16384 * HIDDEN * sizeof(uint16_t));
            w->dense_up = upload(worker, host.dense_up,
                                 (size_t)16384 * HIDDEN * sizeof(uint16_t));
            w->dense_down = upload(worker, host.dense_down,
                                   (size_t)HIDDEN * 16384 * sizeof(uint16_t));
        }
        mimo26_layer_weights_free(&host);
        REQUIRE(w->input_layernorm && w->qkv_proj && w->o_proj,
                "layer weights failed to upload");


    }

    /*
     * One cache over all 48 layers, slots addressed by the global index
     * k3_expert_cache hands out. Layer 0 is dense and never routes, so its
     * slots are simply never planned against -- cheaper than special-casing
     * the indexing, and it keeps slot ids and layer ids aligned.
     */
    REQUIRE(k3_expert_cache_create(&worker->cache, (uint16_t)LAYERS,
                                   config->expert_slots_per_layer, error,
                                   error_size),
            "expert cache creation failed");
    worker->slot_count = k3_expert_cache_slot_count(worker->cache);
    worker->slots =
        (expert_slot *)calloc(worker->slot_count, sizeof *worker->slots);
    REQUIRE(worker->slots != NULL, "expert slot table allocation failed");
    /*
     * Layer 0 is dense and never plans against the cache, so its slot ids
     * exist but are left unbacked -- 8 slots of 12.75 MiB that could never
     * be used. Backing them cost 102 MiB and made the residency ledger
     * exceed the plan, which the qualification harness refused. Keeping the
     * ids aligned with layer ids is still worth it; only the storage is
     * skipped.
     */
    for (uint32_t s = config->expert_slots_per_layer;
         s < worker->slot_count; s++) {
        REQUIRE(hipMalloc(&worker->slots[s].block, PACKED_EXPERT_BYTES) ==
                    hipSuccess,
                "expert slot allocation failed");
        worker->resident_bytes += PACKED_EXPERT_BYTES;
        slot_views(&worker->slots[s]);
    }

    /* Scratch and per-step buffers. */
    const uint64_t capacity = config->global_kv_capacity;
    worker->scratch.attention_capacity = capacity;
    #define DEVICE(field, bytes)                                              \
        REQUIRE(hipMalloc(&worker->field, (bytes)) == hipSuccess,             \
                "device allocation failed")
    DEVICE(hidden, HIDDEN * sizeof(uint16_t));
    DEVICE(normed, HIDDEN * sizeof(uint16_t));
    DEVICE(device_logits, (size_t)VOCAB * sizeof(float));
    DEVICE(device_keys,
           (size_t)capacity * MIMO26_SWA_KV_HEADS * QK * sizeof(uint16_t));
    DEVICE(device_values,
           (size_t)capacity * MIMO26_SWA_KV_HEADS * VD * sizeof(uint16_t));
    DEVICE(cos_table, MIMO26_ROPE_DIM * sizeof(uint16_t));
    DEVICE(sin_table, MIMO26_ROPE_DIM * sizeof(uint16_t));
    #undef DEVICE
    #define SCRATCH(field, bytes)                                             \
        REQUIRE(hipMalloc(&worker->scratch.field, (bytes)) == hipSuccess,     \
                "scratch allocation failed")
    SCRATCH(normed, HIDDEN * sizeof(uint16_t));
    SCRATCH(fused, MIMO26_SWA_QKV_WIDTH * sizeof(uint16_t));
    SCRATCH(query, (size_t)MIMO26_QUERY_HEADS * QK * sizeof(uint16_t));
    SCRATCH(key, (size_t)MIMO26_SWA_KV_HEADS * QK * sizeof(uint16_t));
    SCRATCH(value, (size_t)MIMO26_SWA_KV_HEADS * VD * sizeof(uint16_t));
    SCRATCH(attention, (size_t)MIMO26_QUERY_HEADS * VD * sizeof(uint16_t));
    SCRATCH(projected, HIDDEN * sizeof(uint16_t));
    SCRATCH(mlp_gate, 16384 * sizeof(uint16_t));
    SCRATCH(mlp_up, 16384 * sizeof(uint16_t));
    SCRATCH(mlp_active, 16384 * sizeof(uint16_t));
    SCRATCH(expert_out, HIDDEN * sizeof(uint16_t));
    SCRATCH(accumulator, HIDDEN * sizeof(float));
    SCRATCH(router_logits, MIMO26_ROUTER_EXPERTS * sizeof(float));
    SCRATCH(router_weights, MIMO26_ROUTER_TOP_K * sizeof(float));
    SCRATCH(router_ids, MIMO26_ROUTER_TOP_K * sizeof(uint32_t));
    SCRATCH(attention_scratch,
            mimo26_rocm_attention_scratch_floats(capacity) * sizeof(float));
    #undef SCRATCH

    /*
     * Mapped staging plus a registered ring, following K3's engine. The
     * buffers are GPU-visible, so the copy out of them is a pinned transfer
     * rather than a pageable one, and O_DIRECT reads land straight in them.
     */
    for (uint16_t s = 0; s < MIMO26_STAGING_SLOTS; s++) {
        REQUIRE(hipHostMalloc(&worker->staging_host[s],
                              MIMO26_STAGING_BYTES,
                              hipHostMallocMapped) == hipSuccess,
                "staging allocation failed");
        worker->staging_iov[s].iov_base = worker->staging_host[s];
        worker->staging_iov[s].iov_len = MIMO26_STAGING_BYTES;
    }
    REQUIRE(k3_io_uring_create(&worker->ring, worker->staging_iov,
                               MIMO26_STAGING_SLOTS, error, error_size),
            "io_uring creation failed");
    worker->staging_ready = true;

    /*
     * Prefill scratch. Sized for the configured chunk so the layer entry
     * point can refuse anything wider rather than overrun buffers shaped
     * for a single token.
     */
    worker->prefill_chunk = config->prefill_chunk;
    if (worker->prefill_chunk > 0u) {
        const size_t chunk = worker->prefill_chunk;
        scratch_resize(worker, chunk, error, error_size);
        REQUIRE(worker->scratch.batch_capacity == chunk,
                "prefill scratch allocation failed");
        worker->chunk_keys = (uint16_t *)calloc(
            (size_t)LAYERS * chunk * MIMO26_SWA_KV_HEADS * QK,
            sizeof *worker->chunk_keys);
        worker->chunk_values = (uint16_t *)calloc(
            (size_t)LAYERS * chunk * MIMO26_SWA_KV_HEADS * VD,
            sizeof *worker->chunk_values);
        worker->cos_tables = (uint16_t *)calloc(chunk * MIMO26_ROPE_DIM,
                                                sizeof *worker->cos_tables);
        worker->sin_tables = (uint16_t *)calloc(chunk * MIMO26_ROPE_DIM,
                                                sizeof *worker->sin_tables);
        REQUIRE(worker->chunk_keys && worker->chunk_values &&
                worker->cos_tables && worker->sin_tables,
                "prefill staging allocation failed");
        REQUIRE(hipMalloc(&worker->device_cos_tables,
                          chunk * MIMO26_ROPE_DIM * sizeof(uint16_t)) ==
                    hipSuccess &&
                hipMalloc(&worker->device_sin_tables,
                          chunk * MIMO26_ROPE_DIM * sizeof(uint16_t)) ==
                    hipSuccess,
                "prefill rope table allocation failed");
    }

    worker->staging_key = (uint16_t *)calloc(
        (size_t)MIMO26_SWA_KV_HEADS * QK, sizeof *worker->staging_key);
    worker->staging_value = (uint16_t *)calloc(
        (size_t)MIMO26_SWA_KV_HEADS * VD, sizeof *worker->staging_value);
    REQUIRE(worker->staging_key && worker->staging_value,
            "staging allocation failed");
    REQUIRE(mimo26_kv_create(&worker->kv, capacity, 8u) == MIMO26_KV_OK,
            "kv allocation failed");
    #undef REQUIRE

    struct timespec finished;
    clock_gettime(CLOCK_MONOTONIC, &finished);
    worker->stats.load_seconds =
        (double)(finished.tv_sec - started.tv_sec) +
        (double)(finished.tv_nsec - started.tv_nsec) / 1e9;
    *out = worker;
    return MIMO26_GPU_WORKER_OK;
}

/* ---- decode ---- */

mimo26_gpu_worker_status mimo26_gpu_worker_decode(mimo26_gpu_worker *worker,
                                                  uint32_t token_id,
                                                  float *logits, char *error,
                                                  size_t error_size)
{
    if (worker == NULL || logits == NULL) {
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    }
    if (token_id >= VOCAB) {
        return fail(error, error_size, MIMO26_GPU_WORKER_INVALID_ARGUMENT,
                    "token id %u is beyond the tokenizer vocabulary",
                    token_id);
    }
    if (worker->position >= worker->config.global_kv_capacity) {
        return fail(error, error_size, MIMO26_GPU_WORKER_CAPACITY_EXCEEDED,
                    "context capacity %zu reached",
                    worker->config.global_kv_capacity);
    }
    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);

    const uint64_t position = worker->position;
    /* Phase timing, so a bottleneck is located rather than guessed at. */
    const bool profile = getenv("MIMO26_GPU_PROFILE") != NULL;
    double kv_seconds = 0.0, layer_seconds = 0.0, head_seconds = 0.0;
    double stage_seconds = 0.0;
    g_upload_seconds = 0.0;
    g_read_seconds = 0.0;
    g_copy_seconds = 0.0;
    struct timespec mark, mark2;
    #define TICK() do { if (profile) { hipDeviceSynchronize(); \
        clock_gettime(CLOCK_MONOTONIC, &mark); } } while (0)
    #define TOCK(acc) do { if (profile) { hipDeviceSynchronize(); \
        clock_gettime(CLOCK_MONOTONIC, &mark2); \
        (acc) += (double)(mark2.tv_sec - mark.tv_sec) + \
                 (double)(mark2.tv_nsec - mark.tv_nsec) / 1e9; } } while (0)
    if (mimo26_kv_begin(worker->kv, position) != MIMO26_KV_OK) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "could not open a transaction");
    }

    if (hipMemcpy(worker->hidden,
                  (const uint16_t *)worker->embed_tokens +
                      (size_t)token_id * HIDDEN,
                  HIDDEN * sizeof(uint16_t),
                  hipMemcpyDeviceToDevice) != hipSuccess) {
        mimo26_kv_abort(worker->kv);
        worker->stats.aborted_steps++;
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "embedding lookup failed");
    }

    for (uint32_t l = 0; l < LAYERS; l++) {
        const mimo26_rocm_layer_weights *w = &worker->layers[l];

        /* Mirror this layer's committed history up. Windowed layers keep at
         * most 128 positions, so most of this is small. */
        const uint16_t *view_keys = NULL;
        const uint16_t *view_values = NULL;
        size_t history = 0;
        uint64_t first_position = 0;
        if (mimo26_kv_view(worker->kv, l, &view_keys, &view_values, &history,
                           &first_position) != MIMO26_KV_OK) {
            mimo26_kv_abort(worker->kv);
            worker->stats.aborted_steps++;
            return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                        "kv view failed at layer %u", l);
        }
        TICK();
        if (history > 0) {
            /*
             * Checked, because a silently failed copy here does not crash --
             * it attends over stale or uninitialized history and returns a
             * plausible wrong answer, which is the worst failure mode this
             * worker has.
             */
            if (hipMemcpy(worker->device_keys, view_keys,
                          history * w->kv_heads * QK * sizeof(uint16_t),
                          hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(worker->device_values, view_values,
                          history * w->kv_heads * VD * sizeof(uint16_t),
                          hipMemcpyHostToDevice) != hipSuccess) {
                mimo26_kv_abort(worker->kv);
                worker->stats.aborted_steps++;
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "history upload failed at layer %u", l);
            }
        }

        TOCK(kv_seconds);

        uint16_t cos_host[MIMO26_ROPE_DIM];
        uint16_t sin_host[MIMO26_ROPE_DIM];
        mimo26_attention_config attention_config;
        mimo26_attention_config_for_layer(l, &attention_config);
        mimo26_rope_table(cos_host, sin_host, position,
                          attention_config.rope_theta);
        if (hipMemcpy(worker->cos_table, cos_host, sizeof cos_host,
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(worker->sin_table, sin_host, sizeof sin_host,
                      hipMemcpyHostToDevice) != hipSuccess) {
            mimo26_kv_abort(worker->kv);
            worker->stats.aborted_steps++;
            return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                        "rope table upload failed at layer %u", l);
        }

        TICK();
        mimo26_rocm_layer context;
        memset(&context, 0, sizeof context);
        context.weights = w;
        context.prepare = w->is_moe ? prepare_batch : NULL;
        context.provider = w->is_moe ? provide_expert : NULL;
        context.provider_context = worker;

        const mimo26_rocm_layer_status layer_status =
            mimo26_rocm_layer_decode(&context, &worker->scratch,
                                     worker->hidden, worker->device_keys,
                                     worker->device_values, worker->cos_table,
                                     worker->sin_table, history,
                                     first_position, position, NULL, NULL);
        if (layer_status != MIMO26_ROCM_LAYER_OK) {
            mimo26_kv_abort(worker->kv);
            worker->stats.aborted_steps++;
            return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                        "layer %u failed with status %d", l,
                        (int)layer_status);
        }

        TOCK(layer_seconds);

        /* Stage this layer's key and value back into the host-side history.
         * The commit happens once all 48 have run, so a failure above leaves
         * committed history untouched. */
        TICK();
        if (hipMemcpy(worker->staging_key, worker->scratch.key,
                      w->kv_heads * QK * sizeof(uint16_t),
                      hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(worker->staging_value, worker->scratch.value,
                      w->kv_heads * VD * sizeof(uint16_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            mimo26_kv_abort(worker->kv);
            worker->stats.aborted_steps++;
            return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                        "key/value readback failed at layer %u", l);
        }
        TOCK(stage_seconds);
        if (mimo26_kv_stage(worker->kv, l, worker->staging_key,
                            worker->staging_value) != MIMO26_KV_OK) {
            mimo26_kv_abort(worker->kv);
            worker->stats.aborted_steps++;
            return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                        "kv stage failed at layer %u", l);
        }
    }

    if (mimo26_kv_commit(worker->kv) != MIMO26_KV_OK) {
        mimo26_kv_abort(worker->kv);
        worker->stats.aborted_steps++;
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "commit failed");
    }

    TICK();
    if (!mimo26_rocm_rmsnorm_bf16(worker->normed, worker->hidden,
                                  worker->final_norm, 1u, HIDDEN, 1e-6f,
                                  NULL)) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "final norm failed");
    }
    /* Logits in F32, summed in the CPU's order: this is the answer itself,
     * and a reassociation can flip a near-tie into a different token. */
    /*
     * lm_head uses the tree reduction, unlike the router and RMSNorm.
     *
     * The ordered sum is kept for those two because there it is the dominant
     * error term: the router's row cancels heavily, and pinning its order
     * took the mixing-weight error from 4.5e-04 to about one float ulp.
     * lm_head is a different situation. Its input is the final norm of a
     * hidden state that has already accumulated 48 layers of bounded
     * cross-backend difference, so the input error dominates and the
     * association contributes little -- the ordered version cannot deliver
     * exact logits no matter how it sums, because what it sums is not exact.
     * It was costing 23 ms against 6 ms for a property it does not provide.
     *
     * What the qualification gate actually requires is determinism within a
     * backend, and a fixed-order tree is exactly as deterministic as a
     * sequential sum. Measured over 59 positions the two pick the same
     * argmax every time, which corroborates rather than establishes this --
     * the argument above is the reason.
     *
     * MIMO26_GPU_EXACT_HEAD restores the ordered sum for anyone localizing a
     * divergence.
     */
    const bool exact_head = getenv("MIMO26_GPU_EXACT_HEAD") != NULL;
    if (!(!exact_head
              ? k3_rocm_bf16_gemv_f32(worker->device_logits, worker->lm_head,
                                      worker->normed, VOCAB, HIDDEN, NULL)
              : mimo26_rocm_ordered_gemv_f32(worker->device_logits,
                                             worker->lm_head, worker->normed,
                                             VOCAB, HIDDEN, NULL))) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "lm_head failed");
    }
    if (hipMemcpy(logits, worker->device_logits,
                  (size_t)VOCAB * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "logit readback failed");
    }
    TOCK(head_seconds);
    /* The padded tail decodes to no token, so it must never be sampled. */
    for (uint32_t row = MIMO26_GPU_TOKENIZER_VOCAB; row < VOCAB; row++) {
        logits[row] = -INFINITY;
    }
    if (profile) {
        fprintf(stderr, "    profile pos %llu: layers %.3f s (of which "
                        "expert admission %.3f s [read %.3f, copy %.3f], "
                        "compute %.3f s), "
                        "kv-up %.3f s, kv-down %.3f s, head %.3f s\n",
                (unsigned long long)position, layer_seconds,
                g_upload_seconds, g_read_seconds, g_copy_seconds,
                layer_seconds - g_upload_seconds,
                kv_seconds, stage_seconds, head_seconds);
    }
    #undef TICK
    #undef TOCK

    worker->position++;
    struct timespec finished;
    clock_gettime(CLOCK_MONOTONIC, &finished);
    worker->stats.last_decode_seconds =
        (double)(finished.tv_sec - started.tv_sec) +
        (double)(finished.tv_nsec - started.tv_nsec) / 1e9;
    worker->stats.tokens++;
    return MIMO26_GPU_WORKER_OK;
}

mimo26_gpu_worker_status mimo26_gpu_worker_prefill(mimo26_gpu_worker *worker,
                                                   const uint32_t *tokens,
                                                   size_t count,
                                                   float *logits,
                                                   char *error,
                                                   size_t error_size)
{
    if (worker == NULL || tokens == NULL || logits == NULL) {
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    }
    /* Without a configured chunk this is the decode loop, kept so the old
     * path stays reachable for comparison rather than deleted. */
    if (worker->prefill_chunk == 0u) {
        for (size_t i = 0; i < count; i++) {
            const mimo26_gpu_worker_status status =
                mimo26_gpu_worker_decode(worker, tokens[i], logits, error,
                                         error_size);
            if (status != MIMO26_GPU_WORKER_OK) {
                return status;
            }
        }
        return MIMO26_GPU_WORKER_OK;
    }
    for (size_t i = 0; i < count; i++) {
        if (tokens[i] >= VOCAB) {
            return fail(error, error_size, MIMO26_GPU_WORKER_INVALID_ARGUMENT,
                        "token id %u is beyond the tokenizer vocabulary",
                        tokens[i]);
        }
    }
    if (worker->position + count > worker->config.global_kv_capacity) {
        return fail(error, error_size, MIMO26_GPU_WORKER_CAPACITY_EXCEEDED,
                    "prompt of %zu tokens exceeds the context of %zu",
                    count, worker->config.global_kv_capacity);
    }

    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);
    const bool profile = getenv("MIMO26_GPU_PROFILE") != NULL;
    double kv_seconds = 0.0, rope_seconds = 0.0, layer_seconds = 0.0;
    double readback_seconds = 0.0, commit_seconds = 0.0;
    g_upload_seconds = 0.0;
    struct timespec mark, mark2;
    #define PTICK() do { if (profile) { hipDeviceSynchronize(); \
        clock_gettime(CLOCK_MONOTONIC, &mark); } } while (0)
    #define PTOCK(acc) do { if (profile) { hipDeviceSynchronize(); \
        clock_gettime(CLOCK_MONOTONIC, &mark2); \
        (acc) += (double)(mark2.tv_sec - mark.tv_sec) + \
                 (double)(mark2.tv_nsec - mark.tv_nsec) / 1e9; } } while (0)

    size_t done = 0;
    while (done < count) {
        const size_t chunk = (count - done) < worker->prefill_chunk
                                 ? (count - done) : worker->prefill_chunk;
        const uint64_t base_position = worker->position;

        /* Gather the chunk's embeddings. */
        for (size_t b = 0; b < chunk; b++) {
            if (hipMemcpy((uint16_t *)worker->hidden + b * HIDDEN,
                          (const uint16_t *)worker->embed_tokens +
                              (size_t)tokens[done + b] * HIDDEN,
                          HIDDEN * sizeof(uint16_t),
                          hipMemcpyDeviceToDevice) != hipSuccess) {
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "embedding gather failed");
            }
        }

        for (uint32_t l = 0; l < LAYERS; l++) {
            const mimo26_rocm_layer_weights *w = &worker->layers[l];
            const uint16_t *view_keys = NULL;
            const uint16_t *view_values = NULL;
            size_t history = 0;
            uint64_t first_position = 0;
            if (mimo26_kv_view(worker->kv, l, &view_keys, &view_values,
                               &history, &first_position) != MIMO26_KV_OK) {
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "kv view failed at layer %u", l);
            }
            PTICK();
            if (history > 0 &&
                (hipMemcpy(worker->device_keys, view_keys,
                           history * w->kv_heads * QK * sizeof(uint16_t),
                           hipMemcpyHostToDevice) != hipSuccess ||
                 hipMemcpy(worker->device_values, view_values,
                           history * w->kv_heads * VD * sizeof(uint16_t),
                           hipMemcpyHostToDevice) != hipSuccess)) {
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "history upload failed at layer %u", l);
            }

            PTOCK(kv_seconds);
            PTICK();
            mimo26_attention_config attention_config;
            mimo26_attention_config_for_layer(l, &attention_config);
            for (size_t b = 0; b < chunk; b++) {
                mimo26_rope_table(worker->cos_tables + b * MIMO26_ROPE_DIM,
                                  worker->sin_tables + b * MIMO26_ROPE_DIM,
                                  base_position + b,
                                  attention_config.rope_theta);
            }
            if (hipMemcpy(worker->device_cos_tables, worker->cos_tables,
                          chunk * MIMO26_ROPE_DIM * sizeof(uint16_t),
                          hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(worker->device_sin_tables, worker->sin_tables,
                          chunk * MIMO26_ROPE_DIM * sizeof(uint16_t),
                          hipMemcpyHostToDevice) != hipSuccess) {
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "rope upload failed at layer %u", l);
            }

            PTOCK(rope_seconds);
            PTICK();
            mimo26_rocm_layer context;
            memset(&context, 0, sizeof context);
            context.weights = w;
            context.prepare = w->is_moe ? prepare_batch : NULL;
            context.provider = w->is_moe ? provide_expert : NULL;
            context.provider_context = worker;

            const mimo26_rocm_layer_status status = mimo26_rocm_layer_prefill(
                &context, &worker->scratch, worker->hidden,
                worker->device_keys, worker->device_values,
                worker->device_cos_tables, worker->device_sin_tables,
                history, first_position, base_position, (uint32_t)chunk,
                NULL, NULL);
            if (status != MIMO26_ROCM_LAYER_OK) {
                mimo26_kv_abort(worker->kv);
                worker->stats.aborted_steps++;
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "prefill layer %u failed with status %d", l,
                            (int)status);
            }

            PTOCK(layer_seconds);
            PTICK();
            /* Keep this layer's chunk keys and values for staging once every
             * layer has run, so an abort leaves the journal untouched. */
            uint16_t *key_slot = worker->chunk_keys +
                                 (size_t)l * worker->prefill_chunk *
                                     MIMO26_SWA_KV_HEADS * QK;
            uint16_t *value_slot = worker->chunk_values +
                                   (size_t)l * worker->prefill_chunk *
                                       MIMO26_SWA_KV_HEADS * VD;
            if (hipMemcpy(key_slot, worker->scratch.key,
                          chunk * w->kv_heads * QK * sizeof(uint16_t),
                          hipMemcpyDeviceToHost) != hipSuccess ||
                hipMemcpy(value_slot, worker->scratch.value,
                          chunk * w->kv_heads * VD * sizeof(uint16_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "key/value readback failed at layer %u", l);
            }
        }

        PTOCK(readback_seconds);
        PTICK();
        /* Commit the chunk one position at a time, so the journal sees the
         * same sequence it would have seen from decode. */
        for (size_t b = 0; b < chunk; b++) {
            if (mimo26_kv_begin(worker->kv, base_position + b) !=
                MIMO26_KV_OK) {
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "kv begin failed");
            }
            for (uint32_t l = 0; l < LAYERS; l++) {
                const uint32_t kv_heads = worker->layers[l].kv_heads;
                const uint16_t *key_slot =
                    worker->chunk_keys +
                    (size_t)l * worker->prefill_chunk *
                        MIMO26_SWA_KV_HEADS * QK + b * kv_heads * QK;
                const uint16_t *value_slot =
                    worker->chunk_values +
                    (size_t)l * worker->prefill_chunk *
                        MIMO26_SWA_KV_HEADS * VD + b * kv_heads * VD;
                if (mimo26_kv_stage(worker->kv, l, key_slot, value_slot) !=
                    MIMO26_KV_OK) {
                    mimo26_kv_abort(worker->kv);
                    return fail(error, error_size,
                                MIMO26_GPU_WORKER_DECODE_FAILED,
                                "kv stage failed");
                }
            }
            if (mimo26_kv_commit(worker->kv) != MIMO26_KV_OK) {
                mimo26_kv_abort(worker->kv);
                return fail(error, error_size,
                            MIMO26_GPU_WORKER_DECODE_FAILED,
                            "kv commit failed");
            }
            worker->position++;
            worker->stats.tokens++;
        }
        PTOCK(commit_seconds);
        done += chunk;
    }
    if (profile) {
        fprintf(stderr, "    prefill %zu tokens: layers %.2f s (admission "
                        "%.2f), kv-up %.2f, rope %.2f, readback %.2f, "
                        "commit %.2f\n",
                count, layer_seconds, g_upload_seconds, kv_seconds,
                rope_seconds, readback_seconds, commit_seconds);
    }
    #undef PTICK
    #undef PTOCK

    /* Only the last token's distribution matters for a prompt. */
    const uint16_t *last_hidden = (const uint16_t *)worker->hidden +
                                  ((count - 1u) % worker->prefill_chunk) *
                                      HIDDEN;
    if (!mimo26_rocm_rmsnorm_bf16(worker->normed, last_hidden,
                                  worker->final_norm, 1u, HIDDEN, 1e-6f,
                                  NULL) ||
        !k3_rocm_bf16_gemv_f32(worker->device_logits, worker->lm_head,
                               worker->normed, VOCAB, HIDDEN, NULL) ||
        hipMemcpy(logits, worker->device_logits,
                  (size_t)VOCAB * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "final projection failed");
    }
    for (uint32_t row = MIMO26_GPU_TOKENIZER_VOCAB; row < VOCAB; row++) {
        logits[row] = -INFINITY;
    }

    struct timespec finished;
    clock_gettime(CLOCK_MONOTONIC, &finished);
    worker->stats.last_decode_seconds =
        (double)(finished.tv_sec - started.tv_sec) +
        (double)(finished.tv_nsec - started.tv_nsec) / 1e9;
    return MIMO26_GPU_WORKER_OK;
}

uint32_t mimo26_gpu_worker_argmax(const float *logits)
{
    uint32_t best = 0;
    float best_value = -INFINITY;
    for (uint32_t i = 0; i < MIMO26_GPU_TOKENIZER_VOCAB; i++) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    return best;
}

uint64_t mimo26_gpu_worker_position(const mimo26_gpu_worker *worker)
{
    return worker == NULL ? 0u : worker->position;
}

mimo26_gpu_worker_status mimo26_gpu_worker_rollback(mimo26_gpu_worker *worker,
                                                    size_t count)
{
    if (worker == NULL) {
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    }
    if (count > worker->position) {
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    }
    /*
     * The KV journal is the source of truth for history, and the device
     * copies are re-mirrored from it at the top of every layer, so rewinding
     * the host side is sufficient -- there is no device state to unwind
     * separately. That is a property worth keeping if the mirroring is ever
     * removed for speed.
     */
    if (mimo26_kv_rollback(worker->kv, count) != MIMO26_KV_OK) {
        return MIMO26_GPU_WORKER_DECODE_FAILED;
    }
    worker->position -= count;
    return MIMO26_GPU_WORKER_OK;
}

void mimo26_gpu_worker_reset(mimo26_gpu_worker *worker)
{
    if (worker == NULL) {
        return;
    }
    mimo26_kv_reset(worker->kv);
    worker->position = 0u;
    /* Residency is a cache, so it could be kept -- but reset promises a
     * fresh worker, and leaving stale mappings would make the hit rate of
     * the next run depend on the last one. */
    char error[256];
    k3_expert_cache_reset(worker->cache, error, sizeof error);
    worker->resolved_count = 0u;
}

void mimo26_gpu_worker_get_stats(const mimo26_gpu_worker *worker,
                                 mimo26_gpu_worker_stats *stats)
{
    if (worker == NULL || stats == NULL) {
        return;
    }
    *stats = worker->stats;
    if (worker->cache != NULL) {
        k3_expert_cache_stats cache_stats;
        k3_expert_cache_get_stats(worker->cache, &cache_stats);
        stats->expert_accesses = cache_stats.accesses;
        stats->expert_hits = cache_stats.hits;
    }
}
