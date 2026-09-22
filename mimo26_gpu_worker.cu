#include "mimo26_gpu_worker.h"

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

typedef struct {
    uint32_t expert;       /* which identity occupies this slot */
    uint64_t last_used;    /* for LRU */
    bool     occupied;
    bool     pinned;       /* needed by the step in flight */
    mimo26_rocm_expert view;
    void    *storage[6];   /* the six allocations backing `view` */
} expert_slot;

typedef struct {
    expert_slot *slots;
    uint16_t     count;
} layer_cache;

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

    layer_cache  caches[LAYERS];
    uint64_t     clock;                /* LRU tick */

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

static bool ensure_expert(mimo26_gpu_worker *worker, uint32_t layer,
                          uint32_t expert, mimo26_rocm_expert *out)
{
    layer_cache *cache = &worker->caches[layer];

    /*
     * Counted here, in admission, not in the provider. The provider runs
     * after prepare_batch has already made the whole selection resident, so
     * it hits by construction -- an earlier version reported its hit rate
     * and printed a meaningless 100%. What a caller needs to know is how
     * often an expert had to be fetched, which is exactly this decision.
     */
    worker->stats.expert_accesses++;
    for (uint16_t s = 0; s < cache->count; s++) {
        expert_slot *slot = &cache->slots[s];
        if (slot->occupied && slot->expert == expert) {
            slot->last_used = ++worker->clock;
            slot->pinned = true;
            worker->stats.expert_hits++;
            if (out != NULL) {
                *out = slot->view;
            }
            return true;
        }
    }

    /* Evict the least recently used slot that this step does not need. A
     * pinned slot holds an expert the token in flight will still consume, so
     * taking it would mean reading it back in later in the same step. */
    expert_slot *victim = NULL;
    for (uint16_t s = 0; s < cache->count; s++) {
        expert_slot *slot = &cache->slots[s];
        if (!slot->occupied) {
            victim = slot;
            break;
        }
        if (slot->pinned) {
            continue;
        }
        if (victim == NULL || slot->last_used < victim->last_used) {
            victim = slot;
        }
    }
    if (victim == NULL) {
        return false;   /* every slot pinned: the cache is smaller than top-k */
    }
    if (victim->occupied) {
        /* Subtract exactly what is released. An earlier draft reset the
         * ledger here, which would have made the residency figure lie from
         * the first eviction onward -- and that figure is what the memory
         * guard and the evidence bundle both rest on. */
        for (size_t i = 0; i < 6; i++) {
            if (victim->storage[i] != NULL) {
                hipFree(victim->storage[i]);
                victim->storage[i] = NULL;
            }
        }
        worker->resident_bytes -=
            worker->resident_bytes >= PACKED_EXPERT_BYTES
                ? PACKED_EXPERT_BYTES : worker->resident_bytes;
        victim->occupied = false;
    }

    static const char *kinds[3] = {"gate_proj", "up_proj", "down_proj"};
    const void **packed[3] = {&victim->view.gate_packed,
                              &victim->view.up_packed,
                              &victim->view.down_packed};
    const void **scales[3] = {&victim->view.gate_scales,
                              &victim->view.up_scales,
                              &victim->view.down_scales};
    size_t written = 0;
    for (size_t j = 0; j < 3; j++) {
        char name[320];
        snprintf(name, sizeof name,
                 "model.layers.%u.mlp.experts.%u.%s.weight", layer, expert,
                 kinds[j]);
        void *w = upload_tensor(worker, name, NULL);
        snprintf(name, sizeof name,
                 "model.layers.%u.mlp.experts.%u.%s.weight_scale", layer,
                 expert, kinds[j]);
        void *s = upload_tensor(worker, name, NULL);
        if (w == NULL || s == NULL) {
            if (w != NULL) { hipFree(w); }
            if (s != NULL) { hipFree(s); }
            victim->occupied = false;
            return false;
        }
        *packed[j] = w;
        *scales[j] = s;
        victim->storage[written++] = w;
        victim->storage[written++] = s;
    }
    victim->expert = expert;
    victim->occupied = true;
    victim->pinned = true;
    victim->last_used = ++worker->clock;
    worker->stats.expert_uploads++;
    if (out != NULL) {
        *out = victim->view;
    }
    return true;
}

static bool prepare_batch(void *context, uint32_t layer,
                          const uint32_t *experts, size_t count)
{
    mimo26_gpu_worker *worker = (mimo26_gpu_worker *)context;
    layer_cache *cache = &worker->caches[layer];
    /*
     * Two phases, as the CPU worker does. Unpin first so last step's
     * residents are all evictable, then admit the whole batch before a
     * single expert is consumed -- otherwise admitting the eighth can evict
     * the first, which the same token is about to read.
     */
    for (uint16_t s = 0; s < cache->count; s++) {
        cache->slots[s].pinned = false;
    }
    for (size_t k = 0; k < count; k++) {
        if (!ensure_expert(worker, layer, experts[k], NULL)) {
            return false;
        }
    }
    return true;
}

static bool provide_expert(void *context, uint32_t layer, uint32_t expert,
                           mimo26_rocm_expert *out)
{
    mimo26_gpu_worker *worker = (mimo26_gpu_worker *)context;
    layer_cache *cache = &worker->caches[layer];
    for (uint16_t s = 0; s < cache->count; s++) {
        expert_slot *slot = &cache->slots[s];
        if (slot->occupied && slot->expert == expert) {
            *out = slot->view;
            return true;
        }
    }
    return false;   /* prepare_batch should have admitted it */
}

/* ---- lifecycle ---- */

void mimo26_gpu_worker_destroy(mimo26_gpu_worker *worker)
{
    if (worker == NULL) {
        return;
    }
    for (uint32_t l = 0; l < LAYERS; l++) {
        layer_cache *cache = &worker->caches[l];
        for (uint16_t s = 0; s < cache->count; s++) {
            for (size_t i = 0; i < 6; i++) {
                if (cache->slots[s].storage[i] != NULL) {
                    hipFree(cache->slots[s].storage[i]);
                }
            }
        }
        free(cache->slots);
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
    free(worker->staging_key);
    free(worker->staging_value);
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

        if (w->is_moe) {
            layer_cache *cache = &worker->caches[l];
            cache->count = config->expert_slots_per_layer;
            cache->slots =
                (expert_slot *)calloc(cache->count, sizeof *cache->slots);
            REQUIRE(cache->slots != NULL, "expert cache allocation failed");
        }
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
        if (history > 0) {
            hipMemcpy(worker->device_keys, view_keys,
                      history * w->kv_heads * QK * sizeof(uint16_t),
                      hipMemcpyHostToDevice);
            hipMemcpy(worker->device_values, view_values,
                      history * w->kv_heads * VD * sizeof(uint16_t),
                      hipMemcpyHostToDevice);
        }

        uint16_t cos_host[MIMO26_ROPE_DIM];
        uint16_t sin_host[MIMO26_ROPE_DIM];
        mimo26_attention_config attention_config;
        mimo26_attention_config_for_layer(l, &attention_config);
        mimo26_rope_table(cos_host, sin_host, position,
                          attention_config.rope_theta);
        hipMemcpy(worker->cos_table, cos_host, sizeof cos_host,
                  hipMemcpyHostToDevice);
        hipMemcpy(worker->sin_table, sin_host, sizeof sin_host,
                  hipMemcpyHostToDevice);

        mimo26_rocm_layer context;
        memset(&context, 0, sizeof context);
        context.weights = w;
        context.prepare = w->is_moe ? prepare_batch : NULL;
        context.provider = w->is_moe ? provide_expert : NULL;
        context.provider_context = worker;

        if (mimo26_rocm_layer_decode(&context, &worker->scratch,
                                     worker->hidden, worker->device_keys,
                                     worker->device_values, worker->cos_table,
                                     worker->sin_table, history,
                                     first_position, position, NULL,
                                     NULL) != MIMO26_ROCM_LAYER_OK) {
            mimo26_kv_abort(worker->kv);
            worker->stats.aborted_steps++;
            return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                        "layer %u failed", l);
        }

        /* Stage this layer's key and value back into the host-side history.
         * The commit happens once all 48 have run, so a failure above leaves
         * committed history untouched. */
        hipMemcpy(worker->staging_key, worker->scratch.key,
                  w->kv_heads * QK * sizeof(uint16_t),
                  hipMemcpyDeviceToHost);
        hipMemcpy(worker->staging_value, worker->scratch.value,
                  w->kv_heads * VD * sizeof(uint16_t),
                  hipMemcpyDeviceToHost);
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

    if (!mimo26_rocm_rmsnorm_bf16(worker->normed, worker->hidden,
                                  worker->final_norm, 1u, HIDDEN, 1e-6f,
                                  NULL)) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "final norm failed");
    }
    /* Logits in F32, summed in the CPU's order: this is the answer itself,
     * and a reassociation can flip a near-tie into a different token. */
    if (!mimo26_rocm_ordered_gemv_f32(worker->device_logits, worker->lm_head,
                                      worker->normed, VOCAB, HIDDEN, NULL)) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "lm_head failed");
    }
    if (hipMemcpy(logits, worker->device_logits,
                  (size_t)VOCAB * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "logit readback failed");
    }
    /* The padded tail decodes to no token, so it must never be sampled. */
    for (uint32_t row = MIMO26_GPU_TOKENIZER_VOCAB; row < VOCAB; row++) {
        logits[row] = -INFINITY;
    }

    worker->position++;
    struct timespec finished;
    clock_gettime(CLOCK_MONOTONIC, &finished);
    worker->stats.last_decode_seconds =
        (double)(finished.tv_sec - started.tv_sec) +
        (double)(finished.tv_nsec - started.tv_nsec) / 1e9;
    worker->stats.tokens++;
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

void mimo26_gpu_worker_reset(mimo26_gpu_worker *worker)
{
    if (worker == NULL) {
        return;
    }
    mimo26_kv_reset(worker->kv);
    worker->position = 0u;
}

void mimo26_gpu_worker_get_stats(const mimo26_gpu_worker *worker,
                                 mimo26_gpu_worker_stats *stats)
{
    if (worker != NULL && stats != NULL) {
        *stats = worker->stats;
    }
}
