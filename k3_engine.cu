#include "k3_engine.h"

#include "k3_expert_cache.h"
#include "k3_io_uring.h"
#include "k3_prefill.h"
#include "k3_rocm_ops.h"
#include "k3_safetensors.h"

#include <hip/hip_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

enum {
    K3_ENGINE_MIN_CONTEXT = 64,
    K3_ENGINE_MAX_CONTEXT = 1048576,
    K3_ENGINE_SHARDS = 96,
    K3_ENGINE_LAYERS = 93,
    K3_ENGINE_MOE_LAYERS = 92,
    K3_ENGINE_EXPERTS = 896,
    K3_ENGINE_HEADS = 96,
    K3_ENGINE_MLA_LAYERS = 24,
    K3_ENGINE_MLA_PACKED_ELEMENTS = 96 * 512 * 128,
    K3_ENGINE_STAGING_BYTES = 17551360,
    K3_ENGINE_HIDDEN = 7168,
    K3_ENGINE_KDA_INNER = 12288,
    K3_ENGINE_KDA_HEAD_DIM = 128,
    K3_ENGINE_DENSE_INTERMEDIATE = 33792,
    K3_ENGINE_SCRATCH_COUNT = 20,
    K3_ENGINE_LATENT = 3584,
    K3_ENGINE_EXPERT_HIDDEN = 3072,
    K3_ENGINE_SHARED_HIDDEN = 6144,
    K3_ENGINE_TOP_K = 16,
    K3_ENGINE_EXPERT_TENSOR_COUNT = 6,
    K3_ENGINE_STREAM_QD = 2,
    K3_ENGINE_MOE_SCRATCH_COUNT = 13,
    K3_ENGINE_MLA_Q_LORA = 1536,
    K3_ENGINE_MLA_Q_HEAD = 192,
    K3_ENGINE_MLA_Q = 96 * 192,
    K3_ENGINE_MLA_LATENT = 512,
    K3_ENGINE_MLA_CACHE_DIM = 576,
    K3_ENGINE_MLA_SCRATCH_COUNT = 8,
    K3_ENGINE_VOCAB = 163840,
};

enum {
    S_NORM,
    S_ATTENTION,
    S_MLP_INPUT,
    S_POST_NORM,
    S_MLP_OUTPUT,
    S_Q_PROJECTION,
    S_K_PROJECTION,
    S_V_PROJECTION,
    S_Q,
    S_K,
    S_V,
    S_RAW_GATE,
    S_KDA,
    S_OUTPUT_GATE,
    S_GATED,
    S_F_A,
    S_RAW_BETA,
    S_DENSE_GATE,
    S_DENSE_UP,
    S_DENSE_ACTIVATION,
};

enum {
    M_LOGITS,
    M_LATENT,
    M_EXPERT_GATE,
    M_EXPERT_UP,
    M_EXPERT_ACTIVATION,
    M_EXPERT_OUTPUTS,
    M_ROUTED,
    M_ROUTED_NORM,
    M_ROUTED_FULL,
    M_SHARED_GATE,
    M_SHARED_UP,
    M_SHARED_ACTIVATION,
    M_SHARED_OUTPUT,
};

enum {
    A_MLA_Q_A,
    A_MLA_Q_A_NORM,
    A_MLA_Q,
    A_MLA_KV,
    A_MLA_ABSORBED_Q,
    A_MLA_VALUE,
    A_MLA_GATE,
    A_MLA_GATED,
};

typedef struct {
    uint64_t relative[K3_ENGINE_EXPERT_TENSOR_COUNT];
    uint64_t physical_start;
    uint64_t aligned_start;
    uint32_t aligned_bytes;
    uint16_t shard;
} k3_engine_expert_layout;

static const uint64_t K3_ENGINE_EXPERT_BYTES =
    UINT64_C(17547264);
static const uint64_t K3_ENGINE_KDA_STATE_BYTES =
    UINT64_C(434110464);
static const uint64_t K3_ENGINE_KDA_CONV_BYTES =
    UINT64_C(20348928);
static const uint64_t K3_ENGINE_MLA_PACKED_BYTES =
    UINT64_C(301989888);
static const uint64_t K3_ENGINE_HOST_GUARD_BYTES =
    UINT64_C(4) * UINT64_C(1024) * UINT64_C(1024) *
    UINT64_C(1024);
static const uint64_t K3_ENGINE_BF16_EXTRA_GUARD_BYTES =
    UINT64_C(4) * UINT64_C(1024) * UINT64_C(1024) *
    UINT64_C(1024);

struct k3_engine {
    k3_st_model model;
    bool model_open;
    k3_static_store *static_store;
    k3_expert_cache *cache_policy;
    k3_io_uring *staging_ring;
    void **staging_host;
    void **staging_device;
    struct iovec *staging_iov;
    uint16_t staging_slots;
    uint16_t experts_per_layer;
    uint8_t *cache;
    uint64_t cache_bytes;
    void *kda_state;
    void *kda_conv;
    void *mla_cache;
    uint64_t mla_cache_bytes;
    uint8_t *mla_packed_keys;
    void *workspace;
    uint64_t workspace_bytes;
    void *scratch[K3_ENGINE_SCRATCH_COUNT];
    void *moe_scratch[K3_ENGINE_MOE_SCRATCH_COUNT];
    void *mla_scratch[K3_ENGINE_MLA_SCRATCH_COUNT];
    void *attn_res_blocks;
    void *token_buffer[2];
    uint32_t *route_ids_host;
    uint32_t *route_ids_device;
    float *route_weights_host;
    float *route_weights_device;
    void *logits;
    uint32_t *token_id_host;
    uint32_t *token_id_device;
    float *token_value_host;
    float *token_value_device;
    hipStream_t expert_stream;
    hipStream_t shared_stream;
    k3_rocm_blas_context *blas;
    uint32_t decoded_layers;
    uint32_t token_position;
    uint32_t context;
    uint64_t model_layout_crc64;
    bool q8_projections;
    bool causal_state_valid;
};

static uint64_t engine_model_layout_crc64(
    const k3_st_model *model);

static void engine_error(char *error,
                         size_t error_size,
                         const char *format,
                         ...) {
    if (!error || error_size == 0) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double elapsed_seconds(struct timespec start,
                              struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static bool host_memory_bytes(uint64_t *available,
                              uint64_t *cma_free) {
    *available = 0;
    *cma_free = 0;
    FILE *source = fopen("/proc/meminfo", "r");
    if (!source) return false;
    char key[64];
    unsigned long long value = 0;
    char unit[32];
    while (fscanf(
               source, "%63s %llu %31s",
               key, &value, unit) == 3) {
        const uint64_t bytes =
            value > UINT64_MAX / 1024u ?
                UINT64_MAX : (uint64_t)value * 1024u;
        if (strcmp(key, "MemAvailable:") == 0) {
            *available = bytes;
        } else if (strcmp(key, "CmaFree:") == 0) {
            *cma_free = bytes;
        }
    }
    fclose(source);
    return *available != 0u;
}

static bool add_memory_bytes(uint64_t *total,
                             uint64_t bytes) {
    if (*total > UINT64_MAX - bytes) return false;
    *total += bytes;
    return true;
}

static bool context_memory_bytes(
        uint32_t context,
        uint64_t *mla_cache_bytes,
        uint64_t *workspace_bytes) {
    if (context < K3_ENGINE_MIN_CONTEXT ||
        context > K3_ENGINE_MAX_CONTEXT) {
        return false;
    }
    const uint64_t per_token_cache =
        (uint64_t)K3_ENGINE_MLA_LAYERS *
        K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t);
    const uint64_t per_token_workspace =
        (uint64_t)K3_ENGINE_HEADS *
        (sizeof(float) + sizeof(uint16_t));
    const uint64_t latent_workspace =
        (uint64_t)K3_ENGINE_HEADS *
        K3_ENGINE_MLA_LATENT * sizeof(uint16_t);
    if ((uint64_t)context >
            UINT64_MAX / per_token_cache ||
        (uint64_t)context >
            (UINT64_MAX - latent_workspace) /
                per_token_workspace) {
        return false;
    }
    *mla_cache_bytes =
        (uint64_t)context * per_token_cache;
    *workspace_bytes =
        (uint64_t)context * per_token_workspace +
        latent_workspace;
    return true;
}

static bool residency_preflight(
        const k3_st_model *model,
        bool q8_projections,
        uint64_t cache_bytes,
        uint64_t mla_cache_bytes,
        uint64_t workspace_bytes,
        uint16_t staging_slots,
        char *error,
        size_t error_size) {
    k3_static_store_stats static_plan;
    if (!k3_static_store_plan(
            model, q8_projections, &static_plan,
            error, error_size)) {
        return false;
    }
    uint64_t available = 0;
    uint64_t cma_free = 0;
    if (!host_memory_bytes(&available, &cma_free)) {
        engine_error(
            error, error_size,
            "cannot read /proc/meminfo for K3 residency preflight");
        return false;
    }
    const uint64_t state_bytes =
        K3_ENGINE_KDA_STATE_BYTES +
        K3_ENGINE_KDA_CONV_BYTES +
        mla_cache_bytes +
        K3_ENGINE_MLA_PACKED_BYTES +
        workspace_bytes;
    const uint64_t staging_bytes =
        (uint64_t)staging_slots * K3_ENGINE_STAGING_BYTES;
    const uint64_t guard_bytes =
        K3_ENGINE_HOST_GUARD_BYTES +
        (q8_projections ?
            0u : K3_ENGINE_BF16_EXTRA_GUARD_BYTES);
    uint64_t load_required = static_plan.resident_bytes;
    uint64_t runtime_required = static_plan.resident_bytes;
    if (!add_memory_bytes(
            &load_required, static_plan.peak_transient_bytes) ||
        !add_memory_bytes(&load_required, cma_free) ||
        !add_memory_bytes(&load_required, guard_bytes) ||
        !add_memory_bytes(&runtime_required, cache_bytes) ||
        !add_memory_bytes(&runtime_required, state_bytes) ||
        !add_memory_bytes(&runtime_required, staging_bytes) ||
        !add_memory_bytes(&runtime_required, cma_free) ||
        !add_memory_bytes(&runtime_required, guard_bytes)) {
        engine_error(error, error_size,
                     "K3 residency preflight byte count overflow");
        return false;
    }
    const uint64_t required =
        load_required > runtime_required ?
            load_required : runtime_required;
    if (available < required) {
        engine_error(
            error, error_size,
            "K3 %s residency preflight rejected: "
            "MemAvailable %.3f GiB < required %.3f GiB "
            "(load %.3f, runtime %.3f; "
            "static %.3f + peak transient %.3f + cache %.3f + "
            "runtime %.3f + CMA reserve %.3f + guard %.3f GiB)",
            q8_projections ? "Q8" : "BF16",
            (double)available / 1073741824.0,
            (double)required / 1073741824.0,
            (double)load_required / 1073741824.0,
            (double)runtime_required / 1073741824.0,
            (double)static_plan.resident_bytes / 1073741824.0,
            (double)static_plan.peak_transient_bytes / 1073741824.0,
            (double)cache_bytes / 1073741824.0,
            (double)(state_bytes + staging_bytes) / 1073741824.0,
            (double)cma_free / 1073741824.0,
            (double)guard_bytes / 1073741824.0);
        return false;
    }
    return true;
}

static bool hip_allocate_zero(void **pointer,
                              uint64_t bytes,
                              const char *label,
                              char *error,
                              size_t error_size) {
    hipError_t status = hipMalloc(pointer, bytes);
    if (status == hipSuccess) {
        status = hipMemset(*pointer, 0, bytes);
    }
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "%s allocation failed: %s",
                     label, hipGetErrorString(status));
        if (*pointer) {
            (void)hipFree(*pointer);
            *pointer = NULL;
        }
        return false;
    }
    return true;
}

extern "C" void k3_engine_destroy(k3_engine *engine) {
    if (!engine) return;
    if (engine->expert_stream) {
        (void)hipStreamDestroy(engine->expert_stream);
    }
    if (engine->shared_stream) {
        (void)hipStreamDestroy(engine->shared_stream);
    }
    if (engine->route_weights_host) {
        (void)hipHostFree(engine->route_weights_host);
    }
    if (engine->route_ids_host) {
        (void)hipHostFree(engine->route_ids_host);
    }
    if (engine->token_value_host) {
        (void)hipHostFree(engine->token_value_host);
    }
    if (engine->token_id_host) {
        (void)hipHostFree(engine->token_id_host);
    }
    if (engine->logits) (void)hipFree(engine->logits);
    k3_rocm_blas_context_destroy(engine->blas);
    if (engine->staging_ring) {
        k3_io_uring_destroy(engine->staging_ring);
    }
    if (engine->staging_host) {
        for (uint16_t i = 0; i < engine->staging_slots; i++) {
            if (engine->staging_host[i]) {
                (void)hipHostFree(engine->staging_host[i]);
            }
        }
    }
    free(engine->staging_iov);
    free(engine->staging_device);
    free(engine->staging_host);
    if (engine->workspace) (void)hipFree(engine->workspace);
    for (uint32_t i = 0; i < 2u; i++) {
        if (engine->token_buffer[i]) {
            (void)hipFree(engine->token_buffer[i]);
        }
    }
    if (engine->attn_res_blocks) {
        (void)hipFree(engine->attn_res_blocks);
    }
    for (uint32_t i = 0; i < K3_ENGINE_SCRATCH_COUNT; i++) {
        if (engine->scratch[i]) (void)hipFree(engine->scratch[i]);
    }
    for (uint32_t i = 0; i < K3_ENGINE_MOE_SCRATCH_COUNT; i++) {
        if (engine->moe_scratch[i]) {
            (void)hipFree(engine->moe_scratch[i]);
        }
    }
    for (uint32_t i = 0; i < K3_ENGINE_MLA_SCRATCH_COUNT; i++) {
        if (engine->mla_scratch[i]) {
            (void)hipFree(engine->mla_scratch[i]);
        }
    }
    if (engine->mla_packed_keys) {
        (void)hipFree(engine->mla_packed_keys);
    }
    if (engine->mla_cache) (void)hipFree(engine->mla_cache);
    if (engine->kda_conv) (void)hipFree(engine->kda_conv);
    if (engine->kda_state) (void)hipFree(engine->kda_state);
    if (engine->cache) (void)hipFree(engine->cache);
    k3_expert_cache_destroy(engine->cache_policy);
    k3_static_store_destroy(engine->static_store);
    if (engine->model_open) k3_st_model_close(&engine->model);
    free(engine);
}

static bool pack_mla_keys(k3_engine *engine,
                          double *seconds,
                          char *error,
                          size_t error_size) {
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    uint32_t packed_index = 0;
    for (uint32_t layer = 3u;
         layer < K3_ENGINE_LAYERS; layer += 4u) {
        char name[256];
        snprintf(name, sizeof(name),
                 "language_model.model.layers.%u."
                 "self_attn.kv_b_proj.weight", layer);
        const k3_static_weight *weight =
            k3_static_store_find(engine->static_store, name);
        if (!weight ||
            weight->kind != K3_STATIC_WEIGHT_BF16) {
            engine_error(error, error_size,
                         "missing BF16 MLA kv_b for layer %u",
                         layer);
            return false;
        }
        void *destination =
            engine->mla_packed_keys +
            (uint64_t)packed_index *
                K3_ENGINE_MLA_PACKED_ELEMENTS *
                sizeof(uint16_t);
        if (!k3_rocm_mla_pack_k_weight_bf16(
                destination, weight->data,
                K3_ENGINE_HEADS, NULL)) {
            engine_error(error, error_size,
                         "MLA key pack launch failed at layer %u",
                         layer);
            return false;
        }
        packed_index++;
    }
    if (packed_index != 23u) {
        engine_error(error, error_size,
                     "expected 23 stride-4 MLA layers, got %u",
                     packed_index);
        return false;
    }
    const uint32_t final_layer = 92u;
    const k3_static_weight *final_weight =
        k3_static_store_find(
            engine->static_store,
            "language_model.model.layers.92."
            "self_attn.kv_b_proj.weight");
    if (!final_weight ||
        final_weight->kind != K3_STATIC_WEIGHT_BF16 ||
        !k3_rocm_mla_pack_k_weight_bf16(
            engine->mla_packed_keys +
                (uint64_t)packed_index *
                    K3_ENGINE_MLA_PACKED_ELEMENTS *
                    sizeof(uint16_t),
            final_weight->data, K3_ENGINE_HEADS, NULL)) {
        engine_error(error, error_size,
                     "MLA key pack launch failed at layer %u",
                     final_layer);
        return false;
    }
    packed_index++;
    hipError_t status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "MLA key pack synchronization failed: %s",
                     hipGetErrorString(status));
        return false;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (seconds) *seconds = elapsed_seconds(start, end);
    return packed_index == K3_ENGINE_MLA_LAYERS;
}

static bool allocate_decode_scratch(k3_engine *engine,
                                    char *error,
                                    size_t error_size) {
    const uint64_t hidden_bytes =
        (uint64_t)K3_ENGINE_HIDDEN * sizeof(uint16_t);
    const uint64_t inner_bytes =
        (uint64_t)K3_ENGINE_KDA_INNER * sizeof(uint16_t);
    const uint64_t dense_bytes =
        (uint64_t)K3_ENGINE_DENSE_INTERMEDIATE * sizeof(uint16_t);
    const uint64_t sizes[K3_ENGINE_SCRATCH_COUNT] = {
        hidden_bytes, hidden_bytes, hidden_bytes,
        hidden_bytes, hidden_bytes,
        inner_bytes, inner_bytes, inner_bytes,
        inner_bytes, inner_bytes, inner_bytes,
        inner_bytes, inner_bytes, inner_bytes, inner_bytes,
        (uint64_t)K3_ENGINE_KDA_HEAD_DIM * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_HEADS * sizeof(uint16_t),
        dense_bytes, dense_bytes, dense_bytes,
    };
    hipError_t status = hipSuccess;
    for (uint32_t i = 0; i < K3_ENGINE_SCRATCH_COUNT; i++) {
        status = hipMalloc(&engine->scratch[i], sizes[i]);
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "decode scratch %u allocation failed: %s",
                         i, hipGetErrorString(status));
            return false;
        }
    }
    for (uint32_t i = 0; i < 2u; i++) {
        status = hipMalloc(
            &engine->token_buffer[i],
            (uint64_t)K3_ENGINE_HIDDEN * sizeof(uint16_t));
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "token buffer %u allocation failed: %s",
                         i, hipGetErrorString(status));
            return false;
        }
    }
    status = hipMalloc(
        &engine->attn_res_blocks, 8u * hidden_bytes);
    if (status == hipSuccess) {
        status = hipMemset(
            engine->attn_res_blocks, 0, 8u * hidden_bytes);
    }
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "AttnRes block allocation failed: %s",
                     hipGetErrorString(status));
        return false;
    }
    const uint64_t moe_sizes[K3_ENGINE_MOE_SCRATCH_COUNT] = {
        (uint64_t)K3_ENGINE_EXPERTS * sizeof(float),
        (uint64_t)K3_ENGINE_LATENT * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_EXPERT_HIDDEN * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_EXPERT_HIDDEN * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_EXPERT_HIDDEN * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_TOP_K * K3_ENGINE_LATENT * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_LATENT * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_LATENT * sizeof(uint16_t),
        hidden_bytes,
        (uint64_t)K3_ENGINE_SHARED_HIDDEN * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_SHARED_HIDDEN * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_SHARED_HIDDEN * sizeof(uint16_t),
        hidden_bytes,
    };
    for (uint32_t i = 0; i < K3_ENGINE_MOE_SCRATCH_COUNT; i++) {
        status = hipMalloc(&engine->moe_scratch[i], moe_sizes[i]);
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "MoE scratch %u allocation failed: %s",
                         i, hipGetErrorString(status));
            return false;
        }
    }
    const uint64_t mla_sizes[K3_ENGINE_MLA_SCRATCH_COUNT] = {
        (uint64_t)K3_ENGINE_MLA_Q_LORA * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_MLA_Q_LORA * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_MLA_Q * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_HEADS *
            K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_KDA_INNER * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_KDA_INNER * sizeof(uint16_t),
        (uint64_t)K3_ENGINE_KDA_INNER * sizeof(uint16_t),
    };
    for (uint32_t i = 0; i < K3_ENGINE_MLA_SCRATCH_COUNT; i++) {
        status = hipMalloc(&engine->mla_scratch[i], mla_sizes[i]);
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "MLA scratch %u allocation failed: %s",
                         i, hipGetErrorString(status));
            return false;
        }
    }
    status = hipHostMalloc(
        (void **)&engine->route_ids_host,
        K3_ENGINE_TOP_K * sizeof(*engine->route_ids_host),
        hipHostMallocMapped);
    if (status == hipSuccess) {
        status = hipHostGetDevicePointer(
            (void **)&engine->route_ids_device,
            engine->route_ids_host, 0);
    }
    if (status == hipSuccess) {
        status = hipHostMalloc(
            (void **)&engine->route_weights_host,
            K3_ENGINE_TOP_K * sizeof(*engine->route_weights_host),
            hipHostMallocMapped);
    }
    if (status == hipSuccess) {
        status = hipHostGetDevicePointer(
            (void **)&engine->route_weights_device,
            engine->route_weights_host, 0);
    }
    if (status == hipSuccess) {
        status = hipMalloc(
            &engine->logits,
            (uint64_t)K3_ENGINE_VOCAB * sizeof(uint16_t));
    }
    if (status == hipSuccess) {
        status = hipHostMalloc(
            (void **)&engine->token_id_host,
            sizeof(*engine->token_id_host), hipHostMallocMapped);
    }
    if (status == hipSuccess) {
        status = hipHostGetDevicePointer(
            (void **)&engine->token_id_device,
            engine->token_id_host, 0);
    }
    if (status == hipSuccess) {
        status = hipHostMalloc(
            (void **)&engine->token_value_host,
            sizeof(*engine->token_value_host), hipHostMallocMapped);
    }
    if (status == hipSuccess) {
        status = hipHostGetDevicePointer(
            (void **)&engine->token_value_device,
            engine->token_value_host, 0);
    }
    if (status == hipSuccess) {
        status = hipStreamCreateWithFlags(
            &engine->expert_stream, hipStreamNonBlocking);
    }
    if (status == hipSuccess) {
        status = hipStreamCreateWithFlags(
            &engine->shared_stream, hipStreamNonBlocking);
    }
    if (status == hipSuccess &&
        !k3_rocm_blas_context_create(&engine->blas)) {
        status = hipErrorUnknown;
    }
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "MoE scheduler allocation failed: %s",
                     hipGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool k3_engine_create(
        k3_engine       **out,
        const char       *model_root,
        uint32_t          context,
        uint16_t          experts_per_layer,
        uint16_t          staging_slots,
        bool              q8_projections,
        k3_engine_stats  *stats,
        char             *error,
        size_t            error_size) {
    if (error && error_size) error[0] = '\0';
    if (out) *out = NULL;
    if (stats) memset(stats, 0, sizeof(*stats));
    uint64_t mla_cache_bytes = 0u;
    uint64_t workspace_bytes = 0u;
    if (!out || !model_root ||
        !context_memory_bytes(
            context, &mla_cache_bytes, &workspace_bytes) ||
        experts_per_layer == 0u || staging_slots == 0u ||
        staging_slots > UINT16_MAX) {
        engine_error(error, error_size,
                     "invalid K3 engine configuration");
        return false;
    }
    struct timespec startup_start;
    struct timespec startup_end;
    clock_gettime(CLOCK_MONOTONIC, &startup_start);
    k3_engine *engine =
        (k3_engine *)calloc(1, sizeof(*engine));
    if (!engine) {
        engine_error(error, error_size,
                     "K3 engine allocation failed");
        return false;
    }
    engine->experts_per_layer = experts_per_layer;
    engine->staging_slots = staging_slots;
    engine->context = context;
    engine->mla_cache_bytes = mla_cache_bytes;
    engine->workspace_bytes = workspace_bytes;
    engine->q8_projections = q8_projections;
    if (!k3_st_model_open(
            &engine->model, model_root, K3_ENGINE_SHARDS,
            error, error_size)) {
        k3_engine_destroy(engine);
        return false;
    }
    engine->model_open = true;
    engine->model_layout_crc64 =
        engine_model_layout_crc64(&engine->model);

    const uint64_t cache_slots =
        (uint64_t)K3_ENGINE_MOE_LAYERS *
        experts_per_layer;
    if (cache_slots > UINT32_MAX ||
        cache_slots > UINT64_MAX / K3_ENGINE_EXPERT_BYTES) {
        engine_error(error, error_size,
                     "K3 expert cache dimensions overflow");
        k3_engine_destroy(engine);
        return false;
    }
    engine->cache_bytes =
        cache_slots * K3_ENGINE_EXPERT_BYTES;
    if (!residency_preflight(
            &engine->model, q8_projections,
            engine->cache_bytes, engine->mla_cache_bytes,
            engine->workspace_bytes, staging_slots,
            error, error_size)) {
        k3_engine_destroy(engine);
        return false;
    }

    k3_engine_stats measured;
    memset(&measured, 0, sizeof(measured));
    if (!k3_static_store_load(
            &engine->static_store, &engine->model,
            q8_projections, &measured.static_store,
            error, error_size)) {
        k3_engine_destroy(engine);
        return false;
    }

    hipError_t status =
        hipMalloc(&engine->cache, engine->cache_bytes);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "K3 device cache allocation failed: %s",
                     hipGetErrorString(status));
        k3_engine_destroy(engine);
        return false;
    }
    if (!k3_expert_cache_create(
            &engine->cache_policy,
            K3_ENGINE_MOE_LAYERS,
            experts_per_layer,
            error, error_size)) {
        k3_engine_destroy(engine);
        return false;
    }

    if (!hip_allocate_zero(
            &engine->kda_state, K3_ENGINE_KDA_STATE_BYTES,
            "KDA state", error, error_size) ||
        !hip_allocate_zero(
            &engine->kda_conv, K3_ENGINE_KDA_CONV_BYTES,
            "KDA convolution state", error, error_size) ||
        !hip_allocate_zero(
            &engine->mla_cache, engine->mla_cache_bytes,
            "MLA token cache", error, error_size) ||
        !hip_allocate_zero(
            (void **)&engine->mla_packed_keys,
            K3_ENGINE_MLA_PACKED_BYTES,
            "MLA packed keys", error, error_size) ||
        !hip_allocate_zero(
            &engine->workspace, engine->workspace_bytes,
            "K3 workspace", error, error_size)) {
        k3_engine_destroy(engine);
        return false;
    }

    engine->staging_host = (void **)calloc(
        staging_slots, sizeof(*engine->staging_host));
    engine->staging_device = (void **)calloc(
        staging_slots, sizeof(*engine->staging_device));
    engine->staging_iov = (struct iovec *)calloc(
        staging_slots, sizeof(*engine->staging_iov));
    if (!engine->staging_host ||
        !engine->staging_device ||
        !engine->staging_iov) {
        engine_error(error, error_size,
                     "K3 staging table allocation failed");
        k3_engine_destroy(engine);
        return false;
    }
    for (uint16_t slot = 0; slot < staging_slots; slot++) {
        status = hipHostMalloc(
            &engine->staging_host[slot],
            K3_ENGINE_STAGING_BYTES,
            hipHostMallocMapped);
        if (status == hipSuccess) {
            status = hipHostGetDevicePointer(
                &engine->staging_device[slot],
                engine->staging_host[slot], 0);
        }
        if (status != hipSuccess ||
            (uintptr_t)engine->staging_host[slot] % 4096u != 0u) {
            engine_error(error, error_size,
                         "K3 staging slot %u failed: %s",
                         slot, hipGetErrorString(status));
            k3_engine_destroy(engine);
            return false;
        }
        engine->staging_iov[slot].iov_base =
            engine->staging_host[slot];
        engine->staging_iov[slot].iov_len =
            K3_ENGINE_STAGING_BYTES;
    }
    if (!k3_io_uring_create(
            &engine->staging_ring,
            engine->staging_iov, staging_slots,
            error, error_size)) {
        k3_engine_destroy(engine);
        return false;
    }
    if (!pack_mla_keys(
            engine, &measured.mla_pack_seconds,
            error, error_size)) {
        k3_engine_destroy(engine);
        return false;
    }
    if (!allocate_decode_scratch(
            engine, error, error_size)) {
        k3_engine_destroy(engine);
        return false;
    }
    clock_gettime(CLOCK_MONOTONIC, &startup_end);
    measured.cache_slots = (uint32_t)cache_slots;
    measured.cache_bytes = engine->cache_bytes;
    measured.state_bytes =
        K3_ENGINE_KDA_STATE_BYTES +
        K3_ENGINE_KDA_CONV_BYTES +
        engine->mla_cache_bytes +
        K3_ENGINE_MLA_PACKED_BYTES +
        engine->workspace_bytes;
    measured.staging_slots = staging_slots;
    measured.staging_bytes =
        (uint64_t)staging_slots *
        K3_ENGINE_STAGING_BYTES;
    measured.startup_seconds =
        elapsed_seconds(startup_start, startup_end);
    engine->causal_state_valid = true;
    if (stats) *stats = measured;
    *out = engine;
    return true;
}

extern "C" const k3_static_weight *k3_engine_find_weight(
        const k3_engine *engine,
        const char *name) {
    return engine ?
        k3_static_store_find(engine->static_store, name) :
        NULL;
}

extern "C" void *k3_engine_cache_slot(
        k3_engine *engine,
        uint32_t global_slot) {
    if (!engine ||
        global_slot >=
            (uint32_t)K3_ENGINE_MOE_LAYERS *
                engine->experts_per_layer) {
        return NULL;
    }
    return engine->cache +
        (uint64_t)global_slot * K3_ENGINE_EXPERT_BYTES;
}

extern "C" void *k3_engine_staging_host(
        k3_engine *engine,
        uint16_t slot) {
    return engine && slot < engine->staging_slots ?
        engine->staging_host[slot] : NULL;
}

extern "C" void *k3_engine_staging_device(
        k3_engine *engine,
        uint16_t slot) {
    return engine && slot < engine->staging_slots ?
        engine->staging_device[slot] : NULL;
}

static const k3_static_weight *required_weight(
        k3_engine *engine,
        const char *name,
        char *error,
        size_t error_size) {
    const k3_static_weight *weight =
        k3_static_store_find(engine->static_store, name);
    if (!weight) {
        engine_error(error, error_size,
                     "missing engine weight %s", name);
    }
    return weight;
}

static const k3_static_weight *required_layer_weight(
        k3_engine *engine,
        uint32_t layer,
        const char *suffix,
        char *error,
        size_t error_size) {
    char name[256];
    int length = snprintf(
        name, sizeof(name),
        "language_model.model.layers.%u.%s", layer, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        engine_error(error, error_size,
                     "layer-%u weight name overflow", layer);
        return NULL;
    }
    return required_weight(engine, name, error, error_size);
}

static bool find_expert_layout(
        k3_engine *engine,
        uint32_t layer,
        uint32_t expert,
        k3_engine_expert_layout *layout,
        char *error,
        size_t error_size) {
    static const char *suffix[K3_ENGINE_EXPERT_TENSOR_COUNT] = {
        "w1.weight_packed", "w1.weight_scale",
        "w2.weight_packed", "w2.weight_scale",
        "w3.weight_packed", "w3.weight_scale",
    };
    const k3_st_tensor *tensor[K3_ENGINE_EXPERT_TENSOR_COUNT];
    char name[256];
    memset(layout, 0, sizeof(*layout));
    for (uint32_t i = 0;
         i < K3_ENGINE_EXPERT_TENSOR_COUNT; i++) {
        int length = snprintf(
            name, sizeof(name),
            "language_model.model.layers.%u.block_sparse_moe."
            "experts.%u.%s", layer, expert, suffix[i]);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            engine_error(error, error_size,
                         "layer-%u expert-%u name overflow",
                         layer, expert);
            return false;
        }
        tensor[i] = k3_st_find(&engine->model, name);
        if (!tensor[i]) {
            engine_error(error, error_size,
                         "missing layer-%u expert-%u tensor %s",
                         layer, expert, suffix[i]);
            return false;
        }
        if (i > 0u && tensor[i]->shard != tensor[0]->shard) {
            engine_error(error, error_size,
                         "layer-%u expert-%u crosses shards",
                         layer, expert);
            return false;
        }
    }
    const uint64_t start = tensor[0]->physical_offset;
    const uint64_t end =
        tensor[K3_ENGINE_EXPERT_TENSOR_COUNT - 1u]->physical_offset +
        tensor[K3_ENGINE_EXPERT_TENSOR_COUNT - 1u]->byte_length;
    if (end < start || end - start != K3_ENGINE_EXPERT_BYTES) {
        engine_error(error, error_size,
                     "layer-%u expert-%u has invalid physical span",
                     layer, expert);
        return false;
    }
    layout->physical_start = start;
    layout->aligned_start = start & ~UINT64_C(4095);
    const uint64_t aligned_end =
        (end + UINT64_C(4095)) & ~UINT64_C(4095);
    if (aligned_end < layout->aligned_start ||
        aligned_end - layout->aligned_start >
            K3_ENGINE_STAGING_BYTES) {
        engine_error(error, error_size,
                     "layer-%u expert-%u exceeds staging slot",
                     layer, expert);
        return false;
    }
    layout->aligned_bytes =
        (uint32_t)(aligned_end - layout->aligned_start);
    layout->shard = tensor[0]->shard;
    for (uint32_t i = 0;
         i < K3_ENGINE_EXPERT_TENSOR_COUNT; i++) {
        if (tensor[i]->physical_offset < start ||
            tensor[i]->physical_offset + tensor[i]->byte_length > end) {
            engine_error(error, error_size,
                         "layer-%u expert-%u tensor span is invalid",
                         layer, expert);
            return false;
        }
        layout->relative[i] =
            tensor[i]->physical_offset - start;
    }
    return true;
}

static bool launch_expert(
        k3_engine *engine,
        uint32_t rank,
        uint8_t *base,
        const k3_engine_expert_layout *layout) {
    return
        k3_rocm_mxfp4_gemv_bf16(
            engine->moe_scratch[M_EXPERT_GATE],
            base + layout->relative[0],
            base + layout->relative[1],
            engine->moe_scratch[M_LATENT],
            K3_ENGINE_EXPERT_HIDDEN, K3_ENGINE_LATENT,
            engine->expert_stream) &&
        k3_rocm_mxfp4_gemv_bf16(
            engine->moe_scratch[M_EXPERT_UP],
            base + layout->relative[4],
            base + layout->relative[5],
            engine->moe_scratch[M_LATENT],
            K3_ENGINE_EXPERT_HIDDEN, K3_ENGINE_LATENT,
            engine->expert_stream) &&
        k3_rocm_situ_bf16(
            engine->moe_scratch[M_EXPERT_ACTIVATION],
            engine->moe_scratch[M_EXPERT_GATE],
            engine->moe_scratch[M_EXPERT_UP],
            K3_ENGINE_EXPERT_HIDDEN, 4.0f, 25.0f,
            engine->expert_stream) &&
        k3_rocm_mxfp4_gemv_bf16(
            (uint8_t *)engine->moe_scratch[M_EXPERT_OUTPUTS] +
                (uint64_t)rank * K3_ENGINE_LATENT * sizeof(uint16_t),
            base + layout->relative[2],
            base + layout->relative[3],
            engine->moe_scratch[M_EXPERT_ACTIVATION],
            K3_ENGINE_LATENT, K3_ENGINE_EXPERT_HIDDEN,
            engine->expert_stream);
}

static k3_io_request make_expert_request(
        const k3_engine *engine,
        uint32_t rank,
        const k3_engine_expert_layout *layout) {
    k3_io_request request;
    request.fd = engine->model.shards[layout->shard].direct_fd;
    request.offset = layout->aligned_start;
    request.bytes = layout->aligned_bytes;
    request.buffer_index = (uint16_t)rank;
    request.user_data = rank;
    return request;
}

static void abort_layer_moe(k3_engine *engine, uint16_t cache_layer) {
    char discard[128];
    while (k3_io_uring_outstanding(engine->staging_ring) > 0u) {
        k3_io_completion completions[K3_ENGINE_STREAM_QD];
        uint16_t completion_count = 0;
        if (!k3_io_uring_wait(
                engine->staging_ring, completions,
                K3_ENGINE_STREAM_QD, &completion_count,
                discard, sizeof(discard))) {
            break;
        }
    }
    (void)hipStreamSynchronize(engine->expert_stream);
    (void)hipStreamSynchronize(engine->shared_stream);
    k3_expert_cache_abort(engine->cache_policy, cache_layer);
}

static bool launch_kda_attention(
        k3_engine *engine,
        uint32_t layer,
        const void *normalized,
        void *attention,
        char *error,
        size_t error_size) {
#define KDA_WEIGHT(variable, suffix)                                        \
    const k3_static_weight *variable = required_layer_weight(               \
        engine, layer, suffix, error, error_size);                           \
    if (!variable) return false
    KDA_WEIGHT(q_proj, "self_attn.q_proj.weight");
    KDA_WEIGHT(k_proj, "self_attn.k_proj.weight");
    KDA_WEIGHT(v_proj, "self_attn.v_proj.weight");
    KDA_WEIGHT(q_conv, "self_attn.q_conv1d.weight");
    KDA_WEIGHT(k_conv, "self_attn.k_conv1d.weight");
    KDA_WEIGHT(v_conv, "self_attn.v_conv1d.weight");
    KDA_WEIGHT(f_a, "self_attn.f_a_proj.weight");
    KDA_WEIGHT(f_b, "self_attn.f_b_proj.weight");
    KDA_WEIGHT(b_proj, "self_attn.b_proj.weight");
    KDA_WEIGHT(a_log, "self_attn.A_log");
    KDA_WEIGHT(dt_bias, "self_attn.dt_bias");
    KDA_WEIGHT(g_proj, "self_attn.g_proj.weight");
    KDA_WEIGHT(o_norm, "self_attn.o_norm.weight");
    KDA_WEIGHT(o_proj, "self_attn.o_proj.weight");
#undef KDA_WEIGHT
    const uint32_t kda_index = layer - layer / 4u;
    const uint64_t conv_vector_bytes =
        (uint64_t)K3_ENGINE_KDA_INNER * 4u * sizeof(uint16_t);
    const uint64_t conv_layer_bytes = 3u * conv_vector_bytes;
    const uint64_t state_layer_bytes =
        (uint64_t)K3_ENGINE_HEADS *
        K3_ENGINE_KDA_HEAD_DIM *
        K3_ENGINE_KDA_HEAD_DIM * sizeof(float);
    uint8_t *conv = (uint8_t *)engine->kda_conv +
        (uint64_t)kda_index * conv_layer_bytes;
    uint8_t *state = (uint8_t *)engine->kda_state +
        (uint64_t)kda_index * state_layer_bytes;
    if (!k3_static_weight_gemv_bf16(
            q_proj, engine->scratch[S_Q_PROJECTION],
            normalized, NULL) ||
        !k3_static_weight_gemv_bf16(
            k_proj, engine->scratch[S_K_PROJECTION],
            normalized, NULL) ||
        !k3_static_weight_gemv_bf16(
            v_proj, engine->scratch[S_V_PROJECTION],
            normalized, NULL) ||
        !k3_rocm_short_conv4_silu_bf16_f32_weight(
            engine->scratch[S_Q], conv,
            engine->scratch[S_Q_PROJECTION],
            q_conv->data, K3_ENGINE_KDA_INNER, NULL) ||
        !k3_rocm_short_conv4_silu_bf16_f32_weight(
            engine->scratch[S_K], conv + conv_vector_bytes,
            engine->scratch[S_K_PROJECTION],
            k_conv->data, K3_ENGINE_KDA_INNER, NULL) ||
        !k3_rocm_short_conv4_silu_bf16_f32_weight(
            engine->scratch[S_V], conv + 2u * conv_vector_bytes,
            engine->scratch[S_V_PROJECTION],
            v_conv->data, K3_ENGINE_KDA_INNER, NULL) ||
        !k3_static_weight_gemv_bf16(
            f_a, engine->scratch[S_F_A], normalized, NULL) ||
        !k3_static_weight_gemv_bf16(
            f_b, engine->scratch[S_RAW_GATE],
            engine->scratch[S_F_A], NULL) ||
        !k3_static_weight_gemv_bf16(
            b_proj, engine->scratch[S_RAW_BETA],
            normalized, NULL) ||
        !k3_rocm_kda_recurrent_bf16_f32_state(
            engine->scratch[S_KDA], state,
            engine->scratch[S_Q], engine->scratch[S_K],
            engine->scratch[S_V], engine->scratch[S_RAW_GATE],
            engine->scratch[S_RAW_BETA], a_log->data, dt_bias->data,
            K3_ENGINE_HEADS, K3_ENGINE_KDA_HEAD_DIM,
            -5.0f, NULL) ||
        !k3_static_weight_gemv_bf16(
            g_proj, engine->scratch[S_OUTPUT_GATE],
            normalized, NULL) ||
        !k3_rocm_rms_norm_sigmoid_gate_bf16_f32_weight(
            engine->scratch[S_GATED], engine->scratch[S_KDA],
            engine->scratch[S_OUTPUT_GATE], o_norm->data,
            K3_ENGINE_HEADS, K3_ENGINE_KDA_HEAD_DIM,
            1e-5f, NULL) ||
        !k3_static_weight_gemv_bf16(
            o_proj, attention, engine->scratch[S_GATED], NULL)) {
        engine_error(error, error_size,
                     "layer-%u KDA launch failed", layer);
        return false;
    }
    return true;
}

static bool launch_mla_attention(
        k3_engine *engine,
        uint32_t layer,
        const void *normalized,
        void *attention,
        char *error,
        size_t error_size) {
#define MLA_WEIGHT(variable, suffix)                                        \
    const k3_static_weight *variable = required_layer_weight(               \
        engine, layer, suffix, error, error_size);                           \
    if (!variable) return false
    MLA_WEIGHT(q_a, "self_attn.q_a_proj.weight");
    MLA_WEIGHT(q_a_norm, "self_attn.q_a_layernorm.weight");
    MLA_WEIGHT(q_b, "self_attn.q_b_proj.weight");
    MLA_WEIGHT(kv_a, "self_attn.kv_a_proj_with_mqa.weight");
    MLA_WEIGHT(kv_a_norm, "self_attn.kv_a_layernorm.weight");
    MLA_WEIGHT(kv_b, "self_attn.kv_b_proj.weight");
    MLA_WEIGHT(gate, "self_attn.g_proj.weight");
    MLA_WEIGHT(o_proj, "self_attn.o_proj.weight");
#undef MLA_WEIGHT
    const uint32_t mla_index =
        layer == 92u ? 23u : layer / 4u;
    const uint64_t cache_layer_bytes =
        (uint64_t)engine->context * K3_ENGINE_MLA_CACHE_DIM *
        sizeof(uint16_t);
    uint8_t *cache = (uint8_t *)engine->mla_cache +
        (uint64_t)mla_index * cache_layer_bytes;
    uint16_t *cache_row =
        (uint16_t *)cache +
        (uint64_t)engine->token_position *
            K3_ENGINE_MLA_CACHE_DIM;
    uint8_t *packed_key =
        engine->mla_packed_keys +
        (uint64_t)mla_index *
            K3_ENGINE_MLA_PACKED_ELEMENTS *
            sizeof(uint16_t);
    uint8_t *workspace = (uint8_t *)engine->workspace;
    void *scores = workspace;
    void *probabilities = workspace +
        (uint64_t)K3_ENGINE_HEADS *
            engine->context * sizeof(float);
    void *latent = workspace +
        (uint64_t)K3_ENGINE_HEADS * engine->context *
            (sizeof(float) + sizeof(uint16_t));
    if (!k3_static_weight_gemv_bf16(
            q_a, engine->mla_scratch[A_MLA_Q_A],
            normalized, NULL) ||
        !k3_rocm_rms_norm_bf16(
            engine->mla_scratch[A_MLA_Q_A_NORM],
            engine->mla_scratch[A_MLA_Q_A],
            q_a_norm->data, 1u, K3_ENGINE_MLA_Q_LORA,
            1e-5f, NULL) ||
        !k3_static_weight_gemv_bf16(
            q_b, engine->mla_scratch[A_MLA_Q],
            engine->mla_scratch[A_MLA_Q_A_NORM], NULL) ||
        !k3_static_weight_gemv_bf16(
            kv_a, engine->mla_scratch[A_MLA_KV],
            normalized, NULL) ||
        !k3_rocm_rms_norm_bf16(
            cache_row, engine->mla_scratch[A_MLA_KV],
            kv_a_norm->data, 1u, K3_ENGINE_MLA_LATENT,
            1e-5f, NULL) ||
        hipMemcpyAsync(
            cache_row + K3_ENGINE_MLA_LATENT,
            (const uint16_t *)engine->mla_scratch[A_MLA_KV] +
                K3_ENGINE_MLA_LATENT,
            64u * sizeof(uint16_t),
            hipMemcpyDeviceToDevice, NULL) != hipSuccess ||
        !k3_rocm_mla_absorb_q_bf16(
            engine->mla_scratch[A_MLA_ABSORBED_Q],
            engine->mla_scratch[A_MLA_Q],
            packed_key, K3_ENGINE_HEADS, NULL) ||
        !k3_rocm_blas_mla_attention_bf16(
            engine->blas, latent, scores, probabilities,
            engine->mla_scratch[A_MLA_ABSORBED_Q],
            cache, K3_ENGINE_HEADS,
            engine->token_position + 1u, NULL) ||
        !k3_rocm_mla_decompress_v_bf16(
            engine->mla_scratch[A_MLA_VALUE],
            latent, kv_b->data, K3_ENGINE_HEADS, NULL) ||
        !k3_static_weight_gemv_bf16(
            gate, engine->mla_scratch[A_MLA_GATE],
            normalized, NULL) ||
        !k3_rocm_sigmoid_mul_bf16(
            engine->mla_scratch[A_MLA_GATED],
            engine->mla_scratch[A_MLA_VALUE],
            engine->mla_scratch[A_MLA_GATE],
            K3_ENGINE_KDA_INNER, NULL) ||
        !k3_static_weight_gemv_bf16(
            o_proj, attention,
            engine->mla_scratch[A_MLA_GATED], NULL)) {
        engine_error(error, error_size,
                     "layer-%u MLA launch failed", layer);
        return false;
    }
    return true;
}

extern "C" bool k3_engine_decode_layer0(
        k3_engine *engine,
        const void *input,
        void *output,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!engine || !input || !output ||
        !engine->causal_state_valid ||
        engine->decoded_layers != 0u) {
        engine_error(error, error_size,
                     "invalid or out-of-sequence layer-0 decode");
        return false;
    }
#define WEIGHT(variable, suffix)                                            \
    const k3_static_weight *variable = required_weight(                     \
        engine, "language_model.model.layers.0." suffix,                    \
        error, error_size);                                                 \
    if (!variable) return false
    WEIGHT(input_norm, "input_layernorm.weight");
    WEIGHT(q_proj, "self_attn.q_proj.weight");
    WEIGHT(k_proj, "self_attn.k_proj.weight");
    WEIGHT(v_proj, "self_attn.v_proj.weight");
    WEIGHT(q_conv, "self_attn.q_conv1d.weight");
    WEIGHT(k_conv, "self_attn.k_conv1d.weight");
    WEIGHT(v_conv, "self_attn.v_conv1d.weight");
    WEIGHT(f_a, "self_attn.f_a_proj.weight");
    WEIGHT(f_b, "self_attn.f_b_proj.weight");
    WEIGHT(b_proj, "self_attn.b_proj.weight");
    WEIGHT(a_log, "self_attn.A_log");
    WEIGHT(dt_bias, "self_attn.dt_bias");
    WEIGHT(g_proj, "self_attn.g_proj.weight");
    WEIGHT(o_norm, "self_attn.o_norm.weight");
    WEIGHT(o_proj, "self_attn.o_proj.weight");
    WEIGHT(mlp_res_norm, "mlp_res_norm.weight");
    WEIGHT(mlp_res_proj, "mlp_res_proj.weight");
    WEIGHT(post_norm, "post_attention_layernorm.weight");
    WEIGHT(dense_gate, "mlp.gate_proj.weight");
    WEIGHT(dense_up, "mlp.up_proj.weight");
    WEIGHT(dense_down, "mlp.down_proj.weight");
#undef WEIGHT

    const uint64_t hidden_bytes =
        (uint64_t)K3_ENGINE_HIDDEN * sizeof(uint16_t);
    const uint64_t conv_bytes =
        (uint64_t)K3_ENGINE_KDA_INNER * 4u * sizeof(uint16_t);
    uint8_t *conv = (uint8_t *)engine->kda_conv;
    if (hipMemcpyAsync(
            engine->attn_res_blocks, input, hidden_bytes,
            hipMemcpyDeviceToDevice, NULL) != hipSuccess ||
        !k3_rocm_rms_norm_bf16(
            engine->scratch[S_NORM], input, input_norm->data,
            1u, K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_static_weight_gemv_bf16(
            q_proj, engine->scratch[S_Q_PROJECTION],
            engine->scratch[S_NORM], NULL) ||
        !k3_static_weight_gemv_bf16(
            k_proj, engine->scratch[S_K_PROJECTION],
            engine->scratch[S_NORM], NULL) ||
        !k3_static_weight_gemv_bf16(
            v_proj, engine->scratch[S_V_PROJECTION],
            engine->scratch[S_NORM], NULL) ||
        !k3_rocm_short_conv4_silu_bf16_f32_weight(
            engine->scratch[S_Q], conv,
            engine->scratch[S_Q_PROJECTION],
            q_conv->data, K3_ENGINE_KDA_INNER, NULL) ||
        !k3_rocm_short_conv4_silu_bf16_f32_weight(
            engine->scratch[S_K], conv + conv_bytes,
            engine->scratch[S_K_PROJECTION],
            k_conv->data, K3_ENGINE_KDA_INNER, NULL) ||
        !k3_rocm_short_conv4_silu_bf16_f32_weight(
            engine->scratch[S_V], conv + 2u * conv_bytes,
            engine->scratch[S_V_PROJECTION],
            v_conv->data, K3_ENGINE_KDA_INNER, NULL) ||
        !k3_static_weight_gemv_bf16(
            f_a, engine->scratch[S_F_A],
            engine->scratch[S_NORM], NULL) ||
        !k3_static_weight_gemv_bf16(
            f_b, engine->scratch[S_RAW_GATE],
            engine->scratch[S_F_A], NULL) ||
        !k3_static_weight_gemv_bf16(
            b_proj, engine->scratch[S_RAW_BETA],
            engine->scratch[S_NORM], NULL) ||
        !k3_rocm_kda_recurrent_bf16_f32_state(
            engine->scratch[S_KDA], engine->kda_state,
            engine->scratch[S_Q], engine->scratch[S_K],
            engine->scratch[S_V], engine->scratch[S_RAW_GATE],
            engine->scratch[S_RAW_BETA], a_log->data, dt_bias->data,
            K3_ENGINE_HEADS, K3_ENGINE_KDA_HEAD_DIM,
            -5.0f, NULL) ||
        !k3_static_weight_gemv_bf16(
            g_proj, engine->scratch[S_OUTPUT_GATE],
            engine->scratch[S_NORM], NULL) ||
        !k3_rocm_rms_norm_sigmoid_gate_bf16_f32_weight(
            engine->scratch[S_GATED], engine->scratch[S_KDA],
            engine->scratch[S_OUTPUT_GATE], o_norm->data,
            K3_ENGINE_HEADS, K3_ENGINE_KDA_HEAD_DIM,
            1e-5f, NULL) ||
        !k3_static_weight_gemv_bf16(
            o_proj, engine->scratch[S_ATTENTION],
            engine->scratch[S_GATED], NULL) ||
        !k3_rocm_attn_res_bf16(
            engine->scratch[S_MLP_INPUT],
            engine->scratch[S_ATTENTION],
            engine->attn_res_blocks,
            mlp_res_norm->data, mlp_res_proj->data,
            1u, 8u, 1u, K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_rocm_rms_norm_bf16(
            engine->scratch[S_POST_NORM],
            engine->scratch[S_MLP_INPUT], post_norm->data,
            1u, K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_static_weight_gemv_bf16(
            dense_gate, engine->scratch[S_DENSE_GATE],
            engine->scratch[S_POST_NORM], NULL) ||
        !k3_static_weight_gemv_bf16(
            dense_up, engine->scratch[S_DENSE_UP],
            engine->scratch[S_POST_NORM], NULL) ||
        !k3_rocm_situ_bf16(
            engine->scratch[S_DENSE_ACTIVATION],
            engine->scratch[S_DENSE_GATE],
            engine->scratch[S_DENSE_UP],
            K3_ENGINE_DENSE_INTERMEDIATE,
            4.0f, 25.0f, NULL) ||
        !k3_static_weight_gemv_bf16(
            dense_down, engine->scratch[S_MLP_OUTPUT],
            engine->scratch[S_DENSE_ACTIVATION], NULL) ||
        !k3_rocm_add_bf16(
            output, engine->scratch[S_ATTENTION],
            engine->scratch[S_MLP_OUTPUT],
            K3_ENGINE_HIDDEN, NULL)) {
        engine_error(error, error_size,
                     "layer-0 decode launch failed");
        return false;
    }
    engine->decoded_layers = 1u;
    return true;
}

static bool decode_routed_layer(
        k3_engine *engine,
        uint32_t layer,
        const void *input,
        void *output,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!engine || !input || !output ||
        !engine->causal_state_valid ||
        layer == 0u || layer >= K3_ENGINE_LAYERS ||
        engine->decoded_layers != layer ||
        engine->staging_slots < K3_ENGINE_TOP_K) {
        engine_error(error, error_size,
                     "invalid or out-of-sequence layer-%u decode",
                     layer);
        return false;
    }
#define LAYER_WEIGHT(variable, suffix)                                      \
    const k3_static_weight *variable = required_layer_weight(               \
        engine, layer, suffix, error, error_size);                           \
    if (!variable) return false
    LAYER_WEIGHT(
        self_res_norm, "self_attention_res_norm.weight");
    LAYER_WEIGHT(
        self_res_proj, "self_attention_res_proj.weight");
    LAYER_WEIGHT(input_norm, "input_layernorm.weight");
    LAYER_WEIGHT(mlp_res_norm, "mlp_res_norm.weight");
    LAYER_WEIGHT(mlp_res_proj, "mlp_res_proj.weight");
    LAYER_WEIGHT(post_norm, "post_attention_layernorm.weight");
    LAYER_WEIGHT(
        router_bias,
        "block_sparse_moe.gate.e_score_correction_bias");
    LAYER_WEIGHT(
        router_weight, "block_sparse_moe.gate.weight");
    LAYER_WEIGHT(
        routed_down,
        "block_sparse_moe.routed_expert_down_proj.weight");
    LAYER_WEIGHT(
        routed_norm,
        "block_sparse_moe.routed_expert_norm.weight");
    LAYER_WEIGHT(
        routed_up,
        "block_sparse_moe.routed_expert_up_proj.weight");
    LAYER_WEIGHT(
        shared_down,
        "block_sparse_moe.shared_experts.down_proj.weight");
    LAYER_WEIGHT(
        shared_gate,
        "block_sparse_moe.shared_experts.gate_proj.weight");
    LAYER_WEIGHT(
        shared_up,
        "block_sparse_moe.shared_experts.up_proj.weight");
#undef LAYER_WEIGHT

    const uint32_t pre_attention_blocks =
        (layer + 11u) / 12u;
    const bool block_boundary = layer % 12u == 0u;
    const uint64_t hidden_bytes =
        (uint64_t)K3_ENGINE_HIDDEN * sizeof(uint16_t);
    if (!k3_rocm_attn_res_bf16(
            engine->scratch[S_MLP_INPUT], input,
            engine->attn_res_blocks,
            self_res_norm->data, self_res_proj->data,
            1u, 8u, pre_attention_blocks,
            K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        (block_boundary &&
         hipMemcpyAsync(
             (uint8_t *)engine->attn_res_blocks +
                 (uint64_t)pre_attention_blocks * hidden_bytes,
             input, hidden_bytes,
             hipMemcpyDeviceToDevice, NULL) != hipSuccess) ||
        !k3_rocm_rms_norm_bf16(
            engine->scratch[S_NORM],
            engine->scratch[S_MLP_INPUT], input_norm->data,
            1u, K3_ENGINE_HIDDEN, 1e-5f, NULL)) {
        engine_error(error, error_size,
                     "layer-%u pre-attention residual launch failed",
                     layer);
        return false;
    }
    const bool is_mla = layer % 4u == 3u || layer == 92u;
    if (!(is_mla ?
          launch_mla_attention(
              engine, layer, engine->scratch[S_NORM],
              engine->scratch[S_ATTENTION],
              error, error_size) :
          launch_kda_attention(
              engine, layer, engine->scratch[S_NORM],
              engine->scratch[S_ATTENTION],
              error, error_size))) {
        return false;
    }
    if ((block_boundary ?
         hipMemcpyAsync(
             engine->scratch[S_MLP_OUTPUT],
             engine->scratch[S_ATTENTION], hidden_bytes,
             hipMemcpyDeviceToDevice, NULL) == hipSuccess :
         k3_rocm_add_bf16(
             engine->scratch[S_MLP_OUTPUT], input,
             engine->scratch[S_ATTENTION],
             K3_ENGINE_HIDDEN, NULL)) == false ||
        !k3_rocm_attn_res_bf16(
            engine->scratch[S_MLP_INPUT],
            engine->scratch[S_MLP_OUTPUT],
            engine->attn_res_blocks,
            mlp_res_norm->data, mlp_res_proj->data,
            1u, 8u,
            pre_attention_blocks + (block_boundary ? 1u : 0u),
            K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_rocm_rms_norm_bf16(
            engine->scratch[S_POST_NORM],
            engine->scratch[S_MLP_INPUT], post_norm->data,
            1u, K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_rocm_bf16_gemv_f32(
            engine->moe_scratch[M_LOGITS],
            router_weight->data, engine->scratch[S_POST_NORM],
            K3_ENGINE_EXPERTS, K3_ENGINE_HIDDEN, NULL) ||
        !k3_rocm_router_topk_f32(
            engine->route_ids_device,
            engine->route_weights_device,
            engine->moe_scratch[M_LOGITS],
            router_bias->data,
            K3_ENGINE_EXPERTS, K3_ENGINE_TOP_K, 1.0f, NULL)) {
        engine_error(error, error_size,
                     "layer-%u attention/router launch failed",
                     layer);
        return false;
    }
    hipError_t status = hipStreamSynchronize(NULL);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "layer-%u attention/router failed: %s",
                     layer, hipGetErrorString(status));
        return false;
    }

    const uint16_t cache_layer = (uint16_t)(layer - 1u);
    uint16_t expert_ids[K3_ENGINE_TOP_K];
    k3_engine_expert_layout layouts[K3_ENGINE_TOP_K];
    for (uint32_t rank = 0; rank < K3_ENGINE_TOP_K; rank++) {
        if (engine->route_ids_host[rank] >= K3_ENGINE_EXPERTS) {
            engine_error(error, error_size,
                         "layer-%u router returned invalid expert %u",
                         layer,
                         engine->route_ids_host[rank]);
            return false;
        }
        expert_ids[rank] =
            (uint16_t)engine->route_ids_host[rank];
        if (!find_expert_layout(
                engine, layer, expert_ids[rank], &layouts[rank],
                error, error_size)) {
            return false;
        }
    }

    k3_expert_cache_access accesses[K3_ENGINE_TOP_K];
    if (!k3_expert_cache_plan(
            engine->cache_policy, cache_layer, expert_ids,
            K3_ENGINE_TOP_K, accesses, error, error_size)) {
        return false;
    }
    uint32_t miss_ranks[K3_ENGINE_TOP_K];
    uint32_t miss_count = 0u;
    for (uint32_t rank = 0; rank < K3_ENGINE_TOP_K; rank++) {
        if (!accesses[rank].hit) {
            miss_ranks[miss_count++] = rank;
        }
    }
    uint32_t next_miss = 0u;
    k3_io_request requests[K3_ENGINE_STREAM_QD];
    uint16_t initial_count = 0u;
    while (next_miss < miss_count &&
           initial_count < K3_ENGINE_STREAM_QD) {
        const uint32_t rank = miss_ranks[next_miss++];
        requests[initial_count++] =
            make_expert_request(engine, rank, &layouts[rank]);
    }
    if (initial_count &&
        !k3_io_uring_submit(
            engine->staging_ring, requests, initial_count,
            error, error_size)) {
        abort_layer_moe(engine, cache_layer);
        return false;
    }

    if (!k3_static_weight_gemv_bf16(
            routed_down, engine->moe_scratch[M_LATENT],
            engine->scratch[S_POST_NORM],
            engine->expert_stream) ||
        !k3_static_weight_gemv_bf16(
            shared_gate, engine->moe_scratch[M_SHARED_GATE],
            engine->scratch[S_POST_NORM],
            engine->shared_stream) ||
        !k3_static_weight_gemv_bf16(
            shared_up, engine->moe_scratch[M_SHARED_UP],
            engine->scratch[S_POST_NORM],
            engine->shared_stream) ||
        !k3_rocm_situ_bf16(
            engine->moe_scratch[M_SHARED_ACTIVATION],
            engine->moe_scratch[M_SHARED_GATE],
            engine->moe_scratch[M_SHARED_UP],
            K3_ENGINE_SHARED_HIDDEN, 4.0f, 25.0f,
            engine->shared_stream) ||
        !k3_static_weight_gemv_bf16(
            shared_down, engine->moe_scratch[M_SHARED_OUTPUT],
            engine->moe_scratch[M_SHARED_ACTIVATION],
            engine->shared_stream)) {
        engine_error(error, error_size,
                     "layer-%u static MoE launch failed",
                     layer);
        abort_layer_moe(engine, cache_layer);
        return false;
    }
    for (uint32_t rank = 0; rank < K3_ENGINE_TOP_K; rank++) {
        if (!accesses[rank].hit) continue;
        uint8_t *base = (uint8_t *)k3_engine_cache_slot(
            engine, accesses[rank].source_slot);
        if (!base ||
            !launch_expert(engine, rank, base, &layouts[rank])) {
            engine_error(error, error_size,
                         "layer-%u cache-hit expert %u launch failed",
                         layer,
                         expert_ids[rank]);
            abort_layer_moe(engine, cache_layer);
            return false;
        }
    }

    uint32_t completed_misses = 0u;
    while (completed_misses < miss_count) {
        k3_io_completion completions[K3_ENGINE_STREAM_QD];
        uint16_t completion_count = 0u;
        if (!k3_io_uring_wait(
                engine->staging_ring, completions,
                K3_ENGINE_STREAM_QD, &completion_count,
                error, error_size)) {
            abort_layer_moe(engine, cache_layer);
            return false;
        }
        uint16_t refill_count = 0u;
        while (next_miss < miss_count &&
               refill_count < completion_count) {
            const uint32_t rank = miss_ranks[next_miss++];
            requests[refill_count++] =
                make_expert_request(engine, rank, &layouts[rank]);
        }
        if (refill_count &&
            !k3_io_uring_submit(
                engine->staging_ring, requests, refill_count,
                error, error_size)) {
            abort_layer_moe(engine, cache_layer);
            return false;
        }
        for (uint16_t i = 0; i < completion_count; i++) {
            const uint32_t rank =
                (uint32_t)completions[i].user_data;
            const uint64_t required_bytes =
                rank < K3_ENGINE_TOP_K ?
                layouts[rank].physical_start -
                    layouts[rank].aligned_start +
                    K3_ENGINE_EXPERT_BYTES :
                UINT64_MAX;
            if (rank >= K3_ENGINE_TOP_K ||
                completions[i].buffer_index != rank ||
                completions[i].result < 0 ||
                (uint64_t)completions[i].result < required_bytes) {
                engine_error(error, error_size,
                             "layer-%u expert completion rank=%u "
                             "buffer=%u result=%d required=%llu "
                             "requested=%u",
                             layer, rank,
                             completions[i].buffer_index,
                             completions[i].result,
                             (unsigned long long)required_bytes,
                             rank < K3_ENGINE_TOP_K ?
                                 layouts[rank].aligned_bytes : 0u);
                abort_layer_moe(engine, cache_layer);
                return false;
            }
            uint8_t *base =
                (uint8_t *)engine->staging_device[rank] +
                (layouts[rank].physical_start -
                 layouts[rank].aligned_start);
            if (!launch_expert(
                    engine, rank, base, &layouts[rank])) {
                engine_error(error, error_size,
                             "layer-%u streamed expert %u launch failed",
                             layer,
                             expert_ids[rank]);
                abort_layer_moe(engine, cache_layer);
                return false;
            }
            if (accesses[rank].admit) {
                void *destination = k3_engine_cache_slot(
                    engine, accesses[rank].destination_slot);
                status = destination ?
                    hipMemcpyAsync(
                        destination, base, K3_ENGINE_EXPERT_BYTES,
                        hipMemcpyDeviceToDevice,
                        engine->expert_stream) :
                    hipErrorInvalidValue;
                if (status != hipSuccess) {
                    engine_error(
                        error, error_size,
                        "layer-%u cache admission failed: %s",
                        layer,
                        hipGetErrorString(status));
                    abort_layer_moe(engine, cache_layer);
                    return false;
                }
            }
            completed_misses++;
        }
    }
    if (k3_io_uring_outstanding(engine->staging_ring) != 0u) {
        engine_error(error, error_size,
                     "layer-%u retained outstanding expert reads",
                     layer);
        abort_layer_moe(engine, cache_layer);
        return false;
    }
    status = hipStreamSynchronize(engine->expert_stream);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "layer-%u expert stream failed: %s",
                     layer,
                     hipGetErrorString(status));
        abort_layer_moe(engine, cache_layer);
        return false;
    }

    if (!k3_rocm_weighted_sum_bf16(
            engine->moe_scratch[M_ROUTED],
            engine->moe_scratch[M_EXPERT_OUTPUTS],
            engine->route_weights_device,
            K3_ENGINE_TOP_K, K3_ENGINE_LATENT, NULL) ||
        !k3_rocm_rms_norm_bf16(
            engine->moe_scratch[M_ROUTED_NORM],
            engine->moe_scratch[M_ROUTED],
            routed_norm->data, 1u, K3_ENGINE_LATENT,
            1e-5f, NULL) ||
        !k3_static_weight_gemv_bf16(
            routed_up, engine->moe_scratch[M_ROUTED_FULL],
            engine->moe_scratch[M_ROUTED_NORM], NULL)) {
        engine_error(error, error_size,
                     "layer-%u routed MoE tail launch failed",
                     layer);
        abort_layer_moe(engine, cache_layer);
        return false;
    }
    status = hipStreamSynchronize(engine->shared_stream);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "layer-%u shared MoE stream failed: %s",
                     layer,
                     hipGetErrorString(status));
        abort_layer_moe(engine, cache_layer);
        return false;
    }
    if (!k3_rocm_add_bf16(
            engine->scratch[S_ATTENTION],
            engine->moe_scratch[M_ROUTED_FULL],
            engine->moe_scratch[M_SHARED_OUTPUT],
            K3_ENGINE_HIDDEN, NULL) ||
        !k3_rocm_add_bf16(
            output, engine->scratch[S_MLP_OUTPUT],
            engine->scratch[S_ATTENTION],
            K3_ENGINE_HIDDEN, NULL)) {
        engine_error(error, error_size,
                     "layer-%u final residual launch failed",
                     layer);
        abort_layer_moe(engine, cache_layer);
        return false;
    }
    if (!k3_expert_cache_commit(
            engine->cache_policy, cache_layer,
            error, error_size)) {
        abort_layer_moe(engine, cache_layer);
        return false;
    }
    engine->decoded_layers = layer + 1u;
    return true;
}

extern "C" bool k3_engine_decode_layer1(
        k3_engine *engine,
        const void *input,
        void *output,
        char *error,
        size_t error_size) {
    return decode_routed_layer(
        engine, 1u, input, output, error, error_size);
}

extern "C" bool k3_engine_decode_next_layer(
        k3_engine *engine,
        const void *input,
        void *output,
        char *error,
        size_t error_size) {
    if (!engine || !engine->causal_state_valid ||
        engine->decoded_layers < 2u ||
        engine->decoded_layers >= K3_ENGINE_LAYERS) {
        if (error && error_size) error[0] = '\0';
        engine_error(error, error_size,
                     "no routed K3 layer is ready to decode");
        return false;
    }
    return decode_routed_layer(
        engine, engine->decoded_layers,
        input, output, error, error_size);
}

extern "C" bool k3_engine_load_embedding(
        k3_engine *engine,
        uint32_t token_id,
        void *output,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!engine || !output ||
        !engine->causal_state_valid ||
        engine->decoded_layers != 0u ||
        engine->token_position >= engine->context ||
        token_id >= K3_ENGINE_VOCAB) {
        engine_error(error, error_size,
                     "invalid embedding request");
        return false;
    }
    const k3_st_tensor *embedding = k3_st_find(
        &engine->model,
        "language_model.model.embed_tokens.weight");
    const uint64_t hidden_bytes =
        (uint64_t)K3_ENGINE_HIDDEN * sizeof(uint16_t);
    if (!embedding || embedding->ndim != 2u ||
        embedding->shape[0] != K3_ENGINE_VOCAB ||
        embedding->shape[1] != K3_ENGINE_HIDDEN) {
        engine_error(error, error_size,
                     "invalid K3 embedding tensor");
        return false;
    }
    k3_st_read_view view;
    if (!k3_st_read_span_into(
            &engine->model, embedding->shard,
            embedding->physical_offset +
                (uint64_t)token_id * hidden_bytes,
            hidden_bytes, 4096u,
            engine->staging_host[0],
            K3_ENGINE_STAGING_BYTES,
            &view, error, error_size)) {
        return false;
    }
    const uint64_t data_offset =
        (uint64_t)(view.data -
                   (uint8_t *)engine->staging_host[0]);
    hipError_t status = hipMemcpyAsync(
        output,
        (uint8_t *)engine->staging_device[0] + data_offset,
        hidden_bytes, hipMemcpyDeviceToDevice, NULL);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "embedding upload failed: %s",
                     hipGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool k3_engine_decode_greedy(
        k3_engine *engine,
        const void *input,
        uint32_t *token_id,
        float *token_value,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!engine || !input || !token_id ||
        !engine->causal_state_valid ||
        engine->decoded_layers != K3_ENGINE_LAYERS) {
        engine_error(error, error_size,
                     "invalid or out-of-sequence output decode");
        return false;
    }
    const k3_static_weight *output_res_norm = required_weight(
        engine, "language_model.model.output_attn_res_norm.weight",
        error, error_size);
    const k3_static_weight *output_res_proj = required_weight(
        engine, "language_model.model.output_attn_res_proj.weight",
        error, error_size);
    const k3_static_weight *final_norm = required_weight(
        engine, "language_model.model.norm.weight",
        error, error_size);
    const k3_static_weight *head = required_weight(
        engine, "language_model.lm_head.weight",
        error, error_size);
    if (!output_res_norm || !output_res_proj ||
        !final_norm || !head) {
        return false;
    }
    if (!k3_rocm_attn_res_bf16(
            engine->scratch[S_MLP_INPUT], input,
            engine->attn_res_blocks,
            output_res_norm->data, output_res_proj->data,
            1u, 8u, 8u, K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_rocm_rms_norm_bf16(
            engine->scratch[S_NORM],
            engine->scratch[S_MLP_INPUT], final_norm->data,
            1u, K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_static_weight_gemv_bf16(
            head, engine->logits,
            engine->scratch[S_NORM], NULL) ||
        !k3_rocm_argmax_bf16(
            engine->token_id_device,
            engine->token_value_device,
            engine->logits, K3_ENGINE_VOCAB, NULL)) {
        engine_error(error, error_size,
                     "K3 output-head launch failed");
        return false;
    }
    hipError_t status = hipStreamSynchronize(NULL);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "K3 output-head decode failed: %s",
                     hipGetErrorString(status));
        return false;
    }
    *token_id = *engine->token_id_host;
    if (token_value) *token_value = *engine->token_value_host;
    engine->token_position++;
    engine->decoded_layers = 0u;
    return true;
}

extern "C" void k3_engine_get_cache_stats(
        const k3_engine *engine,
        k3_engine_cache_stats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    if (!engine) return;
    k3_expert_cache_stats internal;
    k3_expert_cache_get_stats(
        engine->cache_policy, &internal);
    stats->batches = internal.batches;
    stats->accesses = internal.accesses;
    stats->hits = internal.hits;
    stats->misses = internal.misses;
    stats->admissions = internal.admissions;
    stats->evictions = internal.evictions;
}

extern "C" bool k3_engine_forward_token(
        k3_engine *engine,
        uint32_t input_token,
        uint32_t *next_token,
        float *token_value,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!engine || !next_token ||
        !engine->causal_state_valid ||
        engine->decoded_layers != 0u) {
        engine_error(error, error_size,
                     "invalid K3 token-forward request");
        return false;
    }
    if (!k3_engine_load_embedding(
            engine, input_token, engine->token_buffer[0],
            error, error_size) ||
        !k3_engine_decode_layer0(
            engine, engine->token_buffer[0],
            engine->token_buffer[1], error, error_size) ||
        !k3_engine_decode_layer1(
            engine, engine->token_buffer[1],
            engine->token_buffer[0], error, error_size)) {
        return false;
    }
    void *input = engine->token_buffer[0];
    void *output = engine->token_buffer[1];
    for (uint32_t layer = 2u;
         layer < K3_ENGINE_LAYERS; layer++) {
        if (!k3_engine_decode_next_layer(
                engine, input, output, error, error_size)) {
            return false;
        }
        void *swap = input;
        input = output;
        output = swap;
    }
    return k3_engine_decode_greedy(
        engine, input, next_token, token_value,
        error, error_size);
}

extern "C" bool k3_engine_reset_state(
        k3_engine *engine,
        bool clear_expert_cache,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!engine) {
        engine_error(error, error_size,
                     "invalid K3 state-reset request");
        return false;
    }
    engine->causal_state_valid = false;
    hipError_t status = hipDeviceSynchronize();
    if (status == hipSuccess) {
        status = hipMemset(
            engine->kda_state, 0,
            K3_ENGINE_KDA_STATE_BYTES);
    }
    if (status == hipSuccess) {
        status = hipMemset(
            engine->kda_conv, 0,
            K3_ENGINE_KDA_CONV_BYTES);
    }
    if (status == hipSuccess) {
        status = hipMemset(
            engine->mla_cache, 0,
            engine->mla_cache_bytes);
    }
    if (status == hipSuccess) {
        status = hipMemset(
            engine->attn_res_blocks, 0,
            UINT64_C(8) * K3_ENGINE_HIDDEN *
                sizeof(uint16_t));
    }
    if (status != hipSuccess) {
        engine_error(
            error, error_size,
            "zeroing K3 causal state failed: %s",
            hipGetErrorString(status));
        return false;
    }
    if (clear_expert_cache) {
        for (uint16_t layer = 0u;
             layer < K3_ENGINE_MOE_LAYERS; layer++) {
            k3_expert_cache_abort(
                engine->cache_policy, layer);
        }
        if (!k3_expert_cache_reset(
                engine->cache_policy,
                error, error_size)) {
            return false;
        }
    }
    engine->decoded_layers = 0u;
    engine->token_position = 0u;
    engine->causal_state_valid = true;
    return true;
}

#include "k3_engine_state.inc"
#include "k3_engine_prefill.inc"
