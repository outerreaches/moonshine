#include "mimo26_gpu_worker.h"
#include "mimo26_expert_group_plan.h"

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
/*
 * Everything runs on the null stream, so the synchronous hipMemcpy below
 * first drains whatever kernels are still enqueued and only then moves bytes.
 * Timing the call as a whole charges GPU compute to "copy", which makes the
 * staging copy look more removable than it is -- the reason the zero-copy
 * lead is gated on this measurement. Under profiling we drain explicitly and
 * bill that separately; hipMemcpy would have waited anyway, so this changes
 * attribution rather than behaviour.
 */
static double g_queue_wait_seconds = 0.0;

static bool admission_profile_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        cached = getenv("MIMO26_GPU_PROFILE") != NULL ? 1 : 0;
    }
    return cached == 1;
}

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
    expert_slot     *slots;            /* slot_count views, global indices */
    /*
     * One allocation backing every slot, rather than one per slot. At 96
     * slots per layer that was 4,512 separate 12.75 MiB hipMallocs and the
     * host began swapping partway through, even though the steady-state
     * footprint fits with roughly 55 GiB to spare -- the cost was in the
     * number of buffer objects, not the bytes. PACKED_EXPERT_BYTES is 3,264
     * pages exactly, so every slot's offset stays 4096-aligned for DMA.
     */
    void            *slot_pool;
    uint64_t         slot_pool_bytes;
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
    bool retention_faulted; /* sticky until destruction, not cleared by reset */
    bool execution_active;  /* single-owner reentrancy guard, not a mutex */
    mimo26_gpu_worker_stats stats;
    char     last_error[512];
};

struct retention_execution_guard {
    mimo26_gpu_worker *worker;
    bool completed = false;
    explicit retention_execution_guard(mimo26_gpu_worker *w) : worker(w) {
        worker->execution_active = true;
    }
    ~retention_execution_guard() {
        if (!completed) worker->retention_faulted = true;
        worker->execution_active = false;
    }
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
    /*
     * 131,072 tokens, paired with the 160 slots below.
     *
     * 262,144 was the default from 2026-09-24 until the serving qualification
     * on 2026-09-25 refused to start on it. At 160 slots that profile predicts
     * 111.4 GiB, and MemAvailable on an IDLE host drifts 118.7-120.5 GiB, so
     * it clears the 8 GiB floor only sometimes -- it was refused outright with
     * nothing else running. A default that starts depending on page-cache
     * state is not a default.
     *
     * 131,072 predicts 108.3 GiB and leaves about 10.5 GiB, and is the pairing
     * that qualified 7/7 through the real server.
     *
     * KV itself remains cheap -- 39 of 48 layers are SWA capped at a 128-token
     * window, so only the 9 global layers grow, at 22.6 KiB/token. What binds
     * is the expert cache, so this is a slots-versus-context trade rather than
     * a hardware limit: 128 slots hold 262,144 comfortably, predicting
     * 92.7 GiB with ~30 GiB spare, and that is the conservative profile to run
     * where the host is unknown.
     *
     * Note what neither buys: prefill falls from ~17.6 tok/s at short prompts
     * to ~10.5 at 11K, so a genuinely large cold prompt is hours of ingest.
     * Large contexts are reached by accumulating across turns with
     * --kv-prefix-reuse, not by single enormous prompts.
     */
    config->global_kv_capacity = 131072u;
    /*
     * The September 24 short-request sweep motivated a larger pooled cache.
     * The rates below divide generated tokens by WHOLE request time, including
     * prefill; they are not pure decode throughput or a hardware ceiling.
     * Admission and GPU compute both matter. Explicit-profile, full-vector
     * and host-memory qualification is recorded separately in the vault.
     *
     *   slots  cached   hit    tok/s  resident
     *      16    6.2%  0.471    1.34   20.5 GiB   (the old default)
     *      64   25.0%  0.721    2.24   48.6 GiB
     *      96   37.5%  0.795    2.71   67.3 GiB
     *     128   50.0%  0.833    3.12   86.1 GiB
     *     160   62.5%  0.849    3.30  104.8 GiB
     *     176   68.8%     --      --   host swaps during load
     *
     * 160 rather than 128, decided 2026-09-24 on a matched ABBA over a mixed
     * code/agentic corpus (tool-calling, code generation, code editing,
     * structured extraction, prose), lookahead on in every arm, full-vector
     * equality 30/30 throughout:
     *
     *   warm prefill total   128 slots 151.14 s   160 slots 128.23 s  -15.2%
     *   per workload         -13.6% to -17.6%, consistent, not carried by one
     *   throughput           13.72 -> 16.17 tok/s
     *
     * This replaces an earlier -17.8% from a single-prompt driver that was
     * not an ABBA comparison. 160 is also the point where grouped prefill
     * starts paying at all: a chunk touches ~135 distinct experts, so below
     * that the working set cannot stay resident.
     *
     * Cost is +18.7 GiB resident, leaving ~16 GiB of host memory. That was
     * previously treated as too thin; the operator's call is that it is not,
     * on a host that runs nothing else. The startup guard's floor moved to
     * 10 GiB to match, which still refuses 176 slots (7.4 GiB).
     */
    config->expert_slots_per_layer = 160u;
    config->memory_limit_bytes = 0u;
    /*
     * 128 tokens per chunk, the maximum the CLI allows. The chunk is a
     * locality knob for the expert cache, not just projection amortization:
     * prefill plans experts per token, so a wider chunk means more tokens
     * sharing a layer's resident slots before the walk moves on. Measured on
     * a 2425-token prompt at the 128-slot default:
     *
     *   chunk   prefill   MB/prompt token   hit
     *       8   11.7 t/s        224         0.956
     *      16   12.7 t/s        212         0.958
     *      32   13.4 t/s        194         0.962   (the old default)
     *      64   14.1 t/s        168         0.967
     *     128   14.7 t/s        141         0.972
     *
     * Monotonic and still improving at the ceiling, though flattening: each
     * doubling is worth less than the last (+8.5%, +5.5%, +5.2%, +4.3%).
     * 128 over 32 is +9.7% prefill and 27% less expert traffic. The scratch
     * cost is trivial -- a few MiB -- now that the buffers are sized right.
     */
    config->prefill_chunk = 128u;
    config->expert_lookahead = true;
    config->expert_major = true;
    /*
     * ON as of 2026-09-27. The weight-reuse tiled MXFP4 GEMM is now the default
     * expert projection kernel.
     *
     * It was off while it was an unqualified behavioural change: it perturbs
     * routing, so ~99% of logits move. What settled it was a paired quality
     * screen with criteria fixed before either arm ran -- and the outcome was
     * stronger than the criteria asked for. Across 60 items in four families the
     * two kernels produced BYTE-IDENTICAL output, because greedy's top-1 margin
     * (~9.2) is 20-30x the logit shift (0.15-0.41). End to end it is 1.25x
     * faster, 1.28x on 6.5K-token prompts.
     *
     * The boundary on that result: greedy only. Margins between ranks 2, 3 and 4
     * are the same size as the perturbation, so if sampling is ever added the two
     * kernels WILL diverge. This server refuses temperature/top_p rather than
     * accepting and ignoring them, which is what makes the default safe.
     *
     * Assigned rather than left to the caller's zeroing, which is how it was
     * missed until 2026-09-26: a default that depends on how the caller
     * allocated the struct is not a default.
     *
     * See [[MiMo Tiled Kernel Quality Screen — Results 2026-09-26]].
     */
    config->expert_weight_reuse = true;
    /*
     * 256 MiB. Full-width sizing would want 8 GiB at the 262144 context above,
     * which is what made that context fail to allocate.
     *
     * Splitting is close to free, which is why the cap can be this small. The
     * kernel grid is (head, query) either way, so the same blocks read the
     * same history whatever the batch width -- narrowing it changes the number
     * of launches, not the work. At 256 MiB the deepest prefill attends 4
     * tokens at a time, about 3000 extra launches per chunk against attention
     * that costs seconds at that depth. Below a ~16K history a full 128-token
     * chunk still fits in one launch, so the cap is inert for short contexts.
     */
    config->attention_scratch_bytes = MIMO26_DEFAULT_ATTENTION_SCRATCH_BYTES;
}

/*
 * Entries (slots * heads) the per-layer key or value staging buffer needs. It
 * holds one layer's view at a time, so the bound is the widest single layer:
 * a global layer is 4 heads over the whole capacity, a windowed one 8 heads
 * over the 128 ring plus the chunk appended on device.
 */
static size_t kv_view_entries(const mimo26_gpu_worker_config *config)
{
    const size_t chunk =
        config->prefill_chunk > 0u ? (size_t)config->prefill_chunk : 1u;
    const size_t global =
        (size_t)config->global_kv_capacity * MIMO26_GLOBAL_KV_HEADS;
    const size_t windowed =
        ((size_t)MIMO26_SLIDING_WINDOW + chunk) * MIMO26_SWA_KV_HEADS;
    return global > windowed ? global : windowed;
}

/*
 * Floats to allocate for the attention scratch: a full chunk at the deepest
 * history, clamped to the configured ceiling, but never below the single row
 * the decode path needs.
 */
uint64_t mimo26_gpu_worker_attention_scratch_floats(
    const mimo26_gpu_worker_config *config)
{
    const uint64_t chunk =
        config->prefill_chunk > 0u ? (uint64_t)config->prefill_chunk : 1u;
    const uint64_t row =
        mimo26_rocm_attention_scratch_floats(config->global_kv_capacity);
    uint64_t floats = chunk * row;
    if (config->attention_scratch_bytes > 0u) {
        const uint64_t cap = config->attention_scratch_bytes / sizeof(float);
        if (floats > cap) {
            floats = cap < row ? row : cap;
        }
    }
    return floats;
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

    /*
     * Everything below scales with the context or the chunk, and none of it
     * was counted until 2026-09-24. The omission was invisible at a 2048
     * context -- a few hundred MiB -- and the agreement with the worker's own
     * resident ledger was no check at all, because that ledger skips the same
     * buffers. At 262144 it was 3.8 GiB, and the guard passed a profile that
     * then failed to allocate.
     */
    /* Per-step device staging for one layer's key and value view. */
    total += (uint64_t)kv_view_entries(config) * (QK + VD) * sizeof(uint16_t);
    /* The attention score scratch, the largest of these by far. */
    total += mimo26_gpu_worker_attention_scratch_floats(config) *
             sizeof(float);
    /* Prefill scratch, all of it linear in the chunk. The widths follow
     * scratch_resize: QKV fused and split, attention out, the projections,
     * the three 16384-wide MLP buffers, the expert-major gather/stash, the
     * F32 accumulator and the router buffers. */
    const uint64_t chunk =
        config->prefill_chunk > 0u ? (uint64_t)config->prefill_chunk : 1u;
    total += chunk * (2ull * HIDDEN + MIMO26_SWA_QKV_WIDTH +
                      MIMO26_QUERY_HEADS * QK +
                      MIMO26_SWA_KV_HEADS * (QK + VD) +
                      MIMO26_QUERY_HEADS * VD +
                      3ull * 16384ull +
                      2ull * HIDDEN +
                      MIMO26_ROUTER_TOP_K * HIDDEN) * sizeof(uint16_t);
    total += chunk * (HIDDEN + MIMO26_ROUTER_EXPERTS + MIMO26_ROUTER_TOP_K) *
             sizeof(float);
    total += chunk * 2ull * MIMO26_ROUTER_TOP_K * sizeof(uint32_t);
    /* Pinned staging ring for expert uploads. */
    total += (uint64_t)MIMO26_STAGING_SLOTS * MIMO26_STAGING_BYTES;
    return total;
}

uint64_t mimo26_gpu_worker_resident_bytes(const mimo26_gpu_worker *worker)
{
    return worker == NULL ? 0u : worker->resident_bytes;
}

bool mimo26_gpu_worker_expert_major(const mimo26_gpu_worker *worker)
{
    return worker == NULL ? false : worker->config.expert_major;
}

bool mimo26_gpu_worker_expert_weight_reuse(const mimo26_gpu_worker *worker)
{
    return worker == NULL ? false : worker->config.expert_weight_reuse;
}

mimo26_gpu_worker_status mimo26_gpu_worker_resolve_overrides(
    mimo26_gpu_worker_config *config, char *error, size_t error_size)
{
    if (config == NULL) {
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    }
    /*
     * The one environment override. Strict on purpose: this used to live in
     * the layer and treat any value but "0" as true, so MIMO26_EXPERT_MAJOR=off
     * silently enabled grouping and bypassed the CLI's coupling to lookahead,
     * while /health went on reporting the configured mode.
     */
    const char *value = getenv("MIMO26_EXPERT_MAJOR");
    if (value != NULL) {
        const bool on = strcmp(value, "1") == 0 || strcmp(value, "on") == 0;
        const bool off = strcmp(value, "0") == 0 || strcmp(value, "off") == 0;
        if (!on && !off) {
            return fail(error, error_size, MIMO26_GPU_WORKER_INVALID_ARGUMENT,
                        "MIMO26_EXPERT_MAJOR must be 0, 1, on or off; got \"%s\". "
                        "Refusing rather than guessing a mode.", value);
        }
        if (on && !config->expert_lookahead) {
            return fail(error, error_size, MIMO26_GPU_WORKER_INVALID_ARGUMENT,
                        "MIMO26_EXPERT_MAJOR=%s needs expert lookahead on; grouped "
                        "prefill without the remaining-group schedule is slower",
                        value);
        }
        config->expert_major = on;
    }
    /*
     * Second override, same strictness, selecting the expert projection kernel.
     * It exists so the weight-reuse tile can be measured against the shipping
     * GEMV in ONE binary: an A/B across two builds would charge this box's
     * 13-21% warm-up and ~2.6% drift to the kernel.
     * See [[perf-screens-need-interleaved-baselines]].
     */
    const char *reuse = getenv("MIMO26_EXPERT_WEIGHT_REUSE");
    if (reuse != NULL) {
        const bool on = strcmp(reuse, "1") == 0 || strcmp(reuse, "on") == 0;
        const bool off = strcmp(reuse, "0") == 0 || strcmp(reuse, "off") == 0;
        if (!on && !off) {
            return fail(error, error_size, MIMO26_GPU_WORKER_INVALID_ARGUMENT,
                        "MIMO26_EXPERT_WEIGHT_REUSE must be 0, 1, on or off; got "
                        "\"%s\". Refusing rather than guessing a mode.", reuse);
        }
        config->expert_weight_reuse = on;
    }
    return MIMO26_GPU_WORKER_OK;
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
static bool prepare_batch_impl(void *context, uint32_t layer,
                              const uint32_t *experts, size_t count,
                              const uint32_t *next_use)
{
    mimo26_gpu_worker *worker = (mimo26_gpu_worker *)context;
    uint16_t ids[MIMO26_ROUTER_TOP_K];
    k3_expert_cache_access accesses[MIMO26_ROUTER_TOP_K];
    if (count > MIMO26_ROUTER_TOP_K) {
        return false;
    }
    for (size_t k = 0; k < count; k++) {
        if (experts[k] >= MIMO26_ROUTER_EXPERTS) return false;
        ids[k] = (uint16_t)experts[k];
    }

    char error[256];
    const bool planned = next_use != NULL
        ? k3_expert_cache_plan_next_use(worker->cache, (uint16_t)layer, ids,
              (uint16_t)count, next_use, MIMO26_ROUTER_EXPERTS,
              accesses, error, sizeof error)
        : k3_expert_cache_plan(worker->cache, (uint16_t)layer, ids,
              (uint16_t)count, accesses, error, sizeof error);
    if (!planned) {
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
    /*
     * Timed separately from the read. Both accumulators used to receive the
     * same `elapsed`, so the profiler always printed "copy 0.000" and the
     * split between waiting on the drive and copying staging into the slot
     * was never actually measured -- which matters, because on an APU the
     * slot is system RAM too, so that copy is RAM-to-RAM of the entire read
     * volume.
     */
    double copy_seconds = 0.0;
    double queue_wait_seconds = 0.0;
    const bool profile_admission = admission_profile_enabled();
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
                struct timespec c0, cs, c1;
                clock_gettime(CLOCK_MONOTONIC, &c0);
                if (profile_admission) {
                    hipDeviceSynchronize();
                }
                clock_gettime(CLOCK_MONOTONIC, &cs);
                const hipError_t copied =
                    hipMemcpy(worker->slots[entry->slot].block, source,
                              PACKED_EXPERT_BYTES, hipMemcpyHostToDevice);
                clock_gettime(CLOCK_MONOTONIC, &c1);
                queue_wait_seconds += (double)(cs.tv_sec - c0.tv_sec) +
                                      (double)(cs.tv_nsec - c0.tv_nsec) / 1e9;
                copy_seconds += (double)(c1.tv_sec - cs.tv_sec) +
                                (double)(c1.tv_nsec - cs.tv_nsec) / 1e9;
                if (copied != hipSuccess) {
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
    g_read_seconds += elapsed - copy_seconds - queue_wait_seconds;
    g_copy_seconds += copy_seconds;
    g_queue_wait_seconds += queue_wait_seconds;
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

static bool prepare_batch(void *context, uint32_t layer,
                          const uint32_t *experts, size_t count)
{
    return prepare_batch_impl(context, layer, experts, count, NULL);
}

static bool prepare_batch_future(void *context, uint32_t layer,
                                 const uint32_t *experts, size_t count,
                                 const uint32_t *future, size_t future_tokens)
{
    const mimo26_gpu_worker *worker = (const mimo26_gpu_worker *)context;
    if (count != MIMO26_ROUTER_TOP_K || future_tokens >= worker->prefill_chunk ||
        (future_tokens && future == NULL)) return false;
    uint32_t next_use[MIMO26_ROUTER_EXPERTS];
    for (unsigned i = 0; i < MIMO26_ROUTER_EXPERTS; ++i) next_use[i] = UINT32_MAX;
    for (size_t t = 0; t < future_tokens; ++t) {
        for (size_t k = 0; k < MIMO26_ROUTER_TOP_K; ++k) {
            uint32_t id = future[t * MIMO26_ROUTER_TOP_K + k];
            if (id >= MIMO26_ROUTER_EXPERTS) return false;
            if (next_use[id] == UINT32_MAX) next_use[id] = (uint32_t)t;
        }
    }
    return prepare_batch_impl(context, layer, experts, count, next_use);
}

static bool prepare_group_future(void *context, uint32_t layer,
                                 uint32_t expert, const uint32_t *remaining,
                                 size_t remaining_count)
{
    uint32_t next_use[MIMO26_ROUTER_EXPERTS];
    if (!mimo26_expert_group_next_use(expert, remaining, remaining_count, next_use))
        return false;
    return prepare_batch_impl(context, layer, &expert, 1u, next_use);
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
        /*
         * Widened for expert-major execution: one batched down-projection
         * writes a row per token in the expert's group, not a single row.
         * The three buffers after it cost ~10 MiB at chunk 128 and exist only
         * on a prefill scratch.
         */
        {&s->expert_out, chunk * HIDDEN * sizeof(uint16_t)},
        {&s->expert_gathered, chunk * HIDDEN * sizeof(uint16_t)},
        {(void **)&s->expert_stash,
         chunk * MIMO26_ROUTER_TOP_K * HIDDEN * sizeof(uint16_t)},
        {(void **)&s->expert_row_ids,
         2u * chunk * MIMO26_ROUTER_TOP_K * sizeof(uint32_t)},
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
    /*
     * attention_scratch is deliberately NOT resized here. It is the largest
     * buffer in the profile at a long context -- chunk * 64 * (capacity + 2)
     * floats is 8 GiB at 262144 -- so it is allocated once at load, under the
     * configured cap and inside the startup guard's accounting. Prefill splits
     * its batch to fit whatever was allocated.
     */
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
    /* Slot blocks are offsets into slot_pool, not separate allocations. */
    if (worker->slot_pool != NULL) {
        hipFree(worker->slot_pool);
        worker->slot_pool = NULL;
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
    hipFree(worker->scratch.expert_gathered);
    hipFree(worker->scratch.expert_stash);
    hipFree(worker->scratch.expert_row_ids);
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
    /*
     * Resolved here as well as by the server, so that every caller -- the
     * gate, the tests, a direct embedder -- gets the same strict handling and
     * the same effective profile. The call is idempotent.
     */
    mimo26_gpu_worker_config effective = *config;
    const mimo26_gpu_worker_status resolved =
        mimo26_gpu_worker_resolve_overrides(&effective, error, error_size);
    if (resolved != MIMO26_GPU_WORKER_OK) {
        return resolved;
    }
    config = &effective;
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
    const uint32_t backed_slots =
        worker->slot_count > config->expert_slots_per_layer
            ? worker->slot_count - config->expert_slots_per_layer
            : 0u;
    if (backed_slots > 0u) {
        worker->slot_pool_bytes =
            (uint64_t)backed_slots * PACKED_EXPERT_BYTES;
        REQUIRE(hipMalloc(&worker->slot_pool, worker->slot_pool_bytes) ==
                    hipSuccess,
                "expert slot pool allocation failed");
        worker->resident_bytes += worker->slot_pool_bytes;
        for (uint32_t s = config->expert_slots_per_layer;
             s < worker->slot_count; s++) {
            const uint64_t index = s - config->expert_slots_per_layer;
            worker->slots[s].block = (uint8_t *)worker->slot_pool +
                                     index * PACKED_EXPERT_BYTES;
            slot_views(&worker->slots[s]);
        }
    }

    /* Scratch and per-step buffers. */
    const uint64_t capacity = config->global_kv_capacity;
    worker->scratch.attention_capacity = capacity;
    #define DEVICE(field, bytes)                                              \
        REQUIRE(hipMalloc(&worker->field, (bytes)) == hipSuccess,             \
                "device allocation failed")
    /*
     * hidden holds the whole prefill chunk, not one token. It was sized for
     * one, and prefill's embedding gather has been writing chunk * HIDDEN
     * into it ever since -- 512 KiB into an 8 KiB allocation at chunk 64.
     * Chunk 128 exposed an "embedding gather failed" error. Successful older
     * requests do not prove that an out-of-bounds write was harmless: allocator
     * layout and overwritten neighbors were not guaranteed. Corrected-worker
     * decode/chunk full-vector and allocation-guard tests cover bounded cases;
     * never rely on adjacent buffers being overwritten later.
     */
    const size_t hidden_tokens =
        config->prefill_chunk > 0u ? (size_t)config->prefill_chunk : 1u;
    DEVICE(hidden, hidden_tokens * HIDDEN * sizeof(uint16_t));
    DEVICE(normed, HIDDEN * sizeof(uint16_t));
    DEVICE(device_logits, (size_t)VOCAB * sizeof(float));
    /*
     * Staging for the layer being processed, so it needs the widest single
     * layer view, not capacity times the larger head count. A global layer is
     * 4 heads over the whole capacity; a windowed one is 8 heads over a 128
     * ring plus the chunk appended on device. Sizing it at capacity * 8 -- as
     * it was until 2026-09-24 -- reserved exactly twice the worst case, 0.6 GiB
     * of it at a 262144 context.
     */
    const size_t kv_view_slots = kv_view_entries(config);
    DEVICE(device_keys, kv_view_slots * QK * sizeof(uint16_t));
    DEVICE(device_values, kv_view_slots * VD * sizeof(uint16_t));
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
    /*
     * Sized once, here, for the widest prefill the cap allows -- not grown
     * later by scratch_resize. Growing it on the first prefill would move the
     * largest allocation in the profile past the startup guard, which is the
     * one place that can still refuse cleanly.
     */
    worker->scratch.attention_scratch_floats =
        mimo26_gpu_worker_attention_scratch_floats(config);
    SCRATCH(attention_scratch,
            worker->scratch.attention_scratch_floats * sizeof(float));
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
    if (worker->execution_active) return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    retention_execution_guard retention_guard(worker);
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
    g_queue_wait_seconds = 0.0;
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
                        "expert admission %.3f s [read %.3f, gpu-wait %.3f, "
                        "copy %.3f], compute %.3f s), "
                        "kv-up %.3f s, kv-down %.3f s, head %.3f s\n",
                (unsigned long long)position, layer_seconds,
                g_upload_seconds, g_read_seconds, g_queue_wait_seconds,
                g_copy_seconds,
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
    retention_guard.completed = true;
    return MIMO26_GPU_WORKER_OK;
}

mimo26_gpu_worker_status mimo26_gpu_worker_prefill(
    mimo26_gpu_worker *worker, const uint32_t *tokens, size_t count,
    float *logits, mimo26_gpu_prefill_progress progress,
    void *progress_context, char *error, size_t error_size)
{
    if (worker == NULL || tokens == NULL || logits == NULL) {
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    }
    if (count == 0u || worker->execution_active)
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
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
            if (progress != NULL) {
                retention_execution_guard callback_guard(worker);
                const bool keep_going = progress(progress_context, i + 1u, count);
                callback_guard.completed = true;
                if (!keep_going) break;
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

    retention_execution_guard retention_guard(worker);

    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);
    const bool profile = getenv("MIMO26_GPU_PROFILE") != NULL;
    double kv_seconds = 0.0, rope_seconds = 0.0, layer_seconds = 0.0;
    double readback_seconds = 0.0, commit_seconds = 0.0;
    g_upload_seconds = 0.0;
    g_read_seconds = 0.0;
    g_copy_seconds = 0.0;
    g_queue_wait_seconds = 0.0;
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
            context.prepare_future = w->is_moe && worker->config.expert_lookahead
                                         ? prepare_batch_future : NULL;
            context.prepare_group_future = w->is_moe && worker->config.expert_lookahead
                                         ? prepare_group_future : NULL;
            context.expert_major = worker->config.expert_major;
            context.expert_weight_reuse = worker->config.expert_weight_reuse;

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
        /* Between chunks is the only safe place to yield: a chunk is
         * transactional across all 48 layers, so stopping inside one would
         * leave the journal half-written. */
        if (progress != NULL && !progress(progress_context, done, count)) {
            retention_guard.completed = true; /* committed chunk boundary */
            return fail(error, error_size, MIMO26_GPU_WORKER_OK,
                        "prefill stopped after %zu of %zu tokens", done,
                        count);
        }
    }
    if (profile) {
        fprintf(stderr, "    prefill %zu tokens: layers %.2f s (admission "
                        "%.2f [read %.2f, gpu-wait %.2f, copy %.2f], "
                        "compute %.2f), kv-up %.2f, rope %.2f, readback %.2f, "
                        "commit %.2f\n",
                count, layer_seconds, g_upload_seconds, g_read_seconds,
                g_queue_wait_seconds, g_copy_seconds,
                layer_seconds - g_upload_seconds, kv_seconds,
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
    retention_guard.completed = true;
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

bool mimo26_gpu_worker_idle(const mimo26_gpu_worker *worker)
{
    return worker != NULL && !worker->execution_active &&
           !worker->retention_faulted &&
           !mimo26_kv_in_transaction(worker->kv) &&
           k3_io_uring_outstanding(worker->ring) == 0u;
}

uint64_t mimo26_gpu_worker_layout_crc64(const mimo26_gpu_worker *worker)
{
    return worker == NULL ? 0u : mimo26_kv_layout_crc64(worker->kv);
}

mimo26_gpu_worker_status mimo26_gpu_worker_export_state(
    const mimo26_gpu_worker *worker, const char *path,
    mimo26_kv_state_info *info, char *error, size_t error_size)
{
    if (worker == NULL || path == NULL) {
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    }
    if (!mimo26_gpu_worker_idle(worker)) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "a checkpoint needs a healthy idle worker");
    }
    if (mimo26_kv_export(worker->kv, path, info) != MIMO26_KV_OK) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "could not write context state to %s", path);
    }
    return MIMO26_GPU_WORKER_OK;
}

mimo26_gpu_worker_status mimo26_gpu_worker_import_state(
    mimo26_gpu_worker *worker, const char *path,
    mimo26_kv_state_info *info, char *error, size_t error_size)
{
    if (worker == NULL || path == NULL) {
        return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    }
    if (!mimo26_gpu_worker_idle(worker)) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "restoring a checkpoint needs a healthy idle worker");
    }
    const mimo26_kv_status status = mimo26_kv_import(worker->kv, path, info);
    /*
     * Take the position from the KV either way. A rejected file leaves the KV
     * as it was; the one failure that can mutate it resets it to empty.
     * Reading the length back rather than assuming a value is what keeps
     * position and history agreeing even on the paths that failed.
     */
    worker->position = mimo26_kv_length(worker->kv);
    worker->resolved_count = 0u;
    if (status != MIMO26_KV_OK) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "could not restore context state from %s", path);
    }
    return MIMO26_GPU_WORKER_OK;
}

mimo26_gpu_worker_status mimo26_gpu_worker_reset_context(
    mimo26_gpu_worker *worker, char *error, size_t error_size)
{
    if (worker == NULL) return MIMO26_GPU_WORKER_INVALID_ARGUMENT;
    if (!mimo26_gpu_worker_idle(worker)) {
        return fail(error, error_size, MIMO26_GPU_WORKER_DECODE_FAILED,
                    "weight retention requires a healthy idle worker; recreate after faults");
    }
    mimo26_kv_reset(worker->kv);
    worker->position = 0u;
    worker->resolved_count = 0u;
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
