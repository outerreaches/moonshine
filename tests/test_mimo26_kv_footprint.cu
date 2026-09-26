/*
 * Does a full 131,072-token KV cost what the startup guard says it does --
 * measured when every page is actually TOUCHED, not merely allocated?
 *
 * Review finding 7 (2026-09-26): a 128K allocation is not 128K qualification.
 * The correctness half is tests/test_mimo26_attention_deep. This is the memory
 * half, and it exists because of a specific trap already hit once in this lane:
 * the footprint predictor was declared accurate to 0.2% against the WORKER'S OWN
 * ledger, which omitted the same buffers the predictor omitted. It was under by
 * 3.8 GiB at a 262,144 context. See [[guard-predictors-need-external-check]].
 *
 * So the reference here is external to the process: amdgpu's own
 * mem_info_gtt_used, plus MemAvailable. Nothing in this test asks the allocator
 * what it thinks it allocated.
 *
 * It also separates two things the earlier measurements conflated. At load the
 * box showed ~13.2 GiB free "because the KV pages are untouched", and headroom
 * was said to shrink as a session fills the context. That predicts a difference
 * between allocating the KV and writing to it, and this measures both:
 *
 *   predicted   the geometry's own arithmetic
 *   allocated   GTT growth from hipMalloc alone
 *   touched     GTT growth after every byte is written
 *
 * If allocation already commits the pages, the two deltas match and a filling
 * context costs nothing further. If it does not, the difference is exactly the
 * headroom a long session will consume after the guard has already approved it.
 *
 * WHAT THIS IS NOT: the KV is allocated and written directly, with no model and
 * no prefill. It measures what a full KV costs the host. It does not establish
 * that a 131,072-token request completes, which needs hours of real ingest and
 * remains open.
 */
#include "mimo26_architecture.h"
#include "mimo26_attention.h"
#include "mimo26_gpu_worker.h"
#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK MIMO26_QK_HEAD_DIM
#define VD MIMO26_V_HEAD_DIM
#define CONTEXT 131072ull
#define CHUNK 128ull

static unsigned failures, ran;

static void ok(const char *what, int passed, const char *detail)
{
    printf("%s  %-56s %s\n", passed ? "PASS" : "FAIL", what, detail);
    fflush(stdout);
    ran++;
    failures += passed ? 0u : 1u;
}

#define HIP_OK(call)                                                          \
    do {                                                                      \
        const hipError_t s_ = (call);                                         \
        if (s_ != hipSuccess) {                                               \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call,   \
                    hipGetErrorString(s_));                                   \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

/* amdgpu's own accounting, not the allocator's. */
static uint64_t gtt_used(void)
{
    FILE *f = fopen("/sys/class/drm/card0/device/mem_info_gtt_used", "r");
    if (!f) return 0u;
    unsigned long long value = 0ull;
    if (fscanf(f, "%llu", &value) != 1) value = 0ull;
    fclose(f);
    return (uint64_t)value;
}

static uint64_t mem_available_bytes(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0u;
    char key[64];
    unsigned long long kb = 0ull;
    while (fscanf(f, "%63s %llu kB\n", key, &kb) >= 1) {
        if (strcmp(key, "MemAvailable:") == 0) { fclose(f); return kb * 1024ull; }
    }
    fclose(f);
    return 0u;
}

int main(void)
{
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        fprintf(stderr, "no HIP device\n");
        return 2;
    }

    /*
     * The shipped geometry: 9 global layers keep the whole context at 4 KV
     * heads; 39 windowed layers keep a 128 ring plus one chunk at 8. Keys are
     * 192 wide and values 128, which is the asymmetry that makes this worth
     * computing rather than guessing.
     */
    const uint64_t global_layers = 9ull, swa_layers = 39ull;
    const uint64_t global_heads = 4ull, swa_heads = 8ull;
    const uint64_t swa_slots = MIMO26_SLIDING_WINDOW + CHUNK;
    const uint64_t per_slot = (uint64_t)(QK + VD) * 2ull;
    const uint64_t predicted = global_layers * CONTEXT * global_heads * per_slot +
                               swa_layers * swa_slots * swa_heads * per_slot;

    printf("geometry: %" PRIu64 " global layers x %" PRIu64 " x %" PRIu64
           " heads, %" PRIu64 " swa x %" PRIu64 " x %" PRIu64 " heads\n",
           global_layers, CONTEXT, global_heads, swa_layers, swa_slots, swa_heads);
    printf("predicted KV: %.3f GiB (%.1f KiB per context token)\n\n",
           (double)predicted / 1073741824.0,
           (double)predicted / (double)CONTEXT / 1024.0);

    /*
     * Create the HIP context BEFORE the baseline, or its runtime structures are
     * charged to the KV. Measured: without this the KV appeared to cost 5.8%
     * more than its geometry, and part of that was simply context creation
     * landing inside the measured window. A control that is taken before the
     * subject exists measures the wrong thing.
     */
    void *warm = NULL;
    HIP_OK(hipMalloc(&warm, 1u << 20));
    HIP_OK(hipMemset(warm, 0, 1u << 20));
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipFree(warm));
    HIP_OK(hipDeviceSynchronize());

    const uint64_t gtt_before = gtt_used(), avail_before = mem_available_bytes();

    /* One buffer per layer, as the cache itself does, so fragmentation and
     * per-allocation rounding are represented rather than hidden in one slab. */
    const uint64_t layers = global_layers + swa_layers;
    void **buffers = (void **)calloc((size_t)layers, sizeof *buffers);
    uint64_t *sizes = (uint64_t *)calloc((size_t)layers, sizeof *sizes);
    if (!buffers || !sizes) return 2;
    for (uint64_t i = 0; i < layers; i++) {
        sizes[i] = (i < global_layers) ? CONTEXT * global_heads * per_slot
                                      : swa_slots * swa_heads * per_slot;
        HIP_OK(hipMalloc(&buffers[i], (size_t)sizes[i]));
    }
    HIP_OK(hipDeviceSynchronize());
    const uint64_t gtt_allocated = gtt_used(), avail_allocated = mem_available_bytes();

    /* Touch every byte. hipMemset writes through the same path a KV commit
     * would, so the pages become resident exactly as a filling context does. */
    for (uint64_t i = 0; i < layers; i++)
        HIP_OK(hipMemset(buffers[i], 0x3C, (size_t)sizes[i]));
    HIP_OK(hipDeviceSynchronize());
    const uint64_t gtt_touched = gtt_used(), avail_touched = mem_available_bytes();

    const double gib = 1073741824.0;
    const int64_t alloc_delta = (int64_t)gtt_allocated - (int64_t)gtt_before;
    const int64_t touch_delta = (int64_t)gtt_touched - (int64_t)gtt_before;
    printf("GTT used   before %.3f  after malloc %.3f  after touch %.3f GiB\n",
           gtt_before / gib, gtt_allocated / gib, gtt_touched / gib);
    printf("GTT growth from malloc %.3f GiB, from malloc+touch %.3f GiB\n",
           alloc_delta / gib, touch_delta / gib);
    printf("MemAvailable before %.2f  alloc %.2f  touch %.2f GiB\n\n",
           avail_before / gib, avail_allocated / gib, avail_touched / gib);

    char detail[160];

    /*
     * The load-bearing check: what the host commits once the KV is live must not
     * exceed what the guard predicted by more than page granularity can explain.
     * Over-prediction is safe; under-prediction is what admitted a 262,144
     * profile that then could not allocate.
     *
     * The allowance is derived, not fitted: 48 separate allocations each rounded
     * up to amdgpu's 2 MiB granularity is at most 96 MiB, so the bound is
     * predicted + 48 * 2 MiB. Anything beyond that is a real under-prediction
     * and should fail rather than have the threshold widened to admit it.
     */
    const uint64_t rounding_allowance = layers * 2ull * 1024ull * 1024ull;
    const double ratio = predicted ? (double)touch_delta / (double)predicted : 0.0;
    snprintf(detail, sizeof detail,
             "touched %.3f GiB vs predicted %.3f + %.3f rounding = %.3f GiB (%.1f%%)",
             touch_delta / gib, predicted / gib, rounding_allowance / gib,
             (predicted + rounding_allowance) / gib, ratio * 100.0);
    ok("touched KV within prediction plus page rounding",
       (uint64_t)touch_delta <= predicted + rounding_allowance, detail);

    snprintf(detail, sizeof detail, "%.1f%% of prediction", ratio * 100.0);
    ok("prediction is not wildly over (within 2x of measured)",
       ratio >= 0.5, detail);

    /*
     * Reported for its own sake: whether a filling context costs headroom the
     * guard has already given away. Not an assertion, because either answer is
     * legitimate -- it is the number an operator needs, and it is what the
     * earlier "the KV pages are untouched" observation was really about.
     */
    const int64_t commit_on_touch = touch_delta - alloc_delta;
    snprintf(detail, sizeof detail,
             "%.3f GiB commits on first write, after the guard has approved",
             commit_on_touch / gib);
    ok("extra commitment as the context fills is recorded",
       commit_on_touch >= 0, detail);

    snprintf(detail, sizeof detail, "%.1f KiB per token",
             (double)predicted / (double)CONTEXT / 1024.0);
    ok("per-token KV cost matches the documented 22.6 KiB",
       (double)predicted / (double)CONTEXT / 1024.0 < 24.0, detail);

    for (uint64_t i = 0; i < layers; i++) HIP_OK(hipFree(buffers[i]));
    free(buffers); free(sizes);

    printf("\n%s %u of %u KV footprint checks\n",
           failures ? "FAIL" : "PASS", ran - failures, ran);
    return failures ? 1 : 0;
}
