/*
 * Deep-history attention: the sub-batch widths production actually uses.
 *
 * WHAT THIS COVERS THAT test_mimo26_gpu_attention.cu DOES NOT.
 *
 * That suite sweeps widths 1, 5, 7, 16, 23, 24 against a chunk of 24 -- but it
 * *forces* each width by allocating the scratch at exactly that size. Its prior
 * histories are 0, 200 and 1000. Production does something different: it hands
 * the kernel a fixed 256 MiB budget and the width falls out of the history,
 *
 *     row_floats = MIMO26_ROCM_QUERY_HEADS * (history + 2)
 *     width      = min(chunk, budget_floats / row_floats)
 *                = min(128, 1048576 / (history + 2))       at 256 MiB
 *
 * so the interesting transition is a function of depth, and at a prior of 1000
 * it has not happened yet -- a 128-token chunk still runs in ONE launch until
 * the history passes 8190. Every case in that suite is on the near side of the
 * boundary this arithmetic creates:
 *
 *     history  8190 -> width 128   one launch, no split at all
 *     history  8191 -> width 127   splits 127 + 1   <-- a tail of ONE, at depth
 *     history 16510 -> width  63   splits 63*2 + 2
 *     history 32896 -> width  31   splits 31*4 + 4
 *     history 65664 -> width  15   splits 15*8 + 8
 *     history 131070 -> width  8   splits 8*16, exactly, no tail
 *     history 131071 -> width  7   splits 7*18 + 2   <-- the shipped context
 *
 * That last pair is worth spelling out, because I got it wrong first time and
 * this test caught it: 64 * 131074 = 8,388,736, which is just OVER 2^23, so
 * floor(1048576 / 131074) is 7 and not 8. The full 131,072 context therefore
 * runs 18 sub-batches of 7 plus a tail of 2 -- a tail, not the clean division
 * I had assumed.
 *
 * The 2026-09-26 review put it as finding 7: a 128K allocation is not 128K
 * qualification. This is the correctness half of that -- the deepest history the
 * shipped context permits, at the widths the shipped scratch budget produces,
 * including the boundary where the split first appears. It is precisely the
 * shape [[thresholds-hide-untested-code-paths]] warns about: a lane that is
 * stable everywhere tested because every test sits on one side of a threshold.
 *
 * WHAT THIS IS NOT. The KV here is synthetic -- random BF16 written directly to
 * device memory, not produced by running the model. That is deliberate: it buys
 * a 131,072-deep history in seconds instead of the many hours a real cold
 * prefill would take at ~10 tok/s. It therefore qualifies the ATTENTION KERNEL
 * at depth and says nothing about a filled-model context end to end. Do not cite
 * it as the latter.
 *
 * The oracle is the same one the shallow suite uses: each query re-run through
 * mimo26_rocm_attention_decode, which sees exactly the history that query may
 * see, compared bit-exact with no tolerance. Both paths run the same kernel, so
 * any difference is a masking, offset or indexing fault rather than arithmetic.
 */
#include "mimo26_architecture.h"
#include "mimo26_attention.h"
#include "mimo26_ops.h"
#include "mimo26_rocm_ops.h"
#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <ctime>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QH MIMO26_QUERY_HEADS
#define QK MIMO26_QK_HEAD_DIM
#define VD MIMO26_V_HEAD_DIM

/*
 * The shipped default, taken from the header rather than copied, so that
 * changing the default fails the boundary assertions below instead of silently
 * relocating the depths this suite covers. That coverage drift is the one thing
 * here the shallow suite cannot notice -- it would keep passing at any budget.
 */
#include "mimo26_gpu_worker.h"
#define PRODUCTION_SCRATCH_BYTES MIMO26_DEFAULT_ATTENTION_SCRATCH_BYTES
#define PRODUCTION_CHUNK 128u
#define PRODUCTION_CONTEXT 131072ull

static unsigned failures;
static unsigned ran;

static void ok(const char *what, int passed, const char *detail)
{
    printf("%s  %-62s %s\n", passed ? "PASS" : "FAIL", what, detail);
    fflush(stdout);
    ran++;
    failures += passed ? 0u : 1u;
}

#define HIP_OK(call)                                                          \
    do {                                                                      \
        const hipError_t status_ = (call);                                    \
        if (status_ != hipSuccess) {                                          \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call,   \
                    hipGetErrorString(status_));                              \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

static double seconds_now(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static uint32_t rng_state = 0x5EED1234u;
static float next_uniform(float low, float high)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    const float unit = (float)((rng_state >> 8) & 0xFFFFu) / 65535.0f;
    return low + unit * (high - low);
}

/* The width production would use at this history, by the kernel's own rule. */
static uint64_t production_width(uint64_t history, uint64_t chunk)
{
    const uint64_t row = mimo26_rocm_attention_scratch_floats(history);
    uint64_t width = (PRODUCTION_SCRATCH_BYTES / sizeof(float)) / row;
    if (width > chunk) width = chunk;
    if (width == 0u) width = 1u;
    return width;
}


/*
 * Timing sweep: does the scratch budget govern deep-prefill attention cost?
 *
 * At the shipped 256 MiB and a 131,072 history the width is 7, so a 128-token
 * chunk needs 19 sub-batch passes and each pass re-reads the whole history's KV.
 * A larger budget means fewer passes. If attention at depth is bandwidth-bound on
 * that re-reading, cost should fall roughly with the pass count -- and a full
 * 131,072 prefill, extrapolated at ~26 h from two live measurements, would be
 * dominated by it.
 *
 * Measures ONE global-layer attention call, which is where the cost is: 9 global
 * layers attend the whole history, the other 39 are capped at a 128 window.
 * Interleaved across budgets with a discarded warm-up, per
 * [[perf-screens-need-interleaved-baselines]] -- this box drifts ~2.6%.
 *
 * MIMO26_DEEP_BENCH=<repeats> selects this instead of the correctness cases.
 */
/*
 * Where does the time go inside one attention call?
 *
 * The kernel has four stages, and the budget sweep showed the cost is not the
 * sub-batch re-reads. mimo26_rocm_attention_scores runs stage 1 alone (it is the
 * scores gate's entry point), so subtracting it from a full call at the same
 * geometry separates:
 *
 *   stage 1  the 192-term QK dots, one slot per thread
 *   stage 2  exponentials, elementwise
 *   stage 2b the denominator -- summed in double by thread 0 ALONE, serially over
 *            every slot. Its comment calls this "a saving of microseconds", which
 *            was written when slots were few; at 131,072 it is 131,072 dependent
 *            adds with 255 threads idle.
 *   stage 3  value accumulation, split over only 128 output dims -- and it
 *            recomputes each slot's probability once PER DIM, so the divide and
 *            two BF16 conversions happen 128 times for every slot.
 *
 * Batch 1, because the scores entry point is fixed at one query. The ratio is
 * what matters, not the absolute.
 */
static int decompose(unsigned repeats)
{
    static const uint64_t depths[] = {8192u, 16384u, 32768u, 65536u, 131072u};
    const size_t kv_heads = 4u, kv_groups = (size_t)QH / 4u;
    printf("one attention call, batch 1, global layer, %u repeats\n\n", repeats);
    printf("%8s %11s %11s %11s %8s\n", "history", "stage1", "stages2+3",
           "total", "2+3 share");
    for (size_t d = 0; d < sizeof depths / sizeof *depths; d++) {
        const uint64_t total = depths[d];
        const size_t key_count = (size_t)total * kv_heads * QK;
        const size_t value_count = (size_t)total * kv_heads * VD;
        uint16_t *hk = (uint16_t *)malloc(key_count * 2);
        uint16_t *hv = (uint16_t *)malloc(value_count * 2);
        uint16_t *hq = (uint16_t *)malloc((size_t)QH * QK * 2);
        if (!hk || !hv || !hq) return 2;
        rng_state = 0xA11CEu + (uint32_t)d;
        for (size_t i = 0; i < key_count; i++) hk[i] = mimo26_f32_to_bf16(next_uniform(-1.5f, 1.5f));
        for (size_t i = 0; i < value_count; i++) hv[i] = mimo26_f32_to_bf16(next_uniform(-2.f, 2.f));
        for (size_t i = 0; i < (size_t)QH * QK; i++) hq[i] = mimo26_f32_to_bf16(next_uniform(-1.5f, 1.5f));

        void *dk = NULL, *dv = NULL, *dq = NULL, *dout = NULL;
        float *scratch = NULL;
        const uint64_t floats = mimo26_rocm_attention_scratch_floats(total);
        HIP_OK(hipMalloc(&dk, key_count * 2));
        HIP_OK(hipMalloc(&dv, value_count * 2));
        HIP_OK(hipMalloc(&dq, (size_t)QH * QK * 2));
        HIP_OK(hipMalloc(&dout, (size_t)QH * VD * 2));
        HIP_OK(hipMalloc(&scratch, floats * sizeof(float)));
        HIP_OK(hipMemcpy(dk, hk, key_count * 2, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(dv, hv, value_count * 2, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(dq, hq, (size_t)QH * QK * 2, hipMemcpyHostToDevice));
        free(hk); free(hv); free(hq);

        double t_scores = 0.0, t_full = 0.0;
        for (unsigned r = 0; r < repeats + 1u; r++) {
            HIP_OK(hipDeviceSynchronize());
            double t0 = seconds_now();
            if (!mimo26_rocm_attention_scores(dq, dk, NULL, NULL, scratch,
                                              (uint32_t)kv_heads, (uint32_t)kv_groups,
                                              0u, total, 0u, total - 1u,
                                              mimo26_attention_scale(), NULL)) return 2;
            HIP_OK(hipDeviceSynchronize());
            const double s = seconds_now() - t0;
            t0 = seconds_now();
            if (!mimo26_rocm_attention_prefill(dout, dq, dk, dv, NULL, scratch, floats,
                                               (uint32_t)kv_heads, (uint32_t)kv_groups,
                                               0u, total, 0u, total - 1u, 1u,
                                               mimo26_attention_scale(), NULL)) return 2;
            HIP_OK(hipDeviceSynchronize());
            const double f = seconds_now() - t0;
            if (r > 0u) { t_scores += s; t_full += f; }
        }
        t_scores /= repeats; t_full /= repeats;
        printf("%8llu %11.5f %11.5f %11.5f %7.0f%%\n",
               (unsigned long long)total, t_scores, t_full - t_scores, t_full,
               100.0 * (t_full - t_scores) / t_full);
        HIP_OK(hipFree(dk)); HIP_OK(hipFree(dv)); HIP_OK(hipFree(dq));
        HIP_OK(hipFree(dout)); HIP_OK(hipFree(scratch));
    }
    return 0;
}

static int bench(unsigned repeats)
{
    static const uint64_t depths[] = {16384u, 32768u, 65536u, 131072u};
    static const uint64_t budgets_mib[] = {256u, 512u, 1024u, 2048u, 4096u};
    const uint64_t chunk = PRODUCTION_CHUNK;
    const size_t kv_heads = 4u, kv_groups = (size_t)QH / 4u;   /* a global layer */

    printf("one global-layer attention call, chunk %llu, interleaved, %u repeats\n\n",
           (unsigned long long)chunk, repeats);
    printf("%8s %7s %6s %7s %11s %9s %8s\n", "history", "budget", "width",
           "passes", "seconds", "GB read", "GB/s");

    for (size_t d = 0; d < sizeof depths / sizeof *depths; d++) {
        const uint64_t total = depths[d], prior = total - chunk;
        const size_t key_count = (size_t)total * kv_heads * QK;
        const size_t value_count = (size_t)total * kv_heads * VD;

        uint16_t *keys = (uint16_t *)malloc(key_count * 2);
        uint16_t *values = (uint16_t *)malloc(value_count * 2);
        uint16_t *queries = (uint16_t *)malloc((size_t)chunk * QH * QK * 2);
        if (!keys || !values || !queries) return 2;
        rng_state = 0xBEEF00u + (uint32_t)d;
        for (size_t i = 0; i < key_count; i++) keys[i] = mimo26_f32_to_bf16(next_uniform(-1.5f, 1.5f));
        for (size_t i = 0; i < value_count; i++) values[i] = mimo26_f32_to_bf16(next_uniform(-2.f, 2.f));
        for (size_t i = 0; i < (size_t)chunk * QH * QK; i++)
            queries[i] = mimo26_f32_to_bf16(next_uniform(-1.5f, 1.5f));

        void *dk = NULL, *dv = NULL, *dq = NULL, *dout = NULL;
        HIP_OK(hipMalloc(&dk, key_count * 2));
        HIP_OK(hipMalloc(&dv, value_count * 2));
        HIP_OK(hipMalloc(&dq, (size_t)chunk * QH * QK * 2));
        HIP_OK(hipMalloc(&dout, (size_t)chunk * QH * VD * 2));
        HIP_OK(hipMemcpy(dk, keys, key_count * 2, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(dv, values, value_count * 2, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(dq, queries, (size_t)chunk * QH * QK * 2, hipMemcpyHostToDevice));
        free(keys); free(values); free(queries);

        const size_t budget_count = sizeof budgets_mib / sizeof *budgets_mib;
        double seconds[8] = {0};
        float *scratch[8] = {NULL};
        uint64_t width[8] = {0}, floats[8] = {0};
        const uint64_t row = mimo26_rocm_attention_scratch_floats(total);
        for (size_t b = 0; b < budget_count; b++) {
            uint64_t w = (budgets_mib[b] * 1024u * 1024u / sizeof(float)) / row;
            if (w > chunk) w = chunk;
            if (w == 0u) w = 1u;
            width[b] = w;
            floats[b] = w * row;
            HIP_OK(hipMalloc(&scratch[b], floats[b] * sizeof(float)));
        }
        /* Interleaved: one pass over every budget per repeat, warm-up discarded. */
        for (unsigned r = 0; r < repeats + 1u; r++) {
            for (size_t b = 0; b < budget_count; b++) {
                HIP_OK(hipDeviceSynchronize());
                const double t0 = seconds_now();
                if (!mimo26_rocm_attention_prefill(
                        dout, dq, dk, dv, NULL, scratch[b], floats[b],
                        (uint32_t)kv_heads, (uint32_t)kv_groups, 0u, total, 0u,
                        prior, (uint32_t)chunk, mimo26_attention_scale(), NULL)) {
                    fprintf(stderr, "launch failed\n");
                    return 2;
                }
                HIP_OK(hipDeviceSynchronize());
                if (r > 0u) seconds[b] += seconds_now() - t0;
            }
        }
        for (size_t b = 0; b < budget_count; b++) {
            const double sec = seconds[b] / repeats;
            const uint64_t passes = (chunk + width[b] - 1u) / width[b];
            /* each pass re-reads the whole history's keys and values */
            const double gb = (double)passes * (double)total * kv_heads *
                              (QK + VD) * 2.0 / 1e9;
            printf("%8llu %6lluM %6llu %7llu %11.5f %9.2f %8.0f\n",
                   (unsigned long long)total, (unsigned long long)budgets_mib[b],
                   (unsigned long long)width[b], (unsigned long long)passes,
                   sec, gb, gb / sec);
            HIP_OK(hipFree(scratch[b]));
        }
        printf("\n");
        HIP_OK(hipFree(dk)); HIP_OK(hipFree(dv));
        HIP_OK(hipFree(dq)); HIP_OK(hipFree(dout));
    }
    return 0;
}

int main(void)
{
    if (const char *d = getenv("MIMO26_DEEP_DECOMPOSE")) {
        int devices = 0;
        if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) return 2;
        return decompose((unsigned)strtoul(d, NULL, 10));
    }
    if (const char *b = getenv("MIMO26_DEEP_BENCH")) {
        int devices = 0;
        if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) return 2;
        return bench((unsigned)strtoul(b, NULL, 10));
    }
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        fprintf(stderr, "no HIP device\n");
        return 2;
    }

    /*
     * Priors chosen so that total = prior + chunk lands exactly where the width
     * arithmetic changes, plus the depths in between and the full context.
     *
     * 8062/8063 straddle the boundary: 8190 is the last history that fits a
     * full chunk in one launch, 8191 the first that does not -- and it splits
     * 127 + 1, so the final sub-batch is a single query at the deepest history
     * in the chunk. That is the case an off-by-one in the offset or the passed
     * history would corrupt, and nothing has ever run it.
     */
    static const uint64_t priors[] = {
        8062u,    /* total  8190 -> width 128, no split (the last such depth) */
        8063u,    /* total  8191 -> width 127, tail of 1                      */
        8064u,    /* total  8192 -> width 127, tail of 1                      */
        16382u,   /* total 16510 -> width  63, tail of 2                      */
        32768u,   /* total 32896 -> width  31, tail of 4                      */
        65536u,   /* total 65664 -> width  15, tail of 8                      */
        131070u - PRODUCTION_CHUNK,             /* total 131070, last width-8  */
        PRODUCTION_CONTEXT - PRODUCTION_CHUNK,  /* full context, width 7+tail 2 */
    };

    for (size_t case_index = 0; case_index < 2u * (sizeof priors / sizeof *priors);
         case_index++) {
        const bool is_swa = (case_index & 1u) != 0u;
        const uint64_t prior = priors[case_index / 2u];
        const uint64_t chunk = PRODUCTION_CHUNK;
        const uint64_t total = prior + chunk;
        const size_t kv_heads = is_swa ? 8u : 4u;
        const size_t kv_groups = (size_t)QH / kv_heads;
        const size_t window = is_swa ? MIMO26_SLIDING_WINDOW : 0u;

        const uint64_t width = production_width(total, chunk);
        const uint64_t row_floats = mimo26_rocm_attention_scratch_floats(total);
        /*
         * Allocated at exactly the width production would use -- so this is the
         * real width for this depth AND an overrun still faults rather than
         * scribbling on a larger buffer and passing.
         */
        const uint64_t scratch_floats = width * row_floats;

        rng_state = 0x5EED1234u + (uint32_t)case_index * 7919u;

        const size_t key_count = (size_t)total * kv_heads * QK;
        const size_t value_count = (size_t)total * kv_heads * VD;
        uint16_t *keys = (uint16_t *)malloc(key_count * sizeof *keys);
        uint16_t *values = (uint16_t *)malloc(value_count * sizeof *values);
        uint16_t *queries = (uint16_t *)malloc((size_t)chunk * QH * QK * sizeof *queries);
        uint16_t *sink = (uint16_t *)malloc(QH * sizeof *sink);
        if (!keys || !values || !queries || !sink) {
            fprintf(stderr, "host allocation failed at total %" PRIu64 "\n", total);
            return 2;
        }
        for (size_t i = 0; i < key_count; i++)
            keys[i] = mimo26_f32_to_bf16(next_uniform(-1.5f, 1.5f));
        for (size_t i = 0; i < value_count; i++)
            values[i] = mimo26_f32_to_bf16(next_uniform(-2.0f, 2.0f));
        for (size_t i = 0; i < (size_t)chunk * QH * QK; i++)
            queries[i] = mimo26_f32_to_bf16(next_uniform(-1.5f, 1.5f));
        for (size_t i = 0; i < (size_t)QH; i++)
            sink[i] = mimo26_f32_to_bf16(next_uniform(-2.0f, 2.0f));

        uint16_t *d_keys = NULL, *d_values = NULL, *d_queries = NULL;
        uint16_t *d_sink = NULL, *d_batched = NULL, *d_single = NULL;
        float *d_scratch = NULL;
        HIP_OK(hipMalloc(&d_keys, key_count * sizeof *keys));
        HIP_OK(hipMalloc(&d_values, value_count * sizeof *values));
        HIP_OK(hipMalloc(&d_queries, (size_t)chunk * QH * QK * sizeof *queries));
        HIP_OK(hipMalloc(&d_sink, QH * sizeof *sink));
        HIP_OK(hipMalloc(&d_batched, (size_t)chunk * QH * VD * sizeof *d_batched));
        HIP_OK(hipMalloc(&d_single, (size_t)QH * VD * sizeof *d_single));
        HIP_OK(hipMalloc(&d_scratch, scratch_floats * sizeof(float)));
        HIP_OK(hipMemcpy(d_keys, keys, key_count * sizeof *keys, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_values, values, value_count * sizeof *values, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_queries, queries, (size_t)chunk * QH * QK * sizeof *queries,
                         hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_sink, sink, QH * sizeof *sink, hipMemcpyHostToDevice));

        const bool launched = mimo26_rocm_attention_prefill(
            d_batched, d_queries, d_keys, d_values, is_swa ? d_sink : NULL,
            d_scratch, scratch_floats, (uint32_t)kv_heads, (uint32_t)kv_groups,
            (uint32_t)window, total, 0u, prior, (uint32_t)chunk,
            mimo26_attention_scale(), NULL);
        HIP_OK(hipDeviceSynchronize());

        uint16_t *batched = (uint16_t *)malloc((size_t)chunk * QH * VD * sizeof *batched);
        uint16_t *single = (uint16_t *)malloc((size_t)QH * VD * sizeof *single);
        if (!batched || !single) return 2;
        HIP_OK(hipMemcpy(batched, d_batched, (size_t)chunk * QH * VD * sizeof *batched,
                         hipMemcpyDeviceToHost));

        size_t differing = 0, first_query = (size_t)-1;
        for (uint64_t b = 0; launched && b < chunk; b++) {
            mimo26_rocm_attention_decode(
                d_single, (const uint8_t *)d_queries + b * QH * QK * 2u,
                (prior + b) ? d_keys : NULL, (prior + b) ? d_values : NULL,
                (const uint8_t *)d_keys + (prior + b) * kv_heads * QK * 2u,
                (const uint8_t *)d_values + (prior + b) * kv_heads * VD * 2u,
                is_swa ? d_sink : NULL, d_scratch, (uint32_t)kv_heads,
                (uint32_t)kv_groups, (uint32_t)window, prior + b, 0u,
                prior + b, mimo26_attention_scale(), NULL);
            HIP_OK(hipDeviceSynchronize());
            HIP_OK(hipMemcpy(single, d_single, (size_t)QH * VD * sizeof *single,
                             hipMemcpyDeviceToHost));
            size_t before = differing;
            for (size_t i = 0; i < (size_t)QH * VD; i++)
                differing += batched[b * QH * VD + i] != single[i];
            if (differing != before && first_query == (size_t)-1) first_query = (size_t)b;
        }

        char label[128], detail[160];
        snprintf(label, sizeof label,
                 "prior %6" PRIu64 " total %6" PRIu64 " %-6s width %3" PRIu64
                 " (%" PRIu64 "x%" PRIu64 "+%" PRIu64 ")",
                 prior, total, is_swa ? "swa" : "global", width,
                 chunk / width, width, chunk % width);
        if (first_query == (size_t)-1)
            snprintf(detail, sizeof detail, "%zu of %zu differ", differing,
                     (size_t)chunk * QH * VD);
        else
            snprintf(detail, sizeof detail, "%zu of %zu differ, first at query %zu",
                     differing, (size_t)chunk * QH * VD, first_query);
        ok(label, launched && differing == 0, detail);

        free(keys); free(values); free(queries); free(sink);
        free(batched); free(single);
        HIP_OK(hipFree(d_keys)); HIP_OK(hipFree(d_values));
        HIP_OK(hipFree(d_queries)); HIP_OK(hipFree(d_sink));
        HIP_OK(hipFree(d_batched)); HIP_OK(hipFree(d_single));
        HIP_OK(hipFree(d_scratch));
    }

    /*
     * Pin the boundary itself, so a change to the scratch budget or the row
     * formula that silently moves it fails here rather than quietly reducing
     * coverage back to "never splits".
     */
    static const struct { uint64_t history; uint64_t width; const char *what; }
    boundaries[] = {
        {8190u,  128u, "8190 is the last depth a full chunk runs in one launch"},
        {8191u,  127u, "8191 is the first depth that splits, tail of 1"},
        {8254u,  127u, "8254 still width 127"},
        {8255u,  126u, "8255 steps down to 126"},
        {131070u,  8u, "131070 is the last width-8 depth"},
        {131071u,  7u, "131071 steps down to 7"},
        {PRODUCTION_CONTEXT, 7u, "the 131072 context runs width 7, tail of 2"},
    };
    for (size_t i = 0; i < sizeof boundaries / sizeof *boundaries; i++) {
        const uint64_t got = production_width(boundaries[i].history, PRODUCTION_CHUNK);
        char detail[96];
        snprintf(detail, sizeof detail, "width %" PRIu64 ", expected %" PRIu64,
                 got, boundaries[i].width);
        ok(boundaries[i].what, got == boundaries[i].width, detail);
    }

    printf("\n%s %u of %u deep-attention cases\n",
           failures ? "FAIL" : "PASS", ran - failures, ran);
    return failures ? 1 : 0;
}
