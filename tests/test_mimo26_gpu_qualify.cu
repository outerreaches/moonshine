/*
 * Operational qualification for the GPU worker, mirroring
 * tests/test_mimo26_qualify.c.
 *
 * The CPU harness found a real bug on its first run -- a 4-slot expert cache
 * cannot serve a top-8 router, and the worker discovered it mid-decode
 * instead of refusing at create. The GPU worker inherited that floor by
 * construction, but nothing had checked the rest of its lifecycle, and it
 * carries state the CPU worker does not: a device-side packed cache with
 * eviction, a host-side KV mirrored up every layer, and a residency ledger
 * that a memory guard depends on.
 *
 * Same discipline as the CPU harness, and for the same reason: comparison is
 * bit-exact on the full logit vector, because a tolerance would hide exactly
 * the nondeterminism this exists to catch. Cross-backend agreement is a
 * different question, already bounded in G1-G5; this asks only whether the
 * GPU worker is reproducible against itself.
 *
 *   MIMO26_ROOT=/path/to/checkpoint tests/test_mimo26_gpu_qualify
 */
#include "mimo26_gpu_worker.h"

#include <hip/hip_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 152576u

/* "The quick brown fox jumps." -- exercises windowed and global attention
 * over a repeated span, and routes to a realistic spread of experts. */
static const uint32_t PROMPT[] = {785, 3974, 13876, 38835, 34208, 13};
#define PROMPT_LEN (sizeof PROMPT / sizeof PROMPT[0])

static int failures = 0;

static void ok(const char *what, int passed, const char *detail)
{
    printf("  %-4s %-52s %s\n", passed ? "ok" : "FAIL", what,
           detail ? detail : "");
    if (!passed) {
        failures++;
    }
}

static size_t first_difference(const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            return i;
        }
    }
    return n;
}

static void compare_logits(const char *what, const float *a, const float *b,
                           size_t n)
{
    char detail[192];
    const size_t at = first_difference(a, b, n);
    if (at == n) {
        snprintf(detail, sizeof detail, "%zu logits bit-identical", n);
        ok(what, 1, detail);
        return;
    }
    snprintf(detail, sizeof detail, "id %zu: %.9g vs %.9g", at, (double)a[at],
             (double)b[at]);
    ok(what, 0, detail);
}

static int decode_run(mimo26_gpu_worker *worker, const uint32_t *tokens,
                      size_t count, float *logits, size_t stride)
{
    char error[512];
    for (size_t i = 0; i < count; i++) {
        if (mimo26_gpu_worker_decode(worker, tokens[i], logits + i * stride,
                                     error, sizeof error) !=
            MIMO26_GPU_WORKER_OK) {
            fprintf(stderr, "decode of token %u failed: %s\n", tokens[i],
                    error);
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *root = getenv("MIMO26_ROOT");
    if (root == NULL) {
        root = "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL";
    }
    hipDeviceProp_t properties;
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        fprintf(stderr, "no device\n");
        return 1;
    }
    printf("gfx %s, %d CUs\n", properties.gcnArchName,
           properties.multiProcessorCount);

    char error[512];
    mimo26_gpu_worker_config config;
    mimo26_gpu_worker_config_defaults(&config);
    config.global_kv_capacity = 64u;

    /* A cache below the router's top-k can never serve a step. Refused at
     * create rather than discovered mid-token. */
    {
        mimo26_gpu_worker_config starved = config;
        starved.expert_slots_per_layer = 4u;
        mimo26_gpu_worker *refused = NULL;
        const mimo26_gpu_worker_status status = mimo26_gpu_worker_create(
            &refused, root, &starved, error, sizeof error);
        ok("fewer slots than the router's top-k is refused",
           status == MIMO26_GPU_WORKER_INVALID_ARGUMENT && refused == NULL,
           status == MIMO26_GPU_WORKER_INVALID_ARGUMENT ? error : "accepted");
        if (refused != NULL) {
            mimo26_gpu_worker_destroy(refused);
        }
    }

    /* The guard must refuse against measured availability before allocating,
     * not discover the shortfall by failing an allocation deep in loading. */
    {
        mimo26_gpu_worker_config tiny = config;
        tiny.expert_slots_per_layer = 8u;
        tiny.memory_limit_bytes = 1024u * 1024u;
        mimo26_gpu_worker *refused = NULL;
        const mimo26_gpu_worker_status status =
            mimo26_gpu_worker_create(&refused, root, &tiny, error,
                                     sizeof error);
        ok("memory guard refuses an over-small limit",
           status == MIMO26_GPU_WORKER_MEMORY_LIMIT && refused == NULL,
           status == MIMO26_GPU_WORKER_MEMORY_LIMIT ? error : "not refused");
        if (refused != NULL) {
            mimo26_gpu_worker_destroy(refused);
        }
    }

    float *a = (float *)malloc(PROMPT_LEN * VOCAB * sizeof *a);
    float *b = (float *)malloc(PROMPT_LEN * VOCAB * sizeof *b);
    float *c = (float *)malloc(PROMPT_LEN * VOCAB * sizeof *c);
    if (a == NULL || b == NULL || c == NULL) {
        return 1;
    }

    config.expert_slots_per_layer = 8u;
    mimo26_gpu_worker *worker = NULL;
    if (mimo26_gpu_worker_create(&worker, root, &config, error,
                                 sizeof error) != MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "create failed: %s\n", error);
        return 1;
    }
    {
        char detail[160];
        const uint64_t planned = mimo26_gpu_worker_planned_bytes(&config);
        const uint64_t resident = mimo26_gpu_worker_resident_bytes(worker);
        snprintf(detail, sizeof detail,
                 "resident %.2f GiB <= planned %.2f GiB",
                 (double)resident / 1073741824.0,
                 (double)planned / 1073741824.0);
        ok("residency stays within the plan", resident <= planned, detail);
    }

    if (!decode_run(worker, PROMPT, PROMPT_LEN, a, VOCAB)) {
        return 1;
    }
    ok("position counts committed tokens",
       mimo26_gpu_worker_position(worker) == PROMPT_LEN, NULL);

    /* Reset must return to a fresh state, and that state must decode
     * identically -- a leak in the KV or the expert cache shows up here and
     * nowhere else. */
    mimo26_gpu_worker_reset(worker);
    ok("reset returns the position to zero",
       mimo26_gpu_worker_position(worker) == 0, NULL);
    if (!decode_run(worker, PROMPT, PROMPT_LEN, b, VOCAB)) {
        return 1;
    }
    compare_logits("decoding is deterministic after reset", a, b,
                   PROMPT_LEN * VOCAB);

    /* Rollback and replay: what a rejected speculative block depends on. */
    {
        const size_t rewind = 2u;
        ok("rollback reports success",
           mimo26_gpu_worker_rollback(worker, rewind) ==
               MIMO26_GPU_WORKER_OK,
           NULL);
        ok("rollback rewinds the position",
           mimo26_gpu_worker_position(worker) == PROMPT_LEN - rewind, NULL);
        ok("rolling back further than the history is refused",
           mimo26_gpu_worker_rollback(worker, PROMPT_LEN * 4u) ==
               MIMO26_GPU_WORKER_INVALID_ARGUMENT,
           NULL);
        if (!decode_run(worker, PROMPT + (PROMPT_LEN - rewind), rewind,
                        c + (PROMPT_LEN - rewind) * VOCAB, VOCAB)) {
            return 1;
        }
        compare_logits("replay after rollback reproduces the logits",
                       a + (PROMPT_LEN - rewind) * VOCAB,
                       c + (PROMPT_LEN - rewind) * VOCAB, rewind * VOCAB);
    }

    /* A rejected token must not disturb committed history. */
    {
        const uint64_t before = mimo26_gpu_worker_position(worker);
        const mimo26_gpu_worker_status status =
            mimo26_gpu_worker_decode(worker, VOCAB + 7u, c, error,
                                     sizeof error);
        ok("an out-of-range token is rejected",
           status != MIMO26_GPU_WORKER_OK,
           status != MIMO26_GPU_WORKER_OK ? error : "accepted");
        ok("a rejected token leaves the position untouched",
           mimo26_gpu_worker_position(worker) == before, NULL);
        mimo26_gpu_worker_reset(worker);
        if (!decode_run(worker, PROMPT, PROMPT_LEN, c, VOCAB)) {
            return 1;
        }
        compare_logits("history survives a rejected token", a, c,
                       PROMPT_LEN * VOCAB);
    }

    /* The context ceiling must be refused at the boundary, not overrun. */
    {
        mimo26_gpu_worker_reset(worker);
        char detail[128];
        mimo26_gpu_worker_status status = MIMO26_GPU_WORKER_OK;
        size_t steps = 0;
        while (steps < config.global_kv_capacity + 4u) {
            status = mimo26_gpu_worker_decode(worker, PROMPT[steps % PROMPT_LEN],
                                              c, error, sizeof error);
            if (status != MIMO26_GPU_WORKER_OK) {
                break;
            }
            steps++;
        }
        snprintf(detail, sizeof detail, "stopped after %zu of %zu", steps,
                 config.global_kv_capacity);
        ok("the context ceiling is refused, not overrun",
           status == MIMO26_GPU_WORKER_CAPACITY_EXCEEDED &&
               steps == config.global_kv_capacity,
           detail);
    }

    mimo26_gpu_worker_stats stats;
    mimo26_gpu_worker_get_stats(worker, &stats);
    ok("no step aborted", stats.aborted_steps == 0, NULL);
    mimo26_gpu_worker_destroy(worker);

    /*
     * Above the floor the expert cache is a cache. Changing its size changes
     * the upload count and must change nothing else -- the property that
     * makes a residency policy safe to tune.
     */
    {
        mimo26_gpu_worker_config roomy = config;
        roomy.expert_slots_per_layer = 24u;
        mimo26_gpu_worker *wide = NULL;
        if (mimo26_gpu_worker_create(&wide, root, &roomy, error,
                                     sizeof error) != MIMO26_GPU_WORKER_OK) {
            fprintf(stderr, "wide create failed: %s\n", error);
            return 1;
        }
        if (!decode_run(wide, PROMPT, PROMPT_LEN, b, VOCAB)) {
            return 1;
        }
        compare_logits("24 expert slots match 8 exactly", a, b,
                       PROMPT_LEN * VOCAB);
        mimo26_gpu_worker_get_stats(wide, &stats);
        char detail[128];
        snprintf(detail, sizeof detail, "%llu uploads, %llu accesses",
                 (unsigned long long)stats.expert_uploads,
                 (unsigned long long)stats.expert_accesses);
        ok("the wider worker still uploaded experts",
           stats.expert_uploads > 0, detail);
        ok("no step aborted at the wider size", stats.aborted_steps == 0,
           NULL);
        mimo26_gpu_worker_destroy(wide);
    }

    free(a);
    free(b);
    free(c);
    printf("test_mimo26_gpu_qualify: %s\n", failures == 0 ? "ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
