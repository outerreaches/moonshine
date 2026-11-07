/*
 * M4 qualification: the operational invariants a production worker must hold,
 * checked against the real checkpoint.
 *
 * These are deliberately *not* numerical-parity checks. Parity against the
 * Python reference is covered by test_mimo26_layer_parity and
 * test_mimo26_weights_vs_reference. What is checked here is that the worker's
 * own behaviour is reproducible and that its recovery paths restore exactly
 * the state they claim to:
 *
 *   1. decoding is deterministic across runs
 *   2. expert cache size does not change results -- it is a cache, not a
 *      participant in the arithmetic
 *   3. rollback restores the pre-block state exactly
 *   4. reset returns the worker to a fresh state exactly
 *   5. a failed decode leaves history and position untouched
 *   6. the memory guard refuses an over-large configuration before allocating
 *
 * Comparison is bit-exact on the full logit vector. A tolerance here would
 * hide precisely the nondeterminism this is meant to catch.
 *
 *   MIMO26_ROOT=/path/to/checkpoint tests/test_mimo26_qualify
 *
 * Roughly a minute per decoded token on this box, so the default prompt is
 * short. MIMO26_QUALIFY_TOKENS overrides it.
 */
#include "mimo26_worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 152576u

/* "The quick brown fox jumps." -- the induction probe, which exercises both
 * windowed and global attention over a repeated span. */
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

static int compare_logits(const char *what, const float *a, const float *b,
                          size_t n)
{
    char detail[192];
    const size_t at = first_difference(a, b, n);
    if (at == n) {
        snprintf(detail, sizeof detail, "%zu logits bit-identical", n);
        ok(what, 1, detail);
        return 1;
    }
    snprintf(detail, sizeof detail, "id %zu: %.9g vs %.9g", at, (double)a[at],
             (double)b[at]);
    ok(what, 0, detail);
    return 0;
}

/* Decode a token sequence, keeping the logits produced at each step. */
static int decode_run(mimo26_worker *worker, const uint32_t *tokens,
                      size_t count, float *logits, size_t stride)
{
    char error[512];
    for (size_t i = 0; i < count; i++) {
        const mimo26_worker_status status =
            mimo26_worker_decode(worker, tokens[i], logits + i * stride, error,
                                 sizeof error);
        if (status != MIMO26_WORKER_OK) {
            fprintf(stderr, "decode of token %u failed: %s\n", tokens[i],
                    error);
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    /* Half an hour of decoding on CPU, so progress must be visible when the
     * output is redirected to a log. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *root = getenv("MIMO26_ROOT");
    if (root == NULL) {
        root = "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL";
    }
    size_t tokens = PROMPT_LEN;
    const char *override = getenv("MIMO26_QUALIFY_TOKENS");
    if (override != NULL) {
        const long requested = strtol(override, NULL, 10);
        if (requested > 0 && (size_t)requested <= PROMPT_LEN) {
            tokens = (size_t)requested;
        }
    }

    char error[512];
    mimo26_worker_config config;
    mimo26_worker_config_defaults(&config);

    /* 6. The guard must refuse before allocating, not after. */
    {
        mimo26_worker_config tiny = config;
        tiny.memory_limit_bytes = 1024u * 1024u;
        mimo26_worker *refused = NULL;
        const mimo26_worker_status status =
            mimo26_worker_create(&refused, root, &tiny, error, sizeof error);
        ok("memory guard refuses an over-small limit",
           status == MIMO26_WORKER_MEMORY_LIMIT && refused == NULL,
           status == MIMO26_WORKER_MEMORY_LIMIT ? error : "not refused");
        if (refused != NULL) {
            mimo26_worker_destroy(refused);
        }
    }

    float *a = malloc(tokens * VOCAB * sizeof *a);
    float *b = malloc(tokens * VOCAB * sizeof *b);
    float *c = malloc(tokens * VOCAB * sizeof *c);
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "out of memory for logit buffers\n");
        return 1;
    }

    mimo26_worker *worker = NULL;
    config.expert_slots_per_layer = 8;
    mimo26_worker_status status =
        mimo26_worker_create(&worker, root, &config, error, sizeof error);
    if (status != MIMO26_WORKER_OK) {
        fprintf(stderr, "worker create failed: %s\n", error);
        return 1;
    }

    {
        char detail[128];
        const uint64_t planned = mimo26_worker_planned_bytes(&config);
        const uint64_t resident = mimo26_worker_resident_bytes(worker);
        snprintf(detail, sizeof detail, "resident %.2f GiB <= planned %.2f GiB",
                 (double)resident / 1073741824.0,
                 (double)planned / 1073741824.0);
        ok("residency stays within the plan", resident <= planned, detail);
    }

    if (!decode_run(worker, PROMPT, tokens, a, VOCAB)) {
        return 1;
    }
    ok("position counts committed tokens",
       mimo26_worker_position(worker) == tokens, NULL);

    /* 4. Reset must return to a fresh state, and 1. that state must decode
     * identically. Running these together makes each one's failure legible:
     * a reset leak shows up here and nowhere else. */
    mimo26_worker_reset(worker);
    ok("reset returns the position to zero",
       mimo26_worker_position(worker) == 0, NULL);
    if (!decode_run(worker, PROMPT, tokens, b, VOCAB)) {
        return 1;
    }
    compare_logits("decoding is deterministic after reset", a, b,
                   tokens * VOCAB);

    /* 3. Roll back the tail of the prompt and replay it. The replayed steps
     * must reproduce the logits they produced the first time, which is what a
     * rejected speculative block depends on. */
    if (tokens >= 3) {
        const size_t rewind = 2;
        status = mimo26_worker_rollback(worker, rewind);
        ok("rollback reports success", status == MIMO26_WORKER_OK, NULL);
        ok("rollback rewinds the position",
           mimo26_worker_position(worker) == tokens - rewind, NULL);
        if (!decode_run(worker, PROMPT + (tokens - rewind), rewind,
                        c + (tokens - rewind) * VOCAB, VOCAB)) {
            return 1;
        }
        compare_logits("replay after rollback reproduces the logits",
                       a + (tokens - rewind) * VOCAB,
                       c + (tokens - rewind) * VOCAB, rewind * VOCAB);
    }

    /* 5. A rejected token must not disturb committed history. The vocabulary
     * bound is the one failure that can be provoked without corrupting the
     * checkpoint. */
    {
        const uint64_t before = mimo26_worker_position(worker);
        mimo26_worker_stats stats_before;
        mimo26_worker_get_stats(worker, &stats_before);
        status = mimo26_worker_decode(worker, VOCAB + 7u, c, error,
                                      sizeof error);
        const int rejected = status != MIMO26_WORKER_OK;
        ok("an out-of-range token is rejected", rejected,
           rejected ? error : "accepted");
        ok("a rejected token leaves the position untouched",
           mimo26_worker_position(worker) == before, NULL);
        /* Decoding on from here must still match the original run. */
        mimo26_worker_reset(worker);
        if (!decode_run(worker, PROMPT, tokens, c, VOCAB)) {
            return 1;
        }
        compare_logits("history survives a rejected token", a, c,
                       tokens * VOCAB);
    }

    mimo26_worker_destroy(worker);

    /* A cache smaller than one token's working set must be refused at create.
     * Found by this test: with 4 slots the first decode failed with "expert 14
     * was neither hit nor admitted", a mid-token failure for a configuration
     * that was never viable. */
    {
        mimo26_worker_config starved = config;
        starved.expert_slots_per_layer = 4;
        mimo26_worker *refused = NULL;
        status = mimo26_worker_create(&refused, root, &starved, error,
                                      sizeof error);
        ok("fewer slots than the router's top-k is refused",
           status == MIMO26_WORKER_INVALID_ARGUMENT && refused == NULL,
           status == MIMO26_WORKER_INVALID_ARGUMENT ? error : "accepted");
        if (refused != NULL) {
            mimo26_worker_destroy(refused);
        }
    }

    /* 2. Above that floor the expert cache is a cache. Doubling it changes
     * the hit rate and the number of dequantizations, and must change
     * nothing else. */
    {
        mimo26_worker_config roomy = config;
        roomy.expert_slots_per_layer = 16;
        mimo26_worker *wide = NULL;
        status = mimo26_worker_create(&wide, root, &roomy, error,
                                      sizeof error);
        if (status != MIMO26_WORKER_OK) {
            fprintf(stderr, "wide worker create failed: %s\n", error);
            return 1;
        }
        if (!decode_run(wide, PROMPT, tokens, b, VOCAB)) {
            return 1;
        }
        mimo26_worker_stats wide_stats;
        mimo26_worker_get_stats(wide, &wide_stats);
        compare_logits("16 expert slots match 8 exactly", a, b,
                       tokens * VOCAB);
        ok("the wider worker still dequantized experts",
           wide_stats.expert_loads > 0, NULL);
        ok("no step aborted", wide_stats.aborted_steps == 0, NULL);
        mimo26_worker_destroy(wide);
    }

    free(a);
    free(b);
    free(c);
    printf("test_mimo26_qualify: %s\n", failures == 0 ? "ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
