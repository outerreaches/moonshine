/*
 * Drive the full 48-layer worker over a token sequence.
 *
 *   mimo26_run CHECKPOINT_ROOT [--slots N] [--context N] [--limit-gib X]
 *              [--generate N] TOKEN [TOKEN ...]
 *
 * Prompt tokens are fed in order; then --generate steps are taken greedily,
 * each feeding back the previous argmax. Token ids are numeric because this
 * binary deliberately carries no tokenizer -- rendering and tokenization are
 * pinned separately in tests/fixtures/mimo26_tokenizer_v1.json.
 */
#include "mimo26_worker.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void trace_layer(void *context, uint32_t layer, const uint16_t *hidden,
                        size_t count)
{
    (void)context;
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        const uint32_t bits = (uint32_t)hidden[i] << 16;
        float value;
        memcpy(&value, &bits, sizeof value);
        sum += (double)value * (double)value;
    }
    printf("    layer %2u  rms %.5f\n", layer, sqrt(sum / (double)count));
}

static void summarize(const float *logits, uint32_t chosen)
{
    double maximum = -INFINITY;
    double sum = 0.0;
    size_t finite = 0;
    for (uint32_t i = 0; i < MIMO26_TOKENIZER_VOCAB; i++) {
        if (!isfinite(logits[i])) {
            continue;
        }
        finite++;
        if (logits[i] > maximum) {
            maximum = logits[i];
        }
    }
    for (uint32_t i = 0; i < MIMO26_TOKENIZER_VOCAB; i++) {
        if (isfinite(logits[i])) {
            sum += exp((double)logits[i] - maximum);
        }
    }
    const double probability = sum > 0.0 ? 1.0 / sum : 0.0;
    printf("    top id %-7u logit %8.4f  p=%.4f  finite=%zu  masked=%u\n",
           chosen, (double)logits[chosen], probability, finite,
           MIMO26_VOCAB_SIZE - MIMO26_TOKENIZER_VOCAB);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s ROOT [--slots N] [--context N] [--limit-gib X] "
                "[--generate N] TOKEN...\n", argv[0]);
        return 2;
    }
    /* Line-buffer so progress is visible when stdout is a file or a pipe:
     * a full 48-layer CPU step takes tens of seconds, and block buffering
     * makes a working run look like a hung one. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *root = argv[1];
    mimo26_worker_config config;
    mimo26_worker_config_defaults(&config);
    size_t generate = 0;
    bool trace = false;

    uint32_t prompt[256];
    size_t prompt_length = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc) {
            config.expert_slots_per_layer = (uint16_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            config.global_kv_capacity = strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--limit-gib") == 0 && i + 1 < argc) {
            config.memory_limit_bytes =
                (uint64_t)(strtod(argv[++i], NULL) * (double)(1u << 30));
        } else if (strcmp(argv[i], "--trace") == 0) {
            trace = true;
        } else if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate = strtoul(argv[++i], NULL, 10);
        } else if (prompt_length < sizeof prompt / sizeof prompt[0]) {
            prompt[prompt_length++] = (uint32_t)strtoul(argv[i], NULL, 10);
        }
    }
    if (prompt_length == 0) {
        fprintf(stderr, "no prompt tokens given\n");
        return 2;
    }

    printf("planned residency %.2f GiB (%u expert slots per layer, "
           "context %zu)\n",
           (double)mimo26_worker_planned_bytes(&config) / (double)(1u << 30),
           config.expert_slots_per_layer, config.global_kv_capacity);

    char error[512] = {0};
    mimo26_worker *worker = NULL;
    const mimo26_worker_status created =
        mimo26_worker_create(&worker, root, &config, error, sizeof error);
    if (created != MIMO26_WORKER_OK) {
        fprintf(stderr, "worker create failed (%d): %s\n", (int)created, error);
        return 1;
    }
    if (trace) {
        mimo26_worker_set_trace(worker, trace_layer, NULL);
    }
    printf("resident %.2f GiB after load\n",
           (double)mimo26_worker_resident_bytes(worker) / (double)(1u << 30));

    float *logits = malloc((size_t)MIMO26_VOCAB_SIZE * sizeof *logits);
    if (logits == NULL) {
        mimo26_worker_destroy(worker);
        return 1;
    }

    int result = 1;
    uint32_t next = 0;
    printf("\nprompt (%zu tokens)\n", prompt_length);
    for (size_t i = 0; i < prompt_length; i++) {
        const mimo26_worker_status status =
            mimo26_worker_decode(worker, prompt[i], logits, error,
                                 sizeof error);
        if (status != MIMO26_WORKER_OK) {
            fprintf(stderr, "decode failed at prompt %zu (%d): %s\n", i,
                    (int)status, error);
            goto done;
        }
        next = mimo26_worker_argmax(logits);
        mimo26_worker_stats stats;
        mimo26_worker_get_stats(worker, &stats);
        printf("  pos %-4llu token %-7u  %6.2fs  hits %llu/%llu\n",
               (unsigned long long)(mimo26_worker_position(worker) - 1u),
               prompt[i], stats.last_decode_seconds,
               (unsigned long long)stats.expert_hits,
               (unsigned long long)stats.expert_accesses);
        summarize(logits, next);
    }

    if (generate > 0) {
        printf("\ngreedy continuation (%zu tokens)\n", generate);
        for (size_t i = 0; i < generate; i++) {
            const mimo26_worker_status status =
                mimo26_worker_decode(worker, next, logits, error,
                                     sizeof error);
            if (status != MIMO26_WORKER_OK) {
                fprintf(stderr, "decode failed at step %zu (%d): %s\n", i,
                        (int)status, error);
                goto done;
            }
            const uint32_t chosen = mimo26_worker_argmax(logits);
            mimo26_worker_stats stats;
            mimo26_worker_get_stats(worker, &stats);
            printf("  pos %-4llu fed %-7u -> %-7u  %6.2fs  hits %llu/%llu\n",
                   (unsigned long long)(mimo26_worker_position(worker) - 1u),
                   next, chosen, stats.last_decode_seconds,
                   (unsigned long long)stats.expert_hits,
                   (unsigned long long)stats.expert_accesses);
            next = chosen;
        }
    }

    {
        mimo26_worker_stats stats;
        mimo26_worker_get_stats(worker, &stats);
        printf("\ntokens %llu, expert loads %llu, hit rate %.1f%%, "
               "aborted steps %llu\n",
               (unsigned long long)stats.tokens,
               (unsigned long long)stats.expert_loads,
               stats.expert_accesses > 0u
                   ? 100.0 * (double)stats.expert_hits /
                         (double)stats.expert_accesses
                   : 0.0,
               (unsigned long long)stats.aborted_steps);
    }
    result = 0;
done:
    free(logits);
    mimo26_worker_destroy(worker);
    return result;
}
