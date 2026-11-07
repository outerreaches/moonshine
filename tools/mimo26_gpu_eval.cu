/*
 * Teacher-forced evaluation of the MiMo GPU worker over a held-out token
 * sequence, for the M4 functional-quality gate.
 *
 * Reads token ids (whitespace-separated) on stdin, feeds every one, and for
 * each position emits one JSON line describing the prediction of the *next*
 * token: its rank, its logit, its log-probability under a full-vocabulary
 * log-softmax, and the argmax. Position 0's prediction is scored against
 * token 1, and the final token has nothing to score, so a sequence of n
 * tokens yields n-1 records.
 *
 * Teacher forcing matters here: each step is fed the true token, so one bad
 * prediction cannot cascade into the rest of the passage and turn a single
 * defect into an apparently total failure.
 *
 * The thresholds live in tests/mimo26_eval_spec.json and are applied by
 * tests/score_mimo26_eval.py, deliberately not by this tool -- it reports,
 * it does not judge.
 *
 *   tests/mimo26_tokenize.py < text | tools/mimo26_eval ROOT > eval.jsonl
 */
#include "mimo26_gpu_worker.h"

#include <hip/hip_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 152576u

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: %s ROOT [--slots N] < tokens\n", argv[0]);
        return 2;
    }
    const char *root = argv[1];

    mimo26_gpu_worker_config config;
    mimo26_gpu_worker_config_defaults(&config);
    for (int i = 2; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--slots") == 0) {
            config.expert_slots_per_layer =
                (uint16_t)strtoul(argv[i + 1], NULL, 10);
        }
    }

    size_t capacity = 1024;
    size_t count = 0;
    uint32_t *tokens = (uint32_t *)malloc(capacity * sizeof *tokens);
    if (tokens == NULL) {
        return 1;
    }
    unsigned long value = 0;
    while (scanf("%lu", &value) == 1) {
        if (count == capacity) {
            capacity *= 2;
            uint32_t *grown = (uint32_t *)realloc(tokens, capacity * sizeof *tokens);
            if (grown == NULL) {
                free(tokens);
                return 1;
            }
            tokens = grown;
        }
        tokens[count++] = (uint32_t)value;
    }
    if (count < 2) {
        fprintf(stderr, "need at least two tokens to score a prediction\n");
        free(tokens);
        return 2;
    }

    char error[512];
    mimo26_gpu_worker *worker = NULL;
    if (mimo26_gpu_worker_create(&worker, root, &config, error, sizeof error) !=
        MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "worker create failed: %s\n", error);
        free(tokens);
        return 1;
    }

    float *logits = (float *)malloc((size_t)VOCAB * sizeof *logits);
    if (logits == NULL) {
        mimo26_gpu_worker_destroy(worker);
        free(tokens);
        return 1;
    }

    for (size_t i = 0; i + 1 < count; i++) {
        if (mimo26_gpu_worker_decode(worker, tokens[i], logits, error,
                                 sizeof error) != MIMO26_GPU_WORKER_OK) {
            fprintf(stderr, "decode at position %zu failed: %s\n", i, error);
            free(logits);
            free(tokens);
            mimo26_gpu_worker_destroy(worker);
            return 1;
        }

        const uint32_t truth = tokens[i + 1];
        const float truth_logit = logits[truth];

        /* Rank, non-finite count and the log-softmax denominator in one pass
         * over the vocabulary. The padded rows are -infinity by contract, so
         * they neither rank nor contribute; anything else non-finite is a
         * defect and is counted rather than silently skipped. */
        uint64_t better = 0;
        uint64_t nonfinite = 0;
        float max_logit = -INFINITY;
        for (uint32_t id = 0; id < VOCAB; id++) {
            const float l = logits[id];
            if (isinf(l) && l < 0.0f && id >= MIMO26_GPU_TOKENIZER_VOCAB) {
                continue;  /* masked padding row, by contract */
            }
            if (!isfinite(l)) {
                nonfinite++;
                continue;
            }
            if (l > truth_logit) {
                better++;
            }
            if (l > max_logit) {
                max_logit = l;
            }
        }
        double sum = 0.0;
        for (uint32_t id = 0; id < MIMO26_GPU_TOKENIZER_VOCAB; id++) {
            const float l = logits[id];
            if (isfinite(l)) {
                sum += exp((double)(l - max_logit));
            }
        }
        const double logprob =
            (double)(truth_logit - max_logit) - log(sum);

        printf("{\"position\":%zu,\"fed\":%u,\"truth\":%u,\"rank\":%llu,"
               "\"logit\":%.6f,\"logprob\":%.6f,\"argmax\":%u,"
               "\"nonfinite\":%llu}\n",
               i, tokens[i], truth, (unsigned long long)(better + 1),
               (double)truth_logit, logprob, mimo26_gpu_worker_argmax(logits),
               (unsigned long long)nonfinite);
    }

    mimo26_gpu_worker_stats stats;
    mimo26_gpu_worker_get_stats(worker, &stats);
    printf("{\"summary\":true,\"tokens\":%llu,\"expert_accesses\":%llu,"
           "\"expert_hits\":%llu,\"expert_loads\":%llu,"
           "\"aborted_steps\":%llu,\"resident_bytes\":%llu,"
           "\"slots_per_layer\":%u}\n",
           (unsigned long long)stats.tokens,
           (unsigned long long)stats.expert_accesses,
           (unsigned long long)stats.expert_hits,
           (unsigned long long)stats.expert_uploads,
           (unsigned long long)stats.aborted_steps,
           (unsigned long long)mimo26_gpu_worker_resident_bytes(worker),
           (unsigned)config.expert_slots_per_layer);

    free(logits);
    free(tokens);
    mimo26_gpu_worker_destroy(worker);
    return 0;
}
