/*
 * Drive the GPU worker over a prompt and optionally continue it greedily.
 *
 * The CPU counterpart is tools/mimo26_run. Keeping both means a divergence
 * can always be localized to a backend rather than argued about.
 *
 *   tools/mimo26_gpu_run ROOT [--slots N] [--context N] [--generate N] TOKEN...
 */
#include "mimo26_gpu_worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 152576u

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s ROOT [--slots N] [--context N] [--generate N] "
                "TOKEN...\n", argv[0]);
        return 2;
    }
    const char *root = argv[1];
    mimo26_gpu_worker_config config;
    mimo26_gpu_worker_config_defaults(&config);
    unsigned generate = 0;

    int index = 2;
    while (index + 1 < argc && argv[index][0] == '-') {
        if (strcmp(argv[index], "--slots") == 0) {
            config.expert_slots_per_layer =
                (uint16_t)strtoul(argv[index + 1], NULL, 10);
        } else if (strcmp(argv[index], "--context") == 0) {
            config.global_kv_capacity = strtoul(argv[index + 1], NULL, 10);
        } else if (strcmp(argv[index], "--generate") == 0) {
            generate = (unsigned)strtoul(argv[index + 1], NULL, 10);
        } else {
            break;
        }
        index += 2;
    }

    const int first_token = index;
    const int token_count = argc - first_token;
    if (token_count <= 0) {
        fprintf(stderr, "no tokens given\n");
        return 2;
    }

    printf("planned %.2f GiB (%u packed expert slots per layer, context %zu)\n",
           (double)mimo26_gpu_worker_planned_bytes(&config) / 1073741824.0,
           (unsigned)config.expert_slots_per_layer,
           config.global_kv_capacity);

    char error[512];
    mimo26_gpu_worker *worker = NULL;
    if (mimo26_gpu_worker_create(&worker, root, &config, error,
                                 sizeof error) != MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "create failed: %s\n", error);
        return 1;
    }
    mimo26_gpu_worker_stats stats;
    mimo26_gpu_worker_get_stats(worker, &stats);
    printf("resident %.2f GiB after %.1f s of loading\n\n",
           (double)mimo26_gpu_worker_resident_bytes(worker) / 1073741824.0,
           stats.load_seconds);

    float *logits = (float *)malloc((size_t)VOCAB * sizeof *logits);
    if (logits == NULL) {
        return 1;
    }

    printf("prompt (%d tokens)\n", token_count);
    uint32_t last = 0;
    for (int i = 0; i < token_count; i++) {
        const uint32_t token = (uint32_t)strtoul(argv[first_token + i], NULL,
                                                 10);
        if (mimo26_gpu_worker_decode(worker, token, logits, error,
                                     sizeof error) != MIMO26_GPU_WORKER_OK) {
            fprintf(stderr, "decode failed: %s\n", error);
            return 1;
        }
        mimo26_gpu_worker_get_stats(worker, &stats);
        last = mimo26_gpu_worker_argmax(logits);
        printf("  pos %-4d token %-8u %6.2f s  -> %-8u logit %8.4f  "
               "resident %llu/%llu\n",
               i, token, stats.last_decode_seconds, last,
               (double)logits[last],
               (unsigned long long)stats.expert_hits,
               (unsigned long long)stats.expert_accesses);
    }

    if (generate > 0) {
        printf("\ngreedy continuation (%u tokens)\n", generate);
        for (unsigned g = 0; g < generate; g++) {
            const uint32_t fed = last;
            if (mimo26_gpu_worker_decode(worker, fed, logits, error,
                                         sizeof error) !=
                MIMO26_GPU_WORKER_OK) {
                fprintf(stderr, "decode failed: %s\n", error);
                return 1;
            }
            mimo26_gpu_worker_get_stats(worker, &stats);
            last = mimo26_gpu_worker_argmax(logits);
            printf("  fed %-8u -> %-8u %6.2f s\n", fed, last,
                   stats.last_decode_seconds);
        }
    }

    mimo26_gpu_worker_get_stats(worker, &stats);
    printf("\ntokens %llu, uploads %llu, hit rate %.1f%%, aborted %llu\n",
           (unsigned long long)stats.tokens,
           (unsigned long long)stats.expert_uploads,
           stats.expert_accesses
               ? 100.0 * (double)stats.expert_hits /
                     (double)stats.expert_accesses
               : 0.0,
           (unsigned long long)stats.aborted_steps);
    free(logits);
    mimo26_gpu_worker_destroy(worker);
    return 0;
}
