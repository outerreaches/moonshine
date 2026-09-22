/*
 * Walk the context ladder the plan asks for, one measured gate at a time.
 *
 * A context depth is not "supported" because the KV fits. The plan asks for
 * prefill latency, retrieval, replay and memory at each rung, so this checks
 * all four and prints what it measured rather than a verdict alone:
 *
 *   PREFILL   wall time for the whole prompt, and per token
 *   RETRIEVAL a fact placed early has to be recoverable from the end, which
 *             is what distinguishes a model that holds context from one that
 *             merely accepts it without erroring
 *   REPLAY    the same prompt twice must give bit-identical logits at depth,
 *             where a KV indexing or eviction bug would first show
 *   MEMORY    resident before and after, since a per-token leak is invisible
 *             at 512 and fatal at 8K
 *
 * The needle is placed early on purpose. A fact near the end is recoverable
 * from the sliding window alone and would pass at any depth, proving
 * nothing about the global layers.
 *
 *   tools/mimo26_context_gate ROOT [--depths 512,2048] [--slots N]
 */
#include "mimo26_gpu_worker.h"
#include "mimo26_tokenizer.h"

#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VOCAB 152576u

static int failures = 0;

static void ok(const char *what, int passed, const char *detail)
{
    printf("    %-4s %-34s %s\n", passed ? "ok" : "FAIL", what,
           detail ? detail : "");
    if (!passed) {
        failures++;
    }
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/*
 * Filler that reads as ordinary prose rather than a repeated token. A
 * degenerate filler compresses into a handful of distinct KV entries and
 * would make retrieval easier than any real workload.
 */
static const char *FILLER[] = {
    "The harbour master recorded the tide each morning before the boats left.",
    "Gulls followed the trawlers out past the breakwater and back again.",
    "Rain moved in from the west and the glass fell steadily through the day.",
    "The ferry ran late, as it often did when the wind came from the north.",
    "Nets were mended on the quay by men who had done it for forty years.",
    "A chandlery on the front sold rope, paint and tinned food to the crews.",
    "The lighthouse keeper kept his own log, separate from the harbour's.",
    "Children counted the boats returning and argued about which was first.",
};
#define FILLER_COUNT (sizeof FILLER / sizeof FILLER[0])

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: %s ROOT [--depths a,b,c] [--slots N]\n",
                argv[0]);
        return 2;
    }
    const char *root = argv[1];
    size_t depths[8] = {512u, 2048u};
    size_t depth_count = 2u;
    uint16_t slots = 48u;

    for (int i = 2; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--depths") == 0) {
            depth_count = 0;
            char *copy = strdup(argv[i + 1]);
            for (char *token = strtok(copy, ","); token != NULL && depth_count < 8;
                 token = strtok(NULL, ",")) {
                depths[depth_count++] = strtoul(token, NULL, 10);
            }
            free(copy);
        } else if (strcmp(argv[i], "--slots") == 0) {
            slots = (uint16_t)strtoul(argv[i + 1], NULL, 10);
        }
    }

    char error[512];
    mimo26_tokenizer *tokenizer = NULL;
    if (!mimo26_tokenizer_create(&tokenizer, root, error, sizeof error)) {
        fprintf(stderr, "tokenizer: %s\n", error);
        return 1;
    }

    for (size_t d = 0; d < depth_count; d++) {
        const size_t depth = depths[d];
        printf("\n=== context %zu ===\n", depth);

        mimo26_gpu_worker_config config;
        mimo26_gpu_worker_config_defaults(&config);
        config.global_kv_capacity = depth + 128u;   /* room for the answer */
        config.expert_slots_per_layer = slots;

        mimo26_gpu_worker *worker = NULL;
        const double load_started = now_seconds();
        if (mimo26_gpu_worker_create(&worker, root, &config, error,
                                     sizeof error) !=
            MIMO26_GPU_WORKER_OK) {
            fprintf(stderr, "  worker: %s\n", error);
            return 1;
        }
        const uint64_t resident_before =
            mimo26_gpu_worker_resident_bytes(worker);
        printf("    loaded %.2f GiB in %.1f s\n",
               (double)resident_before / 1073741824.0,
               now_seconds() - load_started);

        /* Build a prompt of roughly `depth` tokens with the needle early. */
        const int needle = 73219;
        char *prose = (char *)calloc(1u, depth * 12u + 4096u);
        size_t used = 0;
        used += (size_t)snprintf(prose + used, depth * 12u,
                                 "Reference note: the archive code is %d. "
                                 "Remember it.\n", needle);
        mimo26_token_buffer probe;
        memset(&probe, 0, sizeof probe);
        size_t line = 0;
        while (true) {
            const size_t before = used;
            used += (size_t)snprintf(prose + used, depth * 12u - used, "%s\n",
                                     FILLER[line % FILLER_COUNT]);
            line++;
            if ((line % 8u) == 0u) {
                probe.count = 0;
                if (!mimo26_tokenizer_encode(tokenizer, prose, false, &probe,
                                             error, sizeof error)) {
                    fprintf(stderr, "  encode: %s\n", error);
                    return 1;
                }
                if (probe.count >= depth - 48u) {
                    used = before;
                    prose[used] = '\0';
                    break;
                }
            }
        }
        strcat(prose, "\nWhat is the archive code mentioned at the start? "
                      "Answer with the number only.");

        mimo26_chat_message turn;
        memset(&turn, 0, sizeof turn);
        turn.role = "user";
        turn.content = prose;
        mimo26_token_buffer prompt;
        memset(&prompt, 0, sizeof prompt);
        if (!mimo26_tokenizer_encode_chat(tokenizer, &turn, 1u, NULL, true,
                                          false, &prompt, error,
                                          sizeof error)) {
            fprintf(stderr, "  chat encode: %s\n", error);
            return 1;
        }

        float *logits = (float *)malloc((size_t)VOCAB * sizeof *logits);
        float *first_pass = (float *)malloc((size_t)VOCAB * sizeof *logits);
        if (logits == NULL || first_pass == NULL) {
            return 1;
        }

        /* --- prefill --- */
        const double prefill_started = now_seconds();
        bool ok_run = true;
        for (size_t i = 0; i < prompt.count && ok_run; i++) {
            if (mimo26_gpu_worker_decode(worker, prompt.ids[i], logits, error,
                                         sizeof error) !=
                MIMO26_GPU_WORKER_OK) {
                fprintf(stderr, "  prefill failed at %zu: %s\n", i, error);
                ok_run = false;
            }
        }
        const double prefill_seconds = now_seconds() - prefill_started;
        if (!ok_run) {
            return 1;
        }
        memcpy(first_pass, logits, (size_t)VOCAB * sizeof *logits);
        char detail[256];
        snprintf(detail, sizeof detail,
                 "%zu tokens in %.1f s, %.3f s/token", prompt.count,
                 prefill_seconds, prefill_seconds / (double)prompt.count);
        ok("prefill completes", true, detail);

        /* --- retrieval --- */
        const double decode_started = now_seconds();
        char answer[256] = {0};
        size_t answer_used = 0;
        mimo26_decode_stream stream;
        mimo26_decode_stream_init(&stream);
        uint32_t next = mimo26_gpu_worker_argmax(logits);
        size_t generated = 0;
        while (generated < 16u && next != MIMO26_TOK_IM_END &&
               next != MIMO26_TOK_ENDOFTEXT) {
            char piece[64];
            size_t piece_size = 0;
            if (mimo26_tokenizer_decode_stream(tokenizer, &stream, next, piece,
                                               sizeof piece, &piece_size) &&
                answer_used + piece_size < sizeof answer - 1u) {
                memcpy(answer + answer_used, piece, piece_size);
                answer_used += piece_size;
            }
            if (mimo26_gpu_worker_decode(worker, next, logits, error,
                                         sizeof error) !=
                MIMO26_GPU_WORKER_OK) {
                break;
            }
            next = mimo26_gpu_worker_argmax(logits);
            generated++;
        }
        const double decode_seconds = now_seconds() - decode_started;
        char wanted[32];
        snprintf(wanted, sizeof wanted, "%d", needle);
        snprintf(detail, sizeof detail, "answered %s, wanted %s", answer,
                 wanted);
        ok("retrieval from the start of context",
           strstr(answer, wanted) != NULL, detail);
        snprintf(detail, sizeof detail, "%.3f s/token at depth %zu",
                 decode_seconds / (double)(generated ? generated : 1u),
                 prompt.count);
        ok("decode latency at depth", true, detail);

        /* --- replay --- */
        mimo26_gpu_worker_reset(worker);
        for (size_t i = 0; i < prompt.count; i++) {
            if (mimo26_gpu_worker_decode(worker, prompt.ids[i], logits, error,
                                         sizeof error) !=
                MIMO26_GPU_WORKER_OK) {
                break;
            }
        }
        size_t differing = 0;
        for (uint32_t v = 0; v < VOCAB; v++) {
            if (memcmp(&first_pass[v], &logits[v], sizeof(float)) != 0) {
                differing++;
            }
        }
        snprintf(detail, sizeof detail, "%zu of %u logits differ", differing,
                 VOCAB);
        ok("replay is bit-identical at depth", differing == 0, detail);

        /* --- memory --- */
        const uint64_t resident_after =
            mimo26_gpu_worker_resident_bytes(worker);
        snprintf(detail, sizeof detail, "%.2f -> %.2f GiB",
                 (double)resident_before / 1073741824.0,
                 (double)resident_after / 1073741824.0);
        ok("residency is stable across the run",
           resident_after <= resident_before, detail);

        mimo26_gpu_worker_stats stats;
        mimo26_gpu_worker_get_stats(worker, &stats);
        printf("    -- expert hit rate %.1f%%, %llu uploads, %llu aborted\n",
               stats.expert_accesses
                   ? 100.0 * (double)stats.expert_hits /
                         (double)stats.expert_accesses
                   : 0.0,
               (unsigned long long)stats.expert_uploads,
               (unsigned long long)stats.aborted_steps);

        free(prose);
        free(logits);
        free(first_pass);
        mimo26_token_buffer_free(&prompt);
        mimo26_token_buffer_free(&probe);
        mimo26_gpu_worker_destroy(worker);
    }

    mimo26_tokenizer_destroy(tokenizer);
    printf("\nmimo26_context_gate: %s\n", failures == 0 ? "ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
