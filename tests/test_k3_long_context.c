#include "k3_chat.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                     \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

enum {
    K3_RETRIEVAL_DEFAULT_CONTEXT = 16384,
    K3_RETRIEVAL_MAX_CONTEXT = 32768,
    K3_RETRIEVAL_DEFAULT_TARGET = 16000,
    K3_RETRIEVAL_MAX_TARGET = 32000,
    K3_RETRIEVAL_MIN_TARGET = 512,
    K3_RETRIEVAL_MAX_GENERATED = 64,
};

typedef struct {
    struct timespec start;
    k3_chat_lifecycle_event lifecycle[8];
    size_t lifecycle_count;
    uint32_t planned_prompt;
    uint32_t planned_evaluated;
} retrieval_progress;

static bool append_format(FILE *stream, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const int written = vfprintf(stream, format, arguments);
    va_end(arguments);
    return written >= 0;
}

static bool build_case_file(uint32_t record_count,
                            char **text,
                            size_t *text_size,
                            uint32_t *alpha_record,
                            uint32_t *bravo_record,
                            uint32_t *charlie_record) {
    *text = NULL;
    *text_size = 0u;
    FILE *stream = open_memstream(text, text_size);
    if (stream == NULL) {
        return false;
    }

    const uint32_t alpha = record_count / 8u + 1u;
    const uint32_t bravo = record_count / 2u + 1u;
    const uint32_t charlie = record_count * 7u / 8u + 1u;
    bool ok = append_format(
        stream,
        "Aster Harbor field log\n\n"
        "Each numbered entry is an independent operational note. Most "
        "identifier-like strings are routine distractors. Exactly three "
        "entries are explicitly labeled RETRIEVAL KEY. Preserve the value "
        "stated in each of those entries, including capitalization, and "
        "ignore routine shipment codes. Read the complete log before "
        "answering.\n\n");

    static const char *coordinators[] = {
        "Mara", "Owen", "Priya", "Tomas", "Keiko", "Darius",
    };
    static const char *weather[] = {
        "clear", "windy", "overcast", "misty", "cool", "rainy",
    };
    static const char *cargo[] = {
        "canvas", "copper", "timber", "ceramic", "linen", "glass",
    };
    const size_t variant_count =
        sizeof(coordinators) / sizeof(coordinators[0]);

    for (uint32_t record = 1u; ok && record <= record_count; record++) {
        if (record == alpha) {
            ok = append_format(
                stream,
                "Record %04u. RETRIEVAL KEY ALPHA: The observatory access "
                "color is saffron. Preserve the lowercase word exactly; "
                "this is not a shipment code.\n",
                record);
        } else if (record == bravo) {
            ok = append_format(
                stream,
                "Record %04u. RETRIEVAL KEY BRAVO: The sealed archive crate "
                "number is 7319. Preserve all four digits exactly; this is "
                "not a shipment code.\n",
                record);
        } else if (record == charlie) {
            ok = append_format(
                stream,
                "Record %04u. RETRIEVAL KEY CHARLIE: The evening courier's "
                "surname is Nivens. Preserve the capitalization exactly; "
                "this is not a coordinator name.\n",
                record);
        } else {
            const size_t variant = (size_t)(record % variant_count);
            ok = append_format(
                stream,
                "Record %04u. During the routine review, coordinator %s "
                "inspected sector %u. The %s shipment used code HX-%04u-%02u, "
                "conditions were %s, and no retrieval key was assigned.\n",
                record,
                coordinators[variant],
                record % 29u + 1u,
                cargo[(variant + 2u) % variant_count],
                record,
                record % 97u,
                weather[(variant + 4u) % variant_count]);
        }
    }
    if (ok) {
        ok = append_format(
            stream,
            "\nQuestion: Return the three values from RETRIEVAL KEY ALPHA, "
            "BRAVO, and CHARLIE, in that order, separated only by vertical "
            "bars. Return exactly one line with no explanation or extra "
            "punctuation.");
    }
    if (fclose(stream) != 0) {
        ok = false;
    }
    if (!ok) {
        free(*text);
        *text = NULL;
        *text_size = 0u;
        return false;
    }
    *alpha_record = alpha;
    *bravo_record = bravo;
    *charlie_record = charlie;
    return true;
}

static bool encode_prompt(k3_tokenizer *tokenizer,
                          const char *text,
                          k3_token_buffer *tokens,
                          char *error,
                          size_t error_size) {
    const k3_chat_message message = {
        .role = K3_CHAT_ROLE_USER,
        .content = text,
    };
    const k3_chat_options options = {
        .add_generation_prompt = true,
        .thinking = false,
    };
    return k3_tokenizer_encode_chat(
        tokenizer, &message, 1u, &options,
        tokens, error, error_size);
}

static void report_progress(k3_chat_prefill_progress_unit unit,
                            uint32_t completed,
                            uint32_t total,
                            void *user_data) {
    if ((unit == K3_CHAT_PREFILL_PROGRESS_LAYERS &&
         completed != 1u && completed != total && completed % 8u != 0u) ||
        (unit == K3_CHAT_PREFILL_PROGRESS_TOKENS &&
         completed != total && completed % 256u != 0u)) {
        return;
    }
    const retrieval_progress *progress =
        (const retrieval_progress *)user_data;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    const double elapsed =
        (double)(now.tv_sec - progress->start.tv_sec) +
        (double)(now.tv_nsec - progress->start.tv_nsec) / 1e9;
    printf("  progress: %s %u/%u, elapsed %.1f s\n",
           unit == K3_CHAT_PREFILL_PROGRESS_LAYERS ? "layer" : "token",
           completed, total, elapsed);
    fflush(stdout);
}

static void report_lifecycle(
        k3_chat_lifecycle_event event,
        const k3_chat_turn_result *turn,
        void *user_data) {
    retrieval_progress *progress = (retrieval_progress *)user_data;
    if (progress->lifecycle_count <
        sizeof(progress->lifecycle) / sizeof(progress->lifecycle[0])) {
        progress->lifecycle[progress->lifecycle_count++] = event;
    }
    if (event == K3_CHAT_LIFECYCLE_PREFILL_START) {
        progress->planned_prompt = turn->prompt_tokens;
        progress->planned_evaluated = turn->prompt_evaluated_tokens;
    }
}

static bool exact_trimmed_answer(const char *response,
                                 const char *expected) {
    if (response == NULL) {
        return false;
    }
    const char *start = response;
    while (*start == ' ' || *start == '\t' ||
           *start == '\r' || *start == '\n') {
        start++;
    }
    const char *end = response + strlen(response);
    while (end > start &&
           (end[-1] == ' ' || end[-1] == '\t' ||
            end[-1] == '\r' || end[-1] == '\n')) {
        end--;
    }
    const size_t size = (size_t)(end - start);
    return size == strlen(expected) &&
           memcmp(start, expected, size) == 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 ||
        (argc == 3 && strcmp(argv[2], "--plan") != 0)) {
        fprintf(stderr,
                "usage: %s /path/to/moonshotai__Kimi-K3 [--plan]\n",
                argv[0]);
        return 2;
    }
    const bool plan_only = argc == 3;
    uint32_t target_prompt = K3_RETRIEVAL_DEFAULT_TARGET;
    const char *target_text = getenv("MOONSHINE_RETRIEVAL_TARGET");
    if (target_text != NULL && target_text[0] != '\0') {
        char *end = NULL;
        const unsigned long parsed = strtoul(target_text, &end, 10);
        if (end == NULL || *end != '\0' ||
            parsed < K3_RETRIEVAL_MIN_TARGET ||
            parsed > K3_RETRIEVAL_MAX_TARGET) {
            fprintf(stderr,
                    "invalid MOONSHINE_RETRIEVAL_TARGET: %s\n",
                    target_text);
            return 2;
        }
        target_prompt = (uint32_t)parsed;
    }
    const uint32_t minimum_prompt = target_prompt - 200u;
    k3_prefill_projection_backend range_backend =
        K3_PREFILL_PROJECTION_DEFAULT;
    const char *backend_text = getenv("MOONSHINE_RETRIEVAL_BACKEND");
    if (backend_text != NULL && backend_text[0] != '\0') {
        if (strcmp(backend_text, "kda-blas") != 0) {
            fprintf(stderr,
                    "invalid MOONSHINE_RETRIEVAL_BACKEND: %s\n",
                    backend_text);
            return 2;
        }
        range_backend =
            K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT;
    }
    int result = 1;
    char error[512] = { 0 };
    char *prompt_text = NULL;
    size_t prompt_size = 0u;
    k3_tokenizer *tokenizer = NULL;
    k3_token_buffer tokens = { 0 };
    k3_chat_session *session = NULL;
    k3_chat_turn_result turn = { 0 };

    CHECK(k3_tokenizer_create(
              &tokenizer, argv[1], error, sizeof(error)),
          error);

    uint32_t low = 8u;
    uint32_t high = 1024u;
    while (low < high) {
        const uint32_t candidate = low + (high - low + 1u) / 2u;
        uint32_t alpha = 0u;
        uint32_t bravo = 0u;
        uint32_t charlie = 0u;
        free(prompt_text);
        prompt_text = NULL;
        CHECK(build_case_file(
                  candidate, &prompt_text, &prompt_size,
                  &alpha, &bravo, &charlie),
              "building retrieval case file failed");
        CHECK(encode_prompt(
                  tokenizer, prompt_text, &tokens,
                  error, sizeof(error)),
              error);
        if (tokens.count <= target_prompt) {
            low = candidate;
        } else {
            high = candidate - 1u;
        }
    }

    uint32_t alpha_record = 0u;
    uint32_t bravo_record = 0u;
    uint32_t charlie_record = 0u;
    free(prompt_text);
    prompt_text = NULL;
    CHECK(build_case_file(
              low, &prompt_text, &prompt_size,
              &alpha_record, &bravo_record, &charlie_record),
          "building final retrieval case file failed");
    CHECK(encode_prompt(
              tokenizer, prompt_text, &tokens,
              error, sizeof(error)),
          error);
    CHECK(tokens.count >= minimum_prompt &&
              tokens.count <= target_prompt,
          "retrieval prompt did not fill the intended target band");
    if (target_prompt == K3_RETRIEVAL_DEFAULT_TARGET) {
        CHECK(tokens.count == 15993u && low == 389u &&
                  alpha_record == 49u && bravo_record == 195u &&
                  charlie_record == 341u && prompt_size == 69938u,
              "default retrieval prompt construction changed");
    } else if (target_prompt == K3_RETRIEVAL_MAX_TARGET) {
        CHECK(tokens.count == 31999u && low == 781u &&
                  alpha_record == 98u && bravo_record == 391u &&
                  charlie_record == 684u && prompt_size == 139990u,
              "32K retrieval prompt construction changed");
    }
    printf("K3 long-context retrieval plan: %zu/%u tokens, %u records, "
           "needles=%u/%u/%u, text=%zu bytes, backend=%s\n",
           tokens.count, target_prompt, low,
           alpha_record, bravo_record, charlie_record,
           prompt_size,
           range_backend == K3_PREFILL_PROJECTION_DEFAULT ?
               "default" : "kda-blas");
    fflush(stdout);
    if (plan_only) {
        result = 0;
        goto cleanup;
    }

    k3_token_buffer_free(&tokens);
    k3_tokenizer_destroy(tokenizer);
    tokenizer = NULL;

    k3_engine_stats engine_stats;
    const k3_chat_session_config config = {
        .model_root = argv[1],
        .context = target_prompt > K3_RETRIEVAL_DEFAULT_TARGET ?
            K3_RETRIEVAL_MAX_CONTEXT : K3_RETRIEVAL_DEFAULT_CONTEXT,
        .sequential_prefill_limit = K3_CHAT_MEASURED_SEQUENTIAL_LIMIT,
        .experts_per_layer = 32u,
        .staging_slots = 16u,
        .q8_projections = true,
        .range_backend = range_backend,
    };
    CHECK(k3_chat_session_create(
              &session, &config, &engine_stats,
              error, sizeof(error)),
          error);
    const k3_chat_message message = {
        .role = K3_CHAT_ROLE_USER,
        .content = prompt_text,
    };
    retrieval_progress progress = { 0 };
    clock_gettime(CLOCK_MONOTONIC, &progress.start);
    const k3_chat_completion_options options = {
        .progress_callback = report_progress,
        .progress_data = &progress,
        .lifecycle_callback = report_lifecycle,
        .lifecycle_data = &progress,
    };
    CHECK(k3_chat_session_complete_messages_with_options(
              session, &message, 1u,
              K3_RETRIEVAL_MAX_GENERATED,
              &options, NULL, NULL, &turn,
              error, sizeof(error)),
          error);
    CHECK(progress.lifecycle_count >= 3u &&
              progress.lifecycle[0] ==
                  K3_CHAT_LIFECYCLE_PREFILL_START &&
              progress.lifecycle[1] ==
                  K3_CHAT_LIFECYCLE_PREFILL_COMPLETE &&
              progress.lifecycle[2] ==
                  K3_CHAT_LIFECYCLE_DECODE_START,
          "request lifecycle event order changed");
    CHECK(progress.planned_prompt == turn.prompt_tokens &&
              progress.planned_evaluated ==
                  turn.prompt_evaluated_tokens,
          "prefill lifecycle plan did not expose final token accounting");
    CHECK(turn.prefill_strategy == K3_CHAT_PREFILL_LAYER_MAJOR,
          "retrieval prompt did not use selected range prefill");
    CHECK(turn.prompt_tokens >= minimum_prompt &&
              turn.prompt_tokens <= target_prompt,
          "session prompt count differs from retrieval plan");
    printf("  response: %s\n",
           turn.response.data == NULL ? "<empty>" : turn.response.data);
    printf("  reasoning: %s; tool_calls=%zu\n",
           turn.reasoning_content.data == NULL ?
               "<empty>" : turn.reasoning_content.data,
           turn.tool_call_count);
    printf("  startup=%.3f s prompt=%u/%.3f s (%.3f tok/s) "
           "generated=%u/%.3f s (%.3f tok/s) finish=%d forced=%u\n",
           engine_stats.startup_seconds,
           turn.prompt_tokens, turn.prompt_seconds,
           turn.prompt_tokens / turn.prompt_seconds,
           turn.generated_tokens, turn.decode_seconds,
           turn.tokens_per_second,
           (int)turn.finish_reason, turn.forced_trailer_tokens);
    printf("  selected reads=%" PRIu64 " bytes requests=%u "
           "unique=%u/%.1f/%u per layer\n",
           turn.range_stats.routed_physical_read_bytes,
           turn.range_stats.expert_read_requests,
           turn.range_stats.min_unique_experts_per_layer,
           (double)turn.range_stats.unique_experts_across_layers /
               turn.range_stats.routed_layer_sweeps,
           turn.range_stats.max_unique_experts_per_layer);
    const double phase_seconds =
        turn.range_stats.layer0_seconds +
        turn.range_stats.attention_seconds +
        turn.range_stats.router_seconds +
        turn.range_stats.routed_stream_seconds +
        turn.range_stats.moe_tail_seconds +
        turn.range_stats.output_seconds;
    printf("  phases: layer0=%.3f attention=%.3f "
           "(kda=%.3f mla=%.3f) router=%.3f "
           "stream=%.3f tail=%.3f output=%.3f s\n",
           turn.range_stats.layer0_seconds,
           turn.range_stats.attention_seconds,
           turn.range_stats.kda_attention_seconds,
           turn.range_stats.mla_attention_seconds,
           turn.range_stats.router_seconds,
           turn.range_stats.routed_stream_seconds,
           turn.range_stats.moe_tail_seconds,
           turn.range_stats.output_seconds);
    printf("  phase coverage=%.3f/%.3f s (%.2f%%)\n",
           phase_seconds, turn.range_stats.wall_seconds,
           100.0 * phase_seconds / turn.range_stats.wall_seconds);
    const double kda_detail =
        turn.range_stats.kda_projection_seconds +
        turn.range_stats.kda_convolution_seconds +
        turn.range_stats.kda_recurrent_seconds +
        turn.range_stats.kda_gate_norm_seconds +
        turn.range_stats.kda_output_projection_seconds;
    const double stream_detail =
        turn.range_stats.routed_read_wait_seconds +
        turn.range_stats.routed_submit_seconds +
        turn.range_stats.routed_index_seconds +
        turn.range_stats.routed_expert_pipeline_seconds;
    printf("  KDA detail: projection=%.3f conv=%.3f "
           "recurrent=%.3f gate/norm=%.3f output=%.3f "
           "accounted=%.3f other=%.3f s\n",
           turn.range_stats.kda_projection_seconds,
           turn.range_stats.kda_convolution_seconds,
           turn.range_stats.kda_recurrent_seconds,
           turn.range_stats.kda_gate_norm_seconds,
           turn.range_stats.kda_output_projection_seconds,
           kda_detail,
           turn.range_stats.kda_attention_seconds - kda_detail);
    printf("  stream detail: read-wait=%.3f submit=%.3f "
           "index=%.3f expert-pipeline=%.3f "
           "accounted=%.3f other=%.3f s\n",
           turn.range_stats.routed_read_wait_seconds,
           turn.range_stats.routed_submit_seconds,
           turn.range_stats.routed_index_seconds,
           turn.range_stats.routed_expert_pipeline_seconds,
           stream_detail,
           turn.range_stats.routed_stream_seconds - stream_detail);
    fflush(stdout);
    CHECK(exact_trimmed_answer(
              turn.response.data, "saffron|7319|Nivens"),
          "model did not retrieve the three ordered values exactly");
    CHECK(turn.generated_tokens == 17u &&
              turn.finish_reason == K3_CHAT_FINISH_END_OF_MESSAGE &&
              turn.forced_trailer_tokens == 0u &&
              turn.tool_call_count == 0u,
          "retrieval response structure changed");

    printf("K3 long-context natural-text retrieval: PASS\n");
    result = 0;

cleanup:
    k3_chat_turn_result_free(&turn);
    k3_chat_session_destroy(session);
    k3_token_buffer_free(&tokens);
    k3_tokenizer_destroy(tokenizer);
    free(prompt_text);
    return result;
}
