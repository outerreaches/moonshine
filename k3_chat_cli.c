#include "k3_chat.h"
#include "moonshine_version.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const char *model_root;
    const char *system_prompt;
    const char *one_shot_prompt;
    const char *load_path;
    const char *save_path;
    uint32_t    max_tokens;
    uint32_t    context;
    uint32_t    sequential_limit;
} cli_options;

static void usage(FILE *stream, const char *program) {
    fprintf(
        stream,
        "usage: %s MODEL [options]\n"
        "\n"
        "Options:\n"
        "  -p, --prompt TEXT          Run one turn and exit\n"
        "  --system TEXT              Initial system message\n"
        "  -n, --max-tokens N         Maximum generated tokens (default 256)\n"
        "  --context N                Context capacity (default 8192)\n"
        "  --sequential-limit N       Token-major prompt limit (default 92)\n"
        "  --load PATH                Import causal state after startup\n"
        "  --save PATH                Export causal state before exit\n"
        "  -h, --help                 Show this help\n"
        "  --version                  Show the Moonshine version\n"
        "\n"
        "Interactive commands: /save PATH, /load PATH, /help, /quit\n",
        program);
}

static bool parse_u32(const char *text, uint32_t *value) {
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed == 0ul || parsed > UINT32_MAX) {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static bool parse_options(
        int argc, char **argv, cli_options *options) {
    memset(options, 0, sizeof(*options));
    options->max_tokens = 256u;
    options->context = 8192u;
    options->sequential_limit =
        K3_CHAT_MEASURED_SEQUENTIAL_LIMIT;

    int index = 1;
    if (index < argc && argv[index][0] != '-') {
        options->model_root = argv[index++];
    } else {
        options->model_root = getenv("MOONSHINE_MODEL");
    }
    while (index < argc) {
        const char *arg = argv[index++];
        if (strcmp(arg, "-h") == 0 ||
            strcmp(arg, "--help") == 0) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (strcmp(arg, "--version") == 0) {
            printf("%s %s\n", MOONSHINE_NAME, MOONSHINE_VERSION);
            exit(0);
        } else if (strcmp(arg, "-p") == 0 ||
                   strcmp(arg, "--prompt") == 0) {
            if (index >= argc) {
                return false;
            }
            options->one_shot_prompt = argv[index++];
        } else if (strcmp(arg, "--system") == 0) {
            if (index >= argc) {
                return false;
            }
            options->system_prompt = argv[index++];
        } else if (strcmp(arg, "-n") == 0 ||
                   strcmp(arg, "--max-tokens") == 0) {
            if (index >= argc ||
                !parse_u32(argv[index++], &options->max_tokens)) {
                return false;
            }
        } else if (strcmp(arg, "--sequential-limit") == 0) {
            if (index >= argc ||
                !parse_u32(
                    argv[index++], &options->sequential_limit)) {
                return false;
            }
        } else if (strcmp(arg, "--context") == 0) {
            if (index >= argc ||
                !parse_u32(argv[index++], &options->context) ||
                options->context < 64u ||
                options->context > 1048576u) {
                return false;
            }
        } else if (strcmp(arg, "--load") == 0) {
            if (index >= argc) {
                return false;
            }
            options->load_path = argv[index++];
        } else if (strcmp(arg, "--save") == 0) {
            if (index >= argc) {
                return false;
            }
            options->save_path = argv[index++];
        } else {
            return false;
        }
    }
    return options->model_root != NULL &&
           options->model_root[0] != '\0';
}

static void stream_stdout(
        const char *bytes, size_t size, void *user_data) {
    (void)user_data;
    if (size != 0u) {
        fwrite(bytes, 1u, size, stdout);
        fflush(stdout);
    }
}

static const char *prefill_name(k3_chat_prefill_strategy strategy) {
    return strategy == K3_CHAT_PREFILL_LAYER_MAJOR ?
        "layer-major" : "token-major";
}

static const char *finish_name(k3_chat_finish_reason reason) {
    if (reason == K3_CHAT_FINISH_END_OF_MESSAGE) {
        return "end_of_message";
    }
    return reason == K3_CHAT_FINISH_TOOL_CALLS ?
        "tool_calls" : "length";
}

static void print_turn_stats(const k3_chat_turn_result *result) {
    const double prompt_rate =
        result->prompt_seconds > 0.0 ?
            (double)result->prompt_tokens /
                result->prompt_seconds : 0.0;
    const uint64_t accesses =
        result->cache_after.accesses -
        result->cache_before.accesses;
    const uint64_t hits =
        result->cache_after.hits -
        result->cache_before.hits;
    const double hit_rate =
        accesses != 0u ? 100.0 * (double)hits /
            (double)accesses : 0.0;
    fprintf(
        stderr,
        "[prompt=%u %s %.3fs %.3f tok/s; "
        "generated=%u %.3fs %.3f tok/s; "
        "finish=%s; forced=%u; position=%u; "
        "cache=%" PRIu64 "/%" PRIu64 " %.1f%%]\n",
        result->prompt_tokens,
        prefill_name(result->prefill_strategy),
        result->prompt_seconds, prompt_rate,
        result->generated_tokens,
        result->decode_seconds,
        result->tokens_per_second,
        finish_name(result->finish_reason),
        result->forced_trailer_tokens,
        result->position,
        hits, accesses, hit_rate);
    if (result->prefill_strategy ==
        K3_CHAT_PREFILL_LAYER_MAJOR) {
        fprintf(
            stderr,
            "[prefill I/O=%" PRIu64 " bytes; sweeps=%u; requests=%u]\n",
            result->range_stats.routed_physical_read_bytes,
            result->range_stats.routed_layer_sweeps,
            result->range_stats.expert_read_requests);
    }
}

static bool export_state(
        k3_chat_session *session, const char *path) {
    char error[512];
    k3_engine_state_file_info info;
    if (!k3_chat_session_export_state(
            session, path, &info, error, sizeof(error))) {
        fprintf(stderr, "state export failed: %s\n", error);
        return false;
    }
    fprintf(
        stderr,
        "saved position %u to %s (%.3f MiB, %.3fs)\n",
        info.token_position, path,
        (double)info.file_bytes / (1024.0 * 1024.0),
        info.wall_seconds);
    return true;
}

static bool import_state(
        k3_chat_session *session, const char *path) {
    char error[512];
    k3_engine_state_file_info info;
    if (!k3_chat_session_import_state(
            session, path, &info, error, sizeof(error))) {
        fprintf(stderr, "state import failed: %s\n", error);
        return false;
    }
    fprintf(
        stderr,
        "loaded position %u from %s (%.3f MiB, %.3fs)\n",
        info.token_position, path,
        (double)info.file_bytes / (1024.0 * 1024.0),
        info.wall_seconds);
    return true;
}

static bool run_turn(k3_chat_session *session,
                     const char *prompt,
                     uint32_t max_tokens) {
    char error[512];
    k3_chat_turn_result result = { 0 };
    if (!k3_chat_session_turn(
            session, prompt, max_tokens,
            stream_stdout, NULL, &result,
            error, sizeof(error))) {
        fprintf(stderr, "chat turn failed: %s\n", error);
        k3_chat_turn_result_free(&result);
        return false;
    }
    if (result.response.size == 0u) {
        fputs("(no response content)", stdout);
    }
    fputc('\n', stdout);
    fflush(stdout);
    print_turn_stats(&result);
    k3_chat_turn_result_free(&result);
    return true;
}

static const char *command_path(
        const char *line, const char *command) {
    const size_t command_size = strlen(command);
    if (strncmp(line, command, command_size) != 0) {
        return NULL;
    }
    const char *path = line + command_size;
    while (*path == ' ' || *path == '\t') {
        path++;
    }
    return path[0] == '\0' ? NULL : path;
}

static int interactive_loop(k3_chat_session *session,
                            uint32_t max_tokens) {
    char *line = NULL;
    size_t capacity = 0u;
    int result = 0;
    const bool terminal = isatty(STDIN_FILENO);
    if (terminal) {
        fprintf(
            stderr,
            "K3 chat ready. Commands: /save PATH, /load PATH, "
            "/help, /quit\n");
    }
    for (;;) {
        if (terminal) {
            fputs("user> ", stderr);
            fflush(stderr);
        }
        const ssize_t read = getline(&line, &capacity, stdin);
        if (read < 0) {
            break;
        }
        size_t size = (size_t)read;
        while (size > 0u &&
               (line[size - 1u] == '\n' ||
                line[size - 1u] == '\r')) {
            line[--size] = '\0';
        }
        if (size == 0u) {
            continue;
        }
        if (strcmp(line, "/quit") == 0 ||
            strcmp(line, "/exit") == 0) {
            break;
        }
        if (strcmp(line, "/help") == 0) {
            fprintf(
                stderr,
                "/save PATH  export exact causal state\n"
                "/load PATH  import exact causal state\n"
                "/quit       exit\n");
            continue;
        }
        const char *path = command_path(line, "/save");
        if (path != NULL) {
            (void)export_state(session, path);
            continue;
        }
        path = command_path(line, "/load");
        if (path != NULL) {
            (void)import_state(session, path);
            continue;
        }
        if (terminal) {
            fputs("assistant> ", stderr);
            fflush(stderr);
        }
        if (!run_turn(session, line, max_tokens)) {
            result = 1;
        }
    }
    free(line);
    return result;
}

int main(int argc, char **argv) {
    cli_options options;
    if (!parse_options(argc, argv, &options)) {
        usage(stderr, argv[0]);
        return 2;
    }

    const k3_chat_session_config config = {
        .model_root = options.model_root,
        .system_prompt = options.system_prompt,
        .context = options.context,
        .sequential_prefill_limit = options.sequential_limit,
        .experts_per_layer = 32u,
        .staging_slots = 16u,
        .q8_projections = true,
    };
    fprintf(
        stderr, "loading Moonshine Q8/32 context=%u from %s\n",
        options.context, options.model_root);
    char error[512];
    k3_engine_stats engine_stats;
    k3_chat_session *session = NULL;
    if (!k3_chat_session_create(
            &session, &config, &engine_stats,
            error, sizeof(error))) {
        fprintf(stderr, "Moonshine startup failed: %s\n", error);
        return 1;
    }
    fprintf(
        stderr,
        "engine ready in %.3fs; static=%.3f GiB cache=%.3f GiB "
        "state=%.3f GiB\n",
        engine_stats.startup_seconds,
        (double)engine_stats.static_store.resident_bytes /
            (1024.0 * 1024.0 * 1024.0),
        (double)engine_stats.cache_bytes /
            (1024.0 * 1024.0 * 1024.0),
        (double)engine_stats.state_bytes /
            (1024.0 * 1024.0 * 1024.0));

    int result = 0;
    if (options.load_path != NULL &&
        !import_state(session, options.load_path)) {
        result = 1;
        goto cleanup;
    }
    if (options.one_shot_prompt != NULL) {
        result = run_turn(
            session, options.one_shot_prompt,
            options.max_tokens) ? 0 : 1;
    } else {
        result = interactive_loop(
            session, options.max_tokens);
    }
    if (options.save_path != NULL &&
        !export_state(session, options.save_path)) {
        result = 1;
    }

cleanup:
    k3_chat_session_destroy(session);
    return result;
}
