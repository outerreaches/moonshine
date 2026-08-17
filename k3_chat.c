#include "k3_chat.h"

#include "k3_prefix_reuse.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    K3_CHAT_DEFAULT_CONTEXT = 8192,
    K3_CHAT_DEFAULT_EXPERTS_PER_LAYER = 32,
    K3_CHAT_DEFAULT_STAGING_SLOTS = 16,
};

static const uint32_t K3_RESPONSE_TRAILER[] = {
    K3_TOKEN_CLOSE, 12092u, K3_TOKEN_SEP,
    K3_TOKEN_CLOSE, 2778u, K3_TOKEN_SEP,
    K3_TOKEN_END_OF_MSG,
};

struct k3_chat_session {
    k3_engine    *engine;
    k3_tokenizer *tokenizer;
    char         *system_prompt;
    uint32_t      context;
    uint32_t      sequential_prefill_limit;
    k3_prefill_projection_backend range_backend;
    bool          capture_state_digest;
    uint32_t      position;
    k3_token_buffer retained_tokens;
    k3_tool_choice_marker *historical_tool_choices;
    size_t historical_tool_choice_count;
    size_t historical_tool_choice_capacity;
    k3_single_tool_call_marker *historical_single_tool_calls;
    size_t historical_single_tool_call_count;
    size_t historical_single_tool_call_capacity;
    k3_response_format_marker *historical_response_formats;
    size_t historical_response_format_count;
    size_t historical_response_format_capacity;
    bool          started;
    bool          healthy;
};

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error == NULL || error_size == 0u) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static double elapsed_seconds(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static bool reserve_retained_tokens(k3_chat_session *session,
                                    size_t additional,
                                    char *error,
                                    size_t error_size) {
    if (additional > SIZE_MAX - session->retained_tokens.count) {
        set_error(error, error_size,
                  "retained token history size overflow");
        return false;
    }
    const size_t required =
        session->retained_tokens.count + additional;
    if (required <= session->retained_tokens.capacity) {
        return true;
    }
    size_t capacity =
        session->retained_tokens.capacity == 0u ?
            256u : session->retained_tokens.capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            set_error(error, error_size,
                      "retained token history capacity overflow");
            return false;
        }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX /
            sizeof(*session->retained_tokens.data)) {
        set_error(error, error_size,
                  "retained token history byte size overflow");
        return false;
    }
    uint32_t *data = (uint32_t *)realloc(
        session->retained_tokens.data,
        capacity * sizeof(*session->retained_tokens.data));
    if (data == NULL) {
        set_error(error, error_size,
                  "allocating retained token history failed");
        return false;
    }
    session->retained_tokens.data = data;
    session->retained_tokens.capacity = capacity;
    return true;
}

static bool append_retained_tokens(k3_chat_session *session,
                                   const uint32_t *tokens,
                                   size_t count,
                                   char *error,
                                   size_t error_size) {
    if (count == 0u) {
        return true;
    }
    if (!reserve_retained_tokens(
            session, count, error, error_size)) {
        return false;
    }
    memcpy(
        session->retained_tokens.data +
            session->retained_tokens.count,
        tokens, count * sizeof(*tokens));
    session->retained_tokens.count += count;
    return true;
}

static void clear_request_directive_history(k3_chat_session *session) {
    for (size_t i = 0u;
         i < session->historical_response_format_count; i++) {
        free((char *)session->historical_response_formats[i]
            .response_schema_json);
        session->historical_response_formats[i]
            .response_schema_json = NULL;
    }
    session->historical_tool_choice_count = 0u;
    session->historical_single_tool_call_count = 0u;
    session->historical_response_format_count = 0u;
}

/*
 * Request-directive history is an optimization hint, not inference output.
 * If its allocation fails or the caller presents a non-append-only boundary,
 * discard it and let the next request take the exact-prefix mismatch fallback.
 */
static void remember_tool_choice(
        k3_chat_session *session,
        size_t after_message_count,
        k3_tool_choice choice) {
    if (choice == K3_TOOL_CHOICE_AUTO) {
        return;
    }
    if (choice != K3_TOOL_CHOICE_REQUIRED &&
        choice != K3_TOOL_CHOICE_NONE) {
        clear_request_directive_history(session);
        return;
    }
    if (session->historical_tool_choice_count != 0u &&
        session->historical_tool_choices[
            session->historical_tool_choice_count - 1u]
                .after_message_count >= after_message_count) {
        clear_request_directive_history(session);
        return;
    }
    if (session->historical_tool_choice_count ==
        session->historical_tool_choice_capacity) {
        const size_t old_capacity =
            session->historical_tool_choice_capacity;
        if (old_capacity > SIZE_MAX / 2u) {
            clear_request_directive_history(session);
            return;
        }
        const size_t capacity =
            old_capacity == 0u ? 4u : old_capacity * 2u;
        if (capacity > SIZE_MAX /
                sizeof(*session->historical_tool_choices)) {
            clear_request_directive_history(session);
            return;
        }
        k3_tool_choice_marker *markers =
            (k3_tool_choice_marker *)realloc(
                session->historical_tool_choices,
                capacity * sizeof(*markers));
        if (markers == NULL) {
            clear_request_directive_history(session);
            return;
        }
        session->historical_tool_choices = markers;
        session->historical_tool_choice_capacity = capacity;
    }
    session->historical_tool_choices[
        session->historical_tool_choice_count++] =
        (k3_tool_choice_marker) {
            .after_message_count = after_message_count,
            .choice = choice,
        };
}

static void remember_single_tool_call(
        k3_chat_session *session,
        size_t after_message_count,
        bool enforce) {
    if (!enforce) {
        return;
    }
    if (session->historical_single_tool_call_count != 0u &&
        session->historical_single_tool_calls[
            session->historical_single_tool_call_count - 1u]
                .after_message_count >= after_message_count) {
        clear_request_directive_history(session);
        return;
    }
    if (session->historical_single_tool_call_count ==
        session->historical_single_tool_call_capacity) {
        const size_t old_capacity =
            session->historical_single_tool_call_capacity;
        if (old_capacity > SIZE_MAX / 2u) {
            clear_request_directive_history(session);
            return;
        }
        const size_t capacity =
            old_capacity == 0u ? 4u : old_capacity * 2u;
        if (capacity > SIZE_MAX /
                sizeof(*session->historical_single_tool_calls)) {
            clear_request_directive_history(session);
            return;
        }
        k3_single_tool_call_marker *markers =
            (k3_single_tool_call_marker *)realloc(
                session->historical_single_tool_calls,
                capacity * sizeof(*markers));
        if (markers == NULL) {
            clear_request_directive_history(session);
            return;
        }
        session->historical_single_tool_calls = markers;
        session->historical_single_tool_call_capacity = capacity;
    }
    session->historical_single_tool_calls[
        session->historical_single_tool_call_count++] =
        (k3_single_tool_call_marker) {
            .after_message_count = after_message_count,
        };
}

static void remember_response_format(
        k3_chat_session *session,
        size_t after_message_count,
        k3_response_format format,
        const char *schema_json) {
    if (format == K3_RESPONSE_FORMAT_TEXT) {
        return;
    }
    if ((format != K3_RESPONSE_FORMAT_JSON_OBJECT &&
         format != K3_RESPONSE_FORMAT_JSON_SCHEMA) ||
        (format == K3_RESPONSE_FORMAT_JSON_SCHEMA &&
         (schema_json == NULL || schema_json[0] == '\0')) ||
        (format == K3_RESPONSE_FORMAT_JSON_OBJECT &&
         schema_json != NULL)) {
        clear_request_directive_history(session);
        return;
    }
    if (session->historical_response_format_count != 0u &&
        session->historical_response_formats[
            session->historical_response_format_count - 1u]
                .after_message_count >= after_message_count) {
        clear_request_directive_history(session);
        return;
    }
    char *schema_copy = NULL;
    if (format == K3_RESPONSE_FORMAT_JSON_SCHEMA) {
        schema_copy = strdup(schema_json);
        if (schema_copy == NULL) {
            clear_request_directive_history(session);
            return;
        }
    }
    if (session->historical_response_format_count ==
        session->historical_response_format_capacity) {
        const size_t old_capacity =
            session->historical_response_format_capacity;
        if (old_capacity > SIZE_MAX / 2u) {
            free(schema_copy);
            clear_request_directive_history(session);
            return;
        }
        const size_t capacity =
            old_capacity == 0u ? 4u : old_capacity * 2u;
        if (capacity > SIZE_MAX /
                sizeof(*session->historical_response_formats)) {
            free(schema_copy);
            clear_request_directive_history(session);
            return;
        }
        k3_response_format_marker *markers =
            (k3_response_format_marker *)realloc(
                session->historical_response_formats,
                capacity * sizeof(*markers));
        if (markers == NULL) {
            free(schema_copy);
            clear_request_directive_history(session);
            return;
        }
        session->historical_response_formats = markers;
        session->historical_response_format_capacity = capacity;
    }
    session->historical_response_formats[
        session->historical_response_format_count++] =
        (k3_response_format_marker) {
            .after_message_count = after_message_count,
            .format = format,
            .response_schema_json = schema_copy,
        };
}

static bool append_generated_token(k3_token_buffer *buffer,
                                   uint32_t token,
                                   char *error,
                                   size_t error_size) {
    if (buffer->count == buffer->capacity) {
        size_t capacity = buffer->capacity == 0u ?
            64u : buffer->capacity;
        if (capacity > SIZE_MAX / 2u) {
            set_error(error, error_size,
                      "generated token capacity overflow");
            return false;
        }
        capacity *= 2u;
        if (capacity > SIZE_MAX / sizeof(*buffer->data)) {
            set_error(error, error_size,
                      "generated token storage overflow");
            return false;
        }
        uint32_t *data = (uint32_t *)realloc(
            buffer->data, capacity * sizeof(*data));
        if (data == NULL) {
            set_error(error, error_size,
                      "allocating generated token history failed");
            return false;
        }
        buffer->data = data;
        buffer->capacity = capacity;
    }
    buffer->data[buffer->count++] = token;
    return true;
}

static bool append_response_bytes(k3_text_buffer *buffer,
                                  const char *data,
                                  size_t size,
                                  char *error,
                                  size_t error_size) {
    if (buffer->size > SIZE_MAX - size - 1u) {
        set_error(error, error_size, "chat response size overflow");
        return false;
    }
    const size_t required = buffer->size + size + 1u;
    if (required > buffer->capacity) {
        size_t capacity = buffer->capacity == 0u ?
            256u : buffer->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2u) {
                set_error(error, error_size,
                          "chat response capacity overflow");
                return false;
            }
            capacity *= 2u;
        }
        char *data_out = (char *)realloc(buffer->data, capacity);
        if (data_out == NULL) {
            set_error(error, error_size,
                      "allocating chat response failed");
            return false;
        }
        buffer->data = data_out;
        buffer->capacity = capacity;
    }
    if (size != 0u) {
        memcpy(buffer->data + buffer->size, data, size);
    }
    buffer->size += size;
    buffer->data[buffer->size] = '\0';
    return true;
}

static bool feed_token(k3_chat_session *session,
                       uint32_t token,
                       uint32_t *next_token,
                       char *error,
                       size_t error_size) {
    float value = 0.0f;
    if (!k3_engine_forward_token(
            session->engine, token, next_token, &value,
            error, error_size)) {
        session->healthy = false;
        return false;
    }
    session->position++;
    return true;
}

static size_t trailer_prefix_already_generated(
        const uint32_t *tail,
        size_t tail_count,
        const uint32_t *trailer,
        size_t trailer_count) {
    size_t matched = tail_count < trailer_count ?
        tail_count : trailer_count;
    for (;;) {
        if (matched == 0u ||
            memcmp(
                tail + tail_count - matched,
                trailer,
                matched * sizeof(*tail)) == 0) {
            return matched;
        }
        matched--;
    }
}

static void append_tail(uint32_t *tail,
                        size_t *tail_count,
                        size_t capacity,
                        uint32_t token) {
    if (*tail_count < capacity) {
        tail[(*tail_count)++] = token;
        return;
    }
    memmove(tail, tail + 1u,
            (capacity - 1u) * sizeof(*tail));
    tail[capacity - 1u] = token;
}

bool k3_chat_session_create(
        k3_chat_session **out,
        const k3_chat_session_config *config,
        k3_engine_stats *engine_stats,
        char *error,
        size_t error_size) {
    if (out == NULL || config == NULL ||
        config->model_root == NULL ||
        config->model_root[0] == '\0') {
        set_error(error, error_size,
                  "chat session needs output, config, and model root");
        return false;
    }
    *out = NULL;
    k3_chat_session *session =
        (k3_chat_session *)calloc(1u, sizeof(*session));
    if (session == NULL) {
        set_error(error, error_size,
                  "allocating chat session failed");
        return false;
    }
    session->context = config->context == 0u ?
        K3_CHAT_DEFAULT_CONTEXT : config->context;
    session->sequential_prefill_limit =
        config->sequential_prefill_limit == 0u ?
            K3_CHAT_MEASURED_SEQUENTIAL_LIMIT :
            config->sequential_prefill_limit;
    if (config->range_backend != K3_PREFILL_PROJECTION_DEFAULT &&
        config->range_backend !=
            K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT) {
        set_error(error, error_size,
                  "chat session range backend is invalid");
        k3_chat_session_destroy(session);
        return false;
    }
    session->range_backend = config->range_backend;
    session->capture_state_digest = config->capture_state_digest;
    session->healthy = true;
    if (config->system_prompt != NULL &&
        config->system_prompt[0] != '\0') {
        session->system_prompt = strdup(config->system_prompt);
        if (session->system_prompt == NULL) {
            set_error(error, error_size,
                      "copying system prompt failed: %s",
                      strerror(errno));
            k3_chat_session_destroy(session);
            return false;
        }
    }

    if (!k3_tokenizer_create(
            &session->tokenizer, config->model_root,
            error, error_size)) {
        k3_chat_session_destroy(session);
        return false;
    }
    const uint16_t experts =
        config->experts_per_layer == 0u ?
            K3_CHAT_DEFAULT_EXPERTS_PER_LAYER :
            config->experts_per_layer;
    const uint16_t staging =
        config->staging_slots == 0u ?
            K3_CHAT_DEFAULT_STAGING_SLOTS :
            config->staging_slots;
    if (!k3_engine_create(
            &session->engine, config->model_root,
            session->context, experts, staging,
            config->q8_projections,
            engine_stats, error, error_size)) {
        k3_chat_session_destroy(session);
        return false;
    }
    if (config->decode_diagnostics_prefix != NULL &&
        config->decode_diagnostics_prefix[0] != '\0' &&
        !k3_engine_configure_decode_diagnostics(
            session->engine,
            config->decode_diagnostics_prefix,
            error, error_size)) {
        k3_chat_session_destroy(session);
        return false;
    }

    *out = session;
    return true;
}

void k3_chat_session_destroy(k3_chat_session *session) {
    if (session == NULL) {
        return;
    }
    k3_engine_destroy(session->engine);
    k3_tokenizer_destroy(session->tokenizer);
    k3_token_buffer_free(&session->retained_tokens);
    clear_request_directive_history(session);
    free(session->historical_tool_choices);
    free(session->historical_single_tool_calls);
    free(session->historical_response_formats);
    free(session->system_prompt);
    free(session);
}

void k3_chat_turn_result_free(k3_chat_turn_result *result) {
    if (result == NULL) {
        return;
    }
    k3_text_buffer_free(&result->reasoning_content);
    k3_text_buffer_free(&result->response);
    for (size_t i = 0u; i < result->tool_call_count; i++) {
        free((char *)result->tool_calls[i].id);
        free((char *)result->tool_calls[i].name);
        free((char *)result->tool_calls[i].arguments);
    }
    free(result->tool_calls);
    memset(result, 0, sizeof(*result));
}

static bool encode_turn_prompt(k3_chat_session *session,
                               const char *user_text,
                               k3_token_buffer *prompt,
                               char *error,
                               size_t error_size) {
    k3_chat_message messages[2];
    size_t message_count = 0u;
    if (!session->started &&
        session->system_prompt != NULL) {
        messages[message_count++] = (k3_chat_message) {
            .role = K3_CHAT_ROLE_SYSTEM,
            .content = session->system_prompt,
        };
    }
    messages[message_count++] = (k3_chat_message) {
        .role = K3_CHAT_ROLE_USER,
        .content = user_text,
    };
    const k3_chat_options render_options = {
        .add_generation_prompt = true,
        .thinking = false,
        .thinking_effort = NULL,
    };
    return k3_tokenizer_encode_chat(
        session->tokenizer, messages, message_count,
        &render_options, prompt, error, error_size);
}

typedef struct {
    k3_chat_prefill_progress_callback callback;
    void                             *data;
    uint32_t                          completed_tokens;
    uint32_t                          chunk_tokens;
    uint32_t                          total_tokens;
    bool                              chunked;
} prefill_progress_bridge;

static void report_layer_progress(uint32_t completed,
                                  uint32_t total,
                                  void *user_data) {
    prefill_progress_bridge *bridge =
        (prefill_progress_bridge *)user_data;
    if (bridge->callback == NULL) return;
    if (!bridge->chunked) {
        bridge->callback(
            K3_CHAT_PREFILL_PROGRESS_LAYERS,
            completed, total, bridge->data);
        return;
    }
    const uint64_t chunk_progress =
        (uint64_t)bridge->chunk_tokens * completed / total;
    uint64_t prompt_progress =
        (uint64_t)bridge->completed_tokens + chunk_progress;
    if (prompt_progress > bridge->total_tokens) {
        prompt_progress = bridge->total_tokens;
    }
    bridge->callback(
        K3_CHAT_PREFILL_PROGRESS_TOKENS,
        (uint32_t)prompt_progress,
        bridge->total_tokens,
        bridge->data);
}

static void add_prefill_stats(k3_engine_prefill_stats *total,
                              const k3_engine_prefill_stats *part) {
    total->routed_physical_read_bytes +=
        part->routed_physical_read_bytes;
    total->embedding_physical_read_bytes +=
        part->embedding_physical_read_bytes;
    if (part->warm_cache_workspace_bytes >
        total->warm_cache_workspace_bytes) {
        total->warm_cache_workspace_bytes =
            part->warm_cache_workspace_bytes;
    }
    total->routed_layer_sweeps += part->routed_layer_sweeps;
    total->expert_read_requests += part->expert_read_requests;
    total->selected_expert_routes += part->selected_expert_routes;
    total->unique_experts_across_layers +=
        part->unique_experts_across_layers;
    if (part->routed_layer_sweeps != 0u &&
        (total->min_unique_experts_per_layer == 0u ||
         part->min_unique_experts_per_layer <
             total->min_unique_experts_per_layer)) {
        total->min_unique_experts_per_layer =
            part->min_unique_experts_per_layer;
    }
    if (part->max_unique_experts_per_layer >
        total->max_unique_experts_per_layer) {
        total->max_unique_experts_per_layer =
            part->max_unique_experts_per_layer;
    }
    total->layer0_seconds += part->layer0_seconds;
    total->attention_seconds += part->attention_seconds;
    total->kda_attention_seconds += part->kda_attention_seconds;
    total->mla_attention_seconds += part->mla_attention_seconds;
    total->kda_projection_seconds += part->kda_projection_seconds;
    total->kda_convolution_seconds += part->kda_convolution_seconds;
    total->kda_recurrent_seconds += part->kda_recurrent_seconds;
    total->kda_gate_norm_seconds += part->kda_gate_norm_seconds;
    total->kda_output_projection_seconds +=
        part->kda_output_projection_seconds;
    total->kda_dequantize_seconds += part->kda_dequantize_seconds;
    total->kda_blas_seconds += part->kda_blas_seconds;
    total->router_seconds += part->router_seconds;
    total->routed_stream_seconds += part->routed_stream_seconds;
    total->routed_read_wait_seconds +=
        part->routed_read_wait_seconds;
    total->routed_submit_seconds += part->routed_submit_seconds;
    total->routed_index_seconds += part->routed_index_seconds;
    total->routed_expert_pipeline_seconds +=
        part->routed_expert_pipeline_seconds;
    total->moe_tail_seconds += part->moe_tail_seconds;
    total->output_seconds += part->output_seconds;
    total->wall_seconds += part->wall_seconds;
}

static bool largest_current_prefill_chunk(
        k3_chat_session *session,
        uint32_t maximum,
        uint32_t *chunk_tokens,
        char *error,
        size_t error_size) {
    if (maximum < 2u) {
        *chunk_tokens = maximum;
        return true;
    }
    uint32_t low = 2u;
    uint32_t high = maximum;
    uint32_t best = 0u;
    while (low <= high) {
        const uint32_t candidate = low + (high - low) / 2u;
        k3_prefill_plan plan;
        char ignored[512] = { 0 };
        if (k3_engine_plan_prefill_with_projection_backend(
                session->engine, candidate,
                K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE,
                session->range_backend, &plan,
                ignored, sizeof(ignored))) {
            best = candidate;
            low = candidate + 1u;
        } else {
            high = candidate - 1u;
        }
    }
    if (best == 0u) {
        set_error(
            error, error_size,
            "no two-token chunk fits the current prefill workspace");
        return false;
    }
    *chunk_tokens = best;
    return true;
}

static void report_lifecycle(
        k3_chat_lifecycle_callback callback,
        void *data,
        k3_chat_lifecycle_event event,
        const k3_chat_turn_result *result) {
    if (callback != NULL) {
        callback(event, result, data);
    }
}

static bool execute_prompt(
        k3_chat_session *session,
        const k3_token_buffer *prompt,
        uint32_t range_chunk_tokens,
        uint32_t *predicted,
        k3_chat_prefill_progress_callback progress_callback,
        void *progress_data,
        k3_chat_turn_result *result,
        char *error,
        size_t error_size) {
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    bool ok = true;
    if (prompt->count <= session->sequential_prefill_limit) {
        result->prefill_strategy = K3_CHAT_PREFILL_SEQUENTIAL;
        float value = 0.0f;
        for (size_t i = 0u; ok && i < prompt->count; i++) {
            ok = k3_engine_forward_token(
                session->engine, prompt->data[i],
                predicted, &value, error, error_size);
            if (ok && progress_callback != NULL) {
                progress_callback(
                    K3_CHAT_PREFILL_PROGRESS_TOKENS,
                    (uint32_t)(i + 1u),
                    (uint32_t)prompt->count,
                    progress_data);
            }
        }
    } else {
        result->prefill_strategy = K3_CHAT_PREFILL_LAYER_MAJOR;
        /*
         * Range calls advance engine state per chunk. The caller publishes
         * retained prompt tokens only after this function succeeds; any
         * partial failure marks the session unhealthy below.
         */
        memset(&result->range_stats, 0, sizeof(result->range_stats));
        const uint32_t total_tokens = (uint32_t)prompt->count;
        uint32_t completed_tokens = 0u;
        bool first_chunk = true;
        float value = 0.0f;
        while (ok && completed_tokens < total_tokens) {
            const uint32_t remaining = total_tokens - completed_tokens;
            uint32_t chunk_tokens = remaining;
            if (range_chunk_tokens != 0u &&
                chunk_tokens > range_chunk_tokens) {
                chunk_tokens = range_chunk_tokens;
            }
            if (range_chunk_tokens != 0u && !first_chunk &&
                !largest_current_prefill_chunk(
                    session, chunk_tokens, &chunk_tokens,
                    error, error_size)) {
                ok = false;
                break;
            }
            if (remaining > chunk_tokens &&
                remaining - chunk_tokens == 1u &&
                chunk_tokens > 2u) {
                chunk_tokens--;
            }
            if (chunk_tokens == 1u) {
                ok = k3_engine_forward_token(
                    session->engine,
                    prompt->data[completed_tokens],
                    predicted, &value, error, error_size);
                if (ok && progress_callback != NULL) {
                    progress_callback(
                        K3_CHAT_PREFILL_PROGRESS_TOKENS,
                        completed_tokens + 1u, total_tokens,
                        progress_data);
                }
            } else {
                k3_engine_prefill_stats measured;
                memset(&measured, 0, sizeof(measured));
                prefill_progress_bridge bridge = {
                    .callback = progress_callback,
                    .data = progress_data,
                    .completed_tokens = completed_tokens,
                    .chunk_tokens = chunk_tokens,
                    .total_tokens = total_tokens,
                    .chunked = range_chunk_tokens != 0u,
                };
                if (session->range_backend ==
                    K3_PREFILL_PROJECTION_DEFAULT) {
                    ok = k3_engine_forward_range_with_progress(
                        session->engine,
                        prompt->data + completed_tokens,
                        chunk_tokens, predicted, &value, &measured,
                        report_layer_progress, &bridge,
                        error, error_size);
                } else {
                    ok =
                        k3_engine_forward_range_with_projection_backend_and_progress(
                            session->engine,
                            prompt->data + completed_tokens,
                            chunk_tokens, session->range_backend,
                            predicted, &value, &measured,
                            report_layer_progress, &bridge,
                            error, error_size);
                }
                if (ok) {
                    add_prefill_stats(&result->range_stats, &measured);
                }
            }
            if (ok) {
                completed_tokens += chunk_tokens;
                first_chunk = false;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    result->prompt_seconds = elapsed_seconds(start, end);
    if (!ok) {
        session->healthy = false;
        return false;
    }
    session->position += (uint32_t)prompt->count;
    session->started = true;
    return true;
}

static bool execute_encoded_turn(
        k3_chat_session *session,
        const k3_token_buffer *prompt,
        uint32_t total_prompt_tokens,
        uint32_t reused_prompt_tokens,
        uint32_t range_chunk_tokens,
        uint32_t max_generated_tokens,
        k3_chat_prefill_progress_callback progress_callback,
        void *progress_data,
        k3_chat_lifecycle_callback lifecycle_callback,
        void *lifecycle_data,
        bool thinking,
        k3_chat_text_callback reasoning_callback,
        void *reasoning_data,
        k3_chat_text_callback callback,
        void *callback_data,
        k3_chat_turn_result *result,
        char *error,
        size_t error_size) {
    k3_text_buffer token_piece = { 0 };
    k3_token_buffer generated_tokens = { 0 };
    k3_token_buffer thinking_trailer = { 0 };
    bool ok = true;
    bool mutated = false;
    bool diagnostics_active = false;
    if (thinking && !k3_tokenizer_encode(
            session->tokenizer,
            "<|close|>think<|sep|><|open|>response<|sep|>"
            "<|close|>response<|sep|><|close|>message<|sep|>"
            "<|end_of_msg|>",
            true, &thinking_trailer, error, error_size)) {
        ok = false;
        goto cleanup;
    }
    const uint32_t *initial_trailer = thinking ?
        thinking_trailer.data : K3_RESPONSE_TRAILER;
    if (prompt->count < 2u || prompt->count > UINT32_MAX) {
        set_error(error, error_size,
                  "rendered prompt token count %zu is invalid",
                  prompt->count);
        ok = false;
        goto cleanup;
    }
    const size_t response_trailer_count =
        sizeof(K3_RESPONSE_TRAILER) /
        sizeof(K3_RESPONSE_TRAILER[0]);
    const size_t initial_trailer_count = thinking ?
        thinking_trailer.count : response_trailer_count;
    if (thinking &&
        (initial_trailer_count <= response_trailer_count ||
         memcmp(
             thinking_trailer.data + initial_trailer_count -
                 response_trailer_count,
             K3_RESPONSE_TRAILER,
             response_trailer_count *
                 sizeof(*K3_RESPONSE_TRAILER)) != 0)) {
        set_error(error, error_size,
                  "thinking trailer does not end with response closure");
        ok = false;
        goto cleanup;
    }
    const size_t thinking_transition_count = thinking ?
        initial_trailer_count - response_trailer_count : 0u;
    const size_t maximum_trailer_count =
        initial_trailer_count > response_trailer_count ?
            initial_trailer_count : response_trailer_count;
    if (session->position > session->context ||
        prompt->count > session->context - session->position ||
        session->context - session->position -
            (uint32_t)prompt->count <= maximum_trailer_count) {
        set_error(error, error_size,
                  "chat turn does not fit the remaining %u-token context",
                  session->context - session->position);
        ok = false;
        goto cleanup;
    }
    uint32_t available =
        session->context - session->position -
        (uint32_t)prompt->count - (uint32_t)maximum_trailer_count;
    uint32_t generation_limit =
        max_generated_tokens < available ?
            max_generated_tokens : available;
    if (generation_limit == 0u) {
        set_error(error, error_size,
                  "no context remains for generated content");
        ok = false;
        goto cleanup;
    }

    if (!reserve_retained_tokens(
            session,
            prompt->count + (size_t)generation_limit +
                maximum_trailer_count,
            error, error_size)) {
        ok = false;
        goto cleanup;
    }

    result->prompt_tokens = total_prompt_tokens;
    result->prompt_evaluated_tokens =
        (uint32_t)prompt->count;
    result->prompt_reused_tokens = reused_prompt_tokens;
    result->thinking = thinking;
    k3_engine_get_cache_stats(
        session->engine, &result->cache_before);
    uint32_t predicted = 0u;
    if (!execute_prompt(
            session, prompt, range_chunk_tokens, &predicted,
            progress_callback, progress_data,
            result, error, error_size)) {
        ok = false;
        goto cleanup;
    }
    report_lifecycle(
        lifecycle_callback, lifecycle_data,
        K3_CHAT_LIFECYCLE_PREFILL_COMPLETE, result);
    mutated = true;
    if (!append_retained_tokens(
            session, prompt->data, prompt->count,
            error, error_size)) {
        ok = false;
        goto cleanup;
    }

    typedef enum {
        K3_OUTPUT_REASONING = 0,
        K3_OUTPUT_BETWEEN_CHANNELS = 1,
        K3_OUTPUT_RESPONSE = 2,
        K3_OUTPUT_CLOSED = 3,
    } k3_output_channel;
    k3_output_channel channel = thinking ?
        K3_OUTPUT_REASONING : K3_OUTPUT_RESPONSE;
    size_t thinking_transition_progress = 0u;
    bool natural_stop = false;
    uint32_t tail[32];
    size_t tail_count = 0u;
    if (!k3_engine_begin_decode_diagnostics(
            session->engine, error, error_size)) {
        ok = false;
        goto cleanup;
    }
    diagnostics_active = true;
    struct timespec decode_start;
    struct timespec decode_end;
    clock_gettime(CLOCK_MONOTONIC, &decode_start);
    report_lifecycle(
        lifecycle_callback, lifecycle_data,
        K3_CHAT_LIFECYCLE_DECODE_START, result);
    for (uint32_t generated = 0u;
         generated < generation_limit;
         generated++) {
        const uint32_t token = predicted;
        result->generated_tokens++;
        append_tail(
            tail, &tail_count,
            sizeof(tail) / sizeof(tail[0]), token);
        if (!append_generated_token(
                &generated_tokens, token,
                error, error_size)) {
            ok = false;
            goto cleanup;
        }

        if (token == K3_TOKEN_END_OF_MSG) {
            uint32_t discard = 0u;
            if (!feed_token(
                    session, token, &discard,
                    error, error_size)) {
                ok = false;
                goto cleanup;
            }
            if (!append_retained_tokens(
                    session, &token, 1u,
                    error, error_size)) {
                ok = false;
                goto cleanup;
            }
            natural_stop = true;
            break;
        }

        k3_text_buffer *channel_output = NULL;
        k3_chat_text_callback channel_callback = NULL;
        void *channel_data = NULL;
        if (channel == K3_OUTPUT_REASONING) {
            if (token == K3_TOKEN_CLOSE) {
                channel = K3_OUTPUT_BETWEEN_CHANNELS;
                thinking_transition_progress = 1u;
            } else if (token < K3_TOKEN_BOS) {
                channel_output = &result->reasoning_content;
                channel_callback = reasoning_callback;
                channel_data = reasoning_data;
            }
        } else if (channel == K3_OUTPUT_BETWEEN_CHANNELS) {
            if (thinking_transition_progress >=
                    thinking_transition_count ||
                token != initial_trailer[
                    thinking_transition_progress]) {
                set_error(error, error_size,
                          "model emitted an invalid think/response transition");
                ok = false;
                goto cleanup;
            }
            thinking_transition_progress++;
            if (thinking_transition_progress ==
                thinking_transition_count) {
                channel = K3_OUTPUT_RESPONSE;
                report_lifecycle(
                    lifecycle_callback, lifecycle_data,
                    K3_CHAT_LIFECYCLE_RESPONSE_START, result);
            }
        } else if (channel == K3_OUTPUT_RESPONSE) {
            if (token == K3_TOKEN_CLOSE) {
                channel = K3_OUTPUT_CLOSED;
            } else if (token < K3_TOKEN_BOS) {
                channel_output = &result->response;
                channel_callback = callback;
                channel_data = callback_data;
            }
        }
        if (channel_output != NULL) {
            if (!k3_tokenizer_decode(
                    session->tokenizer, &token, 1u, false,
                    &token_piece, error, error_size) ||
                !append_response_bytes(
                    channel_output,
                    token_piece.data, token_piece.size,
                    error, error_size)) {
                ok = false;
                goto cleanup;
            }
            if (channel_callback != NULL &&
                token_piece.size != 0u) {
                channel_callback(
                    token_piece.data, token_piece.size,
                    channel_data);
            }
        }

        if (!feed_token(
                session, token, &predicted,
                error, error_size)) {
            ok = false;
            goto cleanup;
        }
        if (!append_retained_tokens(
                session, &token, 1u,
                error, error_size)) {
            ok = false;
            goto cleanup;
        }
        if ((result->generated_tokens & 63u) == 0u) {
            report_lifecycle(
                lifecycle_callback, lifecycle_data,
                K3_CHAT_LIFECYCLE_DECODE_PROGRESS, result);
        }
    }

    if (!natural_stop) {
        const bool still_reasoning =
            channel == K3_OUTPUT_REASONING;
        const bool between_channels =
            channel == K3_OUTPUT_BETWEEN_CHANNELS;
        const uint32_t *trailer = between_channels ?
            initial_trailer + thinking_transition_progress :
            (still_reasoning ? initial_trailer : K3_RESPONSE_TRAILER);
        const size_t trailer_count = between_channels ?
            initial_trailer_count - thinking_transition_progress :
            (still_reasoning ?
                initial_trailer_count : response_trailer_count);
        const size_t matched = between_channels ? 0u :
            trailer_prefix_already_generated(
                tail, tail_count, trailer, trailer_count);
        uint32_t discard = predicted;
        for (size_t i = matched; i < trailer_count; i++) {
            if (!feed_token(
                    session, trailer[i],
                    &discard, error, error_size)) {
                ok = false;
                goto cleanup;
            }
            if (!append_retained_tokens(
                    session, &trailer[i], 1u,
                    error, error_size)) {
                ok = false;
                goto cleanup;
            }
            result->forced_trailer_tokens++;
        }
        result->finish_reason = K3_CHAT_FINISH_LENGTH;
    } else {
        k3_assistant_output parsed;
        if (!k3_tokenizer_parse_assistant_output(
                session->tokenizer,
                generated_tokens.data,
                generated_tokens.count,
                thinking,
                &parsed, error, error_size)) {
            ok = false;
            goto cleanup;
        }
        k3_text_buffer_free(&result->reasoning_content);
        k3_text_buffer_free(&result->response);
        result->reasoning_content = parsed.reasoning;
        result->response = parsed.response;
        result->tool_calls = parsed.tool_calls;
        result->tool_call_count = parsed.tool_call_count;
        memset(&parsed, 0, sizeof(parsed));
        result->finish_reason = result->tool_call_count == 0u ?
            K3_CHAT_FINISH_END_OF_MESSAGE :
            K3_CHAT_FINISH_TOOL_CALLS;
    }
    clock_gettime(CLOCK_MONOTONIC, &decode_end);
    result->decode_seconds =
        elapsed_seconds(decode_start, decode_end);
    result->tokens_per_second =
        result->decode_seconds > 0.0 ?
            (double)result->generated_tokens /
                result->decode_seconds : 0.0;
    if (session->capture_state_digest) {
        if (!k3_engine_get_state_digest(
                session->engine, &result->state_digest,
                error, error_size)) {
            ok = false;
            goto cleanup;
        }
        result->state_digest_valid = true;
    }
    if (!k3_engine_end_decode_diagnostics(
            session->engine, result->decode_seconds,
            &result->decode_stats, error, error_size)) {
        diagnostics_active = false;
        ok = false;
        goto cleanup;
    }
    diagnostics_active = false;
    result->position = session->position;
    k3_engine_get_cache_stats(
        session->engine, &result->cache_after);

cleanup:
    if (diagnostics_active) {
        k3_engine_abort_decode_diagnostics(session->engine);
    }
    k3_token_buffer_free(&thinking_trailer);
    k3_token_buffer_free(&generated_tokens);
    k3_text_buffer_free(&token_piece);
    if (!ok) {
        k3_text_buffer_free(&result->reasoning_content);
        k3_text_buffer_free(&result->response);
        for (size_t i = 0u; i < result->tool_call_count; i++) {
            free((char *)result->tool_calls[i].id);
            free((char *)result->tool_calls[i].name);
            free((char *)result->tool_calls[i].arguments);
        }
        free(result->tool_calls);
        result->tool_calls = NULL;
        result->tool_call_count = 0u;
        if (mutated || !session->healthy) {
            session->healthy = false;
            session->retained_tokens.count = 0u;
        }
    }
    return ok;
}

static bool preflight_replacement_prefill(
        k3_chat_session *session,
        size_t prompt_tokens,
        bool clear_expert_cache,
        uint32_t *range_chunk_tokens,
        char *error,
        size_t error_size) {
    *range_chunk_tokens = 0u;
    if (prompt_tokens <= session->sequential_prefill_limit) {
        return true;
    }
    k3_prefill_plan plan;
    char full_plan_error[512] = { 0 };
    if (k3_engine_plan_reset_prefill(
            session->engine, (uint32_t)prompt_tokens,
            clear_expert_cache, session->range_backend,
            &plan, full_plan_error, sizeof(full_plan_error))) {
        return true;
    }
    uint32_t low = 2u;
    uint32_t high = (uint32_t)prompt_tokens - 1u;
    uint32_t best = 0u;
    while (low <= high) {
        const uint32_t candidate = low + (high - low) / 2u;
        char ignored[512] = { 0 };
        if (k3_engine_plan_reset_prefill(
                session->engine, candidate,
                clear_expert_cache, session->range_backend,
                &plan, ignored, sizeof(ignored))) {
            best = candidate;
            low = candidate + 1u;
        } else {
            high = candidate - 1u;
        }
    }
    if (best != 0u) {
        *range_chunk_tokens = best;
        return true;
    }
    set_error(
        error, error_size,
        "replacement prefill rejected before causal reset: %s",
        full_plan_error[0] != '\0' ?
            full_plan_error : "no two-token chunk fits");
    return false;
}

bool k3_chat_session_turn(
        k3_chat_session *session,
        const char *user_text,
        uint32_t max_generated_tokens,
        k3_chat_text_callback callback,
        void *callback_data,
        k3_chat_turn_result *result,
        char *error,
        size_t error_size) {
    if (session == NULL || user_text == NULL ||
        result == NULL || max_generated_tokens == 0u) {
        set_error(error, error_size,
                  "chat turn arguments are invalid");
        return false;
    }
    if (!session->healthy) {
        set_error(error, error_size,
                  "chat session is invalid after an earlier execution "
                  "failure; import, reset, or recreate it");
        return false;
    }
    memset(result, 0, sizeof(*result));
    /* The incremental text API cannot later reconstruct OpenAI metadata. */
    clear_request_directive_history(session);
    k3_token_buffer prompt = { 0 };
    bool ok = encode_turn_prompt(
        session, user_text, &prompt, error, error_size);
    if (ok) {
        ok = execute_encoded_turn(
            session, &prompt,
            (uint32_t)prompt.count, 0u, 0u,
            max_generated_tokens, NULL, NULL,
            NULL, NULL,
            false, NULL, NULL,
            callback, callback_data, result,
            error, error_size);
    }
    k3_token_buffer_free(&prompt);
    return ok;
}

bool k3_chat_session_reset(
        k3_chat_session *session,
        bool clear_expert_cache,
        char *error,
        size_t error_size) {
    if (session == NULL) {
        set_error(error, error_size,
                  "chat reset needs a session");
        return false;
    }
    if (!k3_engine_reset_state(
            session->engine, clear_expert_cache,
            error, error_size)) {
        session->healthy = false;
        return false;
    }
    session->position = 0u;
    session->retained_tokens.count = 0u;
    clear_request_directive_history(session);
    session->started = false;
    session->healthy = true;
    return true;
}

bool k3_chat_session_complete_messages(
        k3_chat_session *session,
        const k3_chat_message *messages,
        size_t message_count,
        uint32_t max_generated_tokens,
        k3_chat_text_callback callback,
        void *callback_data,
        k3_chat_turn_result *result,
        char *error,
        size_t error_size) {
    return k3_chat_session_complete_messages_with_options(
        session, messages, message_count,
        max_generated_tokens, NULL,
        callback, callback_data, result,
        error, error_size);
}

bool k3_chat_session_complete_messages_with_options(
        k3_chat_session *session,
        const k3_chat_message *messages,
        size_t message_count,
        uint32_t max_generated_tokens,
        const k3_chat_completion_options *options,
        k3_chat_text_callback callback,
        void *callback_data,
        k3_chat_turn_result *result,
        char *error,
        size_t error_size) {
    if (session == NULL || messages == NULL ||
        message_count == 0u || result == NULL ||
        max_generated_tokens == 0u) {
        set_error(error, error_size,
                  "chat completion arguments are invalid");
        return false;
    }
    memset(result, 0, sizeof(*result));
    const k3_chat_options render_options = {
        .add_generation_prompt = true,
        .thinking = options != NULL && options->thinking,
        .thinking_effort = options == NULL ? NULL :
            options->thinking_effort,
        .tools_json = options == NULL ? NULL : options->tools_json,
        .tool_choice = options == NULL ? K3_TOOL_CHOICE_AUTO :
            options->tool_choice,
        .enforce_single_tool_call = options != NULL &&
            options->enforce_single_tool_call,
        .response_format = options == NULL ?
            K3_RESPONSE_FORMAT_TEXT : options->response_format,
        .response_schema_json = options == NULL ? NULL :
            options->response_schema_json,
        .historical_tool_choices = NULL,
        .historical_tool_choice_count = 0u,
        .historical_single_tool_calls = NULL,
        .historical_single_tool_call_count = 0u,
        .historical_response_formats = NULL,
        .historical_response_format_count = 0u,
    };
    k3_token_buffer canonical_prompt = { 0 };
    k3_token_buffer augmented_prompt = { 0 };
    bool ok = k3_tokenizer_encode_chat(
        session->tokenizer, messages, message_count,
        &render_options, &canonical_prompt, error, error_size);
    const bool clear_expert_cache =
        options != NULL && options->clear_expert_cache;
    size_t reused = 0u;
    size_t reuse_matched = 0u;
    size_t reuse_candidate_count = canonical_prompt.count;
    k3_token_buffer *prompt = &canonical_prompt;
    const bool reuse_eligible =
        ok && options != NULL && options->reuse_prefix &&
        !clear_expert_cache && session->healthy &&
        session->retained_tokens.count ==
            (size_t)session->position &&
        session->retained_tokens.count != 0u;

    bool marker_boundaries_valid = true;
    for (size_t i = 0u;
         i < session->historical_tool_choice_count; i++) {
        if (session->historical_tool_choices[i]
                .after_message_count > message_count) {
            marker_boundaries_valid = false;
            break;
        }
    }
    for (size_t i = 0u;
         marker_boundaries_valid &&
         i < session->historical_single_tool_call_count; i++) {
        if (session->historical_single_tool_calls[i]
                .after_message_count > message_count) {
            marker_boundaries_valid = false;
        }
    }
    for (size_t i = 0u;
         marker_boundaries_valid &&
         i < session->historical_response_format_count; i++) {
        if (session->historical_response_formats[i]
                .after_message_count > message_count) {
            marker_boundaries_valid = false;
        }
    }
    if (reuse_eligible && marker_boundaries_valid &&
        (session->historical_tool_choice_count != 0u ||
         session->historical_single_tool_call_count != 0u ||
         session->historical_response_format_count != 0u)) {
        k3_chat_options augmented_options = render_options;
        augmented_options.historical_tool_choices =
            session->historical_tool_choices;
        augmented_options.historical_tool_choice_count =
            session->historical_tool_choice_count;
        augmented_options.historical_single_tool_calls =
            session->historical_single_tool_calls;
        augmented_options.historical_single_tool_call_count =
            session->historical_single_tool_call_count;
        augmented_options.historical_response_formats =
            session->historical_response_formats;
        augmented_options.historical_response_format_count =
            session->historical_response_format_count;
        char ignored_error[256] = { 0 };
        if (k3_tokenizer_encode_chat(
                session->tokenizer, messages, message_count,
                &augmented_options, &augmented_prompt,
                ignored_error, sizeof(ignored_error))) {
            const size_t matched =
                k3_prefix_reuse_common_tokens(
                    session->retained_tokens.data,
                    session->retained_tokens.count,
                    augmented_prompt.data,
                    augmented_prompt.count);
            if (matched > reuse_matched) {
                reuse_matched = matched;
                reuse_candidate_count = augmented_prompt.count;
            }
            if (k3_prefix_reuse_admits(
                    session->retained_tokens.data,
                    session->retained_tokens.count,
                    augmented_prompt.data,
                    augmented_prompt.count)) {
                prompt = &augmented_prompt;
                reused = session->retained_tokens.count;
            }
        }
    }
    if (reuse_eligible && reused == 0u) {
        const size_t matched =
            k3_prefix_reuse_common_tokens(
                session->retained_tokens.data,
                session->retained_tokens.count,
                canonical_prompt.data,
                canonical_prompt.count);
        if (matched > reuse_matched) {
            reuse_matched = matched;
            reuse_candidate_count = canonical_prompt.count;
        }
        if (k3_prefix_reuse_admits(
                session->retained_tokens.data,
                session->retained_tokens.count,
                canonical_prompt.data,
                canonical_prompt.count)) {
            reused = session->retained_tokens.count;
        }
    }
    if (reuse_eligible && reused == 0u &&
        session->retained_tokens.count <= UINT32_MAX &&
        reuse_matched <= UINT32_MAX &&
        reuse_candidate_count <= UINT32_MAX) {
        result->prompt_reuse_retained_tokens =
            (uint32_t)session->retained_tokens.count;
        result->prompt_reuse_matched_tokens =
            (uint32_t)reuse_matched;
        result->prompt_reuse_candidate_tokens =
            (uint32_t)reuse_candidate_count;
        result->prompt_reuse_declined = true;
    }
    if (ok && (prompt->count < 2u ||
               prompt->count > UINT32_MAX)) {
        set_error(error, error_size,
                  "rendered prompt token count %zu is invalid",
                  prompt->count);
        ok = false;
    }
    if (ok && reused > UINT32_MAX) {
        set_error(error, error_size,
                  "reused prompt token count is too large");
        ok = false;
    }
    if (ok) {
        result->prompt_tokens = (uint32_t)prompt->count;
        result->prompt_evaluated_tokens =
            (uint32_t)(prompt->count - reused);
        result->prompt_reused_tokens = (uint32_t)reused;
        result->prefill_strategy =
            prompt->count - reused <=
                session->sequential_prefill_limit ?
                K3_CHAT_PREFILL_SEQUENTIAL :
                K3_CHAT_PREFILL_LAYER_MAJOR;
        report_lifecycle(
            options == NULL ? NULL :
                options->lifecycle_callback,
            options == NULL ? NULL :
                options->lifecycle_data,
            K3_CHAT_LIFECYCLE_PREFILL_START, result);
    }
    uint32_t range_chunk_tokens = 0u;
    if (ok && reused == 0u) {
        ok = preflight_replacement_prefill(
            session, prompt->count, clear_expert_cache,
            &range_chunk_tokens, error, error_size);
    }
    if (ok && reused == 0u) {
        ok = k3_chat_session_reset(
            session, clear_expert_cache,
            error, error_size);
    }
    if (ok) {
        const k3_token_buffer suffix = {
            .data = prompt->data + reused,
            .count = prompt->count - reused,
            .capacity = prompt->count - reused,
        };
        ok = execute_encoded_turn(
            session, &suffix,
            (uint32_t)prompt->count, (uint32_t)reused,
            range_chunk_tokens, max_generated_tokens,
            options == NULL ? NULL :
                options->progress_callback,
            options == NULL ? NULL :
                options->progress_data,
            options == NULL ? NULL :
                options->lifecycle_callback,
            options == NULL ? NULL :
                options->lifecycle_data,
            options != NULL && options->thinking,
            options == NULL ? NULL :
                options->reasoning_callback,
            options == NULL ? NULL :
                options->reasoning_data,
            callback, callback_data, result,
            error, error_size);
    }
    if (ok && options != NULL &&
        options->preserve_request_directive_history) {
        remember_tool_choice(
            session, message_count, options->tool_choice);
        remember_single_tool_call(
            session, message_count,
            options->enforce_single_tool_call);
        remember_response_format(
            session, message_count,
            options->response_format,
            options->response_schema_json);
    }
    k3_token_buffer_free(&augmented_prompt);
    k3_token_buffer_free(&canonical_prompt);
    return ok;
}

bool k3_chat_session_export_state(
        const k3_chat_session *session,
        const char *path,
        k3_engine_state_file_info *info,
        char *error,
        size_t error_size) {
    if (session == NULL || !session->healthy) {
        set_error(error, error_size,
                  "cannot export an invalid chat session");
        return false;
    }
    return k3_engine_export_state(
        session->engine, path, info, error, error_size);
}

bool k3_chat_session_import_state(
        k3_chat_session *session,
        const char *path,
        k3_engine_state_file_info *info,
        char *error,
        size_t error_size) {
    if (session == NULL) {
        set_error(error, error_size,
                  "chat import needs a session");
        return false;
    }
    k3_engine_state_file_info imported;
    if (!k3_engine_import_state(
            session->engine, path, &imported,
            error, error_size)) {
        return false;
    }
    session->position = imported.token_position;
    session->retained_tokens.count = 0u;
    clear_request_directive_history(session);
    session->started = session->position != 0u;
    session->healthy = true;
    if (info != NULL) {
        *info = imported;
    }
    return true;
}
