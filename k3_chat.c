#include "k3_chat.h"

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
    uint32_t      position;
    k3_token_buffer retained_tokens;
    k3_tool_choice_marker *historical_tool_choices;
    size_t historical_tool_choice_count;
    size_t historical_tool_choice_capacity;
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

static void clear_tool_choice_history(k3_chat_session *session) {
    session->historical_tool_choice_count = 0u;
}

/*
 * Tool-choice history is an optimization hint, not inference output. If its
 * allocation fails or the caller presents a non-append-only boundary, discard
 * it and let the next request take the exact-prefix mismatch fallback.
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
        clear_tool_choice_history(session);
        return;
    }
    if (session->historical_tool_choice_count != 0u &&
        session->historical_tool_choices[
            session->historical_tool_choice_count - 1u]
                .after_message_count >= after_message_count) {
        clear_tool_choice_history(session);
        return;
    }
    if (session->historical_tool_choice_count ==
        session->historical_tool_choice_capacity) {
        const size_t old_capacity =
            session->historical_tool_choice_capacity;
        if (old_capacity > SIZE_MAX / 2u) {
            clear_tool_choice_history(session);
            return;
        }
        const size_t capacity =
            old_capacity == 0u ? 4u : old_capacity * 2u;
        if (capacity > SIZE_MAX /
                sizeof(*session->historical_tool_choices)) {
            clear_tool_choice_history(session);
            return;
        }
        k3_tool_choice_marker *markers =
            (k3_tool_choice_marker *)realloc(
                session->historical_tool_choices,
                capacity * sizeof(*markers));
        if (markers == NULL) {
            clear_tool_choice_history(session);
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
    free(session->historical_tool_choices);
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
} prefill_progress_bridge;

static void report_layer_progress(uint32_t completed,
                                  uint32_t total,
                                  void *user_data) {
    prefill_progress_bridge *bridge =
        (prefill_progress_bridge *)user_data;
    if (bridge->callback != NULL) {
        bridge->callback(
            K3_CHAT_PREFILL_PROGRESS_LAYERS,
            completed, total, bridge->data);
    }
}

static bool execute_prompt(
        k3_chat_session *session,
        const k3_token_buffer *prompt,
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
        float value = 0.0f;
        prefill_progress_bridge bridge = {
            .callback = progress_callback,
            .data = progress_data,
        };
        ok = k3_engine_forward_range_with_progress(
            session->engine, prompt->data,
            (uint32_t)prompt->count,
            predicted, &value, &result->range_stats,
            report_layer_progress, &bridge,
            error, error_size);
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
        uint32_t max_generated_tokens,
        k3_chat_prefill_progress_callback progress_callback,
        void *progress_data,
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
            session, prompt, &predicted,
            progress_callback, progress_data,
            result, error, error_size)) {
        ok = false;
        goto cleanup;
    }
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
    struct timespec decode_start;
    struct timespec decode_end;
    clock_gettime(CLOCK_MONOTONIC, &decode_start);
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
    result->position = session->position;
    k3_engine_get_cache_stats(
        session->engine, &result->cache_after);

cleanup:
    k3_token_buffer_free(&thinking_trailer);
    k3_token_buffer_free(&generated_tokens);
    k3_text_buffer_free(&token_piece);
    if (!ok) {
        k3_text_buffer_free(&result->reasoning_content);
        k3_text_buffer_free(&result->response);
        for (size_t i = 0u; i < result->tool_call_count; i++) {
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
    clear_tool_choice_history(session);
    k3_token_buffer prompt = { 0 };
    bool ok = encode_turn_prompt(
        session, user_text, &prompt, error, error_size);
    if (ok) {
        ok = execute_encoded_turn(
            session, &prompt,
            (uint32_t)prompt.count, 0u,
            max_generated_tokens, NULL, NULL,
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
    clear_tool_choice_history(session);
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
        .response_format = options == NULL ?
            K3_RESPONSE_FORMAT_TEXT : options->response_format,
        .response_schema_json = options == NULL ? NULL :
            options->response_schema_json,
        .historical_tool_choices = NULL,
        .historical_tool_choice_count = 0u,
    };
    k3_token_buffer canonical_prompt = { 0 };
    k3_token_buffer augmented_prompt = { 0 };
    bool ok = k3_tokenizer_encode_chat(
        session->tokenizer, messages, message_count,
        &render_options, &canonical_prompt, error, error_size);
    const bool clear_expert_cache =
        options != NULL && options->clear_expert_cache;
    size_t reused = 0u;
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
    if (reuse_eligible && marker_boundaries_valid &&
        session->historical_tool_choice_count != 0u) {
        k3_chat_options augmented_options = render_options;
        augmented_options.historical_tool_choices =
            session->historical_tool_choices;
        augmented_options.historical_tool_choice_count =
            session->historical_tool_choice_count;
        char ignored_error[256] = { 0 };
        if (k3_tokenizer_encode_chat(
                session->tokenizer, messages, message_count,
                &augmented_options, &augmented_prompt,
                ignored_error, sizeof(ignored_error)) &&
            augmented_prompt.count >=
                session->retained_tokens.count + 2u &&
            memcmp(
                augmented_prompt.data,
                session->retained_tokens.data,
                session->retained_tokens.count *
                    sizeof(*augmented_prompt.data)) == 0) {
            prompt = &augmented_prompt;
            reused = session->retained_tokens.count;
        }
    }
    if (reuse_eligible && reused == 0u &&
        canonical_prompt.count >=
            session->retained_tokens.count + 2u &&
        memcmp(
            canonical_prompt.data, session->retained_tokens.data,
            session->retained_tokens.count *
                sizeof(*canonical_prompt.data)) == 0) {
        reused = session->retained_tokens.count;
    }
    if (ok && reused == 0u) {
        ok = k3_chat_session_reset(
            session, clear_expert_cache,
            error, error_size);
    }
    if (ok) {
        if (prompt->count > UINT32_MAX ||
            reused > UINT32_MAX) {
            set_error(error, error_size,
                      "rendered prompt is too large");
            ok = false;
        }
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
            max_generated_tokens,
            options == NULL ? NULL :
                options->progress_callback,
            options == NULL ? NULL :
                options->progress_data,
            options != NULL && options->thinking,
            options == NULL ? NULL :
                options->reasoning_callback,
            options == NULL ? NULL :
                options->reasoning_data,
            callback, callback_data, result,
            error, error_size);
    }
    if (ok && options != NULL &&
        options->preserve_tool_choice_history) {
        remember_tool_choice(
            session, message_count, options->tool_choice);
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
    clear_tool_choice_history(session);
    session->started = session->position != 0u;
    session->healthy = true;
    if (info != NULL) {
        *info = imported;
    }
    return true;
}
