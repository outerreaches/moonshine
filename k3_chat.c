#include "k3_chat.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    K3_CHAT_DEFAULT_CONTEXT = 8192,
    K3_CHAT_DEFAULT_SEQUENTIAL_LIMIT = 128,
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
        size_t tail_count) {
    const size_t trailer_count =
        sizeof(K3_RESPONSE_TRAILER) /
        sizeof(K3_RESPONSE_TRAILER[0]);
    size_t matched = tail_count < trailer_count ?
        tail_count : trailer_count;
    for (;;) {
        if (matched == 0u ||
            memcmp(
                tail + tail_count - matched,
                K3_RESPONSE_TRAILER,
                matched * sizeof(*tail)) == 0) {
            return matched;
        }
        matched--;
    }
}

static void append_tail(uint32_t tail[7],
                        size_t *tail_count,
                        uint32_t token) {
    const size_t capacity =
        sizeof(K3_RESPONSE_TRAILER) /
        sizeof(K3_RESPONSE_TRAILER[0]);
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
            K3_CHAT_DEFAULT_SEQUENTIAL_LIMIT :
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
    free(session->system_prompt);
    free(session);
}

void k3_chat_turn_result_free(k3_chat_turn_result *result) {
    if (result == NULL) {
        return;
    }
    k3_text_buffer_free(&result->response);
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
    const k3_chat_options options = {
        .add_generation_prompt = true,
        .thinking = false,
        .thinking_effort = NULL,
    };
    return k3_tokenizer_encode_chat(
        session->tokenizer, messages, message_count,
        &options, prompt, error, error_size);
}

static bool execute_prompt(k3_chat_session *session,
                           const k3_token_buffer *prompt,
                           uint32_t *predicted,
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
        }
    } else {
        result->prefill_strategy = K3_CHAT_PREFILL_LAYER_MAJOR;
        float value = 0.0f;
        ok = k3_engine_forward_range(
            session->engine, prompt->data,
            (uint32_t)prompt->count,
            predicted, &value, &result->range_stats,
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
        uint32_t max_generated_tokens,
        k3_chat_text_callback callback,
        void *callback_data,
        k3_chat_turn_result *result,
        char *error,
        size_t error_size) {
    k3_text_buffer token_piece = { 0 };
    bool ok = true;
    if (prompt->count < 2u || prompt->count > UINT32_MAX) {
        set_error(error, error_size,
                  "rendered prompt token count %zu is invalid",
                  prompt->count);
        ok = false;
        goto cleanup;
    }
    const size_t trailer_count =
        sizeof(K3_RESPONSE_TRAILER) /
        sizeof(K3_RESPONSE_TRAILER[0]);
    if (session->position > session->context ||
        prompt->count > session->context - session->position ||
        session->context - session->position -
            (uint32_t)prompt->count <= trailer_count) {
        set_error(error, error_size,
                  "chat turn does not fit the remaining %u-token context",
                  session->context - session->position);
        ok = false;
        goto cleanup;
    }
    uint32_t available =
        session->context - session->position -
        (uint32_t)prompt->count - (uint32_t)trailer_count;
    uint32_t generation_limit =
        max_generated_tokens < available ?
            max_generated_tokens : available;
    if (generation_limit == 0u) {
        set_error(error, error_size,
                  "no context remains for generated content");
        ok = false;
        goto cleanup;
    }

    result->prompt_tokens = (uint32_t)prompt->count;
    k3_engine_get_cache_stats(
        session->engine, &result->cache_before);
    uint32_t predicted = 0u;
    if (!execute_prompt(
            session, prompt, &predicted,
            result, error, error_size)) {
        ok = false;
        goto cleanup;
    }

    bool response_open = true;
    bool natural_stop = false;
    uint32_t tail[7];
    size_t tail_count = 0u;
    struct timespec decode_start;
    struct timespec decode_end;
    clock_gettime(CLOCK_MONOTONIC, &decode_start);
    for (uint32_t generated = 0u;
         generated < generation_limit;
         generated++) {
        const uint32_t token = predicted;
        result->generated_tokens++;
        append_tail(tail, &tail_count, token);

        if (token == K3_TOKEN_END_OF_MSG) {
            uint32_t discard = 0u;
            if (!feed_token(
                    session, token, &discard,
                    error, error_size)) {
                ok = false;
                goto cleanup;
            }
            natural_stop = true;
            break;
        }

        if (response_open && token == K3_TOKEN_CLOSE) {
            response_open = false;
        } else if (response_open && token < K3_TOKEN_BOS) {
            if (!k3_tokenizer_decode(
                    session->tokenizer, &token, 1u, false,
                    &token_piece, error, error_size) ||
                !append_response_bytes(
                    &result->response,
                    token_piece.data, token_piece.size,
                    error, error_size)) {
                ok = false;
                goto cleanup;
            }
            if (callback != NULL && token_piece.size != 0u) {
                callback(
                    token_piece.data, token_piece.size,
                    callback_data);
            }
        }

        if (!feed_token(
                session, token, &predicted,
                error, error_size)) {
            ok = false;
            goto cleanup;
        }
    }

    if (!natural_stop) {
        const size_t matched =
            trailer_prefix_already_generated(tail, tail_count);
        uint32_t discard = predicted;
        for (size_t i = matched; i < trailer_count; i++) {
            if (!feed_token(
                    session, K3_RESPONSE_TRAILER[i],
                    &discard, error, error_size)) {
                ok = false;
                goto cleanup;
            }
            result->forced_trailer_tokens++;
        }
        result->finish_reason = K3_CHAT_FINISH_LENGTH;
    } else {
        result->finish_reason =
            K3_CHAT_FINISH_END_OF_MESSAGE;
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
    k3_text_buffer_free(&token_piece);
    if (!ok) {
        k3_text_buffer_free(&result->response);
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
    k3_token_buffer prompt = { 0 };
    bool ok = encode_turn_prompt(
        session, user_text, &prompt, error, error_size);
    if (ok) {
        ok = execute_encoded_turn(
            session, &prompt, max_generated_tokens,
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
    if (session == NULL || messages == NULL ||
        message_count == 0u || result == NULL ||
        max_generated_tokens == 0u) {
        set_error(error, error_size,
                  "chat completion arguments are invalid");
        return false;
    }
    memset(result, 0, sizeof(*result));
    if (!k3_chat_session_reset(
            session, true, error, error_size)) {
        return false;
    }
    const k3_chat_options options = {
        .add_generation_prompt = true,
        .thinking = false,
        .thinking_effort = NULL,
    };
    k3_token_buffer prompt = { 0 };
    bool ok = k3_tokenizer_encode_chat(
        session->tokenizer, messages, message_count,
        &options, &prompt, error, error_size);
    if (ok) {
        ok = execute_encoded_turn(
            session, &prompt, max_generated_tokens,
            callback, callback_data, result,
            error, error_size);
    }
    k3_token_buffer_free(&prompt);
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
    session->started = session->position != 0u;
    session->healthy = true;
    if (info != NULL) {
        *info = imported;
    }
    return true;
}
