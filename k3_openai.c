#include "k3_openai.h"

#include "k3_json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char   *data;
    size_t  size;
    size_t  capacity;
} text_builder;

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error == NULL || error_size == 0u) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static bool append_text(text_builder *builder,
                        const char *text,
                        size_t size,
                        char *error,
                        size_t error_size) {
    if (builder->size > SIZE_MAX - size - 1u) {
        set_error(error, error_size,
                  "OpenAI text content size overflow");
        return false;
    }
    const size_t required = builder->size + size + 1u;
    if (required > builder->capacity) {
        size_t capacity =
            builder->capacity == 0u ? 256u :
                builder->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2u) {
                set_error(error, error_size,
                          "OpenAI text capacity overflow");
                return false;
            }
            capacity *= 2u;
        }
        char *data = (char *)realloc(
            builder->data, capacity);
        if (data == NULL) {
            set_error(error, error_size,
                      "allocating OpenAI text content failed");
            return false;
        }
        builder->data = data;
        builder->capacity = capacity;
    }
    if (size != 0u) {
        memcpy(builder->data + builder->size, text, size);
    }
    builder->size += size;
    builder->data[builder->size] = '\0';
    return true;
}

static bool token_type(const k3_json_document *document,
                       int32_t token,
                       k3_json_type type) {
    return token >= 0 &&
           (size_t)token < document->token_count &&
           document->tokens[token].type == type;
}

static bool decode_content_parts(
        const k3_json_document *document,
        int32_t content,
        char **decoded,
        char *error,
        size_t error_size) {
    *decoded = NULL;
    if (token_type(document, content, K3_JSON_STRING)) {
        return k3_json_string_dup(
            document, content, decoded,
            error, error_size);
    }
    if (!token_type(document, content, K3_JSON_ARRAY)) {
        set_error(error, error_size,
                  "message content must be a string or text-part array");
        return false;
    }
    text_builder builder = { 0 };
    const size_t part_count =
        document->tokens[content].size;
    for (size_t i = 0u; i < part_count; i++) {
        const int32_t part =
            k3_json_array_get(document, content, i);
        if (!token_type(document, part, K3_JSON_OBJECT)) {
            set_error(error, error_size,
                      "message content part must be an object");
            free(builder.data);
            return false;
        }
        const int32_t type =
            k3_json_object_get(document, part, "type");
        if (!k3_json_string_equal(document, type, "text") &&
            !k3_json_string_equal(document, type, "input_text")) {
            set_error(error, error_size,
                      "only text content parts are supported");
            free(builder.data);
            return false;
        }
        const int32_t text =
            k3_json_object_get(document, part, "text");
        char *piece = NULL;
        if (!k3_json_string_dup(
                document, text, &piece,
                error, error_size) ||
            !append_text(
                &builder, piece, piece == NULL ?
                    0u : strlen(piece),
                error, error_size)) {
            free(piece);
            free(builder.data);
            return false;
        }
        free(piece);
    }
    if (builder.data == NULL) {
        builder.data = strdup("");
        if (builder.data == NULL) {
            set_error(error, error_size,
                      "allocating empty message content failed");
            return false;
        }
    }
    *decoded = builder.data;
    return true;
}

static bool parse_message(
        const k3_json_document *document,
        int32_t token,
        k3_chat_message *message,
        char *error,
        size_t error_size) {
    if (!token_type(document, token, K3_JSON_OBJECT)) {
        set_error(error, error_size,
                  "each message must be an object");
        return false;
    }
    char *role = NULL;
    if (!k3_json_string_dup(
            document,
            k3_json_object_get(document, token, "role"),
            &role, error, error_size)) {
        return false;
    }
    if (strcmp(role, "system") == 0 ||
        strcmp(role, "developer") == 0) {
        message->role = K3_CHAT_ROLE_SYSTEM;
    } else if (strcmp(role, "user") == 0) {
        message->role = K3_CHAT_ROLE_USER;
    } else if (strcmp(role, "assistant") == 0) {
        message->role = K3_CHAT_ROLE_ASSISTANT;
    } else {
        set_error(error, error_size,
                  "unsupported message role \"%s\"", role);
        free(role);
        return false;
    }
    free(role);

    const int32_t content =
        k3_json_object_get(document, token, "content");
    if (token_type(document, content, K3_JSON_NULL) &&
        message->role == K3_CHAT_ROLE_ASSISTANT) {
        message->content = strdup("");
        if (message->content == NULL) {
            set_error(error, error_size,
                      "allocating assistant content failed");
            return false;
        }
    } else {
        char *decoded = NULL;
        if (!decode_content_parts(
                document, content, &decoded,
                error, error_size)) {
            return false;
        }
        message->content = decoded;
    }

    const int32_t name =
        k3_json_object_get(document, token, "name");
    if (name >= 0) {
        char *decoded_name = NULL;
        if (!k3_json_string_dup(
                document, name, &decoded_name,
                error, error_size)) {
            return false;
        }
        message->name = decoded_name;
    }
    const int32_t reasoning =
        k3_json_object_get(
            document, token, "reasoning_content");
    if (reasoning >= 0 &&
        !token_type(document, reasoning, K3_JSON_NULL)) {
        char *decoded_reasoning = NULL;
        if (!k3_json_string_dup(
                document, reasoning, &decoded_reasoning,
                error, error_size)) {
            return false;
        }
        message->reasoning_content = decoded_reasoning;
    }
    const int32_t tool_calls =
        k3_json_object_get(document, token, "tool_calls");
    if (tool_calls >= 0 &&
        !token_type(document, tool_calls, K3_JSON_NULL) &&
        (!token_type(document, tool_calls, K3_JSON_ARRAY) ||
         document->tokens[tool_calls].size != 0u)) {
        set_error(error, error_size,
                  "assistant tool_calls are not implemented yet");
        return false;
    }
    return true;
}

void k3_openai_chat_request_free(
        k3_openai_chat_request *request) {
    if (request == NULL) {
        return;
    }
    for (size_t i = 0u; i < request->message_count; i++) {
        free((char *)request->messages[i].content);
        free((char *)request->messages[i].name);
        free((char *)request->messages[i].reasoning_content);
    }
    free(request->messages);
    free(request->model);
    memset(request, 0, sizeof(*request));
}

bool k3_openai_parse_chat_request(
        const char *json,
        size_t json_size,
        k3_openai_chat_request *request,
        char *error,
        size_t error_size) {
    if (json == NULL || request == NULL) {
        set_error(error, error_size,
                  "OpenAI request arguments are invalid");
        return false;
    }
    memset(request, 0, sizeof(*request));
    request->max_tokens = 256u;
    k3_json_document document;
    if (!k3_json_parse(
            &document, json, json_size,
            error, error_size)) {
        return false;
    }
    bool ok = false;
    if (!token_type(
            &document, document.root, K3_JSON_OBJECT)) {
        set_error(error, error_size,
                  "request root must be an object");
        goto cleanup;
    }
    if (!k3_json_string_dup(
            &document,
            k3_json_object_get(
                &document, document.root, "model"),
            &request->model, error, error_size)) {
        goto cleanup;
    }

    const int32_t messages =
        k3_json_object_get(
            &document, document.root, "messages");
    if (!token_type(&document, messages, K3_JSON_ARRAY) ||
        document.tokens[messages].size == 0u ||
        document.tokens[messages].size >
            SIZE_MAX / sizeof(*request->messages)) {
        set_error(error, error_size,
                  "messages must be a non-empty array");
        goto cleanup;
    }
    request->message_count =
        document.tokens[messages].size;
    request->messages = (k3_chat_message *)calloc(
        request->message_count, sizeof(*request->messages));
    if (request->messages == NULL) {
        set_error(error, error_size,
                  "allocating request messages failed");
        goto cleanup;
    }
    for (size_t i = 0u; i < request->message_count; i++) {
        if (!parse_message(
                &document,
                k3_json_array_get(&document, messages, i),
                &request->messages[i],
                error, error_size)) {
            goto cleanup;
        }
    }
    if (request->messages[request->message_count - 1u].role !=
        K3_CHAT_ROLE_USER) {
        set_error(error, error_size,
                  "the final message must have role user");
        goto cleanup;
    }

    int32_t max_tokens =
        k3_json_object_get(
            &document, document.root,
            "max_completion_tokens");
    if (max_tokens < 0) {
        max_tokens = k3_json_object_get(
            &document, document.root, "max_tokens");
    }
    if (max_tokens >= 0 &&
        (!k3_json_u32(
            &document, max_tokens,
            &request->max_tokens) ||
         request->max_tokens == 0u ||
         request->max_tokens > 8192u)) {
        set_error(error, error_size,
                  "max tokens must be an integer in [1,8192]");
        goto cleanup;
    }
    const int32_t stream =
        k3_json_object_get(
            &document, document.root, "stream");
    if (stream >= 0 &&
        !k3_json_bool(
            &document, stream, &request->stream)) {
        set_error(error, error_size,
                  "stream must be boolean");
        goto cleanup;
    }
    const int32_t n =
        k3_json_object_get(
            &document, document.root, "n");
    uint32_t n_value = 1u;
    if (n >= 0 &&
        (!k3_json_u32(&document, n, &n_value) ||
         n_value != 1u)) {
        set_error(error, error_size,
                  "only n=1 is supported");
        goto cleanup;
    }
    const int32_t tools =
        k3_json_object_get(
            &document, document.root, "tools");
    if (tools >= 0 &&
        !token_type(&document, tools, K3_JSON_NULL) &&
        (!token_type(&document, tools, K3_JSON_ARRAY) ||
         document.tokens[tools].size != 0u)) {
        set_error(error, error_size,
                  "tools are not implemented yet");
        goto cleanup;
    }
    const int32_t response_format =
        k3_json_object_get(
            &document, document.root, "response_format");
    if (response_format >= 0 &&
        !token_type(
            &document, response_format, K3_JSON_NULL)) {
        set_error(error, error_size,
                  "response_format is not implemented yet");
        goto cleanup;
    }
    ok = true;

cleanup:
    k3_json_document_free(&document);
    if (!ok) {
        k3_openai_chat_request_free(request);
    }
    return ok;
}

bool k3_openai_build_chat_response(
        const char *completion_id,
        time_t created,
        const char *model,
        const k3_chat_turn_result *result,
        char **json,
        size_t *json_size,
        char *error,
        size_t error_size) {
    if (completion_id == NULL || model == NULL ||
        result == NULL || json == NULL) {
        set_error(error, error_size,
                  "OpenAI response arguments are invalid");
        return false;
    }
    *json = NULL;
    if (json_size != NULL) {
        *json_size = 0u;
    }
    char *escaped_id = NULL;
    char *escaped_model = NULL;
    char *escaped_content = NULL;
    bool ok =
        k3_json_escape(
            completion_id, strlen(completion_id),
            &escaped_id, NULL, error, error_size) &&
        k3_json_escape(
            model, strlen(model),
            &escaped_model, NULL, error, error_size) &&
        k3_json_escape(
            result->response.data == NULL ? "" :
                result->response.data,
            result->response.size,
            &escaped_content, NULL, error, error_size);
    if (!ok) {
        goto cleanup;
    }
    const char *finish =
        result->finish_reason ==
            K3_CHAT_FINISH_END_OF_MESSAGE ?
            "stop" : "length";
    const int required = snprintf(
        NULL, 0,
        "{\"id\":%s,\"object\":\"chat.completion\","
        "\"created\":%lld,\"model\":%s,"
        "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
        "\"choices\":[{\"index\":0,\"message\":{"
        "\"role\":\"assistant\",\"content\":%s},"
        "\"finish_reason\":\"%s\"}],"
        "\"usage\":{\"prompt_tokens\":%u,"
        "\"completion_tokens\":%u,\"total_tokens\":%u}}",
        escaped_id, (long long)created, escaped_model,
        escaped_content, finish,
        result->prompt_tokens, result->generated_tokens,
        result->prompt_tokens + result->generated_tokens);
    if (required < 0) {
        set_error(error, error_size,
                  "formatting OpenAI response failed");
        goto cleanup;
    }
    char *output = (char *)malloc((size_t)required + 1u);
    if (output == NULL) {
        set_error(error, error_size,
                  "allocating OpenAI response failed");
        goto cleanup;
    }
    snprintf(
        output, (size_t)required + 1u,
        "{\"id\":%s,\"object\":\"chat.completion\","
        "\"created\":%lld,\"model\":%s,"
        "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
        "\"choices\":[{\"index\":0,\"message\":{"
        "\"role\":\"assistant\",\"content\":%s},"
        "\"finish_reason\":\"%s\"}],"
        "\"usage\":{\"prompt_tokens\":%u,"
        "\"completion_tokens\":%u,\"total_tokens\":%u}}",
        escaped_id, (long long)created, escaped_model,
        escaped_content, finish,
        result->prompt_tokens, result->generated_tokens,
        result->prompt_tokens + result->generated_tokens);
    *json = output;
    if (json_size != NULL) {
        *json_size = (size_t)required;
    }
    ok = true;

cleanup:
    free(escaped_content);
    free(escaped_model);
    free(escaped_id);
    return ok && *json != NULL;
}
