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
    if (builder == NULL || (text == NULL && size != 0u)) {
        set_error(error, error_size,
                  "OpenAI text append arguments are invalid");
        return false;
    }
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

static bool schema_keyword_allowed(
        const k3_json_document *document, int32_t key) {
    static const char *allowed[] = {
        "type", "properties", "required", "additionalProperties",
        "items", "title", "description",
    };
    for (size_t i = 0u;
         i < sizeof(allowed) / sizeof(allowed[0]); i++) {
        if (k3_json_string_equal(document, key, allowed[i])) {
            return true;
        }
    }
    return false;
}

static bool json_object_keys_unique(
        const k3_json_document *document,
        int32_t object,
        const char *context,
        char *error,
        size_t error_size) {
    int32_t key = document->tokens[object].first_child;
    while (key >= 0) {
        const int32_t value = document->tokens[key].next_sibling;
        char *name = NULL;
        if (value < 0 || !k3_json_string_dup(
                document, key, &name, error, error_size)) {
            free(name);
            return false;
        }
        int32_t later = document->tokens[value].next_sibling;
        while (later >= 0) {
            const int32_t later_value =
                document->tokens[later].next_sibling;
            if (later_value < 0) {
                free(name);
                return false;
            }
            if (k3_json_string_equal(document, later, name)) {
                set_error(error, error_size,
                          "%s has duplicate property \"%s\"",
                          context, name);
                free(name);
                return false;
            }
            later = document->tokens[later_value].next_sibling;
        }
        free(name);
        key = document->tokens[value].next_sibling;
    }
    return true;
}

static bool validate_schema_definition(
        const k3_json_document *document,
        int32_t schema,
        unsigned depth,
        char *error,
        size_t error_size) {
    if (depth >= 64u || !token_type(
            document, schema, K3_JSON_OBJECT)) {
        set_error(error, error_size,
                  depth >= 64u ? "response schema exceeds 64 levels" :
                  "each response schema must be an object");
        return false;
    }
    if (!json_object_keys_unique(
            document, schema, "response schema",
            error, error_size)) {
        return false;
    }
    int32_t child = document->tokens[schema].first_child;
    while (child >= 0) {
        const int32_t value = document->tokens[child].next_sibling;
        if (value < 0 || !schema_keyword_allowed(document, child)) {
            char *keyword = NULL;
            (void)k3_json_string_dup(
                document, child, &keyword, NULL, 0u);
            set_error(error, error_size,
                      "unsupported response-schema keyword \"%s\"",
                      keyword == NULL ? "?" : keyword);
            free(keyword);
            return false;
        }
        child = document->tokens[value].next_sibling;
    }

    char *type = NULL;
    if (!k3_json_string_dup(
            document,
            k3_json_object_get(document, schema, "type"),
            &type, error, error_size)) {
        set_error(error, error_size,
                  "every response schema needs a string type");
        return false;
    }
    const bool known_type =
        strcmp(type, "object") == 0 ||
        strcmp(type, "array") == 0 ||
        strcmp(type, "string") == 0 ||
        strcmp(type, "number") == 0 ||
        strcmp(type, "integer") == 0 ||
        strcmp(type, "boolean") == 0 ||
        strcmp(type, "null") == 0;
    if (!known_type) {
        set_error(error, error_size,
                  "unsupported response-schema type \"%s\"", type);
        free(type);
        return false;
    }

    const int32_t title =
        k3_json_object_get(document, schema, "title");
    const int32_t description =
        k3_json_object_get(document, schema, "description");
    if ((title >= 0 && !token_type(
            document, title, K3_JSON_STRING)) ||
        (description >= 0 && !token_type(
            document, description, K3_JSON_STRING))) {
        set_error(error, error_size,
                  "response-schema title and description must be strings");
        free(type);
        return false;
    }

    const int32_t properties =
        k3_json_object_get(document, schema, "properties");
    const int32_t required =
        k3_json_object_get(document, schema, "required");
    const int32_t additional =
        k3_json_object_get(document, schema, "additionalProperties");
    const int32_t items =
        k3_json_object_get(document, schema, "items");
    bool ok = true;
    if (strcmp(type, "object") == 0) {
        if (properties >= 0 && !token_type(
                document, properties, K3_JSON_OBJECT)) {
            set_error(error, error_size,
                      "object response-schema properties must be an object");
            ok = false;
        }
        int32_t property = ok && properties >= 0 ?
            document->tokens[properties].first_child : -1;
        if (property >= 0 && !json_object_keys_unique(
                document, properties, "response-schema properties",
                error, error_size)) {
            ok = false;
        }
        while (ok && property >= 0) {
            const int32_t property_schema =
                document->tokens[property].next_sibling;
            ok = property_schema >= 0 && validate_schema_definition(
                document, property_schema, depth + 1u,
                error, error_size);
            if (ok) {
                property = document->tokens[property_schema].next_sibling;
            }
        }
        if (ok && required >= 0) {
            if (!token_type(document, required, K3_JSON_ARRAY)) {
                set_error(error, error_size,
                          "object response-schema required must be an array");
                ok = false;
            }
            for (size_t i = 0u; ok &&
                 i < document->tokens[required].size; i++) {
                const int32_t name =
                    k3_json_array_get(document, required, i);
                char *decoded = NULL;
                if (!k3_json_string_dup(
                        document, name, &decoded,
                        error, error_size) ||
                    properties < 0 || k3_json_object_get(
                        document, properties, decoded) < 0) {
                    set_error(error, error_size,
                              "required response-schema names must exist in properties");
                    ok = false;
                }
                for (size_t j = 0u; ok && j < i; j++) {
                    if (k3_json_string_equal(
                            document,
                            k3_json_array_get(document, required, j),
                            decoded)) {
                        set_error(error, error_size,
                                  "required response-schema names must be unique");
                        ok = false;
                    }
                }
                free(decoded);
            }
        }
        if (ok && additional >= 0) {
            bool ignored = false;
            if (!k3_json_bool(document, additional, &ignored)) {
                set_error(error, error_size,
                          "additionalProperties must be boolean");
                ok = false;
            }
        }
        if (ok && items >= 0) {
            set_error(error, error_size,
                      "items is only valid for array response schemas");
            ok = false;
        }
    } else if (strcmp(type, "array") == 0) {
        if (properties >= 0 || required >= 0 || additional >= 0) {
            set_error(error, error_size,
                      "object keywords require an object response schema");
            ok = false;
        } else if (items < 0) {
            set_error(error, error_size,
                      "array response schemas need items");
            ok = false;
        } else {
            ok = validate_schema_definition(
                document, items, depth + 1u, error, error_size);
        }
    } else if (properties >= 0 || required >= 0 ||
               additional >= 0 || items >= 0) {
        set_error(error, error_size,
                  "container keywords require object or array schemas");
        ok = false;
    }
    free(type);
    return ok;
}

static bool json_number_is_integer(
        const k3_json_document *document, int32_t token) {
    if (!token_type(document, token, K3_JSON_NUMBER)) {
        return false;
    }
    const k3_json_token selected = document->tokens[token];
    size_t fractional_digits = 0u;
    bool after_decimal = false;
    size_t coefficient_end = selected.end;
    size_t exponent_offset = selected.end;
    for (size_t i = selected.start; i < selected.end; i++) {
        const char c = document->source[i];
        if (c == '.') {
            after_decimal = true;
        } else if (c == 'e' || c == 'E') {
            coefficient_end = i;
            exponent_offset = i + 1u;
            break;
        } else if (after_decimal && c >= '0' && c <= '9') {
            fractional_digits++;
        }
    }
    bool negative_exponent = false;
    size_t exponent = 0u;
    if (exponent_offset < selected.end) {
        if (document->source[exponent_offset] == '+' ||
            document->source[exponent_offset] == '-') {
            negative_exponent =
                document->source[exponent_offset] == '-';
            exponent_offset++;
        }
        for (size_t i = exponent_offset; i < selected.end; i++) {
            const unsigned digit =
                (unsigned)(document->source[i] - '0');
            if (exponent > selected.end + fractional_digits) {
                break;
            }
            exponent = exponent * 10u + digit;
        }
    }
    if (!negative_exponent && exponent >= fractional_digits) {
        return true;
    }
    const size_t required_zeroes = negative_exponent ?
        fractional_digits + exponent : fractional_digits - exponent;
    size_t checked = 0u;
    for (size_t i = coefficient_end;
         i > selected.start && checked < required_zeroes;) {
        i--;
        const char c = document->source[i];
        if (c >= '0' && c <= '9') {
            if (c != '0') {
                return false;
            }
            checked++;
        }
    }
    if (checked >= required_zeroes) {
        return true;
    }
    for (size_t i = selected.start; i < selected.end; i++) {
        const char c = document->source[i];
        if (c == 'e' || c == 'E') {
            break;
        }
        if (c >= '1' && c <= '9') {
            return false;
        }
    }
    return true;
}

static bool validate_schema_instance(
        const k3_json_document *schema_document,
        int32_t schema,
        const k3_json_document *instance_document,
        int32_t instance,
        unsigned depth,
        char *error,
        size_t error_size) {
    if (depth >= 64u) {
        set_error(error, error_size,
                  "structured response exceeds 64 schema levels");
        return false;
    }
    char *type = NULL;
    if (!k3_json_string_dup(
            schema_document,
            k3_json_object_get(schema_document, schema, "type"),
            &type, error, error_size)) {
        return false;
    }
    const k3_json_type actual =
        instance_document->tokens[instance].type;
    const bool type_matches =
        (strcmp(type, "object") == 0 && actual == K3_JSON_OBJECT) ||
        (strcmp(type, "array") == 0 && actual == K3_JSON_ARRAY) ||
        (strcmp(type, "string") == 0 && actual == K3_JSON_STRING) ||
        (strcmp(type, "number") == 0 && actual == K3_JSON_NUMBER) ||
        (strcmp(type, "integer") == 0 &&
            json_number_is_integer(instance_document, instance)) ||
        (strcmp(type, "boolean") == 0 &&
            (actual == K3_JSON_TRUE || actual == K3_JSON_FALSE)) ||
        (strcmp(type, "null") == 0 && actual == K3_JSON_NULL);
    if (!type_matches) {
        set_error(error, error_size,
                  "structured response value does not match schema type %s",
                  type);
        free(type);
        return false;
    }

    bool ok = true;
    if (strcmp(type, "object") == 0) {
        if (!json_object_keys_unique(
                instance_document, instance, "structured response",
                error, error_size)) {
            free(type);
            return false;
        }
        const int32_t properties = k3_json_object_get(
            schema_document, schema, "properties");
        const int32_t required = k3_json_object_get(
            schema_document, schema, "required");
        bool allow_additional = true;
        const int32_t additional = k3_json_object_get(
            schema_document, schema, "additionalProperties");
        if (additional >= 0) {
            (void)k3_json_bool(
                schema_document, additional, &allow_additional);
        }
        if (required >= 0) {
            for (size_t i = 0u; ok &&
                 i < schema_document->tokens[required].size; i++) {
                char *name = NULL;
                ok = k3_json_string_dup(
                    schema_document,
                    k3_json_array_get(schema_document, required, i),
                    &name, error, error_size);
                if (ok && k3_json_object_get(
                        instance_document, instance, name) < 0) {
                    set_error(error, error_size,
                              "structured response is missing required property \"%s\"",
                              name);
                    ok = false;
                }
                free(name);
            }
        }
        int32_t key = ok ?
            instance_document->tokens[instance].first_child : -1;
        while (ok && key >= 0) {
            const int32_t value =
                instance_document->tokens[key].next_sibling;
            char *name = NULL;
            ok = value >= 0 && k3_json_string_dup(
                instance_document, key, &name,
                error, error_size);
            const int32_t property_schema = ok && properties >= 0 ?
                k3_json_object_get(
                    schema_document, properties, name) : -1;
            if (ok && property_schema < 0 && !allow_additional) {
                set_error(error, error_size,
                          "structured response has additional property \"%s\"",
                          name);
                ok = false;
            } else if (ok && property_schema >= 0) {
                ok = validate_schema_instance(
                    schema_document, property_schema,
                    instance_document, value, depth + 1u,
                    error, error_size);
            }
            free(name);
            if (ok) {
                key = instance_document->tokens[value].next_sibling;
            }
        }
    } else if (strcmp(type, "array") == 0) {
        const int32_t items = k3_json_object_get(
            schema_document, schema, "items");
        for (size_t i = 0u; ok &&
             i < instance_document->tokens[instance].size; i++) {
            ok = validate_schema_instance(
                schema_document, items,
                instance_document,
                k3_json_array_get(instance_document, instance, i),
                depth + 1u, error, error_size);
        }
    }
    free(type);
    return ok;
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

static bool valid_function_name(const char *name) {
    const size_t size = name == NULL ? 0u : strlen(name);
    if (size == 0u || size > 64u) {
        return false;
    }
    for (size_t i = 0u; i < size; i++) {
        const unsigned char c = (unsigned char)name[i];
        if (!((c >= 'a' && c <= 'z') ||
              (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') ||
              c == '_' || c == '-')) {
            return false;
        }
    }
    return true;
}

static bool validate_tool_object(
        const k3_json_document *document,
        int32_t tool,
        char **name,
        char *error,
        size_t error_size) {
    if (name != NULL) {
        *name = NULL;
    }
    if (!token_type(document, tool, K3_JSON_OBJECT) ||
        !k3_json_string_equal(
            document,
            k3_json_object_get(document, tool, "type"),
            "function")) {
        set_error(error, error_size,
                  "each tool must be a function object");
        return false;
    }
    const int32_t function =
        k3_json_object_get(document, tool, "function");
    if (!token_type(document, function, K3_JSON_OBJECT)) {
        set_error(error, error_size,
                  "tool.function must be an object");
        return false;
    }
    char *decoded_name = NULL;
    if (!k3_json_string_dup(
            document,
            k3_json_object_get(document, function, "name"),
            &decoded_name, error, error_size)) {
        return false;
    }
    if (!valid_function_name(decoded_name)) {
        set_error(error, error_size,
                  "tool function names must be 1-64 letters, digits, '_' or '-'");
        free(decoded_name);
        return false;
    }
    const int32_t description =
        k3_json_object_get(document, function, "description");
    if (description >= 0 &&
        !token_type(document, description, K3_JSON_STRING)) {
        set_error(error, error_size,
                  "tool function description must be a string");
        free(decoded_name);
        return false;
    }
    const int32_t parameters =
        k3_json_object_get(document, function, "parameters");
    if (parameters < 0 ||
        !token_type(document, parameters, K3_JSON_OBJECT)) {
        set_error(error, error_size,
                  "tool function parameters must be a JSON Schema object");
        free(decoded_name);
        return false;
    }
    const int32_t strict =
        k3_json_object_get(document, function, "strict");
    if (strict >= 0 &&
        !token_type(document, strict, K3_JSON_TRUE) &&
        !token_type(document, strict, K3_JSON_FALSE)) {
        set_error(error, error_size,
                  "tool function strict must be boolean");
        free(decoded_name);
        return false;
    }
    if (name != NULL) {
        *name = decoded_name;
    } else {
        free(decoded_name);
    }
    return true;
}

static bool parse_tools(
        const k3_json_document *document,
        int32_t tools,
        char **tools_json,
        size_t *tool_count,
        char *error,
        size_t error_size) {
    *tools_json = NULL;
    *tool_count = 0u;
    if (tools < 0 || token_type(document, tools, K3_JSON_NULL)) {
        return true;
    }
    if (!token_type(document, tools, K3_JSON_ARRAY)) {
        set_error(error, error_size,
                  "tools must be an array");
        return false;
    }
    const size_t count = document->tokens[tools].size;
    for (size_t i = 0u; i < count; i++) {
        char *name = NULL;
        if (!validate_tool_object(
                document,
                k3_json_array_get(document, tools, i),
                &name, error, error_size)) {
            return false;
        }
        for (size_t j = 0u; j < i; j++) {
            char *earlier = NULL;
            if (!validate_tool_object(
                    document,
                    k3_json_array_get(document, tools, j),
                    &earlier, error, error_size)) {
                free(name);
                return false;
            }
            const bool duplicate = strcmp(name, earlier) == 0;
            free(earlier);
            if (duplicate) {
                set_error(error, error_size,
                          "tool function names must be unique");
                free(name);
                return false;
            }
        }
        free(name);
    }
    if (count != 0u &&
        !k3_json_compact_sorted_dup(
            document, tools, tools_json, NULL,
            error, error_size)) {
        return false;
    }
    *tool_count = count;
    return true;
}

static bool compact_arguments(
        const char *arguments,
        char **compacted,
        char *error,
        size_t error_size) {
    k3_json_document document;
    char parse_error[256];
    if (!k3_json_parse(
            &document, arguments, strlen(arguments),
            parse_error, sizeof(parse_error))) {
        *compacted = strdup(arguments);
        if (*compacted == NULL) {
            set_error(error, error_size,
                      "allocating raw tool arguments failed");
            return false;
        }
        return true;
    }
    bool ok = false;
    if (!token_type(&document, document.root, K3_JSON_OBJECT)) {
        set_error(error, error_size,
                  "tool call arguments must encode a JSON object");
    } else {
        ok = k3_json_compact_sorted_dup(
            &document, document.root, compacted, NULL,
            error, error_size);
    }
    k3_json_document_free(&document);
    return ok;
}

static bool parse_assistant_tool_calls(
        const k3_json_document *document,
        int32_t calls,
        k3_chat_message *message,
        char *error,
        size_t error_size) {
    if (calls < 0 || token_type(document, calls, K3_JSON_NULL)) {
        return true;
    }
    if (!token_type(document, calls, K3_JSON_ARRAY)) {
        set_error(error, error_size,
                  "assistant tool_calls must be an array");
        return false;
    }
    const size_t count = document->tokens[calls].size;
    if (count == 0u) {
        return true;
    }
    if (count > SIZE_MAX / sizeof(*message->tool_calls)) {
        set_error(error, error_size,
                  "assistant tool call array is too large");
        return false;
    }
    message->tool_calls = (k3_tool_call *)calloc(
        count, sizeof(*message->tool_calls));
    if (message->tool_calls == NULL) {
        set_error(error, error_size,
                  "allocating assistant tool calls failed");
        return false;
    }
    message->tool_call_count = count;
    for (size_t i = 0u; i < count; i++) {
        const int32_t call =
            k3_json_array_get(document, calls, i);
        if (!token_type(document, call, K3_JSON_OBJECT) ||
            !k3_json_string_equal(
                document,
                k3_json_object_get(document, call, "type"),
                "function")) {
            set_error(error, error_size,
                      "assistant tool_calls must contain function calls");
            return false;
        }
        char *id = NULL;
        if (!k3_json_string_dup(
                document,
                k3_json_object_get(document, call, "id"),
                &id, error, error_size) || id[0] == '\0') {
            free(id);
            set_error(error, error_size,
                      "assistant tool calls need non-empty IDs");
            return false;
        }
        const int32_t function =
            k3_json_object_get(document, call, "function");
        char *name = NULL;
        char *arguments = NULL;
        char *compacted = NULL;
        if (!token_type(document, function, K3_JSON_OBJECT) ||
            !k3_json_string_dup(
                document,
                k3_json_object_get(document, function, "name"),
                &name, error, error_size) ||
            !valid_function_name(name) ||
            !k3_json_string_dup(
                document,
                k3_json_object_get(document, function, "arguments"),
                &arguments, error, error_size) ||
            !compact_arguments(
                arguments, &compacted,
                error, error_size)) {
            free(compacted);
            free(arguments);
            free(name);
            free(id);
            if (error != NULL && error_size != 0u &&
                error[0] == '\0') {
                set_error(error, error_size,
                          "invalid assistant function call");
            }
            return false;
        }
        free(arguments);
        for (size_t j = 0u; j < i; j++) {
            if (strcmp(id, message->tool_calls[j].id) == 0) {
                set_error(error, error_size,
                          "assistant tool call IDs must be unique");
                free(compacted);
                free(name);
                free(id);
                return false;
            }
        }
        message->tool_calls[i] = (k3_tool_call) {
            .id = id,
            .name = name,
            .arguments = compacted,
        };
    }
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
    } else if (strcmp(role, "tool") == 0) {
        message->role = K3_CHAT_ROLE_TOOL;
    } else {
        set_error(error, error_size,
                  "unsupported message role \"%s\"", role);
        free(role);
        return false;
    }
    free(role);

    const int32_t content =
        k3_json_object_get(document, token, "content");
    const int32_t dynamic_tools =
        k3_json_object_get(document, token, "tools");
    if (dynamic_tools >= 0 &&
        message->role != K3_CHAT_ROLE_SYSTEM) {
        set_error(error, error_size,
                  "message tools are only valid on system messages");
        return false;
    }
    if (message->role == K3_CHAT_ROLE_SYSTEM &&
        dynamic_tools >= 0) {
        size_t dynamic_tool_count = 0u;
        char *tools_json = NULL;
        if (!parse_tools(
                document, dynamic_tools,
                &tools_json, &dynamic_tool_count,
                error, error_size) ||
            dynamic_tool_count == 0u) {
            free(tools_json);
            if (dynamic_tool_count == 0u) {
                set_error(error, error_size,
                          "dynamic system tools must be non-empty");
            }
            return false;
        }
        message->tools_json = tools_json;
        if (content >= 0 &&
            !token_type(document, content, K3_JSON_NULL)) {
            set_error(error, error_size,
                      "dynamic tool system messages must omit content");
            return false;
        }
    }
    if ((content < 0 || token_type(document, content, K3_JSON_NULL)) &&
        (message->role == K3_CHAT_ROLE_ASSISTANT ||
         (message->role == K3_CHAT_ROLE_SYSTEM &&
          message->tools_json != NULL))) {
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
    int32_t reasoning = k3_json_object_get(
        document, token, "reasoning_content");
    if (reasoning < 0) {
        reasoning = k3_json_object_get(
            document, token, "reasoning");
    }
    if (reasoning >= 0 &&
        !token_type(document, reasoning, K3_JSON_NULL)) {
        if (message->role != K3_CHAT_ROLE_ASSISTANT) {
            set_error(error, error_size,
                      "reasoning_content is only valid on assistant messages");
            return false;
        }
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
    if (message->role != K3_CHAT_ROLE_ASSISTANT &&
        tool_calls >= 0 &&
        !token_type(document, tool_calls, K3_JSON_NULL)) {
        set_error(error, error_size,
                  "tool_calls are only valid on assistant messages");
        return false;
    }
    if (message->role == K3_CHAT_ROLE_ASSISTANT &&
        !parse_assistant_tool_calls(
            document, tool_calls, message,
            error, error_size)) {
        return false;
    }
    const int32_t tool_call_id =
        k3_json_object_get(document, token, "tool_call_id");
    if (message->role == K3_CHAT_ROLE_TOOL) {
        char *decoded_id = NULL;
        if (!k3_json_string_dup(
                document, tool_call_id, &decoded_id,
                error, error_size) || decoded_id[0] == '\0') {
            free(decoded_id);
            set_error(error, error_size,
                      "tool messages need a non-empty tool_call_id");
            return false;
        }
        message->tool_call_id = decoded_id;
    } else if (tool_call_id >= 0) {
        set_error(error, error_size,
                  "tool_call_id is only valid on tool messages");
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
        free((char *)request->messages[i].tool_call_id);
        free((char *)request->messages[i].tools_json);
        for (size_t j = 0u;
             j < request->messages[i].tool_call_count;
             j++) {
            free((char *)request->messages[i].tool_calls[j].id);
            free((char *)request->messages[i].tool_calls[j].name);
            free((char *)request->messages[i].tool_calls[j].arguments);
        }
        free(request->messages[i].tool_calls);
    }
    free(request->messages);
    free(request->reasoning_effort);
    free(request->tools_json);
    free(request->response_schema_json);
    free(request->model);
    memset(request, 0, sizeof(*request));
}

static bool validate_and_order_tool_history(
        k3_openai_chat_request *request,
        char *error,
        size_t error_size) {
    for (size_t i = 0u; i < request->message_count; i++) {
        k3_chat_message *assistant = &request->messages[i];
        if (assistant->role == K3_CHAT_ROLE_TOOL) {
            set_error(error, error_size,
                      "tool messages must immediately follow assistant tool_calls");
            return false;
        }
        if (assistant->role != K3_CHAT_ROLE_ASSISTANT ||
            assistant->tool_call_count == 0u) {
            continue;
        }
        const size_t count = assistant->tool_call_count;
        if (i + count >= request->message_count) {
            set_error(error, error_size,
                      "every assistant tool call needs a matching tool result");
            return false;
        }
        k3_chat_message *ordered = (k3_chat_message *)calloc(
            count, sizeof(*ordered));
        bool *matched = (bool *)calloc(count, sizeof(*matched));
        if (ordered == NULL || matched == NULL) {
            free(matched);
            free(ordered);
            set_error(error, error_size,
                      "allocating tool result ordering failed");
            return false;
        }
        bool ok = true;
        for (size_t j = 0u; j < count; j++) {
            k3_chat_message *tool = &request->messages[i + 1u + j];
            if (tool->role != K3_CHAT_ROLE_TOOL) {
                set_error(error, error_size,
                          "tool results must be consecutive and complete");
                ok = false;
                break;
            }
            size_t call_index = count;
            for (size_t k = 0u; k < count; k++) {
                if (strcmp(
                        tool->tool_call_id,
                        assistant->tool_calls[k].id) == 0) {
                    call_index = k;
                    break;
                }
            }
            if (call_index == count || matched[call_index]) {
                set_error(error, error_size,
                          "tool_call_id does not uniquely match the preceding assistant");
                ok = false;
                break;
            }
            matched[call_index] = true;
            free((char *)tool->name);
            tool->name = strdup(
                assistant->tool_calls[call_index].name);
            if (tool->name == NULL) {
                set_error(error, error_size,
                          "allocating resolved tool name failed");
                ok = false;
                break;
            }
            ordered[call_index] = *tool;
        }
        if (ok) {
            for (size_t j = 0u; j < count; j++) {
                request->messages[i + 1u + j] = ordered[j];
            }
        }
        free(matched);
        free(ordered);
        if (!ok) {
            return false;
        }
        i += count;
    }
    return true;
}

static int32_t find_tool_by_name(
        const k3_json_document *document,
        int32_t tools,
        const char *wanted) {
    if (!token_type(document, tools, K3_JSON_ARRAY)) {
        return -1;
    }
    for (size_t i = 0u; i < document->tokens[tools].size; i++) {
        const int32_t tool =
            k3_json_array_get(document, tools, i);
        const int32_t function =
            k3_json_object_get(document, tool, "function");
        if (k3_json_string_equal(
                document,
                k3_json_object_get(document, function, "name"),
                wanted)) {
            return tool;
        }
    }
    return -1;
}

static bool filter_specific_tool(
        const k3_json_document *document,
        int32_t tools,
        const char *name,
        char **tools_json,
        char *error,
        size_t error_size) {
    const int32_t tool = find_tool_by_name(document, tools, name);
    if (tool < 0) {
        set_error(error, error_size,
                  "tool_choice names a function absent from tools");
        return false;
    }
    char *object = NULL;
    size_t object_size = 0u;
    if (!k3_json_compact_sorted_dup(
            document, tool, &object, &object_size,
            error, error_size)) {
        return false;
    }
    if (object_size > SIZE_MAX - 3u) {
        free(object);
        set_error(error, error_size,
                  "selected tool definition is too large");
        return false;
    }
    char *array = (char *)malloc(object_size + 3u);
    if (array == NULL) {
        free(object);
        set_error(error, error_size,
                  "allocating selected tool definition failed");
        return false;
    }
    array[0] = '[';
    memcpy(array + 1u, object, object_size);
    array[object_size + 1u] = ']';
    array[object_size + 2u] = '\0';
    free(object);
    free(*tools_json);
    *tools_json = array;
    return true;
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
    request->thinking = true;
    request->reasoning_effort = strdup("max");
    if (request->reasoning_effort == NULL) {
        set_error(error, error_size,
                  "allocating default reasoning effort failed");
        return false;
    }
    request->tool_choice = K3_TOOL_CHOICE_AUTO;
    request->parallel_tool_calls = true;
    k3_json_document document;
    if (!k3_json_parse(
            &document, json, json_size,
            error, error_size)) {
        k3_openai_chat_request_free(request);
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
    if (!validate_and_order_tool_history(
            request, error, error_size)) {
        goto cleanup;
    }
    if (request->messages[request->message_count - 1u].role ==
        K3_CHAT_ROLE_ASSISTANT) {
        set_error(error, error_size,
                  "the final message must request a new assistant generation");
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
    const int32_t reasoning_effort =
        k3_json_object_get(
            &document, document.root, "reasoning_effort");
    if (reasoning_effort >= 0 &&
        !token_type(&document, reasoning_effort, K3_JSON_NULL)) {
        char *decoded_effort = NULL;
        if (!k3_json_string_dup(
                &document, reasoning_effort,
                &decoded_effort, error, error_size) ||
            (strcmp(decoded_effort, "low") != 0 &&
             strcmp(decoded_effort, "high") != 0 &&
             strcmp(decoded_effort, "max") != 0)) {
            free(decoded_effort);
            set_error(error, error_size,
                      "reasoning_effort must be low, high, or max");
            goto cleanup;
        }
        free(request->reasoning_effort);
        request->reasoning_effort = decoded_effort;
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
    if (!parse_tools(
            &document, tools,
            &request->tools_json, &request->tool_count,
            error, error_size)) {
        goto cleanup;
    }
    const int32_t tool_choice =
        k3_json_object_get(
            &document, document.root, "tool_choice");
    if (tool_choice >= 0 &&
        !token_type(&document, tool_choice, K3_JSON_NULL)) {
        if (token_type(&document, tool_choice, K3_JSON_STRING)) {
            if (k3_json_string_equal(
                    &document, tool_choice, "auto")) {
                request->tool_choice = K3_TOOL_CHOICE_AUTO;
            } else if (k3_json_string_equal(
                           &document, tool_choice, "required")) {
                request->tool_choice = K3_TOOL_CHOICE_REQUIRED;
            } else if (k3_json_string_equal(
                           &document, tool_choice, "none")) {
                request->tool_choice = K3_TOOL_CHOICE_NONE;
            } else {
                set_error(error, error_size,
                          "tool_choice must be auto, required, none, or a function object");
                goto cleanup;
            }
        } else if (token_type(
                       &document, tool_choice, K3_JSON_OBJECT) &&
                   k3_json_string_equal(
                       &document,
                       k3_json_object_get(
                           &document, tool_choice, "type"),
                       "function")) {
            const int32_t function = k3_json_object_get(
                &document, tool_choice, "function");
            char *name = NULL;
            if (!token_type(
                    &document, function, K3_JSON_OBJECT) ||
                !k3_json_string_dup(
                    &document,
                    k3_json_object_get(
                        &document, function, "name"),
                    &name, error, error_size) ||
                !valid_function_name(name) ||
                !filter_specific_tool(
                    &document, tools, name,
                    &request->tools_json,
                    error, error_size)) {
                free(name);
                goto cleanup;
            }
            free(name);
            request->tool_choice = K3_TOOL_CHOICE_REQUIRED;
            request->tool_count = 1u;
        } else {
            set_error(error, error_size,
                      "tool_choice must be auto, required, none, or a function object");
            goto cleanup;
        }
    }
    if (request->tool_choice == K3_TOOL_CHOICE_REQUIRED &&
        request->tool_count == 0u) {
        set_error(error, error_size,
                  "tool_choice=required needs at least one tool");
        goto cleanup;
    }
    const int32_t parallel =
        k3_json_object_get(
            &document, document.root, "parallel_tool_calls");
    if (parallel >= 0 &&
        !k3_json_bool(
            &document, parallel,
            &request->parallel_tool_calls)) {
        set_error(error, error_size,
                  "parallel_tool_calls must be boolean");
        goto cleanup;
    }
    const int32_t response_format =
        k3_json_object_get(
            &document, document.root, "response_format");
    if (response_format >= 0 &&
        !token_type(
            &document, response_format, K3_JSON_NULL)) {
        if (!token_type(
                &document, response_format, K3_JSON_OBJECT)) {
            set_error(error, error_size,
                      "response_format must be an object");
            goto cleanup;
        }
        const int32_t format_type = k3_json_object_get(
            &document, response_format, "type");
        if (k3_json_string_equal(
                &document, format_type, "text")) {
            request->response_format = K3_RESPONSE_FORMAT_TEXT;
        } else if (k3_json_string_equal(
                       &document, format_type, "json_object")) {
            request->response_format =
                K3_RESPONSE_FORMAT_JSON_OBJECT;
        } else if (k3_json_string_equal(
                       &document, format_type, "json_schema")) {
            const int32_t wrapper = k3_json_object_get(
                &document, response_format, "json_schema");
            if (!token_type(
                    &document, wrapper, K3_JSON_OBJECT)) {
                set_error(error, error_size,
                          "response_format.json_schema must be an object");
                goto cleanup;
            }
            char *name = NULL;
            if (!k3_json_string_dup(
                    &document,
                    k3_json_object_get(&document, wrapper, "name"),
                    &name, error, error_size) ||
                !valid_function_name(name)) {
                free(name);
                set_error(error, error_size,
                          "response schema names must be 1-64 letters, digits, '_' or '-'");
                goto cleanup;
            }
            free(name);
            const int32_t description = k3_json_object_get(
                &document, wrapper, "description");
            if (description >= 0 && !token_type(
                    &document, description, K3_JSON_STRING)) {
                set_error(error, error_size,
                          "response schema description must be a string");
                goto cleanup;
            }
            const int32_t strict = k3_json_object_get(
                &document, wrapper, "strict");
            bool strict_value = false;
            if (strict >= 0 && !k3_json_bool(
                    &document, strict, &strict_value)) {
                set_error(error, error_size,
                          "response schema strict must be boolean");
                goto cleanup;
            }
            const int32_t schema = k3_json_object_get(
                &document, wrapper, "schema");
            if (!validate_schema_definition(
                    &document, schema, 0u,
                    error, error_size) ||
                !k3_json_compact_sorted_dup(
                    &document, schema,
                    &request->response_schema_json, NULL,
                    error, error_size)) {
                goto cleanup;
            }
            request->response_format =
                K3_RESPONSE_FORMAT_JSON_SCHEMA;
        } else {
            set_error(error, error_size,
                      "response_format.type must be text, json_object, or json_schema");
            goto cleanup;
        }
    }
    ok = true;

cleanup:
    k3_json_document_free(&document);
    if (!ok) {
        k3_openai_chat_request_free(request);
    }
    return ok;
}

bool k3_openai_validate_tool_policy(
        const k3_openai_chat_request *request,
        const k3_chat_turn_result *result,
        char *error,
        size_t error_size) {
    if (request == NULL || result == NULL) {
        set_error(error, error_size,
                  "tool-policy validation arguments are invalid");
        return false;
    }
    if (request->tool_choice == K3_TOOL_CHOICE_NONE &&
        result->tool_call_count != 0u) {
        set_error(error, error_size,
                  "model violated tool_choice=none");
        return false;
    }
    if (request->tool_choice == K3_TOOL_CHOICE_REQUIRED &&
        result->tool_call_count == 0u) {
        set_error(error, error_size,
                  "model ended without satisfying tool_choice=required");
        return false;
    }
    if (!request->parallel_tool_calls &&
        result->tool_call_count > 1u) {
        set_error(error, error_size,
                  "model emitted multiple calls with parallel_tool_calls=false");
        return false;
    }
    return true;
}

bool k3_openai_validate_response_format(
        const k3_openai_chat_request *request,
        const k3_chat_turn_result *result,
        char *error,
        size_t error_size) {
    if (request == NULL || result == NULL) {
        set_error(error, error_size,
                  "response-format validation arguments are invalid");
        return false;
    }
    if (request->response_format == K3_RESPONSE_FORMAT_TEXT ||
        result->tool_call_count != 0u) {
        return true;
    }
    if (request->response_format != K3_RESPONSE_FORMAT_JSON_OBJECT &&
        request->response_format != K3_RESPONSE_FORMAT_JSON_SCHEMA) {
        set_error(error, error_size,
                  "unsupported response-format validator %d",
                  (int)request->response_format);
        return false;
    }
    k3_json_document document;
    char parse_error[256] = { 0 };
    const bool parsed = k3_json_parse(
        &document,
        result->response.data == NULL ? "" : result->response.data,
        result->response.size,
        parse_error, sizeof(parse_error));
    if (!parsed) {
        set_error(error, error_size,
                  "model response is not valid JSON for "
                  "response_format=%s: %s",
                  request->response_format == K3_RESPONSE_FORMAT_JSON_SCHEMA ?
                      "json_schema" : "json_object",
                  parse_error);
        return false;
    }
    if (request->response_format == K3_RESPONSE_FORMAT_JSON_OBJECT) {
        const bool object = document.root >= 0 &&
            document.tokens[document.root].type == K3_JSON_OBJECT;
        k3_json_document_free(&document);
        if (!object) {
            set_error(error, error_size,
                      "model response must be a JSON object for "
                      "response_format=json_object");
            return false;
        }
        return true;
    }

    k3_json_document schema_document;
    if (request->response_schema_json == NULL ||
        !k3_json_parse(
            &schema_document,
            request->response_schema_json == NULL ? "" :
                request->response_schema_json,
            request->response_schema_json == NULL ? 0u :
                strlen(request->response_schema_json),
            parse_error, sizeof(parse_error))) {
        k3_json_document_free(&document);
        set_error(error, error_size,
                  "stored response schema is invalid: %s",
                  request->response_schema_json == NULL ?
                      "schema is missing" : parse_error);
        return false;
    }
    const bool valid = validate_schema_definition(
            &schema_document, schema_document.root, 0u,
            error, error_size) &&
        validate_schema_instance(
            &schema_document, schema_document.root,
            &document, document.root, 0u,
            error, error_size);
    k3_json_document_free(&schema_document);
    k3_json_document_free(&document);
    return valid;
}

static bool build_tool_calls_fragment(
        const char *completion_id,
        const k3_chat_turn_result *result,
        char **fragment,
        char *error,
        size_t error_size) {
    *fragment = NULL;
    if (result->tool_call_count == 0u) {
        *fragment = strdup("");
        if (*fragment == NULL) {
            set_error(error, error_size,
                      "allocating empty tool call fragment failed");
            return false;
        }
        return true;
    }
    text_builder builder = { 0 };
    static const char prefix[] = ",\"tool_calls\":[";
    if (!append_text(
            &builder, prefix, sizeof(prefix) - 1u,
            error, error_size)) {
        return false;
    }
    for (size_t i = 0u; i < result->tool_call_count; i++) {
        const k3_tool_call *call = &result->tool_calls[i];
        char id[384];
        const int id_size = snprintf(
            id, sizeof(id), "call_%s_%zu", completion_id, i + 1u);
        char *escaped_id = NULL;
        char *escaped_name = NULL;
        char *escaped_arguments = NULL;
        bool ok = id_size > 0 && (size_t)id_size < sizeof(id) &&
            k3_json_escape(
                id, (size_t)id_size,
                &escaped_id, NULL,
                error, error_size) &&
            k3_json_escape(
                call->name, strlen(call->name),
                &escaped_name, NULL,
                error, error_size) &&
            k3_json_escape(
                call->arguments, strlen(call->arguments),
                &escaped_arguments, NULL,
                error, error_size);
        if (!ok) {
            free(escaped_arguments);
            free(escaped_name);
            free(escaped_id);
            free(builder.data);
            return false;
        }
        const int required = snprintf(
            NULL, 0,
            "%s{\"id\":%s,\"type\":\"function\","
            "\"function\":{\"name\":%s,\"arguments\":%s}}",
            i == 0u ? "" : ",",
            escaped_id, escaped_name, escaped_arguments);
        char *entry = required < 0 ? NULL :
            (char *)malloc((size_t)required + 1u);
        if (entry != NULL) {
            snprintf(
                entry, (size_t)required + 1u,
                "%s{\"id\":%s,\"type\":\"function\","
                "\"function\":{\"name\":%s,\"arguments\":%s}}",
                i == 0u ? "" : ",",
                escaped_id, escaped_name, escaped_arguments);
        }
        free(escaped_arguments);
        free(escaped_name);
        free(escaped_id);
        if (entry == NULL || !append_text(
                &builder, entry, (size_t)required,
                error, error_size)) {
            free(entry);
            free(builder.data);
            return false;
        }
        free(entry);
    }
    if (!append_text(&builder, "]", 1u, error, error_size)) {
        free(builder.data);
        return false;
    }
    *fragment = builder.data;
    return true;
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
    char *escaped_reasoning = NULL;
    char *reasoning_fragment = NULL;
    char *tool_calls_fragment = NULL;
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
            &escaped_content, NULL, error, error_size) &&
        (!result->thinking || k3_json_escape(
            result->reasoning_content.data == NULL ? "" :
                result->reasoning_content.data,
            result->reasoning_content.size,
            &escaped_reasoning, NULL, error, error_size)) &&
        build_tool_calls_fragment(
            completion_id, result, &tool_calls_fragment,
            error, error_size);
    if (!ok) {
        goto cleanup;
    }
    if (result->thinking) {
        const int reasoning_required = snprintf(
            NULL, 0, ",\"reasoning_content\":%s",
            escaped_reasoning);
        reasoning_fragment = reasoning_required < 0 ? NULL :
            (char *)malloc((size_t)reasoning_required + 1u);
        if (reasoning_fragment == NULL) {
            set_error(error, error_size,
                      "allocating reasoning response field failed");
            goto cleanup;
        }
        snprintf(
            reasoning_fragment, (size_t)reasoning_required + 1u,
            ",\"reasoning_content\":%s", escaped_reasoning);
    } else {
        reasoning_fragment = strdup("");
        if (reasoning_fragment == NULL) {
            set_error(error, error_size,
                      "allocating empty reasoning response field failed");
            goto cleanup;
        }
    }
    if (result->tool_call_count != 0u &&
        result->response.size == 0u) {
        free(escaped_content);
        escaped_content = strdup("null");
        if (escaped_content == NULL) {
            set_error(error, error_size,
                      "allocating null assistant content failed");
            goto cleanup;
        }
    }
    const char *finish = result->finish_reason ==
        K3_CHAT_FINISH_TOOL_CALLS ? "tool_calls" :
        (result->finish_reason ==
            K3_CHAT_FINISH_END_OF_MESSAGE ? "stop" : "length");
    const int required = snprintf(
        NULL, 0,
        "{\"id\":%s,\"object\":\"chat.completion\","
        "\"created\":%lld,\"model\":%s,"
        "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
        "\"choices\":[{\"index\":0,\"message\":{"
        "\"role\":\"assistant\",\"content\":%s%s%s},"
        "\"finish_reason\":\"%s\"}],"
        "\"usage\":{\"prompt_tokens\":%u,"
        "\"completion_tokens\":%u,\"total_tokens\":%u}}",
        escaped_id, (long long)created, escaped_model,
        escaped_content, reasoning_fragment, tool_calls_fragment, finish,
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
        "\"role\":\"assistant\",\"content\":%s%s%s},"
        "\"finish_reason\":\"%s\"}],"
        "\"usage\":{\"prompt_tokens\":%u,"
        "\"completion_tokens\":%u,\"total_tokens\":%u}}",
        escaped_id, (long long)created, escaped_model,
        escaped_content, reasoning_fragment, tool_calls_fragment, finish,
        result->prompt_tokens, result->generated_tokens,
        result->prompt_tokens + result->generated_tokens);
    *json = output;
    if (json_size != NULL) {
        *json_size = (size_t)required;
    }
    ok = true;

cleanup:
    free(reasoning_fragment);
    free(escaped_reasoning);
    free(tool_calls_fragment);
    free(escaped_content);
    free(escaped_model);
    free(escaped_id);
    return ok && *json != NULL;
}
