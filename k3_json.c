#include "k3_json.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    k3_json_document *document;
    size_t            position;
    unsigned          depth;
    char             *error;
    size_t            error_size;
} k3_json_parser;

typedef struct {
    char   *data;
    size_t  size;
    size_t  capacity;
} k3_json_builder;

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error == NULL || error_size == 0u) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static void parser_error(k3_json_parser *parser, const char *message) {
    set_error(parser->error, parser->error_size,
              "JSON offset %zu: %s", parser->position, message);
}

static void skip_space(k3_json_parser *parser) {
    while (parser->position < parser->document->source_size) {
        const char c =
            parser->document->source[parser->position];
        if (c != ' ' && c != '\t' &&
            c != '\r' && c != '\n') {
            break;
        }
        parser->position++;
    }
}

static bool reserve_tokens(k3_json_document *document,
                           size_t required,
                           char *error,
                           size_t error_size) {
    if (required <= document->token_capacity) {
        return true;
    }
    size_t capacity =
        document->token_capacity == 0u ?
            64u : document->token_capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            set_error(error, error_size,
                      "JSON token capacity overflow");
            return false;
        }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(*document->tokens)) {
        set_error(error, error_size,
                  "JSON token storage overflow");
        return false;
    }
    k3_json_token *tokens = (k3_json_token *)realloc(
        document->tokens,
        capacity * sizeof(*document->tokens));
    if (tokens == NULL) {
        set_error(error, error_size,
                  "allocating JSON tokens failed");
        return false;
    }
    document->tokens = tokens;
    document->token_capacity = capacity;
    return true;
}

static int32_t new_token(k3_json_parser *parser,
                         k3_json_type type,
                         size_t start,
                         int32_t parent) {
    k3_json_document *document = parser->document;
    if (document->token_count >= INT32_MAX ||
        !reserve_tokens(
            document, document->token_count + 1u,
            parser->error, parser->error_size)) {
        return -1;
    }
    const int32_t index = (int32_t)document->token_count++;
    document->tokens[index] = (k3_json_token) {
        .type = type,
        .start = start,
        .end = start,
        .size = 0u,
        .parent = parent,
        .first_child = -1,
        .next_sibling = -1,
        .last_child = -1,
    };
    if (parent >= 0) {
        k3_json_token *parent_token =
            &document->tokens[parent];
        if (parent_token->first_child < 0) {
            parent_token->first_child = index;
        } else {
            document->tokens[parent_token->last_child].
                next_sibling = index;
        }
        parent_token->last_child = index;
    }
    return index;
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static size_t utf8_sequence_size(
    const unsigned char *data, size_t remaining);

static int32_t parse_value(k3_json_parser *parser, int32_t parent);

static int32_t parse_string(
        k3_json_parser *parser, int32_t parent) {
    const char *source = parser->document->source;
    const size_t source_size =
        parser->document->source_size;
    if (parser->position >= source_size ||
        source[parser->position] != '"') {
        parser_error(parser, "expected a string");
        return -1;
    }
    parser->position++;
    const size_t start = parser->position;
    while (parser->position < source_size) {
        const unsigned char c =
            (unsigned char)source[parser->position++];
        if (c == '"') {
            const int32_t token = new_token(
                parser, K3_JSON_STRING, start, parent);
            if (token >= 0) {
                parser->document->tokens[token].end =
                    parser->position - 1u;
            }
            return token;
        }
        if (c < 0x20u) {
            parser_error(parser,
                         "unescaped control byte in string");
            return -1;
        }
        if (c >= 0x80u) {
            const size_t width = utf8_sequence_size(
                (const unsigned char *)source +
                    parser->position - 1u,
                source_size - parser->position + 1u);
            if (width == 0u) {
                parser_error(parser,
                             "invalid UTF-8 in string");
                return -1;
            }
            parser->position += width - 1u;
            continue;
        }
        if (c != '\\') {
            continue;
        }
        if (parser->position >= source_size) {
            parser_error(parser,
                         "unterminated string escape");
            return -1;
        }
        const char escaped =
            source[parser->position++];
        if (escaped == '"' || escaped == '\\' ||
            escaped == '/' || escaped == 'b' ||
            escaped == 'f' || escaped == 'n' ||
            escaped == 'r' || escaped == 't') {
            continue;
        }
        if (escaped != 'u' ||
            source_size - parser->position < 4u) {
            parser_error(parser,
                         "invalid string escape");
            return -1;
        }
        uint32_t code_unit = 0u;
        for (unsigned i = 0u; i < 4u; i++) {
            const int digit =
                hex_value(source[parser->position + i]);
            if (digit < 0) {
                parser_error(parser,
                             "invalid Unicode escape");
                return -1;
            }
            code_unit =
                (code_unit << 4u) | (uint32_t)digit;
        }
        parser->position += 4u;
        if (code_unit >= 0xd800u &&
            code_unit <= 0xdbffu) {
            if (source_size - parser->position < 6u ||
                source[parser->position] != '\\' ||
                source[parser->position + 1u] != 'u') {
                parser_error(
                    parser,
                    "high surrogate has no low surrogate");
                return -1;
            }
            uint32_t low = 0u;
            for (unsigned i = 0u; i < 4u; i++) {
                const int digit = hex_value(
                    source[parser->position + 2u + i]);
                if (digit < 0) {
                    parser_error(
                        parser,
                        "invalid low surrogate escape");
                    return -1;
                }
                low = (low << 4u) | (uint32_t)digit;
            }
            if (low < 0xdc00u || low > 0xdfffu) {
                parser_error(
                    parser,
                    "invalid low surrogate");
                return -1;
            }
            parser->position += 6u;
        } else if (code_unit >= 0xdc00u &&
                   code_unit <= 0xdfffu) {
            parser_error(parser, "unexpected low surrogate");
            return -1;
        }
    }
    parser_error(parser, "unterminated string");
    return -1;
}

static bool number_char(char c) {
    return c >= '0' && c <= '9';
}

static int32_t parse_number(
        k3_json_parser *parser, int32_t parent) {
    const char *source = parser->document->source;
    const size_t source_size =
        parser->document->source_size;
    const size_t start = parser->position;
    if (source[parser->position] == '-') {
        parser->position++;
    }
    if (parser->position >= source_size) {
        parser_error(parser, "incomplete number");
        return -1;
    }
    if (source[parser->position] == '0') {
        parser->position++;
        if (parser->position < source_size &&
            number_char(source[parser->position])) {
            parser_error(parser,
                         "leading zero in number");
            return -1;
        }
    } else if (source[parser->position] >= '1' &&
               source[parser->position] <= '9') {
        while (parser->position < source_size &&
               number_char(source[parser->position])) {
            parser->position++;
        }
    } else {
        parser_error(parser, "invalid number");
        return -1;
    }
    if (parser->position < source_size &&
        source[parser->position] == '.') {
        parser->position++;
        if (parser->position >= source_size ||
            !number_char(source[parser->position])) {
            parser_error(parser,
                         "number fraction has no digits");
            return -1;
        }
        while (parser->position < source_size &&
               number_char(source[parser->position])) {
            parser->position++;
        }
    }
    if (parser->position < source_size &&
        (source[parser->position] == 'e' ||
         source[parser->position] == 'E')) {
        parser->position++;
        if (parser->position < source_size &&
            (source[parser->position] == '+' ||
             source[parser->position] == '-')) {
            parser->position++;
        }
        if (parser->position >= source_size ||
            !number_char(source[parser->position])) {
            parser_error(parser,
                         "number exponent has no digits");
            return -1;
        }
        while (parser->position < source_size &&
               number_char(source[parser->position])) {
            parser->position++;
        }
    }
    const int32_t token = new_token(
        parser, K3_JSON_NUMBER, start, parent);
    if (token >= 0) {
        parser->document->tokens[token].end =
            parser->position;
    }
    return token;
}

static int32_t parse_literal(
        k3_json_parser *parser,
        int32_t parent,
        const char *literal,
        k3_json_type type) {
    const size_t size = strlen(literal);
    if (parser->document->source_size -
            parser->position < size ||
        memcmp(
            parser->document->source + parser->position,
            literal, size) != 0) {
        parser_error(parser, "invalid JSON literal");
        return -1;
    }
    const size_t start = parser->position;
    parser->position += size;
    const int32_t token = new_token(
        parser, type, start, parent);
    if (token >= 0) {
        parser->document->tokens[token].end =
            parser->position;
    }
    return token;
}

static int32_t parse_array(
        k3_json_parser *parser, int32_t parent) {
    const size_t start = parser->position++;
    const int32_t array = new_token(
        parser, K3_JSON_ARRAY, start, parent);
    if (array < 0) {
        return -1;
    }
    skip_space(parser);
    if (parser->position < parser->document->source_size &&
        parser->document->source[parser->position] == ']') {
        parser->position++;
        parser->document->tokens[array].end =
            parser->position;
        return array;
    }
    for (;;) {
        if (parse_value(parser, array) < 0) {
            return -1;
        }
        parser->document->tokens[array].size++;
        skip_space(parser);
        if (parser->position >=
            parser->document->source_size) {
            parser_error(parser, "unterminated array");
            return -1;
        }
        const char delimiter =
            parser->document->source[parser->position++];
        if (delimiter == ']') {
            parser->document->tokens[array].end =
                parser->position;
            return array;
        }
        if (delimiter != ',') {
            parser_error(parser,
                         "expected comma or array end");
            return -1;
        }
        skip_space(parser);
    }
}

static int32_t parse_object(
        k3_json_parser *parser, int32_t parent) {
    const size_t start = parser->position++;
    const int32_t object = new_token(
        parser, K3_JSON_OBJECT, start, parent);
    if (object < 0) {
        return -1;
    }
    skip_space(parser);
    if (parser->position < parser->document->source_size &&
        parser->document->source[parser->position] == '}') {
        parser->position++;
        parser->document->tokens[object].end =
            parser->position;
        return object;
    }
    for (;;) {
        if (parse_string(parser, object) < 0) {
            return -1;
        }
        skip_space(parser);
        if (parser->position >=
                parser->document->source_size ||
            parser->document->source[parser->position] != ':') {
            parser_error(parser,
                         "expected colon after object key");
            return -1;
        }
        parser->position++;
        skip_space(parser);
        if (parse_value(parser, object) < 0) {
            return -1;
        }
        parser->document->tokens[object].size++;
        skip_space(parser);
        if (parser->position >=
            parser->document->source_size) {
            parser_error(parser, "unterminated object");
            return -1;
        }
        const char delimiter =
            parser->document->source[parser->position++];
        if (delimiter == '}') {
            parser->document->tokens[object].end =
                parser->position;
            return object;
        }
        if (delimiter != ',') {
            parser_error(parser,
                         "expected comma or object end");
            return -1;
        }
        skip_space(parser);
    }
}

static int32_t parse_value(
        k3_json_parser *parser, int32_t parent) {
    if (parser->depth >= 128u) {
        parser_error(parser,
                     "maximum JSON nesting exceeded");
        return -1;
    }
    skip_space(parser);
    if (parser->position >=
        parser->document->source_size) {
        parser_error(parser, "expected a value");
        return -1;
    }
    parser->depth++;
    const char c =
        parser->document->source[parser->position];
    int32_t token = -1;
    if (c == '{') {
        token = parse_object(parser, parent);
    } else if (c == '[') {
        token = parse_array(parser, parent);
    } else if (c == '"') {
        token = parse_string(parser, parent);
    } else if (c == 't') {
        token = parse_literal(
            parser, parent, "true", K3_JSON_TRUE);
    } else if (c == 'f') {
        token = parse_literal(
            parser, parent, "false", K3_JSON_FALSE);
    } else if (c == 'n') {
        token = parse_literal(
            parser, parent, "null", K3_JSON_NULL);
    } else if (c == '-' || number_char(c)) {
        token = parse_number(parser, parent);
    } else {
        parser_error(parser, "invalid value");
    }
    parser->depth--;
    return token;
}

bool k3_json_parse(k3_json_document *document,
                   const char *source,
                   size_t source_size,
                   char *error,
                   size_t error_size) {
    if (document == NULL || source == NULL) {
        set_error(error, error_size,
                  "JSON parse arguments are invalid");
        return false;
    }
    memset(document, 0, sizeof(*document));
    document->source = source;
    document->source_size = source_size;
    document->root = -1;
    k3_json_parser parser = {
        .document = document,
        .position = 0u,
        .depth = 0u,
        .error = error,
        .error_size = error_size,
    };
    skip_space(&parser);
    const int32_t root = parse_value(&parser, -1);
    if (root < 0) {
        k3_json_document_free(document);
        return false;
    }
    skip_space(&parser);
    if (parser.position != source_size) {
        parser_error(&parser,
                     "trailing data after root value");
        k3_json_document_free(document);
        return false;
    }
    document->root = root;
    return true;
}

void k3_json_document_free(k3_json_document *document) {
    if (document == NULL) {
        return;
    }
    free(document->tokens);
    memset(document, 0, sizeof(*document));
    document->root = -1;
}

static bool token_valid(
        const k3_json_document *document, int32_t token) {
    return document != NULL && token >= 0 &&
           (size_t)token < document->token_count;
}

int32_t k3_json_object_get(
        const k3_json_document *document,
        int32_t object,
        const char *key) {
    if (!token_valid(document, object) ||
        document->tokens[object].type != K3_JSON_OBJECT ||
        key == NULL) {
        return -1;
    }
    int32_t child =
        document->tokens[object].first_child;
    while (child >= 0) {
        const int32_t value =
            document->tokens[child].next_sibling;
        if (value < 0) {
            return -1;
        }
        if (k3_json_string_equal(document, child, key)) {
            return value;
        }
        child =
            document->tokens[value].next_sibling;
    }
    return -1;
}

int32_t k3_json_array_get(
        const k3_json_document *document,
        int32_t array,
        size_t index) {
    if (!token_valid(document, array) ||
        document->tokens[array].type != K3_JSON_ARRAY ||
        index >= document->tokens[array].size) {
        return -1;
    }
    int32_t child =
        document->tokens[array].first_child;
    while (child >= 0 && index != 0u) {
        child = document->tokens[child].next_sibling;
        index--;
    }
    return child;
}

static size_t utf8_sequence_size(
        const unsigned char *data, size_t remaining) {
    if (remaining == 0u) return 0u;
    const unsigned char first = data[0];
    if (first < 0x80u) return 1u;
    size_t size = 0u;
    uint32_t value = 0u;
    uint32_t minimum = 0u;
    if (first >= 0xc2u && first <= 0xdfu) {
        size = 2u;
        value = first & 0x1fu;
        minimum = 0x80u;
    } else if (first >= 0xe0u && first <= 0xefu) {
        size = 3u;
        value = first & 0x0fu;
        minimum = 0x800u;
    } else if (first >= 0xf0u && first <= 0xf4u) {
        size = 4u;
        value = first & 0x07u;
        minimum = 0x10000u;
    } else {
        return 0u;
    }
    if (size > remaining) return 0u;
    for (size_t i = 1u; i < size; i++) {
        if ((data[i] & 0xc0u) != 0x80u) return 0u;
        value = (value << 6u) | (data[i] & 0x3fu);
    }
    if (value < minimum || value > 0x10ffffu ||
        (value >= 0xd800u && value <= 0xdfffu)) {
        return 0u;
    }
    return size;
}

static bool builder_reserve(
        k3_json_builder *builder,
        size_t additional,
        char *error,
        size_t error_size) {
    if (builder->size > SIZE_MAX - additional - 1u) {
        set_error(error, error_size,
                  "JSON string size overflow");
        return false;
    }
    const size_t required =
        builder->size + additional + 1u;
    if (required <= builder->capacity) {
        return true;
    }
    size_t capacity =
        builder->capacity == 0u ? 128u :
            builder->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            set_error(error, error_size,
                      "JSON string capacity overflow");
            return false;
        }
        capacity *= 2u;
    }
    char *data = (char *)realloc(
        builder->data, capacity);
    if (data == NULL) {
        set_error(error, error_size,
                  "allocating JSON string failed");
        return false;
    }
    builder->data = data;
    builder->capacity = capacity;
    return true;
}

static bool builder_append(
        k3_json_builder *builder,
        const void *data,
        size_t size,
        char *error,
        size_t error_size) {
    if (!builder_reserve(
            builder, size, error, error_size)) {
        return false;
    }
    if (size != 0u) {
        memcpy(builder->data + builder->size,
               data, size);
    }
    builder->size += size;
    builder->data[builder->size] = '\0';
    return true;
}

static bool append_codepoint(
        k3_json_builder *builder,
        uint32_t codepoint,
        char *error,
        size_t error_size) {
    char encoded[4];
    size_t size = 0u;
    if (codepoint <= 0x7fu) {
        encoded[0] = (char)codepoint;
        size = 1u;
    } else if (codepoint <= 0x7ffu) {
        encoded[0] = (char)(0xc0u | (codepoint >> 6u));
        encoded[1] = (char)(0x80u | (codepoint & 0x3fu));
        size = 2u;
    } else if (codepoint <= 0xffffu) {
        encoded[0] = (char)(0xe0u | (codepoint >> 12u));
        encoded[1] =
            (char)(0x80u | ((codepoint >> 6u) & 0x3fu));
        encoded[2] = (char)(0x80u | (codepoint & 0x3fu));
        size = 3u;
    } else {
        encoded[0] = (char)(0xf0u | (codepoint >> 18u));
        encoded[1] =
            (char)(0x80u | ((codepoint >> 12u) & 0x3fu));
        encoded[2] =
            (char)(0x80u | ((codepoint >> 6u) & 0x3fu));
        encoded[3] = (char)(0x80u | (codepoint & 0x3fu));
        size = 4u;
    }
    return builder_append(
        builder, encoded, size, error, error_size);
}

static bool unicode_escape(
        const char *source,
        size_t source_size,
        size_t *position,
        uint32_t *codepoint,
        char *error,
        size_t error_size) {
    if (source_size - *position < 4u) {
        set_error(error, error_size,
                  "incomplete Unicode escape");
        return false;
    }
    uint32_t first = 0u;
    for (unsigned i = 0u; i < 4u; i++) {
        const int value = hex_value(source[*position + i]);
        if (value < 0) {
            set_error(error, error_size,
                      "invalid Unicode escape");
            return false;
        }
        first = (first << 4u) | (uint32_t)value;
    }
    *position += 4u;
    if (first >= 0xdc00u && first <= 0xdfffu) {
        set_error(error, error_size,
                  "unpaired low surrogate");
        return false;
    }
    if (first < 0xd800u || first > 0xdbffu) {
        *codepoint = first;
        return true;
    }
    if (source_size - *position < 6u ||
        source[*position] != '\\' ||
        source[*position + 1u] != 'u') {
        set_error(error, error_size,
                  "unpaired high surrogate");
        return false;
    }
    *position += 2u;
    uint32_t second = 0u;
    for (unsigned i = 0u; i < 4u; i++) {
        const int value = hex_value(source[*position + i]);
        if (value < 0) {
            set_error(error, error_size,
                      "invalid low surrogate");
            return false;
        }
        second = (second << 4u) | (uint32_t)value;
    }
    *position += 4u;
    if (second < 0xdc00u || second > 0xdfffu) {
        set_error(error, error_size,
                  "invalid low surrogate");
        return false;
    }
    *codepoint = UINT32_C(0x10000) +
        ((first - 0xd800u) << 10u) +
        (second - 0xdc00u);
    return true;
}

bool k3_json_string_dup(
        const k3_json_document *document,
        int32_t token,
        char **value,
        char *error,
        size_t error_size) {
    if (!token_valid(document, token) ||
        document->tokens[token].type != K3_JSON_STRING ||
        value == NULL) {
        set_error(error, error_size,
                  "JSON token is not a string");
        return false;
    }
    *value = NULL;
    const k3_json_token selected =
        document->tokens[token];
    const char *source =
        document->source + selected.start;
    const size_t source_size =
        selected.end - selected.start;
    k3_json_builder builder = { 0 };
    size_t position = 0u;
    while (position < source_size) {
        const unsigned char c =
            (unsigned char)source[position];
        if (c == '\\') {
            position++;
            const char escaped = source[position++];
            char replacement = '\0';
            if (escaped == '"' || escaped == '\\' ||
                escaped == '/') {
                replacement = escaped;
            } else if (escaped == 'b') {
                replacement = '\b';
            } else if (escaped == 'f') {
                replacement = '\f';
            } else if (escaped == 'n') {
                replacement = '\n';
            } else if (escaped == 'r') {
                replacement = '\r';
            } else if (escaped == 't') {
                replacement = '\t';
            } else if (escaped == 'u') {
                uint32_t codepoint = 0u;
                if (!unicode_escape(
                        source, source_size, &position,
                        &codepoint, error, error_size) ||
                    codepoint == 0u ||
                    !append_codepoint(
                        &builder, codepoint,
                        error, error_size)) {
                    if (codepoint == 0u &&
                        error && error_size) {
                        set_error(error, error_size,
                                  "JSON strings may not contain NUL");
                    }
                    free(builder.data);
                    return false;
                }
                continue;
            } else {
                set_error(error, error_size,
                          "invalid JSON string escape");
                free(builder.data);
                return false;
            }
            if (replacement == '\0') {
                set_error(error, error_size,
                          "JSON strings may not contain NUL");
                free(builder.data);
                return false;
            }
            if (!builder_append(
                    &builder, &replacement, 1u,
                    error, error_size)) {
                free(builder.data);
                return false;
            }
            continue;
        }
        const size_t utf8_size = utf8_sequence_size(
            (const unsigned char *)source + position,
            source_size - position);
        if (utf8_size == 0u) {
            set_error(error, error_size,
                      "invalid UTF-8 in JSON string");
            free(builder.data);
            return false;
        }
        if (!builder_append(
                &builder, source + position, utf8_size,
                error, error_size)) {
            free(builder.data);
            return false;
        }
        position += utf8_size;
    }
    if (!builder_reserve(
            &builder, 0u, error, error_size)) {
        free(builder.data);
        return false;
    }
    builder.data[builder.size] = '\0';
    *value = builder.data;
    return true;
}

bool k3_json_string_equal(
        const k3_json_document *document,
        int32_t token,
        const char *value) {
    if (value == NULL) return false;
    char *decoded = NULL;
    if (!k3_json_string_dup(
            document, token, &decoded, NULL, 0u)) {
        return false;
    }
    const bool equal = strcmp(decoded, value) == 0;
    free(decoded);
    return equal;
}

bool k3_json_u32(const k3_json_document *document,
                 int32_t token,
                 uint32_t *value) {
    if (!token_valid(document, token) ||
        document->tokens[token].type != K3_JSON_NUMBER ||
        value == NULL) {
        return false;
    }
    const k3_json_token selected =
        document->tokens[token];
    const size_t size = selected.end - selected.start;
    if (size == 0u || size >= 32u) {
        return false;
    }
    char text[32];
    memcpy(text, document->source + selected.start, size);
    text[size] = '\0';
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > UINT32_MAX) {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

bool k3_json_bool(const k3_json_document *document,
                  int32_t token,
                  bool *value) {
    if (!token_valid(document, token) || value == NULL) {
        return false;
    }
    if (document->tokens[token].type == K3_JSON_TRUE) {
        *value = true;
        return true;
    }
    if (document->tokens[token].type == K3_JSON_FALSE) {
        *value = false;
        return true;
    }
    return false;
}

typedef struct {
    int32_t key;
    int32_t value;
    char   *decoded_key;
} k3_json_object_pair;

static int compare_object_pairs(const void *left, const void *right) {
    const k3_json_object_pair *a =
        (const k3_json_object_pair *)left;
    const k3_json_object_pair *b =
        (const k3_json_object_pair *)right;
    return strcmp(a->decoded_key, b->decoded_key);
}

static bool compact_sorted_append(
        const k3_json_document *document,
        int32_t token,
        k3_json_builder *builder,
        char *error,
        size_t error_size) {
    if (!token_valid(document, token)) {
        set_error(error, error_size, "invalid JSON token");
        return false;
    }
    const k3_json_token selected = document->tokens[token];
    if (selected.type == K3_JSON_STRING) {
        char *decoded = NULL;
        char *escaped = NULL;
        size_t escaped_size = 0u;
        const bool ok =
            k3_json_string_dup(
                document, token, &decoded,
                error, error_size) &&
            k3_json_escape(
                decoded, strlen(decoded),
                &escaped, &escaped_size,
                error, error_size) &&
            builder_append(
                builder, escaped, escaped_size,
                error, error_size);
        free(escaped);
        free(decoded);
        return ok;
    }
    if (selected.type != K3_JSON_ARRAY &&
        selected.type != K3_JSON_OBJECT) {
        return builder_append(
            builder, document->source + selected.start,
            selected.end - selected.start,
            error, error_size);
    }
    if (selected.type == K3_JSON_ARRAY) {
        if (!builder_append(builder, "[", 1u, error, error_size)) {
            return false;
        }
        int32_t child = selected.first_child;
        for (size_t i = 0u; i < selected.size; i++) {
            if (child < 0 ||
                (i != 0u && !builder_append(
                    builder, ",", 1u, error, error_size)) ||
                !compact_sorted_append(
                    document, child, builder,
                    error, error_size)) {
                return false;
            }
            child = document->tokens[child].next_sibling;
        }
        return builder_append(builder, "]", 1u, error, error_size);
    }

    k3_json_object_pair *pairs = NULL;
    if (selected.size != 0u) {
        if (selected.size > SIZE_MAX / sizeof(*pairs)) {
            set_error(error, error_size, "JSON object is too large");
            return false;
        }
        pairs = (k3_json_object_pair *)calloc(
            selected.size, sizeof(*pairs));
        if (pairs == NULL) {
            set_error(error, error_size,
                      "allocating sorted JSON object failed");
            return false;
        }
    }
    bool ok = true;
    int32_t child = selected.first_child;
    for (size_t i = 0u; i < selected.size; i++) {
        if (child < 0) {
            ok = false;
            set_error(error, error_size,
                      "malformed parsed JSON object");
            break;
        }
        const int32_t value = document->tokens[child].next_sibling;
        if (value < 0 || !k3_json_string_dup(
                document, child, &pairs[i].decoded_key,
                error, error_size)) {
            ok = false;
            break;
        }
        pairs[i].key = child;
        pairs[i].value = value;
        child = document->tokens[value].next_sibling;
    }
    if (ok && selected.size > 1u) {
        qsort(pairs, selected.size, sizeof(*pairs),
              compare_object_pairs);
    }
    if (ok) {
        ok = builder_append(builder, "{", 1u, error, error_size);
    }
    for (size_t i = 0u; ok && i < selected.size; i++) {
        if (i != 0u) {
            ok = builder_append(
                builder, ",", 1u, error, error_size);
        }
        char *escaped = NULL;
        size_t escaped_size = 0u;
        if (ok) {
            ok = k3_json_escape(
                pairs[i].decoded_key,
                strlen(pairs[i].decoded_key),
                &escaped, &escaped_size,
                error, error_size) &&
                builder_append(
                    builder, escaped, escaped_size,
                    error, error_size) &&
                builder_append(
                    builder, ":", 1u,
                    error, error_size) &&
                compact_sorted_append(
                    document, pairs[i].value, builder,
                    error, error_size);
        }
        free(escaped);
    }
    if (ok) {
        ok = builder_append(builder, "}", 1u, error, error_size);
    }
    for (size_t i = 0u; i < selected.size; i++) {
        free(pairs[i].decoded_key);
    }
    free(pairs);
    return ok;
}

bool k3_json_compact_sorted_dup(
        const k3_json_document *document,
        int32_t token,
        char **json,
        size_t *json_size,
        char *error,
        size_t error_size) {
    if (json == NULL || !token_valid(document, token)) {
        set_error(error, error_size,
                  "JSON compact arguments are invalid");
        return false;
    }
    *json = NULL;
    if (json_size != NULL) {
        *json_size = 0u;
    }
    k3_json_builder builder = { 0 };
    if (!compact_sorted_append(
            document, token, &builder,
            error, error_size)) {
        free(builder.data);
        return false;
    }
    if (!builder_reserve(&builder, 0u, error, error_size)) {
        free(builder.data);
        return false;
    }
    builder.data[builder.size] = '\0';
    *json = builder.data;
    if (json_size != NULL) {
        *json_size = builder.size;
    }
    return true;
}

bool k3_json_escape(const char *text,
                    size_t text_size,
                    char **escaped,
                    size_t *escaped_size,
                    char *error,
                    size_t error_size) {
    if (text == NULL || escaped == NULL) {
        set_error(error, error_size,
                  "JSON escape arguments are invalid");
        return false;
    }
    *escaped = NULL;
    if (escaped_size != NULL) {
        *escaped_size = 0u;
    }
    k3_json_builder builder = { 0 };
    if (!builder_append(
            &builder, "\"", 1u, error, error_size)) {
        return false;
    }
    size_t position = 0u;
    while (position < text_size) {
        const unsigned char c =
            (unsigned char)text[position];
        const char *replacement = NULL;
        size_t replacement_size = 0u;
        switch (c) {
            case '"':  replacement = "\\\""; replacement_size = 2u; break;
            case '\\': replacement = "\\\\"; replacement_size = 2u; break;
            case '\b': replacement = "\\b";  replacement_size = 2u; break;
            case '\f': replacement = "\\f";  replacement_size = 2u; break;
            case '\n': replacement = "\\n";  replacement_size = 2u; break;
            case '\r': replacement = "\\r";  replacement_size = 2u; break;
            case '\t': replacement = "\\t";  replacement_size = 2u; break;
            default: break;
        }
        if (replacement != NULL) {
            if (!builder_append(
                    &builder, replacement, replacement_size,
                    error, error_size)) {
                free(builder.data);
                return false;
            }
            position++;
            continue;
        }
        if (c < 0x20u) {
            char control[7];
            snprintf(control, sizeof(control), "\\u%04x", c);
            if (!builder_append(
                    &builder, control, 6u,
                    error, error_size)) {
                free(builder.data);
                return false;
            }
            position++;
            continue;
        }
        const size_t utf8_size = utf8_sequence_size(
            (const unsigned char *)text + position,
            text_size - position);
        if (utf8_size == 0u) {
            static const char replacement_utf8[] = "\xef\xbf\xbd";
            if (!builder_append(
                    &builder, replacement_utf8, 3u,
                    error, error_size)) {
                free(builder.data);
                return false;
            }
            position++;
            continue;
        }
        if (!builder_append(
                &builder, text + position, utf8_size,
                error, error_size)) {
            free(builder.data);
            return false;
        }
        position += utf8_size;
    }
    if (!builder_append(
            &builder, "\"", 1u, error, error_size)) {
        free(builder.data);
        return false;
    }
    *escaped = builder.data;
    if (escaped_size != NULL) {
        *escaped_size = builder.size;
    }
    return true;
}
