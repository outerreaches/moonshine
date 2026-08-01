#include "k3_tokenizer.h"

#include "k3_json.h"

#include <unicode/uregex.h>
#include <unicode/utext.h>
#include <unicode/utypes.h>

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    K3_BASE_TOKEN_COUNT = 163584,
    K3_SPECIAL_TOKEN_COUNT = 256,
    K3_VOCAB_HASH_CAPACITY = 1 << 19,
};

typedef struct {
    uint32_t offset;
    uint32_t size;
} k3_vocab_piece;

typedef struct {
    uint64_t hash;
    uint32_t token_id;
    bool     used;
} k3_vocab_hash_entry;

struct k3_tokenizer {
    uint8_t             *piece_data;
    size_t               piece_data_size;
    size_t               piece_data_capacity;
    k3_vocab_piece      *pieces;
    k3_vocab_hash_entry *hash_table;
    size_t               hash_capacity;
    URegularExpression  *pretokenizer;
};

static const char *K3_PRETOKENIZER_PATTERN =
    "[\\p{Han}]+|"
    "[^\\r\\n\\p{L}\\p{N}]?"
    "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "[^\\r\\n\\p{L}\\p{N}]?"
    "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "\\p{N}{1,3}|"
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
    "\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|"
    "\\s+";

typedef struct {
    uint32_t id;
    const char *text;
} k3_special_override;

static const k3_special_override K3_SPECIAL_OVERRIDES[] = {
    { 163584u, "[BOS]" },
    { 163585u, "[EOS]" },
    { 163586u, "<|end_of_msg|>" },
    { 163587u, "<|open|>" },
    { 163588u, "<|close|>" },
    { 163589u, "<|sep|>" },
    { 163590u, "[start_header_id]" },
    { 163591u, "[end_header_id]" },
    { 163593u, "[EOT]" },
    { 163602u, "<|media_begin|>" },
    { 163603u, "<|media_content|>" },
    { 163604u, "<|media_end|>" },
    { 163605u, "<|media_pad|>" },
    { 163649u, "<osagent_mode>" },
    { 163838u, "[UNK]" },
    { 163839u, "[PAD]" },
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

static bool checked_add_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool reserve_tokens(k3_token_buffer *buffer,
                           size_t required,
                           char *error,
                           size_t error_size) {
    if (required <= buffer->capacity) {
        return true;
    }
    size_t capacity = buffer->capacity == 0u ? 64u : buffer->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            set_error(error, error_size, "token buffer capacity overflow");
            return false;
        }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(*buffer->data)) {
        set_error(error, error_size, "token buffer byte size overflow");
        return false;
    }
    uint32_t *data = (uint32_t *)realloc(
        buffer->data, capacity * sizeof(*buffer->data));
    if (data == NULL) {
        set_error(error, error_size,
                  "allocating %zu tokenizer output tokens failed", capacity);
        return false;
    }
    buffer->data = data;
    buffer->capacity = capacity;
    return true;
}

static bool append_token(k3_token_buffer *buffer,
                         uint32_t token,
                         char *error,
                         size_t error_size) {
    size_t required = 0u;
    if (!checked_add_size(buffer->count, 1u, &required) ||
        !reserve_tokens(buffer, required, error, error_size)) {
        return false;
    }
    buffer->data[buffer->count++] = token;
    return true;
}

static bool reserve_text(k3_text_buffer *buffer,
                         size_t required,
                         char *error,
                         size_t error_size) {
    if (required <= buffer->capacity) {
        return true;
    }
    size_t capacity = buffer->capacity == 0u ? 256u : buffer->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            set_error(error, error_size, "text buffer capacity overflow");
            return false;
        }
        capacity *= 2u;
    }
    char *data = (char *)realloc(buffer->data, capacity);
    if (data == NULL) {
        set_error(error, error_size,
                  "allocating %zu tokenizer output bytes failed", capacity);
        return false;
    }
    buffer->data = data;
    buffer->capacity = capacity;
    return true;
}

static bool append_text_bytes(k3_text_buffer *buffer,
                              const void *data,
                              size_t size,
                              char *error,
                              size_t error_size) {
    size_t with_data = 0u;
    size_t required = 0u;
    if (!checked_add_size(buffer->size, size, &with_data) ||
        !checked_add_size(with_data, 1u, &required) ||
        !reserve_text(buffer, required, error, error_size)) {
        return false;
    }
    if (size != 0u) {
        memcpy(buffer->data + buffer->size, data, size);
    }
    buffer->size = with_data;
    buffer->data[buffer->size] = '\0';
    return true;
}

void k3_token_buffer_free(k3_token_buffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

void k3_text_buffer_free(k3_text_buffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

static uint64_t hash_bytes(const uint8_t *data, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0u; i < size; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool vocab_lookup(const k3_tokenizer *tokenizer,
                         const uint8_t *data,
                         size_t size,
                         uint32_t *token_id) {
    if (size > UINT32_MAX) {
        return false;
    }
    const uint64_t hash = hash_bytes(data, size);
    size_t slot = (size_t)hash & (tokenizer->hash_capacity - 1u);
    for (size_t probes = 0u;
         probes < tokenizer->hash_capacity;
         probes++) {
        const k3_vocab_hash_entry *entry =
            &tokenizer->hash_table[slot];
        if (!entry->used) {
            return false;
        }
        if (entry->hash == hash) {
            const k3_vocab_piece piece =
                tokenizer->pieces[entry->token_id];
            if (piece.size == size &&
                memcmp(tokenizer->piece_data + piece.offset,
                       data, size) == 0) {
                *token_id = entry->token_id;
                return true;
            }
        }
        slot = (slot + 1u) & (tokenizer->hash_capacity - 1u);
    }
    return false;
}

static bool vocab_insert(k3_tokenizer *tokenizer,
                         uint32_t token_id,
                         char *error,
                         size_t error_size) {
    const k3_vocab_piece piece = tokenizer->pieces[token_id];
    const uint8_t *data = tokenizer->piece_data + piece.offset;
    const uint64_t hash = hash_bytes(data, piece.size);
    size_t slot = (size_t)hash & (tokenizer->hash_capacity - 1u);
    for (size_t probes = 0u;
         probes < tokenizer->hash_capacity;
         probes++) {
        k3_vocab_hash_entry *entry = &tokenizer->hash_table[slot];
        if (!entry->used) {
            entry->used = true;
            entry->hash = hash;
            entry->token_id = token_id;
            return true;
        }
        if (entry->hash == hash) {
            const k3_vocab_piece existing =
                tokenizer->pieces[entry->token_id];
            if (existing.size == piece.size &&
                memcmp(tokenizer->piece_data + existing.offset,
                       data, piece.size) == 0) {
                set_error(error, error_size,
                          "duplicate tiktoken piece at ranks %u and %u",
                          entry->token_id, token_id);
                return false;
            }
        }
        slot = (slot + 1u) & (tokenizer->hash_capacity - 1u);
    }
    set_error(error, error_size, "tiktoken hash table is full");
    return false;
}

static int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') {
        return (int)(c - 'A');
    }
    if (c >= 'a' && c <= 'z') {
        return (int)(c - 'a') + 26;
    }
    if (c >= '0' && c <= '9') {
        return (int)(c - '0') + 52;
    }
    if (c == '+') {
        return 62;
    }
    if (c == '/') {
        return 63;
    }
    return -1;
}

static bool decode_base64(const char *input,
                          size_t input_size,
                          uint8_t **output,
                          size_t *output_size,
                          char *error,
                          size_t error_size) {
    if (input_size == 0u || input_size % 4u != 0u) {
        set_error(error, error_size,
                  "invalid tiktoken base64 length %zu", input_size);
        return false;
    }
    size_t capacity = input_size / 4u * 3u;
    uint8_t *data = (uint8_t *)malloc(capacity == 0u ? 1u : capacity);
    if (data == NULL) {
        set_error(error, error_size,
                  "allocating base64 decode buffer failed");
        return false;
    }
    size_t written = 0u;
    for (size_t i = 0u; i < input_size; i += 4u) {
        int values[4];
        for (size_t j = 0u; j < 4u; j++) {
            const unsigned char c = (unsigned char)input[i + j];
            if (c == '=') {
                values[j] = -2;
            } else {
                values[j] = base64_value(c);
            }
        }
        if (values[0] < 0 || values[1] < 0 ||
            values[2] == -1 || values[3] == -1 ||
            (values[2] == -2 && values[3] != -2) ||
            (i + 4u != input_size &&
             (values[2] == -2 || values[3] == -2))) {
            free(data);
            set_error(error, error_size,
                      "invalid tiktoken base64 data");
            return false;
        }
        const uint32_t packed =
            ((uint32_t)values[0] << 18u) |
            ((uint32_t)values[1] << 12u) |
            ((uint32_t)(values[2] < 0 ? 0 : values[2]) << 6u) |
            (uint32_t)(values[3] < 0 ? 0 : values[3]);
        data[written++] = (uint8_t)(packed >> 16u);
        if (values[2] >= 0) {
            data[written++] = (uint8_t)(packed >> 8u);
        }
        if (values[3] >= 0) {
            data[written++] = (uint8_t)packed;
        }
    }
    *output = data;
    *output_size = written;
    return true;
}

static bool append_piece_data(k3_tokenizer *tokenizer,
                              const uint8_t *data,
                              size_t size,
                              uint32_t *offset,
                              char *error,
                              size_t error_size) {
    size_t required = 0u;
    if (size > UINT32_MAX ||
        !checked_add_size(tokenizer->piece_data_size, size, &required) ||
        required > UINT32_MAX) {
        set_error(error, error_size,
                  "tiktoken piece storage exceeds format bounds");
        return false;
    }
    if (required > tokenizer->piece_data_capacity) {
        size_t capacity = tokenizer->piece_data_capacity == 0u ?
            4u * 1024u * 1024u : tokenizer->piece_data_capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2u) {
                set_error(error, error_size,
                          "tiktoken piece storage capacity overflow");
                return false;
            }
            capacity *= 2u;
        }
        uint8_t *piece_data =
            (uint8_t *)realloc(tokenizer->piece_data, capacity);
        if (piece_data == NULL) {
            set_error(error, error_size,
                      "allocating tiktoken piece storage failed");
            return false;
        }
        tokenizer->piece_data = piece_data;
        tokenizer->piece_data_capacity = capacity;
    }
    *offset = (uint32_t)tokenizer->piece_data_size;
    memcpy(tokenizer->piece_data + tokenizer->piece_data_size, data, size);
    tokenizer->piece_data_size = required;
    return true;
}

static bool load_vocab(k3_tokenizer *tokenizer,
                       const char *path,
                       char *error,
                       size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        set_error(error, error_size, "opening %s failed: %s",
                  path, strerror(errno));
        return false;
    }

    char *line = NULL;
    size_t line_capacity = 0u;
    ssize_t line_size = 0;
    uint32_t loaded = 0u;
    bool ok = true;
    while ((line_size = getline(&line, &line_capacity, file)) >= 0) {
        while (line_size > 0 &&
               (line[line_size - 1] == '\n' ||
                line[line_size - 1] == '\r')) {
            line[--line_size] = '\0';
        }
        char *space = strrchr(line, ' ');
        if (space == NULL || space == line || space[1] == '\0') {
            set_error(error, error_size,
                      "invalid tiktoken line %u", loaded + 1u);
            ok = false;
            break;
        }
        *space = '\0';
        char *end = NULL;
        errno = 0;
        const unsigned long rank = strtoul(space + 1, &end, 10);
        if (errno != 0 || end == space + 1 || *end != '\0' ||
            rank != loaded || rank >= K3_BASE_TOKEN_COUNT) {
            set_error(error, error_size,
                      "unexpected tiktoken rank on line %u", loaded + 1u);
            ok = false;
            break;
        }

        uint8_t *decoded = NULL;
        size_t decoded_size = 0u;
        if (!decode_base64(
                line, strlen(line), &decoded, &decoded_size,
                error, error_size)) {
            ok = false;
            break;
        }
        uint32_t offset = 0u;
        if (!append_piece_data(
                tokenizer, decoded, decoded_size, &offset,
                error, error_size)) {
            free(decoded);
            ok = false;
            break;
        }
        free(decoded);
        tokenizer->pieces[loaded].offset = offset;
        tokenizer->pieces[loaded].size = (uint32_t)decoded_size;
        loaded++;
    }

    if (ok && ferror(file)) {
        set_error(error, error_size, "reading %s failed: %s",
                  path, strerror(errno));
        ok = false;
    }
    if (ok && loaded != K3_BASE_TOKEN_COUNT) {
        set_error(error, error_size,
                  "tiktoken model has %u base tokens, expected %u",
                  loaded, K3_BASE_TOKEN_COUNT);
        ok = false;
    }
    free(line);
    fclose(file);

    if (!ok) {
        return false;
    }
    for (uint32_t id = 0u; id < loaded; id++) {
        if (!vocab_insert(tokenizer, id, error, error_size)) {
            return false;
        }
    }
    return true;
}

static const char *special_name(uint32_t token_id,
                                char generated[64]) {
    for (size_t i = 0u;
         i < sizeof(K3_SPECIAL_OVERRIDES) /
             sizeof(K3_SPECIAL_OVERRIDES[0]);
         i++) {
        if (K3_SPECIAL_OVERRIDES[i].id == token_id) {
            return K3_SPECIAL_OVERRIDES[i].text;
        }
    }
    snprintf(generated, 64u, "<|reserved_token_%u|>", token_id);
    return generated;
}

static bool validate_specials(k3_tokenizer *tokenizer,
                              char *error,
                              size_t error_size) {
    for (uint32_t id = K3_BASE_TOKEN_COUNT;
         id < K3_TOKEN_VOCAB_SIZE;
         id++) {
        char generated[64];
        const char *name = special_name(id, generated);
        uint32_t ordinary = 0u;
        if (vocab_lookup(
                tokenizer, (const uint8_t *)name, strlen(name),
                &ordinary)) {
            set_error(error, error_size,
                      "special token %u duplicates base token %u",
                      id, ordinary);
            return false;
        }
    }
    return true;
}

bool k3_tokenizer_create(k3_tokenizer **out,
                         const char *model_root,
                         char *error,
                         size_t error_size) {
    if (out == NULL || model_root == NULL || model_root[0] == '\0') {
        set_error(error, error_size,
                  "tokenizer creation needs output and model root");
        return false;
    }
    *out = NULL;
    k3_tokenizer *tokenizer =
        (k3_tokenizer *)calloc(1u, sizeof(*tokenizer));
    if (tokenizer == NULL) {
        set_error(error, error_size, "allocating tokenizer failed");
        return false;
    }
    tokenizer->pieces = (k3_vocab_piece *)calloc(
        K3_BASE_TOKEN_COUNT, sizeof(*tokenizer->pieces));
    tokenizer->hash_capacity = K3_VOCAB_HASH_CAPACITY;
    tokenizer->hash_table = (k3_vocab_hash_entry *)calloc(
        tokenizer->hash_capacity, sizeof(*tokenizer->hash_table));
    if (tokenizer->pieces == NULL || tokenizer->hash_table == NULL) {
        set_error(error, error_size,
                  "allocating tokenizer vocabulary index failed");
        k3_tokenizer_destroy(tokenizer);
        return false;
    }

    const size_t root_size = strlen(model_root);
    static const char suffix[] = "/tiktoken.model";
    size_t path_size = 0u;
    if (!checked_add_size(root_size, sizeof(suffix), &path_size)) {
        set_error(error, error_size, "tokenizer model path is too long");
        k3_tokenizer_destroy(tokenizer);
        return false;
    }
    char *path = (char *)malloc(path_size);
    if (path == NULL) {
        set_error(error, error_size,
                  "allocating tokenizer model path failed");
        k3_tokenizer_destroy(tokenizer);
        return false;
    }
    snprintf(path, path_size, "%s%s", model_root, suffix);
    const bool loaded = load_vocab(
        tokenizer, path, error, error_size);
    free(path);
    if (!loaded ||
        !validate_specials(tokenizer, error, error_size)) {
        k3_tokenizer_destroy(tokenizer);
        return false;
    }

    UErrorCode status = U_ZERO_ERROR;
    UParseError parse_error;
    memset(&parse_error, 0, sizeof(parse_error));
    tokenizer->pretokenizer = uregex_openC(
        K3_PRETOKENIZER_PATTERN,
        UREGEX_ERROR_ON_UNKNOWN_ESCAPES,
        &parse_error, &status);
    if (U_FAILURE(status) || tokenizer->pretokenizer == NULL) {
        set_error(error, error_size,
                  "compiling K3 tokenizer regex failed at line %d offset %d: %s",
                  parse_error.line, parse_error.offset,
                  u_errorName(status));
        k3_tokenizer_destroy(tokenizer);
        return false;
    }

    *out = tokenizer;
    return true;
}

void k3_tokenizer_destroy(k3_tokenizer *tokenizer) {
    if (tokenizer == NULL) {
        return;
    }
    if (tokenizer->pretokenizer != NULL) {
        uregex_close(tokenizer->pretokenizer);
    }
    free(tokenizer->hash_table);
    free(tokenizer->pieces);
    free(tokenizer->piece_data);
    free(tokenizer);
}

static bool encode_bpe_piece(k3_tokenizer *tokenizer,
                             const uint8_t *piece,
                             size_t piece_size,
                             k3_token_buffer *output,
                             char *error,
                             size_t error_size) {
    if (piece_size == 0u) {
        return true;
    }
    uint32_t whole = 0u;
    if (vocab_lookup(tokenizer, piece, piece_size, &whole)) {
        return append_token(output, whole, error, error_size);
    }
    if (piece_size > UINT32_MAX ||
        piece_size == SIZE_MAX) {
        set_error(error, error_size, "BPE piece is too large");
        return false;
    }
    uint32_t *boundaries = (uint32_t *)malloc(
        (piece_size + 1u) * sizeof(*boundaries));
    if (boundaries == NULL) {
        set_error(error, error_size,
                  "allocating BPE merge workspace failed");
        return false;
    }
    size_t boundary_count = piece_size + 1u;
    for (size_t i = 0u; i < boundary_count; i++) {
        boundaries[i] = (uint32_t)i;
    }

    while (boundary_count > 2u) {
        uint32_t best_rank = UINT32_MAX;
        size_t best_index = SIZE_MAX;
        for (size_t i = 0u; i + 2u < boundary_count; i++) {
            const size_t start = boundaries[i];
            const size_t end = boundaries[i + 2u];
            uint32_t rank = 0u;
            if (vocab_lookup(
                    tokenizer, piece + start, end - start, &rank) &&
                (rank < best_rank ||
                 (rank == best_rank && i < best_index))) {
                best_rank = rank;
                best_index = i;
            }
        }
        if (best_index == SIZE_MAX) {
            break;
        }
        memmove(&boundaries[best_index + 1u],
                &boundaries[best_index + 2u],
                (boundary_count - best_index - 2u) *
                    sizeof(*boundaries));
        boundary_count--;
    }

    bool ok = true;
    for (size_t i = 0u; i + 1u < boundary_count; i++) {
        const size_t start = boundaries[i];
        const size_t end = boundaries[i + 1u];
        uint32_t token = 0u;
        if (!vocab_lookup(
                tokenizer, piece + start, end - start, &token)) {
            set_error(error, error_size,
                      "K3 vocabulary lacks a BPE byte segment");
            ok = false;
            break;
        }
        if (!append_token(output, token, error, error_size)) {
            ok = false;
            break;
        }
    }
    free(boundaries);
    return ok;
}

static bool encode_ordinary(k3_tokenizer *tokenizer,
                            const char *text,
                            size_t text_size,
                            k3_token_buffer *output,
                            char *error,
                            size_t error_size) {
    if (text_size == 0u) {
        return true;
    }
    if (text_size > INT64_MAX) {
        set_error(error, error_size,
                  "tokenizer input exceeds ICU bounds");
        return false;
    }

    UErrorCode status = U_ZERO_ERROR;
    UText subject = UTEXT_INITIALIZER;
    UText *opened = utext_openUTF8(
        &subject, text, (int64_t)text_size, &status);
    if (U_FAILURE(status) || opened == NULL) {
        set_error(error, error_size,
                  "opening UTF-8 tokenizer input failed: %s",
                  u_errorName(status));
        return false;
    }
    uregex_setUText(tokenizer->pretokenizer, &subject, &status);
    if (U_FAILURE(status)) {
        set_error(error, error_size,
                  "setting tokenizer regex input failed: %s",
                  u_errorName(status));
        utext_close(&subject);
        return false;
    }

    int64_t consumed = 0;
    bool ok = true;
    while (uregex_findNext(tokenizer->pretokenizer, &status)) {
        const int64_t start =
            uregex_start64(tokenizer->pretokenizer, 0, &status);
        const int64_t end =
            uregex_end64(tokenizer->pretokenizer, 0, &status);
        if (U_FAILURE(status) ||
            start != consumed || end <= start ||
            end > (int64_t)text_size) {
            set_error(error, error_size,
                      "K3 tokenizer regex did not partition UTF-8 input");
            ok = false;
            break;
        }
        if (!encode_bpe_piece(
                tokenizer, (const uint8_t *)text + start,
                (size_t)(end - start), output,
                error, error_size)) {
            ok = false;
            break;
        }
        consumed = end;
    }
    if (ok && U_FAILURE(status)) {
        set_error(error, error_size,
                  "K3 tokenizer regex failed: %s",
                  u_errorName(status));
        ok = false;
    }
    if (ok && consumed != (int64_t)text_size) {
        set_error(error, error_size,
                  "K3 tokenizer regex left %zu UTF-8 bytes unmatched",
                  text_size - (size_t)consumed);
        ok = false;
    }
    utext_close(&subject);
    return ok;
}

static bool find_next_special(const char *text,
                              size_t offset,
                              size_t *special_offset,
                              size_t *special_size,
                              uint32_t *special_id) {
    const char *best = NULL;
    size_t best_size = 0u;
    uint32_t best_id = 0u;
    for (uint32_t id = K3_BASE_TOKEN_COUNT;
         id < K3_TOKEN_VOCAB_SIZE;
         id++) {
        char generated[64];
        const char *name = special_name(id, generated);
        const char *found = strstr(text + offset, name);
        if (found == NULL) {
            continue;
        }
        const size_t name_size = strlen(name);
        if (best == NULL || found < best ||
            (found == best && name_size > best_size)) {
            best = found;
            best_size = name_size;
            best_id = id;
        }
    }
    if (best == NULL) {
        return false;
    }
    *special_offset = (size_t)(best - text);
    *special_size = best_size;
    *special_id = best_id;
    return true;
}

static bool encode_append(k3_tokenizer *tokenizer,
                          const char *text,
                          bool allow_special,
                          k3_token_buffer *output,
                          char *error,
                          size_t error_size) {
    if (text == NULL) {
        set_error(error, error_size,
                  "tokenizer input must not be NULL");
        return false;
    }
    const size_t text_size = strlen(text);
    if (!allow_special) {
        return encode_ordinary(
            tokenizer, text, text_size, output,
            error, error_size);
    }

    size_t offset = 0u;
    while (offset < text_size) {
        size_t special_offset = 0u;
        size_t special_size = 0u;
        uint32_t special_id = 0u;
        if (!find_next_special(
                text, offset, &special_offset,
                &special_size, &special_id)) {
            return encode_ordinary(
                tokenizer, text + offset, text_size - offset,
                output, error, error_size);
        }
        if (special_offset > offset &&
            !encode_ordinary(
                tokenizer, text + offset, special_offset - offset,
                output, error, error_size)) {
            return false;
        }
        if (!append_token(output, special_id, error, error_size)) {
            return false;
        }
        offset = special_offset + special_size;
    }
    return true;
}

bool k3_tokenizer_encode(k3_tokenizer *tokenizer,
                         const char *text,
                         bool allow_special,
                         k3_token_buffer *output,
                         char *error,
                         size_t error_size) {
    if (tokenizer == NULL || output == NULL) {
        set_error(error, error_size,
                  "tokenizer encode needs tokenizer and output");
        return false;
    }
    output->count = 0u;
    return encode_append(
        tokenizer, text, allow_special, output,
        error, error_size);
}

bool k3_tokenizer_decode(const k3_tokenizer *tokenizer,
                         const uint32_t *tokens,
                         size_t token_count,
                         bool include_special,
                         k3_text_buffer *output,
                         char *error,
                         size_t error_size) {
    if (tokenizer == NULL || output == NULL ||
        (tokens == NULL && token_count != 0u)) {
        set_error(error, error_size,
                  "tokenizer decode arguments are invalid");
        return false;
    }
    output->size = 0u;
    if (!reserve_text(output, 1u, error, error_size)) {
        return false;
    }
    output->data[0] = '\0';
    for (size_t i = 0u; i < token_count; i++) {
        const uint32_t token = tokens[i];
        if (token < K3_BASE_TOKEN_COUNT) {
            const k3_vocab_piece piece = tokenizer->pieces[token];
            if (!append_text_bytes(
                    output,
                    tokenizer->piece_data + piece.offset,
                    piece.size, error, error_size)) {
                return false;
            }
        } else if (token < K3_TOKEN_VOCAB_SIZE) {
            if (!include_special) {
                continue;
            }
            char generated[64];
            const char *name = special_name(token, generated);
            if (!append_text_bytes(
                    output, name, strlen(name),
                    error, error_size)) {
                return false;
            }
        } else {
            set_error(error, error_size,
                      "token ID %u exceeds K3 vocabulary", token);
            return false;
        }
    }
    return true;
}

static bool append_control(k3_token_buffer *output,
                           uint32_t token,
                           char *error,
                           size_t error_size) {
    return append_token(output, token, error, error_size);
}

static bool append_ordinary(k3_tokenizer *tokenizer,
                            const char *text,
                            k3_token_buffer *output,
                            char *error,
                            size_t error_size) {
    return encode_append(
        tokenizer, text, false, output,
        error, error_size);
}

static bool append_escaped_attr_value(k3_tokenizer *tokenizer,
                                      const char *value,
                                      k3_token_buffer *output,
                                      char *error,
                                      size_t error_size) {
    k3_text_buffer escaped = { 0 };
    for (const char *cursor = value; *cursor != '\0'; cursor++) {
        const char *replacement = NULL;
        if (*cursor == '&') {
            replacement = "&amp;";
        } else if (*cursor == '"') {
            replacement = "&quot;";
        }
        if (replacement != NULL) {
            if (!append_text_bytes(
                    &escaped, replacement, strlen(replacement),
                    error, error_size)) {
                k3_text_buffer_free(&escaped);
                return false;
            }
        } else if (!append_text_bytes(
                       &escaped, cursor, 1u, error, error_size)) {
            k3_text_buffer_free(&escaped);
            return false;
        }
    }
    const char *text = escaped.data == NULL ? "" : escaped.data;
    const bool ok = append_ordinary(
        tokenizer, text, output, error, error_size);
    k3_text_buffer_free(&escaped);
    return ok;
}

static bool append_attr(k3_tokenizer *tokenizer,
                        const char *key,
                        const char *value,
                        k3_token_buffer *output,
                        char *error,
                        size_t error_size) {
    char *key_segment = NULL;
    const size_t key_size = strlen(key);
    if (key_size == SIZE_MAX) {
        set_error(error, error_size, "XTML attribute key is too long");
        return false;
    }
    key_segment = (char *)malloc(key_size + 2u);
    if (key_segment == NULL) {
        set_error(error, error_size,
                  "allocating XTML attribute segment failed");
        return false;
    }
    key_segment[0] = ' ';
    memcpy(key_segment + 1u, key, key_size + 1u);
    bool ok =
        append_ordinary(
            tokenizer, key_segment, output, error, error_size) &&
        append_ordinary(
            tokenizer, "=\"", output, error, error_size) &&
        append_escaped_attr_value(
            tokenizer, value, output, error, error_size) &&
        append_ordinary(
            tokenizer, "\"", output, error, error_size);
    free(key_segment);
    return ok;
}

static bool open_tag(k3_tokenizer *tokenizer,
                     const char *tag,
                     const char *const *attr_keys,
                     const char *const *attr_values,
                     size_t attr_count,
                     k3_token_buffer *output,
                     char *error,
                     size_t error_size) {
    if (!append_control(
            output, K3_TOKEN_OPEN, error, error_size) ||
        !append_ordinary(
            tokenizer, tag, output, error, error_size)) {
        return false;
    }
    for (size_t i = 0u; i < attr_count; i++) {
        if (!append_attr(
                tokenizer, attr_keys[i], attr_values[i],
                output, error, error_size)) {
            return false;
        }
    }
    return append_control(
        output, K3_TOKEN_SEP, error, error_size);
}

static bool close_tag(k3_tokenizer *tokenizer,
                      const char *tag,
                      k3_token_buffer *output,
                      char *error,
                      size_t error_size) {
    return
        append_control(
            output, K3_TOKEN_CLOSE, error, error_size) &&
        append_ordinary(
            tokenizer, tag, output, error, error_size) &&
        append_control(
            output, K3_TOKEN_SEP, error, error_size);
}

static bool end_message(k3_token_buffer *output,
                        char *error,
                        size_t error_size) {
    return append_control(
        output, K3_TOKEN_END_OF_MSG, error, error_size);
}

static bool open_message(k3_tokenizer *tokenizer,
                         const char *role,
                         const char *name,
                         const char *type,
                         k3_token_buffer *output,
                         char *error,
                         size_t error_size) {
    const char *keys[3];
    const char *values[3];
    size_t count = 0u;
    keys[count] = "role";
    values[count++] = role;
    if (name != NULL && name[0] != '\0') {
        keys[count] = "name";
        values[count++] = name;
    }
    if (type != NULL && type[0] != '\0') {
        keys[count] = "type";
        values[count++] = type;
    }
    return open_tag(
        tokenizer, "message", keys, values, count,
        output, error, error_size);
}

static bool append_internal_system_message(
        k3_tokenizer *tokenizer,
        const char *type,
        const char *body,
        k3_token_buffer *output,
        char *error,
        size_t error_size) {
    return
        open_message(
            tokenizer, "system", NULL, type,
            output, error, error_size) &&
        append_ordinary(
            tokenizer, body, output, error, error_size) &&
        close_tag(
            tokenizer, "message", output,
            error, error_size) &&
        end_message(output, error, error_size);
}

static bool append_thinking_effort(
        k3_tokenizer *tokenizer,
        const char *effort,
        k3_token_buffer *output,
        char *error,
        size_t error_size) {
    if (effort == NULL) {
        return true;
    }
    if (strcmp(effort, "low") != 0 &&
        strcmp(effort, "high") != 0 &&
        strcmp(effort, "max") != 0) {
        set_error(error, error_size,
                  "unsupported K3 thinking effort \"%s\"", effort);
        return false;
    }
    static const char prefix[] =
        "`thinking_effort` guides on how much to think in your thinking "
        "channel (not including the response channel), supported values "
        "include `low`, `medium`, `high`, and `max`.\n"
        "Now the system is invoked with `thinking_effort=";
    static const char suffix[] = "`.";
    const size_t size =
        sizeof(prefix) - 1u + strlen(effort) + sizeof(suffix);
    char *body = (char *)malloc(size);
    if (body == NULL) {
        set_error(error, error_size,
                  "allocating thinking-effort message failed");
        return false;
    }
    snprintf(body, size, "%s%s%s", prefix, effort, suffix);
    const bool ok = append_internal_system_message(
        tokenizer, "thinking-effort", body,
        output, error, error_size);
    free(body);
    return ok;
}

static bool append_tool_declare(
        k3_tokenizer *tokenizer,
        const char *tools_json,
        bool dynamic,
        k3_token_buffer *output,
        char *error,
        size_t error_size) {
    if (tools_json == NULL || tools_json[0] == '\0') {
        return true;
    }
    static const char normal_prefix[] =
        "# Tools\n"
        "Here are the available tools, described in JSONSchema.\n\n"
        "```json\n";
    static const char dynamic_prefix[] =
        "## New Tools Available\n"
        "The system dynamically extends the toolset via lazy-loading.\n"
        "You have access to all existing and extended tools.\n"
        "Here are the specs for the extended tools.\n\n"
        "```json\n";
    static const char suffix[] = "\n```";
    const char *prefix = dynamic ? dynamic_prefix : normal_prefix;
    const size_t prefix_size = strlen(prefix);
    const size_t tools_size = strlen(tools_json);
    const size_t suffix_size = sizeof(suffix) - 1u;
    if (prefix_size > SIZE_MAX - tools_size ||
        prefix_size + tools_size > SIZE_MAX - suffix_size - 1u) {
        set_error(error, error_size,
                  "tool declaration size overflow");
        return false;
    }
    const size_t body_size = prefix_size + tools_size + suffix_size;
    char *body = (char *)malloc(body_size + 1u);
    if (body == NULL) {
        set_error(error, error_size,
                  "allocating tool declaration failed");
        return false;
    }
    memcpy(body, prefix, prefix_size);
    memcpy(body + prefix_size, tools_json, tools_size);
    memcpy(body + prefix_size + tools_size, suffix, suffix_size + 1u);
    const bool ok = append_internal_system_message(
        tokenizer, "tool-declare", body,
        output, error, error_size);
    free(body);
    return ok;
}

static const char *json_type_name(k3_json_type type) {
    switch (type) {
        case K3_JSON_STRING: return "string";
        case K3_JSON_NUMBER: return "number";
        case K3_JSON_TRUE:
        case K3_JSON_FALSE: return "boolean";
        case K3_JSON_NULL: return "null";
        case K3_JSON_OBJECT: return "object";
        case K3_JSON_ARRAY: return "array";
        default: return NULL;
    }
}

static bool append_tool_call(
        k3_tokenizer *tokenizer,
        const k3_tool_call *call,
        size_t index,
        k3_token_buffer *output,
        char *error,
        size_t error_size) {
    if (call->name == NULL || call->name[0] == '\0' ||
        call->arguments == NULL) {
        set_error(error, error_size,
                  "tool calls need a name and JSON arguments");
        return false;
    }
    char index_text[32];
    snprintf(index_text, sizeof(index_text), "%zu", index);
    const char *call_keys[] = { "tool", "index" };
    const char *call_values[] = { call->name, index_text };
    if (!open_tag(
            tokenizer, "call", call_keys, call_values, 2u,
            output, error, error_size)) {
        return false;
    }

    k3_json_document document;
    char parse_error[256];
    if (!k3_json_parse(
            &document, call->arguments, strlen(call->arguments),
            parse_error, sizeof(parse_error))) {
        const char *json_keys[] = { "type" };
        const char *json_values[] = { "object" };
        return
            open_tag(
                tokenizer, "json", json_keys, json_values, 1u,
                output, error, error_size) &&
            append_ordinary(
                tokenizer, call->arguments,
                output, error, error_size) &&
            close_tag(
                tokenizer, "json", output,
                error, error_size) &&
            close_tag(
                tokenizer, "call", output,
                error, error_size);
    }
    bool ok = false;
    if (document.tokens[document.root].type != K3_JSON_OBJECT) {
        set_error(error, error_size,
                  "tool call arguments must be a JSON object");
        goto cleanup;
    }
    int32_t key = document.tokens[document.root].first_child;
    ok = true;
    for (size_t i = 0u;
         ok && i < document.tokens[document.root].size;
         i++) {
        if (key < 0) {
            set_error(error, error_size,
                      "malformed tool arguments object");
            ok = false;
            break;
        }
        const int32_t value = document.tokens[key].next_sibling;
        char *decoded_key = NULL;
        char *rendered_value = NULL;
        const char *type = value < 0 ? NULL :
            json_type_name(document.tokens[value].type);
        if (value < 0 || type == NULL ||
            !k3_json_string_dup(
                &document, key, &decoded_key,
                error, error_size)) {
            free(decoded_key);
            ok = false;
            break;
        }
        if (document.tokens[value].type == K3_JSON_STRING) {
            ok = k3_json_string_dup(
                &document, value, &rendered_value,
                error, error_size);
        } else {
            ok = k3_json_compact_sorted_dup(
                &document, value, &rendered_value, NULL,
                error, error_size);
        }
        const char *argument_keys[] = { "key", "type" };
        const char *argument_values[] = { decoded_key, type };
        if (ok) {
            ok = open_tag(
                    tokenizer, "argument",
                    argument_keys, argument_values, 2u,
                    output, error, error_size) &&
                append_ordinary(
                    tokenizer, rendered_value,
                    output, error, error_size) &&
                close_tag(
                    tokenizer, "argument", output,
                    error, error_size);
        }
        free(rendered_value);
        free(decoded_key);
        key = document.tokens[value].next_sibling;
    }
    if (ok) {
        ok = close_tag(
            tokenizer, "call", output,
            error, error_size);
    }

cleanup:
    k3_json_document_free(&document);
    return ok;
}

static bool append_assistant_tool_calls(
        k3_tokenizer *tokenizer,
        const k3_chat_message *message,
        k3_token_buffer *output,
        char *error,
        size_t error_size) {
    if (message->tool_call_count == 0u) {
        return true;
    }
    if (message->tool_calls == NULL ||
        !open_tag(
            tokenizer, "tools", NULL, NULL, 0u,
            output, error, error_size)) {
        if (message->tool_calls == NULL) {
            set_error(error, error_size,
                      "tool call count has no call array");
        }
        return false;
    }
    for (size_t i = 0u; i < message->tool_call_count; i++) {
        if (!append_tool_call(
                tokenizer, &message->tool_calls[i], i + 1u,
                output, error, error_size)) {
            return false;
        }
    }
    return close_tag(
        tokenizer, "tools", output,
        error, error_size);
}

static bool append_tool_result_message(
        k3_tokenizer *tokenizer,
        const k3_chat_message *message,
        size_t index,
        k3_token_buffer *output,
        char *error,
        size_t error_size) {
    if (message->name == NULL || message->name[0] == '\0') {
        set_error(error, error_size,
                  "tool result needs a resolved tool name");
        return false;
    }
    char index_text[32];
    snprintf(index_text, sizeof(index_text), "%zu", index);
    const char *keys[] = { "role", "tool", "index" };
    const char *values[] = { "tool", message->name, index_text };
    return
        open_tag(
            tokenizer, "message", keys, values, 3u,
            output, error, error_size) &&
        append_ordinary(
            tokenizer,
            message->content == NULL ? "" : message->content,
            output, error, error_size) &&
        close_tag(
            tokenizer, "message", output,
            error, error_size) &&
        end_message(output, error, error_size);
}

static bool append_chat_message(
        k3_tokenizer *tokenizer,
        const k3_chat_message *message,
        bool thinking,
        k3_token_buffer *output,
        char *error,
        size_t error_size) {
    const char *role = NULL;
    switch (message->role) {
        case K3_CHAT_ROLE_SYSTEM:
            role = "system";
            break;
        case K3_CHAT_ROLE_USER:
            role = "user";
            break;
        case K3_CHAT_ROLE_ASSISTANT:
            role = "assistant";
            break;
        case K3_CHAT_ROLE_TOOL:
            set_error(error, error_size,
                      "tool messages use the tool-result renderer");
            return false;
        default:
            set_error(error, error_size,
                      "unsupported K3 chat role %d", (int)message->role);
            return false;
    }
    if (!open_message(
            tokenizer, role, message->name, NULL,
            output, error, error_size)) {
        return false;
    }

    const char *content =
        message->content == NULL ? "" : message->content;
    if (message->role == K3_CHAT_ROLE_ASSISTANT) {
        if (thinking) {
            if (!open_tag(
                    tokenizer, "think", NULL, NULL, 0u,
                    output, error, error_size)) {
                return false;
            }
            const char *reasoning = message->reasoning_content;
            if (reasoning != NULL && reasoning[0] != '\0' &&
                !append_ordinary(
                    tokenizer, reasoning, output,
                    error, error_size)) {
                return false;
            }
            if (!close_tag(
                    tokenizer, "think", output,
                    error, error_size)) {
                return false;
            }
        }
        if (!open_tag(
                tokenizer, "response", NULL, NULL, 0u,
                output, error, error_size) ||
            !append_ordinary(
                tokenizer, content, output,
                error, error_size) ||
            !close_tag(
                tokenizer, "response", output,
                error, error_size)) {
            return false;
        }
        if (!append_assistant_tool_calls(
                tokenizer, message, output,
                error, error_size)) {
            return false;
        }
    } else if (!append_ordinary(
                   tokenizer, content, output,
                   error, error_size)) {
        return false;
    }

    return
        close_tag(
            tokenizer, "message", output,
            error, error_size) &&
        end_message(output, error, error_size);
}

bool k3_tokenizer_encode_chat(
        k3_tokenizer *tokenizer,
        const k3_chat_message *messages,
        size_t message_count,
        const k3_chat_options *options,
        k3_token_buffer *output,
        char *error,
        size_t error_size) {
    if (tokenizer == NULL || output == NULL ||
        (messages == NULL && message_count != 0u)) {
        set_error(error, error_size,
                  "chat encoding arguments are invalid");
        return false;
    }
    const k3_chat_options defaults = {
        .add_generation_prompt = true,
        .thinking = false,
        .thinking_effort = NULL,
        .tools_json = NULL,
        .tool_choice = K3_TOOL_CHOICE_AUTO,
    };
    const k3_chat_options *selected =
        options == NULL ? &defaults : options;
    output->count = 0u;

    if (!append_tool_declare(
            tokenizer, selected->tools_json, false,
            output, error, error_size)) {
        return false;
    }
    if (selected->thinking &&
        !append_thinking_effort(
            tokenizer, selected->thinking_effort,
            output, error, error_size)) {
        return false;
    }
    size_t tool_result_index = 0u;
    for (size_t i = 0u; i < message_count; i++) {
        if (messages[i].role == K3_CHAT_ROLE_SYSTEM &&
            messages[i].tools_json != NULL) {
            if (!append_tool_declare(
                    tokenizer, messages[i].tools_json, true,
                    output, error, error_size)) {
                return false;
            }
            continue;
        }
        if (messages[i].role == K3_CHAT_ROLE_ASSISTANT) {
            tool_result_index = 0u;
        }
        if (messages[i].role == K3_CHAT_ROLE_TOOL) {
            tool_result_index++;
            if (!append_tool_result_message(
                    tokenizer, &messages[i], tool_result_index,
                    output, error, error_size)) {
                return false;
            }
        } else if (!append_chat_message(
                       tokenizer, &messages[i], selected->thinking,
                       output, error, error_size)) {
            return false;
        }
    }
    if (selected->tool_choice == K3_TOOL_CHOICE_REQUIRED &&
        !append_internal_system_message(
            tokenizer, "tool-choice",
            "The system is invoked with `tool_choice=required`.\n"
            "You MUST call tools in the next message.",
            output, error, error_size)) {
        return false;
    }
    if (selected->tool_choice == K3_TOOL_CHOICE_NONE &&
        !append_internal_system_message(
            tokenizer, "tool-choice",
            "The system is invoked with `tool_choice=none`.\n"
            "You MUST NOT call any tools in the next message.",
            output, error, error_size)) {
        return false;
    }
    if (!selected->add_generation_prompt) {
        return true;
    }
    if (!open_message(
            tokenizer, "assistant", NULL, NULL,
            output, error, error_size)) {
        return false;
    }
    return open_tag(
        tokenizer, selected->thinking ? "think" : "response",
        NULL, NULL, 0u, output, error, error_size);
}

typedef struct {
    const k3_tokenizer *tokenizer;
    const uint32_t     *tokens;
    size_t              count;
    size_t              position;
} k3_xtml_cursor;

static bool decode_range(
        const k3_xtml_cursor *cursor,
        size_t begin,
        size_t end,
        k3_text_buffer *text,
        char *error,
        size_t error_size) {
    if (begin > end || end > cursor->count) {
        set_error(error, error_size,
                  "invalid XTML token range");
        return false;
    }
    for (size_t i = begin; i < end; i++) {
        if (cursor->tokens[i] >= K3_TOKEN_BOS) {
            set_error(error, error_size,
                      "unexpected control token inside XTML text");
            return false;
        }
    }
    return k3_tokenizer_decode(
        cursor->tokenizer, cursor->tokens + begin, end - begin,
        false, text, error, error_size);
}

static bool consume_tag(
        k3_xtml_cursor *cursor,
        uint32_t marker,
        const char *expected_tag,
        char **header,
        char *error,
        size_t error_size) {
    if (header != NULL) {
        *header = NULL;
    }
    if (cursor->position >= cursor->count ||
        cursor->tokens[cursor->position] != marker) {
        set_error(error, error_size,
                  "expected XTML %s tag %s",
                  marker == K3_TOKEN_OPEN ? "open" : "close",
                  expected_tag);
        return false;
    }
    const size_t begin = ++cursor->position;
    while (cursor->position < cursor->count &&
           cursor->tokens[cursor->position] != K3_TOKEN_SEP) {
        if (cursor->tokens[cursor->position] >= K3_TOKEN_BOS) {
            set_error(error, error_size,
                      "malformed XTML tag header");
            return false;
        }
        cursor->position++;
    }
    if (cursor->position >= cursor->count) {
        set_error(error, error_size,
                  "unterminated XTML tag header");
        return false;
    }
    k3_text_buffer decoded = { 0 };
    if (!decode_range(
            cursor, begin, cursor->position,
            &decoded, error, error_size)) {
        k3_text_buffer_free(&decoded);
        return false;
    }
    cursor->position++;
    const size_t tag_size = strlen(expected_tag);
    const bool matches = decoded.size >= tag_size &&
        memcmp(decoded.data, expected_tag, tag_size) == 0 &&
        (decoded.size == tag_size || decoded.data[tag_size] == ' ');
    if (!matches) {
        set_error(error, error_size,
                  "expected XTML tag %s, got %s",
                  expected_tag,
                  decoded.data == NULL ? "" : decoded.data);
        k3_text_buffer_free(&decoded);
        return false;
    }
    if (header != NULL) {
        *header = decoded.data;
        decoded.data = NULL;
    }
    k3_text_buffer_free(&decoded);
    return true;
}

static bool next_open_tag_is(
        const k3_xtml_cursor *cursor,
        const char *tag) {
    if (cursor->position >= cursor->count ||
        cursor->tokens[cursor->position] != K3_TOKEN_OPEN) {
        return false;
    }
    size_t end = cursor->position + 1u;
    while (end < cursor->count &&
           cursor->tokens[end] < K3_TOKEN_BOS) {
        end++;
    }
    if (end >= cursor->count ||
        cursor->tokens[end] != K3_TOKEN_SEP) {
        return false;
    }
    k3_text_buffer header = { 0 };
    if (!decode_range(
            cursor, cursor->position + 1u, end,
            &header, NULL, 0u)) {
        k3_text_buffer_free(&header);
        return false;
    }
    const size_t tag_size = strlen(tag);
    const bool matches = header.size >= tag_size &&
        memcmp(header.data, tag, tag_size) == 0 &&
        (header.size == tag_size || header.data[tag_size] == ' ');
    k3_text_buffer_free(&header);
    return matches;
}

static char *xtml_attr_dup(
        const char *header,
        const char *key,
        char *error,
        size_t error_size) {
    const size_t key_size = strlen(key);
    const char *cursor = header;
    while ((cursor = strchr(cursor, ' ')) != NULL) {
        cursor++;
        if (strlen(cursor) < key_size + 2u ||
            strncmp(cursor, key, key_size) != 0 ||
            cursor[key_size] != '=' ||
            cursor[key_size + 1u] != '"') {
            continue;
        }
        const char *value = cursor + key_size + 2u;
        const char *end = strchr(value, '"');
        if (end == NULL) {
            break;
        }
        k3_text_buffer decoded = { 0 };
        const char *part = value;
        while (part < end) {
            if (*part != '&') {
                if (!append_text_bytes(
                        &decoded, part, 1u,
                        error, error_size)) {
                    k3_text_buffer_free(&decoded);
                    return NULL;
                }
                part++;
                continue;
            }
            const char *replacement = NULL;
            size_t consumed = 0u;
            if ((size_t)(end - part) >= 5u &&
                memcmp(part, "&amp;", 5u) == 0) {
                replacement = "&";
                consumed = 5u;
            } else if ((size_t)(end - part) >= 6u &&
                       memcmp(part, "&quot;", 6u) == 0) {
                replacement = "\"";
                consumed = 6u;
            } else {
                set_error(error, error_size,
                          "unsupported XTML attribute escape");
                k3_text_buffer_free(&decoded);
                return NULL;
            }
            if (!append_text_bytes(
                    &decoded, replacement, 1u,
                    error, error_size)) {
                k3_text_buffer_free(&decoded);
                return NULL;
            }
            part += consumed;
        }
        if (decoded.data == NULL) {
            decoded.data = strdup("");
            if (decoded.data == NULL) {
                set_error(error, error_size,
                          "allocating XTML attribute failed");
                return NULL;
            }
        }
        return decoded.data;
    }
    set_error(error, error_size,
              "XTML tag lacks attribute %s", key);
    return NULL;
}

static bool append_json_argument(
        k3_text_buffer *arguments,
        const char *key,
        const char *type,
        const char *value,
        bool first,
        char *error,
        size_t error_size) {
    char *escaped_key = NULL;
    char *rendered_value = NULL;
    size_t escaped_key_size = 0u;
    size_t rendered_value_size = 0u;
    bool ok = k3_json_escape(
        key, strlen(key), &escaped_key, &escaped_key_size,
        error, error_size);
    if (ok && strcmp(type, "string") == 0) {
        ok = k3_json_escape(
            value, strlen(value),
            &rendered_value, &rendered_value_size,
            error, error_size);
    } else if (ok) {
        k3_json_document document;
        memset(&document, 0, sizeof(document));
        ok = k3_json_parse(
            &document, value, strlen(value),
            error, error_size);
        k3_json_type expected = K3_JSON_NULL;
        if (strcmp(type, "number") == 0) {
            expected = K3_JSON_NUMBER;
        } else if (strcmp(type, "boolean") == 0) {
            if (ok && document.tokens[document.root].type != K3_JSON_TRUE &&
                document.tokens[document.root].type != K3_JSON_FALSE) {
                ok = false;
            }
            expected = K3_JSON_TRUE;
        } else if (strcmp(type, "null") == 0) {
            expected = K3_JSON_NULL;
        } else if (strcmp(type, "object") == 0) {
            expected = K3_JSON_OBJECT;
        } else if (strcmp(type, "array") == 0) {
            expected = K3_JSON_ARRAY;
        } else {
            ok = false;
        }
        if (ok && expected != K3_JSON_TRUE &&
            document.tokens[document.root].type != expected) {
            ok = false;
        }
        if (!ok && error != NULL && error_size != 0u) {
            set_error(error, error_size,
                      "invalid XTML %s argument value", type);
        }
        if (ok) {
            ok = k3_json_compact_sorted_dup(
                &document, document.root,
                &rendered_value, &rendered_value_size,
                error, error_size);
        }
        k3_json_document_free(&document);
    }
    if (ok) {
        ok = (first || append_text_bytes(
                arguments, ",", 1u,
                error, error_size)) &&
            append_text_bytes(
                arguments, escaped_key, escaped_key_size,
                error, error_size) &&
            append_text_bytes(
                arguments, ":", 1u,
                error, error_size) &&
            append_text_bytes(
                arguments, rendered_value, rendered_value_size,
                error, error_size);
    }
    free(rendered_value);
    free(escaped_key);
    return ok;
}

static bool append_output_call(
        k3_assistant_output *output,
        char *name,
        char *arguments,
        char *error,
        size_t error_size) {
    if (output->tool_call_count == SIZE_MAX /
            sizeof(*output->tool_calls)) {
        set_error(error, error_size,
                  "too many generated tool calls");
        return false;
    }
    const size_t count = output->tool_call_count + 1u;
    k3_tool_call *calls = (k3_tool_call *)realloc(
        output->tool_calls, count * sizeof(*calls));
    if (calls == NULL) {
        set_error(error, error_size,
                  "allocating generated tool calls failed");
        return false;
    }
    output->tool_calls = calls;
    output->tool_calls[output->tool_call_count] = (k3_tool_call) {
        .name = name,
        .arguments = arguments,
    };
    output->tool_call_count = count;
    return true;
}

void k3_assistant_output_free(k3_assistant_output *output) {
    if (output == NULL) {
        return;
    }
    k3_text_buffer_free(&output->response);
    for (size_t i = 0u; i < output->tool_call_count; i++) {
        free((char *)output->tool_calls[i].id);
        free((char *)output->tool_calls[i].name);
        free((char *)output->tool_calls[i].arguments);
    }
    free(output->tool_calls);
    memset(output, 0, sizeof(*output));
}

bool k3_tokenizer_parse_assistant_output(
        const k3_tokenizer *tokenizer,
        const uint32_t *tokens,
        size_t token_count,
        k3_assistant_output *output,
        char *error,
        size_t error_size) {
    if (tokenizer == NULL || output == NULL ||
        (tokens == NULL && token_count != 0u)) {
        set_error(error, error_size,
                  "assistant output parse arguments are invalid");
        return false;
    }
    memset(output, 0, sizeof(*output));
    k3_xtml_cursor cursor = {
        .tokenizer = tokenizer,
        .tokens = tokens,
        .count = token_count,
    };
    while (cursor.position < cursor.count &&
           cursor.tokens[cursor.position] < K3_TOKEN_BOS) {
        cursor.position++;
    }
    if (!decode_range(
            &cursor, 0u, cursor.position,
            &output->response, error, error_size) ||
        !consume_tag(
            &cursor, K3_TOKEN_CLOSE, "response", NULL,
            error, error_size)) {
        goto fail;
    }

    if (cursor.position < cursor.count &&
        cursor.tokens[cursor.position] == K3_TOKEN_OPEN) {
        if (!consume_tag(
                &cursor, K3_TOKEN_OPEN, "tools", NULL,
                error, error_size)) {
            goto fail;
        }
        for (;;) {
            if (cursor.position >= cursor.count) {
                set_error(error, error_size,
                          "unterminated XTML tools block");
                goto fail;
            }
            if (cursor.tokens[cursor.position] == K3_TOKEN_CLOSE) {
                if (!consume_tag(
                        &cursor, K3_TOKEN_CLOSE, "tools", NULL,
                        error, error_size)) {
                    goto fail;
                }
                break;
            }
            char *call_header = NULL;
            char *name = NULL;
            char *index_text = NULL;
            if (!consume_tag(
                    &cursor, K3_TOKEN_OPEN, "call", &call_header,
                    error, error_size)) {
                free(call_header);
                goto fail;
            }
            name = xtml_attr_dup(
                call_header, "tool", error, error_size);
            index_text = xtml_attr_dup(
                call_header, "index", error, error_size);
            free(call_header);
            char expected_index[32];
            snprintf(expected_index, sizeof(expected_index), "%zu",
                     output->tool_call_count + 1u);
            if (name == NULL || index_text == NULL ||
                strcmp(index_text, expected_index) != 0) {
                if (name != NULL && index_text != NULL) {
                    set_error(error, error_size,
                              "XTML tool call index is out of order");
                }
                free(index_text);
                free(name);
                goto fail;
            }
            free(index_text);

            k3_text_buffer arguments = { 0 };
            if (!append_text_bytes(
                    &arguments, "{", 1u,
                    error, error_size)) {
                free(name);
                goto fail;
            }
            size_t argument_count = 0u;
            bool raw_json = false;
            if (next_open_tag_is(&cursor, "json")) {
                char *json_header = NULL;
                if (!consume_tag(
                        &cursor, K3_TOKEN_OPEN, "json", &json_header,
                        error, error_size)) {
                    free(json_header);
                    free(name);
                    k3_text_buffer_free(&arguments);
                    goto fail;
                }
                char *json_type = xtml_attr_dup(
                    json_header, "type", error, error_size);
                free(json_header);
                const size_t json_begin = cursor.position;
                while (cursor.position < cursor.count &&
                       cursor.tokens[cursor.position] < K3_TOKEN_BOS) {
                    cursor.position++;
                }
                k3_text_buffer raw = { 0 };
                const bool decoded = json_type != NULL &&
                    strcmp(json_type, "object") == 0 &&
                    decode_range(
                        &cursor, json_begin, cursor.position,
                        &raw, error, error_size) &&
                    consume_tag(
                        &cursor, K3_TOKEN_CLOSE, "json", NULL,
                        error, error_size);
                free(json_type);
                if (!decoded) {
                    free(name);
                    k3_text_buffer_free(&raw);
                    k3_text_buffer_free(&arguments);
                    goto fail;
                }
                k3_text_buffer_free(&arguments);
                arguments = raw;
                raw_json = true;
            }
            while (!raw_json &&
                   cursor.position < cursor.count &&
                   cursor.tokens[cursor.position] == K3_TOKEN_OPEN) {
                char *header = NULL;
                if (!consume_tag(
                        &cursor, K3_TOKEN_OPEN, "argument", &header,
                        error, error_size)) {
                    free(header);
                    free(name);
                    k3_text_buffer_free(&arguments);
                    goto fail;
                }
                char *key = xtml_attr_dup(
                    header, "key", error, error_size);
                char *type = xtml_attr_dup(
                    header, "type", error, error_size);
                free(header);
                const size_t value_begin = cursor.position;
                while (cursor.position < cursor.count &&
                       cursor.tokens[cursor.position] < K3_TOKEN_BOS) {
                    cursor.position++;
                }
                k3_text_buffer value = { 0 };
                const bool decoded = key != NULL && type != NULL &&
                    decode_range(
                        &cursor, value_begin, cursor.position,
                        &value, error, error_size) &&
                    consume_tag(
                        &cursor, K3_TOKEN_CLOSE, "argument", NULL,
                        error, error_size) &&
                    append_json_argument(
                        &arguments, key, type,
                        value.data == NULL ? "" : value.data,
                        argument_count == 0u,
                        error, error_size);
                k3_text_buffer_free(&value);
                free(type);
                free(key);
                if (!decoded) {
                    free(name);
                    k3_text_buffer_free(&arguments);
                    goto fail;
                }
                argument_count++;
            }
            if (!raw_json &&
                !append_text_bytes(
                    &arguments, "}", 1u,
                    error, error_size)) {
                free(name);
                k3_text_buffer_free(&arguments);
                goto fail;
            }
            if (!consume_tag(
                    &cursor, K3_TOKEN_CLOSE, "call", NULL,
                    error, error_size) ||
                !append_output_call(
                    output, name, arguments.data,
                    error, error_size)) {
                free(name);
                k3_text_buffer_free(&arguments);
                goto fail;
            }
            arguments.data = NULL;
            k3_text_buffer_free(&arguments);
        }
    }
    if (!consume_tag(
            &cursor, K3_TOKEN_CLOSE, "message", NULL,
            error, error_size) ||
        cursor.position >= cursor.count ||
        cursor.tokens[cursor.position] != K3_TOKEN_END_OF_MSG) {
        set_error(error, error_size,
                  "assistant XTML lacks end_of_msg");
        goto fail;
    }
    cursor.position++;
    if (cursor.position != cursor.count) {
        set_error(error, error_size,
                  "trailing tokens after assistant XTML message");
        goto fail;
    }
    return true;

fail:
    k3_assistant_output_free(output);
    return false;
}
