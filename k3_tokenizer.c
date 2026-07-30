#include "k3_tokenizer.h"

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
    };
    const k3_chat_options *selected =
        options == NULL ? &defaults : options;
    output->count = 0u;

    if (selected->thinking &&
        !append_thinking_effort(
            tokenizer, selected->thinking_effort,
            output, error, error_size)) {
        return false;
    }
    for (size_t i = 0u; i < message_count; i++) {
        if (!append_chat_message(
                tokenizer, &messages[i], selected->thinking,
                output, error, error_size)) {
            return false;
        }
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
