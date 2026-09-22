#include "mimo26_tokenizer.h"

#include "k3_json.h"

#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/uregex.h>
#include <unicode/utext.h>
#include <unicode/utypes.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * The pre-tokenizer split, taken verbatim from tokenizer.json's Split
 * pattern. ICU's dialect accepts it unchanged, which is the whole reason
 * this reuses K3's mechanism rather than hand-rolling Unicode categories.
 */
#define MIMO26_PRETOKENIZER_PATTERN \
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|" \
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

#define MAX_PIECE_BYTES 512u

typedef struct {
    char    *key;      /* byte-level spelling */
    uint32_t id;
    bool     occupied;
} vocab_entry;

typedef struct {
    char    *pair;     /* "left\0right" concatenated with a NUL between */
    size_t   left_len;
    size_t   total_len;
    uint32_t rank;
    bool     occupied;
} merge_entry;

typedef struct {
    char    *content;
    uint32_t id;
    bool     special;
} added_token;

struct mimo26_tokenizer {
    vocab_entry *vocab;        /* open-addressed */
    size_t       vocab_mask;
    uint32_t     vocab_count;

    merge_entry *merges;
    size_t       merge_mask;

    char       **id_to_piece;  /* byte-level spelling per id, or NULL */
    uint32_t     id_capacity;

    added_token *added;
    size_t       added_count;

    URegularExpression *pretokenizer;
    const UNormalizer2 *nfc;

    /* GPT-2 byte-level alphabet, both directions. */
    char     byte_to_unicode[256][4];
    uint8_t  byte_to_unicode_len[256];
    int16_t  unicode_to_byte[512];   /* code point -> byte, -1 when unused */
};

static void set_error(char *error, size_t size, const char *format, ...)
{
    if (error == NULL || size == 0) {
        return;
    }
    va_list args;
    va_start(args, format);
    vsnprintf(error, size, format, args);
    va_end(args);
}

void mimo26_token_buffer_free(mimo26_token_buffer *buffer)
{
    if (buffer != NULL) {
        free(buffer->ids);
        buffer->ids = NULL;
        buffer->count = 0;
        buffer->capacity = 0;
    }
}

static bool buffer_push(mimo26_token_buffer *buffer, uint32_t id)
{
    if (buffer->count == buffer->capacity) {
        const size_t grown = buffer->capacity ? buffer->capacity * 2u : 64u;
        uint32_t *ids = realloc(buffer->ids, grown * sizeof *ids);
        if (ids == NULL) {
            return false;
        }
        buffer->ids = ids;
        buffer->capacity = grown;
    }
    buffer->ids[buffer->count++] = id;
    return true;
}

/* FNV-1a over a byte range. */
static uint64_t hash_bytes(const char *data, size_t length)
{
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < length; i++) {
        hash ^= (unsigned char)data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

/*
 * The GPT-2 byte-level alphabet: every byte maps to a printable code point
 * so that BPE operates on text, and the mapping is reversible. Bytes that
 * are already printable ASCII (and two Latin-1 ranges) map to themselves;
 * the remaining 68 map to U+0100 upward in order.
 */
static void build_byte_alphabet(mimo26_tokenizer *tokenizer)
{
    for (size_t i = 0; i < 512; i++) {
        tokenizer->unicode_to_byte[i] = -1;
    }
    int next = 0;
    for (int byte = 0; byte < 256; byte++) {
        int code;
        if ((byte >= '!' && byte <= '~') || (byte >= 0xA1 && byte <= 0xAC) ||
            (byte >= 0xAE && byte <= 0xFF)) {
            code = byte;
        } else {
            code = 256 + next;
            next++;
        }
        /* Encode `code` as UTF-8 into the forward table. */
        char *slot = tokenizer->byte_to_unicode[byte];
        if (code < 0x80) {
            slot[0] = (char)code;
            tokenizer->byte_to_unicode_len[byte] = 1u;
        } else if (code < 0x800) {
            slot[0] = (char)(0xC0 | (code >> 6));
            slot[1] = (char)(0x80 | (code & 0x3F));
            tokenizer->byte_to_unicode_len[byte] = 2u;
        } else {
            slot[0] = (char)(0xE0 | (code >> 12));
            slot[1] = (char)(0x80 | ((code >> 6) & 0x3F));
            slot[2] = (char)(0x80 | (code & 0x3F));
            tokenizer->byte_to_unicode_len[byte] = 3u;
        }
        tokenizer->unicode_to_byte[code] = (int16_t)byte;
    }
}

/* Decode one UTF-8 code point; returns bytes consumed or 0. */
static size_t utf8_next(const char *text, size_t available, uint32_t *code)
{
    const unsigned char *p = (const unsigned char *)text;
    if (available == 0) {
        return 0;
    }
    if (p[0] < 0x80) {
        *code = p[0];
        return 1;
    }
    if ((p[0] & 0xE0) == 0xC0 && available >= 2) {
        *code = (uint32_t)((p[0] & 0x1F) << 6 | (p[1] & 0x3F));
        return 2;
    }
    if ((p[0] & 0xF0) == 0xE0 && available >= 3) {
        *code = (uint32_t)((p[0] & 0x0F) << 12 | (p[1] & 0x3F) << 6 |
                           (p[2] & 0x3F));
        return 3;
    }
    if ((p[0] & 0xF8) == 0xF0 && available >= 4) {
        *code = (uint32_t)((p[0] & 0x07) << 18 | (p[1] & 0x3F) << 12 |
                           (p[2] & 0x3F) << 6 | (p[3] & 0x3F));
        return 4;
    }
    return 0;
}

/* ---- vocabulary and merge tables ---- */

static bool vocab_insert(mimo26_tokenizer *tokenizer, const char *key,
                         size_t length, uint32_t id)
{
    size_t index = (size_t)hash_bytes(key, length) & tokenizer->vocab_mask;
    while (tokenizer->vocab[index].occupied) {
        index = (index + 1u) & tokenizer->vocab_mask;
    }
    char *copy = malloc(length + 1u);
    if (copy == NULL) {
        return false;
    }
    memcpy(copy, key, length);
    copy[length] = '\0';
    tokenizer->vocab[index].key = copy;
    tokenizer->vocab[index].id = id;
    tokenizer->vocab[index].occupied = true;
    tokenizer->vocab_count++;
    return true;
}

static bool vocab_lookup(const mimo26_tokenizer *tokenizer, const char *key,
                         size_t length, uint32_t *id)
{
    size_t index = (size_t)hash_bytes(key, length) & tokenizer->vocab_mask;
    while (tokenizer->vocab[index].occupied) {
        const char *candidate = tokenizer->vocab[index].key;
        if (strlen(candidate) == length &&
            memcmp(candidate, key, length) == 0) {
            *id = tokenizer->vocab[index].id;
            return true;
        }
        index = (index + 1u) & tokenizer->vocab_mask;
    }
    return false;
}

static bool merge_insert(mimo26_tokenizer *tokenizer, const char *left,
                         size_t left_len, const char *right, size_t right_len,
                         uint32_t rank)
{
    const size_t total = left_len + 1u + right_len;
    char *joined = malloc(total + 1u);
    if (joined == NULL) {
        return false;
    }
    memcpy(joined, left, left_len);
    joined[left_len] = '\0';
    memcpy(joined + left_len + 1u, right, right_len);
    joined[total] = '\0';

    size_t index = (size_t)hash_bytes(joined, total) & tokenizer->merge_mask;
    while (tokenizer->merges[index].occupied) {
        index = (index + 1u) & tokenizer->merge_mask;
    }
    tokenizer->merges[index].pair = joined;
    tokenizer->merges[index].left_len = left_len;
    tokenizer->merges[index].total_len = total;
    tokenizer->merges[index].rank = rank;
    tokenizer->merges[index].occupied = true;
    return true;
}

static bool merge_rank(const mimo26_tokenizer *tokenizer, const char *left,
                       size_t left_len, const char *right, size_t right_len,
                       uint32_t *rank)
{
    char probe[MAX_PIECE_BYTES * 2u + 2u];
    const size_t total = left_len + 1u + right_len;
    if (total > sizeof probe) {
        return false;
    }
    memcpy(probe, left, left_len);
    probe[left_len] = '\0';
    memcpy(probe + left_len + 1u, right, right_len);

    size_t index = (size_t)hash_bytes(probe, total) & tokenizer->merge_mask;
    while (tokenizer->merges[index].occupied) {
        const merge_entry *entry = &tokenizer->merges[index];
        if (entry->total_len == total &&
            memcmp(entry->pair, probe, total) == 0) {
            *rank = entry->rank;
            return true;
        }
        index = (index + 1u) & tokenizer->merge_mask;
    }
    return false;
}

/* ---- BPE over one pre-token ---- */

typedef struct {
    const char *start;
    size_t      length;
} symbol;

/*
 * Standard byte-level BPE: start from single code points and repeatedly
 * merge the adjacent pair with the lowest rank. Quadratic in the number of
 * symbols, which is fine because a pre-token is bounded by the split regex
 * and is short in practice.
 */
static bool bpe_encode_piece(const mimo26_tokenizer *tokenizer,
                             const char *piece, size_t length,
                             mimo26_token_buffer *out, char *error,
                             size_t error_size)
{
    symbol symbols[MAX_PIECE_BYTES];
    size_t count = 0;
    size_t offset = 0;
    while (offset < length) {
        uint32_t code = 0;
        const size_t step = utf8_next(piece + offset, length - offset, &code);
        if (step == 0 || count == MAX_PIECE_BYTES) {
            set_error(error, error_size, "pre-token is malformed or too long");
            return false;
        }
        symbols[count].start = piece + offset;
        symbols[count].length = step;
        count++;
        offset += step;
    }

    while (count > 1u) {
        uint32_t best_rank = UINT32_MAX;
        size_t best = SIZE_MAX;
        for (size_t i = 0; i + 1u < count; i++) {
            uint32_t rank = 0;
            if (merge_rank(tokenizer, symbols[i].start, symbols[i].length,
                           symbols[i + 1u].start, symbols[i + 1u].length,
                           &rank) &&
                rank < best_rank) {
                best_rank = rank;
                best = i;
            }
        }
        if (best == SIZE_MAX) {
            break;
        }
        /* Merging adjacent symbols keeps them contiguous in the source, so
         * the result is still one span rather than a copy. */
        symbols[best].length += symbols[best + 1u].length;
        for (size_t i = best + 1u; i + 1u < count; i++) {
            symbols[i] = symbols[i + 1u];
        }
        count--;
    }

    for (size_t i = 0; i < count; i++) {
        uint32_t id = 0;
        if (!vocab_lookup(tokenizer, symbols[i].start, symbols[i].length,
                          &id)) {
            set_error(error, error_size,
                      "no vocabulary entry for a byte-level piece");
            return false;
        }
        if (!buffer_push(out, id)) {
            set_error(error, error_size, "out of memory encoding");
            return false;
        }
    }
    return true;
}

/*
 * NFC-normalize a span. tokenizer.json declares {"type": "NFC"}, so this is
 * not optional decoration: without it "e" followed by a combining acute
 * encodes as three tokens where the reference produces one, and every id
 * after that point differs. Note the consequence -- encoding is not
 * round-trip faithful for decomposed input, because the reference is not
 * either; decode returns the composed form.
 */
static bool normalize_nfc(mimo26_tokenizer *tokenizer, const char *text,
                          size_t length, char **out, size_t *out_length,
                          char *error, size_t error_size)
{
    UErrorCode status = U_ZERO_ERROR;
    int32_t utf16_length = 0;
    u_strFromUTF8(NULL, 0, &utf16_length, text, (int32_t)length, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        set_error(error, error_size, "u_strFromUTF8: %s", u_errorName(status));
        return false;
    }
    status = U_ZERO_ERROR;
    UChar *utf16 = malloc(((size_t)utf16_length + 1u) * sizeof *utf16);
    if (utf16 == NULL) {
        set_error(error, error_size, "out of memory normalizing");
        return false;
    }
    u_strFromUTF8(utf16, utf16_length + 1, &utf16_length, text,
                  (int32_t)length, &status);
    if (U_FAILURE(status)) {
        free(utf16);
        set_error(error, error_size, "u_strFromUTF8: %s", u_errorName(status));
        return false;
    }

    status = U_ZERO_ERROR;
    int32_t composed_length =
        unorm2_normalize(tokenizer->nfc, utf16, utf16_length, NULL, 0,
                         &status);
    status = U_ZERO_ERROR;
    UChar *composed = malloc(((size_t)composed_length + 1u) * sizeof *composed);
    if (composed == NULL) {
        free(utf16);
        set_error(error, error_size, "out of memory normalizing");
        return false;
    }
    composed_length = unorm2_normalize(tokenizer->nfc, utf16, utf16_length,
                                       composed, composed_length + 1,
                                       &status);
    free(utf16);
    if (U_FAILURE(status)) {
        free(composed);
        set_error(error, error_size, "unorm2_normalize: %s",
                  u_errorName(status));
        return false;
    }

    status = U_ZERO_ERROR;
    int32_t utf8_length = 0;
    u_strToUTF8(NULL, 0, &utf8_length, composed, composed_length, &status);
    status = U_ZERO_ERROR;
    char *utf8 = malloc((size_t)utf8_length + 1u);
    if (utf8 == NULL) {
        free(composed);
        set_error(error, error_size, "out of memory normalizing");
        return false;
    }
    u_strToUTF8(utf8, utf8_length + 1, &utf8_length, composed,
                composed_length, &status);
    free(composed);
    if (U_FAILURE(status)) {
        free(utf8);
        set_error(error, error_size, "u_strToUTF8: %s", u_errorName(status));
        return false;
    }
    *out = utf8;
    *out_length = (size_t)utf8_length;
    return true;
}

/* Encode a plain span of UTF-8 with no special-token handling. */
static bool encode_span(mimo26_tokenizer *tokenizer, const char *raw,
                        size_t raw_length, mimo26_token_buffer *out,
                        char *error, size_t error_size)
{
    if (raw_length == 0) {
        return true;
    }
    char *normalized = NULL;
    size_t length = 0;
    if (!normalize_nfc(tokenizer, raw, raw_length, &normalized, &length,
                       error, error_size)) {
        return false;
    }
    const char *text = normalized;
    if (length == 0) {
        free(normalized);
        return true;
    }
    UErrorCode status = U_ZERO_ERROR;
    UText subject = UTEXT_INITIALIZER;
    utext_openUTF8(&subject, text, (int64_t)length, &status);
    if (U_FAILURE(status)) {
        free(normalized);
        set_error(error, error_size, "utext_openUTF8: %s",
                  u_errorName(status));
        return false;
    }
    uregex_setUText(tokenizer->pretokenizer, &subject, &status);
    if (U_FAILURE(status)) {
        utext_close(&subject);
        free(normalized);
        set_error(error, error_size, "uregex_setUText: %s",
                  u_errorName(status));
        return false;
    }

    bool ok = true;
    char mapped[MAX_PIECE_BYTES];
    while (ok && uregex_findNext(tokenizer->pretokenizer, &status) &&
           U_SUCCESS(status)) {
        const int64_t begin = uregex_start64(tokenizer->pretokenizer, 0,
                                             &status);
        const int64_t end = uregex_end64(tokenizer->pretokenizer, 0, &status);
        if (U_FAILURE(status) || end < begin) {
            ok = false;
            set_error(error, error_size, "pre-tokenizer match failed");
            break;
        }
        /* Map the raw bytes through the byte-level alphabet before BPE. */
        size_t mapped_length = 0;
        for (int64_t i = begin; i < end; i++) {
            const unsigned char byte = (unsigned char)text[i];
            const uint8_t width = tokenizer->byte_to_unicode_len[byte];
            if (mapped_length + width > sizeof mapped) {
                ok = false;
                set_error(error, error_size, "pre-token exceeds the bound");
                break;
            }
            memcpy(mapped + mapped_length, tokenizer->byte_to_unicode[byte],
                   width);
            mapped_length += width;
        }
        if (ok) {
            ok = bpe_encode_piece(tokenizer, mapped, mapped_length, out,
                                  error, error_size);
        }
    }
    utext_close(&subject);
    free(normalized);
    return ok;
}

bool mimo26_tokenizer_encode(mimo26_tokenizer *tokenizer, const char *text,
                             bool allow_special, mimo26_token_buffer *out,
                             char *error, size_t error_size)
{
    if (tokenizer == NULL || text == NULL || out == NULL) {
        set_error(error, error_size, "invalid encode arguments");
        return false;
    }
    out->count = 0;
    const size_t length = strlen(text);
    if (!allow_special) {
        return encode_span(tokenizer, text, length, out, error, error_size);
    }
    /*
     * Scan for added tokens and encode the gaps between them. Longest match
     * wins so that a marker which is a prefix of another cannot shadow it.
     */
    size_t cursor = 0;
    size_t span_start = 0;
    while (cursor < length) {
        const added_token *found = NULL;
        size_t found_length = 0;
        for (size_t a = 0; a < tokenizer->added_count; a++) {
            const size_t marker_length = strlen(tokenizer->added[a].content);
            if (marker_length > found_length &&
                cursor + marker_length <= length &&
                memcmp(text + cursor, tokenizer->added[a].content,
                       marker_length) == 0) {
                found = &tokenizer->added[a];
                found_length = marker_length;
            }
        }
        if (found == NULL) {
            cursor++;
            continue;
        }
        if (!encode_span(tokenizer, text + span_start, cursor - span_start,
                         out, error, error_size) ||
            !buffer_push(out, found->id)) {
            return false;
        }
        cursor += found_length;
        span_start = cursor;
    }
    return encode_span(tokenizer, text + span_start, length - span_start, out,
                       error, error_size);
}

/* ---- decode ---- */

/* Append one id's raw bytes to `sink`, returning false when it will not
 * fit. Reverses the byte-level alphabet. */
static bool append_token_bytes(const mimo26_tokenizer *tokenizer, uint32_t id,
                               char *sink, size_t capacity, size_t *used,
                               bool include_special)
{
    if (id >= tokenizer->id_capacity || tokenizer->id_to_piece[id] == NULL) {
        return true;   /* padding row: decodes to nothing */
    }
    for (size_t a = 0; a < tokenizer->added_count; a++) {
        if (tokenizer->added[a].id == id) {
            if (!include_special) {
                return true;
            }
            const size_t length = strlen(tokenizer->added[a].content);
            if (*used + length > capacity) {
                return false;
            }
            memcpy(sink + *used, tokenizer->added[a].content, length);
            *used += length;
            return true;
        }
    }
    const char *piece = tokenizer->id_to_piece[id];
    const size_t piece_length = strlen(piece);
    size_t offset = 0;
    while (offset < piece_length) {
        uint32_t code = 0;
        const size_t step = utf8_next(piece + offset, piece_length - offset,
                                      &code);
        if (step == 0 || code >= 512u ||
            tokenizer->unicode_to_byte[code] < 0) {
            return false;
        }
        if (*used + 1u > capacity) {
            return false;
        }
        sink[(*used)++] = (char)tokenizer->unicode_to_byte[code];
        offset += step;
    }
    return true;
}

char *mimo26_tokenizer_decode(const mimo26_tokenizer *tokenizer,
                              const uint32_t *ids, size_t count,
                              bool include_special, char *error,
                              size_t error_size)
{
    if (tokenizer == NULL || (ids == NULL && count > 0)) {
        set_error(error, error_size, "invalid decode arguments");
        return NULL;
    }
    size_t capacity = count * 8u + 64u;
    char *sink = malloc(capacity);
    if (sink == NULL) {
        set_error(error, error_size, "out of memory decoding");
        return NULL;
    }
    size_t used = 0;
    for (size_t i = 0; i < count; i++) {
        while (!append_token_bytes(tokenizer, ids[i], sink, capacity, &used,
                                   include_special)) {
            if (used + 64u <= capacity) {
                /* Not a capacity problem: the piece is unmappable. */
                free(sink);
                set_error(error, error_size,
                          "token %u does not decode to bytes", ids[i]);
                return NULL;
            }
            capacity *= 2u;
            char *grown = realloc(sink, capacity);
            if (grown == NULL) {
                free(sink);
                set_error(error, error_size, "out of memory decoding");
                return NULL;
            }
            sink = grown;
        }
    }
    sink[used] = '\0';
    return sink;
}

void mimo26_decode_stream_init(mimo26_decode_stream *stream)
{
    if (stream != NULL) {
        stream->pending_bytes = 0;
    }
}

/* How many bytes a UTF-8 lead byte announces, or 0 if it is not a lead. */
static size_t utf8_sequence_length(unsigned char lead)
{
    if (lead < 0x80) { return 1u; }
    if ((lead & 0xE0) == 0xC0) { return 2u; }
    if ((lead & 0xF0) == 0xE0) { return 3u; }
    if ((lead & 0xF8) == 0xF0) { return 4u; }
    return 0u;
}

bool mimo26_tokenizer_decode_stream(const mimo26_tokenizer *tokenizer,
                                    mimo26_decode_stream *stream, uint32_t id,
                                    char *out, size_t out_size,
                                    size_t *produced)
{
    if (tokenizer == NULL || stream == NULL || out == NULL ||
        out_size < 64u || produced == NULL) {
        return false;
    }
    char scratch[64];
    size_t used = 0;
    memcpy(scratch, stream->pending, stream->pending_bytes);
    used = stream->pending_bytes;
    if (!append_token_bytes(tokenizer, id, scratch, sizeof scratch, &used,
                            false)) {
        return false;
    }

    /*
     * Emit only whole sequences. A byte-level BPE token can end mid
     * character, and a server that sends that fragment produces invalid
     * UTF-8 on the wire.
     */
    size_t complete = 0;
    size_t offset = 0;
    while (offset < used) {
        const size_t width =
            utf8_sequence_length((unsigned char)scratch[offset]);
        if (width == 0u) {
            offset++;             /* stray continuation: pass it through */
            complete = offset;
            continue;
        }
        if (offset + width > used) {
            break;                /* incomplete: hold it back */
        }
        offset += width;
        complete = offset;
    }
    memcpy(out, scratch, complete);
    stream->pending_bytes = used - complete;
    memcpy(stream->pending, scratch + complete, stream->pending_bytes);
    *produced = complete;
    return true;
}

/* ---- chat template ---- */

static bool append_text(char **buffer, size_t *used, size_t *capacity,
                        const char *text)
{
    const size_t length = strlen(text);
    if (*used + length + 1u > *capacity) {
        size_t grown = *capacity ? *capacity : 256u;
        while (*used + length + 1u > grown) {
            grown *= 2u;
        }
        char *next = realloc(*buffer, grown);
        if (next == NULL) {
            return false;
        }
        *buffer = next;
        *capacity = grown;
    }
    memcpy(*buffer + *used, text, length);
    *used += length;
    (*buffer)[*used] = '\0';
    return true;
}

bool mimo26_tokenizer_encode_chat(mimo26_tokenizer *tokenizer,
                                  const mimo26_chat_message *messages,
                                  size_t message_count,
                                  bool add_generation_prompt,
                                  bool enable_thinking,
                                  mimo26_token_buffer *out, char *error,
                                  size_t error_size)
{
    if (tokenizer == NULL || out == NULL ||
        (messages == NULL && message_count > 0)) {
        set_error(error, error_size, "invalid chat arguments");
        return false;
    }
    out->count = 0;

    /*
     * Rendered once as text, then encoded with specials enabled. Doing it
     * this way rather than emitting ids directly means the markers go through
     * exactly the same matching path the fixtures exercise, so a mistake in
     * the template shows up as a rendering difference rather than as a
     * plausible but wrong id sequence.
     *
     * Message content is inserted verbatim. If it happens to spell a marker,
     * the marker-matching pass will encode it as one -- which is why request
     * content must be validated by the caller before it reaches here.
     */
    char *rendered = NULL;
    size_t used = 0, capacity = 0;
    bool ok = true;
    for (size_t i = 0; ok && i < message_count; i++) {
        const char *role = messages[i].role != NULL ? messages[i].role : "";
        ok = append_text(&rendered, &used, &capacity, "<|im_start|>") &&
             append_text(&rendered, &used, &capacity, role) &&
             append_text(&rendered, &used, &capacity, "\n");
        /* An assistant turn always carries a think block: the recorded
         * reasoning when there is some, an empty one when there is not. */
        if (ok && strcmp(role, "assistant") == 0) {
            ok = append_text(&rendered, &used, &capacity, "<think>") &&
                 append_text(&rendered, &used, &capacity,
                             messages[i].reasoning != NULL
                                 ? messages[i].reasoning : "") &&
                 append_text(&rendered, &used, &capacity, "</think>");
        }
        ok = ok &&
             append_text(&rendered, &used, &capacity,
                         messages[i].content != NULL ? messages[i].content
                                                     : "") &&
             append_text(&rendered, &used, &capacity, "<|im_end|>");
    }
    if (ok && add_generation_prompt) {
        ok = append_text(&rendered, &used, &capacity,
                         "<|im_start|>assistant\n");
        if (ok && !enable_thinking) {
            ok = append_text(&rendered, &used, &capacity,
                             "<think></think>");
        }
    }
    if (!ok) {
        free(rendered);
        set_error(error, error_size, "out of memory rendering the chat");
        return false;
    }
    if (rendered == NULL) {
        return true;   /* no messages and no generation prompt */
    }
    ok = mimo26_tokenizer_encode(tokenizer, rendered, true, out, error,
                                 error_size);
    free(rendered);
    return ok;
}

/* ---- lifecycle ---- */

uint32_t mimo26_tokenizer_vocab_size(const mimo26_tokenizer *tokenizer)
{
    return tokenizer == NULL ? 0u : tokenizer->id_capacity;
}

void mimo26_tokenizer_destroy(mimo26_tokenizer *tokenizer)
{
    if (tokenizer == NULL) {
        return;
    }
    if (tokenizer->vocab != NULL) {
        for (size_t i = 0; i <= tokenizer->vocab_mask; i++) {
            free(tokenizer->vocab[i].key);
        }
        free(tokenizer->vocab);
    }
    if (tokenizer->merges != NULL) {
        for (size_t i = 0; i <= tokenizer->merge_mask; i++) {
            free(tokenizer->merges[i].pair);
        }
        free(tokenizer->merges);
    }
    free(tokenizer->id_to_piece);
    if (tokenizer->added != NULL) {
        for (size_t i = 0; i < tokenizer->added_count; i++) {
            free(tokenizer->added[i].content);
        }
        free(tokenizer->added);
    }
    if (tokenizer->pretokenizer != NULL) {
        uregex_close(tokenizer->pretokenizer);
    }
    free(tokenizer);
}

static size_t round_up_pow2(size_t value)
{
    size_t result = 1u;
    while (result < value) {
        result <<= 1u;
    }
    return result;
}

bool mimo26_tokenizer_create(mimo26_tokenizer **out, const char *root,
                             char *error, size_t error_size)
{
    if (out == NULL || root == NULL) {
        set_error(error, error_size, "invalid tokenizer arguments");
        return false;
    }
    *out = NULL;

    char path[1024];
    snprintf(path, sizeof path, "%s/tokenizer.json", root);
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        set_error(error, error_size, "cannot open %s", path);
        return false;
    }
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = size > 0 ? malloc((size_t)size + 1u) : NULL;
    if (text == NULL || fread(text, 1u, (size_t)size, file) != (size_t)size) {
        free(text);
        fclose(file);
        set_error(error, error_size, "cannot read %s", path);
        return false;
    }
    fclose(file);
    text[size] = '\0';

    k3_json_document document;
    memset(&document, 0, sizeof document);
    if (!k3_json_parse(&document, text, (size_t)size, error, error_size)) {
        free(text);
        return false;
    }

    mimo26_tokenizer *tokenizer = calloc(1u, sizeof *tokenizer);
    if (tokenizer == NULL) {
        k3_json_document_free(&document);
        free(text);
        set_error(error, error_size, "out of memory");
        return false;
    }
    build_byte_alphabet(tokenizer);

    #define FAIL(message)                                                     \
        do {                                                                  \
            set_error(error, error_size, "%s", (message));                    \
            k3_json_document_free(&document);                                 \
            free(text);                                                       \
            mimo26_tokenizer_destroy(tokenizer);                              \
            return false;                                                     \
        } while (0)

    const int32_t model = k3_json_object_get(&document, document.root,
                                            "model");
    if (model < 0) { FAIL("tokenizer.json has no model"); }
    const int32_t vocab = k3_json_object_get(&document, model, "vocab");
    const int32_t merges = k3_json_object_get(&document, model, "merges");
    if (vocab < 0 || merges < 0) {
        FAIL("tokenizer.json model lacks vocab or merges");
    }

    /* `size` counts key and value tokens, so members is half. */
    const size_t vocab_members =
        (size_t)document.tokens[vocab].size / 2u + 1u;
    tokenizer->vocab_mask = round_up_pow2(vocab_members * 2u) - 1u;
    tokenizer->vocab = calloc(tokenizer->vocab_mask + 1u,
                              sizeof *tokenizer->vocab);
    if (tokenizer->vocab == NULL) { FAIL("out of memory for the vocabulary"); }

    uint32_t highest = 0;
    /* Object members are laid out flat: key, value, key, value. */
    for (int32_t child = document.tokens[vocab].first_child; child >= 0;) {
        const int32_t value = document.tokens[child].next_sibling;
        if (value < 0) { FAIL("a vocabulary entry has no value"); }
        char *key = NULL;
        if (!k3_json_string_dup(&document, child, &key, error, error_size)) {
            FAIL("a vocabulary key is not a string");
        }
        uint32_t id = 0;
        if (value < 0 || !k3_json_u32(&document, value, &id)) {
            free(key);
            FAIL("a vocabulary value is not an integer");
        }
        const bool inserted = vocab_insert(tokenizer, key, strlen(key), id);
        free(key);
        if (!inserted) { FAIL("out of memory inserting a vocabulary entry"); }
        if (id > highest) { highest = id; }
        child = document.tokens[value].next_sibling;
    }

    const size_t merge_members = (size_t)document.tokens[merges].size;
    tokenizer->merge_mask = round_up_pow2(merge_members * 2u) - 1u;
    tokenizer->merges = calloc(tokenizer->merge_mask + 1u,
                               sizeof *tokenizer->merges);
    if (tokenizer->merges == NULL) { FAIL("out of memory for the merges"); }

    uint32_t rank = 0;
    for (int32_t child = document.tokens[merges].first_child; child >= 0;
         child = document.tokens[child].next_sibling, rank++) {
        const int32_t left_node = document.tokens[child].first_child;
        if (left_node < 0) { FAIL("a merge entry is malformed"); }
        const int32_t right_node = document.tokens[left_node].next_sibling;
        if (right_node < 0) { FAIL("a merge entry is malformed"); }
        char *left = NULL, *right = NULL;
        if (!k3_json_string_dup(&document, left_node, &left, error,
                                error_size) ||
            !k3_json_string_dup(&document, right_node, &right, error,
                                error_size)) {
            free(left);
            FAIL("a merge entry is not a pair of strings");
        }
        const bool inserted = merge_insert(tokenizer, left, strlen(left),
                                           right, strlen(right), rank);
        free(left);
        free(right);
        if (!inserted) { FAIL("out of memory inserting a merge"); }
    }

    /* Added tokens carry ids above the BPE vocabulary. */
    const int32_t added = k3_json_object_get(&document, document.root,
                                            "added_tokens");
    if (added >= 0) {
        tokenizer->added = calloc((size_t)document.tokens[added].size,
                                  sizeof *tokenizer->added);
        if (tokenizer->added == NULL) { FAIL("out of memory for added tokens"); }
        for (int32_t child = document.tokens[added].first_child; child >= 0;
             child = document.tokens[child].next_sibling) {
            const int32_t id_node = k3_json_object_get(&document, child, "id");
            const int32_t content = k3_json_object_get(&document, child,
                                                       "content");
            if (id_node < 0 || content < 0) { FAIL("an added token is malformed"); }
            char *spelling = NULL;
            if (!k3_json_string_dup(&document, content, &spelling, error,
                                    error_size)) {
                FAIL("an added token content is not a string");
            }
            uint32_t added_id = 0;
            if (!k3_json_u32(&document, id_node, &added_id)) {
                free(spelling);
                FAIL("an added token id is not an integer");
            }
            added_token *slot = &tokenizer->added[tokenizer->added_count++];
            slot->content = spelling;
            slot->id = added_id;
            slot->special = true;
            if (slot->id > highest) { highest = slot->id; }
        }
    }

    /* Reverse index for decoding. */
    tokenizer->id_capacity = highest + 1u;
    tokenizer->id_to_piece = calloc(tokenizer->id_capacity,
                                    sizeof *tokenizer->id_to_piece);
    if (tokenizer->id_to_piece == NULL) { FAIL("out of memory for decoding"); }
    for (size_t i = 0; i <= tokenizer->vocab_mask; i++) {
        if (tokenizer->vocab[i].occupied) {
            tokenizer->id_to_piece[tokenizer->vocab[i].id] =
                tokenizer->vocab[i].key;
        }
    }
    for (size_t i = 0; i < tokenizer->added_count; i++) {
        /* Marks the id as defined; the spelling comes from `added`. */
        tokenizer->id_to_piece[tokenizer->added[i].id] =
            tokenizer->added[i].content;
    }
    #undef FAIL

    k3_json_document_free(&document);
    free(text);

    UErrorCode status = U_ZERO_ERROR;
    UParseError parse_error;
    memset(&parse_error, 0, sizeof parse_error);
    tokenizer->nfc = unorm2_getNFCInstance(&status);
    if (U_FAILURE(status) || tokenizer->nfc == NULL) {
        set_error(error, error_size, "NFC normalizer unavailable: %s",
                  u_errorName(status));
        mimo26_tokenizer_destroy(tokenizer);
        return false;
    }
    status = U_ZERO_ERROR;
    tokenizer->pretokenizer = uregex_openC(MIMO26_PRETOKENIZER_PATTERN, 0,
                                           &parse_error, &status);
    if (U_FAILURE(status) || tokenizer->pretokenizer == NULL) {
        set_error(error, error_size,
                  "compiling the pre-tokenizer failed at offset %d: %s",
                  parse_error.offset, u_errorName(status));
        mimo26_tokenizer_destroy(tokenizer);
        return false;
    }

    *out = tokenizer;
    return true;
}
