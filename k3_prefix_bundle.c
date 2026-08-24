#include "k3_prefix_bundle.h"

#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

enum {
    K3_PREFIX_ENDIAN = 0x01020304,
    K3_PREFIX_META_VERSION = 1,
    K3_PREFIX_META_HEADER = 256,
    K3_PREFIX_MANIFEST_VERSION = 1,
    K3_PREFIX_MANIFEST_HEADER = 256,
    K3_PREFIX_MANIFEST_ENTRY = 96,
    K3_PREFIX_ID_BYTES = 32,
};

static const uint8_t K3_PREFIX_META_MAGIC[8] = {
    'K', '3', 'P', 'F', 'X', 'M', '1', '\0',
};
static const uint8_t K3_PREFIX_MANIFEST_MAGIC[8] = {
    'K', '3', 'P', 'F', 'X', 'I', '1', '\0',
};

typedef struct {
    char id[K3_PREFIX_ID_BYTES + 1u];
    char state_path[PATH_MAX];
    char metadata_path[PATH_MAX];
    uint32_t *tokens;
    size_t token_count;
    k3_tool_choice_marker *tool_choices;
    size_t tool_choice_count;
    k3_single_tool_call_marker *single_tool_calls;
    size_t single_tool_call_count;
    k3_response_format_marker *response_formats;
    size_t response_format_count;
    k3_engine_state_file_info state_info;
    uint64_t publication_sequence;
    uint64_t last_use_sequence;
    uint64_t metadata_bytes;
} bundle_entry;

struct k3_prefix_bundle {
    char root[PATH_MAX];
    char entries_root[PATH_MAX];
    char manifest_path[PATH_MAX];
    k3_prefix_bundle_identity identity;
    uint32_t entry_limit;
    uint64_t byte_limit;
    uint64_t generation;
    uint64_t next_id;
    bundle_entry *entries;
    size_t count;
};

static void bundle_error(char *error, size_t error_size,
                         const char *format, ...) {
    if (error == NULL || error_size == 0u) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static void put_u32(uint8_t *output, uint32_t value) {
    for (uint32_t index = 0u; index < 4u; index++) {
        output[index] = (uint8_t)(value >> (index * 8u));
    }
}

static void put_u64(uint8_t *output, uint64_t value) {
    for (uint32_t index = 0u; index < 8u; index++) {
        output[index] = (uint8_t)(value >> (index * 8u));
    }
}

static uint32_t get_u32(const uint8_t *input) {
    uint32_t value = 0u;
    for (uint32_t index = 0u; index < 4u; index++) {
        value |= (uint32_t)input[index] << (index * 8u);
    }
    return value;
}

static uint64_t get_u64(const uint8_t *input) {
    uint64_t value = 0u;
    for (uint32_t index = 0u; index < 8u; index++) {
        value |= (uint64_t)input[index] << (index * 8u);
    }
    return value;
}

static void crc_table(uint64_t table[256]) {
    const uint64_t polynomial = UINT64_C(0x42f0e1eba9ea3693);
    for (uint32_t value = 0u; value < 256u; value++) {
        uint64_t crc = (uint64_t)value << 56u;
        for (uint32_t bit = 0u; bit < 8u; bit++) {
            crc = crc & (UINT64_C(1) << 63u) ?
                (crc << 1u) ^ polynomial : crc << 1u;
        }
        table[value] = crc;
    }
}

static uint64_t crc_update(const uint64_t table[256], uint64_t crc,
                           const void *data, uint64_t bytes) {
    const uint8_t *input = (const uint8_t *)data;
    for (uint64_t index = 0u; index < bytes; index++) {
        const uint8_t slot = (uint8_t)((crc >> 56u) ^ input[index]);
        crc = table[slot] ^ (crc << 8u);
    }
    return crc;
}

static uint64_t bytes_crc(const void *data, uint64_t bytes) {
    uint64_t table[256];
    crc_table(table);
    return crc_update(table, 0u, data, bytes);
}

static bool write_full(int fd, const void *data, uint64_t bytes) {
    const uint8_t *input = (const uint8_t *)data;
    while (bytes != 0u) {
        const size_t count = bytes > SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        const ssize_t written = write(fd, input, count);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        input += written;
        bytes -= (uint64_t)written;
    }
    return true;
}

static bool read_full(int fd, void *data, uint64_t bytes) {
    uint8_t *output = (uint8_t *)data;
    while (bytes != 0u) {
        const size_t count = bytes > SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        const ssize_t got = read(fd, output, count);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        output += got;
        bytes -= (uint64_t)got;
    }
    return true;
}

static bool path_join(char *output, size_t output_size,
                      const char *left, const char *right) {
    const int required = snprintf(output, output_size, "%s/%s", left, right);
    return required > 0 && (size_t)required < output_size;
}

static bool private_directory(const char *path, char *error,
                              size_t error_size) {
    struct stat status;
    if (lstat(path, &status) != 0) {
        if (errno != ENOENT || mkdir(path, S_IRWXU) != 0) {
            bundle_error(error, error_size, "creating %s failed: %s",
                         path, strerror(errno));
            return false;
        }
        if (lstat(path, &status) != 0) return false;
    }
    if (!S_ISDIR(status.st_mode) || S_ISLNK(status.st_mode) ||
        chmod(path, S_IRWXU) != 0) {
        bundle_error(error, error_size, "%s is not a private directory", path);
        return false;
    }
    return true;
}

static bool private_regular(const char *path, uint64_t *bytes,
                            char *error, size_t error_size) {
    struct stat status;
    if (lstat(path, &status) != 0 || !S_ISREG(status.st_mode) ||
        S_ISLNK(status.st_mode) || (status.st_mode & 077u) != 0u ||
        status.st_size < 0) {
        bundle_error(error, error_size, "%s is not a private regular file", path);
        return false;
    }
    if (bytes != NULL) *bytes = (uint64_t)status.st_size;
    return true;
}

static bool valid_id(const char *id) {
    if (id == NULL || strlen(id) != K3_PREFIX_ID_BYTES) return false;
    for (size_t index = 0u; index < K3_PREFIX_ID_BYTES; index++) {
        const char value = id[index];
        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f'))) return false;
    }
    return true;
}

static bool sync_directory(const char *path) {
    const int fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = fsync(fd) == 0;
    (void)close(fd);
    return ok;
}

static bool entry_paths(const k3_prefix_bundle *bundle, const char *id,
                        char *state, size_t state_size,
                        char *metadata, size_t metadata_size) {
    char state_name[48];
    char metadata_name[48];
    if (!valid_id(id) ||
        snprintf(state_name, sizeof(state_name), "%s.state", id) <= 0 ||
        snprintf(metadata_name, sizeof(metadata_name), "%s.meta", id) <= 0) {
        return false;
    }
    return path_join(state, state_size, bundle->entries_root, state_name) &&
        path_join(metadata, metadata_size, bundle->entries_root, metadata_name);
}

static void entry_free(bundle_entry *entry) {
    if (entry == NULL) return;
    for (size_t index = 0u; index < entry->response_format_count; index++) {
        free((char *)entry->response_formats[index].response_schema_json);
    }
    free(entry->response_formats);
    free(entry->single_tool_calls);
    free(entry->tool_choices);
    free(entry->tokens);
    memset(entry, 0, sizeof(*entry));
}

static bool identity_equal(const k3_prefix_bundle_identity *left,
                           const k3_prefix_bundle_identity *right) {
    return left->format_version == right->format_version &&
        left->context == right->context &&
        left->model_layout_crc64 == right->model_layout_crc64 &&
        left->q8_projections == right->q8_projections;
}

static bool snapshot_clone(bundle_entry *entry, const char *id,
                           const char *state_path, const char *metadata_path,
                           const k3_prefix_bundle_snapshot *snapshot,
                           const k3_engine_state_file_info *state_info,
                           char *error, size_t error_size) {
    memset(entry, 0, sizeof(*entry));
    if (snapshot == NULL || snapshot->tokens == NULL ||
        snapshot->token_count == 0u || snapshot->token_count > UINT32_MAX ||
        state_info == NULL || state_info->token_position != snapshot->token_count ||
        !valid_id(id) || strlen(state_path) >= sizeof(entry->state_path) ||
        strlen(metadata_path) >= sizeof(entry->metadata_path)) {
        bundle_error(error, error_size, "invalid checkpoint snapshot");
        return false;
    }
    entry->tokens = (uint32_t *)malloc(
        snapshot->token_count * sizeof(*entry->tokens));
    entry->tool_choices = snapshot->tool_choice_count == 0u ? NULL :
        (k3_tool_choice_marker *)malloc(snapshot->tool_choice_count *
                                        sizeof(*entry->tool_choices));
    entry->single_tool_calls = snapshot->single_tool_call_count == 0u ? NULL :
        (k3_single_tool_call_marker *)malloc(snapshot->single_tool_call_count *
                                              sizeof(*entry->single_tool_calls));
    entry->response_formats = snapshot->response_format_count == 0u ? NULL :
        (k3_response_format_marker *)calloc(snapshot->response_format_count,
                                             sizeof(*entry->response_formats));
    if (entry->tokens == NULL ||
        (snapshot->tool_choice_count != 0u && entry->tool_choices == NULL) ||
        (snapshot->single_tool_call_count != 0u && entry->single_tool_calls == NULL) ||
        (snapshot->response_format_count != 0u && entry->response_formats == NULL)) {
        bundle_error(error, error_size, "allocating checkpoint metadata failed");
        entry_free(entry);
        return false;
    }
    memcpy(entry->tokens, snapshot->tokens,
           snapshot->token_count * sizeof(*entry->tokens));
    if (snapshot->tool_choice_count != 0u) {
        memcpy(entry->tool_choices, snapshot->tool_choices,
               snapshot->tool_choice_count * sizeof(*entry->tool_choices));
    }
    if (snapshot->single_tool_call_count != 0u) {
        memcpy(entry->single_tool_calls, snapshot->single_tool_calls,
               snapshot->single_tool_call_count * sizeof(*entry->single_tool_calls));
    }
    for (size_t index = 0u; index < snapshot->response_format_count; index++) {
        entry->response_formats[index] = snapshot->response_formats[index];
        const char *schema = snapshot->response_formats[index].response_schema_json;
        if (schema != NULL) {
            entry->response_formats[index].response_schema_json = strdup(schema);
            if (entry->response_formats[index].response_schema_json == NULL) {
                bundle_error(error, error_size, "copying checkpoint schema failed");
                entry_free(entry);
                return false;
            }
        }
    }
    memcpy(entry->id, id, K3_PREFIX_ID_BYTES + 1u);
    memcpy(entry->state_path, state_path, strlen(state_path) + 1u);
    memcpy(entry->metadata_path, metadata_path, strlen(metadata_path) + 1u);
    entry->token_count = snapshot->token_count;
    entry->tool_choice_count = snapshot->tool_choice_count;
    entry->single_tool_call_count =
        snapshot->single_tool_call_count;
    entry->response_format_count =
        snapshot->response_format_count;
    for (size_t index = 0u; index < entry->tool_choice_count; index++) {
        if ((entry->tool_choices[index].choice !=
                K3_TOOL_CHOICE_REQUIRED &&
             entry->tool_choices[index].choice != K3_TOOL_CHOICE_NONE) ||
            (index != 0u &&
             entry->tool_choices[index - 1u].after_message_count >=
                 entry->tool_choices[index].after_message_count)) {
            bundle_error(error, error_size,
                         "checkpoint tool-choice markers are invalid");
            entry_free(entry);
            return false;
        }
    }
    for (size_t index = 1u;
         index < entry->single_tool_call_count; index++) {
        if (entry->single_tool_calls[index - 1u]
                .after_message_count >=
            entry->single_tool_calls[index].after_message_count) {
            bundle_error(error, error_size,
                         "checkpoint single-call markers are invalid");
            entry_free(entry);
            return false;
        }
    }
    for (size_t index = 0u;
         index < entry->response_format_count; index++) {
        const k3_response_format_marker *marker =
            &entry->response_formats[index];
        const char *schema = marker->response_schema_json;
        if ((marker->format != K3_RESPONSE_FORMAT_JSON_OBJECT &&
             marker->format != K3_RESPONSE_FORMAT_JSON_SCHEMA) ||
            (marker->format == K3_RESPONSE_FORMAT_JSON_SCHEMA) !=
                (schema != NULL) ||
            (schema != NULL && strlen(schema) > UINT32_MAX) ||
            (index != 0u &&
             entry->response_formats[index - 1u]
                     .after_message_count >=
                 marker->after_message_count)) {
            bundle_error(error, error_size,
                         "checkpoint response-format markers are invalid");
            entry_free(entry);
            return false;
        }
    }
    entry->state_info = *state_info;
    return true;
}

static bool metadata_body(const bundle_entry *entry, uint8_t **body_out,
                          uint64_t *body_bytes_out, uint64_t *schema_bytes_out,
                          char *error, size_t error_size) {
    uint64_t schema_bytes = 0u;
    for (size_t index = 0u; index < entry->response_format_count; index++) {
        const char *schema = entry->response_formats[index].response_schema_json;
        if (schema != NULL) {
            const size_t length = strlen(schema);
            if (schema_bytes > UINT64_MAX - length) return false;
            schema_bytes += length;
        }
    }
    const uint64_t token_bytes = (uint64_t)entry->token_count * 4u;
    const uint64_t tool_bytes = (uint64_t)entry->tool_choice_count * 16u;
    const uint64_t single_bytes = (uint64_t)entry->single_tool_call_count * 8u;
    const uint64_t response_bytes = (uint64_t)entry->response_format_count * 24u;
    if (token_bytes > UINT64_MAX - tool_bytes ||
        token_bytes + tool_bytes > UINT64_MAX - single_bytes ||
        token_bytes + tool_bytes + single_bytes > UINT64_MAX - response_bytes ||
        token_bytes + tool_bytes + single_bytes + response_bytes >
            UINT64_MAX - schema_bytes) {
        bundle_error(error, error_size, "checkpoint metadata size overflow");
        return false;
    }
    const uint64_t body_bytes = token_bytes + tool_bytes + single_bytes +
        response_bytes + schema_bytes;
    if (body_bytes > SIZE_MAX) return false;
    uint8_t *body = (uint8_t *)calloc(1u, (size_t)body_bytes);
    if (body == NULL) return false;
    uint64_t cursor = 0u;
    for (size_t index = 0u; index < entry->token_count; index++, cursor += 4u) {
        put_u32(body + cursor, entry->tokens[index]);
    }
    for (size_t index = 0u; index < entry->tool_choice_count; index++, cursor += 16u) {
        put_u64(body + cursor, entry->tool_choices[index].after_message_count);
        put_u32(body + cursor + 8u, (uint32_t)entry->tool_choices[index].choice);
    }
    for (size_t index = 0u; index < entry->single_tool_call_count; index++, cursor += 8u) {
        put_u64(body + cursor, entry->single_tool_calls[index].after_message_count);
    }
    const uint64_t schema_start = token_bytes + tool_bytes + single_bytes + response_bytes;
    uint64_t schema_cursor = 0u;
    for (size_t index = 0u; index < entry->response_format_count; index++, cursor += 24u) {
        const k3_response_format_marker *marker = &entry->response_formats[index];
        const char *schema = marker->response_schema_json;
        const uint64_t length = schema == NULL ? 0u : strlen(schema);
        put_u64(body + cursor, marker->after_message_count);
        put_u32(body + cursor + 8u, (uint32_t)marker->format);
        put_u32(body + cursor + 12u, (uint32_t)length);
        put_u64(body + cursor + 16u, schema_cursor);
        if (length != 0u) {
            memcpy(body + schema_start + schema_cursor, schema, (size_t)length);
            schema_cursor += length;
        }
    }
    *body_out = body;
    *body_bytes_out = body_bytes;
    *schema_bytes_out = schema_bytes;
    return true;
}

static bool metadata_write(bundle_entry *entry, char *error, size_t error_size) {
    uint8_t *body = NULL;
    uint64_t body_bytes = 0u;
    uint64_t schema_bytes = 0u;
    if (!metadata_body(entry, &body, &body_bytes, &schema_bytes,
                       error, error_size)) return false;
    uint8_t header[K3_PREFIX_META_HEADER];
    memset(header, 0, sizeof(header));
    memcpy(header, K3_PREFIX_META_MAGIC, sizeof(K3_PREFIX_META_MAGIC));
    put_u32(header + 8u, K3_PREFIX_META_VERSION);
    put_u32(header + 12u, K3_PREFIX_META_HEADER);
    put_u32(header + 16u, K3_PREFIX_ENDIAN);
    put_u32(header + 20u, entry->state_info.q8_projections ? 1u : 0u);
    put_u32(header + 24u, entry->state_info.context);
    put_u32(header + 28u, (uint32_t)entry->token_count);
    put_u32(header + 32u, entry->state_info.format_version);
    put_u32(header + 36u, (uint32_t)entry->tool_choice_count);
    put_u32(header + 40u, (uint32_t)entry->single_tool_call_count);
    put_u32(header + 44u, (uint32_t)entry->response_format_count);
    put_u64(header + 48u, entry->state_info.model_layout_crc64);
    put_u64(header + 56u, entry->state_info.payload_bytes);
    put_u64(header + 64u, entry->state_info.file_bytes);
    put_u64(header + 72u, entry->state_info.payload_crc64);
    put_u64(header + 80u, schema_bytes);
    put_u64(header + 88u, body_bytes);
    put_u64(header + 96u, bytes_crc(body, body_bytes));
    put_u64(header + 104u, 0u);
    put_u64(header + 104u, bytes_crc(header, sizeof(header)));

    char temporary[PATH_MAX];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp.XXXXXX",
                 entry->metadata_path) <= 0) {
        free(body);
        return false;
    }
    int fd = mkstemp(temporary);
    if (fd < 0) {
        bundle_error(error, error_size, "creating metadata temporary failed: %s",
                     strerror(errno));
        free(body);
        return false;
    }
    bool ok = fchmod(fd, S_IRUSR | S_IWUSR) == 0 &&
        write_full(fd, header, sizeof(header)) &&
        write_full(fd, body, body_bytes) && fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    fd = -1;
    if (ok) ok = rename(temporary, entry->metadata_path) == 0;
    if (!ok) {
        bundle_error(error, error_size, "publishing checkpoint metadata failed: %s",
                     strerror(errno));
        if (fd >= 0) (void)close(fd);
        (void)unlink(temporary);
    } else {
        entry->metadata_bytes = K3_PREFIX_META_HEADER + body_bytes;
    }
    free(body);
    return ok;
}

static bool metadata_load(bundle_entry *entry,
                          const k3_prefix_bundle_identity *identity,
                          char *error, size_t error_size) {
    uint64_t file_bytes = 0u;
    if (!private_regular(entry->metadata_path, &file_bytes, error, error_size) ||
        file_bytes < K3_PREFIX_META_HEADER || file_bytes > SIZE_MAX) return false;
    int fd = open(entry->metadata_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    uint8_t header[K3_PREFIX_META_HEADER];
    bool ok = read_full(fd, header, sizeof(header));
    if (!ok || memcmp(header, K3_PREFIX_META_MAGIC, 8u) != 0 ||
        get_u32(header + 8u) != K3_PREFIX_META_VERSION ||
        get_u32(header + 12u) != K3_PREFIX_META_HEADER ||
        get_u32(header + 16u) != K3_PREFIX_ENDIAN) {
        bundle_error(error, error_size, "invalid checkpoint metadata header");
        (void)close(fd);
        return false;
    }
    const uint64_t expected_header_crc = get_u64(header + 104u);
    put_u64(header + 104u, 0u);
    if (bytes_crc(header, sizeof(header)) != expected_header_crc) {
        bundle_error(error, error_size, "checkpoint metadata header CRC mismatch");
        (void)close(fd);
        return false;
    }
    const uint32_t token_count = get_u32(header + 28u);
    const uint32_t tool_count = get_u32(header + 36u);
    const uint32_t single_count = get_u32(header + 40u);
    const uint32_t response_count = get_u32(header + 44u);
    const uint64_t schema_bytes = get_u64(header + 80u);
    const uint64_t body_bytes = get_u64(header + 88u);
    if (token_count == 0u || body_bytes != file_bytes - K3_PREFIX_META_HEADER ||
        body_bytes > SIZE_MAX || schema_bytes > body_bytes ||
        get_u32(header + 24u) != identity->context ||
        get_u64(header + 48u) != identity->model_layout_crc64 ||
        ((get_u32(header + 20u) & 1u) != 0u) != identity->q8_projections ||
        get_u32(header + 32u) != identity->format_version) {
        bundle_error(error, error_size, "checkpoint metadata identity/size mismatch");
        (void)close(fd);
        return false;
    }
    uint8_t *body = (uint8_t *)malloc((size_t)body_bytes);
    bool body_ok = body != NULL && read_full(fd, body, body_bytes);
    if (close(fd) != 0) body_ok = false;
    if (!body_ok ||
        bytes_crc(body, body_bytes) != get_u64(header + 96u)) {
        bundle_error(error, error_size,
                     "checkpoint metadata body read/CRC failed");
        free(body);
        return false;
    }
    const uint64_t token_bytes = (uint64_t)token_count * 4u;
    const uint64_t tool_bytes = (uint64_t)tool_count * 16u;
    const uint64_t single_bytes = (uint64_t)single_count * 8u;
    const uint64_t response_bytes = (uint64_t)response_count * 24u;
    if (token_bytes + tool_bytes + single_bytes + response_bytes > body_bytes ||
        token_bytes + tool_bytes + single_bytes + response_bytes +
            schema_bytes != body_bytes) {
        bundle_error(error, error_size,
                     "checkpoint metadata body ledger mismatch");
        free(body);
        return false;
    }
    entry->tokens = (uint32_t *)malloc((size_t)token_count * sizeof(uint32_t));
    entry->tool_choices = tool_count == 0u ? NULL :
        (k3_tool_choice_marker *)calloc(tool_count, sizeof(*entry->tool_choices));
    entry->single_tool_calls = single_count == 0u ? NULL :
        (k3_single_tool_call_marker *)calloc(single_count, sizeof(*entry->single_tool_calls));
    entry->response_formats = response_count == 0u ? NULL :
        (k3_response_format_marker *)calloc(response_count, sizeof(*entry->response_formats));
    if (entry->tokens == NULL ||
        (tool_count != 0u && entry->tool_choices == NULL) ||
        (single_count != 0u && entry->single_tool_calls == NULL) ||
        (response_count != 0u && entry->response_formats == NULL)) {
        bundle_error(error, error_size, "allocating loaded checkpoint metadata failed");
        free(body);
        entry_free(entry);
        return false;
    }
    uint64_t cursor = 0u;
    for (uint32_t index = 0u; index < token_count; index++, cursor += 4u) {
        entry->tokens[index] = get_u32(body + cursor);
    }
    for (uint32_t index = 0u; index < tool_count; index++, cursor += 16u) {
        const uint64_t boundary = get_u64(body + cursor);
        const uint32_t choice = get_u32(body + cursor + 8u);
        if (boundary > SIZE_MAX || choice > K3_TOOL_CHOICE_NONE) goto invalid_body;
        entry->tool_choices[index].after_message_count = (size_t)boundary;
        entry->tool_choices[index].choice = (k3_tool_choice)choice;
    }
    for (uint32_t index = 0u; index < single_count; index++, cursor += 8u) {
        const uint64_t boundary = get_u64(body + cursor);
        if (boundary > SIZE_MAX) goto invalid_body;
        entry->single_tool_calls[index].after_message_count = (size_t)boundary;
    }
    const uint64_t schema_start = token_bytes + tool_bytes + single_bytes + response_bytes;
    for (uint32_t index = 0u; index < response_count; index++, cursor += 24u) {
        const uint64_t boundary = get_u64(body + cursor);
        const uint32_t format = get_u32(body + cursor + 8u);
        const uint32_t length = get_u32(body + cursor + 12u);
        const uint64_t schema_offset = get_u64(body + cursor + 16u);
        if (boundary > SIZE_MAX || format > K3_RESPONSE_FORMAT_JSON_SCHEMA ||
            schema_offset > schema_bytes || length > schema_bytes - schema_offset ||
            (format == K3_RESPONSE_FORMAT_JSON_SCHEMA && length == 0u) ||
            (format != K3_RESPONSE_FORMAT_JSON_SCHEMA && length != 0u)) goto invalid_body;
        entry->response_formats[index].after_message_count = (size_t)boundary;
        entry->response_formats[index].format = (k3_response_format)format;
        if (length != 0u) {
            char *schema = (char *)malloc((size_t)length + 1u);
            if (schema == NULL) goto invalid_body;
            memcpy(schema, body + schema_start + schema_offset, length);
            schema[length] = '\0';
            entry->response_formats[index].response_schema_json = schema;
        }
    }
    entry->token_count = token_count;
    entry->tool_choice_count = tool_count;
    entry->single_tool_call_count = single_count;
    entry->response_format_count = response_count;
    entry->state_info.format_version = get_u32(header + 32u);
    entry->state_info.context = get_u32(header + 24u);
    entry->state_info.token_position = token_count;
    entry->state_info.model_layout_crc64 = get_u64(header + 48u);
    entry->state_info.payload_bytes = get_u64(header + 56u);
    entry->state_info.file_bytes = get_u64(header + 64u);
    entry->state_info.payload_crc64 = get_u64(header + 72u);
    entry->state_info.q8_projections = (get_u32(header + 20u) & 1u) != 0u;
    entry->metadata_bytes = file_bytes;
    free(body);
    return true;

invalid_body:
    bundle_error(error, error_size, "checkpoint metadata marker ledger is invalid");
    free(body);
    entry_free(entry);
    return false;
}

static uint64_t entry_total_bytes(const bundle_entry *entry) {
    return entry->state_info.file_bytes + entry->metadata_bytes;
}

static bool manifest_write(const k3_prefix_bundle *bundle,
                           const bundle_entry *entries, size_t count,
                           uint64_t generation,
                           char *error, size_t error_size) {
    if (count > UINT32_MAX || count > SIZE_MAX / K3_PREFIX_MANIFEST_ENTRY) return false;
    const uint64_t body_bytes = (uint64_t)count * K3_PREFIX_MANIFEST_ENTRY;
    uint8_t *body = (uint8_t *)calloc(1u, (size_t)body_bytes);
    if (body == NULL && body_bytes != 0u) return false;
    for (size_t index = 0u; index < count; index++) {
        uint8_t *record = body + index * K3_PREFIX_MANIFEST_ENTRY;
        memcpy(record, entries[index].id, K3_PREFIX_ID_BYTES);
        put_u64(record + 32u, entries[index].publication_sequence);
        put_u64(record + 40u, entries[index].last_use_sequence);
        put_u64(record + 48u, entries[index].state_info.file_bytes);
        put_u64(record + 56u, entries[index].metadata_bytes);
        put_u32(record + 64u, (uint32_t)entries[index].token_count);
        put_u32(record + 68u, entries[index].state_info.format_version);
        put_u64(record + 72u, entries[index].state_info.payload_bytes);
        put_u64(record + 80u, entries[index].state_info.payload_crc64);
        put_u64(record + 88u, entries[index].state_info.model_layout_crc64);
    }
    uint8_t header[K3_PREFIX_MANIFEST_HEADER];
    memset(header, 0, sizeof(header));
    memcpy(header, K3_PREFIX_MANIFEST_MAGIC, 8u);
    put_u32(header + 8u, K3_PREFIX_MANIFEST_VERSION);
    put_u32(header + 12u, K3_PREFIX_MANIFEST_HEADER);
    put_u32(header + 16u, K3_PREFIX_ENDIAN);
    put_u32(header + 20u, bundle->identity.q8_projections ? 1u : 0u);
    put_u32(header + 24u, bundle->identity.context);
    put_u32(header + 28u, (uint32_t)count);
    put_u64(header + 32u, bundle->identity.model_layout_crc64);
    put_u64(header + 40u, generation);
    put_u32(header + 48u, bundle->entry_limit);
    put_u32(header + 52u, bundle->identity.format_version);
    put_u64(header + 56u, bundle->byte_limit);
    put_u64(header + 64u, body_bytes);
    put_u64(header + 72u, bytes_crc(body, body_bytes));
    put_u64(header + 80u, 0u);
    put_u64(header + 80u, bytes_crc(header, sizeof(header)));

    char temporary[PATH_MAX];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp.XXXXXX",
                 bundle->manifest_path) <= 0) {
        free(body);
        return false;
    }
    int fd = mkstemp(temporary);
    if (fd < 0) {
        free(body);
        return false;
    }
    bool ok = fchmod(fd, S_IRUSR | S_IWUSR) == 0 &&
        write_full(fd, header, sizeof(header)) &&
        write_full(fd, body, body_bytes) && fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    fd = -1;
    if (ok) {
        ok = rename(temporary, bundle->manifest_path) == 0;
    }
    if (ok) {
        const int root_fd = open(
            bundle->root, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (root_fd >= 0) {
            (void)fsync(root_fd);
            (void)close(root_fd);
        }
    }
    if (!ok) {
        bundle_error(error, error_size, "publishing checkpoint manifest failed: %s",
                     strerror(errno));
        if (fd >= 0) (void)close(fd);
        (void)unlink(temporary);
    }
    free(body);
    return ok;
}

static bool manifest_load(k3_prefix_bundle *bundle,
                          char *error, size_t error_size) {
    int fd = open(bundle->manifest_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0 && errno == ENOENT) return true;
    if (fd < 0) {
        bundle_error(error, error_size,
                     "opening checkpoint manifest failed: %s",
                     strerror(errno));
        return false;
    }
    struct stat status;
    uint8_t header[K3_PREFIX_MANIFEST_HEADER];
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
        (status.st_mode & 077u) != 0u ||
        status.st_size < (off_t)sizeof(header) ||
        !read_full(fd, header, sizeof(header)) ||
        memcmp(header, K3_PREFIX_MANIFEST_MAGIC, 8u) != 0 ||
        get_u32(header + 8u) != K3_PREFIX_MANIFEST_VERSION ||
        get_u32(header + 12u) != K3_PREFIX_MANIFEST_HEADER ||
        get_u32(header + 16u) != K3_PREFIX_ENDIAN) {
        bundle_error(error, error_size,
                     "invalid checkpoint manifest header");
        (void)close(fd);
        return false;
    }
    const uint64_t header_crc = get_u64(header + 80u);
    put_u64(header + 80u, 0u);
    const uint32_t count = get_u32(header + 28u);
    const uint64_t body_bytes = get_u64(header + 64u);
    const k3_prefix_bundle_identity stored = {
        .format_version = get_u32(header + 52u),
        .context = get_u32(header + 24u),
        .model_layout_crc64 = get_u64(header + 32u),
        .q8_projections = (get_u32(header + 20u) & 1u) != 0u,
    };
    if (bytes_crc(header, sizeof(header)) != header_crc ||
        !identity_equal(&stored, &bundle->identity) ||
        count > bundle->entry_limit ||
        body_bytes != (uint64_t)count * K3_PREFIX_MANIFEST_ENTRY ||
        (uint64_t)status.st_size != sizeof(header) + body_bytes) {
        bundle_error(error, error_size,
                     "checkpoint manifest identity/size mismatch");
        (void)close(fd);
        return false;
    }
    uint8_t *body = (uint8_t *)malloc((size_t)body_bytes);
    bool body_ok = (body != NULL || body_bytes == 0u) &&
        read_full(fd, body, body_bytes);
    if (close(fd) != 0) body_ok = false;
    if (!body_ok ||
        bytes_crc(body, body_bytes) != get_u64(header + 72u)) {
        bundle_error(error, error_size,
                     "checkpoint manifest body read/CRC failed");
        free(body);
        return false;
    }
    bundle_entry *entries = count == 0u ? NULL :
        (bundle_entry *)calloc(count, sizeof(*entries));
    if (entries == NULL && count != 0u) {
        free(body);
        return false;
    }
    uint64_t total = 0u;
    for (uint32_t index = 0u; index < count; index++) {
        const uint8_t *record = body + (uint64_t)index * K3_PREFIX_MANIFEST_ENTRY;
        memcpy(entries[index].id, record, K3_PREFIX_ID_BYTES);
        entries[index].id[K3_PREFIX_ID_BYTES] = '\0';
        if (!entry_paths(bundle, entries[index].id,
                         entries[index].state_path, sizeof(entries[index].state_path),
                         entries[index].metadata_path, sizeof(entries[index].metadata_path))) {
            bundle_error(error, error_size, "checkpoint manifest ID is invalid");
            goto fail;
        }
        entries[index].publication_sequence = get_u64(record + 32u);
        entries[index].last_use_sequence = get_u64(record + 40u);
        const uint64_t state_bytes = get_u64(record + 48u);
        const uint64_t metadata_bytes = get_u64(record + 56u);
        if (!metadata_load(&entries[index], &bundle->identity, error, error_size)) goto fail;
        uint64_t actual_state_bytes = 0u;
        if (!private_regular(entries[index].state_path, &actual_state_bytes,
                             error, error_size) ||
            actual_state_bytes != state_bytes ||
            entries[index].metadata_bytes != metadata_bytes ||
            entries[index].token_count != get_u32(record + 64u) ||
            entries[index].state_info.format_version != get_u32(record + 68u) ||
            entries[index].state_info.payload_bytes != get_u64(record + 72u) ||
            entries[index].state_info.payload_crc64 != get_u64(record + 80u) ||
            entries[index].state_info.model_layout_crc64 != get_u64(record + 88u) ||
            entries[index].state_info.file_bytes != state_bytes ||
            state_bytes > UINT64_MAX - metadata_bytes ||
            total > UINT64_MAX - (state_bytes + metadata_bytes)) {
            bundle_error(error, error_size, "checkpoint manifest entry mismatch");
            goto fail;
        }
        total += state_bytes + metadata_bytes;
        for (uint32_t prior = 0u; prior < index; prior++) {
            if (strcmp(entries[prior].id, entries[index].id) == 0 ||
                (entries[prior].token_count == entries[index].token_count &&
                 memcmp(entries[prior].tokens, entries[index].tokens,
                        entries[index].token_count * sizeof(uint32_t)) == 0)) {
                bundle_error(error, error_size, "duplicate checkpoint manifest entry");
                goto fail;
            }
        }
    }
    if (total > bundle->byte_limit) {
        bundle_error(error, error_size, "checkpoint manifest exceeds byte limit");
        goto fail;
    }
    free(body);
    bundle->entries = entries;
    bundle->count = count;
    bundle->generation = get_u64(header + 40u);
    bundle->next_id = bundle->generation + 1u;
    return true;

fail:
    for (uint32_t index = 0u; index < count; index++) entry_free(&entries[index]);
    free(entries);
    free(body);
    return false;
}

static bool bundle_references_id(
        const k3_prefix_bundle *bundle,
        const char *id) {
    for (size_t index = 0u; index < bundle->count; index++) {
        if (strcmp(bundle->entries[index].id, id) == 0) return true;
    }
    return false;
}

static void cleanup_orphans(k3_prefix_bundle *bundle) {
    DIR *directory = opendir(bundle->entries_root);
    if (directory == NULL) return;
    struct dirent *item;
    while ((item = readdir(directory)) != NULL) {
        const size_t length = strlen(item->d_name);
        if (length < K3_PREFIX_ID_BYTES + 5u) continue;
        char id[K3_PREFIX_ID_BYTES + 1u];
        memcpy(id, item->d_name, K3_PREFIX_ID_BYTES);
        id[K3_PREFIX_ID_BYTES] = '\0';
        if (!valid_id(id) || bundle_references_id(bundle, id)) continue;
        const char *suffix = item->d_name + K3_PREFIX_ID_BYTES;
        if (strcmp(suffix, ".state") != 0 &&
            strcmp(suffix, ".meta") != 0 &&
            strstr(suffix, ".tmp.") == NULL) {
            continue;
        }
        char path[PATH_MAX];
        if (path_join(path, sizeof(path),
                      bundle->entries_root, item->d_name)) {
            (void)unlink(path);
        }
    }
    (void)closedir(directory);
}
bool k3_prefix_bundle_open(
        k3_prefix_bundle **out,
        const char *absolute_root,
        const k3_prefix_bundle_identity *identity,
        uint32_t entry_limit,
        uint64_t byte_limit,
        char *error,
        size_t error_size) {
    if (out != NULL) *out = NULL;
    if (out == NULL || absolute_root == NULL || absolute_root[0] != '/' ||
        strlen(absolute_root) >= PATH_MAX || identity == NULL ||
        identity->format_version == 0u || identity->context == 0u ||
        identity->model_layout_crc64 == 0u || entry_limit == 0u ||
        entry_limit > 64u || byte_limit == 0u) {
        bundle_error(error, error_size, "invalid checkpoint bundle configuration");
        return false;
    }
    k3_prefix_bundle *bundle = (k3_prefix_bundle *)calloc(1u, sizeof(*bundle));
    if (bundle == NULL) return false;
    memcpy(bundle->root, absolute_root, strlen(absolute_root) + 1u);
    bundle->identity = *identity;
    bundle->entry_limit = entry_limit;
    bundle->byte_limit = byte_limit;
    if (!path_join(bundle->entries_root, sizeof(bundle->entries_root),
                   bundle->root, "entries") ||
        !path_join(bundle->manifest_path, sizeof(bundle->manifest_path),
                   bundle->root, "manifest.bin") ||
        !private_directory(bundle->root, error, error_size) ||
        !private_directory(bundle->entries_root, error, error_size) ||
        !manifest_load(bundle, error, error_size)) {
        k3_prefix_bundle_destroy(bundle);
        return false;
    }
    cleanup_orphans(bundle);
    *out = bundle;
    return true;
}

void k3_prefix_bundle_destroy(k3_prefix_bundle *bundle) {
    if (bundle == NULL) return;
    for (size_t index = 0u; index < bundle->count; index++) {
        entry_free(&bundle->entries[index]);
    }
    free(bundle->entries);
    free(bundle);
}

size_t k3_prefix_bundle_count(const k3_prefix_bundle *bundle) {
    return bundle == NULL ? 0u : bundle->count;
}

bool k3_prefix_bundle_entry_at(const k3_prefix_bundle *bundle, size_t index,
                               k3_prefix_bundle_entry *entry) {
    if (entry != NULL) memset(entry, 0, sizeof(*entry));
    if (bundle == NULL || entry == NULL || index >= bundle->count) return false;
    const bundle_entry *source = &bundle->entries[index];
    entry->id = source->id;
    entry->state_path = source->state_path;
    entry->tokens = source->tokens;
    entry->token_count = source->token_count;
    entry->tool_choices = source->tool_choices;
    entry->tool_choice_count = source->tool_choice_count;
    entry->single_tool_calls = source->single_tool_calls;
    entry->single_tool_call_count = source->single_tool_call_count;
    entry->response_formats = source->response_formats;
    entry->response_format_count = source->response_format_count;
    entry->state_info = &source->state_info;
    entry->publication_sequence = source->publication_sequence;
    entry->last_use_sequence = source->last_use_sequence;
    entry->metadata_bytes = source->metadata_bytes;
    return true;
}

bool k3_prefix_bundle_find_exact(const k3_prefix_bundle *bundle,
                                 const uint32_t *tokens, size_t token_count,
                                 size_t *index) {
    if (index != NULL) *index = SIZE_MAX;
    if (bundle == NULL || tokens == NULL || token_count == 0u || index == NULL) return false;
    for (size_t item = 0u; item < bundle->count; item++) {
        if (bundle->entries[item].token_count == token_count &&
            memcmp(bundle->entries[item].tokens, tokens,
                   token_count * sizeof(*tokens)) == 0) {
            *index = item;
            return true;
        }
    }
    return false;
}

bool k3_prefix_bundle_allocate_state_path(
        k3_prefix_bundle *bundle, char *id, size_t id_size,
        char *state_path, size_t state_path_size,
        char *error, size_t error_size) {
    if (bundle == NULL || id == NULL || id_size < K3_PREFIX_ID_BYTES + 1u ||
        state_path == NULL) return false;
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    for (uint32_t attempt = 0u; attempt < 1024u; attempt++) {
        const uint64_t sequence = bundle->next_id++;
        const uint64_t stamp = (uint64_t)now.tv_sec * UINT64_C(1000000000) +
            (uint64_t)now.tv_nsec + attempt;
        snprintf(id, id_size, "%016llx%016llx",
                 (unsigned long long)stamp, (unsigned long long)sequence);
        char metadata[PATH_MAX];
        if (!entry_paths(bundle, id, state_path, state_path_size,
                         metadata, sizeof(metadata))) return false;
        struct stat ignored;
        if (lstat(state_path, &ignored) != 0 && errno == ENOENT &&
            lstat(metadata, &ignored) != 0 && errno == ENOENT) return true;
    }
    bundle_error(error, error_size, "allocating checkpoint ID failed");
    return false;
}

static size_t oldest_entry(const bundle_entry *entries, size_t count,
                           const bool *keep) {
    size_t oldest = SIZE_MAX;
    for (size_t index = 0u; index < count; index++) {
        if (!keep[index]) continue;
        if (oldest == SIZE_MAX ||
            entries[index].last_use_sequence < entries[oldest].last_use_sequence ||
            (entries[index].last_use_sequence == entries[oldest].last_use_sequence &&
             entries[index].publication_sequence < entries[oldest].publication_sequence)) {
            oldest = index;
        }
    }
    return oldest;
}

bool k3_prefix_bundle_publish(
        k3_prefix_bundle *bundle, const char *id, const char *state_path,
        const k3_prefix_bundle_snapshot *snapshot,
        const k3_engine_state_file_info *state_info,
        char *error, size_t error_size) {
    if (bundle == NULL || id == NULL || state_path == NULL ||
        snapshot == NULL || state_info == NULL) return false;
    if (state_info->format_version != bundle->identity.format_version ||
        state_info->context != bundle->identity.context ||
        state_info->model_layout_crc64 !=
            bundle->identity.model_layout_crc64 ||
        state_info->q8_projections != bundle->identity.q8_projections ||
        state_info->payload_bytes == 0u ||
        state_info->file_bytes <= state_info->payload_bytes) {
        bundle_error(error, error_size,
                     "checkpoint state identity is invalid");
        return false;
    }
    char expected_state[PATH_MAX];
    char metadata_path[PATH_MAX];
    if (!entry_paths(bundle, id, expected_state, sizeof(expected_state),
                     metadata_path, sizeof(metadata_path)) ||
        strcmp(expected_state, state_path) != 0) {
        bundle_error(error, error_size, "checkpoint state path is not generated");
        return false;
    }
    uint64_t state_bytes = 0u;
    if (!private_regular(state_path, &state_bytes, error, error_size) ||
        state_bytes != state_info->file_bytes) return false;
    size_t duplicate = SIZE_MAX;
    if (k3_prefix_bundle_find_exact(bundle, snapshot->tokens,
                                    snapshot->token_count, &duplicate)) {
        bundle_error(error, error_size, "checkpoint token entry already exists");
        return false;
    }
    bundle_entry added;
    if (!snapshot_clone(&added, id, state_path, metadata_path,
                        snapshot, state_info, error, error_size) ||
        !metadata_write(&added, error, error_size)) {
        entry_free(&added);
        return false;
    }
    if (!sync_directory(bundle->entries_root)) {
        bundle_error(error, error_size,
                     "syncing checkpoint entries failed: %s",
                     strerror(errno));
        (void)unlink(added.metadata_path);
        entry_free(&added);
        return false;
    }
    const uint64_t added_bytes = entry_total_bytes(&added);
    if (added_bytes > bundle->byte_limit) {
        bundle_error(error, error_size, "checkpoint exceeds byte limit");
        (void)unlink(added.metadata_path);
        entry_free(&added);
        return false;
    }
    bool *keep = bundle->count == 0u ? NULL :
        (bool *)malloc(bundle->count * sizeof(*keep));
    if (keep == NULL && bundle->count != 0u) {
        (void)unlink(added.metadata_path);
        entry_free(&added);
        return false;
    }
    uint64_t total = added_bytes;
    size_t kept = bundle->count;
    for (size_t index = 0u; index < bundle->count; index++) {
        keep[index] = true;
        if (total > UINT64_MAX - entry_total_bytes(&bundle->entries[index])) {
            free(keep);
            entry_free(&added);
            return false;
        }
        total += entry_total_bytes(&bundle->entries[index]);
    }
    while (kept + 1u > bundle->entry_limit || total > bundle->byte_limit) {
        const size_t evict = oldest_entry(bundle->entries, bundle->count, keep);
        if (evict == SIZE_MAX) break;
        keep[evict] = false;
        kept--;
        total -= entry_total_bytes(&bundle->entries[evict]);
    }
    bundle_entry *published = (bundle_entry *)calloc(kept + 1u, sizeof(*published));
    if (published == NULL) {
        free(keep);
        (void)unlink(added.metadata_path);
        entry_free(&added);
        return false;
    }
    size_t cursor = 0u;
    for (size_t index = 0u; index < bundle->count; index++) {
        if (keep[index]) published[cursor++] = bundle->entries[index];
    }
    added.publication_sequence = bundle->generation + 1u;
    added.last_use_sequence = added.publication_sequence;
    published[cursor++] = added;
    if (!manifest_write(bundle, published, cursor,
                        bundle->generation + 1u, error, error_size)) {
        free(published);
        free(keep);
        (void)unlink(added.metadata_path);
        entry_free(&added);
        return false;
    }
    for (size_t index = 0u; index < bundle->count; index++) {
        if (!keep[index]) {
            (void)unlink(bundle->entries[index].state_path);
            (void)unlink(bundle->entries[index].metadata_path);
            entry_free(&bundle->entries[index]);
        } else {
            memset(&bundle->entries[index], 0, sizeof(bundle->entries[index]));
        }
    }
    free(bundle->entries);
    free(keep);
    bundle->entries = published;
    bundle->count = cursor;
    bundle->generation++;
    return true;
}

bool k3_prefix_bundle_remove(k3_prefix_bundle *bundle, size_t index,
                             char *error, size_t error_size) {
    if (bundle == NULL || index >= bundle->count) return false;
    const size_t next_count = bundle->count - 1u;
    bundle_entry *next = next_count == 0u ? NULL :
        (bundle_entry *)calloc(next_count, sizeof(*next));
    if (next == NULL && next_count != 0u) return false;
    size_t cursor = 0u;
    for (size_t item = 0u; item < bundle->count; item++) {
        if (item != index) next[cursor++] = bundle->entries[item];
    }
    if (!manifest_write(bundle, next, next_count,
                        bundle->generation + 1u, error, error_size)) {
        free(next);
        return false;
    }
    (void)unlink(bundle->entries[index].state_path);
    (void)unlink(bundle->entries[index].metadata_path);
    entry_free(&bundle->entries[index]);
    for (size_t item = 0u; item < bundle->count; item++) {
        if (item != index) memset(&bundle->entries[item], 0,
                                  sizeof(bundle->entries[item]));
    }
    free(bundle->entries);
    bundle->entries = next;
    bundle->count = next_count;
    bundle->generation++;
    return true;
}
