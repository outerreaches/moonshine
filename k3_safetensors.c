#include "k3_safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

typedef struct {
    const char *p;
    const char *end;
} k3_json;

static void k3_set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (!error || error_size == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static void k3_json_ws(k3_json *json) {
    while (json->p < json->end &&
           (*json->p == ' ' || *json->p == '\n' ||
            *json->p == '\r' || *json->p == '\t')) {
        json->p++;
    }
}

static bool k3_json_take(k3_json *json, char expected) {
    k3_json_ws(json);
    if (json->p >= json->end || *json->p != expected) return false;
    json->p++;
    return true;
}

static bool k3_json_string(k3_json *json, char **out) {
    *out = NULL;
    k3_json_ws(json);
    if (json->p >= json->end || *json->p++ != '"') return false;

    const char *start = json->p;
    size_t max_bytes = (size_t)(json->end - start);
    char *value = malloc(max_bytes + 1u);
    if (!value) return false;

    size_t used = 0;
    while (json->p < json->end) {
        unsigned char c = (unsigned char)*json->p++;
        if (c == '"') {
            value[used] = '\0';
            *out = value;
            return true;
        }
        if (c < 0x20) break;
        if (c != '\\') {
            value[used++] = (char)c;
            continue;
        }
        if (json->p >= json->end) break;
        c = (unsigned char)*json->p++;
        switch (c) {
        case '"': case '\\': case '/': value[used++] = (char)c; break;
        case 'b': value[used++] = '\b'; break;
        case 'f': value[used++] = '\f'; break;
        case 'n': value[used++] = '\n'; break;
        case 'r': value[used++] = '\r'; break;
        case 't': value[used++] = '\t'; break;
        /*
         * Tensor names and dtype strings are ASCII. Reject escaped Unicode
         * instead of silently constructing a different lookup key.
         */
        default:
            free(value);
            return false;
        }
    }
    free(value);
    return false;
}

static bool k3_json_u64(k3_json *json, uint64_t *out) {
    k3_json_ws(json);
    if (json->p >= json->end || *json->p < '0' || *json->p > '9') {
        return false;
    }
    uint64_t value = 0;
    do {
        unsigned digit = (unsigned)(*json->p++ - '0');
        if (value > (UINT64_MAX - digit) / 10u) return false;
        value = value * 10u + digit;
    } while (json->p < json->end && *json->p >= '0' && *json->p <= '9');
    *out = value;
    return true;
}

static bool k3_json_skip(k3_json *json, unsigned depth);

static bool k3_json_skip_object(k3_json *json, unsigned depth) {
    if (!k3_json_take(json, '{')) return false;
    k3_json_ws(json);
    if (json->p < json->end && *json->p == '}') {
        json->p++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        if (!k3_json_string(json, &key) || !k3_json_take(json, ':')) {
            free(key);
            return false;
        }
        free(key);
        if (!k3_json_skip(json, depth + 1u)) return false;
        k3_json_ws(json);
        if (json->p < json->end && *json->p == '}') {
            json->p++;
            return true;
        }
        if (!k3_json_take(json, ',')) return false;
    }
}

static bool k3_json_skip_array(k3_json *json, unsigned depth) {
    if (!k3_json_take(json, '[')) return false;
    k3_json_ws(json);
    if (json->p < json->end && *json->p == ']') {
        json->p++;
        return true;
    }
    for (;;) {
        if (!k3_json_skip(json, depth + 1u)) return false;
        k3_json_ws(json);
        if (json->p < json->end && *json->p == ']') {
            json->p++;
            return true;
        }
        if (!k3_json_take(json, ',')) return false;
    }
}

static bool k3_json_skip(k3_json *json, unsigned depth) {
    if (depth > 64u) return false;
    k3_json_ws(json);
    if (json->p >= json->end) return false;
    if (*json->p == '{') return k3_json_skip_object(json, depth);
    if (*json->p == '[') return k3_json_skip_array(json, depth);
    if (*json->p == '"') {
        char *value = NULL;
        bool ok = k3_json_string(json, &value);
        free(value);
        return ok;
    }
    const char *start = json->p;
    while (json->p < json->end &&
           *json->p != ',' && *json->p != '}' && *json->p != ']' &&
           *json->p != ' ' && *json->p != '\n' &&
           *json->p != '\r' && *json->p != '\t') {
        json->p++;
    }
    return json->p != start;
}

static bool k3_json_u64_array(k3_json *json,
                              uint64_t *values,
                              size_t capacity,
                              size_t *count) {
    *count = 0;
    if (!k3_json_take(json, '[')) return false;
    k3_json_ws(json);
    if (json->p < json->end && *json->p == ']') {
        json->p++;
        return true;
    }
    for (;;) {
        uint64_t value = 0;
        if (*count >= capacity || !k3_json_u64(json, &value)) return false;
        values[(*count)++] = value;
        k3_json_ws(json);
        if (json->p < json->end && *json->p == ']') {
            json->p++;
            return true;
        }
        if (!k3_json_take(json, ',')) return false;
    }
}

static bool k3_append_tensor(k3_st_model *model, const k3_st_tensor *tensor) {
    if (model->tensor_count == model->tensor_capacity) {
        size_t capacity = model->tensor_capacity ?
            model->tensor_capacity * 2u : 8192u;
        if (capacity < model->tensor_capacity ||
            capacity > SIZE_MAX / sizeof(*model->tensors)) {
            return false;
        }
        void *next = realloc(model->tensors,
                             capacity * sizeof(*model->tensors));
        if (!next) return false;
        model->tensors = next;
        model->tensor_capacity = capacity;
    }
    model->tensors[model->tensor_count++] = *tensor;
    return true;
}

static k3_st_dtype k3_dtype_parse(const char *dtype) {
    if (strcmp(dtype, "BF16") == 0) return K3_ST_DTYPE_BF16;
    if (strcmp(dtype, "F32") == 0) return K3_ST_DTYPE_F32;
    if (strcmp(dtype, "U8") == 0) return K3_ST_DTYPE_U8;
    return K3_ST_DTYPE_UNKNOWN;
}

static uint64_t k3_dtype_bytes(k3_st_dtype dtype) {
    switch (dtype) {
    case K3_ST_DTYPE_BF16: return 2;
    case K3_ST_DTYPE_F32: return 4;
    case K3_ST_DTYPE_U8: return 1;
    default: return 0;
    }
}

static bool k3_tensor_shape_bytes(const k3_st_tensor *tensor,
                                  uint64_t *bytes) {
    uint64_t elements = 1;
    for (uint8_t i = 0; i < tensor->ndim; i++) {
        if (tensor->shape[i] != 0 &&
            elements > UINT64_MAX / tensor->shape[i]) {
            return false;
        }
        elements *= tensor->shape[i];
    }
    uint64_t width = k3_dtype_bytes(tensor->dtype);
    if (width == 0 || elements > UINT64_MAX / width) return false;
    *bytes = elements * width;
    return true;
}

static bool k3_parse_tensor_value(k3_json *json,
                                  k3_st_tensor *tensor,
                                  uint64_t data_offset,
                                  uint64_t file_bytes) {
    bool have_dtype = false;
    bool have_shape = false;
    bool have_offsets = false;
    uint64_t offsets[2] = {0, 0};
    size_t offset_count = 0;

    if (!k3_json_take(json, '{')) return false;
    for (;;) {
        k3_json_ws(json);
        if (json->p < json->end && *json->p == '}') {
            json->p++;
            break;
        }
        char *key = NULL;
        if (!k3_json_string(json, &key) || !k3_json_take(json, ':')) {
            free(key);
            return false;
        }
        bool ok = true;
        if (strcmp(key, "dtype") == 0) {
            char *dtype = NULL;
            ok = k3_json_string(json, &dtype);
            if (ok) {
                tensor->dtype = k3_dtype_parse(dtype);
                have_dtype = tensor->dtype != K3_ST_DTYPE_UNKNOWN;
            }
            free(dtype);
        } else if (strcmp(key, "shape") == 0) {
            size_t ndim = 0;
            ok = k3_json_u64_array(json, tensor->shape,
                                   K3_ST_MAX_DIMS, &ndim);
            if (ok && ndim <= UINT8_MAX) {
                tensor->ndim = (uint8_t)ndim;
                have_shape = true;
            } else {
                ok = false;
            }
        } else if (strcmp(key, "data_offsets") == 0) {
            ok = k3_json_u64_array(json, offsets, 2u, &offset_count);
            have_offsets = ok && offset_count == 2u;
        } else {
            ok = k3_json_skip(json, 0);
        }
        free(key);
        if (!ok) return false;
        k3_json_ws(json);
        if (json->p < json->end && *json->p == '}') {
            json->p++;
            break;
        }
        if (!k3_json_take(json, ',')) return false;
    }

    if (!have_dtype || !have_shape || !have_offsets ||
        offsets[1] < offsets[0] ||
        data_offset > file_bytes ||
        offsets[1] > file_bytes - data_offset) {
        return false;
    }
    tensor->physical_offset = data_offset + offsets[0];
    tensor->byte_length = offsets[1] - offsets[0];

    uint64_t shape_bytes = 0;
    return k3_tensor_shape_bytes(tensor, &shape_bytes) &&
           shape_bytes == tensor->byte_length;
}

static bool k3_parse_header(k3_st_model *model,
                            uint16_t shard,
                            const char *header,
                            uint64_t header_bytes,
                            char *error,
                            size_t error_size) {
    k3_json json = {header, header + header_bytes};
    k3_st_shard *source = &model->shards[shard];
    if (!k3_json_take(&json, '{')) {
        k3_set_error(error, error_size, "%s: header is not a JSON object",
                     source->path);
        return false;
    }
    for (;;) {
        k3_json_ws(&json);
        if (json.p < json.end && *json.p == '}') {
            json.p++;
            break;
        }
        char *name = NULL;
        if (!k3_json_string(&json, &name) || !k3_json_take(&json, ':')) {
            free(name);
            k3_set_error(error, error_size, "%s: malformed tensor key",
                         source->path);
            return false;
        }
        if (strcmp(name, "__metadata__") == 0) {
            free(name);
            if (!k3_json_skip(&json, 0)) {
                k3_set_error(error, error_size, "%s: malformed metadata",
                             source->path);
                return false;
            }
        } else {
            k3_st_tensor tensor;
            memset(&tensor, 0, sizeof(tensor));
            tensor.name = name;
            tensor.shard = shard;
            if (!k3_parse_tensor_value(&json, &tensor, source->data_offset,
                                       source->file_bytes)) {
                k3_set_error(error, error_size,
                             "%s: invalid tensor entry %s",
                             source->path, name);
                free(name);
                return false;
            }
            if (!k3_append_tensor(model, &tensor)) {
                free(name);
                k3_set_error(error, error_size,
                             "out of memory building tensor directory");
                return false;
            }
        }
        k3_json_ws(&json);
        if (json.p < json.end && *json.p == '}') {
            json.p++;
            break;
        }
        if (!k3_json_take(&json, ',')) {
            k3_set_error(error, error_size, "%s: malformed header delimiter",
                         source->path);
            return false;
        }
    }
    k3_json_ws(&json);
    if (json.p != json.end) {
        k3_set_error(error, error_size, "%s: trailing header content",
                     source->path);
        return false;
    }
    return true;
}

static int k3_tensor_compare(const void *left, const void *right) {
    const k3_st_tensor *a = left;
    const k3_st_tensor *b = right;
    return strcmp(a->name, b->name);
}

static bool k3_read_full(int fd, void *buffer, uint64_t bytes,
                         uint64_t offset) {
    uint8_t *out = buffer;
    uint64_t done = 0;
    while (done < bytes) {
        size_t request = bytes - done > (uint64_t)SSIZE_MAX ?
            (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t got = pread(fd, out + done, request, (off_t)(offset + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        done += (uint64_t)got;
    }
    return true;
}

static uint64_t k3_load_le64(const uint8_t bytes[8]) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++) {
        value |= (uint64_t)bytes[i] << (i * 8u);
    }
    return value;
}

bool k3_st_model_open(k3_st_model *model,
                      const char *root,
                      size_t shard_count,
                      char *error,
                      size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model || !root || shard_count == 0 || shard_count > UINT16_MAX) {
        k3_set_error(error, error_size, "invalid model-open arguments");
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->shards = calloc(shard_count, sizeof(*model->shards));
    if (!model->shards) {
        k3_set_error(error, error_size, "out of memory allocating shards");
        return false;
    }
    model->shard_count = shard_count;
    for (size_t i = 0; i < shard_count; i++) {
        model->shards[i].fd = -1;
        model->shards[i].direct_fd = -1;
    }

    for (size_t i = 0; i < shard_count; i++) {
        size_t path_bytes = strlen(root) + 64u;
        char *path = malloc(path_bytes);
        if (!path) {
            k3_set_error(error, error_size, "out of memory allocating path");
            goto fail;
        }
        snprintf(path, path_bytes, "%s/model-%05zu-of-%06zu.safetensors",
                 root, i + 1u, shard_count);
        k3_st_shard *shard = &model->shards[i];
        shard->path = path;
        shard->fd = open(path, O_RDONLY | O_CLOEXEC);
        if (shard->fd < 0) {
            k3_set_error(error, error_size, "open %s: %s",
                         path, strerror(errno));
            goto fail;
        }
        struct stat st;
        if (fstat(shard->fd, &st) != 0 || st.st_size < 8) {
            k3_set_error(error, error_size, "stat %s: %s",
                         path, strerror(errno));
            goto fail;
        }
        shard->file_bytes = (uint64_t)st.st_size;

        uint8_t prefix[8];
        if (!k3_read_full(shard->fd, prefix, sizeof(prefix), 0)) {
            k3_set_error(error, error_size, "read header prefix %s: %s",
                         path, strerror(errno));
            goto fail;
        }
        uint64_t header_bytes = k3_load_le64(prefix);
        if (header_bytes == 0 || header_bytes > shard->file_bytes - 8u ||
            header_bytes > SIZE_MAX - 1u) {
            k3_set_error(error, error_size,
                         "%s: invalid header length %" PRIu64,
                         path, header_bytes);
            goto fail;
        }
        shard->data_offset = 8u + header_bytes;
        char *header = malloc((size_t)header_bytes + 1u);
        if (!header) {
            k3_set_error(error, error_size,
                         "out of memory reading %" PRIu64 "-byte header",
                         header_bytes);
            goto fail;
        }
        if (!k3_read_full(shard->fd, header, header_bytes, 8u)) {
            free(header);
            k3_set_error(error, error_size, "read header %s: %s",
                         path, strerror(errno));
            goto fail;
        }
        header[header_bytes] = '\0';
        bool parsed = k3_parse_header(model, (uint16_t)i, header,
                                      header_bytes, error, error_size);
        free(header);
        if (!parsed) goto fail;

        if (O_DIRECT != 0) {
            shard->direct_fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
        }
    }

    qsort(model->tensors, model->tensor_count,
          sizeof(*model->tensors), k3_tensor_compare);
    for (size_t i = 1; i < model->tensor_count; i++) {
        if (strcmp(model->tensors[i - 1u].name,
                   model->tensors[i].name) == 0) {
            k3_set_error(error, error_size, "duplicate tensor name: %s",
                         model->tensors[i].name);
            goto fail;
        }
    }
    return true;

fail:
    k3_st_model_close(model);
    return false;
}

void k3_st_model_close(k3_st_model *model) {
    if (!model) return;
    for (size_t i = 0; i < model->tensor_count; i++) {
        free(model->tensors[i].name);
    }
    free(model->tensors);
    for (size_t i = 0; i < model->shard_count; i++) {
        if (model->shards[i].direct_fd >= 0) {
            close(model->shards[i].direct_fd);
        }
        if (model->shards[i].fd >= 0) close(model->shards[i].fd);
        free(model->shards[i].path);
    }
    free(model->shards);
    memset(model, 0, sizeof(*model));
}

const k3_st_tensor *k3_st_find(const k3_st_model *model, const char *name) {
    if (!model || !name) return NULL;
    size_t low = 0;
    size_t high = model->tensor_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        int cmp = strcmp(name, model->tensors[middle].name);
        if (cmp == 0) return &model->tensors[middle];
        if (cmp < 0) high = middle;
        else low = middle + 1u;
    }
    return NULL;
}

static bool k3_st_read_range(const k3_st_model *model,
                             uint16_t shard_index,
                             uint64_t physical_offset,
                             uint64_t byte_length,
                             uint64_t alignment,
                             uint64_t *aligned_start,
                             uint64_t *allocation_bytes,
                             char *error,
                             size_t error_size) {
    if (!model || shard_index >= model->shard_count || byte_length == 0 ||
        alignment < sizeof(void *) ||
        (alignment & (alignment - 1u)) != 0) {
        k3_set_error(error, error_size, "invalid read-span arguments");
        return false;
    }
    const k3_st_shard *shard = &model->shards[shard_index];
    if (physical_offset > shard->file_bytes ||
        byte_length > shard->file_bytes - physical_offset) {
        k3_set_error(error, error_size, "read span exceeds %s", shard->path);
        return false;
    }
    *aligned_start = physical_offset & ~(alignment - 1u);
    uint64_t data_end = physical_offset + byte_length;
    if (data_end > UINT64_MAX - (alignment - 1u)) {
        k3_set_error(error, error_size, "aligned read range overflow");
        return false;
    }
    uint64_t aligned_end =
        (data_end + alignment - 1u) & ~(alignment - 1u);
    if (aligned_end > shard->file_bytes) {
        /*
         * O_DIRECT cannot safely extend past EOF. The buffered fallback reads
         * the exact span while retaining an aligned destination allocation.
         */
        *aligned_start = physical_offset;
        aligned_end = data_end;
    }
    *allocation_bytes = aligned_end - *aligned_start;
    return true;
}

bool k3_st_read_span_into(const k3_st_model *model,
                          uint16_t shard_index,
                          uint64_t physical_offset,
                          uint64_t byte_length,
                          uint64_t alignment,
                          void *allocation,
                          uint64_t allocation_capacity,
                          k3_st_read_view *view,
                          char *error,
                          size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (view) memset(view, 0, sizeof(*view));
    uint64_t aligned_start = 0;
    uint64_t allocation_bytes = 0;
    if (!view || !allocation) {
        if (error && error_size && error[0] == '\0') {
            k3_set_error(error, error_size,
                         "invalid caller-owned read buffer");
        }
        return false;
    }
    if (!k3_st_read_range(model, shard_index, physical_offset, byte_length,
                          alignment, &aligned_start, &allocation_bytes,
                          error, error_size)) {
        return false;
    }
    if ((uintptr_t)allocation % alignment != 0) {
        k3_set_error(error, error_size,
                     "caller-owned read buffer is not %" PRIu64
                     "-byte aligned", alignment);
        return false;
    }
    if (allocation_bytes > allocation_capacity) {
        k3_set_error(error, error_size,
                     "read needs %" PRIu64 " bytes, buffer has %" PRIu64,
                     allocation_bytes, allocation_capacity);
        return false;
    }

    const k3_st_shard *shard = &model->shards[shard_index];
    int fd = shard->fd;
    bool direct = shard->direct_fd >= 0 &&
                  aligned_start % alignment == 0 &&
                  allocation_bytes % alignment == 0;
    if (direct) fd = shard->direct_fd;
    if (!k3_read_full(fd, allocation, allocation_bytes, aligned_start)) {
        int saved_errno = errno;
        if (!direct ||
            !k3_read_full(shard->fd, allocation, allocation_bytes,
                          aligned_start)) {
            k3_set_error(error, error_size, "read %s: %s",
                         shard->path, strerror(saved_errno));
            return false;
        }
        direct = false;
    }
    view->allocation_bytes = allocation_bytes;
    view->physical_offset = aligned_start;
    view->data = (uint8_t *)allocation +
                 (size_t)(physical_offset - aligned_start);
    view->data_bytes = byte_length;
    view->used_direct_io = direct;
    return true;
}

bool k3_st_read_span(const k3_st_model *model,
                     uint16_t shard_index,
                     uint64_t physical_offset,
                     uint64_t byte_length,
                     uint64_t alignment,
                     k3_st_read *read,
                     char *error,
                     size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (read) memset(read, 0, sizeof(*read));
    uint64_t aligned_start = 0;
    uint64_t allocation_bytes = 0;
    if (!read ||
        !k3_st_read_range(model, shard_index, physical_offset, byte_length,
                          alignment, &aligned_start, &allocation_bytes,
                          error, error_size)) {
        return false;
    }
    if (allocation_bytes > SIZE_MAX) {
        k3_set_error(error, error_size, "read span is too large");
        return false;
    }
    void *allocation = NULL;
    int alloc_rc = posix_memalign(&allocation, (size_t)alignment,
                                  (size_t)allocation_bytes);
    if (alloc_rc != 0) {
        k3_set_error(error, error_size, "aligned allocation: %s",
                     strerror(alloc_rc));
        return false;
    }

    k3_st_read_view view;
    if (!k3_st_read_span_into(model, shard_index, physical_offset, byte_length,
                              alignment, allocation, allocation_bytes, &view,
                              error, error_size)) {
        free(allocation);
        return false;
    }
    read->allocation = allocation;
    read->allocation_bytes = view.allocation_bytes;
    read->physical_offset = view.physical_offset;
    read->data = view.data;
    read->data_bytes = view.data_bytes;
    read->used_direct_io = view.used_direct_io;
    return true;
}

void k3_st_read_release(k3_st_read *read) {
    if (!read) return;
    free(read->allocation);
    memset(read, 0, sizeof(*read));
}
