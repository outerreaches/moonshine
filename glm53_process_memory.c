#include "glm53_process_memory.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define GLM53_LINE_CAP 16384
#define GLM53_PATH_CAP 4096
#define GLM53_GIB UINT64_C(1073741824)

static glm53_process_memory_status read_line(FILE *file, char *line,
                                               size_t capacity, bool *got) {
    size_t n;
    if (!fgets(line, (int)capacity, file)) {
        *got = false;
        return ferror(file) ? GLM53_PROCESS_MEMORY_IO_ERROR
                            : GLM53_PROCESS_MEMORY_OK;
    }
    *got = true;
    n = strlen(line);
    if (n != 0 && line[n - 1] == '\n') {
        line[--n] = '\0';
        if (n != 0 && line[n - 1] == '\r') line[n - 1] = '\0';
    } else if (!feof(file)) {
        return GLM53_PROCESS_MEMORY_MALFORMED;
    }
    return GLM53_PROCESS_MEMORY_OK;
}

/* 0: not this field, 1: value, negative: malformed/overflow. */
static int parse_kb_field(const char *line, const char *name,
                          uint64_t *bytes) {
    size_t name_len = strlen(name);
    const char *p;
    char *end;
    unsigned long long value;
    if (strncmp(line, name, name_len) != 0 || line[name_len] != ':') return 0;
    p = line + name_len + 1;
    while (isspace((unsigned char)*p)) ++p;
    if (!isdigit((unsigned char)*p)) return -1;
    errno = 0;
    value = strtoull(p, &end, 10);
    if (errno == ERANGE || end == p) return -2;
    p = end;
    while (isspace((unsigned char)*p)) ++p;
    if (p[0] != 'k' || p[1] != 'B') return -1;
    p += 2;
    while (isspace((unsigned char)*p)) ++p;
    if (*p != '\0') return -1;
    if (value > UINT64_MAX / UINT64_C(1024)) return -2;
    *bytes = (uint64_t)value * UINT64_C(1024);
    return 1;
}

static glm53_process_memory_status parse_two_fields(
    FILE *file, const char *first_name, const char *second_name,
    uint64_t *first_out, uint64_t *second_out) {
    char line[GLM53_LINE_CAP];
    uint64_t first = 0, second = 0, value;
    bool have_first = false, have_second = false, got;
    glm53_process_memory_status status;
    if (!file || !first_out || !second_out)
        return GLM53_PROCESS_MEMORY_INVALID_ARGUMENT;
    for (;;) {
        int parsed;
        status = read_line(file, line, sizeof(line), &got);
        if (status != GLM53_PROCESS_MEMORY_OK) return status;
        if (!got) break;
        parsed = parse_kb_field(line, first_name, &value);
        if (parsed != 0) {
            if (parsed == -2) return GLM53_PROCESS_MEMORY_OVERFLOW;
            if (parsed < 0) return GLM53_PROCESS_MEMORY_MALFORMED;
            if (have_first) return GLM53_PROCESS_MEMORY_DUPLICATE_FIELD;
            first = value;
            have_first = true;
            continue;
        }
        parsed = parse_kb_field(line, second_name, &value);
        if (parsed == -2) return GLM53_PROCESS_MEMORY_OVERFLOW;
        if (parsed < 0) return GLM53_PROCESS_MEMORY_MALFORMED;
        if (parsed > 0) {
            if (have_second) return GLM53_PROCESS_MEMORY_DUPLICATE_FIELD;
            second = value;
            have_second = true;
        }
    }
    if (!have_first || !have_second)
        return GLM53_PROCESS_MEMORY_MISSING_FIELD;
    *first_out = first;
    *second_out = second;
    return GLM53_PROCESS_MEMORY_OK;
}

glm53_process_memory_status glm53_process_memory_parse_meminfo(
    FILE *file, uint64_t *mem_available_bytes, uint64_t *swap_free_bytes) {
    return parse_two_fields(file, "MemAvailable", "SwapFree",
                            mem_available_bytes, swap_free_bytes);
}

glm53_process_memory_status glm53_process_memory_parse_status(
    FILE *file, uint64_t *vm_swap_bytes) {
    char line[GLM53_LINE_CAP];
    uint64_t result = 0, value;
    bool have = false, got;
    glm53_process_memory_status status;
    if (!file || !vm_swap_bytes) return GLM53_PROCESS_MEMORY_INVALID_ARGUMENT;
    for (;;) {
        int parsed;
        status = read_line(file, line, sizeof(line), &got);
        if (status != GLM53_PROCESS_MEMORY_OK) return status;
        if (!got) break;
        parsed = parse_kb_field(line, "VmSwap", &value);
        if (parsed == -2) return GLM53_PROCESS_MEMORY_OVERFLOW;
        if (parsed < 0) return GLM53_PROCESS_MEMORY_MALFORMED;
        if (parsed > 0) {
            if (have) return GLM53_PROCESS_MEMORY_DUPLICATE_FIELD;
            result = value;
            have = true;
        }
    }
    if (!have) return GLM53_PROCESS_MEMORY_MISSING_FIELD;
    *vm_swap_bytes = result;
    return GLM53_PROCESS_MEMORY_OK;
}

glm53_process_memory_status glm53_process_memory_parse_smaps_rollup(
    FILE *file, uint64_t *swap_bytes, uint64_t *swap_pss_bytes) {
    return parse_two_fields(file, "Swap", "SwapPss", swap_bytes,
                            swap_pss_bytes);
}

static bool parse_uint_token(const char **cursor, int base, char terminator,
                             uint64_t *value) {
    const char *p = *cursor;
    char *end;
    unsigned long long parsed;
    if ((base == 16 && !isxdigit((unsigned char)*p)) ||
        (base == 10 && !isdigit((unsigned char)*p))) return false;
    errno = 0;
    parsed = strtoull(p, &end, base);
    if (end == p || errno == ERANGE || (terminator && *end != terminator) ||
        parsed > UINT64_MAX) return false;
    *value = (uint64_t)parsed;
    *cursor = end + (terminator ? 1 : 0);
    return true;
}

static bool skip_token(const char **cursor) {
    const char *p = *cursor;
    if (*p == '\0' || isspace((unsigned char)*p)) return false;
    while (*p && !isspace((unsigned char)*p)) ++p;
    *cursor = p;
    return true;
}

static bool require_space(const char **cursor) {
    const char *p = *cursor;
    if (!isspace((unsigned char)*p)) return false;
    while (isspace((unsigned char)*p)) ++p;
    *cursor = p;
    return true;
}

static bool valid_hex_token(const char *begin, const char *end) {
    if (begin == end) return false;
    while (begin != end) {
        if (!isxdigit((unsigned char)*begin++)) return false;
    }
    return true;
}

static bool valid_permissions(const char *begin, const char *end) {
    return end - begin == 4 && (begin[0] == 'r' || begin[0] == '-') &&
           (begin[1] == 'w' || begin[1] == '-') &&
           (begin[2] == 'x' || begin[2] == '-') &&
           (begin[3] == 'p' || begin[3] == 's');
}

static bool path_has_suffix(const char *path, const char *suffix) {
    static const char deleted[] = " (deleted)";
    size_t path_len = strlen(path), suffix_len = strlen(suffix);
    const size_t deleted_len = sizeof(deleted) - 1u;
    if (path_len >= deleted_len &&
        memcmp(path + path_len - deleted_len, deleted, deleted_len) == 0)
        path_len -= deleted_len;
    return path_len >= suffix_len &&
           memcmp(path + path_len - suffix_len, suffix, suffix_len) == 0;
}

glm53_process_memory_status glm53_process_memory_parse_maps(
    FILE *file, const char *model_path_prefix, uint64_t *vma_count,
    uint64_t *vma_bytes, uint64_t *largest_vma_bytes) {
    char line[GLM53_LINE_CAP];
    uint64_t count = 0, total = 0, largest = 0;
    size_t prefix_len = model_path_prefix ? strlen(model_path_prefix) : 0;
    bool got;
    glm53_process_memory_status status;
    if (!file || !vma_count || !vma_bytes || !largest_vma_bytes)
        return GLM53_PROCESS_MEMORY_INVALID_ARGUMENT;
    for (;;) {
        const char *p, *token_begin, *path = NULL;
        uint64_t start, end, ignored, bytes;
        status = read_line(file, line, sizeof(line), &got);
        if (status != GLM53_PROCESS_MEMORY_OK) return status;
        if (!got) break;
        if (line[0] == '\0') return GLM53_PROCESS_MEMORY_MALFORMED;
        p = line;
        if (!parse_uint_token(&p, 16, '-', &start) ||
            !parse_uint_token(&p, 16, '\0', &end) ||
            !require_space(&p)) return GLM53_PROCESS_MEMORY_MALFORMED;
        /* permissions */
        token_begin = p;
        if (!skip_token(&p) || !valid_permissions(token_begin, p) ||
            !require_space(&p)) return GLM53_PROCESS_MEMORY_MALFORMED;
        /* file offset */
        token_begin = p;
        if (!skip_token(&p) || !valid_hex_token(token_begin, p) ||
            !require_space(&p)) return GLM53_PROCESS_MEMORY_MALFORMED;
        /* device major:minor */
        token_begin = p;
        while (*p && !isspace((unsigned char)*p) && *p != ':') ++p;
        if (*p != ':' || !valid_hex_token(token_begin, p))
            return GLM53_PROCESS_MEMORY_MALFORMED;
        token_begin = ++p;
        while (*p && !isspace((unsigned char)*p)) ++p;
        if (!valid_hex_token(token_begin, p) || !require_space(&p))
            return GLM53_PROCESS_MEMORY_MALFORMED;
        /* inode */
        if (!parse_uint_token(&p, 10, '\0', &ignored))
            return GLM53_PROCESS_MEMORY_MALFORMED;
        if (*p != '\0') {
            if (!require_space(&p)) return GLM53_PROCESS_MEMORY_MALFORMED;
            if (*p != '\0') path = p;
        }
        if (end <= start) return GLM53_PROCESS_MEMORY_MALFORMED;
        if (!path || !((prefix_len != 0 &&
                        strncmp(path, model_path_prefix, prefix_len) == 0 &&
                        (model_path_prefix[prefix_len - 1u] == '/' ||
                         path[prefix_len] == '\0' ||
                         path[prefix_len] == '/')) ||
                       path_has_suffix(path, ".safetensors"))) continue;
        bytes = end - start;
        if (count == UINT64_MAX || total > UINT64_MAX - bytes)
            return GLM53_PROCESS_MEMORY_OVERFLOW;
        ++count;
        total += bytes;
        if (bytes > largest) largest = bytes;
    }
    *vma_count = count;
    *vma_bytes = total;
    *largest_vma_bytes = largest;
    return GLM53_PROCESS_MEMORY_OK;
}

static glm53_process_memory_status open_parse(
    const char *path, int kind, const char *prefix, glm53_process_memory *s) {
    FILE *file = fopen(path, "r");
    glm53_process_memory_status status;
    int close_status;
    if (!file) return GLM53_PROCESS_MEMORY_OPEN_FAILED;
    if (kind == 0)
        status = glm53_process_memory_parse_meminfo(
            file, &s->mem_available_bytes, &s->swap_free_bytes);
    else if (kind == 1)
        status = glm53_process_memory_parse_status(file, &s->vm_swap_bytes);
    else if (kind == 2)
        status = glm53_process_memory_parse_smaps_rollup(
            file, &s->smaps_swap_bytes, &s->smaps_swap_pss_bytes);
    else
        status = glm53_process_memory_parse_maps(
            file, prefix, &s->model_vma_count, &s->model_vma_bytes,
            &s->largest_model_vma_bytes);
    close_status = fclose(file);
    if (status == GLM53_PROCESS_MEMORY_OK && close_status != 0)
        status = GLM53_PROCESS_MEMORY_IO_ERROR;
    return status;
}

glm53_process_memory_status glm53_process_memory_sample_root(
    const char *proc_root, const char *model_path_prefix,
    glm53_process_memory *out) {
    static const char *const relative[] = {
        "meminfo", "self/status", "self/smaps_rollup", "self/maps"
    };
    glm53_process_memory result;
    glm53_process_memory_status status;
    char path[GLM53_PATH_CAP];
    size_t i;
    if (!proc_root || !*proc_root || !out)
        return GLM53_PROCESS_MEMORY_INVALID_ARGUMENT;
    memset(&result, 0, sizeof(result));
    for (i = 0; i < sizeof(relative) / sizeof(relative[0]); ++i) {
        int n = snprintf(path, sizeof(path), "%s%s%s", proc_root,
                         proc_root[strlen(proc_root) - 1] == '/' ? "" : "/",
                         relative[i]);
        if (n < 0 || (size_t)n >= sizeof(path))
            return GLM53_PROCESS_MEMORY_INVALID_ARGUMENT;
        status = open_parse(path, (int)i, model_path_prefix, &result);
        if (status != GLM53_PROCESS_MEMORY_OK) return status;
    }
    *out = result;
    return GLM53_PROCESS_MEMORY_OK;
}

glm53_process_memory_status glm53_process_memory_sample(
    const char *model_path_prefix, glm53_process_memory *out) {
    return glm53_process_memory_sample_root("/proc", model_path_prefix, out);
}

bool glm53_process_memory_policy_allows(
    const glm53_process_memory *sample, uint64_t remaining_uncommitted_bytes) {
    uint64_t reserve = UINT64_C(16) * GLM53_GIB;
    uint64_t needed;
    if (!sample || remaining_uncommitted_bytes > UINT64_MAX - reserve)
        return false;
    needed = remaining_uncommitted_bytes + reserve;
    return sample->mem_available_bytes >= needed &&
           sample->vm_swap_bytes == 0 && sample->smaps_swap_bytes == 0 &&
           sample->model_vma_count == 0;
}

const char *glm53_process_memory_status_string(
    glm53_process_memory_status status) {
    switch (status) {
        case GLM53_PROCESS_MEMORY_OK: return "ok";
        case GLM53_PROCESS_MEMORY_INVALID_ARGUMENT: return "invalid argument";
        case GLM53_PROCESS_MEMORY_OPEN_FAILED: return "open failed";
        case GLM53_PROCESS_MEMORY_IO_ERROR: return "I/O error";
        case GLM53_PROCESS_MEMORY_MISSING_FIELD: return "missing field";
        case GLM53_PROCESS_MEMORY_DUPLICATE_FIELD: return "duplicate field";
        case GLM53_PROCESS_MEMORY_MALFORMED: return "malformed input";
        case GLM53_PROCESS_MEMORY_OVERFLOW: return "overflow";
        default: return "unknown";
    }
}
