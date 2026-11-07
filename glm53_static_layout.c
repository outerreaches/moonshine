#include "glm53_static_layout.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t size, const char *format, ...) {
    va_list ap;
    if (!error || size == 0u) return;
    va_start(ap, format);
    (void)vsnprintf(error, size, format, ap);
    va_end(ap);
}

static glm53_static_layout_status reject(
        char *error, size_t error_size, glm53_static_layout_status status,
        const char *message) {
    set_error(error, error_size, "%s", message);
    return status;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static int add_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static int align_up(uint64_t value, uint64_t *out) {
    uint64_t remainder = value % GLM53_STATIC_LAYOUT_ALIGNMENT;
    uint64_t extra = remainder == 0u ? 0u :
                     GLM53_STATIC_LAYOUT_ALIGNMENT - remainder;
    return add_u64(value, extra, out);
}

static int ledger_valid(const glm53_weight_ledger *ledger) {
    uint64_t total;
    if (!add_u64(ledger->f8_bytes, ledger->f32_bytes, &total) ||
        !add_u64(total, ledger->bf16_bytes, &total)) return 0;
    return total == ledger->total_bytes;
}

static int ledger_add_tensor(glm53_weight_ledger *ledger,
                             const glm53_static_layout_entry *entry) {
    uint64_t *dtype_bytes;
    uint64_t next;
    if (entry->dtype == K3_ST_DTYPE_F8_E4M3) dtype_bytes = &ledger->f8_bytes;
    else if (entry->dtype == K3_ST_DTYPE_F32) dtype_bytes = &ledger->f32_bytes;
    else if (entry->dtype == K3_ST_DTYPE_BF16) dtype_bytes = &ledger->bf16_bytes;
    else return 0;
    if (!add_u64(*dtype_bytes, entry->logical_bytes, &next)) return 0;
    *dtype_bytes = next;
    if (!add_u64(ledger->total_bytes, entry->logical_bytes, &next)) return 0;
    ledger->total_bytes = next;
    if (ledger->tensor_count == SIZE_MAX) return 0;
    ++ledger->tensor_count;
    return 1;
}

static int ledger_equal(const glm53_weight_ledger *a,
                        const glm53_weight_ledger *b) {
    return a->tensor_count == b->tensor_count &&
           a->total_bytes == b->total_bytes &&
           a->f8_bytes == b->f8_bytes &&
           a->f32_bytes == b->f32_bytes &&
           a->bf16_bytes == b->bf16_bytes;
}

static int entry_name_compare(const void *left, const void *right) {
    const glm53_static_layout_entry *a =
        (const glm53_static_layout_entry *)left;
    const glm53_static_layout_entry *b =
        (const glm53_static_layout_entry *)right;
    return strcmp(a->tensor->name, b->tensor->name);
}

static int tensor_metadata_valid(const k3_st_tensor *tensor) {
    uint64_t elements = 1u;
    uint64_t element_bytes;
    uint64_t expected_bytes;
    size_t i;
    if (!tensor || !tensor->name || tensor->name[0] == '\0' ||
        tensor->byte_length == 0u || tensor->ndim == 0u ||
        tensor->ndim > K3_ST_MAX_DIMS) return 0;
    if (tensor->dtype == K3_ST_DTYPE_F8_E4M3) element_bytes = 1u;
    else if (tensor->dtype == K3_ST_DTYPE_F32) element_bytes = 4u;
    else if (tensor->dtype == K3_ST_DTYPE_BF16) element_bytes = 2u;
    else return 0;
    for (i = 0u; i < tensor->ndim; ++i) {
        if (tensor->shape[i] == 0u ||
            elements > UINT64_MAX / tensor->shape[i]) return 0;
        elements *= tensor->shape[i];
    }
    if (elements > UINT64_MAX / element_bytes) return 0;
    expected_bytes = elements * element_bytes;
    return expected_bytes == tensor->byte_length &&
           tensor->physical_offset <= UINT64_MAX - tensor->byte_length;
}

static void copy_entry(glm53_static_layout_entry *entry,
                       const k3_st_tensor *tensor) {
    memset(entry, 0, sizeof(*entry));
    entry->tensor = tensor;
    entry->source_shard = tensor->shard;
    entry->source_physical_offset = tensor->physical_offset;
    entry->logical_bytes = tensor->byte_length;
    entry->dtype = tensor->dtype;
}

static glm53_static_layout_status append_tensor(
        glm53_static_layout_entry *entries, size_t capacity, size_t *count,
        uint64_t *class_bytes, const k3_st_tensor *tensor,
        int require_scale, char *error, size_t error_size) {
    uint64_t next;
    if (*count >= capacity)
        return reject(error, error_size, GLM53_STATIC_LAYOUT_BAD_PLAN,
                      "resident tensor count exceeds plan ledger");
    if (!tensor_metadata_valid(tensor))
        return reject(error, error_size, GLM53_STATIC_LAYOUT_BAD_PLAN,
                      "resident tensor metadata is invalid");
    if (require_scale && tensor->dtype != K3_ST_DTYPE_F32)
        return reject(error, error_size, GLM53_STATIC_LAYOUT_BAD_PLAN,
                      "routed resident scale is not F32");
    if (!add_u64(*class_bytes, tensor->byte_length, &next))
        return reject(error, error_size, GLM53_STATIC_LAYOUT_OVERFLOW,
                      "resident logical byte ledger overflows");
    *class_bytes = next;
    copy_entry(&entries[*count], tensor);
    ++*count;
    return GLM53_STATIC_LAYOUT_OK;
}

glm53_static_layout_status glm53_static_layout_build(
        glm53_static_layout *layout, const glm53_weight_plan *weights,
        char *error, size_t error_size) {
    glm53_static_layout next;
    glm53_static_layout_entry *entries = NULL;
    glm53_static_layout_status status;
    size_t expected, count = 0u, static_count = 0u;
    size_t i, j;
    uint64_t static_bytes = 0u, scale_bytes = 0u;
    uint64_t logical = 0u, padded = 0u;

    if (error && error_size != 0u) error[0] = '\0';
    if (!layout || !weights)
        return reject(error, error_size, GLM53_STATIC_LAYOUT_INVALID_ARGUMENT,
                      "invalid static-layout arguments");
    if (!weights->built)
        return reject(error, error_size, GLM53_STATIC_LAYOUT_UNBUILT_PLAN,
                      "weight plan is not built");
    if (!ledger_valid(&weights->resident_static) ||
        !ledger_valid(&weights->resident_routed_scales) ||
        (weights->routed_experts.expert_count != 0u &&
         !weights->routed_experts.experts))
        return reject(error, error_size, GLM53_STATIC_LAYOUT_BAD_PLAN,
                      "resident weight ledger or routed plan is invalid");
    if (!add_size(weights->resident_static.tensor_count,
                  weights->resident_routed_scales.tensor_count, &expected) ||
        (expected != 0u && expected > SIZE_MAX / sizeof(*entries)))
        return reject(error, error_size, GLM53_STATIC_LAYOUT_OVERFLOW,
                      "resident tensor allocation size overflows");
    if (expected == 0u)
        return reject(error, error_size, GLM53_STATIC_LAYOUT_BAD_PLAN,
                      "resident layout is empty");
    entries = (glm53_static_layout_entry *)calloc(expected, sizeof(*entries));
    if (!entries)
        return reject(error, error_size, GLM53_STATIC_LAYOUT_ALLOCATION_FAILED,
                      "static-layout allocation failed");

    for (i = 0u; i < GLM53_WEIGHT_GLOBAL_COUNT; ++i) {
        if (!weights->globals[i]) continue;
        status = append_tensor(entries, expected, &count, &static_bytes,
                               weights->globals[i], 0, error, error_size);
        if (status != GLM53_STATIC_LAYOUT_OK) goto fail;
        ++static_count;
    }
    for (i = 0u; i < GLM53_WEIGHT_LAYER_COUNT; ++i) {
        for (j = 0u; j < GLM53_WEIGHT_LAYER_ROLE_COUNT; ++j) {
            if (!weights->layers[i].roles[j]) continue;
            status = append_tensor(entries, expected, &count, &static_bytes,
                                   weights->layers[i].roles[j], 0,
                                   error, error_size);
            if (status != GLM53_STATIC_LAYOUT_OK) goto fail;
            ++static_count;
        }
    }
    if (static_count != weights->resident_static.tensor_count ||
        static_bytes != weights->resident_static.total_bytes) {
        status = reject(error, error_size, GLM53_STATIC_LAYOUT_BAD_PLAN,
                        "resident static entries disagree with ledger");
        goto fail;
    }

    /* Expert-plan role order is weight, scale, weight, scale, weight, scale.
     * Only the three odd roles enter this resident device allocation. */
    for (i = 0u; i < weights->routed_experts.expert_count; ++i) {
        const glm53_expert_plan *expert = &weights->routed_experts.experts[i];
        for (j = 1u; j < GLM53_EXPERT_TENSOR_COUNT; j += 2u) {
            if (!expert->tensors[j]) continue;
            status = append_tensor(entries, expected, &count, &scale_bytes,
                                   expert->tensors[j], 1, error, error_size);
            if (status != GLM53_STATIC_LAYOUT_OK) goto fail;
        }
    }
    if (count - static_count !=
            weights->resident_routed_scales.tensor_count ||
        scale_bytes != weights->resident_routed_scales.total_bytes ||
        count != expected) {
        status = reject(error, error_size, GLM53_STATIC_LAYOUT_BAD_PLAN,
                        "resident routed scales disagree with ledger");
        goto fail;
    }

    {
        glm53_weight_ledger observed_static;
        glm53_weight_ledger observed_scales;
        memset(&observed_static, 0, sizeof(observed_static));
        memset(&observed_scales, 0, sizeof(observed_scales));
        for (i = 0u; i < static_count; ++i)
            if (!ledger_add_tensor(&observed_static, &entries[i])) {
                status = reject(error, error_size, GLM53_STATIC_LAYOUT_OVERFLOW,
                                "resident static dtype ledger overflows");
                goto fail;
            }
        for (; i < count; ++i)
            if (!ledger_add_tensor(&observed_scales, &entries[i])) {
                status = reject(error, error_size, GLM53_STATIC_LAYOUT_OVERFLOW,
                                "resident scale dtype ledger overflows");
                goto fail;
            }
        if (!ledger_equal(&observed_static, &weights->resident_static) ||
            !ledger_equal(&observed_scales,
                          &weights->resident_routed_scales)) {
            status = reject(error, error_size, GLM53_STATIC_LAYOUT_BAD_PLAN,
                            "resident dtype subledgers disagree with tensors");
            goto fail;
        }
    }

    qsort(entries, count, sizeof(*entries), entry_name_compare);
    for (i = 1u; i < count; ++i) {
        if (entries[i - 1u].tensor == entries[i].tensor ||
            strcmp(entries[i - 1u].tensor->name,
                   entries[i].tensor->name) == 0) {
            status = reject(error, error_size,
                            GLM53_STATIC_LAYOUT_DUPLICATE_TENSOR,
                            "duplicate resident tensor");
            goto fail;
        }
    }
    for (i = 0u; i < count; ++i) {
        uint64_t end;
        entries[i].device_offset = padded;
        if ((entries[i].device_offset &
             (GLM53_STATIC_LAYOUT_ALIGNMENT - UINT64_C(1))) != 0u ||
            !add_u64(logical, entries[i].logical_bytes, &logical) ||
            !add_u64(padded, entries[i].logical_bytes, &end) ||
            !align_up(end, &padded)) {
            status = reject(error, error_size, GLM53_STATIC_LAYOUT_OVERFLOW,
                            "aligned static layout overflows");
            goto fail;
        }
    }
    memset(&next, 0, sizeof(next));
    next.entries = entries;
    next.entry_count = count;
    next.tensor_count = count;
    next.logical_bytes = logical;
    next.padded_bytes = padded;
    for (i = 0u; i < count; ++i)
        if (entries[i].logical_bytes > next.max_tensor_bytes)
            next.max_tensor_bytes = entries[i].logical_bytes;
    next.built = true;
    *layout = next;
    return GLM53_STATIC_LAYOUT_OK;

fail:
    free(entries);
    return status;
}

const glm53_static_layout_entry *glm53_static_layout_find_name(
        const glm53_static_layout *layout, const char *name) {
    size_t low = 0u, high;
    if (!layout || !layout->built || !layout->entries || !name) return NULL;
    high = layout->entry_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        int order = strcmp(layout->entries[middle].tensor->name, name);
        if (order < 0) low = middle + 1u;
        else high = middle;
    }
    if (low == layout->entry_count ||
        strcmp(layout->entries[low].tensor->name, name) != 0) return NULL;
    return &layout->entries[low];
}

const glm53_static_layout_entry *glm53_static_layout_find_tensor(
        const glm53_static_layout *layout, const k3_st_tensor *tensor) {
    const glm53_static_layout_entry *entry;
    if (!tensor || !tensor->name) return NULL;
    entry = glm53_static_layout_find_name(layout, tensor->name);
    return entry && entry->tensor == tensor ? entry : NULL;
}

const glm53_static_layout_entry *glm53_static_layout_find_by_name(
        const glm53_static_layout *layout, const char *name) {
    return glm53_static_layout_find_name(layout, name);
}

const glm53_static_layout_entry *glm53_static_layout_find_by_tensor(
        const glm53_static_layout *layout, const k3_st_tensor *tensor) {
    return glm53_static_layout_find_tensor(layout, tensor);
}

const char *glm53_static_layout_status_string(glm53_static_layout_status s) {
    switch (s) {
        case GLM53_STATIC_LAYOUT_OK: return "ok";
        case GLM53_STATIC_LAYOUT_INVALID_ARGUMENT: return "invalid argument";
        case GLM53_STATIC_LAYOUT_UNBUILT_PLAN: return "unbuilt plan";
        case GLM53_STATIC_LAYOUT_BAD_PLAN: return "bad plan";
        case GLM53_STATIC_LAYOUT_DUPLICATE_TENSOR: return "duplicate tensor";
        case GLM53_STATIC_LAYOUT_ALLOCATION_FAILED: return "allocation failed";
        case GLM53_STATIC_LAYOUT_OVERFLOW: return "overflow";
        default: return "unknown status";
    }
}

void glm53_static_layout_free(glm53_static_layout *layout) {
    if (!layout) return;
    free(layout->entries);
    memset(layout, 0, sizeof(*layout));
}
