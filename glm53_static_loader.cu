#include "glm53_static_loader.h"

#include <hip/hip_runtime.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct glm53_static_store {
    glm53_static_runtime_entry *entries;
    size_t entry_count;
    void *device_base;
    uint64_t device_bytes;
    glm53_static_loader_stats ledger;
};

typedef struct {
    glm53_static_loader_status status;
    char *error;
    size_t error_size;
} load_error;

static void fail_message(load_error *failure,
                         glm53_static_loader_status status,
                         const char *format, ...) {
    va_list arguments;
    if (failure->status != GLM53_STATIC_LOADER_OK) return;
    failure->status = status;
    if (!failure->error || failure->error_size == 0u) return;
    va_start(arguments, format);
    (void)vsnprintf(failure->error, failure->error_size, format, arguments);
    va_end(arguments);
}

static bool add_u64(uint64_t *value, uint64_t amount) {
    if (*value > UINT64_MAX - amount) return false;
    *value += amount;
    return true;
}

static uint64_t crc64_update(uint64_t crc, const void *source, size_t bytes) {
    const unsigned char *data = (const unsigned char *)source;
    const uint64_t polynomial = UINT64_C(0x42f0e1eba9ea3693);
    for (size_t i = 0u; i < bytes; ++i) {
        crc ^= (uint64_t)data[i] << 56u;
        for (unsigned bit = 0u; bit < 8u; ++bit)
            crc = (crc & (UINT64_C(1) << 63u)) != 0u ?
                (crc << 1u) ^ polynomial : crc << 1u;
    }
    return crc;
}

static uint64_t crc_le64(uint64_t crc, uint64_t value) {
    unsigned char encoded[8];
    for (unsigned i = 0u; i < 8u; ++i)
        encoded[i] = (unsigned char)(value >> (8u * i));
    return crc64_update(crc, encoded, sizeof(encoded));
}

static uint64_t crc_le32(uint64_t crc, uint32_t value) {
    unsigned char encoded[4];
    for (unsigned i = 0u; i < 4u; ++i)
        encoded[i] = (unsigned char)(value >> (8u * i));
    return crc64_update(crc, encoded, sizeof(encoded));
}

typedef struct {
    uint16_t shard;
    uint64_t start;
    uint64_t end;
} loader_source_range;

static int source_range_compare(const void *left, const void *right) {
    const loader_source_range *a = (const loader_source_range *)left;
    const loader_source_range *b = (const loader_source_range *)right;
    if (a->shard != b->shard) return a->shard < b->shard ? -1 : 1;
    if (a->start != b->start) return a->start < b->start ? -1 : 1;
    if (a->end != b->end) return a->end < b->end ? -1 : 1;
    return 0;
}

static bool tensor_identity_matches(const k3_st_tensor *a,
                                    const k3_st_tensor *b) {
    if (!a || !b || !a->name || !b->name || strcmp(a->name, b->name) != 0 ||
        a->physical_offset != b->physical_offset ||
        a->byte_length != b->byte_length || a->shard != b->shard ||
        a->ndim != b->ndim || a->dtype != b->dtype) return false;
    for (size_t i = 0u; i < a->ndim; ++i)
        if (a->shape[i] != b->shape[i]) return false;
    return true;
}

static bool layout_valid(const k3_st_model *model,
                         const glm53_static_layout *layout) {
    uint64_t logical = 0u;
    uint64_t previous_end = 0u;
    bool valid = false;
    loader_source_range *ranges = NULL;
    if (!model || !model->shards || model->shard_count == 0u ||
        !model->tensors || model->tensor_count == 0u ||
        !layout || !layout->built || !layout->entries ||
        layout->entry_count == 0u ||
        layout->tensor_count != layout->entry_count ||
        layout->padded_bytes == 0u || layout->padded_bytes > SIZE_MAX ||
        layout->entry_count > SIZE_MAX / sizeof(*ranges)) return false;
    ranges = (loader_source_range *)calloc(layout->entry_count,
                                            sizeof(*ranges));
    if (!ranges) return false;
    for (size_t i = 0u; i < layout->entry_count; ++i) {
        const glm53_static_layout_entry *entry = &layout->entries[i];
        const k3_st_tensor *tensor = entry->tensor;
        const k3_st_tensor *canonical;
        uint64_t device_end;
        if (!tensor || !tensor->name || tensor->name[0] == '\0') goto done;
        canonical = k3_st_find(model, tensor->name);
        if (!tensor_identity_matches(tensor, canonical) ||
            entry->logical_bytes == 0u ||
            entry->source_shard >= model->shard_count ||
            entry->source_shard != canonical->shard ||
            entry->source_physical_offset != canonical->physical_offset ||
            entry->logical_bytes != canonical->byte_length ||
            entry->dtype != canonical->dtype ||
            entry->device_offset % GLM53_STATIC_LAYOUT_ALIGNMENT != 0u ||
            entry->device_offset < previous_end ||
            entry->device_offset > UINT64_MAX - entry->logical_bytes ||
            logical > UINT64_MAX - entry->logical_bytes) goto done;
        device_end = entry->device_offset + entry->logical_bytes;
        if (device_end > layout->padded_bytes ||
            entry->source_physical_offset >
                model->shards[entry->source_shard].file_bytes ||
            entry->logical_bytes >
                model->shards[entry->source_shard].file_bytes -
                    entry->source_physical_offset ||
            (i != 0u && strcmp(layout->entries[i - 1u].tensor->name,
                              tensor->name) >= 0)) goto done;
        ranges[i].shard = entry->source_shard;
        ranges[i].start = entry->source_physical_offset;
        ranges[i].end = entry->source_physical_offset + entry->logical_bytes;
        logical += entry->logical_bytes;
        previous_end = device_end;
    }
    qsort(ranges, layout->entry_count, sizeof(*ranges), source_range_compare);
    for (size_t i = 1u; i < layout->entry_count; ++i)
        if (ranges[i - 1u].shard == ranges[i].shard &&
            ranges[i].start < ranges[i - 1u].end) goto done;
    valid = logical == layout->logical_bytes &&
            previous_end <= layout->padded_bytes;
done:
    free(ranges);
    return valid;
}

static bool cancelled_now(glm53_static_loader_cancel_fn cancelled,
                          void *context) {
    return cancelled && cancelled(context);
}

extern "C" glm53_static_loader_status glm53_static_loader_load(
        glm53_static_store **out,
        const k3_st_model *model,
        const glm53_static_layout *layout,
        glm53_static_loader_cancel_fn cancelled,
        void *cancel_context,
        glm53_static_loader_stats *stats,
        char *error,
        size_t error_size) {
    load_error failure = {GLM53_STATIC_LOADER_OK, error, error_size};
    glm53_static_loader_stats measured;
    glm53_static_store *store = NULL;
    hipStream_t stream = NULL;
    hipEvent_t event = NULL;
    void *host_allocation = NULL;
    unsigned char *io = NULL;
    uint64_t io_capacity = 0u;
    hipError_t hip_status = hipSuccess;

    memset(&measured, 0, sizeof(measured));
    if (error && error_size != 0u) error[0] = '\0';
    if (!out || !model || !layout) {
        fail_message(&failure, GLM53_STATIC_LOADER_INVALID_ARGUMENT,
                     "invalid static-loader arguments");
        goto finish;
    }
    if (!layout_valid(model, layout)) {
        fail_message(&failure, GLM53_STATIC_LOADER_BAD_LAYOUT,
                     "invalid static-loader layout");
        goto finish;
    }
    if (cancelled_now(cancelled, cancel_context)) {
        fail_message(&failure, GLM53_STATIC_LOADER_CANCELLED,
                     "static load cancelled before allocation");
        goto finish;
    }

    store = (glm53_static_store *)calloc(1u, sizeof(*store));
    if (!store) {
        fail_message(&failure, GLM53_STATIC_LOADER_ALLOCATION_FAILED,
                     "static-loader store allocation failed");
        goto finish;
    }
    store->entries = (glm53_static_runtime_entry *)calloc(
        layout->entry_count, sizeof(*store->entries));
    if (!store->entries) {
        fail_message(&failure, GLM53_STATIC_LOADER_ALLOCATION_FAILED,
                     "static-loader entry allocation failed");
        goto finish;
    }
    store->entry_count = layout->entry_count;
    for (size_t i = 0u; i < layout->entry_count; ++i) {
        const glm53_static_layout_entry *source = &layout->entries[i];
        glm53_static_runtime_entry *destination = &store->entries[i];
        destination->name = strdup(source->tensor->name);
        if (!destination->name) {
            fail_message(&failure, GLM53_STATIC_LOADER_ALLOCATION_FAILED,
                         "static-loader name allocation failed");
            goto finish;
        }
        destination->dtype = source->dtype;
        destination->device_offset = source->device_offset;
        destination->logical_bytes = source->logical_bytes;
    }

    hip_status = hipStreamCreateWithFlags(
        &stream, hipStreamNonBlocking);
    if (hip_status != hipSuccess) {
        fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                     "hipStreamCreateWithFlags: %s",
                     hipGetErrorString(hip_status));
        goto finish;
    }
    hip_status = hipEventCreateWithFlags(&event, hipEventDisableTiming);
    if (hip_status != hipSuccess) {
        fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                     "hipEventCreateWithFlags: %s",
                     hipGetErrorString(hip_status));
        goto finish;
    }
    hip_status = hipHostMalloc(&host_allocation,
                               (size_t)GLM53_STATIC_LOADER_IO_BYTES,
                               hipHostMallocMapped);
    if (hip_status != hipSuccess) {
        fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                     "hipHostMallocMapped: %s", hipGetErrorString(hip_status));
        goto finish;
    }
    {
        uintptr_t raw = (uintptr_t)host_allocation;
        uintptr_t aligned = (raw + (uintptr_t)GLM53_STATIC_LOADER_IO_ALIGNMENT - 1u) &
                            ~((uintptr_t)GLM53_STATIC_LOADER_IO_ALIGNMENT - 1u);
        uint64_t displacement = (uint64_t)(aligned - raw);
        io = (unsigned char *)aligned;
        io_capacity = GLM53_STATIC_LOADER_IO_BYTES - displacement;
    }
    hip_status = hipMalloc(&store->device_base, (size_t)layout->padded_bytes);
    if (hip_status != hipSuccess) {
        fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                     "hipMalloc static allocation: %s",
                     hipGetErrorString(hip_status));
        goto finish;
    }
    store->device_bytes = layout->padded_bytes;
    measured.device_allocation_bytes = layout->padded_bytes;
    measured.entry_count = layout->entry_count;

    for (size_t i = 0u; i < layout->entry_count; ++i) {
        const glm53_static_layout_entry *entry = &layout->entries[i];
        glm53_static_runtime_entry *runtime = &store->entries[i];
        const size_t name_bytes = strlen(runtime->name);
        uint64_t entry_crc = 0u;
        measured.aggregate_crc64_ecma = crc_le64(
            measured.aggregate_crc64_ecma, (uint64_t)name_bytes);
        measured.aggregate_crc64_ecma = crc64_update(
            measured.aggregate_crc64_ecma, runtime->name, name_bytes);
        measured.aggregate_crc64_ecma = crc_le32(
            measured.aggregate_crc64_ecma, (uint32_t)runtime->dtype);
        measured.aggregate_crc64_ecma = crc_le64(
            measured.aggregate_crc64_ecma, runtime->device_offset);
        measured.aggregate_crc64_ecma = crc_le64(
            measured.aggregate_crc64_ecma, runtime->logical_bytes);

        uint64_t done = 0u;
        while (done < entry->logical_bytes) {
            uint64_t chunk = entry->logical_bytes - done;
            k3_st_read_view view;
            char read_error[512];
            if (chunk > GLM53_STATIC_LOADER_CHUNK_BYTES)
                chunk = GLM53_STATIC_LOADER_CHUNK_BYTES;
            if (cancelled_now(cancelled, cancel_context)) {
                fail_message(&failure, GLM53_STATIC_LOADER_CANCELLED,
                             "static load cancelled before read of %s",
                             runtime->name);
                goto finish;
            }
            memset(&view, 0, sizeof(view));
            read_error[0] = '\0';
            if (!k3_st_read_span_into(
                    model, entry->source_shard,
                    entry->source_physical_offset + done, chunk,
                    GLM53_STATIC_LOADER_IO_ALIGNMENT, io, io_capacity,
                    &view, read_error, sizeof(read_error))) {
                fail_message(&failure, GLM53_STATIC_LOADER_READ_FAILED,
                             "read %s at offset %llu: %s", runtime->name,
                             (unsigned long long)done, read_error);
                goto finish;
            }
            if (view.data_bytes != chunk || view.allocation_bytes > io_capacity ||
                !add_u64(&measured.logical_read_bytes, chunk) ||
                !add_u64(&measured.physical_read_bytes,
                         view.allocation_bytes) ||
                !add_u64(&measured.read_requests, 1u)) {
                fail_message(&failure, GLM53_STATIC_LOADER_OVERFLOW,
                             "static-loader read ledger overflow");
                goto finish;
            }
            if (view.used_direct_io) {
                if (!add_u64(&measured.direct_logical_bytes, chunk) ||
                    !add_u64(&measured.direct_physical_bytes,
                             view.allocation_bytes) ||
                    !add_u64(&measured.direct_requests, 1u)) {
                    fail_message(&failure, GLM53_STATIC_LOADER_OVERFLOW,
                                 "static-loader direct ledger overflow");
                    goto finish;
                }
            } else if (!add_u64(&measured.buffered_logical_bytes, chunk) ||
                       !add_u64(&measured.buffered_physical_bytes,
                                view.allocation_bytes) ||
                       !add_u64(&measured.buffered_requests, 1u)) {
                fail_message(&failure, GLM53_STATIC_LOADER_OVERFLOW,
                             "static-loader buffered ledger overflow");
                goto finish;
            }
            entry_crc = crc64_update(entry_crc, view.data, (size_t)chunk);
            measured.aggregate_crc64_ecma = crc64_update(
                measured.aggregate_crc64_ecma, view.data, (size_t)chunk);
            unsigned char *device_destination =
                (unsigned char *)store->device_base +
                (size_t)(entry->device_offset + done);
            hip_status = hipMemcpyAsync(device_destination, view.data,
                                        (size_t)chunk,
                                        hipMemcpyHostToDevice, stream);
            if (hip_status != hipSuccess) {
                fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                             "hipMemcpyAsync %s: %s", runtime->name,
                             hipGetErrorString(hip_status));
                goto finish;
            }
            if (!add_u64(&measured.logical_submitted_bytes, chunk) ||
                !add_u64(&measured.submitted_requests, 1u)) {
                fail_message(&failure, GLM53_STATIC_LOADER_OVERFLOW,
                             "static-loader submitted ledger overflow");
                goto finish;
            }
            hip_status = hipEventRecord(event, stream);
            if (hip_status == hipSuccess)
                hip_status = hipEventSynchronize(event);
            if (hip_status != hipSuccess) {
                fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                             "HIP copy completion %s: %s", runtime->name,
                             hipGetErrorString(hip_status));
                goto finish;
            }
            if (!add_u64(&measured.logical_completed_bytes, chunk) ||
                !add_u64(&measured.completed_requests, 1u)) {
                fail_message(&failure, GLM53_STATIC_LOADER_OVERFLOW,
                             "static-loader completion ledger overflow");
                goto finish;
            }
            done += chunk;
            if (cancelled_now(cancelled, cancel_context)) {
                fail_message(&failure, GLM53_STATIC_LOADER_CANCELLED,
                             "static load cancelled after copy of %s",
                             runtime->name);
                goto finish;
            }
        }
        runtime->crc64_ecma = entry_crc;
    }

    if (measured.logical_read_bytes != layout->logical_bytes ||
        measured.logical_submitted_bytes != layout->logical_bytes ||
        measured.logical_completed_bytes != layout->logical_bytes ||
        measured.device_allocation_bytes != layout->padded_bytes ||
        measured.direct_logical_bytes + measured.buffered_logical_bytes !=
            measured.logical_read_bytes ||
        measured.direct_physical_bytes + measured.buffered_physical_bytes !=
            measured.physical_read_bytes ||
        measured.direct_requests + measured.buffered_requests !=
            measured.read_requests ||
        measured.read_requests != measured.submitted_requests ||
        measured.submitted_requests != measured.completed_requests) {
        fail_message(&failure, GLM53_STATIC_LOADER_LEDGER_MISMATCH,
                     "static-loader final ledger mismatch");
        goto finish;
    }
    store->ledger = measured;

finish:
    /* A failed event record/synchronize can leave a queued copy referring to
     * the one host staging allocation. Drain the stream before freeing or
     * reusing that host storage. Preserve the original failure message. */
    if (stream) {
        hipError_t cleanup = hipStreamSynchronize(stream);
        if (cleanup != hipSuccess)
            fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                         "hipStreamSynchronize during cleanup: %s",
                         hipGetErrorString(cleanup));
    }
    if (event) {
        hipError_t cleanup = hipEventDestroy(event);
        if (cleanup != hipSuccess)
            fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                         "hipEventDestroy: %s", hipGetErrorString(cleanup));
    }
    if (stream) {
        hipError_t cleanup = hipStreamDestroy(stream);
        if (cleanup != hipSuccess)
            fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                         "hipStreamDestroy: %s", hipGetErrorString(cleanup));
    }
    if (host_allocation) {
        hipError_t cleanup = hipHostFree(host_allocation);
        if (cleanup != hipSuccess)
            fail_message(&failure, GLM53_STATIC_LOADER_HIP_FAILED,
                         "hipHostFree: %s", hipGetErrorString(cleanup));
    }
    if (failure.status != GLM53_STATIC_LOADER_OK) {
        glm53_static_store_destroy(store);
        store = NULL;
    } else {
        *out = store;
    }
    if (stats) *stats = measured;
    return failure.status;
}

extern "C" const glm53_static_runtime_entry *glm53_static_store_find(
        const glm53_static_store *store, const char *name) {
    if (!store || !name) return NULL;
    size_t low = 0u, high = store->entry_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        int order = strcmp(store->entries[middle].name, name);
        if (order < 0) low = middle + 1u;
        else high = middle;
    }
    if (low == store->entry_count ||
        strcmp(store->entries[low].name, name) != 0) return NULL;
    return &store->entries[low];
}

extern "C" void *glm53_static_store_device_pointer(
        const glm53_static_store *store,
        const glm53_static_runtime_entry *entry) {
    if (!store || !entry || !store->device_base) return NULL;
    bool owned = false;
    for (size_t i = 0u; i < store->entry_count; ++i)
        if (&store->entries[i] == entry) { owned = true; break; }
    if (!owned || entry->device_offset > store->device_bytes ||
        entry->logical_bytes > store->device_bytes - entry->device_offset)
        return NULL;
    return (unsigned char *)store->device_base + (size_t)entry->device_offset;
}

extern "C" void *glm53_static_store_device_base(
        const glm53_static_store *store) {
    return store ? store->device_base : NULL;
}

extern "C" size_t glm53_static_store_entry_count(
        const glm53_static_store *store) {
    return store ? store->entry_count : 0u;
}

extern "C" uint64_t glm53_static_store_device_bytes(
        const glm53_static_store *store) {
    return store ? store->device_bytes : 0u;
}

extern "C" const glm53_static_loader_stats *glm53_static_store_stats(
        const glm53_static_store *store) {
    return store ? &store->ledger : NULL;
}

extern "C" void glm53_static_store_destroy(glm53_static_store *store) {
    if (!store) return;
    if (store->device_base) (void)hipFree(store->device_base);
    for (size_t i = 0u; i < store->entry_count; ++i)
        free(store->entries[i].name);
    free(store->entries);
    free(store);
}

extern "C" const char *glm53_static_loader_status_string(
        glm53_static_loader_status status) {
    switch (status) {
    case GLM53_STATIC_LOADER_OK: return "ok";
    case GLM53_STATIC_LOADER_INVALID_ARGUMENT: return "invalid argument";
    case GLM53_STATIC_LOADER_BAD_LAYOUT: return "bad layout";
    case GLM53_STATIC_LOADER_ALLOCATION_FAILED: return "allocation failed";
    case GLM53_STATIC_LOADER_READ_FAILED: return "read failed";
    case GLM53_STATIC_LOADER_HIP_FAILED: return "HIP failed";
    case GLM53_STATIC_LOADER_CANCELLED: return "cancelled";
    case GLM53_STATIC_LOADER_OVERFLOW: return "overflow";
    case GLM53_STATIC_LOADER_LEDGER_MISMATCH: return "ledger mismatch";
    default: return "unknown status";
    }
}
