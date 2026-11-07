#include "../glm53_static_loader.h"

#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    goto fail; } } while (0)
#define HIP_CHECK(expression) do { hipError_t h_ = (expression); if (h_ != hipSuccess) { \
    fprintf(stderr, "FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, #expression, \
            hipGetErrorString(h_)); goto fail; } } while (0)

static const uint64_t BIG_BYTES =
    GLM53_STATIC_LOADER_CHUNK_BYTES + UINT64_C(123);
static const uint64_t TAIL_BYTES = UINT64_C(5003);

static unsigned char payload_byte(uint64_t position) {
    uint64_t value = position * UINT64_C(6364136223846793005) +
                     UINT64_C(1442695040888963407);
    return (unsigned char)((value >> 29u) ^ (value >> 47u));
}

static bool write_full(int fd, const void *data, size_t bytes) {
    const unsigned char *source = (const unsigned char *)data;
    size_t done = 0u;
    while (done < bytes) {
        ssize_t written = write(fd, source + done, bytes - done);
        if (written < 0) return false;
        done += (size_t)written;
    }
    return true;
}

static bool make_fixture(char path[64]) {
    (void)strcpy(path, "/tmp/glm53-loader-XXXXXX");
    int fd = mkstemp(path);
    if (fd < 0) return false;
    char header[1024];
    int header_length = snprintf(
        header, sizeof(header),
        "{\"alpha.big\":{\"dtype\":\"F8_E4M3\",\"shape\":[%llu],"
        "\"data_offsets\":[0,%llu]},"
        "\"zeta.tail\":{\"dtype\":\"F8_E4M3\",\"shape\":[%llu],"
        "\"data_offsets\":[%llu,%llu]}} ",
        (unsigned long long)BIG_BYTES, (unsigned long long)BIG_BYTES,
        (unsigned long long)TAIL_BYTES, (unsigned long long)BIG_BYTES,
        (unsigned long long)(BIG_BYTES + TAIL_BYTES));
    if (header_length <= 0 || (size_t)header_length >= sizeof(header)) {
        (void)close(fd); (void)unlink(path); return false;
    }
    unsigned char prefix[8];
    for (unsigned i = 0u; i < 8u; ++i)
        prefix[i] = (unsigned char)((uint64_t)header_length >> (8u * i));
    bool ok = write_full(fd, prefix, sizeof(prefix)) &&
              write_full(fd, header, (size_t)header_length);
    unsigned char block[65536];
    uint64_t total = BIG_BYTES + TAIL_BYTES;
    uint64_t done = 0u;
    while (ok && done < total) {
        size_t amount = (size_t)(total - done > sizeof(block) ?
                                 sizeof(block) : total - done);
        for (size_t i = 0u; i < amount; ++i)
            block[i] = payload_byte(done + i);
        ok = write_full(fd, block, amount);
        done += amount;
    }
    if (close(fd) != 0) ok = false;
    if (!ok) (void)unlink(path);
    return ok;
}

static uint64_t crc_update(uint64_t crc, const void *source, size_t bytes) {
    const unsigned char *data = (const unsigned char *)source;
    for (size_t i = 0u; i < bytes; ++i) {
        crc ^= (uint64_t)data[i] << 56u;
        for (unsigned bit = 0u; bit < 8u; ++bit)
            crc = (crc & (UINT64_C(1) << 63u)) ?
                (crc << 1u) ^ UINT64_C(0x42f0e1eba9ea3693) : crc << 1u;
    }
    return crc;
}

static uint64_t crc_le64(uint64_t crc, uint64_t value) {
    unsigned char bytes[8];
    for (unsigned i = 0u; i < 8u; ++i)
        bytes[i] = (unsigned char)(value >> (8u * i));
    return crc_update(crc, bytes, sizeof(bytes));
}

static uint64_t crc_le32(uint64_t crc, uint32_t value) {
    unsigned char bytes[4];
    for (unsigned i = 0u; i < 4u; ++i)
        bytes[i] = (unsigned char)(value >> (8u * i));
    return crc_update(crc, bytes, sizeof(bytes));
}

static uint64_t expected_entry_crc(uint64_t start, uint64_t bytes) {
    unsigned char block[65536];
    uint64_t crc = 0u, done = 0u;
    while (done < bytes) {
        size_t amount = (size_t)(bytes - done > sizeof(block) ?
                                 sizeof(block) : bytes - done);
        for (size_t i = 0u; i < amount; ++i)
            block[i] = payload_byte(start + done + i);
        crc = crc_update(crc, block, amount);
        done += amount;
    }
    return crc;
}

static uint64_t expected_aggregate(const glm53_static_layout *layout) {
    unsigned char block[65536];
    uint64_t aggregate = 0u;
    uint64_t starts[2] = {0u, BIG_BYTES};
    for (size_t entry = 0u; entry < 2u; ++entry) {
        const char *name = layout->entries[entry].tensor->name;
        size_t name_bytes = strlen(name);
        aggregate = crc_le64(aggregate, name_bytes);
        aggregate = crc_update(aggregate, name, name_bytes);
        aggregate = crc_le32(aggregate, (uint32_t)layout->entries[entry].dtype);
        aggregate = crc_le64(aggregate, layout->entries[entry].device_offset);
        aggregate = crc_le64(aggregate, layout->entries[entry].logical_bytes);
        uint64_t done = 0u;
        while (done < layout->entries[entry].logical_bytes) {
            size_t amount = (size_t)(layout->entries[entry].logical_bytes - done >
                                      sizeof(block) ? sizeof(block) :
                                      layout->entries[entry].logical_bytes - done);
            for (size_t i = 0u; i < amount; ++i)
                block[i] = payload_byte(starts[entry] + done + i);
            aggregate = crc_update(aggregate, block, amount);
            done += amount;
        }
    }
    return aggregate;
}

typedef struct { unsigned calls; unsigned cancel_on; } cancel_state;
static bool cancel_at(void *context) {
    cancel_state *state = (cancel_state *)context;
    ++state->calls;
    return state->calls == state->cancel_on;
}

int main(void) {
    int result = 1;
    int device_count = 0;
    char path[64] = {0};
    char error[512];
    k3_st_model model;
    glm53_static_layout layout;
    glm53_static_layout_entry layout_entries[2];
    glm53_static_store *store = NULL;
    glm53_static_store *sentinel = NULL;
    glm53_static_loader_stats partial;
    glm53_static_loader_stats stats;
    cancel_state cancellation;
    hipDeviceProp_t properties;
    const glm53_static_runtime_entry *big = NULL;
    const glm53_static_runtime_entry *tail = NULL;
    const glm53_static_runtime_entry *runtime_entries[2] = {NULL, NULL};
    uint64_t big_offset = 0u;
    uint64_t tail_offset = 0u;
    uint64_t starts[2] = {0u, BIG_BYTES};
    uint64_t saved_source_offset = 0u;
    k3_st_tensor forged_tensor;
    unsigned char *host = NULL;
    memset(&model, 0, sizeof(model));
    memset(&layout, 0, sizeof(layout));

    HIP_CHECK(hipGetDeviceCount(&device_count));
    CHECK(device_count > 0);
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    CHECK(strstr(properties.gcnArchName, "gfx1151") != NULL);
    CHECK(make_fixture(path));
    CHECK(k3_st_model_open_file(&model, path, error, sizeof(error)));
    CHECK(model.tensor_count == 2u);
    CHECK(model.tensors[0].physical_offset % 4096u != 0u);

    memset(layout_entries, 0, sizeof(layout_entries));
    layout_entries[0].tensor = &model.tensors[0];
    layout_entries[0].source_shard = model.tensors[0].shard;
    layout_entries[0].source_physical_offset = model.tensors[0].physical_offset;
    layout_entries[0].logical_bytes = model.tensors[0].byte_length;
    layout_entries[0].dtype = model.tensors[0].dtype;
    layout_entries[0].device_offset = 0u;
    layout_entries[1].tensor = &model.tensors[1];
    layout_entries[1].source_shard = model.tensors[1].shard;
    layout_entries[1].source_physical_offset = model.tensors[1].physical_offset;
    layout_entries[1].logical_bytes = model.tensors[1].byte_length;
    layout_entries[1].dtype = model.tensors[1].dtype;
    layout_entries[1].device_offset = (BIG_BYTES + 255u) & ~UINT64_C(255);
    layout.entries = layout_entries;
    layout.entry_count = 2u;
    layout.tensor_count = 2u;
    layout.logical_bytes = BIG_BYTES + TAIL_BYTES;
    layout.padded_bytes = (layout_entries[1].device_offset + TAIL_BYTES + 255u) &
                          ~UINT64_C(255);
    layout.max_tensor_bytes = BIG_BYTES;
    layout.built = true;

    sentinel = (glm53_static_store *)(uintptr_t)1u;
    memset(&partial, 0, sizeof(partial));
    forged_tensor = model.tensors[0];
    forged_tensor.physical_offset += 1u;
    layout_entries[0].tensor = &forged_tensor;
    layout_entries[0].source_physical_offset = forged_tensor.physical_offset;
    CHECK(glm53_static_loader_load(&sentinel, &model, &layout,
              NULL, NULL, &partial, error, sizeof(error)) ==
          GLM53_STATIC_LOADER_BAD_LAYOUT);
    CHECK(sentinel == (glm53_static_store *)(uintptr_t)1u);
    layout_entries[0].tensor = &model.tensors[0];
    layout_entries[0].source_physical_offset = model.tensors[0].physical_offset;

    saved_source_offset = model.tensors[1].physical_offset;
    model.tensors[1].physical_offset = model.tensors[0].physical_offset + 1u;
    layout_entries[1].source_physical_offset = model.tensors[1].physical_offset;
    CHECK(glm53_static_loader_load(&sentinel, &model, &layout,
              NULL, NULL, &partial, error, sizeof(error)) ==
          GLM53_STATIC_LOADER_BAD_LAYOUT);
    model.tensors[1].physical_offset = saved_source_offset;
    layout_entries[1].source_physical_offset = saved_source_offset;

    memset(&partial, 0, sizeof(partial));
    cancellation.calls = 0u;
    cancellation.cancel_on = 1u;
    CHECK(glm53_static_loader_load(&sentinel, &model, &layout,
              cancel_at, &cancellation, &partial, error, sizeof(error)) ==
          GLM53_STATIC_LOADER_CANCELLED);
    CHECK(sentinel == (glm53_static_store *)(uintptr_t)1u);
    CHECK(partial.device_allocation_bytes == 0u &&
          partial.logical_read_bytes == 0u);

    cancellation.calls = 0u;
    cancellation.cancel_on = 3u; /* pre-allocation, pre-read, post-copy */
    memset(&partial, 0, sizeof(partial));
    CHECK(glm53_static_loader_load(&sentinel, &model, &layout,
              cancel_at, &cancellation, &partial, error, sizeof(error)) ==
          GLM53_STATIC_LOADER_CANCELLED);
    CHECK(sentinel == (glm53_static_store *)(uintptr_t)1u);
    CHECK(partial.logical_read_bytes == GLM53_STATIC_LOADER_CHUNK_BYTES);
    CHECK(partial.logical_submitted_bytes == partial.logical_read_bytes &&
          partial.logical_completed_bytes == partial.logical_read_bytes);

    memset(&stats, 0, sizeof(stats));
    CHECK(glm53_static_loader_load(&store, &model, &layout, NULL, NULL,
                                   &stats, error, sizeof(error)) ==
          GLM53_STATIC_LOADER_OK);
    CHECK(store != NULL && glm53_static_store_entry_count(store) == 2u);
    CHECK(glm53_static_store_device_bytes(store) == layout.padded_bytes);
    CHECK(stats.logical_read_bytes == layout.logical_bytes &&
          stats.logical_submitted_bytes == layout.logical_bytes &&
          stats.logical_completed_bytes == layout.logical_bytes);
    CHECK(stats.device_allocation_bytes == layout.padded_bytes);
    CHECK(stats.read_requests == 3u && stats.submitted_requests == 3u &&
          stats.completed_requests == 3u);
    CHECK(stats.direct_requests + stats.buffered_requests == 3u);
    CHECK(stats.buffered_requests >= 1u); /* EOF tail cannot use O_DIRECT. */
    CHECK(stats.direct_logical_bytes + stats.buffered_logical_bytes ==
          stats.logical_read_bytes);
    CHECK(stats.direct_physical_bytes + stats.buffered_physical_bytes ==
          stats.physical_read_bytes);
    CHECK(stats.aggregate_crc64_ecma == expected_aggregate(&layout));

    big = glm53_static_store_find(store, "alpha.big");
    tail = glm53_static_store_find(store, "zeta.tail");
    CHECK(big && tail && !glm53_static_store_find(store, "missing"));
    CHECK(big->crc64_ecma == expected_entry_crc(0u, BIG_BYTES));
    CHECK(tail->crc64_ecma == expected_entry_crc(BIG_BYTES, TAIL_BYTES));

    /* Prove runtime names and offsets do not borrow model/layout metadata. */
    big_offset = big->device_offset;
    tail_offset = tail->device_offset;
    k3_st_model_close(&model);
    memset(layout_entries, 0xa5, sizeof(layout_entries));
    CHECK(strcmp(big->name, "alpha.big") == 0 && big->device_offset == big_offset);
    CHECK(strcmp(tail->name, "zeta.tail") == 0 && tail->device_offset == tail_offset);

    host = (unsigned char *)malloc((size_t)GLM53_STATIC_LOADER_CHUNK_BYTES);
    CHECK(host != NULL);
    runtime_entries[0] = big;
    runtime_entries[1] = tail;
    for (size_t e = 0u; e < 2u; ++e) {
        uint64_t done = 0u;
        while (done < runtime_entries[e]->logical_bytes) {
            size_t amount = (size_t)(runtime_entries[e]->logical_bytes - done >
                GLM53_STATIC_LOADER_CHUNK_BYTES ? GLM53_STATIC_LOADER_CHUNK_BYTES :
                runtime_entries[e]->logical_bytes - done);
            HIP_CHECK(hipMemcpy(host,
                (unsigned char *)glm53_static_store_device_pointer(store,
                    runtime_entries[e]) + done,
                amount, hipMemcpyDeviceToHost));
            for (size_t i = 0u; i < amount; ++i)
                CHECK(host[i] == payload_byte(starts[e] + done + i));
            done += amount;
        }
    }

    free(host); host = NULL;
    glm53_static_store_destroy(store); store = NULL;
    (void)unlink(path); path[0] = '\0';
    printf("glm53 static loader on %s (%s): PASS; requests %llu direct/%llu buffered; physical %llu bytes\n",
           properties.name, properties.gcnArchName,
           (unsigned long long)stats.direct_requests,
           (unsigned long long)stats.buffered_requests,
           (unsigned long long)stats.physical_read_bytes);
    result = 0;
fail:
    free(host);
    glm53_static_store_destroy(store);
    k3_st_model_close(&model);
    if (path[0]) (void)unlink(path);
    return result;
}
