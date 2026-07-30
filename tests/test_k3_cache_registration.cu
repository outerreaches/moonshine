#include "k3_io_uring.h"
#include "k3_safetensors.h"

#include <hip/hip_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/uio.h>
#include <time.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                        \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    K3_MOE_LAYERS = 92,
    K3_CACHE_PER_LAYER = 32,
    K3_CACHE_SLOTS = K3_MOE_LAYERS * K3_CACHE_PER_LAYER,
    K3_SLOT_BYTES = 17551360,
    K3_STAGING_SLOTS = 16,
    K3_COPY_ITERATIONS = 20,
};

static double elapsed_seconds(struct timespec start,
                              struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const uint8_t *values = (const uint8_t *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        hash ^= values[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 cache registration: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 cache registration on %s (%s)\n",
           properties.name, properties.gcnArchName);

    char error[512];
    k3_st_model model;
    CHECK(k3_st_model_open(&model, root, 96u,
                           error, sizeof(error)),
          error);
    const k3_st_tensor *first = k3_st_find(
        &model,
        "language_model.model.layers.1.block_sparse_moe."
        "experts.0.w1.weight_packed");
    const k3_st_tensor *last = k3_st_find(
        &model,
        "language_model.model.layers.1.block_sparse_moe."
        "experts.0.w3.weight_scale");
    CHECK(first && last && first->shard == last->shard,
          "representative expert lookup");
    const uint64_t logical_start = first->physical_offset;
    const uint64_t logical_end =
        last->physical_offset + last->byte_length;
    const uint64_t aligned_start =
        logical_start & ~UINT64_C(4095);
    const uint64_t aligned_end =
        (logical_end + UINT64_C(4095)) & ~UINT64_C(4095);
    CHECK(logical_end - logical_start == UINT64_C(17547264) &&
          aligned_end - aligned_start <= K3_SLOT_BYTES,
          "representative expert span");
    CHECK(model.shards[first->shard].direct_fd >= 0,
          "representative shard has no O_DIRECT descriptor");

    const uint64_t cache_bytes =
        (uint64_t)K3_CACHE_SLOTS * K3_SLOT_BYTES;
    void *cache_host = NULL;
    void *cache_device = NULL;
    struct timespec allocation_start;
    struct timespec allocation_end;
    clock_gettime(CLOCK_MONOTONIC, &allocation_start);
    HIP_CHECK(hipHostMalloc(
        &cache_host, cache_bytes, hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer(
        &cache_device, cache_host, 0));
    clock_gettime(CLOCK_MONOTONIC, &allocation_end);
    CHECK((uintptr_t)cache_host % 4096u == 0u &&
          cache_device != NULL,
          "mapped cache alignment/device alias");

    struct iovec *iovecs = (struct iovec *)calloc(
        K3_CACHE_SLOTS, sizeof(*iovecs));
    CHECK(iovecs, "cache iovec allocation");
    for (uint32_t slot = 0; slot < K3_CACHE_SLOTS; slot++) {
        iovecs[slot].iov_base =
            (uint8_t *)cache_host + (uint64_t)slot * K3_SLOT_BYTES;
        iovecs[slot].iov_len = K3_SLOT_BYTES;
    }
    k3_io_uring *ring = NULL;
    struct timespec registration_start;
    struct timespec registration_end;
    clock_gettime(CLOCK_MONOTONIC, &registration_start);
    const bool full_registration = k3_io_uring_create(
        &ring, iovecs, K3_CACHE_SLOTS,
        error, sizeof(error));
    clock_gettime(CLOCK_MONOTONIC, &registration_end);
    char full_registration_error[512];
    snprintf(full_registration_error,
             sizeof(full_registration_error), "%s", error);
    if (!full_registration) {
        CHECK(k3_io_uring_create(
                  &ring, iovecs, K3_STAGING_SLOTS,
                  error, sizeof(error)),
              error);
    }

    k3_io_request request = {
        model.shards[first->shard].direct_fd,
        aligned_start,
        (uint32_t)(aligned_end - aligned_start),
        0u,
        UINT64_C(0x4b33434143484530),
    };
    CHECK(k3_io_uring_submit(
              ring, &request, 1u, error, sizeof(error)),
          error);
    k3_io_completion completion;
    uint16_t completion_count = 0;
    CHECK(k3_io_uring_wait(
              ring, &completion, 1u, &completion_count,
              error, sizeof(error)),
          error);
    CHECK(completion_count == 1u &&
          completion.buffer_index == 0u &&
          completion.user_data == request.user_data &&
          completion.result == (int32_t)request.bytes,
          "registered cache-slot read completion");
    const uint8_t *logical_data =
        (const uint8_t *)cache_host +
        (logical_start - aligned_start);
    const uint64_t hash =
        fnv1a64(logical_data, logical_end - logical_start);
    CHECK(hash == UINT64_C(0xeb0eff9b9d8ae0ef),
          "registered cache-slot expert hash");

    uint8_t *mapped_destination =
        (uint8_t *)cache_host +
        (uint64_t)K3_STAGING_SLOTS * K3_SLOT_BYTES;
    struct timespec cpu_copy_start;
    struct timespec cpu_copy_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_copy_start);
    for (uint32_t i = 0; i < K3_COPY_ITERATIONS; i++) {
        memcpy(mapped_destination, cache_host, K3_SLOT_BYTES);
    }
    clock_gettime(CLOCK_MONOTONIC, &cpu_copy_end);
    CHECK(fnv1a64(
              mapped_destination +
                  (logical_start - aligned_start),
              logical_end - logical_start) == hash,
          "mapped admission copy hash");
    const double cpu_copy_ms =
        elapsed_seconds(cpu_copy_start, cpu_copy_end) *
        1000.0 / K3_COPY_ITERATIONS;

    void *device_destination = NULL;
    HIP_CHECK(hipMalloc(&device_destination, K3_SLOT_BYTES));
    hipEvent_t copy_start;
    hipEvent_t copy_stop;
    HIP_CHECK(hipEventCreate(&copy_start));
    HIP_CHECK(hipEventCreate(&copy_stop));
    HIP_CHECK(hipEventRecord(copy_start, NULL));
    for (uint32_t i = 0; i < K3_COPY_ITERATIONS; i++) {
        HIP_CHECK(hipMemcpyAsync(
            device_destination, cache_host, K3_SLOT_BYTES,
            hipMemcpyHostToDevice, NULL));
    }
    HIP_CHECK(hipEventRecord(copy_stop, NULL));
    HIP_CHECK(hipEventSynchronize(copy_stop));
    float host_to_device_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(
        &host_to_device_ms, copy_start, copy_stop));
    host_to_device_ms /= K3_COPY_ITERATIONS;

    HIP_CHECK(hipEventRecord(copy_start, NULL));
    for (uint32_t i = 0; i < K3_COPY_ITERATIONS; i++) {
        HIP_CHECK(hipMemcpyAsync(
            device_destination, cache_device, K3_SLOT_BYTES,
            hipMemcpyDeviceToDevice, NULL));
    }
    HIP_CHECK(hipEventRecord(copy_stop, NULL));
    HIP_CHECK(hipEventSynchronize(copy_stop));
    float mapped_to_device_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(
        &mapped_to_device_ms, copy_start, copy_stop));
    mapped_to_device_ms /= K3_COPY_ITERATIONS;

    struct rlimit memlock;
    CHECK(getrlimit(RLIMIT_MEMLOCK, &memlock) == 0,
          "read RLIMIT_MEMLOCK");
    printf("  mapped cache: %u slots × %.3f MiB = %.3f GiB\n",
           K3_CACHE_SLOTS, K3_SLOT_BYTES / 1048576.0,
           cache_bytes / 1073741824.0);
    printf("  allocation/device alias: %.3f s\n",
           elapsed_seconds(allocation_start, allocation_end));
    if (full_registration) {
        printf("  all-slot io_uring registration: PASS in %.3f s\n",
               elapsed_seconds(registration_start, registration_end));
    } else {
        printf("  all-slot io_uring registration: rejected (%s)\n",
               full_registration_error);
        printf("  fallback: %u registered staging slots; "
               "memlock hard limit %.3f GiB\n",
               K3_STAGING_SLOTS,
               memlock.rlim_max / 1073741824.0);
    }
    printf("  fixed read: %.3f MiB, expert hash 0x%016llx\n",
           request.bytes / 1048576.0,
           (unsigned long long)hash);
    printf("  admission copy: mapped CPU %.3f ms; "
           "mapped-host→device %.3f ms; "
           "mapped-device→device %.3f ms\n",
           cpu_copy_ms, host_to_device_ms,
           mapped_to_device_ms);

    HIP_CHECK(hipEventDestroy(copy_stop));
    HIP_CHECK(hipEventDestroy(copy_start));
    HIP_CHECK(hipFree(device_destination));
    k3_io_uring_destroy(ring);
    free(iovecs);
    HIP_CHECK(hipHostFree(cache_host));
    k3_st_model_close(&model);
    printf("K3 cache registration: PASS\n");
    return 0;
}
