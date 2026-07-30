#include "k3_safetensors.h"

#include <errno.h>
#include <inttypes.h>
#include <linux/io_uring.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define K3_ALIGNMENT UINT64_C(4096)
#define K3_EXPERT_BYTES UINT64_C(17547264)
#define K3_LAYER_COUNT 92u
#define K3_EXPERT_COUNT 896u
#define K3_MAX_QD 32u
#define K3_DEFAULT_OPS 256u

typedef struct {
    int fd;
    uint64_t offset;
    uint32_t bytes;
    uint16_t shard;
    uint16_t layer;
    uint16_t expert;
} read_request;

typedef struct {
    int fd;
    struct io_uring_params params;
    void *sq_ring;
    size_t sq_ring_bytes;
    void *cq_ring;
    size_t cq_ring_bytes;
    struct io_uring_sqe *sqes;
    size_t sqes_bytes;
    unsigned *sq_head;
    unsigned *sq_tail;
    unsigned *sq_mask;
    unsigned *sq_entries;
    unsigned *sq_array;
    unsigned *cq_head;
    unsigned *cq_tail;
    unsigned *cq_mask;
    struct io_uring_cqe *cqes;
} raw_ring;

typedef struct {
    uint32_t request_index;
    struct timespec submitted_at;
} active_slot;

static double elapsed_seconds(struct timespec start, struct timespec end) {
    return (double) (end.tv_sec - start.tv_sec) +
           (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

static void ring_close(raw_ring *ring) {
    if (!ring) return;
    if (ring->sqes && ring->sqes != MAP_FAILED) {
        munmap(ring->sqes, ring->sqes_bytes);
    }
    if (ring->sq_ring && ring->sq_ring != MAP_FAILED) {
        munmap(ring->sq_ring, ring->sq_ring_bytes);
    }
    if (ring->cq_ring && ring->cq_ring != MAP_FAILED &&
        ring->cq_ring != ring->sq_ring) {
        munmap(ring->cq_ring, ring->cq_ring_bytes);
    }
    if (ring->fd >= 0) close(ring->fd);
    memset(ring, 0, sizeof(*ring));
    ring->fd = -1;
}

static bool ring_open(raw_ring *ring, unsigned entries,
                      char *error, size_t error_size) {
    memset(ring, 0, sizeof(*ring));
    ring->fd = -1;
    ring->fd = (int) syscall(SYS_io_uring_setup, entries, &ring->params);
    if (ring->fd < 0) {
        snprintf(error, error_size, "io_uring_setup: %s", strerror(errno));
        return false;
    }

    ring->sq_ring_bytes = ring->params.sq_off.array +
                          ring->params.sq_entries * sizeof(unsigned);
    ring->cq_ring_bytes = ring->params.cq_off.cqes +
                          ring->params.cq_entries * sizeof(struct io_uring_cqe);
    if (ring->params.features & IORING_FEAT_SINGLE_MMAP) {
        if (ring->cq_ring_bytes > ring->sq_ring_bytes) {
            ring->sq_ring_bytes = ring->cq_ring_bytes;
        }
        ring->cq_ring_bytes = ring->sq_ring_bytes;
    }

    ring->sq_ring = mmap(NULL, ring->sq_ring_bytes, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, ring->fd,
                         IORING_OFF_SQ_RING);
    if (ring->sq_ring == MAP_FAILED) {
        ring->sq_ring = NULL;
        snprintf(error, error_size, "mmap SQ ring: %s", strerror(errno));
        ring_close(ring);
        return false;
    }
    if (ring->params.features & IORING_FEAT_SINGLE_MMAP) {
        ring->cq_ring = ring->sq_ring;
    } else {
        ring->cq_ring = mmap(NULL, ring->cq_ring_bytes,
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_POPULATE, ring->fd,
                             IORING_OFF_CQ_RING);
        if (ring->cq_ring == MAP_FAILED) {
            ring->cq_ring = NULL;
            snprintf(error, error_size, "mmap CQ ring: %s", strerror(errno));
            ring_close(ring);
            return false;
        }
    }

    ring->sqes_bytes =
        ring->params.sq_entries * sizeof(struct io_uring_sqe);
    ring->sqes = mmap(NULL, ring->sqes_bytes, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_SQES);
    if (ring->sqes == MAP_FAILED) {
        ring->sqes = NULL;
        snprintf(error, error_size, "mmap SQEs: %s", strerror(errno));
        ring_close(ring);
        return false;
    }

    uint8_t *sq = (uint8_t *) ring->sq_ring;
    uint8_t *cq = (uint8_t *) ring->cq_ring;
    ring->sq_head = (unsigned *) (sq + ring->params.sq_off.head);
    ring->sq_tail = (unsigned *) (sq + ring->params.sq_off.tail);
    ring->sq_mask = (unsigned *) (sq + ring->params.sq_off.ring_mask);
    ring->sq_entries = (unsigned *) (sq + ring->params.sq_off.ring_entries);
    ring->sq_array = (unsigned *) (sq + ring->params.sq_off.array);
    ring->cq_head = (unsigned *) (cq + ring->params.cq_off.head);
    ring->cq_tail = (unsigned *) (cq + ring->params.cq_off.tail);
    ring->cq_mask = (unsigned *) (cq + ring->params.cq_off.ring_mask);
    ring->cqes =
        (struct io_uring_cqe *) (cq + ring->params.cq_off.cqes);
    return true;
}

static bool ring_register_buffers(raw_ring *ring, struct iovec *iovecs,
                                  unsigned count,
                                  char *error, size_t error_size) {
    int result = (int) syscall(SYS_io_uring_register, ring->fd,
                               IORING_REGISTER_BUFFERS, iovecs, count);
    if (result < 0) {
        snprintf(error, error_size, "register fixed buffers: %s",
                 strerror(errno));
        return false;
    }
    return true;
}

static bool ring_submit(raw_ring *ring, unsigned count,
                        char *error, size_t error_size) {
    unsigned submitted = 0;
    while (submitted < count) {
        int result = (int) syscall(SYS_io_uring_enter, ring->fd,
                                   count - submitted, 0, 0, NULL, 0);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) {
            snprintf(error, error_size, "io_uring submit: %s",
                     result < 0 ? strerror(errno) : "no progress");
            return false;
        }
        submitted += (unsigned) result;
    }
    return true;
}

static bool ring_wait(raw_ring *ring, char *error, size_t error_size) {
    for (;;) {
        unsigned head = __atomic_load_n(ring->cq_head, __ATOMIC_ACQUIRE);
        unsigned tail = __atomic_load_n(ring->cq_tail, __ATOMIC_ACQUIRE);
        if (head != tail) return true;
        int result = (int) syscall(SYS_io_uring_enter, ring->fd, 0, 1,
                                   IORING_ENTER_GETEVENTS, NULL, 0);
        if (result >= 0 || errno == EINTR) continue;
        snprintf(error, error_size, "io_uring wait: %s", strerror(errno));
        return false;
    }
}

static bool queue_read(raw_ring *ring, const read_request *request,
                       void *buffer, uint16_t buffer_index, uint64_t user_data,
                       char *error, size_t error_size) {
    unsigned head = __atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
    unsigned tail = __atomic_load_n(ring->sq_tail, __ATOMIC_RELAXED);
    if (tail - head >= *ring->sq_entries) {
        snprintf(error, error_size, "SQ ring is full");
        return false;
    }
    unsigned index = tail & *ring->sq_mask;
    struct io_uring_sqe *sqe = &ring->sqes[index];
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ_FIXED;
    sqe->fd = request->fd;
    sqe->off = request->offset;
    sqe->addr = (uint64_t) (uintptr_t) buffer;
    sqe->len = request->bytes;
    sqe->buf_index = buffer_index;
    sqe->user_data = user_data;
    ring->sq_array[index] = index;
    __atomic_store_n(ring->sq_tail, tail + 1u, __ATOMIC_RELEASE);
    return true;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *) left;
    double b = *(const double *) right;
    return (a > b) - (a < b);
}

static double percentile(const double *sorted, uint32_t count,
                         uint32_t numerator, uint32_t denominator) {
    uint64_t rank =
        ((uint64_t) numerator * count + denominator - 1u) / denominator;
    if (rank == 0) rank = 1;
    if (rank > count) rank = count;
    return sorted[rank - 1u];
}

static bool benchmark_qd(raw_ring *ring,
                         const read_request *requests, uint32_t operation_count,
                         void **buffers, uint32_t qd,
                         double *throughput_gib_s, double *p50_ms,
                         double *p95_ms, double *p99_ms,
                         char *error, size_t error_size) {
    active_slot slots[K3_MAX_QD];
    double *latencies = (double *) calloc(operation_count, sizeof(double));
    if (!latencies) {
        snprintf(error, error_size, "latency allocation failed");
        return false;
    }

    uint32_t issued = 0;
    uint32_t completed = 0;
    unsigned initial = operation_count < qd ? operation_count : qd;
    for (unsigned slot = 0; slot < initial; slot++) {
        slots[slot].request_index = issued;
        clock_gettime(CLOCK_MONOTONIC, &slots[slot].submitted_at);
        if (!queue_read(ring, &requests[issued], buffers[slot],
                        (uint16_t) slot, slot, error, error_size)) {
            free(latencies);
            return false;
        }
        issued++;
    }

    struct timespec run_start, run_end;
    clock_gettime(CLOCK_MONOTONIC, &run_start);
    if (!ring_submit(ring, initial, error, error_size)) {
        free(latencies);
        return false;
    }

    while (completed < operation_count) {
        if (!ring_wait(ring, error, error_size)) {
            free(latencies);
            return false;
        }
        unsigned cq_head = __atomic_load_n(ring->cq_head, __ATOMIC_RELAXED);
        unsigned cq_tail = __atomic_load_n(ring->cq_tail, __ATOMIC_ACQUIRE);
        unsigned queued = 0;
        while (cq_head != cq_tail) {
            struct io_uring_cqe *cqe =
                &ring->cqes[cq_head & *ring->cq_mask];
            unsigned slot = (unsigned) cqe->user_data;
            if (slot >= qd) {
                snprintf(error, error_size,
                         "completion has invalid slot %u", slot);
                free(latencies);
                return false;
            }
            const read_request *done =
                &requests[slots[slot].request_index];
            if (cqe->res < 0 || (uint32_t) cqe->res != done->bytes) {
                snprintf(error, error_size,
                         "read layer=%u expert=%u shard=%u: %s (%d/%u)",
                         done->layer, done->expert, done->shard,
                         cqe->res < 0 ? strerror(-cqe->res) : "short read",
                         cqe->res, done->bytes);
                free(latencies);
                return false;
            }
            struct timespec finished;
            clock_gettime(CLOCK_MONOTONIC, &finished);
            latencies[completed++] =
                elapsed_seconds(slots[slot].submitted_at, finished) * 1000.0;
            cq_head++;

            if (issued < operation_count) {
                slots[slot].request_index = issued;
                clock_gettime(CLOCK_MONOTONIC, &slots[slot].submitted_at);
                if (!queue_read(ring, &requests[issued], buffers[slot],
                                (uint16_t) slot, slot,
                                error, error_size)) {
                    free(latencies);
                    return false;
                }
                issued++;
                queued++;
            }
        }
        __atomic_store_n(ring->cq_head, cq_head, __ATOMIC_RELEASE);
        if (queued && !ring_submit(ring, queued, error, error_size)) {
            free(latencies);
            return false;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &run_end);

    uint64_t total_bytes = 0;
    for (uint32_t i = 0; i < operation_count; i++) {
        total_bytes += requests[i].bytes;
    }
    double seconds = elapsed_seconds(run_start, run_end);
    *throughput_gib_s =
        (double) total_bytes / seconds / (1024.0 * 1024.0 * 1024.0);
    qsort(latencies, operation_count, sizeof(*latencies), compare_double);
    *p50_ms = percentile(latencies, operation_count, 50, 100);
    *p95_ms = percentile(latencies, operation_count, 95, 100);
    *p99_ms = percentile(latencies, operation_count, 99, 100);
    free(latencies);
    return true;
}

static uint32_t next_random(uint32_t *state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static bool build_requests(const k3_st_model *model,
                           read_request *requests, uint32_t count,
                           uint32_t *maximum_bytes,
                           char *error, size_t error_size) {
    uint32_t state = UINT32_C(0x4b334944);
    uint32_t built = 0;
    uint32_t attempts = 0;
    while (built < count && attempts < count * 8u) {
        attempts++;
        unsigned layer = 1u + next_random(&state) % K3_LAYER_COUNT;
        unsigned expert = next_random(&state) % K3_EXPERT_COUNT;
        char first_name[256];
        char last_name[256];
        snprintf(first_name, sizeof(first_name),
                 "language_model.model.layers.%u.block_sparse_moe."
                 "experts.%u.w1.weight_packed", layer, expert);
        snprintf(last_name, sizeof(last_name),
                 "language_model.model.layers.%u.block_sparse_moe."
                 "experts.%u.w3.weight_scale", layer, expert);
        const k3_st_tensor *first = k3_st_find(model, first_name);
        const k3_st_tensor *last = k3_st_find(model, last_name);
        if (!first || !last || first->shard != last->shard) continue;
        uint64_t physical_end = last->physical_offset + last->byte_length;
        if (physical_end < last->physical_offset ||
            physical_end - first->physical_offset != K3_EXPERT_BYTES) {
            continue;
        }
        const k3_st_shard *shard = &model->shards[first->shard];
        if (shard->direct_fd < 0) {
            snprintf(error, error_size,
                     "shard %u has no O_DIRECT descriptor", first->shard);
            return false;
        }
        uint64_t aligned_start =
            first->physical_offset & ~(K3_ALIGNMENT - 1u);
        uint64_t aligned_end =
            (physical_end + K3_ALIGNMENT - 1u) & ~(K3_ALIGNMENT - 1u);
        uint64_t bytes = aligned_end - aligned_start;
        if (aligned_end > shard->file_bytes || bytes > UINT32_MAX) continue;
        requests[built].fd = shard->direct_fd;
        requests[built].offset = aligned_start;
        requests[built].bytes = (uint32_t) bytes;
        requests[built].shard = first->shard;
        requests[built].layer = (uint16_t) layer;
        requests[built].expert = (uint16_t) expert;
        if (requests[built].bytes > *maximum_bytes) {
            *maximum_bytes = requests[built].bytes;
        }
        built++;
    }
    if (built != count) {
        snprintf(error, error_size,
                 "only built %u/%u contiguous expert requests", built, count);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    uint32_t operation_count = K3_DEFAULT_OPS;
    if (argc > 2) {
        char *end = NULL;
        unsigned long parsed = strtoul(argv[2], &end, 10);
        if (!end || *end != '\0' || parsed < K3_MAX_QD || parsed > 4096u) {
            fprintf(stderr, "operations must be in [%u, 4096]\n", K3_MAX_QD);
            return 2;
        }
        operation_count = (uint32_t) parsed;
    }

    char error[512];
    k3_st_model model;
    struct timespec open_start, open_end;
    clock_gettime(CLOCK_MONOTONIC, &open_start);
    if (!k3_st_model_open(&model, root, 96, error, sizeof(error))) {
        fprintf(stderr, "test_k3_io_qd: FAIL: %s\n", error);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &open_end);

    read_request *requests =
        (read_request *) calloc(operation_count, sizeof(*requests));
    uint32_t maximum_bytes = 0;
    if (!requests ||
        !build_requests(&model, requests, operation_count, &maximum_bytes,
                        error, sizeof(error))) {
        fprintf(stderr, "test_k3_io_qd: FAIL: %s\n",
                requests ? error : "request allocation failed");
        free(requests);
        k3_st_model_close(&model);
        return 1;
    }

    void *buffers[K3_MAX_QD] = { 0 };
    struct iovec iovecs[K3_MAX_QD];
    for (unsigned i = 0; i < K3_MAX_QD; i++) {
        int result = posix_memalign(&buffers[i], (size_t) K3_ALIGNMENT,
                                    maximum_bytes);
        if (result != 0) {
            snprintf(error, sizeof(error), "buffer %u allocation: %s",
                     i, strerror(result));
            for (unsigned j = 0; j < i; j++) free(buffers[j]);
            free(requests);
            k3_st_model_close(&model);
            fprintf(stderr, "test_k3_io_qd: FAIL: %s\n", error);
            return 1;
        }
        iovecs[i].iov_base = buffers[i];
        iovecs[i].iov_len = maximum_bytes;
    }

    raw_ring ring;
    if (!ring_open(&ring, K3_MAX_QD, error, sizeof(error)) ||
        !ring_register_buffers(&ring, iovecs, K3_MAX_QD,
                               error, sizeof(error))) {
        fprintf(stderr, "test_k3_io_qd: FAIL: %s\n", error);
        ring_close(&ring);
        for (unsigned i = 0; i < K3_MAX_QD; i++) free(buffers[i]);
        free(requests);
        k3_st_model_close(&model);
        return 1;
    }

    fprintf(stderr,
            "K3 exact-block io_uring sweep: %u operations/QD, "
            "%u-byte maximum request, %.3f s index open\n",
            operation_count, maximum_bytes,
            elapsed_seconds(open_start, open_end));
    printf("qd,operations,request_bytes,throughput_gib_s,p50_ms,p95_ms,p99_ms\n");
    const uint32_t depths[] = { 1u, 2u, 4u, 8u, 16u, 32u };
    bool passed = true;
    for (size_t i = 0; i < sizeof(depths) / sizeof(depths[0]); i++) {
        double gib_s, p50, p95, p99;
        if (!benchmark_qd(&ring, requests, operation_count, buffers, depths[i],
                          &gib_s, &p50, &p95, &p99,
                          error, sizeof(error))) {
            fprintf(stderr, "test_k3_io_qd: FAIL at QD%u: %s\n",
                    depths[i], error);
            passed = false;
            break;
        }
        printf("%u,%u,%u,%.6f,%.6f,%.6f,%.6f\n",
               depths[i], operation_count, maximum_bytes,
               gib_s, p50, p95, p99);
        fflush(stdout);
    }

    syscall(SYS_io_uring_register, ring.fd, IORING_UNREGISTER_BUFFERS,
            NULL, 0);
    ring_close(&ring);
    for (unsigned i = 0; i < K3_MAX_QD; i++) free(buffers[i]);
    free(requests);
    k3_st_model_close(&model);
    if (passed) fprintf(stderr, "K3 exact-block io_uring sweep: PASS\n");
    return passed ? 0 : 1;
}
