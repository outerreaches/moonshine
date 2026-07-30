#include "k3_io_uring.h"

#include <errno.h>
#include <linux/io_uring.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef struct {
    uint64_t user_data;
    bool busy;
} k3_io_buffer_state;

struct k3_io_uring {
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
    struct iovec *buffers;
    k3_io_buffer_state *buffer_state;
    uint16_t buffer_count;
    uint16_t outstanding;
    bool buffers_registered;
};

static void k3_io_error(char *error, size_t error_size,
                        const char *message) {
    if (error && error_size) {
        snprintf(error, error_size, "%s", message);
    }
}

static void k3_io_errno(char *error, size_t error_size,
                        const char *operation) {
    if (error && error_size) {
        snprintf(error, error_size, "%s: %s", operation, strerror(errno));
    }
}

void k3_io_uring_destroy(k3_io_uring *ring) {
    if (!ring) return;
    if (ring->buffers_registered && ring->fd >= 0) {
        syscall(SYS_io_uring_register, ring->fd,
                IORING_UNREGISTER_BUFFERS, NULL, 0);
    }
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
    free(ring->buffer_state);
    free(ring->buffers);
    free(ring);
}

bool k3_io_uring_create(k3_io_uring **out,
                        const struct iovec *buffers,
                        uint16_t buffer_count,
                        char *error,
                        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (out) *out = NULL;
    if (!out || !buffers || buffer_count == 0) {
        k3_io_error(error, error_size, "invalid io_uring create arguments");
        return false;
    }
    for (uint16_t i = 0; i < buffer_count; i++) {
        if (!buffers[i].iov_base || buffers[i].iov_len == 0) {
            k3_io_error(error, error_size,
                        "registered buffer is empty");
            return false;
        }
    }

    k3_io_uring *ring = (k3_io_uring *)calloc(1, sizeof(*ring));
    if (!ring) {
        k3_io_error(error, error_size, "io_uring allocation failed");
        return false;
    }
    ring->fd = -1;
    ring->buffer_count = buffer_count;
    ring->buffers =
        (struct iovec *)malloc((size_t)buffer_count * sizeof(*ring->buffers));
    ring->buffer_state = (k3_io_buffer_state *)calloc(
        buffer_count, sizeof(*ring->buffer_state));
    if (!ring->buffers || !ring->buffer_state) {
        k3_io_error(error, error_size, "io_uring buffer table allocation failed");
        k3_io_uring_destroy(ring);
        return false;
    }
    memcpy(ring->buffers, buffers,
           (size_t)buffer_count * sizeof(*ring->buffers));

    ring->fd = (int)syscall(SYS_io_uring_setup, buffer_count, &ring->params);
    if (ring->fd < 0) {
        k3_io_errno(error, error_size, "io_uring_setup");
        k3_io_uring_destroy(ring);
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

    ring->sq_ring = mmap(NULL, ring->sq_ring_bytes,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE,
                         ring->fd, IORING_OFF_SQ_RING);
    if (ring->sq_ring == MAP_FAILED) {
        ring->sq_ring = NULL;
        k3_io_errno(error, error_size, "mmap SQ ring");
        k3_io_uring_destroy(ring);
        return false;
    }
    if (ring->params.features & IORING_FEAT_SINGLE_MMAP) {
        ring->cq_ring = ring->sq_ring;
    } else {
        ring->cq_ring = mmap(NULL, ring->cq_ring_bytes,
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_POPULATE,
                             ring->fd, IORING_OFF_CQ_RING);
        if (ring->cq_ring == MAP_FAILED) {
            ring->cq_ring = NULL;
            k3_io_errno(error, error_size, "mmap CQ ring");
            k3_io_uring_destroy(ring);
            return false;
        }
    }
    ring->sqes_bytes =
        ring->params.sq_entries * sizeof(struct io_uring_sqe);
    ring->sqes = (struct io_uring_sqe *)mmap(
        NULL, ring->sqes_bytes, PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_SQES);
    if (ring->sqes == MAP_FAILED) {
        ring->sqes = NULL;
        k3_io_errno(error, error_size, "mmap SQEs");
        k3_io_uring_destroy(ring);
        return false;
    }

    uint8_t *sq = (uint8_t *)ring->sq_ring;
    uint8_t *cq = (uint8_t *)ring->cq_ring;
    ring->sq_head = (unsigned *)(sq + ring->params.sq_off.head);
    ring->sq_tail = (unsigned *)(sq + ring->params.sq_off.tail);
    ring->sq_mask = (unsigned *)(sq + ring->params.sq_off.ring_mask);
    ring->sq_entries = (unsigned *)(sq + ring->params.sq_off.ring_entries);
    ring->sq_array = (unsigned *)(sq + ring->params.sq_off.array);
    ring->cq_head = (unsigned *)(cq + ring->params.cq_off.head);
    ring->cq_tail = (unsigned *)(cq + ring->params.cq_off.tail);
    ring->cq_mask = (unsigned *)(cq + ring->params.cq_off.ring_mask);
    ring->cqes =
        (struct io_uring_cqe *)(cq + ring->params.cq_off.cqes);

    int result = (int)syscall(SYS_io_uring_register, ring->fd,
                              IORING_REGISTER_BUFFERS,
                              ring->buffers, buffer_count);
    if (result < 0) {
        k3_io_errno(error, error_size, "register fixed buffers");
        k3_io_uring_destroy(ring);
        return false;
    }
    ring->buffers_registered = true;
    *out = ring;
    return true;
}

bool k3_io_uring_submit(k3_io_uring *ring,
                        const k3_io_request *requests,
                        uint16_t request_count,
                        char *error,
                        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!ring || !requests || request_count == 0 ||
        request_count > ring->buffer_count - ring->outstanding) {
        k3_io_error(error, error_size, "invalid io_uring submit arguments");
        return false;
    }
    unsigned head = __atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
    unsigned tail = __atomic_load_n(ring->sq_tail, __ATOMIC_RELAXED);
    if (request_count > *ring->sq_entries - (tail - head)) {
        k3_io_error(error, error_size, "io_uring submission queue is full");
        return false;
    }
    for (uint16_t i = 0; i < request_count; i++) {
        const k3_io_request *request = &requests[i];
        if (request->fd < 0 || request->bytes == 0 ||
            request->buffer_index >= ring->buffer_count ||
            request->bytes >
                ring->buffers[request->buffer_index].iov_len ||
            ring->buffer_state[request->buffer_index].busy) {
            k3_io_error(error, error_size,
                        "invalid or busy fixed-buffer request");
            return false;
        }
        for (uint16_t j = 0; j < i; j++) {
            if (requests[j].buffer_index == request->buffer_index) {
                k3_io_error(error, error_size,
                            "duplicate fixed buffer in request batch");
                return false;
            }
        }
    }

    for (uint16_t i = 0; i < request_count; i++) {
        const k3_io_request *request = &requests[i];
        unsigned index = tail & *ring->sq_mask;
        struct io_uring_sqe *sqe = &ring->sqes[index];
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ_FIXED;
        sqe->fd = request->fd;
        sqe->off = request->offset;
        sqe->addr = (uint64_t)(uintptr_t)
            ring->buffers[request->buffer_index].iov_base;
        sqe->len = request->bytes;
        sqe->buf_index = request->buffer_index;
        sqe->user_data = request->buffer_index;
        ring->sq_array[index] = index;
        ring->buffer_state[request->buffer_index].busy = true;
        ring->buffer_state[request->buffer_index].user_data =
            request->user_data;
        tail++;
    }
    __atomic_store_n(ring->sq_tail, tail, __ATOMIC_RELEASE);

    unsigned submitted = 0;
    while (submitted < request_count) {
        int result = (int)syscall(SYS_io_uring_enter, ring->fd,
                                  request_count - submitted,
                                  0, 0, NULL, 0);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) {
            /*
             * The SQEs are already visible to the kernel. Keep the buffers
             * marked busy; the caller must destroy the ring on this failure.
             */
            if (result < 0) {
                k3_io_errno(error, error_size, "io_uring submit");
            } else {
                k3_io_error(error, error_size,
                            "io_uring submit made no progress");
            }
            return false;
        }
        submitted += (unsigned)result;
    }
    ring->outstanding += request_count;
    return true;
}

bool k3_io_uring_wait(k3_io_uring *ring,
                      k3_io_completion *completions,
                      uint16_t capacity,
                      uint16_t *completion_count,
                      char *error,
                      size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (completion_count) *completion_count = 0;
    if (!ring || !completions || capacity == 0 || !completion_count ||
        ring->outstanding == 0) {
        k3_io_error(error, error_size, "invalid io_uring wait arguments");
        return false;
    }
    for (;;) {
        unsigned head = __atomic_load_n(ring->cq_head, __ATOMIC_ACQUIRE);
        unsigned tail = __atomic_load_n(ring->cq_tail, __ATOMIC_ACQUIRE);
        if (head != tail) break;
        int result = (int)syscall(SYS_io_uring_enter, ring->fd, 0, 1,
                                  IORING_ENTER_GETEVENTS, NULL, 0);
        if (result >= 0 || errno == EINTR) continue;
        k3_io_errno(error, error_size, "io_uring wait");
        return false;
    }

    unsigned head = __atomic_load_n(ring->cq_head, __ATOMIC_RELAXED);
    unsigned tail = __atomic_load_n(ring->cq_tail, __ATOMIC_ACQUIRE);
    uint16_t count = 0;
    while (head != tail && count < capacity) {
        struct io_uring_cqe *cqe = &ring->cqes[head & *ring->cq_mask];
        uint16_t buffer_index = (uint16_t)cqe->user_data;
        if (buffer_index >= ring->buffer_count ||
            !ring->buffer_state[buffer_index].busy) {
            k3_io_error(error, error_size,
                        "io_uring completion has invalid buffer");
            return false;
        }
        completions[count].result = cqe->res;
        completions[count].buffer_index = buffer_index;
        completions[count].user_data =
            ring->buffer_state[buffer_index].user_data;
        ring->buffer_state[buffer_index].busy = false;
        ring->outstanding--;
        count++;
        head++;
    }
    __atomic_store_n(ring->cq_head, head, __ATOMIC_RELEASE);
    *completion_count = count;
    return true;
}

uint16_t k3_io_uring_outstanding(const k3_io_uring *ring) {
    return ring ? ring->outstanding : 0;
}
