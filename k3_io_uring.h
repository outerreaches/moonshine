#ifndef K3_IO_URING_H
#define K3_IO_URING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/uio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k3_io_uring k3_io_uring;

typedef struct {
    int fd;
    uint64_t offset;
    uint32_t bytes;
    uint16_t buffer_index;
    uint64_t user_data;
} k3_io_request;

typedef struct {
    int32_t result;
    uint16_t buffer_index;
    uint64_t user_data;
} k3_io_completion;

/*
 * Register caller-owned aligned buffers and create a raw io_uring. The caller
 * may supply ordinary aligned memory or GPU-visible HIP host allocations.
 * A registered buffer can have at most one outstanding request.
 */
bool k3_io_uring_create(k3_io_uring **out,
                        const struct iovec *buffers,
                        uint16_t buffer_count,
                        char *error,
                        size_t error_size);

bool k3_io_uring_submit(k3_io_uring *ring,
                        const k3_io_request *requests,
                        uint16_t request_count,
                        char *error,
                        size_t error_size);

/*
 * Wait for at least one completion and drain up to capacity entries.
 * Negative completion results are returned to the caller unchanged.
 */
bool k3_io_uring_wait(k3_io_uring *ring,
                      k3_io_completion *completions,
                      uint16_t capacity,
                      uint16_t *completion_count,
                      char *error,
                      size_t error_size);

uint16_t k3_io_uring_outstanding(const k3_io_uring *ring);
void k3_io_uring_destroy(k3_io_uring *ring);

#ifdef __cplusplus
}
#endif

#endif
