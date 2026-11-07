#ifndef GLM53_EXPERT_STREAM_H
#define GLM53_EXPERT_STREAM_H

#include "glm53_expert_plan.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GLM53_EXPERT_STREAM_REQUEST_COUNT = 1,
    GLM53_EXPERT_STREAM_COPY_COUNT = 1
};

/* A 24 MiB weight extent has the same phase at both ends.  It therefore
 * needs either 24 MiB (aligned) or 24 MiB + one page (unaligned). */
#define GLM53_EXPERT_STREAM_MAX_COVER     (GLM53_EXPERT_WEIGHT_BYTES + (uint64_t)GLM53_EXPERT_IO_ALIGNMENT)

typedef enum {
    GLM53_EXPERT_STREAM_OK = 0,
    GLM53_EXPERT_STREAM_INVALID_ARGUMENT,
    GLM53_EXPERT_STREAM_BAD_PLAN,
    GLM53_EXPERT_STREAM_OUT_OF_BOUNDS,
    GLM53_EXPERT_STREAM_OVERFLOW,
    GLM53_EXPERT_STREAM_COVER_TOO_LARGE
} glm53_expert_stream_status;

typedef struct {
    uint16_t shard;
    uint64_t file_offset; /* aligned, or exact when EOF forces fallback */
    uint64_t byte_count;  /* aligned cover, or exact logical byte count */
    bool direct_io;       /* range is eligible for an aligned direct read */
} glm53_expert_stream_request;

typedef struct {
    uint64_t request_offset; /* source offset in the staged request */
    uint64_t slot_offset;    /* destination offset in the serialized slot */
    uint64_t byte_count;
} glm53_expert_stream_copy;

typedef struct {
    glm53_expert_stream_request request;
    glm53_expert_stream_copy copy;
    size_t request_count;
    size_t copy_count;
    uint64_t logical_bytes;
    uint64_t requested_bytes;
} glm53_expert_stream_stage;

/* A small accounting object for the single serialized staging slot.  The
 * record helpers use checked addition and leave the ledger unchanged on
 * failure. */
typedef struct {
    uint64_t request_count;
    uint64_t requested_bytes;
    uint64_t copy_count;
    uint64_t copied_bytes;
} glm53_expert_stream_ledger;

/* Derive a read and copy plan solely from expert->logical[0] (weights).
 * expert->logical[1] (resident scales) is deliberately not covered. */
glm53_expert_stream_status glm53_expert_stream_stage_build(
    glm53_expert_stream_stage *stage,
    const glm53_expert_plan *expert,
    const k3_st_model *model,
    char *error,
    size_t error_size);

void glm53_expert_stream_ledger_reset(glm53_expert_stream_ledger *ledger);

glm53_expert_stream_status glm53_expert_stream_ledger_record_request(
    glm53_expert_stream_ledger *ledger,
    const glm53_expert_stream_request *request);

glm53_expert_stream_status glm53_expert_stream_ledger_record_copy(
    glm53_expert_stream_ledger *ledger,
    const glm53_expert_stream_copy *copy);

/* Convenience helper for a completed serialized stage: one request and one
 * logical copy.  It resets ledger first, including on failure. */
glm53_expert_stream_status glm53_expert_stream_ledger_from_stage(
    glm53_expert_stream_ledger *ledger,
    const glm53_expert_stream_stage *stage);

#ifdef __cplusplus
}
#endif

#endif
