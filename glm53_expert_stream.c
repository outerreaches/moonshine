#include "glm53_expert_stream.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t size, const char *format, ...) {
    va_list ap;
    if (!error || size == 0u) return;
    va_start(ap, format);
    (void)vsnprintf(error, size, format, ap);
    va_end(ap);
}

static glm53_expert_stream_status reject(
        char *error, size_t error_size, glm53_expert_stream_status status,
        const char *message) {
    set_error(error, error_size, "%s", message);
    return status;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

glm53_expert_stream_status glm53_expert_stream_stage_build(
        glm53_expert_stream_stage *stage,
        const glm53_expert_plan *expert,
        const k3_st_model *model,
        char *error,
        size_t error_size) {
    glm53_expert_stream_stage temp;
    const glm53_expert_extent *weight;
    const k3_st_shard *shard;
    const uint64_t alignment = (uint64_t)GLM53_EXPERT_IO_ALIGNMENT;
    const uint64_t mask = alignment - UINT64_C(1);
    uint64_t logical_end;
    uint64_t rounded_end;
    uint64_t cover_end;

    if (error && error_size != 0u) error[0] = '\0';
    if (stage) memset(stage, 0, sizeof(*stage));
    if (!stage || !expert || !model || !model->shards ||
        model->shard_count == 0u)
        return reject(error, error_size, GLM53_EXPERT_STREAM_INVALID_ARGUMENT,
                      "invalid expert-stream arguments");

    weight = &expert->logical[0];
    if (weight->length != GLM53_EXPERT_WEIGHT_BYTES ||
        weight->shard != expert->shard)
        return reject(error, error_size, GLM53_EXPERT_STREAM_BAD_PLAN,
                      "invalid logical weight extent");
    if ((size_t)weight->shard >= model->shard_count)
        return reject(error, error_size, GLM53_EXPERT_STREAM_OUT_OF_BOUNDS,
                      "logical weight shard is out of bounds");
    shard = &model->shards[weight->shard];
    /* Match k3_st_read_range: the logical span must be in the file, but an
     * aligned cover may include safetensors header bytes before data_offset. */
    if (weight->offset > shard->file_bytes ||
        weight->length > shard->file_bytes - weight->offset)
        return reject(error, error_size, GLM53_EXPERT_STREAM_OUT_OF_BOUNDS,
                      "logical weight extent exceeds shard file");
    if (!add_u64(weight->offset, weight->length, &logical_end) ||
        !add_u64(logical_end, mask, &rounded_end))
        return reject(error, error_size, GLM53_EXPERT_STREAM_OVERFLOW,
                      "logical or aligned weight extent overflows");

    memset(&temp, 0, sizeof(temp));
    temp.request.shard = weight->shard;
    temp.request.file_offset = weight->offset & ~mask;
    cover_end = rounded_end & ~mask;
    temp.request.direct_io = true;
    if (cover_end > shard->file_bytes) {
        /* O_DIRECT cannot extend past EOF.  Mirror k3_st_read_range's exact
         * buffered fallback while retaining the same logical copy. */
        temp.request.file_offset = weight->offset;
        cover_end = logical_end;
        temp.request.direct_io = false;
    }
    temp.request.byte_count = cover_end - temp.request.file_offset;
    temp.copy.request_offset = weight->offset - temp.request.file_offset;
    temp.copy.slot_offset = 0u;
    temp.copy.byte_count = weight->length;
    temp.request_count = GLM53_EXPERT_STREAM_REQUEST_COUNT;
    temp.copy_count = GLM53_EXPERT_STREAM_COPY_COUNT;
    temp.logical_bytes = weight->length;
    temp.requested_bytes = temp.request.byte_count;

    if (temp.request.byte_count == 0u ||
        temp.request.byte_count > GLM53_EXPERT_STREAM_MAX_COVER)
        return reject(error, error_size, GLM53_EXPERT_STREAM_COVER_TOO_LARGE,
                      "weight cover exceeds staging maximum");
    if (!add_u64(temp.request.file_offset, temp.request.byte_count,
                 &cover_end))
        return reject(error, error_size, GLM53_EXPERT_STREAM_OVERFLOW,
                      "physical weight cover overflows");
    if (cover_end > shard->file_bytes)
        return reject(error, error_size, GLM53_EXPERT_STREAM_OUT_OF_BOUNDS,
                      "weight cover exceeds shard file");

    *stage = temp;
    return GLM53_EXPERT_STREAM_OK;
}

void glm53_expert_stream_ledger_reset(glm53_expert_stream_ledger *ledger) {
    if (ledger) memset(ledger, 0, sizeof(*ledger));
}

glm53_expert_stream_status glm53_expert_stream_ledger_record_request(
        glm53_expert_stream_ledger *ledger,
        const glm53_expert_stream_request *request) {
    glm53_expert_stream_ledger temp;
    uint64_t ignored;
    if (!ledger || !request || request->byte_count == 0u)
        return GLM53_EXPERT_STREAM_INVALID_ARGUMENT;
    if (!add_u64(request->file_offset, request->byte_count, &ignored))
        return GLM53_EXPERT_STREAM_OVERFLOW;
    temp = *ledger;
    if (!add_u64(temp.request_count, UINT64_C(1), &temp.request_count) ||
        !add_u64(temp.requested_bytes, request->byte_count,
                 &temp.requested_bytes))
        return GLM53_EXPERT_STREAM_OVERFLOW;
    *ledger = temp;
    return GLM53_EXPERT_STREAM_OK;
}

glm53_expert_stream_status glm53_expert_stream_ledger_record_copy(
        glm53_expert_stream_ledger *ledger,
        const glm53_expert_stream_copy *copy) {
    glm53_expert_stream_ledger temp;
    uint64_t ignored;
    if (!ledger || !copy || copy->byte_count == 0u)
        return GLM53_EXPERT_STREAM_INVALID_ARGUMENT;
    /* The ranges must themselves be representable even though this helper
     * does not know their allocation sizes. */
    if (!add_u64(copy->request_offset, copy->byte_count, &ignored) ||
        !add_u64(copy->slot_offset, copy->byte_count, &ignored))
        return GLM53_EXPERT_STREAM_OVERFLOW;
    temp = *ledger;
    if (!add_u64(temp.copy_count, UINT64_C(1), &temp.copy_count) ||
        !add_u64(temp.copied_bytes, copy->byte_count, &temp.copied_bytes))
        return GLM53_EXPERT_STREAM_OVERFLOW;
    *ledger = temp;
    return GLM53_EXPERT_STREAM_OK;
}

glm53_expert_stream_status glm53_expert_stream_ledger_from_stage(
        glm53_expert_stream_ledger *ledger,
        const glm53_expert_stream_stage *stage) {
    glm53_expert_stream_ledger temp;
    glm53_expert_stream_status status;
    if (ledger) memset(ledger, 0, sizeof(*ledger));
    if (!ledger || !stage ||
        stage->request_count != GLM53_EXPERT_STREAM_REQUEST_COUNT ||
        stage->copy_count != GLM53_EXPERT_STREAM_COPY_COUNT ||
        stage->requested_bytes != stage->request.byte_count ||
        stage->logical_bytes != stage->copy.byte_count)
        return GLM53_EXPERT_STREAM_INVALID_ARGUMENT;
    memset(&temp, 0, sizeof(temp));
    status = glm53_expert_stream_ledger_record_request(&temp, &stage->request);
    if (status != GLM53_EXPERT_STREAM_OK) return status;
    status = glm53_expert_stream_ledger_record_copy(&temp, &stage->copy);
    if (status != GLM53_EXPERT_STREAM_OK) return status;
    *ledger = temp;
    return GLM53_EXPERT_STREAM_OK;
}
