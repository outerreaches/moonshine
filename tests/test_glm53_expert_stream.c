#include "../glm53_expert_stream.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)
#define OK(c) CHECK((c) == GLM53_EXPERT_STREAM_OK)

typedef struct {
    k3_st_shard shard;
    k3_st_model model;
    glm53_expert_plan expert;
} fixture;

static void init_fixture(fixture *f, uint64_t weight_at, uint64_t scale_at,
                         uint64_t file_bytes) {
    memset(f, 0, sizeof(*f));
    f->shard.data_offset = 4096u;
    f->shard.file_bytes = file_bytes;
    f->model.shards = &f->shard;
    f->model.shard_count = 1u;
    f->expert.shard = 0u;
    f->expert.logical[0].shard = 0u;
    f->expert.logical[0].offset = weight_at;
    f->expert.logical[0].length = GLM53_EXPERT_WEIGHT_BYTES;
    f->expert.logical[1].shard = 0u;
    f->expert.logical[1].offset = scale_at;
    f->expert.logical[1].length = GLM53_EXPERT_SCALE_BYTES;
}

static bool all_zero(const void *value, size_t size) {
    const unsigned char *p = (const unsigned char *)value;
    size_t i;
    for (i = 0u; i < size; ++i) if (p[i] != 0u) return false;
    return true;
}

static bool test_aligned_24mib_cover(void) {
    fixture f;
    glm53_expert_stream_stage s;
    glm53_expert_stream_ledger ledger;
    char error[128];
    init_fixture(&f, 8192u, UINT64_C(50000000), UINT64_C(100000000));
    OK(glm53_expert_stream_stage_build(&s, &f.expert, &f.model,
                                       error, sizeof(error)));
    CHECK(error[0] == '\0');
    CHECK(s.request_count == 1u && s.copy_count == 1u);
    CHECK(s.request.shard == 0u && s.request.file_offset == 8192u);
    CHECK(s.request.byte_count == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(s.request.direct_io);
    CHECK(s.copy.request_offset == 0u && s.copy.slot_offset == 0u);
    CHECK(s.copy.byte_count == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(s.logical_bytes == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(s.requested_bytes == GLM53_EXPERT_WEIGHT_BYTES);
    OK(glm53_expert_stream_ledger_from_stage(&ledger, &s));
    CHECK(ledger.request_count == 1u && ledger.copy_count == 1u);
    CHECK(ledger.requested_bytes == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(ledger.copied_bytes == GLM53_EXPERT_WEIGHT_BYTES);
    return true;
}

static bool test_alignment_edges(void) {
    fixture f;
    glm53_expert_stream_stage s;
    char error[64];
    init_fixture(&f, 8193u, UINT64_C(60000000), UINT64_C(100000000));
    OK(glm53_expert_stream_stage_build(&s, &f.expert, &f.model,
                                       error, sizeof(error)));
    CHECK(s.request.file_offset == 8192u);
    CHECK(s.request.byte_count == GLM53_EXPERT_STREAM_MAX_COVER);
    CHECK(s.request.direct_io);
    CHECK(s.copy.request_offset == 1u);
    CHECK(s.copy.request_offset + s.copy.byte_count < s.request.byte_count);

    init_fixture(&f, 12287u, UINT64_C(60000000), UINT64_C(100000000));
    OK(glm53_expert_stream_stage_build(&s, &f.expert, &f.model,
                                       error, sizeof(error)));
    CHECK(s.request.file_offset == 8192u);
    CHECK(s.request.byte_count == GLM53_EXPERT_STREAM_MAX_COVER);
    CHECK(s.copy.request_offset == 4095u);
    CHECK(s.copy.request_offset + s.copy.byte_count ==
          s.request.byte_count - 1u);
    return true;
}

static bool test_resident_scale_separation(void) {
    fixture f;
    glm53_expert_stream_stage s;
    char error[64];
    const uint64_t weight = 8192u;
    const uint64_t weight_end = weight + GLM53_EXPERT_WEIGHT_BYTES;

    /* These scale placements generate 8 KiB and 12 KiB aligned covers in the
     * expert plan.  Neither is part of this weight-only staging request. */
    init_fixture(&f, weight, weight_end + 1u, UINT64_C(100000000));
    OK(glm53_expert_stream_stage_build(&s, &f.expert, &f.model,
                                       error, sizeof(error)));
    CHECK(s.request.file_offset + s.request.byte_count == weight_end);
    CHECK(s.requested_bytes == GLM53_EXPERT_WEIGHT_BYTES);

    init_fixture(&f, weight, weight_end + 3000u, UINT64_C(100000000));
    OK(glm53_expert_stream_stage_build(&s, &f.expert, &f.model,
                                       error, sizeof(error)));
    CHECK(s.request.file_offset + s.request.byte_count == weight_end);
    CHECK(s.requested_bytes == GLM53_EXPERT_WEIGHT_BYTES);
    return true;
}

static bool expect_failure(fixture *f, glm53_expert_stream_status wanted) {
    glm53_expert_stream_stage s;
    char error[96];
    memset(&s, 0xa5, sizeof(s));
    memset(error, 0, sizeof(error));
    CHECK(glm53_expert_stream_stage_build(&s, &f->expert, &f->model,
                                          error, sizeof(error)) == wanted);
    CHECK(all_zero(&s, sizeof(s)));
    CHECK(error[0] != '\0');
    return true;
}

static bool test_failures_and_zeroing(void) {
    fixture f;
    glm53_expert_stream_stage s;
    char error[64];

    init_fixture(&f, 8192u, UINT64_C(60000000), UINT64_C(100000000));
    f.expert.logical[0].length--;
    CHECK(expect_failure(&f, GLM53_EXPERT_STREAM_BAD_PLAN));

    init_fixture(&f, UINT64_MAX - GLM53_EXPERT_WEIGHT_BYTES,
                 8192u, UINT64_MAX);
    CHECK(expect_failure(&f, GLM53_EXPERT_STREAM_OVERFLOW));
    init_fixture(&f, UINT64_MAX - GLM53_EXPERT_WEIGHT_BYTES + 1u,
                 8192u, UINT64_MAX);
    CHECK(expect_failure(&f, GLM53_EXPERT_STREAM_OUT_OF_BOUNDS));

    /* The aligned cover may include header bytes before data_offset. */
    init_fixture(&f, 5001u, UINT64_C(60000000), UINT64_C(100000000));
    f.shard.data_offset = 5000u;
    OK(glm53_expert_stream_stage_build(&s, &f.expert, &f.model,
                                       error, sizeof(error)));
    CHECK(s.request.file_offset == 4096u && s.request.direct_io);

    /* At EOF, match k3_st_read_range's exact buffered fallback. */
    init_fixture(&f, 8193u, UINT64_C(60000000),
                 8193u + GLM53_EXPERT_WEIGHT_BYTES);
    OK(glm53_expert_stream_stage_build(&s, &f.expert, &f.model,
                                       error, sizeof(error)));
    CHECK(s.request.file_offset == 8193u);
    CHECK(s.request.byte_count == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(!s.request.direct_io);
    CHECK(s.copy.request_offset == 0u);

    init_fixture(&f, 8193u, UINT64_C(60000000),
                 8193u + GLM53_EXPERT_WEIGHT_BYTES - 1u);
    CHECK(expect_failure(&f, GLM53_EXPERT_STREAM_OUT_OF_BOUNDS));

    init_fixture(&f, 8192u, UINT64_C(60000000), UINT64_C(100000000));
    f.expert.logical[0].shard = 1u;
    CHECK(expect_failure(&f, GLM53_EXPERT_STREAM_BAD_PLAN));

    init_fixture(&f, 8192u, UINT64_C(60000000), UINT64_C(100000000));
    f.expert.logical[0].shard = 1u;
    f.expert.shard = 1u;
    CHECK(expect_failure(&f, GLM53_EXPERT_STREAM_OUT_OF_BOUNDS));

    memset(&s, 0xa5, sizeof(s));
    memset(error, 0, sizeof(error));
    CHECK(glm53_expert_stream_stage_build(&s, NULL, &f.model,
          error, sizeof(error)) == GLM53_EXPERT_STREAM_INVALID_ARGUMENT);
    CHECK(all_zero(&s, sizeof(s)) && error[0] != '\0');
    return true;
}

static bool test_ledger_checked_arithmetic(void) {
    glm53_expert_stream_ledger ledger;
    glm53_expert_stream_ledger before;
    glm53_expert_stream_request request = { 0u, 4096u, 4096u, true };
    glm53_expert_stream_copy copy = { 0u, 0u, 1024u };
    glm53_expert_stream_stage bad;

    glm53_expert_stream_ledger_reset(&ledger);
    OK(glm53_expert_stream_ledger_record_request(&ledger, &request));
    OK(glm53_expert_stream_ledger_record_copy(&ledger, &copy));
    CHECK(ledger.request_count == 1u && ledger.requested_bytes == 4096u);
    CHECK(ledger.copy_count == 1u && ledger.copied_bytes == 1024u);

    ledger.requested_bytes = UINT64_MAX - 100u;
    before = ledger;
    CHECK(glm53_expert_stream_ledger_record_request(&ledger, &request) ==
          GLM53_EXPERT_STREAM_OVERFLOW);
    CHECK(memcmp(&ledger, &before, sizeof(ledger)) == 0);
    request.file_offset = UINT64_MAX;
    request.byte_count = 1u;
    CHECK(glm53_expert_stream_ledger_record_request(&ledger, &request) ==
          GLM53_EXPERT_STREAM_OVERFLOW);
    CHECK(memcmp(&ledger, &before, sizeof(ledger)) == 0);

    ledger = before;
    ledger.copy_count = UINT64_MAX;
    before = ledger;
    CHECK(glm53_expert_stream_ledger_record_copy(&ledger, &copy) ==
          GLM53_EXPERT_STREAM_OVERFLOW);
    CHECK(memcmp(&ledger, &before, sizeof(ledger)) == 0);

    copy.request_offset = UINT64_MAX;
    copy.byte_count = 1u;
    CHECK(glm53_expert_stream_ledger_record_copy(&ledger, &copy) ==
          GLM53_EXPERT_STREAM_OVERFLOW);
    CHECK(memcmp(&ledger, &before, sizeof(ledger)) == 0);

    memset(&bad, 0, sizeof(bad));
    memset(&ledger, 0xa5, sizeof(ledger));
    CHECK(glm53_expert_stream_ledger_from_stage(&ledger, &bad) ==
          GLM53_EXPERT_STREAM_INVALID_ARGUMENT);
    CHECK(all_zero(&ledger, sizeof(ledger)));
    return true;
}

int main(void) {
    CHECK(test_aligned_24mib_cover());
    CHECK(test_alignment_edges());
    CHECK(test_resident_scale_separation());
    CHECK(test_failures_and_zeroing());
    CHECK(test_ledger_checked_arithmetic());
    puts("glm53 expert stream: all tests passed");
    return 0;
}
