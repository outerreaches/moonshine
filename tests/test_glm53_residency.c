#include "glm53_residency.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)
#define OK(x) CHECK((x) == GLM53_RESIDENCY_OK)

static bool commit_routes(glm53_residency *r, uint16_t layer,
                          const uint16_t *ids, uint16_t count,
                          glm53_residency_access *a) {
    OK(glm53_residency_plan(r, layer, ids, count, a));
    OK(glm53_residency_commit(r, layer));
    return true;
}

static bool snapshot_is(glm53_residency *r, uint16_t layer,
                        const uint16_t *want, uint16_t want_count) {
    uint16_t got[GLM53_RESIDENCY_EXPERTS_PER_LAYER];
    uint16_t got_count = 0;
    OK(glm53_residency_snapshot(r, layer, got,
                                GLM53_RESIDENCY_EXPERTS_PER_LAYER,
                                &got_count));
    CHECK(got_count == want_count);
    CHECK(memcmp(got, want, (size_t)want_count * sizeof(want[0])) == 0);
    return true;
}

static bool test_geometry_capacity_and_invalid(void) {
    glm53_residency r;
    uint64_t bytes = 99;
    uint16_t id = 0;
    glm53_residency_access a;
    OK(glm53_residency_init(&r, 0));
    CHECK(r.slots_per_layer == 48);
    CHECK(glm53_residency_slot_count(&r) == 2016);
    OK(glm53_residency_logical_capacity(0, 100, &bytes));
    CHECK(bytes == UINT64_C(201600));
    OK(glm53_residency_logical_capacity(
        48, UINT64_C(25165824), &bytes));
    CHECK(bytes == UINT64_C(50734301184));
    CHECK(glm53_residency_logical_capacity(
              48, UINT64_MAX / 2016 + 1, &bytes) ==
          GLM53_RESIDENCY_OVERFLOW);
    CHECK(bytes == 0);
    CHECK(glm53_residency_init(NULL, 48) ==
          GLM53_RESIDENCY_INVALID_ARGUMENT);
    CHECK(glm53_residency_init(&r, 289) ==
          GLM53_RESIDENCY_INVALID_ARGUMENT);
    OK(glm53_residency_init(&r, 48));
    CHECK(glm53_residency_plan(&r, 2, &id, 1, &a) ==
          GLM53_RESIDENCY_INVALID_LAYER);
    CHECK(glm53_residency_plan(&r, 45, &id, 1, &a) ==
          GLM53_RESIDENCY_INVALID_LAYER);
    id = 288;
    CHECK(glm53_residency_plan(&r, 3, &id, 1, &a) ==
          GLM53_RESIDENCY_INVALID_EXPERT);
    CHECK(glm53_residency_plan(&r, 3, &id, 0, &a) ==
          GLM53_RESIDENCY_INVALID_ARGUMENT);
    CHECK(glm53_residency_commit(&r, 3) ==
          GLM53_RESIDENCY_NO_PENDING_PLAN);
    return true;
}

static bool test_fill_repeat_and_49th(void) {
    glm53_residency r;
    uint16_t fill[48];
    uint16_t expected[48];
    glm53_residency_access a[48];
    glm53_residency_access one;
    uint16_t i;
    OK(glm53_residency_init(&r, 0));
    for (i = 0; i < 48; ++i) fill[i] = i;
    CHECK(commit_routes(&r, 3, fill, 48, a));
    for (i = 0; i < 48; ++i) {
        CHECK(!a[i].hit);
        CHECK(a[i].source_slot == GLM53_RESIDENCY_NO_SLOT);
        CHECK(a[i].destination_slot == i);
    }
    CHECK(snapshot_is(&r, 3, fill, 48));
    CHECK(commit_routes(&r, 3, fill, 48, a));
    for (i = 0; i < 48; ++i) {
        CHECK(a[i].hit && a[i].source_slot == i);
        CHECK(a[i].destination_slot == GLM53_RESIDENCY_NO_SLOT);
    }
    i = 48;
    OK(glm53_residency_plan(&r, 3, &i, 1, &one));
    CHECK(!one.hit && one.destination_slot == 0);
    /* Planning, including a planned overwrite, leaves committed LRU intact. */
    CHECK(snapshot_is(&r, 3, fill, 48));
    OK(glm53_residency_commit(&r, 3));
    for (i = 0; i < 47; ++i) expected[i] = (uint16_t)(i + 1);
    expected[47] = 48;
    CHECK(snapshot_is(&r, 3, expected, 48));
    return true;
}

static bool test_hit_refresh_and_abort(void) {
    glm53_residency r;
    uint16_t fill[48];
    uint16_t expected[48];
    glm53_residency_access a[48];
    glm53_residency_access one;
    uint16_t i;
    OK(glm53_residency_init(&r, 48));
    for (i = 0; i < 48; ++i) fill[i] = i;
    CHECK(commit_routes(&r, 10, fill, 48, a));
    i = 0;
    CHECK(commit_routes(&r, 10, &i, 1, &one));
    CHECK(one.hit && one.source_slot == 0);
    for (i = 0; i < 47; ++i) expected[i] = (uint16_t)(i + 1);
    expected[47] = 0;
    CHECK(snapshot_is(&r, 10, expected, 48));
    i = 48;
    OK(glm53_residency_plan(&r, 10, &i, 1, &one));
    CHECK(one.destination_slot == 1); /* expert 1 is oldest untouched */
    OK(glm53_residency_abort(&r, 10));
    CHECK(snapshot_is(&r, 10, expected, 48));
    CHECK(glm53_residency_abort(&r, 10) ==
          GLM53_RESIDENCY_NO_PENDING_PLAN);
    return true;
}

static bool test_mixed_hit_miss(void) {
    glm53_residency r;
    uint16_t fill[] = { 10, 11, 12, 13 };
    uint16_t mixed[] = { 11, 20, 13 };
    uint16_t expected[] = { 12, 11, 20, 13 };
    glm53_residency_access a[4];
    OK(glm53_residency_init(&r, 4));
    CHECK(commit_routes(&r, 44, fill, 4, a));
    OK(glm53_residency_plan(&r, 44, mixed, 3, a));
    CHECK(a[0].hit && a[0].source_slot == 1);
    CHECK(!a[1].hit && a[1].destination_slot == 0);
    CHECK(a[2].hit && a[2].source_slot == 3);
    CHECK(a[1].destination_slot != a[0].source_slot);
    CHECK(a[1].destination_slot != a[2].source_slot);
    CHECK(snapshot_is(&r, 44, fill, 4));
    OK(glm53_residency_commit(&r, 44));
    CHECK(snapshot_is(&r, 44, expected, 4));
    return true;
}

static bool test_duplicates_pending_and_ranges(void) {
    glm53_residency r;
    uint16_t duplicate[] = { 7, 7 };
    uint16_t ids[49];
    glm53_residency_access a[49];
    uint16_t i;
    OK(glm53_residency_init(&r, 48));
    CHECK(glm53_residency_plan(&r, 3, duplicate, 2, a) ==
          GLM53_RESIDENCY_DUPLICATE_EXPERT);
    for (i = 0; i < 49; ++i) ids[i] = i;
    CHECK(glm53_residency_plan(&r, 3, ids, 49, a) ==
          GLM53_RESIDENCY_TOO_MANY_ROUTES);
    OK(glm53_residency_plan(&r, 3, ids, 1, a));
    CHECK(glm53_residency_plan(&r, 3, ids + 1, 1, a) ==
          GLM53_RESIDENCY_PLAN_PENDING);
    /* Pending plans are per-layer, not global. */
    OK(glm53_residency_plan(&r, 4, ids + 1, 1, a + 1));
    OK(glm53_residency_abort(&r, 3));
    OK(glm53_residency_abort(&r, 4));
    CHECK(snapshot_is(&r, 3, NULL, 0));
    CHECK(snapshot_is(&r, 4, NULL, 0));

    /* Caller-owned metadata still fails closed if corrupted before commit. */
    OK(glm53_residency_plan(&r, 3, ids, 1, a));
    r.pending_entries[0][0].slot = r.slots_per_layer;
    CHECK(glm53_residency_commit(&r, 3) ==
          GLM53_RESIDENCY_INVALID_ARGUMENT);
    OK(glm53_residency_abort(&r, 3));
    return true;
}

int main(void) {
    CHECK(test_geometry_capacity_and_invalid());
    CHECK(test_fill_repeat_and_49th());
    CHECK(test_hit_refresh_and_abort());
    CHECK(test_mixed_hit_miss());
    CHECK(test_duplicates_pending_and_ranges());
    puts("glm53_residency: all tests passed");
    return 0;
}
