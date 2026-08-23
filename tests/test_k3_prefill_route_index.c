#include "k3_prefill_route_index.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                     \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static bool slice_equal(
        const k3_prefill_route_index *index,
        uint32_t expert,
        const uint32_t *tokens,
        const uint32_t *outputs,
        uint32_t count) {
    k3_prefill_route_slice slice;
    if (!k3_prefill_route_index_slice(index, expert, &slice) ||
        slice.count != count) {
        return false;
    }
    for (uint32_t item = 0u; item < count; item++) {
        if (slice.tokens[item] != tokens[item] ||
            slice.outputs[item] != outputs[item]) {
            return false;
        }
    }
    return true;
}

static int test_stable_grouping(void) {
    static const uint32_t routes[] = {
        2u, 0u, 4u,
        1u, 2u, 3u,
        2u, 4u, 1u,
    };
    k3_prefill_route_index *index = NULL;
    CHECK(k3_prefill_route_index_create(&index, 6u), "index creation");
    CHECK(k3_prefill_route_index_build(index, routes, 3u, 3u),
          "route index build");
    CHECK(k3_prefill_route_index_expert_count(index) == 6u &&
              k3_prefill_route_index_token_count(index) == 3u &&
              k3_prefill_route_index_top_k(index) == 3u &&
              k3_prefill_route_index_route_count(index) == 9u &&
              k3_prefill_route_index_selected_count(index) == 5u,
          "route index ledger");

    static const uint32_t token0[] = {0u};
    static const uint32_t output0[] = {1u};
    static const uint32_t token1[] = {1u, 2u};
    static const uint32_t output1[] = {3u, 8u};
    static const uint32_t token2[] = {0u, 1u, 2u};
    static const uint32_t output2[] = {0u, 4u, 6u};
    static const uint32_t token3[] = {1u};
    static const uint32_t output3[] = {5u};
    static const uint32_t token4[] = {0u, 2u};
    static const uint32_t output4[] = {2u, 7u};
    CHECK(slice_equal(index, 0u, token0, output0, 1u) &&
              slice_equal(index, 1u, token1, output1, 2u) &&
              slice_equal(index, 2u, token2, output2, 3u) &&
              slice_equal(index, 3u, token3, output3, 1u) &&
              slice_equal(index, 4u, token4, output4, 2u),
          "stable expert slices");
    k3_prefill_route_slice empty;
    CHECK(k3_prefill_route_index_slice(index, 5u, &empty) &&
              empty.count == 0u,
          "unselected expert slice");
    k3_prefill_route_index_destroy(index);
    return 0;
}

static int test_failed_build_preserves_previous_index(void) {
    static const uint32_t valid[] = {0u, 1u, 2u, 3u};
    static const uint32_t duplicate[] = {0u, 0u};
    static const uint32_t out_of_range[] = {0u, 4u};
    k3_prefill_route_index *index = NULL;
    CHECK(k3_prefill_route_index_create(&index, 4u) &&
              k3_prefill_route_index_build(index, valid, 2u, 2u),
          "initial route index");
    CHECK(!k3_prefill_route_index_build(index, duplicate, 1u, 2u),
          "duplicate route was accepted");
    CHECK(!k3_prefill_route_index_build(index, out_of_range, 1u, 2u),
          "out-of-range route was accepted");
    CHECK(!k3_prefill_route_index_build(
              index, valid, UINT32_MAX, 2u),
          "overflowing route count was accepted");
    static const uint32_t token2[] = {1u};
    static const uint32_t output2[] = {2u};
    CHECK(k3_prefill_route_index_token_count(index) == 2u &&
              k3_prefill_route_index_route_count(index) == 4u &&
              slice_equal(index, 2u, token2, output2, 1u),
          "failed build changed the prior index");
    k3_prefill_route_index_destroy(index);
    return 0;
}

static int test_rebuild_and_large_stable_order(void) {
    enum { TOKENS = 100, TOP_K = 16, EXPERTS = 896 };
    uint32_t *routes = (uint32_t *)malloc(
        TOKENS * TOP_K * sizeof(*routes));
    CHECK(routes != NULL, "large route allocation");
    for (uint32_t token = 0u; token < TOKENS; token++) {
        for (uint32_t rank = 0u; rank < TOP_K; rank++) {
            routes[token * TOP_K + rank] =
                (token + rank * 7u) % EXPERTS;
        }
    }
    k3_prefill_route_index *index = NULL;
    CHECK(k3_prefill_route_index_create(&index, EXPERTS) &&
              k3_prefill_route_index_build(
                  index, routes, TOKENS, TOP_K),
          "large route index build");
    CHECK(k3_prefill_route_index_route_count(index) ==
              TOKENS * TOP_K,
          "large route ledger");
    for (uint32_t expert = 0u; expert < EXPERTS; expert++) {
        k3_prefill_route_slice slice;
        CHECK(k3_prefill_route_index_slice(index, expert, &slice),
              "large route slice");
        for (uint32_t item = 1u; item < slice.count; item++) {
            CHECK(slice.outputs[item - 1u] < slice.outputs[item],
                  "expert route order is not stable");
        }
    }

    static const uint32_t replacement[] = {7u, 3u, 5u};
    CHECK(k3_prefill_route_index_build(index, replacement, 1u, 3u) &&
              k3_prefill_route_index_token_count(index) == 1u &&
              k3_prefill_route_index_route_count(index) == 3u &&
              k3_prefill_route_index_selected_count(index) == 3u,
          "route index rebuild");
    static const uint32_t token[] = {0u};
    static const uint32_t output[] = {1u};
    CHECK(slice_equal(index, 3u, token, output, 1u),
          "rebuilt route slice");
    k3_prefill_route_index_destroy(index);
    free(routes);
    return 0;
}

static int test_invalid_api_inputs(void) {
    k3_prefill_route_index *index = (k3_prefill_route_index *)(uintptr_t)1u;
    CHECK(!k3_prefill_route_index_create(NULL, 4u),
          "null output was accepted");
    CHECK(!k3_prefill_route_index_create(&index, 0u) && index == NULL,
          "zero experts were accepted");
    CHECK(k3_prefill_route_index_expert_count(NULL) == 0u &&
              k3_prefill_route_index_token_count(NULL) == 0u &&
              k3_prefill_route_index_route_count(NULL) == 0u,
          "null getters are not empty");
    return 0;
}

int main(void) {
    CHECK(test_stable_grouping() == 0, "stable grouping");
    CHECK(test_failed_build_preserves_previous_index() == 0,
          "failed-build atomicity");
    CHECK(test_rebuild_and_large_stable_order() == 0,
          "large stable route order");
    CHECK(test_invalid_api_inputs() == 0, "invalid API inputs");
    printf("K3 model-free prefill route index: PASS\n");
    return 0;
}
