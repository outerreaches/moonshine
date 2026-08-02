#include "k3_prefix_reuse.h"

#include <stdio.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", (message)); \
            return 1; \
        } \
    } while (0)

/*
 * Session prefix reuse skips re-evaluating already-committed positions. A
 * false admission does not fail loudly: the engine continues from causal state
 * that does not match the supplied history and returns a plausible completion
 * for the wrong prefix. Every case below that must be REJECTED is therefore
 * more important than the accepted ones.
 */

static int test_exact_append(void) {
    static const uint32_t retained[] = { 10u, 11u, 12u, 13u };
    static const uint32_t candidate[] = {
        10u, 11u, 12u, 13u, 20u, 21u, 22u };
    CHECK(k3_prefix_reuse_admits(retained, 4u, candidate, 7u),
          "exact append with a three-token suffix must be admitted");
    CHECK(k3_prefix_reuse_common_tokens(
              retained, 4u, candidate, 7u) == 4u,
          "exact append diagnostic must report the retained span");
    return 0;
}

static int test_minimum_suffix_boundary(void) {
    static const uint32_t retained[] = { 10u, 11u, 12u, 13u };
    static const uint32_t two_new[] = {
        10u, 11u, 12u, 13u, 20u, 21u };
    static const uint32_t one_new[] = { 10u, 11u, 12u, 13u, 20u };

    CHECK(k3_prefix_reuse_admits(retained, 4u, two_new, 6u),
          "a two-token suffix is the documented minimum and must admit");
    CHECK(!k3_prefix_reuse_admits(retained, 4u, one_new, 5u),
          "a one-token suffix is below the executor minimum");
    CHECK(!k3_prefix_reuse_admits(retained, 4u, retained, 4u),
          "an identical history leaves no suffix to evaluate");
    return 0;
}

/*
 * The case this test exists for. An edited history of identical token count
 * passes any length check and any count-based heuristic. A byte-exact
 * comparison over the whole retained span rejects it without relying on a
 * collision-bearing summary.
 */
static int test_edited_same_length_history(void) {
    static const uint32_t retained[] = {
        10u, 11u, 12u, 13u, 14u, 15u };
    static const uint32_t edited_first[] = {
        99u, 11u, 12u, 13u, 14u, 15u, 20u, 21u };
    static const uint32_t edited_middle[] = {
        10u, 11u, 99u, 13u, 14u, 15u, 20u, 21u };
    static const uint32_t edited_last[] = {
        10u, 11u, 12u, 13u, 14u, 99u, 20u, 21u };

    CHECK(!k3_prefix_reuse_admits(retained, 6u, edited_first, 8u),
          "an edit at the first retained position must be rejected");
    CHECK(!k3_prefix_reuse_admits(retained, 6u, edited_middle, 8u),
          "an edit inside the retained span must be rejected");
    CHECK(k3_prefix_reuse_common_tokens(
              retained, 6u, edited_middle, 8u) == 2u,
          "diagnostic overlap must stop at the edited token");
    CHECK(!k3_prefix_reuse_admits(retained, 6u, edited_last, 8u),
          "an edit at the final retained position must be rejected");
    return 0;
}

/*
 * A long prefix differing only at its final token catches a comparison that
 * checks a truncated span, a sampled span, or a digest of the leading bytes.
 */
static int test_deep_single_token_divergence(void) {
    enum { SPAN = 4096u };
    static uint32_t retained[SPAN];
    static uint32_t candidate[SPAN + 2u];
    for (size_t i = 0u; i < SPAN; i++) {
        retained[i] = (uint32_t)(i * 7u + 1u);
        candidate[i] = retained[i];
    }
    candidate[SPAN] = 5u;
    candidate[SPAN + 1u] = 6u;

    CHECK(k3_prefix_reuse_admits(
              retained, SPAN, candidate, SPAN + 2u),
          "an untouched long prefix must be admitted");

    candidate[SPAN - 1u] ^= 1u;
    CHECK(!k3_prefix_reuse_admits(
              retained, SPAN, candidate, SPAN + 2u),
          "a single differing token at the end of a long retained "
          "span must be rejected");
    return 0;
}

static int test_shorter_and_forked_histories(void) {
    static const uint32_t retained[] = {
        10u, 11u, 12u, 13u, 14u, 15u };
    static const uint32_t shorter[] = { 10u, 11u, 12u };
    static const uint32_t forked[] = { 10u, 11u, 99u, 20u, 21u };

    CHECK(!k3_prefix_reuse_admits(retained, 6u, shorter, 3u),
          "a shorter history must be rejected");
    CHECK(!k3_prefix_reuse_admits(retained, 6u, forked, 5u),
          "a fork below the retained length must be rejected");
    return 0;
}

static int test_degenerate_inputs(void) {
    static const uint32_t tokens[] = { 10u, 11u, 12u, 13u };

    CHECK(!k3_prefix_reuse_admits(tokens, 0u, tokens, 4u),
          "an empty retained history must never admit; a zero-length "
          "comparison would otherwise be vacuously equal");
    CHECK(!k3_prefix_reuse_admits(NULL, 4u, tokens, 8u),
          "a null retained buffer must be rejected");
    CHECK(!k3_prefix_reuse_admits(tokens, 4u, NULL, 8u),
          "a null candidate buffer must be rejected");
    CHECK(k3_prefix_reuse_common_tokens(
              NULL, 4u, tokens, 4u) == 0u &&
          k3_prefix_reuse_common_tokens(
              tokens, 4u, NULL, 4u) == 0u,
          "null diagnostic buffers must report no overlap");
    return 0;
}

static int test_count_arithmetic_boundaries(void) {
    static const uint32_t tokens[] = { 10u, 11u, 12u, 13u };
    const size_t oversized_span =
        SIZE_MAX / sizeof(tokens[0]) + 1u;

    CHECK(!k3_prefix_reuse_admits(
              tokens, SIZE_MAX, tokens, SIZE_MAX),
          "retained-count suffix arithmetic must not wrap");
    CHECK(!k3_prefix_reuse_admits(
              tokens, oversized_span,
              tokens, oversized_span + 2u),
          "retained byte-span arithmetic must not wrap");
    return 0;
}

int main(void) {
    if (test_exact_append() != 0 ||
        test_minimum_suffix_boundary() != 0 ||
        test_edited_same_length_history() != 0 ||
        test_deep_single_token_divergence() != 0 ||
        test_shorter_and_forked_histories() != 0 ||
        test_degenerate_inputs() != 0 ||
        test_count_arithmetic_boundaries() != 0) {
        return 1;
    }
    printf("K3 session prefix reuse admission: PASS\n");
    return 0;
}
