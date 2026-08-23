#include "k3_prefix_catalog.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                     \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static k3_engine_state_file_info state_info(
        uint32_t position,
        uint32_t context,
        uint64_t layout,
        bool q8,
        uint64_t payload_crc) {
    k3_engine_state_file_info info;
    memset(&info, 0, sizeof(info));
    info.format_version = 1u;
    info.context = context;
    info.token_position = position;
    info.model_layout_crc64 = layout;
    info.payload_bytes = UINT64_C(454600000) + position * UINT64_C(27648);
    info.file_bytes = info.payload_bytes + 256u;
    info.payload_crc64 = payload_crc;
    info.q8_projections = q8;
    return info;
}

static int test_longest_exact_eligible_prefix(void) {
    k3_prefix_catalog *catalog = NULL;
    CHECK(k3_prefix_catalog_create(&catalog), "catalog creation");
    const uint32_t short_prefix[] = {10u, 11u};
    const uint32_t long_prefix[] = {10u, 11u, 12u, 13u};
    const uint32_t candidate[] = {10u, 11u, 12u, 13u, 20u, 21u};
    const k3_engine_state_file_info short_info =
        state_info(2u, 8192u, UINT64_C(0x1234), true, UINT64_C(0xa1));
    const k3_engine_state_file_info long_info =
        state_info(4u, 8192u, UINT64_C(0x1234), true, UINT64_C(0xa2));
    CHECK(k3_prefix_catalog_add(
              catalog, short_prefix, 2u, "/states/short.k3", &short_info) &&
          k3_prefix_catalog_add(
              catalog, long_prefix, 4u, "/states/long.k3", &long_info),
          "catalog insertion");
    const k3_prefix_catalog_identity identity = {
        .format_version = 1u,
        .context = 8192u,
        .model_layout_crc64 = UINT64_C(0x1234),
        .q8_projections = true,
    };
    k3_prefix_catalog_match match;
    CHECK(k3_prefix_catalog_find(
              catalog, candidate,
              sizeof(candidate) / sizeof(candidate[0]),
              &identity, &match),
          "longest prefix lookup");
    CHECK(match.token_count == 4u &&
              strcmp(match.state_path, "/states/long.k3") == 0 &&
              match.state_info->payload_crc64 == UINT64_C(0xa2),
          "longest eligible prefix was not selected");

    const uint32_t one_after_long[] = {10u, 11u, 12u, 13u, 20u};
    CHECK(k3_prefix_catalog_find(
              catalog, one_after_long,
              sizeof(one_after_long) / sizeof(one_after_long[0]),
              &identity, &match) &&
              match.token_count == 2u,
          "lookup did not fall back to an eligible shorter prefix");
    k3_prefix_catalog_destroy(catalog);
    return 0;
}

static int test_identity_and_exactness_rejection(void) {
    k3_prefix_catalog *catalog = NULL;
    CHECK(k3_prefix_catalog_create(&catalog), "catalog creation");
    const uint32_t prefix[] = {1u, 2u, 3u, 4u};
    const uint32_t exact[] = {1u, 2u, 3u, 4u, 5u, 6u};
    const uint32_t forked[] = {1u, 2u, 9u, 4u, 5u, 6u};
    const k3_engine_state_file_info info =
        state_info(4u, 131072u, UINT64_C(0xfeed), true, UINT64_C(0xbeef));
    CHECK(k3_prefix_catalog_add(
              catalog, prefix, 4u, "/states/prefix.k3", &info),
          "catalog insertion");
    k3_prefix_catalog_match match;
    k3_prefix_catalog_identity identity = {
        .format_version = 1u,
        .context = 131072u,
        .model_layout_crc64 = UINT64_C(0xfeed),
        .q8_projections = true,
    };
    CHECK(k3_prefix_catalog_find(
              catalog, exact, 6u, &identity, &match),
          "exact identity lookup");
    CHECK(!k3_prefix_catalog_find(
              catalog, forked, 6u, &identity, &match),
          "forked prefix was admitted");
    CHECK(!k3_prefix_catalog_find(
              catalog, prefix, 4u, &identity, &match),
          "identical history without suffix was admitted");
    identity.context = 8192u;
    CHECK(!k3_prefix_catalog_find(
              catalog, exact, 6u, &identity, &match),
          "wrong context identity was admitted");
    identity.context = 131072u;
    identity.model_layout_crc64 ^= 1u;
    CHECK(!k3_prefix_catalog_find(
              catalog, exact, 6u, &identity, &match),
          "wrong model identity was admitted");
    identity.model_layout_crc64 ^= 1u;
    identity.q8_projections = false;
    CHECK(!k3_prefix_catalog_find(
              catalog, exact, 6u, &identity, &match),
          "wrong static tier was admitted");
    k3_prefix_catalog_destroy(catalog);
    return 0;
}

static int test_duplicate_remove_and_metadata_validation(void) {
    k3_prefix_catalog *catalog = NULL;
    CHECK(k3_prefix_catalog_create(&catalog), "catalog creation");
    const uint32_t prefix[] = {7u, 8u};
    k3_engine_state_file_info info =
        state_info(2u, 8192u, UINT64_C(0x77), true, UINT64_C(0x88));
    CHECK(k3_prefix_catalog_add(
              catalog, prefix, 2u, "/states/a.k3", &info),
          "catalog insertion");
    CHECK(k3_prefix_catalog_add(
              catalog, prefix, 2u, "/states/a.k3", &info) &&
              k3_prefix_catalog_count(catalog) == 1u,
          "idempotent insertion duplicated an entry");
    CHECK(!k3_prefix_catalog_add(
              catalog, prefix, 2u, "/states/conflict.k3", &info),
          "conflicting state path was accepted");

    k3_engine_state_file_info other = info;
    other.context = 32768u;
    CHECK(k3_prefix_catalog_add(
              catalog, prefix, 2u, "/states/other-context.k3", &other) &&
              k3_prefix_catalog_count(catalog) == 2u,
          "identity-specific entry was rejected");
    const k3_prefix_catalog_identity identity = {
        .format_version = 1u,
        .context = 8192u,
        .model_layout_crc64 = UINT64_C(0x77),
        .q8_projections = true,
    };
    CHECK(k3_prefix_catalog_remove(
              catalog, prefix, 2u, &identity) &&
              k3_prefix_catalog_count(catalog) == 1u,
          "identity-specific removal failed");
    CHECK(!k3_prefix_catalog_remove(
              catalog, prefix, 2u, &identity),
          "missing removal unexpectedly succeeded");

    info.token_position = 3u;
    CHECK(!k3_prefix_catalog_add(
              catalog, prefix, 2u, "/states/invalid.k3", &info),
          "token/state position mismatch was accepted");
    info.token_position = 2u;
    info.context = 1u;
    CHECK(!k3_prefix_catalog_add(
              catalog, prefix, 2u, "/states/invalid.k3", &info),
          "state position beyond context was accepted");
    k3_prefix_catalog_destroy(catalog);
    return 0;
}

static int test_deep_prefix_compares_every_token(void) {
    enum { COUNT = 1024 };
    uint32_t *prefix = (uint32_t *)malloc(COUNT * sizeof(*prefix));
    uint32_t *candidate =
        (uint32_t *)malloc((COUNT + 2u) * sizeof(*candidate));
    CHECK(prefix != NULL && candidate != NULL, "deep-prefix allocation");
    for (uint32_t index = 0u; index < COUNT; index++) {
        prefix[index] = index * 17u + 3u;
        candidate[index] = prefix[index];
    }
    candidate[COUNT] = 9u;
    candidate[COUNT + 1u] = 10u;
    const k3_engine_state_file_info info =
        state_info(COUNT, 8192u, UINT64_C(0x99), true, UINT64_C(0x55));
    const k3_prefix_catalog_identity identity = {
        .format_version = 1u,
        .context = 8192u,
        .model_layout_crc64 = UINT64_C(0x99),
        .q8_projections = true,
    };
    k3_prefix_catalog *catalog = NULL;
    CHECK(k3_prefix_catalog_create(&catalog) &&
              k3_prefix_catalog_add(
                  catalog, prefix, COUNT, "/states/deep.k3", &info),
          "deep-prefix insertion");
    k3_prefix_catalog_match match;
    CHECK(k3_prefix_catalog_find(
              catalog, candidate, COUNT + 2u, &identity, &match),
          "deep exact prefix lookup");
    candidate[COUNT - 1u] ^= 1u;
    CHECK(!k3_prefix_catalog_find(
              catalog, candidate, COUNT + 2u, &identity, &match),
          "deep final-token mismatch was admitted");
    k3_prefix_catalog_destroy(catalog);
    free(candidate);
    free(prefix);
    return 0;
}

int main(void) {
    CHECK(test_longest_exact_eligible_prefix() == 0,
          "longest exact eligible prefix");
    CHECK(test_identity_and_exactness_rejection() == 0,
          "identity and exactness rejection");
    CHECK(test_duplicate_remove_and_metadata_validation() == 0,
          "duplicate/remove/metadata validation");
    CHECK(test_deep_prefix_compares_every_token() == 0,
          "deep exact comparison");
    printf("K3 exact prefix checkpoint catalog: PASS\n");
    return 0;
}
