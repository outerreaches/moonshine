#include "k3_expert_cache.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                       \
            return false;                                                   \
        }                                                                   \
    } while (0)

enum {
    K3_TRACE_MOE_LAYERS = 92,
    K3_TRACE_TOP_K = 16,
};

static bool commit_batch(k3_expert_cache *cache,
                         uint16_t layer,
                         const uint16_t *ids,
                         uint16_t count,
                         k3_expert_cache_access *accesses) {
    char error[256];
    CHECK(k3_expert_cache_plan(cache, layer, ids, count, accesses,
                               error, sizeof(error)),
          error);
    CHECK(k3_expert_cache_commit(cache, layer, error, sizeof(error)), error);
    return true;
}

static bool unit_test(void) {
    char error[256];
    k3_expert_cache *cache = NULL;
    CHECK(k3_expert_cache_create(&cache, 2u, 2u,
                                 error, sizeof(error)),
          error);
    CHECK(k3_expert_cache_slot_count(cache) == 4u, "slot count");
    CHECK(k3_expert_cache_storage_bytes(cache, 100u) == 400u,
          "storage-byte count");

    const uint16_t first[] = { 1u, 2u, 3u };
    k3_expert_cache_access first_access[3];
    CHECK(commit_batch(cache, 0u, first, 3u, first_access),
          "first cache batch");
    CHECK(!first_access[0].hit && !first_access[0].admit,
          "oldest miss should fall outside capacity");
    CHECK(first_access[1].admit && first_access[2].admit,
          "two newest misses should be admitted");
    CHECK(first_access[1].destination_slot !=
              first_access[2].destination_slot,
          "admissions must use distinct slots");

    const uint16_t second[] = { 3u, 2u, 4u };
    k3_expert_cache_access second_access[3];
    CHECK(k3_expert_cache_plan(cache, 0u, second, 3u, second_access,
                               error, sizeof(error)),
          error);
    CHECK(second_access[0].hit && second_access[1].hit,
          "token-batch hits are tested before eviction");
    CHECK(second_access[2].admit, "newest miss should be admitted");
    CHECK(second_access[2].destination_slot ==
              second_access[0].source_slot,
          "evicted hit slot should be reused only after compute");
    CHECK(!k3_expert_cache_plan(cache, 0u, second, 3u, second_access,
                                error, sizeof(error)),
          "a layer must reject a second pending plan");
    k3_expert_cache_abort(cache, 0u);

    const uint16_t third[] = { 2u, 3u };
    k3_expert_cache_access third_access[2];
    CHECK(commit_batch(cache, 0u, third, 2u, third_access),
          "post-abort cache batch");
    CHECK(third_access[0].hit && third_access[1].hit,
          "abort must preserve live cache state");

    const uint16_t other_layer[] = { 9u };
    k3_expert_cache_access other_access[1];
    CHECK(commit_batch(cache, 1u, other_layer, 1u, other_access),
          "second-layer cache batch");
    CHECK(!other_access[0].hit && other_access[0].admit,
          "layers must have independent caches");

    k3_expert_cache_stats stats;
    k3_expert_cache_get_stats(cache, &stats);
    CHECK(stats.batches == 3u, "committed batch telemetry");
    CHECK(stats.accesses == 6u && stats.hits == 2u &&
              stats.misses == 4u,
          "hit/miss telemetry");
    CHECK(stats.admissions == 3u && stats.evictions == 0u,
          "admission/eviction telemetry");
    CHECK(k3_expert_cache_invalidate_slots(
              cache, other_access[0].destination_slot, 1u,
              error, sizeof(error)),
          error);
    const uint16_t invalidated_probe[] = { 9u };
    k3_expert_cache_access invalidated_access[1];
    CHECK(k3_expert_cache_plan(
              cache, 1u, invalidated_probe, 1u,
              invalidated_access, error, sizeof(error)),
          error);
    CHECK(!invalidated_access[0].hit &&
              invalidated_access[0].admit &&
              invalidated_access[0].destination_slot ==
                  other_access[0].destination_slot,
          "slot invalidation must forget only its mapping");
    k3_expert_cache_abort(cache, 1u);
    const uint16_t preserved_probe[] = { 2u };
    k3_expert_cache_access preserved_access[1];
    CHECK(k3_expert_cache_plan(
              cache, 0u, preserved_probe, 1u,
              preserved_access, error, sizeof(error)),
          error);
    CHECK(preserved_access[0].hit,
          "slot invalidation must preserve other layers");
    k3_expert_cache_abort(cache, 0u);
    k3_expert_cache_get_stats(cache, &stats);
    CHECK(stats.evictions == 1u,
          "slot invalidation must count the removed mapping");
    CHECK(!k3_expert_cache_invalidate_slots(
              cache, 4u, 1u, error, sizeof(error)),
          "out-of-range invalidation must fail");
    CHECK(k3_expert_cache_reset(cache, error, sizeof(error)),
          error);
    k3_expert_cache_get_stats(cache, &stats);
    CHECK(stats.batches == 0u && stats.accesses == 0u,
          "reset must clear cache telemetry");
    const uint16_t reset_probe[] = { 2u };
    k3_expert_cache_access reset_access[1];
    CHECK(commit_batch(
              cache, 0u, reset_probe, 1u, reset_access),
          "post-reset cache batch");
    CHECK(!reset_access[0].hit && reset_access[0].admit,
          "reset must forget resident expert mappings");
    k3_expert_cache_destroy(cache);

    k3_expert_cache *range_cache = NULL;
    CHECK(k3_expert_cache_create(
              &range_cache, 3u, 3u, error, sizeof(error)),
          error);
    const uint16_t range_seed[][3] = {
        { 10u, 11u, 12u },
        { 20u, 21u, 22u },
        { 30u, 31u, 32u },
    };
    for (uint16_t layer = 0u; layer < 3u; layer++) {
        k3_expert_cache_access seeded[3];
        CHECK(commit_batch(
                  range_cache, layer, range_seed[layer], 3u, seeded),
              "range invalidation seed batch");
    }
    CHECK(k3_expert_cache_invalidate_slots(
              range_cache, 2u, 5u, error, sizeof(error)),
          error);
    const uint16_t range_preserved0[] = { 10u, 11u };
    k3_expert_cache_access range_preserved0_access[2];
    CHECK(k3_expert_cache_plan(
              range_cache, 0u, range_preserved0, 2u,
              range_preserved0_access, error, sizeof(error)),
          error);
    CHECK(range_preserved0_access[0].hit &&
              range_preserved0_access[1].hit,
          "cross-layer invalidation must preserve the prefix slots");
    k3_expert_cache_abort(range_cache, 0u);
    const uint16_t range_removed1[] = { 20u };
    k3_expert_cache_access range_removed1_access[1];
    CHECK(k3_expert_cache_plan(
              range_cache, 1u, range_removed1, 1u,
              range_removed1_access, error, sizeof(error)),
          error);
    CHECK(!range_removed1_access[0].hit &&
              range_removed1_access[0].admit,
          "cross-layer invalidation must remove a fully covered layer");
    k3_expert_cache_abort(range_cache, 1u);
    const uint16_t range_mixed2[] = { 30u, 31u, 32u };
    k3_expert_cache_access range_mixed2_access[3];
    CHECK(k3_expert_cache_plan(
              range_cache, 2u, range_mixed2, 3u,
              range_mixed2_access, error, sizeof(error)),
          error);
    CHECK(!range_mixed2_access[0].hit &&
              range_mixed2_access[1].hit &&
              range_mixed2_access[2].hit,
          "cross-layer invalidation must preserve the suffix slots");
    k3_expert_cache_abort(range_cache, 2u);
    k3_expert_cache_get_stats(range_cache, &stats);
    CHECK(stats.evictions == 5u,
          "cross-layer invalidation telemetry must count every mapping");
    k3_expert_cache_destroy(range_cache);

    k3_expert_cache *edge_cache = NULL;
    CHECK(k3_expert_cache_create(&edge_cache, 1u, 2u,
                                 error, sizeof(error)),
          error);
    const uint16_t edge_seed[] = { 1u, 2u };
    k3_expert_cache_access edge_seed_access[2];
    CHECK(commit_batch(edge_cache, 0u, edge_seed, 2u, edge_seed_access),
          "edge seed batch");
    const uint16_t edge_reentry[] = { 3u, 1u };
    k3_expert_cache_access edge_reentry_access[2];
    CHECK(commit_batch(edge_cache, 0u, edge_reentry, 2u,
                       edge_reentry_access),
          "same-batch hit re-entry");
    CHECK(edge_reentry_access[1].hit,
          "re-entered expert should be a token-batch hit");
    uint32_t old_source = edge_reentry_access[1].source_slot;
    const uint16_t edge_probe[] = { 1u };
    k3_expert_cache_access edge_probe_access[1];
    CHECK(k3_expert_cache_plan(edge_cache, 0u, edge_probe, 1u,
                               edge_probe_access, error, sizeof(error)),
          error);
    CHECK(edge_probe_access[0].hit &&
              edge_probe_access[0].source_slot == old_source,
          "same-batch re-entry must preserve the resident bytes");
    k3_expert_cache_abort(edge_cache, 0u);
    k3_expert_cache_destroy(edge_cache);

    printf("K3 expert cache unit test: PASS\n");
    return true;
}

static bool replay_cell(k3_expert_cache *cache,
                        uint16_t layer,
                        const uint16_t ids[K3_TRACE_TOP_K]) {
    k3_expert_cache_access accesses[K3_TRACE_TOP_K];
    return commit_batch(cache, layer, ids, K3_TRACE_TOP_K, accesses);
}

static bool replay_trace(const char *path, uint16_t capacity) {
    FILE *input = fopen(path, "r");
    CHECK(input != NULL, "open normalized router trace");
    char error[256];
    k3_expert_cache *cache = NULL;
    CHECK(k3_expert_cache_create(&cache, K3_TRACE_MOE_LAYERS, capacity,
                                 error, sizeof(error)),
          error);

    char line[512];
    CHECK(fgets(line, sizeof(line), input) != NULL, "trace header");
    CHECK(strncmp(
              line,
              "token_index,token_id,layer,selection_slot,expert_id,gate_weight",
              strlen("token_index,token_id,layer,selection_slot,expert_id,"
                     "gate_weight")) == 0,
          "normalized trace header");
    uint64_t current_token = UINT64_MAX;
    uint16_t current_layer = UINT16_MAX;
    uint16_t ids[K3_TRACE_TOP_K];
    bool present[K3_TRACE_TOP_K] = { false };
    uint16_t row_count = 0;

    while (fgets(line, sizeof(line), input)) {
        uint64_t token;
        unsigned token_id, layer, rank, expert;
        double weight;
        CHECK(sscanf(line, "%" SCNu64 ",%u,%u,%u,%u,%lf",
                     &token, &token_id, &layer, &rank, &expert,
                     &weight) == 6,
              "parse normalized trace row");
        (void)token_id;
        (void)weight;
        CHECK(layer >= 1u && layer <= K3_TRACE_MOE_LAYERS &&
                  rank < K3_TRACE_TOP_K &&
                  expert < UINT16_MAX,
              "normalized trace value range");
        if (current_token != UINT64_MAX &&
            (token != current_token || layer != current_layer)) {
            CHECK(row_count == K3_TRACE_TOP_K,
                  "normalized trace cell row count");
            CHECK(replay_cell(cache, current_layer - 1u, ids),
                  "replay normalized trace cell");
            memset(present, 0, sizeof(present));
            row_count = 0;
        }
        current_token = token;
        current_layer = (uint16_t)layer;
        CHECK(!present[rank], "duplicate normalized trace rank");
        ids[rank] = (uint16_t)expert;
        present[rank] = true;
        row_count++;
    }
    if (current_token != UINT64_MAX) {
        CHECK(row_count == K3_TRACE_TOP_K,
              "final normalized trace cell row count");
        CHECK(replay_cell(cache, current_layer - 1u, ids),
              "replay final normalized trace cell");
    }
    fclose(input);

    k3_expert_cache_stats stats;
    k3_expert_cache_get_stats(cache, &stats);
    double hit_rate =
        stats.accesses ? (double)stats.hits / (double)stats.accesses : 0.0;
    printf("K3 expert cache trace replay: capacity=%u layers=%u "
           "batches=%" PRIu64 " accesses=%" PRIu64 " hits=%" PRIu64
           " hit_rate=%.6f storage_gib=%.3f\n",
           capacity, K3_TRACE_MOE_LAYERS, stats.batches, stats.accesses,
           stats.hits, hit_rate,
           (double)k3_expert_cache_storage_bytes(
               cache, UINT64_C(17551360)) /
               (1024.0 * 1024.0 * 1024.0));
    k3_expert_cache_destroy(cache);
    return true;
}

int main(int argc, char **argv) {
    if (!unit_test()) return 1;
    if (argc == 1) return 0;
    if (argc != 3) {
        fprintf(stderr, "usage: %s [NORMALIZED_TRACE CAPACITY]\n", argv[0]);
        return 2;
    }
    unsigned long parsed = strtoul(argv[2], NULL, 10);
    if (parsed == 0 || parsed > UINT16_MAX) {
        fprintf(stderr, "invalid cache capacity: %s\n", argv[2]);
        return 2;
    }
    return replay_trace(argv[1], (uint16_t)parsed) ? 0 : 1;
}
