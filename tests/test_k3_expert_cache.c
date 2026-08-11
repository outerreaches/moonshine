#include "k3_expert_cache.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    K3_TRACE_EXPERTS = 896,
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
    uint16_t snapshot[2] = { 0u, 0u };
    uint16_t snapshot_count = 0u;
    CHECK(k3_expert_cache_snapshot_layer(
              cache, 0u, snapshot, 2u, &snapshot_count,
              error, sizeof(error)),
          error);
    CHECK(snapshot_count == 2u &&
              snapshot[0] == 2u && snapshot[1] == 3u,
          "cache snapshot must preserve oldest-to-newest LRU order");

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
                        const uint16_t ids[K3_TRACE_TOP_K],
                        uint32_t *hit_mask) {
    k3_expert_cache_access accesses[K3_TRACE_TOP_K];
    if (!commit_batch(
            cache, layer, ids, K3_TRACE_TOP_K, accesses)) {
        return false;
    }
    if (hit_mask) {
        *hit_mask = 0u;
        for (uint16_t rank = 0u; rank < K3_TRACE_TOP_K; rank++) {
            if (accesses[rank].hit) {
                *hit_mask |= UINT32_C(1) << rank;
            }
        }
    }
    return true;
}

typedef struct {
    bool has_capture;
    uint64_t capture;
    uint16_t occupancy[K3_TRACE_MOE_LAYERS];
} decode_seed_info;

typedef struct {
    uint64_t capture;
    uint64_t steps;
    uint64_t accesses;
    uint64_t hits;
    uint64_t misses;
    uint64_t read_requests;
    uint64_t logical_expert_bytes;
    uint64_t physical_read_bytes;
    uint64_t wait_calls;
    uint64_t completions;
    uint32_t max_inflight;
    double timings[6];
} decode_ledger_info;

static bool parse_unsigned_csv(
        char *line,
        uint64_t *values,
        size_t count) {
    char *cursor = line;
    for (size_t i = 0u; i < count; i++) {
        if (*cursor == '-') return false;
        errno = 0;
        char *end = NULL;
        values[i] = strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor) return false;
        if (i + 1u < count) {
            if (*end != ',') return false;
            cursor = end + 1;
        } else {
            while (*end == '\r' || *end == '\n') end++;
            if (*end != '\0') return false;
        }
    }
    return true;
}

static bool add_u64_checked(uint64_t *total, uint64_t value) {
    if (UINT64_MAX - *total < value) return false;
    *total += value;
    return true;
}

static bool multiply_u64_checked(
        uint64_t left,
        uint64_t right,
        uint64_t *product) {
    if (left != 0u && right > UINT64_MAX / left) return false;
    *product = left * right;
    return true;
}

static bool timing_nearly_equal(double left, double right) {
    const double scale = fmax(1.0, fmax(fabs(left), fabs(right)));
    return fabs(left - right) <= 1e-6 * scale;
}

static bool validate_decode_ledger(
        const char *path,
        decode_ledger_info *info) {
    memset(info, 0, sizeof(*info));
    FILE *input = fopen(path, "r");
    if (!input) {
        fprintf(stderr, "FAIL: open decode ledger\n");
        return false;
    }
    bool ok = false;
    char line[1024];
    static const char expected_header[] =
        "capture,scope,layer,steps,accesses,hits,misses,"
        "read_requests,logical_expert_bytes,physical_read_bytes,"
        "wait_calls,completions,max_inflight,pre_moe_seconds,"
        "io_wait_seconds,expert_pipeline_seconds,"
        "expert_sync_seconds,shared_sync_seconds,"
        "host_interval_seconds";
    if (!fgets(line, sizeof(line), input)) {
        fprintf(stderr, "FAIL: decode ledger header\n");
        goto done;
    }
    line[strcspn(line, "\r\n")] = '\0';
    if (strcmp(line, expected_header) != 0) {
        fprintf(stderr, "FAIL: decode ledger header\n");
        goto done;
    }

    uint64_t layer_steps = UINT64_MAX;
    uint64_t sum_accesses = 0u;
    uint64_t sum_hits = 0u;
    uint64_t sum_misses = 0u;
    uint64_t sum_reads = 0u;
    uint64_t sum_logical = 0u;
    uint64_t sum_physical = 0u;
    uint64_t sum_waits = 0u;
    uint64_t sum_completions = 0u;
    uint32_t maximum_inflight = 0u;
    double sum_timings[6] = { 0 };
    uint32_t row = 0u;
    while (fgets(line, sizeof(line), input)) {
        uint64_t capture = 0u;
        char scope[16] = { 0 };
        unsigned layer = 0u;
        uint64_t steps = 0u;
        uint64_t accesses = 0u;
        uint64_t hits = 0u;
        uint64_t misses = 0u;
        uint64_t reads = 0u;
        uint64_t logical = 0u;
        uint64_t physical = 0u;
        uint64_t waits = 0u;
        uint64_t completions = 0u;
        unsigned max_inflight = 0u;
        double timings[6] = { 0 };
        int consumed = 0;
        const int parsed = sscanf(
            line,
            "%" SCNu64 ",%15[^,],%u,%" SCNu64 ",%" SCNu64
            ",%" SCNu64 ",%" SCNu64 ",%" SCNu64 ",%" SCNu64
            ",%" SCNu64 ",%" SCNu64 ",%" SCNu64
            ",%u,%lf,%lf,%lf,%lf,%lf,%lf%n",
            &capture, scope, &layer, &steps, &accesses,
            &hits, &misses, &reads, &logical, &physical,
            &waits, &completions, &max_inflight,
            &timings[0], &timings[1], &timings[2],
            &timings[3], &timings[4], &timings[5], &consumed);
        if (parsed != 19 || consumed <= 0) {
            fprintf(stderr, "FAIL: parse decode ledger row\n");
            goto done;
        }
        const char *tail = line + consumed;
        while (*tail == '\r' || *tail == '\n') tail++;
        if (*tail != '\0') {
            fprintf(stderr, "FAIL: trailing decode ledger data\n");
            goto done;
        }
        uint64_t expected_accesses = 0u;
        uint64_t expected_logical = 0u;
        const uint64_t access_factor = row == 0u ?
            K3_TRACE_TOP_K * K3_TRACE_MOE_LAYERS : K3_TRACE_TOP_K;
        bool timing_ok = true;
        for (uint32_t i = 0u; i < 6u; i++) {
            if (!isfinite(timings[i]) || timings[i] < 0.0) {
                timing_ok = false;
            }
        }
        if (!multiply_u64_checked(
                steps, access_factor, &expected_accesses) ||
            !multiply_u64_checked(
                reads, UINT64_C(17547264), &expected_logical) ||
            accesses != expected_accesses || hits > accesses ||
            misses != accesses - hits || reads != misses ||
            logical != expected_logical || physical < logical ||
            completions != reads || waits > completions ||
            max_inflight > 2u ||
            (reads == 0u ? max_inflight != 0u : max_inflight == 0u) ||
            !timing_ok) {
            fprintf(stderr, "FAIL: decode ledger row accounting\n");
            goto done;
        }
        if (row == 0u) {
            if (strcmp(scope, "summary") != 0 || layer != 0u) {
                fprintf(stderr, "FAIL: decode ledger summary row\n");
                goto done;
            }
            info->capture = capture;
            info->steps = steps;
            info->accesses = accesses;
            info->hits = hits;
            info->misses = misses;
            info->read_requests = reads;
            info->logical_expert_bytes = logical;
            info->physical_read_bytes = physical;
            info->wait_calls = waits;
            info->completions = completions;
            info->max_inflight = (uint32_t)max_inflight;
            memcpy(info->timings, timings, sizeof(info->timings));
        } else {
            if (capture != info->capture ||
                strcmp(scope, "layer") != 0 || layer != row ||
                row > K3_TRACE_MOE_LAYERS) {
                fprintf(stderr, "FAIL: decode ledger layer/capture order\n");
                goto done;
            }
            if (layer_steps == UINT64_MAX) layer_steps = steps;
            if (steps != layer_steps ||
                !add_u64_checked(&sum_accesses, accesses) ||
                !add_u64_checked(&sum_hits, hits) ||
                !add_u64_checked(&sum_misses, misses) ||
                !add_u64_checked(&sum_reads, reads) ||
                !add_u64_checked(&sum_logical, logical) ||
                !add_u64_checked(&sum_physical, physical) ||
                !add_u64_checked(&sum_waits, waits) ||
                !add_u64_checked(&sum_completions, completions)) {
                fprintf(stderr, "FAIL: decode ledger layer totals\n");
                goto done;
            }
            if (max_inflight > maximum_inflight) {
                maximum_inflight = (uint32_t)max_inflight;
            }
            for (uint32_t i = 0u; i < 6u; i++) {
                sum_timings[i] += timings[i];
            }
        }
        row++;
    }
    if (ferror(input)) {
        fprintf(stderr, "FAIL: read decode ledger\n");
        goto done;
    }
    if (row != 1u + K3_TRACE_MOE_LAYERS ||
        layer_steps == UINT64_MAX || info->steps != layer_steps ||
        info->accesses != sum_accesses ||
        info->hits != sum_hits || info->misses != sum_misses ||
        info->read_requests != sum_reads ||
        info->logical_expert_bytes != sum_logical ||
        info->physical_read_bytes != sum_physical ||
        info->wait_calls != sum_waits ||
        info->completions != sum_completions ||
        info->max_inflight != maximum_inflight) {
        fprintf(stderr, "FAIL: incomplete or unreconciled decode ledger\n");
        goto done;
    }
    for (uint32_t i = 0u; i < 6u; i++) {
        if (!timing_nearly_equal(info->timings[i], sum_timings[i])) {
            fprintf(stderr, "FAIL: decode ledger timing totals\n");
            goto done;
        }
    }
    ok = true;

done:
    if (fclose(input) != 0) {
        fprintf(stderr, "FAIL: close decode ledger\n");
        ok = false;
    }
    return ok;
}

static bool seed_decode_cache(
        const char *path,
        k3_expert_cache *cache,
        uint16_t source_capacity,
        decode_seed_info *info) {
    memset(info, 0, sizeof(*info));
    FILE *input = fopen(path, "r");
    if (!input) {
        fprintf(stderr, "FAIL: open decode cache snapshot\n");
        return false;
    }
    bool ok = false;
    char line[256];
    if (!fgets(line, sizeof(line), input)) {
        fprintf(stderr, "FAIL: decode cache snapshot header\n");
        goto done;
    }
    line[strcspn(line, "\r\n")] = '\0';
    if (strcmp(line, "capture,layer,lru_rank,expert_id") != 0) {
        fprintf(stderr, "FAIL: decode cache snapshot header\n");
        goto done;
    }
    uint16_t ids[K3_TRACE_EXPERTS];
    k3_expert_cache_access accesses[K3_TRACE_EXPERTS];
    uint16_t current_layer = 0u;
    uint16_t count = 0u;
    while (fgets(line, sizeof(line), input)) {
        uint64_t values[4];
        if (!parse_unsigned_csv(line, values, 4u)) {
            fprintf(stderr, "FAIL: parse decode cache snapshot row\n");
            goto done;
        }
        const uint64_t capture = values[0];
        const uint64_t layer = values[1];
        const uint64_t rank = values[2];
        const uint64_t expert = values[3];
        if (!info->has_capture) {
            info->has_capture = true;
            info->capture = capture;
        }
        if (capture != info->capture ||
            layer < 1u || layer > K3_TRACE_MOE_LAYERS ||
            rank >= K3_TRACE_EXPERTS ||
            expert >= K3_TRACE_EXPERTS) {
            fprintf(stderr, "FAIL: decode cache snapshot value range\n");
            goto done;
        }
        if (current_layer != 0u && layer != current_layer) {
            if (layer <= current_layer || count > source_capacity) {
                fprintf(stderr, "FAIL: decode cache snapshot layer order/capacity\n");
                goto done;
            }
            if (!commit_batch(
                    cache, current_layer - 1u, ids, count, accesses)) {
                goto done;
            }
            info->occupancy[current_layer - 1u] = count;
            count = 0u;
        }
        if (rank != count) {
            fprintf(stderr, "FAIL: decode cache snapshot rank order\n");
            goto done;
        }
        current_layer = (uint16_t)layer;
        ids[count++] = (uint16_t)expert;
    }
    if (ferror(input)) {
        fprintf(stderr, "FAIL: read decode cache snapshot\n");
        goto done;
    }
    if (current_layer != 0u) {
        if (count > source_capacity) {
            fprintf(stderr, "FAIL: decode cache snapshot exceeds source capacity\n");
            goto done;
        }
        if (!commit_batch(
                cache, current_layer - 1u, ids, count, accesses)) {
            goto done;
        }
        info->occupancy[current_layer - 1u] = count;
    }
    ok = true;

done:
    if (fclose(input) != 0) {
        fprintf(stderr, "FAIL: close decode cache snapshot\n");
        ok = false;
    }
    return ok;
}

static bool replay_decode_routes(
        FILE *input,
        k3_expert_cache *cache,
        uint16_t capacity,
        uint16_t source_capacity,
        const decode_seed_info *seed,
        const decode_ledger_info *ledger,
        const k3_expert_cache_stats *seed_stats,
        bool fresh_empty_source,
        bool report) {
    char line[1024];
    uint64_t rows = 0u;
    uint64_t observed_mismatches = 0u;
    uint64_t expected_capture = ledger->capture;
    if (seed->has_capture && seed->capture != expected_capture) {
        fprintf(stderr, "FAIL: cache snapshot capture differs from ledger\n");
        return false;
    }
    uint64_t current_step = UINT64_MAX;
    uint64_t current_position = UINT64_MAX;
    while (fgets(line, sizeof(line), input)) {
        uint64_t values[5u + K3_TRACE_TOP_K];
        if (!parse_unsigned_csv(
                line, values, 5u + K3_TRACE_TOP_K)) {
            fprintf(stderr, "FAIL: parse decode route row\n");
            return false;
        }
        const uint64_t capture = values[0];
        const uint64_t step = values[1];
        const uint64_t position = values[2];
        const uint64_t layer = values[3];
        const uint64_t observed_hit_mask = values[4];
        const uint64_t expected_layer =
            rows % K3_TRACE_MOE_LAYERS + 1u;
        if (capture != expected_capture ||
            layer != expected_layer ||
            observed_hit_mask > UINT16_MAX) {
            fprintf(stderr, "FAIL: decode route capture/layer/value range\n");
            return false;
        }
        if (expected_layer == 1u) {
            if (rows == 0u) {
                if (step != 0u) {
                    fprintf(stderr, "FAIL: decode route first step\n");
                    return false;
                }
            } else if (current_step == UINT64_MAX ||
                       current_position == UINT64_MAX ||
                       step != current_step + 1u ||
                       position != current_position + 1u) {
                fprintf(stderr, "FAIL: decode route step/position order\n");
                return false;
            }
            current_step = step;
            current_position = position;
        } else if (step != current_step ||
                   position != current_position) {
            fprintf(stderr, "FAIL: decode route step cell consistency\n");
            return false;
        }
        uint16_t ids[K3_TRACE_TOP_K];
        bool seen[K3_TRACE_EXPERTS] = { false };
        for (uint16_t rank = 0u; rank < K3_TRACE_TOP_K; rank++) {
            if (values[5u + rank] >= K3_TRACE_EXPERTS) {
                fprintf(stderr, "FAIL: decode route expert range\n");
                return false;
            }
            ids[rank] = (uint16_t)values[5u + rank];
            if (seen[ids[rank]]) {
                fprintf(stderr, "FAIL: duplicate decode route expert\n");
                return false;
            }
            seen[ids[rank]] = true;
        }
        uint32_t replayed_hit_mask = 0u;
        if (!replay_cell(
                cache, (uint16_t)(layer - 1u),
                ids, &replayed_hit_mask)) {
            return false;
        }
        if (replayed_hit_mask != (uint32_t)observed_hit_mask) {
            observed_mismatches++;
        }
        rows++;
    }
    if (ferror(input)) {
        fprintf(stderr, "FAIL: read decode route trace\n");
        return false;
    }
    if (rows == 0u || rows % K3_TRACE_MOE_LAYERS != 0u ||
        rows / K3_TRACE_MOE_LAYERS != ledger->steps) {
        fprintf(stderr, "FAIL: incomplete decode route capture\n");
        return false;
    }
    k3_expert_cache_stats stats;
    k3_expert_cache_get_stats(cache, &stats);
    stats.batches -= seed_stats->batches;
    stats.accesses -= seed_stats->accesses;
    stats.hits -= seed_stats->hits;
    stats.misses -= seed_stats->misses;
    stats.admissions -= seed_stats->admissions;
    stats.evictions -= seed_stats->evictions;
    if (stats.batches != rows ||
        stats.accesses != rows * K3_TRACE_TOP_K) {
        fprintf(stderr, "FAIL: decode route replay accounting\n");
        return false;
    }
    if (capacity == source_capacity &&
        (stats.accesses != ledger->accesses ||
         stats.hits != ledger->hits || stats.misses != ledger->misses)) {
        fprintf(stderr, "FAIL: source-capacity replay differs from ledger\n");
        return false;
    }
    const double hit_rate =
        stats.accesses ?
            (double)stats.hits / (double)stats.accesses : 0.0;
    if (report) {
        printf("K3 decode cache replay: capture=%" PRIu64
               " source_capacity=%u capacity=%u fresh_empty_source=%s "
               "rows=%" PRIu64
               " accesses=%" PRIu64 " hits=%" PRIu64
               " misses=%" PRIu64 " hit_rate=%.6f"
               " observed_mask_mismatches=%" PRIu64
               " storage_gib=%.3f\n",
               expected_capture, source_capacity, capacity,
               fresh_empty_source ? "yes" : "no", rows,
               stats.accesses, stats.hits, stats.misses,
               hit_rate, observed_mismatches,
               (double)k3_expert_cache_storage_bytes(
                   cache, UINT64_C(17547264)) /
                   (1024.0 * 1024.0 * 1024.0));
    }
    if (capacity == source_capacity && observed_mismatches != 0u) {
        fprintf(stderr,
                "FAIL: source-capacity observed hit masks differ\n");
        return false;
    }
    return true;
}

static bool replay_trace(
        const char *seed_path,
        const char *ledger_path,
        const char *path,
        uint16_t source_capacity,
        uint16_t capacity,
        bool fresh_empty_source) {
    decode_ledger_info ledger = { 0 };
    if (ledger_path != NULL &&
        !validate_decode_ledger(ledger_path, &ledger)) {
        return false;
    }
    FILE *input = fopen(path, "r");
    if (!input) {
        fprintf(stderr, "FAIL: open normalized router trace\n");
        return false;
    }
    char error[256];
    k3_expert_cache *cache = NULL;
    if (!k3_expert_cache_create(
            &cache, K3_TRACE_MOE_LAYERS, capacity,
            error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        (void)fclose(input);
        return false;
    }
    bool ok = false;
    k3_expert_cache_stats seed_stats = { 0 };
    decode_seed_info seed = { 0 };
    if (seed_path != NULL) {
        if (!seed_decode_cache(
                seed_path, cache, source_capacity, &seed)) {
            goto done;
        }
        k3_expert_cache_get_stats(cache, &seed_stats);
        if (capacity > source_capacity) {
            if (!fresh_empty_source) {
                fprintf(
                    stderr,
                    "FAIL: target capacity exceeds source capacity without "
                    "explicit fresh-empty provenance\n");
                goto done;
            }
            for (uint16_t layer = 0u;
                 layer < K3_TRACE_MOE_LAYERS; layer++) {
                if (seed.occupancy[layer] != 0u) {
                    fprintf(
                        stderr,
                        "FAIL: target capacity exceeds a nonempty source "
                        "snapshot; prior evictions or invalidations are "
                        "unknown\n");
                    goto done;
                }
            }
        }
    }

    char line[512];
    if (!fgets(line, sizeof(line), input)) {
        fprintf(stderr, "FAIL: trace header\n");
        goto done;
    }
    if (strncmp(
            line,
            "capture,step,position,layer,observed_hit_mask,",
            strlen("capture,step,position,layer,observed_hit_mask,")) == 0) {
        if (ledger_path == NULL) {
            fprintf(stderr, "FAIL: decode routes require a committed ledger\n");
            goto done;
        }
        ok = replay_decode_routes(
            input, cache, capacity, source_capacity,
            &seed, &ledger, &seed_stats,
            fresh_empty_source, true);
        goto done;
    }
    if (strncmp(
            line,
            "token_index,token_id,layer,selection_slot,expert_id,gate_weight",
            strlen("token_index,token_id,layer,selection_slot,expert_id,"
                   "gate_weight")) != 0) {
        fprintf(stderr, "FAIL: normalized trace header\n");
        goto done;
    }
    uint64_t current_token = UINT64_MAX;
    uint16_t current_layer = UINT16_MAX;
    uint16_t ids[K3_TRACE_TOP_K];
    bool present[K3_TRACE_TOP_K] = { false };
    uint16_t row_count = 0;

    while (fgets(line, sizeof(line), input)) {
        uint64_t token;
        unsigned token_id, layer, rank, expert;
        double weight;
        if (sscanf(line, "%" SCNu64 ",%u,%u,%u,%u,%lf",
                   &token, &token_id, &layer, &rank, &expert,
                   &weight) != 6) {
            fprintf(stderr, "FAIL: parse normalized trace row\n");
            goto done;
        }
        (void)token_id;
        (void)weight;
        if (layer < 1u || layer > K3_TRACE_MOE_LAYERS ||
            rank >= K3_TRACE_TOP_K || expert >= UINT16_MAX) {
            fprintf(stderr, "FAIL: normalized trace value range\n");
            goto done;
        }
        if (current_token != UINT64_MAX &&
            (token != current_token || layer != current_layer)) {
            if (row_count != K3_TRACE_TOP_K ||
                !replay_cell(cache, current_layer - 1u, ids, NULL)) {
                fprintf(stderr, "FAIL: normalized trace cell\n");
                goto done;
            }
            memset(present, 0, sizeof(present));
            row_count = 0;
        }
        current_token = token;
        current_layer = (uint16_t)layer;
        if (present[rank]) {
            fprintf(stderr, "FAIL: duplicate normalized trace rank\n");
            goto done;
        }
        ids[rank] = (uint16_t)expert;
        present[rank] = true;
        row_count++;
    }
    if (ferror(input)) {
        fprintf(stderr, "FAIL: read normalized trace\n");
        goto done;
    }
    if (current_token != UINT64_MAX) {
        if (row_count != K3_TRACE_TOP_K ||
            !replay_cell(cache, current_layer - 1u, ids, NULL)) {
            fprintf(stderr, "FAIL: final normalized trace cell\n");
            goto done;
        }
    }

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
    ok = true;

done:
    if (fclose(input) != 0) {
        fprintf(stderr, "FAIL: close trace input\n");
        ok = false;
    }
    k3_expert_cache_destroy(cache);
    return ok;
}

static bool replay_validation_unit_test(void) {
    char valid[] = "1,2,3,4\n";
    uint64_t values[4];
    CHECK(parse_unsigned_csv(valid, values, 4u),
          "parse valid unsigned CSV row");
    char negative[] = "1,-2,3,4\n";
    CHECK(!parse_unsigned_csv(negative, values, 4u),
          "reject negative unsigned CSV value");
    char overflow[] = "18446744073709551616,2,3,4\n";
    CHECK(!parse_unsigned_csv(overflow, values, 4u),
          "reject overflowing unsigned CSV value");
    char trailing[] = "1,2,3,4,5\n";
    CHECK(!parse_unsigned_csv(trailing, values, 4u),
          "reject trailing CSV field");

    char ledger_path[] = "/tmp/moonshine-ledger-unit-XXXXXX";
    const int ledger_fd = mkstemp(ledger_path);
    CHECK(ledger_fd >= 0, "create decode ledger unit fixture");
    FILE *ledger_output = fdopen(ledger_fd, "w");
    CHECK(ledger_output != NULL, "open decode ledger unit fixture");
    CHECK(fprintf(
              ledger_output,
              "capture,scope,layer,steps,accesses,hits,misses,"
              "read_requests,logical_expert_bytes,physical_read_bytes,"
              "wait_calls,completions,max_inflight,pre_moe_seconds,"
              "io_wait_seconds,expert_pipeline_seconds,"
              "expert_sync_seconds,shared_sync_seconds,"
              "host_interval_seconds\n") > 0,
          "write decode ledger unit header");
    const uint64_t layer_accesses = K3_TRACE_TOP_K;
    const uint64_t total_accesses =
        K3_TRACE_MOE_LAYERS * layer_accesses;
    const uint64_t layer_bytes =
        layer_accesses * UINT64_C(17547264);
    const uint64_t total_bytes =
        total_accesses * UINT64_C(17547264);
    CHECK(fprintf(
              ledger_output,
              "7,summary,0,1,%" PRIu64 ",0,%" PRIu64
              ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
              ",%" PRIu64 ",%" PRIu64
              ",2,0,0,0,0,0,0\n",
              total_accesses, total_accesses, total_accesses,
              total_bytes, total_bytes,
              total_accesses, total_accesses) > 0,
          "write decode ledger unit summary");
    for (uint16_t layer = 1u;
         layer <= K3_TRACE_MOE_LAYERS; layer++) {
        CHECK(fprintf(
                  ledger_output,
                  "7,layer,%u,1,%" PRIu64 ",0,%" PRIu64
                  ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                  ",%" PRIu64 ",%" PRIu64
                  ",2,0,0,0,0,0,0\n",
                  layer, layer_accesses, layer_accesses,
                  layer_accesses, layer_bytes, layer_bytes,
                  layer_accesses, layer_accesses) > 0,
              "write decode ledger unit layer");
    }
    CHECK(fclose(ledger_output) == 0,
          "close decode ledger unit fixture");
    decode_ledger_info ledger = { 0 };
    const bool ledger_ok = validate_decode_ledger(
        ledger_path, &ledger);
    CHECK(unlink(ledger_path) == 0,
          "remove decode ledger unit fixture");
    CHECK(ledger_ok, "validate complete decode ledger fixture");

    FILE *routes = tmpfile();
    CHECK(routes != NULL, "create decode route unit fixture");
    for (uint16_t layer = 1u;
         layer <= K3_TRACE_MOE_LAYERS; layer++) {
        CHECK(fprintf(routes, "7,0,100,%u,0", layer) > 0,
              "write decode route unit fixture");
        for (uint16_t expert = 0u;
             expert < K3_TRACE_TOP_K; expert++) {
            CHECK(fprintf(routes, ",%u", expert) > 0,
                  "write decode route unit fixture experts");
        }
        CHECK(fputc('\n', routes) != EOF,
              "finish decode route unit fixture row");
    }
    CHECK(fflush(routes) == 0 && fseek(routes, 0L, SEEK_SET) == 0,
          "rewind decode route unit fixture");
    char error[256];
    k3_expert_cache *cache = NULL;
    CHECK(k3_expert_cache_create(
              &cache, K3_TRACE_MOE_LAYERS, 30u,
              error, sizeof(error)),
          error);
    const decode_seed_info seed = { 0 };
    const k3_expert_cache_stats seed_stats = { 0 };
    const bool replay_ok = replay_decode_routes(
        routes, cache, 30u, 30u, &seed, &ledger,
        &seed_stats, false, false);
    k3_expert_cache_destroy(cache);
    CHECK(fclose(routes) == 0,
          "close decode route unit fixture");
    CHECK(replay_ok, "replay valid complete decode capture");
    printf("K3 decode cache replay validation: PASS\n");
    return true;
}

static bool parse_capacity_argument(
        const char *text,
        uint16_t *capacity) {
    errno = 0;
    char *end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed == 0u || parsed > K3_TRACE_EXPERTS) {
        return false;
    }
    *capacity = (uint16_t)parsed;
    return true;
}

int main(int argc, char **argv) {
    if (!unit_test() || !replay_validation_unit_test()) return 1;
    if (argc == 1) return 0;
    if (argc != 3 && argc != 6 && argc != 7) {
        fprintf(
            stderr,
            "usage: %s ROUTE_TRACE CAPACITY\n"
            "       %s CACHE_SNAPSHOT LEDGER ROUTE_TRACE "
            "SOURCE_CAPACITY CAPACITY [fresh-empty-source]\n",
            argv[0], argv[0]);
        return 2;
    }
    const bool seeded = argc >= 6;
    const char *seed_path = seeded ? argv[1] : NULL;
    const char *ledger_path = seeded ? argv[2] : NULL;
    const char *trace_path = seeded ? argv[3] : argv[1];
    const bool fresh_empty_source = argc == 7;
    if (fresh_empty_source &&
        strcmp(argv[6], "fresh-empty-source") != 0) {
        fprintf(stderr, "invalid source provenance: %s\n", argv[6]);
        return 2;
    }
    uint16_t source_capacity = 0u;
    uint16_t capacity = 0u;
    if (seeded &&
        !parse_capacity_argument(argv[4], &source_capacity)) {
        fprintf(stderr, "invalid source cache capacity: %s\n", argv[4]);
        return 2;
    }
    const char *capacity_text = seeded ? argv[5] : argv[2];
    if (!parse_capacity_argument(capacity_text, &capacity)) {
        fprintf(stderr, "invalid cache capacity: %s\n", capacity_text);
        return 2;
    }
    if (!seeded) source_capacity = capacity;
    return replay_trace(
        seed_path, ledger_path, trace_path,
        source_capacity, capacity, fresh_empty_source) ? 0 : 1;
}
