/*
 * M2 transactional KV. The load-bearing properties:
 *
 *   - an abort at any point cannot alter committed state, including a windowed
 *     layer that is already full and evicting;
 *   - a commit publishes all 48 layers or none;
 *   - a committed step can be rewound byte-exactly, which for a full windowed
 *     layer means restoring bytes a ring index reset would have lost;
 *   - rollback refuses rather than approximating when the journal is too
 *     shallow.
 *
 * Every state comparison is a full byte comparison of both arrays across all
 * 48 layers, not a spot check of indices.
 */
#include "mimo26_kv.h"
#include "mimo26_architecture.h"
#include "mimo26_ops.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GLOBAL_CAPACITY 512u
#define ROLLBACK_DEPTH 8u   /* the MTP and DFlash draft block size */

static uint32_t next_random(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

/* A full snapshot of every layer's visible bytes and indices. */
typedef struct {
    uint16_t *keys[MIMO26_TEXT_LAYER_COUNT];
    uint16_t *values[MIMO26_TEXT_LAYER_COUNT];
    size_t    history[MIMO26_TEXT_LAYER_COUNT];
    uint64_t  first[MIMO26_TEXT_LAYER_COUNT];
    size_t    key_bytes[MIMO26_TEXT_LAYER_COUNT];
    size_t    value_bytes[MIMO26_TEXT_LAYER_COUNT];
    uint64_t  length;
} snapshot;

static void snapshot_take(const mimo26_kv_cache *cache, snapshot *shot)
{
    memset(shot, 0, sizeof *shot);
    shot->length = mimo26_kv_length(cache);
    for (uint32_t layer = 0; layer < MIMO26_TEXT_LAYER_COUNT; layer++) {
        mimo26_attention_config config;
        assert(mimo26_attention_config_for_layer(layer, &config) ==
               MIMO26_ATTENTION_OK);
        const uint16_t *keys = NULL;
        const uint16_t *values = NULL;
        size_t history = 0;
        uint64_t first = 0;
        assert(mimo26_kv_view(cache, layer, &keys, &values, &history, &first) ==
               MIMO26_KV_OK);
        const size_t capacity = config.window != 0u ? MIMO26_SLIDING_WINDOW
                                                    : GLOBAL_CAPACITY;
        shot->key_bytes[layer] =
            capacity * config.kv_heads * MIMO26_QK_HEAD_DIM * sizeof(uint16_t);
        shot->value_bytes[layer] =
            capacity * config.kv_heads * MIMO26_V_HEAD_DIM * sizeof(uint16_t);
        shot->keys[layer] = malloc(shot->key_bytes[layer]);
        shot->values[layer] = malloc(shot->value_bytes[layer]);
        assert(shot->keys[layer] && shot->values[layer]);
        memcpy(shot->keys[layer], keys, shot->key_bytes[layer]);
        memcpy(shot->values[layer], values, shot->value_bytes[layer]);
        shot->history[layer] = history;
        shot->first[layer] = first;
    }
}

static void snapshot_free(snapshot *shot)
{
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        free(shot->keys[i]);
        free(shot->values[i]);
    }
}

/* Compare only the live prefix: bytes past history are scratch. */
static bool snapshot_equal_live(const mimo26_kv_cache *cache,
                                const snapshot *shot)
{
    if (mimo26_kv_length(cache) != shot->length) {
        return false;
    }
    for (uint32_t layer = 0; layer < MIMO26_TEXT_LAYER_COUNT; layer++) {
        mimo26_attention_config config;
        if (mimo26_attention_config_for_layer(layer, &config) !=
            MIMO26_ATTENTION_OK) {
            return false;
        }
        const uint16_t *keys = NULL;
        const uint16_t *values = NULL;
        size_t history = 0;
        uint64_t first = 0;
        if (mimo26_kv_view(cache, layer, &keys, &values, &history, &first) !=
            MIMO26_KV_OK) {
            return false;
        }
        if (history != shot->history[layer] || first != shot->first[layer]) {
            return false;
        }
        const size_t key_span = config.kv_heads * MIMO26_QK_HEAD_DIM;
        const size_t value_span = config.kv_heads * MIMO26_V_HEAD_DIM;
        if (memcmp(keys, shot->keys[layer],
                   history * key_span * sizeof(uint16_t)) != 0) {
            return false;
        }
        if (memcmp(values, shot->values[layer],
                   history * value_span * sizeof(uint16_t)) != 0) {
            return false;
        }
    }
    return true;
}

/* Stage every layer with content derived from the token index, so any
 * misplacement is detectable. */
static void stage_all(mimo26_kv_cache *cache, uint64_t token, size_t skip_layer)
{
    for (uint32_t layer = 0; layer < MIMO26_TEXT_LAYER_COUNT; layer++) {
        if ((size_t)layer == skip_layer) {
            continue;
        }
        mimo26_attention_config config;
        assert(mimo26_attention_config_for_layer(layer, &config) ==
               MIMO26_ATTENTION_OK);
        uint16_t keys[MIMO26_SWA_KV_HEADS * MIMO26_QK_HEAD_DIM];
        uint16_t values[MIMO26_SWA_KV_HEADS * MIMO26_V_HEAD_DIM];
        uint32_t state = (uint32_t)(token * 7919u + layer * 104729u + 1u);
        for (size_t i = 0; i < config.kv_heads * MIMO26_QK_HEAD_DIM; i++) {
            keys[i] = (uint16_t)(next_random(&state) >> 17);
        }
        for (size_t i = 0; i < config.kv_heads * MIMO26_V_HEAD_DIM; i++) {
            values[i] = (uint16_t)(next_random(&state) >> 17);
        }
        assert(mimo26_kv_stage(cache, layer, keys, values) == MIMO26_KV_OK);
    }
}

static void commit_token(mimo26_kv_cache *cache, uint64_t token)
{
    assert(mimo26_kv_begin(cache, token) == MIMO26_KV_OK);
    stage_all(cache, token, SIZE_MAX);
    assert(mimo26_kv_commit(cache) == MIMO26_KV_OK);
}

int main(void)
{
    mimo26_kv_cache *cache = NULL;
    assert(mimo26_kv_create(&cache, GLOBAL_CAPACITY, ROLLBACK_DEPTH) ==
           MIMO26_KV_OK);
    printf("  ok  cache allocated, %.3f MiB for %u global positions and a "
           "depth-%u journal\n",
           (double)mimo26_kv_allocated_bytes(cache) / (1024.0 * 1024.0),
           GLOBAL_CAPACITY, ROLLBACK_DEPTH);

    /* Position discipline: begin must match the committed length. */
    assert(mimo26_kv_begin(cache, 1u) == MIMO26_KV_INVALID_ARGUMENT);
    assert(mimo26_kv_begin(cache, 0u) == MIMO26_KV_OK);
    assert(mimo26_kv_begin(cache, 0u) == MIMO26_KV_INVALID_STATE);
    assert(mimo26_kv_abort(cache) == MIMO26_KV_OK);
    assert(mimo26_kv_abort(cache) == MIMO26_KV_INVALID_STATE);
    assert(mimo26_kv_length(cache) == 0u);
    printf("  ok  begin rejects a skipped or replayed position\n");

    /* A partial step cannot publish. */
    assert(mimo26_kv_begin(cache, 0u) == MIMO26_KV_OK);
    stage_all(cache, 0u, 17u); /* withhold layer 17 */
    assert(mimo26_kv_commit(cache) == MIMO26_KV_INVALID_STATE);
    assert(mimo26_kv_length(cache) == 0u);
    /* Staging the same layer twice in one step is refused. */
    uint16_t dummy_keys[MIMO26_SWA_KV_HEADS * MIMO26_QK_HEAD_DIM] = {0};
    uint16_t dummy_values[MIMO26_SWA_KV_HEADS * MIMO26_V_HEAD_DIM] = {0};
    assert(mimo26_kv_stage(cache, 1u, dummy_keys, dummy_values) ==
           MIMO26_KV_INVALID_STATE);
    assert(mimo26_kv_abort(cache) == MIMO26_KV_OK);
    printf("  ok  commit refuses a step missing a layer; double-stage refused\n");

    /* Fill past the window so windowed layers are evicting. */
    const uint64_t filled = 200u;
    for (uint64_t token = 0; token < filled; token++) {
        commit_token(cache, token);
    }
    assert(mimo26_kv_length(cache) == filled);
    for (uint32_t layer = 0; layer < MIMO26_TEXT_LAYER_COUNT; layer++) {
        mimo26_attention_config config;
        assert(mimo26_attention_config_for_layer(layer, &config) ==
               MIMO26_ATTENTION_OK);
        size_t history = 0;
        uint64_t first = 0;
        assert(mimo26_kv_view(cache, layer, NULL, NULL, &history, &first) ==
               MIMO26_KV_OK);
        if (config.window != 0u) {
            assert(history == MIMO26_SLIDING_WINDOW);
            assert(first == filled - MIMO26_SLIDING_WINDOW);
        } else {
            assert(history == filled);
            assert(first == 0u);
        }
    }
    printf("  ok  %llu tokens committed; windowed layers hold 128 from "
           "position %llu, global layers hold all %llu\n",
           (unsigned long long)filled,
           (unsigned long long)(filled - MIMO26_SLIDING_WINDOW),
           (unsigned long long)filled);

    /* An abort while full and evicting must not disturb committed bytes. */
    snapshot before;
    snapshot_take(cache, &before);
    assert(mimo26_kv_begin(cache, filled) == MIMO26_KV_OK);
    stage_all(cache, 9999u, SIZE_MAX); /* stage everything, then abandon it */
    assert(mimo26_kv_abort(cache) == MIMO26_KV_OK);
    assert(snapshot_equal_live(cache, &before));
    /* Also abort after staging only part of the step. */
    assert(mimo26_kv_begin(cache, filled) == MIMO26_KV_OK);
    stage_all(cache, 8888u, 30u);
    assert(mimo26_kv_abort(cache) == MIMO26_KV_OK);
    assert(snapshot_equal_live(cache, &before));
    printf("  ok  abort while evicting leaves all 48 layers byte-identical\n");

    /* Rollback of an evicting commit must restore the displaced position. */
    commit_token(cache, filled);
    assert(mimo26_kv_length(cache) == filled + 1u);
    assert(!snapshot_equal_live(cache, &before));
    assert(mimo26_kv_rollback(cache, 1u) == MIMO26_KV_OK);
    assert(snapshot_equal_live(cache, &before));
    printf("  ok  single-step rollback restores the evicted position "
           "byte-exactly\n");

    /* Rollback at the draft block size, which is what speculation needs. */
    for (uint64_t step = 0; step < ROLLBACK_DEPTH; step++) {
        commit_token(cache, filled + step);
    }
    assert(mimo26_kv_length(cache) == filled + ROLLBACK_DEPTH);
    assert(mimo26_kv_rollback_available(cache) == ROLLBACK_DEPTH);
    assert(mimo26_kv_rollback(cache, ROLLBACK_DEPTH) == MIMO26_KV_OK);
    assert(snapshot_equal_live(cache, &before));
    assert(mimo26_kv_length(cache) == filled);
    printf("  ok  %u-step rollback (the draft block size) restores state\n",
           ROLLBACK_DEPTH);

    /* Deeper than the journal is refused, and refusal changes nothing. */
    for (uint64_t step = 0; step < ROLLBACK_DEPTH; step++) {
        commit_token(cache, filled + step);
    }
    snapshot deep;
    snapshot_take(cache, &deep);
    assert(mimo26_kv_rollback(cache, ROLLBACK_DEPTH + 1u) ==
           MIMO26_KV_ROLLBACK_UNAVAILABLE);
    assert(snapshot_equal_live(cache, &deep));
    assert(mimo26_kv_length(cache) == filled + ROLLBACK_DEPTH);
    printf("  ok  rollback deeper than the journal is refused without "
           "touching state\n");

    /* Rollback is refused mid-transaction. */
    assert(mimo26_kv_begin(cache, filled + ROLLBACK_DEPTH) == MIMO26_KV_OK);
    assert(mimo26_kv_rollback(cache, 1u) == MIMO26_KV_INVALID_STATE);
    assert(mimo26_kv_abort(cache) == MIMO26_KV_OK);
    snapshot_free(&deep);

    /* Re-committing after a rollback must reproduce the same state, which is
     * what exact replay after a rejected draft requires. */
    assert(mimo26_kv_rollback(cache, ROLLBACK_DEPTH) == MIMO26_KV_OK);
    for (uint64_t step = 0; step < ROLLBACK_DEPTH; step++) {
        commit_token(cache, filled + step);
    }
    snapshot replayed;
    snapshot_take(cache, &replayed);
    assert(mimo26_kv_rollback(cache, ROLLBACK_DEPTH) == MIMO26_KV_OK);
    for (uint64_t step = 0; step < ROLLBACK_DEPTH; step++) {
        commit_token(cache, filled + step);
    }
    assert(snapshot_equal_live(cache, &replayed));
    printf("  ok  rollback then replay is deterministic\n");
    snapshot_free(&replayed);

    /* Global capacity is enforced rather than silently wrapping. */
    mimo26_kv_reset(cache);
    assert(mimo26_kv_length(cache) == 0u);
    assert(mimo26_kv_rollback_available(cache) == 0u);
    for (uint64_t token = 0; token < GLOBAL_CAPACITY; token++) {
        commit_token(cache, token);
    }
    assert(mimo26_kv_begin(cache, GLOBAL_CAPACITY) == MIMO26_KV_OK);
    /* Layer 0 is a full-attention layer and is now at capacity. */
    assert(mimo26_kv_stage(cache, 0u, dummy_keys, dummy_values) ==
           MIMO26_KV_CAPACITY_EXCEEDED);
    /* A windowed layer still accepts, since it evicts instead. */
    assert(mimo26_kv_stage(cache, 1u, dummy_keys, dummy_values) ==
           MIMO26_KV_OK);
    assert(mimo26_kv_abort(cache) == MIMO26_KV_OK);
    assert(mimo26_kv_length(cache) == GLOBAL_CAPACITY);
    printf("  ok  global capacity refuses a %u-th position; windowed layers "
           "evict instead\n", GLOBAL_CAPACITY);

    snapshot_free(&before);
    mimo26_kv_destroy(cache);

    /* A zero-depth journal is legal but admits no rollback. */
    mimo26_kv_cache *no_journal = NULL;
    assert(mimo26_kv_create(&no_journal, 64u, 0u) == MIMO26_KV_OK);
    for (uint64_t token = 0; token < 4u; token++) {
        commit_token(no_journal, token);
    }
    assert(mimo26_kv_rollback_available(no_journal) == 0u);
    assert(mimo26_kv_rollback(no_journal, 1u) ==
           MIMO26_KV_ROLLBACK_UNAVAILABLE);
    mimo26_kv_destroy(no_journal);
    assert(mimo26_kv_create(&no_journal, 0u, 4u) == MIMO26_KV_INVALID_ARGUMENT);
    printf("  ok  zero-depth journal admits no rollback; zero capacity "
           "refused\n");

    printf("test_mimo26_kv: ok\n");
    return 0;
}
