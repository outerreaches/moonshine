#include "mimo26_kv.h"

#include "mimo26_architecture.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint16_t *keys;        /* [capacity][kv_heads][192], ascending */
    uint16_t *values;      /* [capacity][kv_heads][128], ascending */
    size_t    kv_heads;
    size_t    capacity;
    size_t    history;
    uint64_t  first_position;
    bool      windowed;

    /* Open transaction. */
    bool      staged;
    uint16_t *staged_keys;
    uint16_t *staged_values;

    /*
     * Rollback journal, one slot per rewindable commit. Only commits that
     * evicted need their displaced position stored; the rest rewind by
     * decrementing history.
     */
    uint16_t *journal_keys;    /* [depth][kv_heads][192] */
    uint16_t *journal_values;  /* [depth][kv_heads][128] */
} mimo26_kv_layer;

typedef struct {
    bool evicted; /* this commit pushed a position out of a windowed layer */
} mimo26_kv_journal_entry;

struct mimo26_kv_cache {
    mimo26_kv_layer layers[MIMO26_TEXT_LAYER_COUNT];
    uint64_t        length;
    bool            in_transaction;
    size_t          staged_count;

    size_t          depth;      /* rollback capacity in steps */
    size_t          journaled;  /* steps currently rewindable */
    size_t          journal_head; /* next slot to write, modulo depth */
    mimo26_kv_journal_entry *journal; /* [depth], per step (all layers) */

    size_t          allocated_bytes;
};

static size_t key_stride(const mimo26_kv_layer *layer)
{
    return layer->kv_heads * MIMO26_QK_HEAD_DIM;
}

static size_t value_stride(const mimo26_kv_layer *layer)
{
    return layer->kv_heads * MIMO26_V_HEAD_DIM;
}

void mimo26_kv_destroy(mimo26_kv_cache *cache)
{
    if (cache == NULL) {
        return;
    }
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        mimo26_kv_layer *layer = &cache->layers[i];
        free(layer->keys);
        free(layer->values);
        free(layer->staged_keys);
        free(layer->staged_values);
        free(layer->journal_keys);
        free(layer->journal_values);
    }
    free(cache->journal);
    free(cache);
}

mimo26_kv_status mimo26_kv_create(mimo26_kv_cache **out, size_t global_capacity,
                                  size_t rollback_depth)
{
    if (out == NULL || global_capacity == 0u) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    *out = NULL;

    mimo26_kv_cache *cache = calloc(1u, sizeof *cache);
    if (cache == NULL) {
        return MIMO26_KV_OUT_OF_MEMORY;
    }
    cache->allocated_bytes = sizeof *cache;
    cache->depth = rollback_depth;

    if (rollback_depth > 0u) {
        cache->journal = calloc(rollback_depth, sizeof *cache->journal);
        if (cache->journal == NULL) {
            mimo26_kv_destroy(cache);
            return MIMO26_KV_OUT_OF_MEMORY;
        }
        cache->allocated_bytes += rollback_depth * sizeof *cache->journal;
    }

    for (uint32_t index = 0; index < MIMO26_TEXT_LAYER_COUNT; index++) {
        mimo26_attention_config config;
        if (mimo26_attention_config_for_layer(index, &config) !=
            MIMO26_ATTENTION_OK) {
            mimo26_kv_destroy(cache);
            return MIMO26_KV_INVALID_ARGUMENT;
        }
        mimo26_kv_layer *layer = &cache->layers[index];
        layer->kv_heads = config.kv_heads;
        layer->windowed = (config.window != 0u);
        layer->capacity = layer->windowed ? MIMO26_SLIDING_WINDOW
                                          : global_capacity;

        const size_t keys_span = key_stride(layer);
        const size_t values_span = value_stride(layer);
        const size_t keys_bytes = layer->capacity * keys_span * sizeof(uint16_t);
        const size_t values_bytes =
            layer->capacity * values_span * sizeof(uint16_t);
        const size_t staged_keys_bytes = keys_span * sizeof(uint16_t);
        const size_t staged_values_bytes = values_span * sizeof(uint16_t);

        layer->keys = calloc(1u, keys_bytes);
        layer->values = calloc(1u, values_bytes);
        layer->staged_keys = calloc(1u, staged_keys_bytes);
        layer->staged_values = calloc(1u, staged_values_bytes);
        if (layer->keys == NULL || layer->values == NULL ||
            layer->staged_keys == NULL || layer->staged_values == NULL) {
            mimo26_kv_destroy(cache);
            return MIMO26_KV_OUT_OF_MEMORY;
        }
        cache->allocated_bytes +=
            keys_bytes + values_bytes + staged_keys_bytes + staged_values_bytes;

        /* Only windowed layers can evict, so only they need journal storage. */
        if (rollback_depth > 0u && layer->windowed) {
            layer->journal_keys =
                calloc(rollback_depth, staged_keys_bytes);
            layer->journal_values =
                calloc(rollback_depth, staged_values_bytes);
            if (layer->journal_keys == NULL || layer->journal_values == NULL) {
                mimo26_kv_destroy(cache);
                return MIMO26_KV_OUT_OF_MEMORY;
            }
            cache->allocated_bytes +=
                rollback_depth * (staged_keys_bytes + staged_values_bytes);
        }
    }

    *out = cache;
    return MIMO26_KV_OK;
}

void mimo26_kv_reset(mimo26_kv_cache *cache)
{
    if (cache == NULL) {
        return;
    }
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        cache->layers[i].history = 0u;
        cache->layers[i].first_position = 0u;
        cache->layers[i].staged = false;
    }
    cache->length = 0u;
    cache->in_transaction = false;
    cache->staged_count = 0u;
    cache->journaled = 0u;
    cache->journal_head = 0u;
}

uint64_t mimo26_kv_length(const mimo26_kv_cache *cache)
{
    return cache != NULL ? cache->length : 0u;
}

bool mimo26_kv_in_transaction(const mimo26_kv_cache *cache)
{
    return cache != NULL && cache->in_transaction;
}

size_t mimo26_kv_rollback_available(const mimo26_kv_cache *cache)
{
    return cache != NULL ? cache->journaled : 0u;
}

size_t mimo26_kv_allocated_bytes(const mimo26_kv_cache *cache)
{
    return cache != NULL ? cache->allocated_bytes : 0u;
}

mimo26_kv_status mimo26_kv_begin(mimo26_kv_cache *cache, uint64_t position)
{
    if (cache == NULL) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    if (cache->in_transaction) {
        return MIMO26_KV_INVALID_STATE;
    }
    if (position != cache->length) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        cache->layers[i].staged = false;
    }
    cache->in_transaction = true;
    cache->staged_count = 0u;
    return MIMO26_KV_OK;
}

mimo26_kv_status mimo26_kv_stage(mimo26_kv_cache *cache, uint32_t layer_index,
                                 const uint16_t *keys, const uint16_t *values)
{
    if (cache == NULL || keys == NULL || values == NULL ||
        layer_index >= MIMO26_TEXT_LAYER_COUNT) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    if (!cache->in_transaction) {
        return MIMO26_KV_INVALID_STATE;
    }
    mimo26_kv_layer *layer = &cache->layers[layer_index];
    if (layer->staged) {
        return MIMO26_KV_INVALID_STATE;
    }
    if (!layer->windowed && layer->history >= layer->capacity) {
        return MIMO26_KV_CAPACITY_EXCEEDED;
    }
    memcpy(layer->staged_keys, keys, key_stride(layer) * sizeof(uint16_t));
    memcpy(layer->staged_values, values,
           value_stride(layer) * sizeof(uint16_t));
    layer->staged = true;
    cache->staged_count++;
    return MIMO26_KV_OK;
}

mimo26_kv_status mimo26_kv_commit(mimo26_kv_cache *cache)
{
    if (cache == NULL) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    if (!cache->in_transaction) {
        return MIMO26_KV_INVALID_STATE;
    }
    if (cache->staged_count != MIMO26_TEXT_LAYER_COUNT) {
        return MIMO26_KV_INVALID_STATE;
    }

    const size_t slot = cache->depth > 0u
                            ? (cache->journal_head % cache->depth)
                            : 0u;
    bool evicted_any = false;

    /* No failure path below this point: the step publishes completely. */
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        mimo26_kv_layer *layer = &cache->layers[i];
        const size_t keys_span = key_stride(layer);
        const size_t values_span = value_stride(layer);

        if (layer->history < layer->capacity) {
            memcpy(layer->keys + layer->history * keys_span,
                   layer->staged_keys, keys_span * sizeof(uint16_t));
            memcpy(layer->values + layer->history * values_span,
                   layer->staged_values, values_span * sizeof(uint16_t));
            layer->history++;
        } else {
            /* Journal the position about to be evicted before it is gone. */
            if (cache->depth > 0u && layer->journal_keys != NULL) {
                memcpy(layer->journal_keys + slot * keys_span, layer->keys,
                       keys_span * sizeof(uint16_t));
                memcpy(layer->journal_values + slot * values_span,
                       layer->values, values_span * sizeof(uint16_t));
            }
            evicted_any = true;

            memmove(layer->keys, layer->keys + keys_span,
                    (layer->capacity - 1u) * keys_span * sizeof(uint16_t));
            memmove(layer->values, layer->values + values_span,
                    (layer->capacity - 1u) * values_span * sizeof(uint16_t));
            memcpy(layer->keys + (layer->capacity - 1u) * keys_span,
                   layer->staged_keys, keys_span * sizeof(uint16_t));
            memcpy(layer->values + (layer->capacity - 1u) * values_span,
                   layer->staged_values, values_span * sizeof(uint16_t));
            layer->first_position++;
        }
        layer->staged = false;
    }

    if (cache->depth > 0u) {
        cache->journal[slot].evicted = evicted_any;
        cache->journal_head = (cache->journal_head + 1u) % cache->depth;
        if (cache->journaled < cache->depth) {
            cache->journaled++;
        }
    }

    cache->length++;
    cache->in_transaction = false;
    cache->staged_count = 0u;
    return MIMO26_KV_OK;
}

mimo26_kv_status mimo26_kv_abort(mimo26_kv_cache *cache)
{
    if (cache == NULL) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    if (!cache->in_transaction) {
        return MIMO26_KV_INVALID_STATE;
    }
    /*
     * Staged bytes never entered the visible arrays, so committed history is
     * already intact and there is nothing to undo. That is the point of
     * staging: an abort cannot corrupt a windowed layer even mid-step.
     */
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        cache->layers[i].staged = false;
    }
    cache->in_transaction = false;
    cache->staged_count = 0u;
    return MIMO26_KV_OK;
}

mimo26_kv_status mimo26_kv_rollback(mimo26_kv_cache *cache, size_t count)
{
    if (cache == NULL) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    if (cache->in_transaction) {
        return MIMO26_KV_INVALID_STATE;
    }
    if (count == 0u) {
        return MIMO26_KV_OK;
    }
    /* Refuse rather than approximate: a shallower journal cannot restore the
     * evicted bytes, and continuing from damaged history is worse than an
     * error the caller can handle. */
    if (count > cache->journaled || count > (size_t)cache->length) {
        return MIMO26_KV_ROLLBACK_UNAVAILABLE;
    }

    for (size_t step = 0; step < count; step++) {
        const size_t slot =
            (cache->journal_head + cache->depth - 1u) % cache->depth;
        const bool evicted = cache->journal[slot].evicted;

        for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
            mimo26_kv_layer *layer = &cache->layers[i];
            const size_t keys_span = key_stride(layer);
            const size_t values_span = value_stride(layer);

            if (!evicted || !layer->windowed || layer->journal_keys == NULL) {
                /* Nothing was pushed out, so dropping the last slot suffices. */
                if (layer->history > 0u) {
                    layer->history--;
                }
                continue;
            }
            /* Shift back up and restore the journaled position at the front. */
            memmove(layer->keys + keys_span, layer->keys,
                    (layer->capacity - 1u) * keys_span * sizeof(uint16_t));
            memmove(layer->values + values_span, layer->values,
                    (layer->capacity - 1u) * values_span * sizeof(uint16_t));
            memcpy(layer->keys, layer->journal_keys + slot * keys_span,
                   keys_span * sizeof(uint16_t));
            memcpy(layer->values, layer->journal_values + slot * values_span,
                   values_span * sizeof(uint16_t));
            layer->first_position--;
        }

        cache->journal_head = slot;
        cache->journaled--;
        cache->length--;
    }
    return MIMO26_KV_OK;
}

mimo26_kv_status mimo26_kv_view(const mimo26_kv_cache *cache,
                                uint32_t layer_index, const uint16_t **keys,
                                const uint16_t **values, size_t *history,
                                uint64_t *first_position)
{
    if (cache == NULL || layer_index >= MIMO26_TEXT_LAYER_COUNT) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    const mimo26_kv_layer *layer = &cache->layers[layer_index];
    if (keys != NULL) {
        *keys = layer->keys;
    }
    if (values != NULL) {
        *values = layer->values;
    }
    if (history != NULL) {
        *history = layer->history;
    }
    if (first_position != NULL) {
        *first_position = layer->first_position;
    }
    return MIMO26_KV_OK;
}
