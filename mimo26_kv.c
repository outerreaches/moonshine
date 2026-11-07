#include "mimo26_kv.h"

#include "mimo26_architecture.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

/* ---- persisted prefix state ---- */

/*
 * CRC-64/ECMA, the same polynomial the static loader uses, so checkpoint
 * integrity is checked the way the rest of this tree checks it.
 */
static uint64_t kv_crc64_table[256];
static bool kv_crc64_ready = false;

static void kv_crc64_init(void)
{
    const uint64_t polynomial = UINT64_C(0x42f0e1eba9ea3693);
    for (unsigned byte = 0u; byte < 256u; ++byte) {
        uint64_t crc = (uint64_t)byte << 56u;
        for (unsigned bit = 0u; bit < 8u; ++bit) {
            crc = (crc & (UINT64_C(1) << 63u)) != 0u
                      ? (crc << 1u) ^ polynomial
                      : crc << 1u;
        }
        kv_crc64_table[byte] = crc;
    }
    kv_crc64_ready = true;
}

/*
 * Table-driven, a byte at a time. The bitwise form this replaces ran at about
 * 181 MB/s, which made the CRC rather than the SSD the cost of a checkpoint --
 * 1.4 s for a 244 MiB state the drive moves in a fraction of that. Same
 * polynomial and byte order, so files stay compatible.
 */
static uint64_t kv_crc64(uint64_t crc, const void *source, size_t bytes)
{
    if (!kv_crc64_ready) {
        kv_crc64_init();
    }
    const unsigned char *data = (const unsigned char *)source;
    for (size_t i = 0u; i < bytes; ++i) {
        crc = (crc << 8u) ^
              kv_crc64_table[((crc >> 56u) ^ (uint64_t)data[i]) & 0xffu];
    }
    return crc;
}

static void put_u64(unsigned char *out, uint64_t value)
{
    for (unsigned i = 0u; i < 8u; ++i) {
        out[i] = (unsigned char)(value >> (8u * i));
    }
}

static uint64_t get_u64(const unsigned char *in)
{
    uint64_t value = 0u;
    for (unsigned i = 0u; i < 8u; ++i) {
        value |= (uint64_t)in[i] << (8u * i);
    }
    return value;
}

#define KV_STATE_MAGIC "MIMO26KV"
#define KV_STATE_VERSION 1u
/* magic 8, version 8, capacity 8, length 8, layers 8, layout crc 8,
 * payload bytes 8, payload crc 8 */
#define KV_STATE_HEADER_BYTES 64u

/*
 * Identity of the geometry, not of the contents: layer count, and each
 * layer's head count, window flag and capacity. A file written under a
 * different context or a different architecture must not be loadable, because
 * the payload would be read with the wrong strides.
 */
static uint64_t kv_layout_crc(const mimo26_kv_cache *cache)
{
    uint64_t crc = 0u;
    unsigned char field[8];
    put_u64(field, (uint64_t)MIMO26_TEXT_LAYER_COUNT);
    crc = kv_crc64(crc, field, sizeof field);
    put_u64(field, (uint64_t)MIMO26_QK_HEAD_DIM);
    crc = kv_crc64(crc, field, sizeof field);
    put_u64(field, (uint64_t)MIMO26_V_HEAD_DIM);
    crc = kv_crc64(crc, field, sizeof field);
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        const mimo26_kv_layer *layer = &cache->layers[i];
        put_u64(field, (uint64_t)layer->kv_heads);
        crc = kv_crc64(crc, field, sizeof field);
        put_u64(field, (uint64_t)layer->capacity);
        crc = kv_crc64(crc, field, sizeof field);
        put_u64(field, layer->windowed ? 1u : 0u);
        crc = kv_crc64(crc, field, sizeof field);
    }
    return crc;
}

/* Payload: per layer, history and first_position, then the committed rows. */
static uint64_t kv_payload_bytes(const mimo26_kv_cache *cache)
{
    uint64_t total = 0u;
    for (size_t i = 0; i < MIMO26_TEXT_LAYER_COUNT; i++) {
        const mimo26_kv_layer *layer = &cache->layers[i];
        total += 16u;                    /* history, first_position */
        total += (uint64_t)layer->history * key_stride(layer) *
                 sizeof(uint16_t);
        total += (uint64_t)layer->history * value_stride(layer) *
                 sizeof(uint16_t);
    }
    return total;
}

static double kv_now_seconds(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

uint64_t mimo26_kv_layout_crc64(const mimo26_kv_cache *cache)
{
    return cache == NULL ? 0u : kv_layout_crc(cache);
}

mimo26_kv_status mimo26_kv_export(const mimo26_kv_cache *cache,
                                  const char *path,
                                  mimo26_kv_state_info *info)
{
    if (cache == NULL || path == NULL) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    /* A staged-but-uncommitted step is not history and has no position yet. */
    if (cache->in_transaction) {
        return MIMO26_KV_INVALID_STATE;
    }
    const double started = kv_now_seconds();

    char temporary[4096];
    const int written = snprintf(temporary, sizeof temporary, "%s.partial",
                                 path);
    if (written <= 0 || (size_t)written >= sizeof temporary) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    /*
     * Created 0600 explicitly rather than through the umask. A checkpoint is
     * verbatim conversation state, so it should not be world-readable, and the
     * store that indexes these files refuses anything that is not a private
     * regular file.
     */
    const int descriptor =
        open(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        return MIMO26_KV_INVALID_STATE;
    }
    FILE *file = fdopen(descriptor, "wb");
    if (file == NULL) {
        (void)close(descriptor);
        (void)remove(temporary);
        return MIMO26_KV_INVALID_STATE;
    }

    const uint64_t payload_bytes = kv_payload_bytes(cache);
    const uint64_t layout = kv_layout_crc(cache);

    /* The payload CRC is accumulated while writing, so the header cannot be
     * written first. Reserve it and seek back. */
    unsigned char header[KV_STATE_HEADER_BYTES];
    memset(header, 0, sizeof header);
    bool ok = fwrite(header, 1, sizeof header, file) == sizeof header;

    uint64_t crc = 0u;
    unsigned char field[16];
    for (size_t i = 0; ok && i < MIMO26_TEXT_LAYER_COUNT; i++) {
        const mimo26_kv_layer *layer = &cache->layers[i];
        put_u64(field, (uint64_t)layer->history);
        put_u64(field + 8, layer->first_position);
        crc = kv_crc64(crc, field, sizeof field);
        ok = fwrite(field, 1, sizeof field, file) == sizeof field;
        const size_t keys = layer->history * key_stride(layer) *
                            sizeof(uint16_t);
        const size_t values = layer->history * value_stride(layer) *
                              sizeof(uint16_t);
        if (ok && keys > 0u) {
            crc = kv_crc64(crc, layer->keys, keys);
            ok = fwrite(layer->keys, 1, keys, file) == keys;
        }
        if (ok && values > 0u) {
            crc = kv_crc64(crc, layer->values, values);
            ok = fwrite(layer->values, 1, values, file) == values;
        }
    }

    if (ok) {
        memcpy(header, KV_STATE_MAGIC, 8u);
        put_u64(header + 8u, (uint64_t)KV_STATE_VERSION);
        put_u64(header + 16u, (uint64_t)cache->layers[0].capacity);
        put_u64(header + 24u, cache->length);
        put_u64(header + 32u, (uint64_t)MIMO26_TEXT_LAYER_COUNT);
        put_u64(header + 40u, layout);
        put_u64(header + 48u, payload_bytes);
        put_u64(header + 56u, crc);
        ok = fseek(file, 0, SEEK_SET) == 0 &&
             fwrite(header, 1, sizeof header, file) == sizeof header;
    }
    /* Durable before the rename, or a crash could publish a name pointing at
     * an incomplete payload. */
    if (ok) {
        ok = fflush(file) == 0 && fsync(fileno(file)) == 0;
    }
    if (fclose(file) != 0) {
        ok = false;
    }
    if (!ok || rename(temporary, path) != 0) {
        (void)remove(temporary);
        return MIMO26_KV_INVALID_STATE;
    }
    if (info != NULL) {
        memset(info, 0, sizeof *info);
        info->format_version = KV_STATE_VERSION;
        info->global_capacity = (uint64_t)cache->layers[0].capacity;
        info->length = cache->length;
        info->payload_bytes = payload_bytes;
        info->file_bytes = payload_bytes + KV_STATE_HEADER_BYTES;
        info->payload_crc64 = crc;
        info->layout_crc64 = layout;
        info->wall_seconds = kv_now_seconds() - started;
    }
    return MIMO26_KV_OK;
}

/*
 * Shared by inspect and import. Validates the header against this cache's
 * geometry, then streams the payload to confirm the CRC and the exact length,
 * without writing anything into the cache.
 */
static mimo26_kv_status kv_validate_file(const mimo26_kv_cache *cache,
                                         const char *path, FILE **out_file,
                                         mimo26_kv_state_info *info)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return MIMO26_KV_INVALID_STATE;
    }
    unsigned char header[KV_STATE_HEADER_BYTES];
    if (fread(header, 1, sizeof header, file) != sizeof header ||
        memcmp(header, KV_STATE_MAGIC, 8u) != 0 ||
        get_u64(header + 8u) != (uint64_t)KV_STATE_VERSION ||
        get_u64(header + 32u) != (uint64_t)MIMO26_TEXT_LAYER_COUNT ||
        get_u64(header + 40u) != kv_layout_crc(cache)) {
        fclose(file);
        return MIMO26_KV_INVALID_STATE;
    }
    const uint64_t capacity = get_u64(header + 16u);
    const uint64_t length = get_u64(header + 24u);
    const uint64_t payload_bytes = get_u64(header + 48u);
    const uint64_t payload_crc = get_u64(header + 56u);
    if (capacity != (uint64_t)cache->layers[0].capacity ||
        length > capacity) {
        fclose(file);
        return MIMO26_KV_INVALID_STATE;
    }

    /* Stream the payload once for the CRC, and require the file to end
     * exactly where the header says it should. */
    unsigned char buffer[1u << 16];
    uint64_t seen = 0u;
    uint64_t crc = 0u;
    while (seen < payload_bytes) {
        uint64_t want = payload_bytes - seen;
        if (want > sizeof buffer) {
            want = sizeof buffer;
        }
        const size_t got = fread(buffer, 1, (size_t)want, file);
        if (got == 0u) {
            fclose(file);
            return MIMO26_KV_INVALID_STATE;
        }
        crc = kv_crc64(crc, buffer, got);
        seen += got;
    }
    if (crc != payload_crc || fgetc(file) != EOF) {
        fclose(file);
        return MIMO26_KV_INVALID_STATE;
    }
    if (info != NULL) {
        memset(info, 0, sizeof *info);
        info->format_version = KV_STATE_VERSION;
        info->global_capacity = capacity;
        info->length = length;
        info->payload_bytes = payload_bytes;
        info->file_bytes = payload_bytes + KV_STATE_HEADER_BYTES;
        info->payload_crc64 = payload_crc;
        info->layout_crc64 = get_u64(header + 40u);
    }
    if (out_file != NULL) {
        *out_file = file;
    } else {
        fclose(file);
    }
    return MIMO26_KV_OK;
}

mimo26_kv_status mimo26_kv_inspect(const mimo26_kv_cache *cache,
                                   const char *path,
                                   mimo26_kv_state_info *info)
{
    if (cache == NULL || path == NULL) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    return kv_validate_file(cache, path, NULL, info);
}

mimo26_kv_status mimo26_kv_import(mimo26_kv_cache *cache, const char *path,
                                  mimo26_kv_state_info *info)
{
    if (cache == NULL || path == NULL) {
        return MIMO26_KV_INVALID_ARGUMENT;
    }
    if (cache->in_transaction) {
        return MIMO26_KV_INVALID_STATE;
    }
    const double started = kv_now_seconds();
    mimo26_kv_state_info local;
    FILE *file = NULL;
    const mimo26_kv_status valid =
        kv_validate_file(cache, path, &file, &local);
    if (valid != MIMO26_KV_OK) {
        return valid;      /* nothing was touched */
    }
    if (fseek(file, (long)KV_STATE_HEADER_BYTES, SEEK_SET) != 0) {
        fclose(file);
        return MIMO26_KV_INVALID_STATE;
    }

    /*
     * From here the cache is being rewritten. The file has already been
     * validated end to end, so only an I/O error can intervene; that leaves
     * the cache reset rather than half-loaded.
     */
    mimo26_kv_reset(cache);
    bool ok = true;
    unsigned char field[16];
    for (size_t i = 0; ok && i < MIMO26_TEXT_LAYER_COUNT; i++) {
        mimo26_kv_layer *layer = &cache->layers[i];
        if (fread(field, 1, sizeof field, file) != sizeof field) {
            ok = false;
            break;
        }
        const uint64_t history = get_u64(field);
        const uint64_t first_position = get_u64(field + 8);
        if (history > (uint64_t)layer->capacity) {
            ok = false;
            break;
        }
        const size_t keys = (size_t)history * key_stride(layer) *
                            sizeof(uint16_t);
        const size_t values = (size_t)history * value_stride(layer) *
                              sizeof(uint16_t);
        if ((keys > 0u && fread(layer->keys, 1, keys, file) != keys) ||
            (values > 0u &&
             fread(layer->values, 1, values, file) != values)) {
            ok = false;
            break;
        }
        layer->history = (size_t)history;
        layer->first_position = first_position;
    }
    fclose(file);
    if (!ok) {
        mimo26_kv_reset(cache);
        return MIMO26_KV_INVALID_STATE;
    }
    cache->length = local.length;
    if (info != NULL) {
        local.wall_seconds = kv_now_seconds() - started;
        *info = local;
    }
    return MIMO26_KV_OK;
}
