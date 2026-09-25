#ifndef MIMO26_KV_H
#define MIMO26_KV_H

#include "mimo26_attention.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIMO26_KV_OK = 0,
    MIMO26_KV_INVALID_ARGUMENT,
    MIMO26_KV_OUT_OF_MEMORY,
    MIMO26_KV_CAPACITY_EXCEEDED,
    MIMO26_KV_INVALID_STATE,
    MIMO26_KV_ROLLBACK_UNAVAILABLE
} mimo26_kv_status;

typedef struct mimo26_kv_cache mimo26_kv_cache;

/*
 * Transactional KV storage for all 48 text layers.
 *
 * Windowed layers keep 128 positions; full-attention layers keep up to
 * global_capacity. Values are stored exactly as mimo26_attention_split_qkv
 * produces them, which means pre-scaled by 0.707.
 *
 * Two separate guarantees, because they need different machinery:
 *
 * 1. A decode step is a transaction. Staged data is held outside the visible
 *    arrays until commit, so an abort at any point -- cancellation, a device
 *    fault, a failure between projection and residual publication -- cannot
 *    have touched committed state. Commit then applies all 48 layers with no
 *    failure path inside it.
 *
 * 2. Committed tokens can be rewound up to rollback_depth steps. This is the
 *    hard direction: once a full windowed layer shifts, the evicted position's
 *    bytes are gone, so a ring index reset cannot restore them. Every commit
 *    that evicts therefore journals what it displaced. Speculation needs this
 *    at the draft block size, which is 8 for both the indexed MTP layers and
 *    the separate DFlash draft.
 *
 * Windowed layers are kept physically ordered by shifting, so a read view is
 * always contiguous and ascending. That costs one 2,560-byte move per windowed
 * layer per token, about 100 KB per token across 39 layers, against roughly
 * 4.7 GiB of expert traffic for the same token. A modular ring is the later
 * optimization; ordered storage is the version that is easy to prove correct.
 */
mimo26_kv_status mimo26_kv_create(mimo26_kv_cache **cache,
                                  size_t global_capacity,
                                  size_t rollback_depth);
void mimo26_kv_destroy(mimo26_kv_cache *cache);

/* Drop all history, any open transaction, and the rollback journal. */
void mimo26_kv_reset(mimo26_kv_cache *cache);

/* Positions committed so far; the next token takes this position. */
uint64_t mimo26_kv_length(const mimo26_kv_cache *cache);
bool mimo26_kv_in_transaction(const mimo26_kv_cache *cache);

/* Committed steps that can currently be rewound. */
size_t mimo26_kv_rollback_available(const mimo26_kv_cache *cache);

/*
 * Open a transaction for one token. position must equal mimo26_kv_length(),
 * so a caller cannot silently skip or replay a position.
 */
mimo26_kv_status mimo26_kv_begin(mimo26_kv_cache *cache, uint64_t position);

/*
 * Stage one layer's key and value. keys is [kv_heads][192], values is
 * [kv_heads][128], both BF16. At most once per layer per transaction, and
 * invisible to mimo26_kv_view until commit.
 */
mimo26_kv_status mimo26_kv_stage(mimo26_kv_cache *cache, uint32_t layer,
                                 const uint16_t *keys, const uint16_t *values);

/* Publish every staged layer. Fails unless all 48 were staged, so a partially
 * built step cannot become visible. */
mimo26_kv_status mimo26_kv_commit(mimo26_kv_cache *cache);

/* Discard the open transaction. Committed state is untouched by construction. */
mimo26_kv_status mimo26_kv_abort(mimo26_kv_cache *cache);

/*
 * Rewind `count` committed steps, restoring evicted positions from the
 * journal. Refuses rather than approximating when the journal is too shallow,
 * so a caller can never silently continue from damaged history.
 */
mimo26_kv_status mimo26_kv_rollback(mimo26_kv_cache *cache, size_t count);

/*
 * Committed history for one layer, ascending. first_position is the absolute
 * position of element 0, nonzero once a windowed layer has evicted. Pointers
 * stay valid until the next commit, rollback or reset.
 */
mimo26_kv_status mimo26_kv_view(const mimo26_kv_cache *cache, uint32_t layer,
                                const uint16_t **keys, const uint16_t **values,
                                size_t *history, uint64_t *first_position);

/* Bytes currently allocated, for the memory ledger. */
size_t mimo26_kv_allocated_bytes(const mimo26_kv_cache *cache);

/*
 * Persisted committed history, so a prefix can be restored instead of
 * re-evaluated. Follows K3's prefix checkpoints, which is where the design
 * comes from: re-prefilling 10K tokens costs about 1,000 s at the measured
 * rate, while the same state is ~244 MiB and reloads in tens of milliseconds.
 *
 * Only semantic continuation state is written -- committed keys, values,
 * per-layer history and first_position, and the cache length. The rollback
 * journal and any open transaction are deliberately excluded: they are
 * recovery scaffolding rather than history, and restoring them would let a
 * later rewind cross a checkpoint boundary into positions the file never
 * described.
 */
typedef struct {
    uint32_t format_version;
    uint64_t global_capacity;  /* capacity the file was written at */
    uint64_t length;           /* committed positions in the file */
    uint64_t payload_bytes;
    uint64_t file_bytes;
    uint64_t payload_crc64;
    uint64_t layout_crc64;     /* per-layer kv_heads, window and capacity */
    double   wall_seconds;
} mimo26_kv_state_info;

/*
 * Write committed history to `path`, atomically: the payload goes to a
 * temporary beside it and is renamed only once fully written and synced, so a
 * reader never observes a short file. Refuses while a transaction is open.
 */
mimo26_kv_status mimo26_kv_export(const mimo26_kv_cache *cache,
                                  const char *path,
                                  mimo26_kv_state_info *info);

/*
 * Header, identity and payload CRC only. Never mutates the cache, so a caller
 * can qualify a candidate file before deciding to load it.
 */
mimo26_kv_status mimo26_kv_inspect(const mimo26_kv_cache *cache,
                                   const char *path,
                                   mimo26_kv_state_info *info);

/*
 * Replace committed history with the file's.
 *
 * Everything is validated before the cache is touched -- magic, version,
 * layout identity, exact file length and the whole payload CRC -- so a
 * rejected file leaves the cache exactly as it was. The one case that can
 * fail after mutation begins is an I/O error during the apply pass, and that
 * resets the cache to empty rather than leaving a partial history: an empty
 * cache is merely slow, a partial one answers from history it does not have.
 */
mimo26_kv_status mimo26_kv_import(mimo26_kv_cache *cache, const char *path,
                                  mimo26_kv_state_info *info);

#ifdef __cplusplus
}
#endif

#endif
