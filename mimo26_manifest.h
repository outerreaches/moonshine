#ifndef MIMO26_MANIFEST_H
#define MIMO26_MANIFEST_H

#include "k3_safetensors.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIMO26_SHARD_COUNT 65u
#define MIMO26_INDEX_TENSOR_COUNT 73081u

/* index metadata.total_size is the payload sum, ~8.7 MiB below the sum of
 * file sizes: the difference is the per-shard SafeTensors headers. */
#define MIMO26_INDEX_TOTAL_BYTES UINT64_C(172923364096)
#define MIMO26_INDEX_SAVE_FORMAT "mxfp4"
#define MIMO26_INDEX_TP_SIZE 4u

#define MIMO26_EXPERT_PROJECTION_COUNT 3u

typedef struct {
    char    *name;
    uint16_t shard; /* index into mimo26_manifest.shard_names */
} mimo26_index_entry;

typedef struct {
    char              **shard_names; /* deterministic sorted order */
    size_t              shard_count;
    mimo26_index_entry *entries;     /* sorted by name */
    size_t              entry_count;
    uint64_t            total_size;
    uint32_t            tp_size;
    char               *save_format;
} mimo26_manifest;

/*
 * Parse and strictly validate model.safetensors.index.json. Shard names are
 * rejected unless they are plain, non-empty, traversal-free basenames ending
 * in .safetensors; this is the boundary that keeps an index from naming a
 * file outside the checkpoint directory.
 */
bool mimo26_manifest_parse(mimo26_manifest *manifest,
                           const char      *index_json,
                           size_t           index_size,
                           char            *error,
                           size_t           error_size);

bool mimo26_manifest_load(mimo26_manifest *manifest,
                          const char      *root,
                          char            *error,
                          size_t           error_size);

void mimo26_manifest_free(mimo26_manifest *manifest);

/* Shard index for a tensor name, or -1 when the index does not list it. */
int32_t mimo26_manifest_shard_of(const mimo26_manifest *manifest,
                                 const char            *tensor_name);

/*
 * Open every shard the index names, in manifest order, so shard indices in
 * the resulting directory match mimo26_manifest.shard_names. MiMo's shard
 * filenames do not follow the numbered model-NNNNN-of-NNNNN family, so this
 * is the only supported way to open the checkpoint.
 */
bool mimo26_manifest_open_model(const mimo26_manifest *manifest,
                                const char            *root,
                                k3_st_model           *model,
                                char                  *error,
                                size_t                 error_size);

/*
 * Reconcile parsed index metadata against payload-free SafeTensors headers:
 * one-to-one name coverage, index-to-shard agreement, and a payload byte sum
 * matching metadata.total_size.
 */
bool mimo26_manifest_reconcile(const mimo26_manifest *manifest,
                               const k3_st_model     *model,
                               char                  *error,
                               size_t                 error_size);

/* One routed expert projection: packed weight plus its E8M0 scale block. */
typedef struct {
    uint16_t shard;
    uint64_t weight_offset;
    uint64_t weight_bytes;
    uint64_t scale_offset;
    uint64_t scale_bytes;
} mimo26_expert_span;

/*
 * Resolve the read spans for one routed expert. Projections are returned in
 * gate, up, down order. Weight and scale are cached as one expert identity,
 * so both spans are resolved together or not at all.
 */
bool mimo26_expert_spans(const k3_st_model  *model,
                         uint32_t            layer,
                         uint32_t            expert,
                         mimo26_expert_span  spans[MIMO26_EXPERT_PROJECTION_COUNT],
                         char               *error,
                         size_t              error_size);

/* Total bytes a resident expert occupies, weights plus scales: 12.75 MiB. */
uint64_t mimo26_expert_bytes(void);

#ifdef __cplusplus
}
#endif

#endif
