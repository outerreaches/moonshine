#ifndef K3_PREFIX_CATALOG_H
#define K3_PREFIX_CATALOG_H

#include "k3_engine.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k3_prefix_catalog k3_prefix_catalog;

typedef struct {
    uint32_t format_version;
    uint32_t context;
    uint64_t model_layout_crc64;
    bool q8_projections;
} k3_prefix_catalog_identity;

/*
 * Borrowed view into one catalog entry. It remains valid until the next catalog
 * mutation or destruction. Lookup always validates the full token span; the
 * catalog never admits collision-bearing or semantic/fuzzy matches.
 */
typedef struct {
    const uint32_t *tokens;
    size_t token_count;
    const char *state_path;
    const k3_engine_state_file_info *state_info;
} k3_prefix_catalog_match;

bool k3_prefix_catalog_create(k3_prefix_catalog **out);
void k3_prefix_catalog_destroy(k3_prefix_catalog *catalog);
size_t k3_prefix_catalog_count(const k3_prefix_catalog *catalog);

/*
 * Add an immutable exact-prefix checkpoint. TOKEN_COUNT must equal the state
 * file's committed token position. An identical entry is idempotent; a second
 * path or conflicting state metadata for the same tokens and identity is
 * rejected.
 */
bool k3_prefix_catalog_add(
    k3_prefix_catalog *catalog,
    const uint32_t *tokens,
    size_t token_count,
    const char *state_path,
    const k3_engine_state_file_info *state_info);

bool k3_prefix_catalog_remove(
    k3_prefix_catalog *catalog,
    const uint32_t *tokens,
    size_t token_count,
    const k3_prefix_catalog_identity *identity);

/*
 * Return the longest identity-compatible exact prefix that leaves at least the
 * turn executor's required two-token suffix. Lookup performs no allocation.
 */
bool k3_prefix_catalog_find(
    const k3_prefix_catalog *catalog,
    const uint32_t *candidate,
    size_t candidate_count,
    const k3_prefix_catalog_identity *identity,
    k3_prefix_catalog_match *match);

#ifdef __cplusplus
}
#endif

#endif
