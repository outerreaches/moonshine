#ifndef K3_PREFIX_REUSE_H
#define K3_PREFIX_REUSE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Exact append-only prefix admission for session causal-state reuse.
 *
 * Reuse skips re-evaluating `retained_count` already-committed positions, so a
 * wrong answer here does not fail loudly: the engine would continue from causal
 * state that does not correspond to the supplied history and return a plausible
 * completion computed against the wrong prefix. Every rejection path must
 * therefore be conservative.
 *
 * Admission requires all of:
 *
 *   - a non-empty retained history. An empty retained buffer would make the
 *     comparison below vacuously true, so it is rejected explicitly rather
 *     than relying on callers to pre-check.
 *   - room for a real suffix. The evaluated remainder must be at least two
 *     tokens, matching the minimum the turn executor accepts.
 *   - byte-exact equality over the whole retained span. Equal length is not
 *     sufficient: an edited history of identical token count must be
 *     rejected without relying on a collision-bearing summary.
 *   - counts whose suffix or byte-span arithmetic cannot be represented by
 *     size_t are rejected before either pointer is dereferenced.
 *
 * Returns false for null buffers.
 */
bool k3_prefix_reuse_admits(
        const uint32_t *retained,
        size_t          retained_count,
        const uint32_t *candidate,
        size_t          candidate_count);

/*
 * Return the exact leading-token overlap for diagnostics. This never admits
 * reuse by itself; callers must still apply k3_prefix_reuse_admits(). Null
 * buffers have no overlap.
 */
size_t k3_prefix_reuse_common_tokens(
        const uint32_t *retained,
        size_t          retained_count,
        const uint32_t *candidate,
        size_t          candidate_count);

#ifdef __cplusplus
}
#endif

#endif
