#ifndef GLM53_STATE_ORACLE_H
#define GLM53_STATE_ORACLE_H

/* Small, allocation-free CPU reference helpers for GLM-5.3-Flash state/layout
 * semantics.  The equations and layouts are pinned to transformers eb4d9e2
 * and llama.cpp a771613a. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of float elements required in scratch.  Zero means that the shape is
 * invalid or that the size calculation overflowed. */
size_t glm53_kda_scratch_floats(size_t key_dim, size_t value_dim,
                                size_t steps);

/* One KDA head.  Chunk arrays are time-major. beta has one scalar per step.
 * State is row-major [key_dim, value_dim].  No destination is changed on
 * failure, including arithmetic failure.  Scratch is caller-owned and may not
 * alias any input or output. */
bool glm53_kda_step_f32(const float *q, const float *k, const float *v,
                        const float *g, float beta,
                        float *state, size_t key_dim, size_t value_dim,
                        float *output, float *scratch,
                        size_t scratch_floats);
bool glm53_kda_chunk_f32(const float *q, const float *k, const float *v,
                         const float *g, const float *beta, size_t steps,
                         float *state, size_t key_dim, size_t value_dim,
                         float *output, float *scratch,
                         size_t scratch_floats);

/* Checkpoint-native combined no-PE MLA projection layout:
 * [head, key_dim + value_dim, rank]. */
bool glm53_mla_kv_b_key_index(size_t heads, size_t key_dim,
                              size_t value_dim, size_t rank,
                              size_t head, size_t key_channel,
                              size_t rank_channel, size_t *flat_index);
bool glm53_mla_kv_b_value_index(size_t heads, size_t key_dim,
                                size_t value_dim, size_t rank,
                                size_t head, size_t value_channel,
                                size_t rank_channel, size_t *flat_index);

enum { GLM53_INDEX_POOL_SIZE = 4 };
typedef struct glm53_index_pool4 {
    size_t start;       /* inclusive raw token index */
    size_t end;         /* inclusive raw token index; visibility is tested here */
    bool end_visible;   /* valid[end] && end <= visible_through */
} glm53_index_pool4;

typedef struct glm53_index_pool4_metadata {
    size_t first_valid; /* length when there is no valid token */
    size_t pool_count;  /* complete, all-valid pools only */
    size_t tail_count;  /* visible raw indices in the incomplete pool: 0..3 */
} glm53_index_pool4_metadata;

/* Build pool metadata for one sequence. Pools start at first_valid. Only
 * complete/all-valid pools are emitted. The visible incomplete tail is
 * appended as raw indices. visible_through is an inclusive causal position;
 * SIZE_MAX means no token is visible. Caller arrays may be NULL iff their
 * required count is zero. The function fails before writing. */
bool glm53_index_pool4_build(const uint8_t *valid, size_t length,
                             size_t visible_through,
                             glm53_index_pool4 *pools, size_t pool_capacity,
                             size_t *tail_indices, size_t tail_capacity,
                             glm53_index_pool4_metadata *metadata);

#ifdef __cplusplus
}
#endif
#endif
