#ifndef GLM53_KDA_OPS_H
#define GLM53_KDA_OPS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_KDA_HEAD_DIM 128u

/*
 * Allocation-free, asynchronous one-token GLM-5.3 KDA recurrence.
 *
 * q, k, v, and g are contiguous F32 [heads,128], beta is F32 [heads],
 * output is F32 [heads,128], and both states are F32
 * [heads,128 key,128 value].  Counts are elements.  destination_state must
 * be separate from source_state; source_state is never modified.  Every
 * complete accessible span described by a pointer/count pair must be
 * pairwise non-overlapping.  The call only enqueues work on the supplied
 * hipStream_t (passed as void *) and reports validation/immediate launch
 * status.
 */
bool glm53_kda_recurrent_f32(
    void *output, size_t output_count,
    void *destination_state, size_t destination_state_count,
    const void *q, size_t q_count,
    const void *k, size_t k_count,
    const void *v, size_t v_count,
    const void *g, size_t g_count,
    const void *beta, size_t beta_count,
    const void *source_state, size_t source_state_count,
    size_t heads, void *stream);

#ifdef __cplusplus
}
#endif
#endif
