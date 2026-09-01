#ifndef GLM53_MHC_OPS_H
#define GLM53_MHC_OPS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_MHC_OPS_HIDDEN 4096u
#define GLM53_MHC_OPS_STREAMS 4u
#define GLM53_MHC_OPS_MIX 24u

/* Allocation-free asynchronous mHC primitives. All buffers are contiguous
 * device buffers. Counts are elements, not bytes. width may be smaller than
 * 4096 for tests; production callers must pass 4096. stream is a hipStream_t
 * passed as void *. Calls only enqueue work on that stream. Every buffer in a
 * call must be disjoint. BF16/F32 inputs are required to be finite; signed
 * finite values, including negative coefficients and data, are accepted. */

/* hidden[width] -> streams[4,width], all BF16. */
bool glm53_mhc_replicate_bf16(
    void *streams, size_t streams_count,
    const void *hidden, size_t hidden_count, size_t width, void *stream);

/* Prepare one mHC layer. fn is row-major BF16 [24,4*width]. streams is BF16
 * [4,width]. mix is F32[24]. pre is F32[4]. post and comb are BF16-rounded
 * [4] and [4,4]. collapsed is BF16[width]. The RMS epsilon is exactly 1e-5
 * and the Sinkhorn epsilon is exactly 1e-6. Collapse uses unrounded pre. */
bool glm53_mhc_prepare_bf16(
    void *mix, size_t mix_count,
    void *pre, size_t pre_count,
    void *post, size_t post_count,
    void *comb, size_t comb_count,
    void *collapsed, size_t collapsed_count,
    const void *streams, size_t streams_count,
    const void *fn, size_t fn_count,
    const void *base, size_t base_count,
    const void *scale, size_t scale_count,
    size_t width, void *stream);

/* out[dst,d] = post[dst]*branch[d] +
 * sum_src comb[src,dst]*residual[src,d].  To match the pinned BF16 graph,
 * the post product and four-term matmul result each round to BF16 before the
 * final BF16 add. The source-to-destination transpose is intentional. */
bool glm53_mhc_expand_bf16(
    void *out, size_t out_count,
    const void *branch, size_t branch_count,
    const void *residual, size_t residual_count,
    const void *post, size_t post_count,
    const void *comb, size_t comb_count,
    size_t width, void *stream);

/* Unweighted stream mean: out[d] = (sum_src streams[src,d]) / 4, BF16. */
bool glm53_mhc_hyper_mean_bf16(
    void *out, size_t out_count,
    const void *streams, size_t streams_count,
    size_t width, void *stream);

#ifdef __cplusplus
}
#endif
#endif
