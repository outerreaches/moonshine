#ifndef GLM53_KDA_AUX_OPS_H
#define GLM53_KDA_AUX_OPS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_KDA_AUX_CHANNELS 8192u
#define GLM53_KDA_AUX_HEADS 64u
#define GLM53_KDA_AUX_HEAD_DIM 128u
#define GLM53_KDA_AUX_CONV_WIDTH 4u

/* Allocation-free asynchronous KDA auxiliary primitives. All pointers name
 * contiguous device BF16 storage. Counts are elements. stream is a
 * hipStream_t passed as void *. Each call requires every named buffer to be
 * pairwise non-overlapping and only enqueues work on the caller's stream. */

/* Transactional one-token causal depthwise convolution. dst_cache and
 * src_cache are distinct [channels,4] buffers and weight is checkpoint layout
 * [channels,1,4]. For each channel dst_cache becomes
 * {src_cache[1],src_cache[2],src_cache[3],input}. The four products accumulate
 * in F32 in increasing tap order. The convolution rounds to BF16 before SiLU
 * is evaluated in F32 and output is rounded to BF16. src_cache is read-only. */
bool glm53_kda_conv4_silu_bf16(
    void *output, size_t output_count,
    void *dst_cache, size_t dst_cache_count,
    const void *input, size_t input_count,
    const void *src_cache, size_t src_cache_count,
    const void *weight, size_t weight_count,
    size_t channels, void *stream);

/* Per-head RMSNormGated. input, gate, and output are [heads,128]; weight is
 * shared [128]. Mean square and arithmetic are F32. The gate uses a stable
 * sigmoid and the final result rounds once to BF16. Production uses eps=1e-5. */
bool glm53_kda_rmsnorm_gated_bf16(
    void *output, size_t output_count,
    const void *input, size_t input_count,
    const void *gate, size_t gate_count,
    const void *weight, size_t weight_count,
    size_t heads, float eps, void *stream);

#ifdef __cplusplus
}
#endif
#endif
