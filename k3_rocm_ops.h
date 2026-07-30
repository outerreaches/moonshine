#ifndef K3_ROCM_OPS_H
#define K3_ROCM_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k3_rocm_blas_context k3_rocm_blas_context;

/*
 * All tensor pointers are device pointers to contiguous BF16 storage. The
 * stream argument is a hipStream_t passed as void *; NULL selects the default
 * stream. Launch errors are returned synchronously, while execution remains
 * asynchronous with respect to the host.
 */
bool k3_rocm_situ_bf16(void       *output,
                       const void *gate,
                       const void *up,
                       uint64_t    element_count,
                       float       beta,
                       float       linear_beta,
                       void       *stream);

bool k3_rocm_attn_res_bf16(void       *output,
                           const void *prefix,
                           const void *blocks,
                           const void *norm_weight,
                           const void *qk_weight,
                           uint32_t    token_count,
                           uint32_t    block_capacity,
                           uint32_t    num_blocks,
                           uint32_t    hidden_size,
                           float       epsilon,
                           void       *stream);

/*
 * Matrix-vector product using the checkpoint's native MXFP4 representation:
 * packed has shape [rows, columns / 2] with adjacent low/high nibbles and
 * scales has shape [rows, columns / 32] in OCP E8M0. No repack is required.
 */
bool k3_rocm_mxfp4_gemv_bf16(void       *output,
                             const void *packed,
                             const void *scales,
                             const void *input,
                             uint32_t    rows,
                             uint32_t    columns,
                             void       *stream);

/*
 * Token-row batch variants preserve the GEMV reduction order independently
 * for every row. Inputs are [vector_count, columns] and outputs are
 * [vector_count, rows]; weights are shared.
 */
bool k3_rocm_mxfp4_gemm_bf16(void       *output,
                             const void *packed,
                             const void *scales,
                             const void *input,
                             uint32_t    vector_count,
                             uint32_t    rows,
                             uint32_t    columns,
                             void       *stream);

/*
 * Exact-shape tuning entry points. These retain the production reduction
 * order while selecting a 16-, 32-, or 64-vector weight-reuse tile.
 */
bool k3_rocm_mxfp4_gemm_tiled_bf16(
        void       *output,
        const void *packed,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        uint32_t    batch_tile,
        void       *stream);

bool k3_rocm_bf16_gemv_bf16(void       *output,
                             const void *weights,
                             const void *input,
                             uint32_t    rows,
                             uint32_t    columns,
                             void       *stream);

bool k3_rocm_bf16_gemm_tiled_bf16(
        void       *output,
        const void *weights,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        uint32_t    batch_tile,
        void       *stream);

bool k3_rocm_bf16_gemm_bf16(void       *output,
                             const void *weights,
                             const void *input,
                             uint32_t    vector_count,
                             uint32_t    rows,
                             uint32_t    columns,
                             void       *stream);

bool k3_rocm_bf16_gemv_f32(void       *output,
                            const void *weights,
                            const void *input,
                            uint32_t    rows,
                            uint32_t    columns,
                            void       *stream);

bool k3_rocm_bf16_gemm_f32(void       *output,
                            const void *weights,
                            const void *input,
                            uint32_t    vector_count,
                            uint32_t    rows,
                            uint32_t    columns,
                            void       *stream);

bool k3_rocm_bf16_gemm_tiled_f32(
        void       *output,
        const void *weights,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        uint32_t    batch_tile,
        void       *stream);

/*
 * Optional static-projection compression. Weights are quantized independently
 * in symmetric 128-element blocks to signed int8 plus one F32 scale. Columns
 * must be divisible by 128. The BF16 reference path remains available for
 * correctness and quality A/B tests.
 */
bool k3_rocm_bf16_quantize_q8_128(
        void       *quantized,
        void       *scales,
        const void *weights,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream);

bool k3_rocm_q8_128_dequantize_bf16(
        void       *weights,
        const void *quantized,
        const void *scales,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream);

bool k3_rocm_q8_128_gemv_bf16(
        void       *output,
        const void *quantized,
        const void *scales,
        const void *input,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream);

bool k3_rocm_q8_128_gemm_bf16(
        void       *output,
        const void *quantized,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        void       *stream);

bool k3_rocm_q8_128_gemm_tiled_bf16(
        void       *output,
        const void *quantized,
        const void *scales,
        const void *input,
        uint32_t    vector_count,
        uint32_t    rows,
        uint32_t    columns,
        uint32_t    batch_tile,
        void       *stream);

/*
 * rocBLAS-backed BF16 matrix paths for large resident projections.
 * The explicit context owns its hipBLAS handle and is safe to keep for the
 * lifetime of one engine instance.
 */
bool k3_rocm_blas_context_create(k3_rocm_blas_context **context);
void k3_rocm_blas_context_destroy(k3_rocm_blas_context *context);

bool k3_rocm_blas_bf16_gemv_bf16(
        k3_rocm_blas_context *context,
        void                 *output,
        const void           *weights,
        const void           *input,
        uint32_t              rows,
        uint32_t              columns,
        void                 *stream);

bool k3_rocm_blas_bf16_gemm_bf16(
        k3_rocm_blas_context *context,
        void                 *output,
        const void           *weights,
        const void           *input,
        uint32_t              vector_count,
        uint32_t              rows,
        uint32_t              columns,
        void                 *stream);

bool k3_rocm_blas_bf16_gemm_f32(
        k3_rocm_blas_context *context,
        void                 *output,
        const void           *weights,
        const void           *input,
        uint32_t              vector_count,
        uint32_t              rows,
        uint32_t              columns,
        void                 *stream);

/*
 * K3 MLA decode helpers. kv_b is the checkpoint-native BF16
 * [head_count * 256, 512] matrix, where each head stores 128 key rows then
 * 128 value rows. packed_k is a one-time reordered [head_count, 512, 128]
 * matrix used to absorb the no-PE query. The absorbed query is
 * [head_count, 576] (512 latent + 64 pass-through no-PE dimensions).
 */
bool k3_rocm_mla_pack_k_weight_bf16(
        void       *packed_k,
        const void *kv_b,
        uint32_t    head_count,
        void       *stream);

bool k3_rocm_mla_absorb_q_bf16(
        void       *absorbed_q,
        const void *q,
        const void *packed_k,
        uint32_t    head_count,
        void       *stream);

bool k3_rocm_mla_decompress_v_bf16(
        void       *output,
        const void *latent,
        const void *kv_b,
        uint32_t    head_count,
        void       *stream);

/*
 * Compressed-cache MLA attention for one decode token. cache is BF16
 * [token_count, 576], absorbed_q is BF16 [head_count, 576], and output is
 * BF16 [head_count, 512]. The caller supplies F32 [head_count, token_count]
 * score and BF16 probability workspaces. Softmax remains F32 and is cast to
 * BF16 before the value GEMM, matching the official eager path.
 */
bool k3_rocm_blas_mla_attention_bf16(
        k3_rocm_blas_context *context,
        void                 *output,
        void                 *score_workspace,
        void                 *probability_workspace,
        const void           *absorbed_q,
        const void           *cache,
        uint32_t              head_count,
        uint32_t              token_count,
        void                 *stream);

bool k3_rocm_rms_norm_bf16(void       *output,
                            const void *input,
                            const void *weight,
                            uint32_t    vector_count,
                            uint32_t    hidden_size,
                            float       epsilon,
                            void       *stream);

/*
 * One-token Kimi Delta Attention recurrence. q/k/v/raw_gate are BF16
 * [head_count, head_dim], raw_beta is BF16 [head_count], A_log is F32
 * [head_count], and dt_bias is F32 [head_count, head_dim].
 *
 * state is an in-place F32 [head_count, head_dim, head_dim] tensor in the
 * checkpoint runtime's V-first [head, value, key] layout. output is BF16
 * [head_count, head_dim]. The kernel fuses q/k L2 normalization (epsilon
 * 1e-6), 1/sqrt(head_dim) query scaling, the lower-bound decay activation,
 * beta sigmoid, state update, and output projection. K3 currently requires
 * head_dim=128 and lower_bound=-5.
 */
bool k3_rocm_kda_recurrent_bf16_f32_state(
        void       *output,
        void       *state,
        const void *q,
        const void *k,
        const void *v,
        const void *raw_gate,
        const void *raw_beta,
        const void *A_log,
        const void *dt_bias,
        uint32_t    head_count,
        uint32_t    head_dim,
        float       lower_bound,
        void       *stream);

/*
 * One-token update for K3's depthwise width-4 Q/K/V convolution. input,
 * output, and the in-place [channel_count, 4] cache are BF16; weights are
 * the checkpoint-native F32 [channel_count, 1, 4] values. SiLU is fused.
 */
bool k3_rocm_short_conv4_silu_bf16_f32_weight(
        void       *output,
        void       *cache,
        const void *input,
        const void *weight,
        uint32_t    channel_count,
        void       *stream);

/*
 * Causal token-range form of the same depthwise convolution. Each channel
 * scans token_count BF16 rows in order and leaves the exact final width-4
 * cache that repeated one-token calls would produce.
 */
bool k3_rocm_short_conv4_silu_sequence_bf16_f32_weight(
        void       *output,
        void       *cache,
        const void *input,
        const void *weight,
        uint32_t    token_count,
        uint32_t    channel_count,
        void       *stream);

/*
 * Per-vector RMSNorm followed by K3's sigmoid output gate. input/gate/output
 * are BF16, while the shared norm weight is checkpoint-native F32.
 */
bool k3_rocm_rms_norm_sigmoid_gate_bf16_f32_weight(
        void       *output,
        const void *input,
        const void *gate,
        const void *weight,
        uint32_t    vector_count,
        uint32_t    hidden_size,
        float       epsilon,
        void       *stream);

/*
 * K3's no-aux router: sigmoid(logits), select top-k using the correction
 * bias, then gather and renormalize the uncorrected sigmoid scores.
 * IDs are returned in descending corrected-score order with lower-ID tie
 * break. K3 currently uses expert_count=896 and top_k=16.
 */
bool k3_rocm_router_topk_f32(void       *expert_ids,
                             void       *expert_weights,
                             const void *logits,
                             const void *correction_bias,
                             uint32_t    expert_count,
                             uint32_t    top_k,
                             float       scaling_factor,
                             void       *stream);

bool k3_rocm_router_topk_f32_batch(
        void       *expert_ids,
        void       *expert_weights,
        const void *logits,
        const void *correction_bias,
        uint32_t    vector_count,
        uint32_t    expert_count,
        uint32_t    top_k,
        float       scaling_factor,
        void       *stream);

bool k3_rocm_weighted_sum_bf16(void       *output,
                                const void *vectors,
                                const void *weights,
                                uint32_t    vector_count,
                                uint32_t    hidden_size,
                                void       *stream);

bool k3_rocm_weighted_sum_bf16_batch(
        void       *output,
        const void *vectors,
        const void *weights,
        uint32_t    token_count,
        uint32_t    vector_count,
        uint32_t    hidden_size,
        void       *stream);

bool k3_rocm_gather_rows_bf16(
        void       *output,
        const void *input,
        const void *row_ids,
        uint32_t    row_count,
        uint32_t    row_size,
        void       *stream);

bool k3_rocm_scatter_rows_bf16(
        void       *output,
        const void *input,
        const void *row_ids,
        uint32_t    row_count,
        uint32_t    row_size,
        void       *stream);

bool k3_rocm_add_bf16(void       *output,
                       const void *left,
                       const void *right,
                       uint64_t    element_count,
                       void       *stream);

bool k3_rocm_sigmoid_mul_bf16(void       *output,
                               const void *input,
                               const void *gate,
                               uint64_t    element_count,
                               void       *stream);

/*
 * Greedy BF16 logit selection. Ties resolve to the lower token ID. token_id
 * and token_value may be HIP-mapped coherent host allocations.
 */
bool k3_rocm_argmax_bf16(void       *token_id,
                          void       *token_value,
                          const void *logits,
                          uint32_t    count,
                          void       *stream);

#ifdef __cplusplus
}
#endif

#endif
