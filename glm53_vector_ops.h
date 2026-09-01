#ifndef GLM53_VECTOR_OPS_H
#define GLM53_VECTOR_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Allocation-free, asynchronous vector primitives. All pointers name
 * contiguous device storage and every count is in elements of the pointed-to
 * type. stream is a hipStream_t passed as void *; NULL selects the default
 * stream. Calls validate capacity, arithmetic/address overflow, and require
 * all buffers named by a call to be pairwise non-overlapping. They enqueue
 * work only and report argument and kernel-launch status, not later execution
 * errors. Input data must be finite unless IEEE propagation is desired.
 */

/* table is row-major BF16 [rows, columns]; output is F32 [columns]. */
bool glm53_vector_gather_bf16_f32(
    void *output, size_t output_count,
    const void *table, size_t table_count,
    uint32_t rows, uint32_t columns, uint32_t row_index, void *stream);

/* output/input are F32 [count]. weight is F32 or BF16 [count]. */
bool glm53_vector_rmsnorm_f32(
    void *output, size_t output_count, const void *input, size_t input_count,
    const void *weight, size_t weight_count, size_t count, float eps,
    void *stream);
bool glm53_vector_rmsnorm_bf16_weight_f32(
    void *output, size_t output_count, const void *input, size_t input_count,
    const void *weight, size_t weight_count, size_t count, float eps,
    void *stream);

bool glm53_vector_add_f32(
    void *output, size_t output_count, const void *left, size_t left_count,
    const void *right, size_t right_count, size_t count, void *stream);
bool glm53_vector_multiply_f32(
    void *output, size_t output_count, const void *left, size_t left_count,
    const void *right, size_t right_count, size_t count, void *stream);
bool glm53_vector_sigmoid_f32(
    void *output, size_t output_count, const void *input, size_t input_count,
    size_t count, void *stream);
bool glm53_vector_silu_f32(
    void *output, size_t output_count, const void *input, size_t input_count,
    size_t count, void *stream);
/* Official limited gate: gate is capped above at limit; up is clamped to [-limit, limit], then output = silu(gate) * up. */
bool glm53_vector_swiglu_f32(
    void *output, size_t output_count, const void *gate, size_t gate_count,
    const void *up, size_t up_count, size_t count, float limit, void *stream);

bool glm53_vector_cast_f32_bf16(
    void *output, size_t output_count, const void *input, size_t input_count,
    size_t count, void *stream);
bool glm53_vector_cast_bf16_f32(
    void *output, size_t output_count, const void *input, size_t input_count,
    size_t count, void *stream);

/* Device scalar output[0] = dot(left, right). */
bool glm53_vector_dot_f32(
    void *output, size_t output_count, const void *left, size_t left_count,
    const void *right, size_t right_count, size_t count, void *stream);

/* matrix is row-major [rows, columns]. rows and columns are nonzero; the official mHC projection is 24 x 16384. */
bool glm53_vector_matvec_f32(
    void *output, size_t output_count,
    const void *matrix, size_t matrix_count,
    const void *input, size_t input_count,
    uint32_t rows, uint32_t columns, void *stream);
bool glm53_vector_matvec_bf16_weight_f32(
    void *output, size_t output_count,
    const void *matrix, size_t matrix_count,
    const void *input, size_t input_count,
    uint32_t rows, uint32_t columns, void *stream);

#ifdef __cplusplus
}
#endif
#endif
