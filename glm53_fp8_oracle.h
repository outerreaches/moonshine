#ifndef GLM53_FP8_ORACLE_H
#define GLM53_FP8_ORACLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_FP8_ORACLE_BLOCK_ROWS 128u
#define GLM53_FP8_ORACLE_BLOCK_COLS 128u

typedef enum glm53_fp8_oracle_status {
    GLM53_FP8_ORACLE_OK = 0,
    GLM53_FP8_ORACLE_INVALID_ARGUMENT,
    GLM53_FP8_ORACLE_DIMENSION_OVERFLOW,
    GLM53_FP8_ORACLE_BUFFER_TOO_SMALL,
    GLM53_FP8_ORACLE_NAN_ENCODING,
    GLM53_FP8_ORACLE_NONFINITE_VALUE
} glm53_fp8_oracle_status;

/*
 * Decode one OCP E4M3FN byte to binary32.  0x7f and 0xff are the only NaN
 * encodings.  For either one, *value is set to the canonical quiet binary32
 * NaN (bits 0x7fc00000) and GLM53_FP8_ORACLE_NAN_ENCODING is returned.
 * Every other byte, including signed zero, is decoded exactly.
 */
glm53_fp8_oracle_status glm53_fp8_e4m3fn_decode(uint8_t encoded,
                                                 float *value);

/*
 * Compute y = W x with block-scaled FP8 weights.
 *
 * Tensor orientation is explicit: W is [rows, cols], row-major with
 * weight_row_stride elements between row starts; x is [cols]; and y is [rows].
 * inverse_scales is [ceil(rows/128), ceil(cols/128)], row-major.  Element
 * W[r,c] projects as decode(W[r,c]) * inverse_scales[r/128,c/128].
 * All strides and counts are in elements, not bytes.  The count arguments
 * describe the complete accessible span starting at each pointer.
 *
 * The operation uses float multiplication and accumulation.  It allocates no
 * memory.  All geometry, spans, FP8 NaNs, and non-finite x/scale values are
 * validated before y is modified.  Thus an error leaves y unchanged.
 * Buffers must not overlap.
 */
glm53_fp8_oracle_status glm53_fp8_project_f32(
    float *y, size_t y_count, size_t y_stride,
    const uint8_t *weights, size_t weight_count, size_t weight_row_stride,
    const float *inverse_scales, size_t inverse_scale_count,
    size_t inverse_scale_row_stride,
    const float *x, size_t x_count, size_t x_stride,
    size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif
