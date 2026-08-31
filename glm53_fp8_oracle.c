#include "glm53_fp8_oracle.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static float canonical_nan(void) {
    const uint32_t bits = UINT32_C(0x7fc00000);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

glm53_fp8_oracle_status glm53_fp8_e4m3fn_decode(uint8_t encoded,
                                                 float *value) {
    unsigned sign;
    unsigned exponent;
    unsigned fraction;
    float magnitude;

    if (value == NULL) {
        return GLM53_FP8_ORACLE_INVALID_ARGUMENT;
    }

    sign = (unsigned)(encoded >> 7);
    exponent = ((unsigned)encoded >> 3) & 15u;
    fraction = (unsigned)encoded & 7u;

    /* OCP E4M3FN reserves only S.1111.111 for NaN. */
    if (exponent == 15u && fraction == 7u) {
        *value = canonical_nan();
        return GLM53_FP8_ORACLE_NAN_ENCODING;
    }

    if (exponent == 0u) {
        /* Subnormal: (fraction / 8) * 2^-6 = fraction * 2^-9. */
        magnitude = ldexpf((float)fraction, -9);
    } else {
        /* Normal and extended exponent-15 finite encodings. */
        magnitude = ldexpf(1.0f + (float)fraction * 0.125f,
                            (int)exponent - 7);
    }
    *value = sign != 0u ? -magnitude : magnitude;
    return GLM53_FP8_ORACLE_OK;
}

static int checked_span(size_t outer, size_t inner, size_t stride,
                        size_t *span) {
    size_t row_offset;
    size_t last;

    if (outer == 0u || inner == 0u || stride < inner) {
        return 0;
    }
    if (outer - 1u > SIZE_MAX / stride) {
        return -1;
    }
    row_offset = (outer - 1u) * stride;
    if (row_offset > SIZE_MAX - (inner - 1u)) {
        return -1;
    }
    last = row_offset + inner - 1u;
    if (last == SIZE_MAX) {
        return -1;
    }
    *span = last + 1u;
    return 1;
}

static int checked_vector_span(size_t length, size_t stride, size_t *span) {
    return checked_span(length, 1u, stride, span);
}

glm53_fp8_oracle_status glm53_fp8_project_f32(
    float *y, size_t y_count, size_t y_stride,
    const uint8_t *weights, size_t weight_count, size_t weight_row_stride,
    const float *inverse_scales, size_t inverse_scale_count,
    size_t inverse_scale_row_stride,
    const float *x, size_t x_count, size_t x_stride,
    size_t rows, size_t cols) {
    size_t scale_rows;
    size_t scale_cols;
    size_t needed_y;
    size_t needed_weights;
    size_t needed_scales;
    size_t needed_x;
    int check;
    size_t r;
    size_t c;

    if (y == NULL || weights == NULL || inverse_scales == NULL || x == NULL ||
        rows == 0u || cols == 0u || y_stride == 0u || x_stride == 0u) {
        return GLM53_FP8_ORACLE_INVALID_ARGUMENT;
    }

    scale_rows = rows / GLM53_FP8_ORACLE_BLOCK_ROWS +
                 (rows % GLM53_FP8_ORACLE_BLOCK_ROWS != 0u ? 1u : 0u);
    scale_cols = cols / GLM53_FP8_ORACLE_BLOCK_COLS +
                 (cols % GLM53_FP8_ORACLE_BLOCK_COLS != 0u ? 1u : 0u);

    check = checked_vector_span(rows, y_stride, &needed_y);
    if (check < 0) return GLM53_FP8_ORACLE_DIMENSION_OVERFLOW;
    if (check == 0) return GLM53_FP8_ORACLE_INVALID_ARGUMENT;
    check = checked_span(rows, cols, weight_row_stride, &needed_weights);
    if (check < 0) return GLM53_FP8_ORACLE_DIMENSION_OVERFLOW;
    if (check == 0) return GLM53_FP8_ORACLE_INVALID_ARGUMENT;
    check = checked_span(scale_rows, scale_cols, inverse_scale_row_stride,
                         &needed_scales);
    if (check < 0) return GLM53_FP8_ORACLE_DIMENSION_OVERFLOW;
    if (check == 0) return GLM53_FP8_ORACLE_INVALID_ARGUMENT;
    check = checked_vector_span(cols, x_stride, &needed_x);
    if (check < 0) return GLM53_FP8_ORACLE_DIMENSION_OVERFLOW;
    if (check == 0) return GLM53_FP8_ORACLE_INVALID_ARGUMENT;

    if (needed_y > y_count || needed_weights > weight_count ||
        needed_scales > inverse_scale_count || needed_x > x_count) {
        return GLM53_FP8_ORACLE_BUFFER_TOO_SMALL;
    }

    /* Validate all source data before producing any output. */
    for (c = 0u; c < cols; ++c) {
        if (!isfinite(x[c * x_stride])) {
            return GLM53_FP8_ORACLE_NONFINITE_VALUE;
        }
    }
    for (r = 0u; r < scale_rows; ++r) {
        for (c = 0u; c < scale_cols; ++c) {
            if (!isfinite(inverse_scales[r * inverse_scale_row_stride + c])) {
                return GLM53_FP8_ORACLE_NONFINITE_VALUE;
            }
        }
    }
    for (r = 0u; r < rows; ++r) {
        for (c = 0u; c < cols; ++c) {
            const uint8_t encoded = weights[r * weight_row_stride + c];
            if ((encoded & UINT8_C(0x7f)) == UINT8_C(0x7f)) {
                return GLM53_FP8_ORACLE_NAN_ENCODING;
            }
        }
    }

    for (r = 0u; r < rows; ++r) {
        float sum = 0.0f;
        const size_t scale_r = r / GLM53_FP8_ORACLE_BLOCK_ROWS;
        for (c = 0u; c < cols; ++c) {
            float decoded;
            float scaled;
            (void)glm53_fp8_e4m3fn_decode(
                weights[r * weight_row_stride + c], &decoded);
            scaled = decoded * inverse_scales[
                scale_r * inverse_scale_row_stride +
                c / GLM53_FP8_ORACLE_BLOCK_COLS];
            sum += scaled * x[c * x_stride];
        }
        y[r * y_stride] = sum;
    }
    return GLM53_FP8_ORACLE_OK;
}
