#include "glm53_fp8_dynamic.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/*
 * Semantics pinned to vLLM commit
 * c28feab98919739ae6d2041c31c94a3b11718590, function
 * _per_token_group_quant_fp8 in
 * vllm/model_executor/layers/quantization/utils/fp8_utils.py: load F32,
 * group absmax, max with the default eps=1e-10, multiply by the rounded
 * reciprocal of 448, divide, clamp, then cast. This CPU oracle abstracts the
 * platform dtype to the model's released OCP E4M3FN rather than ROCm FNUZ.
 */

static float decode_finite_magnitude(unsigned code) {
    const unsigned exponent = (code >> 3) & 15u;
    const unsigned fraction = code & 7u;
    if (exponent == 0u) {
        return ldexpf((float)fraction, -9);
    }
    return ldexpf(1.0f + (float)fraction * 0.125f, (int)exponent - 7);
}

glm53_fp8_dynamic_status glm53_fp8_dynamic_decode(uint8_t encoded,
                                                   float *value) {
    float magnitude;
    if (value == NULL) return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;
    if ((encoded & UINT8_C(0x7f)) == UINT8_C(0x7f)) {
        return GLM53_FP8_DYNAMIC_NAN_ENCODING;
    }
    magnitude = decode_finite_magnitude((unsigned)encoded & 0x7fu);
    if ((encoded & UINT8_C(0x80)) != 0u) {
        uint32_t bits;
        memcpy(&bits, &magnitude, sizeof(bits));
        bits |= UINT32_C(0x80000000);
        memcpy(&magnitude, &bits, sizeof(magnitude));
    }
    *value = magnitude;
    return GLM53_FP8_DYNAMIC_OK;
}

glm53_fp8_dynamic_status glm53_fp8_e4m3fn_encode(float value,
                                                  uint8_t *encoded) {
    float magnitude;
    float best_distance;
    unsigned best;
    unsigned code;
    uint8_t result;

    if (encoded == NULL) return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;
    if (!isfinite(value)) return GLM53_FP8_DYNAMIC_NONFINITE_VALUE;

    magnitude = fabsf(value);
    /* In torch.float8_e4m3fn, 0x7f is NaN and acts as the next RNE code.
     * Its midpoint with 448 is 464; the even 0x7e wins the exact tie. */
    if (magnitude > 464.0f) return GLM53_FP8_DYNAMIC_NAN_ENCODING;

    best = 0u;
    best_distance = magnitude;
    for (code = 1u; code <= 0x7eu; ++code) {
        const float candidate = decode_finite_magnitude(code);
        const float distance = fabsf(magnitude - candidate);
        if (distance < best_distance ||
            (distance == best_distance && (code & 1u) == 0u &&
             (best & 1u) != 0u)) {
            best = code;
            best_distance = distance;
        }
    }
    result = (uint8_t)best;
    {
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        if ((bits & UINT32_C(0x80000000)) != 0u) result |= UINT8_C(0x80);
    }
    *encoded = result;
    return GLM53_FP8_DYNAMIC_OK;
}

static int checked_span(size_t rows, size_t width, size_t stride,
                        size_t *span) {
    size_t offset;
    if (rows == 0u || width == 0u || stride < width) return 0;
    if (rows - 1u > SIZE_MAX / stride) return -1;
    offset = (rows - 1u) * stride;
    if (offset > SIZE_MAX - width) return -1;
    *span = offset + width;
    return 1;
}

/* Return one for overlap, zero for disjoint spans, and minus one when a byte
 * range cannot be represented. This keeps the success-only output contract
 * honest even when callers accidentally alias input and workspace buffers. */
static int spans_overlap(const void *a, size_t a_count, size_t a_element_bytes,
                         const void *b, size_t b_count, size_t b_element_bytes) {
    const uintptr_t a_start = (uintptr_t)a;
    const uintptr_t b_start = (uintptr_t)b;
    size_t a_bytes, b_bytes;
    uintptr_t a_end, b_end;
    if (a_count > SIZE_MAX / a_element_bytes ||
        b_count > SIZE_MAX / b_element_bytes) return -1;
    a_bytes = a_count * a_element_bytes;
    b_bytes = b_count * b_element_bytes;
    if (a_bytes > UINTPTR_MAX - a_start || b_bytes > UINTPTR_MAX - b_start)
        return -1;
    a_end = a_start + (uintptr_t)a_bytes;
    b_end = b_start + (uintptr_t)b_bytes;
    return a_start < b_end && b_start < a_end;
}

glm53_fp8_dynamic_status glm53_fp8_dynamic_quantize_f32(
    uint8_t *quantized, size_t quantized_count, size_t quantized_row_stride,
    float *inverse_scales, size_t inverse_scale_count,
    size_t inverse_scale_row_stride,
    const float *input, size_t input_count, size_t input_row_stride,
    size_t tokens, size_t columns) {
    size_t groups;
    size_t needed_q;
    size_t needed_s;
    size_t needed_x;
    int check;
    size_t token;
    size_t group;
    size_t column;

    if (quantized == NULL || inverse_scales == NULL || input == NULL ||
        tokens == 0u || columns == 0u ||
        columns % GLM53_FP8_DYNAMIC_GROUP_COLS != 0u) {
        return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;
    }
    groups = columns / GLM53_FP8_DYNAMIC_GROUP_COLS;

    check = checked_span(tokens, columns, quantized_row_stride, &needed_q);
    if (check < 0) return GLM53_FP8_DYNAMIC_DIMENSION_OVERFLOW;
    if (check == 0) return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;
    check = checked_span(tokens, groups, inverse_scale_row_stride, &needed_s);
    if (check < 0) return GLM53_FP8_DYNAMIC_DIMENSION_OVERFLOW;
    if (check == 0) return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;
    check = checked_span(tokens, columns, input_row_stride, &needed_x);
    if (check < 0) return GLM53_FP8_DYNAMIC_DIMENSION_OVERFLOW;
    if (check == 0) return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;
    if (needed_q > quantized_count || needed_s > inverse_scale_count ||
        needed_x > input_count) {
        return GLM53_FP8_DYNAMIC_BUFFER_TOO_SMALL;
    }
    check = spans_overlap(quantized, needed_q, sizeof(*quantized),
                          inverse_scales, needed_s, sizeof(*inverse_scales));
    if (check < 0) return GLM53_FP8_DYNAMIC_DIMENSION_OVERFLOW;
    if (check != 0) return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;
    check = spans_overlap(quantized, needed_q, sizeof(*quantized),
                          input, needed_x, sizeof(*input));
    if (check < 0) return GLM53_FP8_DYNAMIC_DIMENSION_OVERFLOW;
    if (check != 0) return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;
    check = spans_overlap(inverse_scales, needed_s, sizeof(*inverse_scales),
                          input, needed_x, sizeof(*input));
    if (check < 0) return GLM53_FP8_DYNAMIC_DIMENSION_OVERFLOW;
    if (check != 0) return GLM53_FP8_DYNAMIC_INVALID_ARGUMENT;

    /* Complete the only data-dependent failure pass before either output. */
    for (token = 0u; token < tokens; ++token) {
        for (column = 0u; column < columns; ++column) {
            if (!isfinite(input[token * input_row_stride + column])) {
                return GLM53_FP8_DYNAMIC_NONFINITE_VALUE;
            }
        }
    }

    for (token = 0u; token < tokens; ++token) {
        for (group = 0u; group < groups; ++group) {
            const size_t first = group * GLM53_FP8_DYNAMIC_GROUP_COLS;
            float absmax = 0.0f;
            float scale;
            for (column = 0u; column < GLM53_FP8_DYNAMIC_GROUP_COLS;
                 ++column) {
                const float magnitude =
                    fabsf(input[token * input_row_stride + first + column]);
                if (magnitude > absmax) absmax = magnitude;
            }
            if (absmax < GLM53_FP8_DYNAMIC_EPS) {
                absmax = GLM53_FP8_DYNAMIC_EPS;
            }
            scale = absmax * GLM53_FP8_DYNAMIC_INV_MAX;
            inverse_scales[token * inverse_scale_row_stride + group] = scale;

            for (column = 0u; column < GLM53_FP8_DYNAMIC_GROUP_COLS;
                 ++column) {
                const float original =
                    input[token * input_row_stride + first + column];
                float scaled = original / scale;
                uint8_t code;
                if (scaled > GLM53_FP8_DYNAMIC_MAX) {
                    scaled = GLM53_FP8_DYNAMIC_MAX;
                } else if (scaled < -GLM53_FP8_DYNAMIC_MAX) {
                    scaled = -GLM53_FP8_DYNAMIC_MAX;
                }
                /* Clamp guarantees a successful finite encoding. Preserve
                 * the original sign bit explicitly because fast-math is
                 * permitted elsewhere in the project and may fold -0 / s. */
                (void)glm53_fp8_e4m3fn_encode(scaled, &code);
                {
                    uint32_t bits;
                    memcpy(&bits, &original, sizeof(bits));
                    code = (uint8_t)((code & UINT8_C(0x7f)) |
                        (uint8_t)((bits >> 24u) & UINT32_C(0x80)));
                }
                quantized[token * quantized_row_stride + first + column] = code;
            }
        }
    }
    return GLM53_FP8_DYNAMIC_OK;
}
