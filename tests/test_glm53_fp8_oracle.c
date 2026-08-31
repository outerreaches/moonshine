#include "glm53_fp8_oracle.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/*
 * Independent expected-value construction: map E4M3 fields directly into
 * IEEE binary32 fields.  This does not call the decoder or use its arithmetic.
 */
static uint32_t expected_bits(uint8_t encoded, bool *is_nan_encoding) {
    const uint32_t sign = (uint32_t)(encoded & UINT8_C(0x80)) << 24;
    const unsigned exponent = ((unsigned)encoded >> 3) & 15u;
    const unsigned fraction = (unsigned)encoded & 7u;

    if (exponent == 15u && fraction == 7u) {
        *is_nan_encoding = true;
        return UINT32_C(0x7fc00000);
    }
    *is_nan_encoding = false;
    if (exponent == 0u) {
        unsigned top;
        uint32_t ieee_exponent;
        uint32_t ieee_fraction;
        if (fraction == 0u) return sign;
        top = fraction >= 4u ? 2u : (fraction >= 2u ? 1u : 0u);
        ieee_exponent = (uint32_t)(118u + top); /* (top - 9) + 127 */
        ieee_fraction = (uint32_t)(fraction - (1u << top)) << (23u - top);
        return sign | (ieee_exponent << 23) | ieee_fraction;
    }
    return sign | ((uint32_t)(exponent + 120u) << 23) |
           ((uint32_t)fraction << 20);
}

static bool test_all_decodings(void) {
    unsigned i;
    CHECK(sizeof(float) == sizeof(uint32_t));
    CHECK(glm53_fp8_e4m3fn_decode(0u, NULL) ==
          GLM53_FP8_ORACLE_INVALID_ARGUMENT);
    for (i = 0u; i < 256u; ++i) {
        float got = 123.0f;
        bool want_nan;
        const uint32_t want = expected_bits((uint8_t)i, &want_nan);
        const glm53_fp8_oracle_status status =
            glm53_fp8_e4m3fn_decode((uint8_t)i, &got);
        CHECK(status == (want_nan ? GLM53_FP8_ORACLE_NAN_ENCODING :
                                    GLM53_FP8_ORACLE_OK));
        CHECK(float_bits(got) == want);
    }
    return true;
}

static bool test_hand_projection(void) {
    /* 1.5, minimum subnormal, -2, maximum finite. */
    const uint8_t weights[] = { UINT8_C(0x3c), UINT8_C(0x01),
                                UINT8_C(0xc0), UINT8_C(0x7e) };
    const float scale[] = { 0.5f };
    const float x[] = { 2.0f, 256.0f, 3.0f, 0.25f };
    float y[] = { -99.0f };
    CHECK(glm53_fp8_project_f32(y, 1u, 1u, weights, 4u, 4u,
                                scale, 1u, 1u, x, 4u, 1u, 1u, 4u) ==
          GLM53_FP8_ORACLE_OK);
    /* .5 * (1.5*2 + 2^-9*256 + -2*3 + 448*.25) = 54.75 */
    CHECK(float_bits(y[0]) == float_bits(54.75f));
    return true;
}

static bool test_partial_blocks_and_strides(void) {
    const size_t rows = 129u;
    const size_t cols = 129u;
    const size_t weight_stride = 131u;
    const size_t vector_stride = 2u;
    const size_t weight_count = (rows - 1u) * weight_stride + cols;
    const size_t vector_count = (rows - 1u) * vector_stride + 1u;
    uint8_t *weights = (uint8_t *)calloc(weight_count, sizeof(*weights));
    float *x = (float *)calloc(vector_count, sizeof(*x));
    float *y = (float *)malloc(vector_count * sizeof(*y));
    /* Logical shape is 2x2, with one padding element between scale rows. */
    float scales[] = { 1.0f, 2.0f, -77.0f, 3.0f, 4.0f };
    size_t i;
    bool ok = true;

    CHECK(weights != NULL && x != NULL && y != NULL);
    for (i = 0u; i < vector_count; ++i) y[i] = -77.0f;
    weights[0u * weight_stride + 0u] = UINT8_C(0x38);   /* 1 */
    weights[0u * weight_stride + 128u] = UINT8_C(0x38);
    weights[128u * weight_stride + 0u] = UINT8_C(0x38);
    weights[128u * weight_stride + 128u] = UINT8_C(0x38);
    x[0u * vector_stride] = 1.0f;
    x[128u * vector_stride] = 2.0f;

    if (glm53_fp8_project_f32(y, vector_count, vector_stride,
            weights, weight_count, weight_stride, scales, 5u, 3u,
            x, vector_count, vector_stride, rows, cols) !=
        GLM53_FP8_ORACLE_OK) ok = false;
    if (ok && float_bits(y[0u]) != float_bits(5.0f)) ok = false;
    if (ok && float_bits(y[128u * vector_stride]) != float_bits(11.0f))
        ok = false;
    for (i = 1u; ok && i < 128u; ++i) {
        if (float_bits(y[i * vector_stride]) != float_bits(0.0f)) ok = false;
    }
    /* Output stride padding must not be touched. */
    for (i = 1u; ok && i < vector_count; i += 2u) {
        if (float_bits(y[i]) != float_bits(-77.0f)) ok = false;
    }
    free(y);
    free(x);
    free(weights);
    CHECK(ok);
    return true;
}

static bool test_nan_and_nonfinite_rejection(void) {
    uint8_t weights[] = { UINT8_C(0x38), UINT8_C(0x7f) };
    float scales[] = { 1.0f };
    float x[] = { 1.0f, 1.0f };
    float y[] = { 19.0f };

    CHECK(glm53_fp8_project_f32(y, 1u, 1u, weights, 2u, 2u,
                                scales, 1u, 1u, x, 2u, 1u, 1u, 2u) ==
          GLM53_FP8_ORACLE_NAN_ENCODING);
    CHECK(float_bits(y[0]) == float_bits(19.0f));
    weights[1] = UINT8_C(0xff);
    CHECK(glm53_fp8_project_f32(y, 1u, 1u, weights, 2u, 2u,
                                scales, 1u, 1u, x, 2u, 1u, 1u, 2u) ==
          GLM53_FP8_ORACLE_NAN_ENCODING);
    CHECK(float_bits(y[0]) == float_bits(19.0f));
    weights[1] = UINT8_C(0x38);
    scales[0] = NAN;
    CHECK(glm53_fp8_project_f32(y, 1u, 1u, weights, 2u, 2u,
                                scales, 1u, 1u, x, 2u, 1u, 1u, 2u) ==
          GLM53_FP8_ORACLE_NONFINITE_VALUE);
    CHECK(float_bits(y[0]) == float_bits(19.0f));
    scales[0] = 1.0f;
    x[0] = INFINITY;
    CHECK(glm53_fp8_project_f32(y, 1u, 1u, weights, 2u, 2u,
                                scales, 1u, 1u, x, 2u, 1u, 1u, 2u) ==
          GLM53_FP8_ORACLE_NONFINITE_VALUE);
    CHECK(float_bits(y[0]) == float_bits(19.0f));
    return true;
}

static bool test_geometry_fail_closed(void) {
    const uint8_t w[] = { UINT8_C(0x38), UINT8_C(0x38) };
    const float scale[] = { 1.0f };
    const float x[] = { 1.0f, 1.0f };
    float y[] = { 23.0f };

    CHECK(glm53_fp8_project_f32(y, 1u, 1u, w, 2u, 1u,
                                scale, 1u, 1u, x, 2u, 1u, 1u, 2u) ==
          GLM53_FP8_ORACLE_INVALID_ARGUMENT); /* row stride < cols */
    CHECK(float_bits(y[0]) == float_bits(23.0f));
    CHECK(glm53_fp8_project_f32(y, 1u, 1u, w, 1u, 2u,
                                scale, 1u, 1u, x, 2u, 1u, 1u, 2u) ==
          GLM53_FP8_ORACLE_BUFFER_TOO_SMALL);
    CHECK(float_bits(y[0]) == float_bits(23.0f));
    CHECK(glm53_fp8_project_f32(y, 1u, 2u, w, 2u, 1u,
                                scale, 1u, 1u, x, 2u, 1u,
                                SIZE_MAX, 1u) ==
          GLM53_FP8_ORACLE_DIMENSION_OVERFLOW);
    CHECK(float_bits(y[0]) == float_bits(23.0f));
    CHECK(glm53_fp8_project_f32(y, 1u, 1u, w, 2u, 2u,
                                scale, 1u, 1u, x, 2u, 1u, 0u, 2u) ==
          GLM53_FP8_ORACLE_INVALID_ARGUMENT);
    CHECK(float_bits(y[0]) == float_bits(23.0f));
    return true;
}

int main(void) {
    if (!test_all_decodings() || !test_hand_projection() ||
        !test_partial_blocks_and_strides() ||
        !test_nan_and_nonfinite_rejection() ||
        !test_geometry_fail_closed()) return 1;
    puts("glm53 fp8 oracle tests: ok");
    return 0;
}
