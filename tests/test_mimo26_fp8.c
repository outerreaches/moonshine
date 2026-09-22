/*
 * M1: confirm GLM's block-FP8 oracle is reusable for MiMo's dense and QKV
 * projections, rather than writing a second implementation.
 *
 * The E4M3 code table is derived here from the OCP bit fields instead of being
 * copied from the decoder under test, so a disagreement is a real semantic
 * difference. The geometry cases include MiMo's over-provisioned global QKV
 * scale grid, which is the case most likely to be rejected by a stricter
 * validator.
 */
#include "glm53_fp8_oracle.h"
#include "mimo26_architecture.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* OCP FP8 E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa, no infinities,
 * and S.1111.111 as the sole NaN pattern. */
static bool spec_e4m3fn(uint8_t encoded, float *value)
{
    const uint32_t sign = (uint32_t)(encoded >> 7);
    const uint32_t exponent = (uint32_t)((encoded >> 3) & 0xFu);
    const uint32_t mantissa = (uint32_t)(encoded & 0x7u);

    if (exponent == 0xFu && mantissa == 0x7u) {
        return false; /* NaN */
    }
    double magnitude;
    if (exponent == 0u) {
        magnitude = (double)mantissa * ldexp(1.0, -9); /* m/8 * 2^-6 */
    } else {
        magnitude = (1.0 + (double)mantissa / 8.0) *
                    ldexp(1.0, (int)exponent - 7);
    }
    *value = (float)(sign ? -magnitude : magnitude);
    return true;
}

static void check_code_table(void)
{
    size_t nan_count = 0;
    float max_finite = 0.0f;
    float min_positive_subnormal = INFINITY;
    for (unsigned encoded = 0; encoded < 256u; encoded++) {
        float expected = 0.0f;
        const bool finite = spec_e4m3fn((uint8_t)encoded, &expected);
        float actual = 0.0f;
        const glm53_fp8_oracle_status status =
            glm53_fp8_e4m3fn_decode((uint8_t)encoded, &actual);
        if (!finite) {
            assert(status == GLM53_FP8_ORACLE_NAN_ENCODING);
            assert(isnan(actual));
            nan_count++;
            continue;
        }
        assert(status == GLM53_FP8_ORACLE_OK);
        /* Bit-exact: every E4M3 value is representable in binary32. */
        assert(memcmp(&actual, &expected, sizeof actual) == 0);
        if (fabsf(actual) > max_finite) {
            max_finite = fabsf(actual);
        }
        if (((uint8_t)encoded & 0x78u) == 0u && (encoded & 0x7u) != 0u &&
            fabsf(actual) < min_positive_subnormal) {
            min_positive_subnormal = fabsf(actual);
        }
    }
    assert(nan_count == 2); /* 0x7f and 0xff only */
    assert(max_finite == 448.0f);
    assert(min_positive_subnormal == ldexpf(1.0f, -9));
    /* Both zeros decode, and they are distinguishable by sign bit. */
    float positive_zero = 1.0f;
    float negative_zero = 1.0f;
    assert(glm53_fp8_e4m3fn_decode(0x00u, &positive_zero) ==
           GLM53_FP8_ORACLE_OK);
    assert(glm53_fp8_e4m3fn_decode(0x80u, &negative_zero) ==
           GLM53_FP8_ORACLE_OK);
    assert(positive_zero == 0.0f && negative_zero == 0.0f);
    assert(signbit(negative_zero) && !signbit(positive_zero));
    printf("  ok  256 E4M3 codes match an independent spec derivation "
           "(2 NaN, max 448, min subnormal 2^-9, both zeros)\n");
}

/* Reference projection built from the spec decoder, independent of the
 * oracle's own loop order. Accumulates in double to stay distinguishable
 * from the implementation's float accumulation. */
static void reference_project(double *y, const uint8_t *weights,
                              const float *scales, size_t scale_stride,
                              const float *x, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        double sum = 0.0;
        for (size_t c = 0; c < cols; c++) {
            float decoded = 0.0f;
            const bool finite = spec_e4m3fn(weights[r * cols + c], &decoded);
            assert(finite);
            sum += (double)decoded *
                   (double)scales[(r / 128u) * scale_stride + (c / 128u)] *
                   (double)x[c];
        }
        y[r] = sum;
    }
}

static void check_geometry(const char *label, size_t rows, size_t cols,
                           size_t scale_rows, size_t scale_cols)
{
    /* Keep the exercised block grid small: only the first rows are compared,
     * but the declared scale grid is the checkpoint's real one. */
    const size_t probe_rows = rows < 256u ? rows : 256u;
    uint8_t *weights = malloc(probe_rows * cols);
    float *scales = malloc(scale_rows * scale_cols * sizeof *scales);
    float *x = malloc(cols * sizeof *x);
    float *y = malloc(probe_rows * sizeof *y);
    double *expected = malloc(probe_rows * sizeof *expected);
    assert(weights && scales && x && y && expected);

    /* Deterministic, finite, and avoiding the two NaN encodings. */
    for (size_t i = 0; i < probe_rows * cols; i++) {
        uint8_t code = (uint8_t)((i * 37u + 11u) & 0xFFu);
        if (code == 0x7Fu || code == 0xFFu) {
            code = 0x3Cu;
        }
        weights[i] = code;
    }
    for (size_t i = 0; i < scale_rows * scale_cols; i++) {
        scales[i] = ldexpf(1.0f, -(int)(i % 9u) - 2);
    }
    for (size_t i = 0; i < cols; i++) {
        x[i] = (float)((i % 17u) - 8) * 0.125f;
    }

    const glm53_fp8_oracle_status status = glm53_fp8_project_f32(
        y, probe_rows, 1u,
        weights, probe_rows * cols, cols,
        scales, scale_rows * scale_cols, scale_cols,
        x, cols, 1u,
        probe_rows, cols);
    assert(status == GLM53_FP8_ORACLE_OK);

    reference_project(expected, weights, scales, scale_cols, x, probe_rows,
                      cols);
    double worst = 0.0;
    for (size_t r = 0; r < probe_rows; r++) {
        const double scale = fabs(expected[r]) > 1e-12 ? fabs(expected[r]) : 1.0;
        const double relative = fabs((double)y[r] - expected[r]) / scale;
        if (relative > worst) {
            worst = relative;
        }
    }
    /* float accumulation against a double reference over `cols` terms. */
    assert(worst < 1e-5);
    printf("  ok  %-26s rows=%-6zu cols=%-6zu scales=[%zu,%zu]  "
           "max rel %.2e\n", label, rows, cols, scale_rows, scale_cols, worst);

    free(weights);
    free(scales);
    free(x);
    free(y);
    free(expected);
}

int main(void)
{
    check_code_table();

    /* MiMo's four FP8 geometries. The global QKV grid is [108,32] where an
     * exact ceil(13568/128) would be 106: the surplus block rows must be
     * tolerated, not rejected. */
    check_geometry("global qkv", 13568u, 4096u, 108u, 32u);
    check_geometry("swa qkv", 14848u, 4096u, 116u, 32u);
    check_geometry("dense gate/up", 16384u, 4096u, 128u, 32u);
    check_geometry("dense down", 4096u, 16384u, 32u, 128u);

    /* A grid smaller than ceil(rows/128) must still be refused. */
    {
        uint8_t weights[256];
        float scales[2];
        float x[128];
        float y[2];
        memset(weights, 0x3Cu, sizeof weights);
        scales[0] = 1.0f;
        scales[1] = 1.0f;
        for (size_t i = 0; i < 128u; i++) {
            x[i] = 1.0f;
        }
        /* rows=129 needs two block rows; offering one is too small. */
        const glm53_fp8_oracle_status status = glm53_fp8_project_f32(
            y, 2u, 1u, weights, sizeof weights, 128u,
            scales, 1u, 1u, x, 128u, 1u, 129u, 128u);
        assert(status == GLM53_FP8_ORACLE_BUFFER_TOO_SMALL);
        printf("  ok  undersized scale grid is refused\n");
    }

    /* A NaN weight encoding must be reported, not silently propagated. */
    {
        uint8_t weights[128];
        float scales[1] = {1.0f};
        float x[128];
        float y[1] = {12345.0f};
        memset(weights, 0x3Cu, sizeof weights);
        weights[64] = 0x7Fu;
        for (size_t i = 0; i < 128u; i++) {
            x[i] = 1.0f;
        }
        const glm53_fp8_oracle_status status = glm53_fp8_project_f32(
            y, 1u, 1u, weights, sizeof weights, 128u,
            scales, 1u, 1u, x, 128u, 1u, 1u, 128u);
        assert(status == GLM53_FP8_ORACLE_NAN_ENCODING);
        assert(y[0] == 12345.0f); /* output untouched on error */
        printf("  ok  NaN weight encoding is reported and y is unchanged\n");
    }

    printf("test_mimo26_fp8: ok  (GLM block-FP8 oracle is reusable for MiMo)\n");
    return 0;
}
