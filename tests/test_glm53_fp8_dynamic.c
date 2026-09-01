#include "glm53_fp8_dynamic.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)

static uint32_t fbits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static bool encode_is(float x, unsigned want) {
    uint8_t got = UINT8_C(0xa5);
    return glm53_fp8_e4m3fn_encode(x, &got) == GLM53_FP8_DYNAMIC_OK &&
           got == (uint8_t)want;
}

static bool test_representable_roundtrip(void) {
    unsigned i;
    for (i = 0u; i < 256u; ++i) {
        float x = 99.0f;
        uint8_t encoded = UINT8_C(0xa5);
        const bool nan_code = (i & 0x7fu) == 0x7fu;
        CHECK(glm53_fp8_dynamic_decode((uint8_t)i, &x) ==
              (nan_code ? GLM53_FP8_DYNAMIC_NAN_ENCODING :
                          GLM53_FP8_DYNAMIC_OK));
        if (nan_code) {
            CHECK(fbits(x) == fbits(99.0f));
        } else {
            CHECK(glm53_fp8_e4m3fn_encode(x, &encoded) ==
                  GLM53_FP8_DYNAMIC_OK);
            CHECK(encoded == (uint8_t)i);
        }
    }
    return true;
}

static bool test_torch_goldens_and_special_policy(void) {
    uint8_t out = UINT8_C(0xa5);
    CHECK(encode_is(0.0f, 0x00u));
    CHECK(encode_is(-0.0f, 0x80u));
    CHECK(encode_is(ldexpf(1.0f, -9), 0x01u));
    CHECK(encode_is(ldexpf(1.0f, -10), 0x00u)); /* zero/subnormal tie */
    CHECK(encode_is(ldexpf(3.0f, -10), 0x02u)); /* odd/even tie */
    CHECK(encode_is(1.0f, 0x38u));
    CHECK(encode_is(1.0625f, 0x38u));
    CHECK(encode_is(1.1875f, 0x3au));
    CHECK(encode_is(448.0f, 0x7eu));
    CHECK(encode_is(464.0f, 0x7eu)); /* finite/NaN tie chooses even */
    CHECK(encode_is(-464.0f, 0xfeu));
    CHECK(glm53_fp8_e4m3fn_encode(nextafterf(464.0f, INFINITY), &out) ==
          GLM53_FP8_DYNAMIC_NAN_ENCODING);
    CHECK(out == UINT8_C(0xa5));
    CHECK(glm53_fp8_e4m3fn_encode(-nextafterf(464.0f, INFINITY), &out) ==
          GLM53_FP8_DYNAMIC_NAN_ENCODING);
    CHECK(out == UINT8_C(0xa5));
    CHECK(glm53_fp8_e4m3fn_encode(NAN, &out) ==
          GLM53_FP8_DYNAMIC_NONFINITE_VALUE);
    CHECK(out == UINT8_C(0xa5));
    CHECK(glm53_fp8_e4m3fn_encode(INFINITY, &out) ==
          GLM53_FP8_DYNAMIC_NONFINITE_VALUE);
    CHECK(out == UINT8_C(0xa5));
    CHECK(glm53_fp8_e4m3fn_encode(1.0f, NULL) ==
          GLM53_FP8_DYNAMIC_INVALID_ARGUMENT);
    return true;
}

static bool test_every_midpoint(void) {
    unsigned lower;
    for (lower = 0u; lower < 0x7eu; ++lower) {
        float a;
        float b;
        float midpoint;
        unsigned tie;
        CHECK(glm53_fp8_dynamic_decode((uint8_t)lower, &a) ==
              GLM53_FP8_DYNAMIC_OK);
        CHECK(glm53_fp8_dynamic_decode((uint8_t)(lower + 1u), &b) ==
              GLM53_FP8_DYNAMIC_OK);
        midpoint = (a + b) * 0.5f;
        tie = (lower & 1u) == 0u ? lower : lower + 1u;
        CHECK(encode_is(midpoint, tie));
        CHECK(encode_is(nextafterf(midpoint, -INFINITY), lower));
        CHECK(encode_is(nextafterf(midpoint, INFINITY), lower + 1u));
        CHECK(encode_is(-midpoint, tie | 0x80u));
    }
    return true;
}

static bool test_groups_floor_strides_and_dequant(void) {
    enum { TOKENS = 2, COLS = 256, XSTRIDE = 259, QSTRIDE = 261,
           SSTRIDE = 4 };
    float x[TOKENS * XSTRIDE];
    uint8_t q[TOKENS * QSTRIDE];
    float scales[TOKENS * SSTRIDE];
    float max_error = 0.0f;
    size_t t;
    size_t c;
    memset(x, 0, sizeof(x));
    memset(q, 0xcd, sizeof(q));
    for (t = 0u; t < TOKENS * SSTRIDE; ++t) scales[t] = -77.0f;

    /* Token 0 proves vLLM's default eps=1e-10 floor and signed-zero path. */
    { const uint32_t negative_zero = UINT32_C(0x80000000);
      memcpy(&x[0], &negative_zero, sizeof(negative_zero)); }
    x[1] = GLM53_FP8_DYNAMIC_SCALE_FLOOR;
    /* Its second group has scale 2 and exercises both saturation endpoints. */
    x[128] = 896.0f;
    x[129] = -896.0f;
    x[130] = 2.0f;
    x[131] = 1.1f; /* non-representable: make the diagnostic meaningful */

    /* Token 1 groups must have independent maxima. */
    x[XSTRIDE + 0] = 448.0f;
    x[XSTRIDE + 127] = -224.0f;
    x[XSTRIDE + 128] = 0.5f;
    x[XSTRIDE + 255] = -0.25f;

    CHECK(glm53_fp8_dynamic_quantize_f32(
        q, sizeof(q), QSTRIDE, scales, TOKENS * SSTRIDE, SSTRIDE,
        x, TOKENS * XSTRIDE, XSTRIDE, TOKENS, COLS) ==
        GLM53_FP8_DYNAMIC_OK);
    CHECK(fbits(scales[0]) == fbits(GLM53_FP8_DYNAMIC_SCALE_FLOOR));
    CHECK(fbits(scales[1]) == fbits(2.0f));
    CHECK(fbits(scales[SSTRIDE]) == fbits(1.0f));
    CHECK(fbits(scales[SSTRIDE + 1]) == fbits(0.5f / 448.0f));
    CHECK(q[0] == UINT8_C(0x80));
    CHECK(q[1] == UINT8_C(0x38));
    CHECK(q[128] == UINT8_C(0x7e));
    CHECK(q[129] == UINT8_C(0xfe));
    CHECK(q[QSTRIDE] == UINT8_C(0x7e));
    CHECK(q[QSTRIDE + 128] == UINT8_C(0x7e));
    /* Row and scale padding are untouched. */
    CHECK(q[COLS] == UINT8_C(0xcd));
    CHECK(scales[2] == -77.0f && scales[3] == -77.0f);

    for (t = 0u; t < TOKENS; ++t) {
        for (c = 0u; c < COLS; ++c) {
            float decoded;
            const float scale = scales[t * SSTRIDE + c / 128u];
            const float original = x[t * XSTRIDE + c];
            float error;
            CHECK(glm53_fp8_dynamic_decode(q[t * QSTRIDE + c], &decoded) ==
                  GLM53_FP8_DYNAMIC_OK);
            error = fabsf(decoded * scale - original);
            if (error > max_error) max_error = error;
            /* Half of the largest E4M3 spacing is 16 scale units. */
            CHECK(error <= 16.0f * scale);
        }
    }
    printf("glm53 fp8 dynamic dequant max_abs_error=%g\n", (double)max_error);
    return true;
}


static bool test_vllm_scale_rounding(void) {
    float x[128] = {0.0f};
    uint8_t q[128];
    float scale = -1.0f;
    const float absmax = 7.5660152f;
    x[37] = -absmax;
    CHECK(glm53_fp8_dynamic_quantize_f32(
        q, sizeof(q), 128u, &scale, 1u, 1u,
        x, 128u, 128u, 1u, 128u) == GLM53_FP8_DYNAMIC_OK);
    CHECK(fbits(scale) == fbits(absmax * GLM53_FP8_DYNAMIC_INV_MAX));
    CHECK(q[37] == UINT8_C(0xfe));
    return true;
}

static bool test_fail_closed_geometry_and_nonfinite(void) {
    float x[256];
    uint8_t q[256];
    float scales[2];
    size_t i;
    memset(x, 0, sizeof(x));
    memset(q, 0x5a, sizeof(q));
    scales[0] = 31.0f;
    scales[1] = 32.0f;

    x[255] = NAN;
    CHECK(glm53_fp8_dynamic_quantize_f32(q, 256u, 256u, scales, 2u, 2u,
        x, 256u, 256u, 1u, 256u) == GLM53_FP8_DYNAMIC_NONFINITE_VALUE);
    for (i = 0u; i < 256u; ++i) CHECK(q[i] == UINT8_C(0x5a));
    CHECK(scales[0] == 31.0f && scales[1] == 32.0f);
    x[255] = 0.0f;

    CHECK(glm53_fp8_dynamic_quantize_f32(q, 255u, 256u, scales, 2u, 2u,
        x, 256u, 256u, 1u, 256u) == GLM53_FP8_DYNAMIC_BUFFER_TOO_SMALL);
    CHECK(glm53_fp8_dynamic_quantize_f32(q, 256u, 127u, scales, 2u, 2u,
        x, 256u, 256u, 1u, 256u) == GLM53_FP8_DYNAMIC_INVALID_ARGUMENT);
    CHECK(glm53_fp8_dynamic_quantize_f32(q, 256u, 256u, scales, 2u, 2u,
        x, 256u, 256u, 1u, 255u) == GLM53_FP8_DYNAMIC_INVALID_ARGUMENT);
    CHECK(glm53_fp8_dynamic_quantize_f32(q, 256u, 256u, scales, 2u, 2u,
        x, 256u, 256u, 0u, 256u) == GLM53_FP8_DYNAMIC_INVALID_ARGUMENT);
    CHECK(glm53_fp8_dynamic_quantize_f32(q, 256u, SIZE_MAX, scales, 2u,
        SIZE_MAX, x, 256u, SIZE_MAX, 2u, 256u) ==
        GLM53_FP8_DYNAMIC_DIMENSION_OVERFLOW);
    CHECK(glm53_fp8_dynamic_quantize_f32((uint8_t *)x, sizeof(x), 256u,
        scales, 2u, 2u, x, 256u, 256u, 1u, 256u) ==
        GLM53_FP8_DYNAMIC_INVALID_ARGUMENT);
    CHECK(glm53_fp8_dynamic_quantize_f32(q, 256u, 256u, (float *)q, 2u,
        2u, x, 256u, 256u, 1u, 256u) ==
        GLM53_FP8_DYNAMIC_INVALID_ARGUMENT);
    for (i = 0u; i < 256u; ++i) CHECK(q[i] == UINT8_C(0x5a));
    CHECK(scales[0] == 31.0f && scales[1] == 32.0f);
    return true;
}

int main(void) {
    if (!test_representable_roundtrip() ||
        !test_torch_goldens_and_special_policy() ||
        !test_every_midpoint() ||
        !test_vllm_scale_rounding() ||
        !test_groups_floor_strides_and_dequant() ||
        !test_fail_closed_geometry_and_nonfinite()) return 1;
    puts("glm53 fp8 dynamic tests: ok");
    return 0;
}
