#include "glm53_arch_math.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)

static bool closef(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static bool test_swiglu(void) {
    const float gate[4] = {-100.0f, -1.0f, 1.0f, 100.0f};
    const float up[4] = {-100.0f, -2.0f, 3.0f, 100.0f};
    /* Independent high-precision goldens for limit=2. */
    const float want[4] = {7.44015195e-42f, 0.537882843f,
                           1.46211716f, 3.52318831f};
    float out[4] = {-9.0f, -9.0f, -9.0f, -9.0f};
    float bad_gate[4];
    size_t i;
    CHECK(glm53_limited_swiglu_f32(out, gate, up, 4u, 2.0f) ==
          GLM53_ARCH_MATH_OK);
    for (i = 0u; i < 4u; ++i) CHECK(closef(out[i], want[i], 5e-7f));
    memcpy(bad_gate, gate, sizeof(gate));
    bad_gate[3] = NAN;
    for (i = 0u; i < 4u; ++i) out[i] = -7.0f;
    CHECK(glm53_limited_swiglu_f32(out, bad_gate, up, 4u, 2.0f) ==
          GLM53_ARCH_MATH_NONFINITE_VALUE);
    for (i = 0u; i < 4u; ++i) CHECK(out[i] == -7.0f);
    return true;
}

static bool test_router(void) {
    const float logits[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    const float bias[5] = {0.0f, 0.5f, 0.0f, -0.4f, -1.0f};
    const float want[3] = {0.268941421f, 0.5f, 0.731058579f};
    size_t index[3] = {99u, 99u, 99u};
    float weight[3] = {-1.0f, -1.0f, -1.0f};
    float tie_logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tie_bias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t i;
    CHECK(glm53_router_topk_f32(index, weight, logits, bias, 5u, 3u, 1.5f) ==
          GLM53_ARCH_MATH_OK);
    CHECK(index[0] == 1u && index[1] == 2u && index[2] == 3u);
    for (i = 0u; i < 3u; ++i) CHECK(closef(weight[i], want[i], 2e-7f));
    CHECK(closef(weight[0] + weight[1] + weight[2], 1.5f, 2e-7f));
    CHECK(glm53_router_topk_f32(index, weight, tie_logits, tie_bias,
                                4u, 3u, 1.0f) == GLM53_ARCH_MATH_OK);
    CHECK(index[0] == 0u && index[1] == 1u && index[2] == 2u);
    tie_logits[3] = INFINITY;
    index[0] = 77u; weight[0] = -8.0f;
    CHECK(glm53_router_topk_f32(index, weight, tie_logits, tie_bias,
                                4u, 3u, 1.0f) ==
          GLM53_ARCH_MATH_NONFINITE_VALUE);
    CHECK(index[0] == 77u && weight[0] == -8.0f);
    return true;
}

static bool test_router_fixed(void) {
    float logits[288];
    float bias[288];
    size_t index[8];
    float weight[8];
    size_t i;
    for (i = 0u; i < 288u; ++i) { logits[i] = 0.0f; bias[i] = (float)i; }
    CHECK(glm53_router_top8_288_f32(index, weight, logits, bias, 2.0f) ==
          GLM53_ARCH_MATH_OK);
    for (i = 0u; i < 8u; ++i) {
        CHECK(index[i] == 287u - i);
        CHECK(closef(weight[i], 0.25f, 1e-7f));
    }
    return true;
}

static bool test_mhc_weights(void) {
    float mix[24], base[24], scale[3] = {0.5f, -0.75f, 1.25f};
    const float want_pre[4] = {0.47751618f, 0.45760306f,
                               0.53743085f, 0.46754669f};
    const float want_post[4] = {0.88303778f, 1.15377052f,
                                0.71097689f, 1.29131261f};
    const float want_c[16] = {
        0.10221775f, 0.43268412f, 0.07427372f, 0.39082342f,
        0.20019646f, 0.31176462f, 0.20642771f, 0.28161020f,
        0.29953490f, 0.17162236f, 0.30885812f, 0.21998361f,
        0.39804988f, 0.08392790f, 0.41043945f, 0.10758177f
    };
    float pre[4], post[4], c[16];
    size_t i, src, dst;
    for (i = 0u; i < 24u; ++i) {
        mix[i] = (i & 1u ? -1.0f : 1.0f) * (float)(i + 1u) / 10.0f;
        base[i] = ((float)(i % 5u) - 2.0f) * 0.07f;
    }
    CHECK(glm53_mhc_weights4_f32(pre, post, c, mix, base, scale, 1e-6f) ==
          GLM53_ARCH_MATH_OK);
    for (i = 0u; i < 4u; ++i) {
        CHECK(closef(pre[i], want_pre[i], 3e-6f));
        CHECK(closef(post[i], want_post[i], 3e-6f));
    }
    for (i = 0u; i < 16u; ++i) CHECK(closef(c[i], want_c[i], 4e-6f));
    for (src = 0u; src < 4u; ++src) {
        float sum = 0.0f;
        for (dst = 0u; dst < 4u; ++dst) sum += c[src * 4u + dst];
        CHECK(closef(sum, 0.999999f, 4e-6f));
    }
    for (dst = 0u; dst < 4u; ++dst) {
        float sum = 0.0f;
        for (src = 0u; src < 4u; ++src) sum += c[src * 4u + dst];
        CHECK(closef(sum, 0.999999f, 4e-6f));
    }
    mix[23] = NAN; pre[0] = -22.0f; post[0] = -22.0f; c[0] = -22.0f;
    CHECK(glm53_mhc_weights4_f32(pre, post, c, mix, base, scale, 1e-6f) ==
          GLM53_ARCH_MATH_NONFINITE_VALUE);
    CHECK(pre[0] == -22.0f && post[0] == -22.0f && c[0] == -22.0f);
    return true;
}

static bool test_mhc_apply(void) {
    const float streams[12] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    const float pre[4] = {0.1f,0.2f,0.3f,0.4f};
    const float post[4] = {1,2,3,4};
    const float c[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    const float branch[3] = {0.5f,1.0f,1.5f};
    float collapsed[3];
    float out[12];
    float inplace[12];
    size_t i;
    CHECK(glm53_mhc_collapse4_f32(collapsed, streams, 3u, pre) ==
          GLM53_ARCH_MATH_OK);
    CHECK(closef(collapsed[0], 7.0f, 1e-6f));
    CHECK(closef(collapsed[1], 8.0f, 1e-6f));
    CHECK(closef(collapsed[2], 9.0f, 1e-6f));
    CHECK(glm53_mhc_post_apply4_f32(out, streams, branch, 3u, post, c) ==
          GLM53_ARCH_MATH_OK);
    for (i = 0u; i < 12u; ++i) {
        const size_t dst = i / 3u, d = i % 3u;
        CHECK(closef(out[i], streams[i] + post[dst] * branch[d], 1e-6f));
    }
    memcpy(inplace, streams, sizeof(streams));
    CHECK(glm53_mhc_post_apply4_f32(inplace, inplace, branch, 3u, post, c) ==
          GLM53_ARCH_MATH_OK);
    for (i = 0u; i < 12u; ++i) CHECK(inplace[i] == out[i]);
    return true;
}

int main(void) {
    if (!test_swiglu() || !test_router() || !test_router_fixed() ||
        !test_mhc_weights() || !test_mhc_apply()) return 1;
    puts("glm53 arch math tests: ok");
    return 0;
}
