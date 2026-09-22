/*
 * M1 norm, activation, accumulation and residual contract.
 *
 * Each case also shows that the plausible-looking alternative gives a
 * different answer, so the contract distinctions are demonstrated rather than
 * asserted: single-rounding RMSNorm, GLM's clamped SwiGLU, and per-expert
 * BF16 rounding are each measurably wrong here.
 */
#include "mimo26_ops.h"
#include "glm53_arch_math.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define COUNT 4096u

static uint32_t next_random(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float unit_random(uint32_t *state)
{
    return (float)(next_random(state) >> 8) / (float)(1u << 24);
}

static void check_bf16_roundtrip(void)
{
    /* Exactly representable values survive a round trip. */
    const float exact[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 256.0f, -0.125f};
    for (size_t i = 0; i < sizeof exact / sizeof exact[0]; i++) {
        assert(mimo26_bf16_to_f32(mimo26_f32_to_bf16(exact[i])) == exact[i]);
    }

    /* Round-to-nearest-even at the tie, matching hip_bfloat16 on gfx1151. */
    struct { uint32_t bits; uint16_t expected; } cases[] = {
        {0x3F808000u, 0x3F80u}, /* tie, even upper -> stays */
        {0x3F818000u, 0x3F82u}, /* tie, odd upper  -> rounds up */
        {0x3F808001u, 0x3F81u}, /* above tie       -> rounds up */
        {0x3F807FFFu, 0x3F80u}, /* below tie       -> stays */
        {0xBF818000u, 0xBF82u}, /* negative tie, odd upper */
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        float value;
        memcpy(&value, &cases[i].bits, sizeof value);
        assert(mimo26_f32_to_bf16(value) == cases[i].expected);
    }

    /* Signed zero is preserved, and NaN does not decay into infinity. */
    assert(mimo26_f32_to_bf16(-0.0f) == 0x8000u);
    assert(mimo26_f32_to_bf16(0.0f) == 0x0000u);
    const uint32_t nan_bits = 0x7F800001u; /* NaN with only a low mantissa bit */
    float nan_value;
    memcpy(&nan_value, &nan_bits, sizeof nan_value);
    assert(isnan(mimo26_bf16_to_f32(mimo26_f32_to_bf16(nan_value))));
    printf("  ok  BF16 conversion is RNE, preserves signed zero and NaN\n");
}

static void check_rmsnorm(void)
{
    uint32_t state = 20260921u;
    uint16_t input[COUNT];
    uint16_t weight[COUNT];
    uint16_t out[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        input[i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 6.0f);
        weight[i] = mimo26_f32_to_bf16(0.5f + unit_random(&state));
    }

    assert(mimo26_rmsnorm_bf16(out, input, weight, COUNT,
                               MIMO26_LAYERNORM_EPSILON) == MIMO26_OPS_OK);

    /* Reference in double, following the same cast sequence. */
    double sum_squares = 0.0;
    for (size_t i = 0; i < COUNT; i++) {
        const double x = (double)mimo26_bf16_to_f32(input[i]);
        sum_squares += x * x;
    }
    const double variance = sum_squares / (double)COUNT;
    const double inverse = 1.0 / sqrt(variance + (double)MIMO26_LAYERNORM_EPSILON);

    size_t differs_from_single_rounding = 0;
    for (size_t i = 0; i < COUNT; i++) {
        const float x = mimo26_bf16_to_f32(input[i]);
        const float w = mimo26_bf16_to_f32(weight[i]);

        /* Contract: round the normalized value, then multiply. */
        const float rounded =
            mimo26_bf16_to_f32(mimo26_f32_to_bf16((float)((double)x * inverse)));
        const uint16_t expected = mimo26_f32_to_bf16(w * rounded);

        /* The natural-looking alternative: multiply in f32, round once. */
        const uint16_t single =
            mimo26_f32_to_bf16((float)((double)w * (double)x * inverse));

        /* Allow one ulp against the double-precision inverse. */
        const int delta = (int)out[i] - (int)expected;
        assert(delta <= 1 && delta >= -1);
        if (out[i] != single) {
            differs_from_single_rounding++;
        }
    }
    /* If these never differed, the cast ordering would not matter. */
    assert(differs_from_single_rounding > 0);
    printf("  ok  RMSNorm matches the double-rounding contract; "
           "%zu/%u elements differ from single rounding\n",
           differs_from_single_rounding, COUNT);

    /* Rejection leaves the output untouched. */
    uint16_t guard[8];
    uint16_t small_in[8];
    uint16_t small_w[8];
    for (size_t i = 0; i < 8u; i++) {
        guard[i] = 0xDEADu;
        small_in[i] = mimo26_f32_to_bf16(1.0f);
        small_w[i] = mimo26_f32_to_bf16(1.0f);
    }
    small_in[3] = 0x7FC0u; /* NaN */
    assert(mimo26_rmsnorm_bf16(guard, small_in, small_w, 8u,
                               MIMO26_LAYERNORM_EPSILON) ==
           MIMO26_OPS_NONFINITE_VALUE);
    for (size_t i = 0; i < 8u; i++) {
        assert(guard[i] == 0xDEADu);
    }
    assert(mimo26_rmsnorm_bf16(guard, small_in, small_w, 0u,
                               MIMO26_LAYERNORM_EPSILON) ==
           MIMO26_OPS_INVALID_ARGUMENT);
    printf("  ok  RMSNorm refuses NaN input and zero count without writing\n");
}

static void check_silu_product(void)
{
    uint32_t state = 4242u;
    float gate[COUNT];
    float up[COUNT];
    float out[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        gate[i] = (unit_random(&state) - 0.5f) * 20.0f; /* wide enough to clamp */
        up[i] = (unit_random(&state) - 0.5f) * 20.0f;
    }
    assert(mimo26_silu_product_f32(out, gate, up, COUNT) == MIMO26_OPS_OK);

    double worst = 0.0;
    for (size_t i = 0; i < COUNT; i++) {
        const double sigmoid = 1.0 / (1.0 + exp(-(double)gate[i]));
        const double reference = (double)gate[i] * sigmoid * (double)up[i];
        const double scale = fabs(reference) > 1e-9 ? fabs(reference) : 1.0;
        const double relative = fabs((double)out[i] - reference) / scale;
        if (relative > worst) {
            worst = relative;
        }
    }
    assert(worst < 1e-5);
    printf("  ok  silu(gate)*up matches a double reference (max rel %.2e)\n",
           worst);

    /* GLM's limited SwiGLU clamps, so it is not interchangeable. */
    float limited[COUNT];
    assert(glm53_limited_swiglu_f32(limited, gate, up, COUNT, 7.0f) ==
           GLM53_ARCH_MATH_OK);
    size_t clamped = 0;
    for (size_t i = 0; i < COUNT; i++) {
        if (fabsf(limited[i] - out[i]) > 1e-4f) {
            clamped++;
        }
    }
    assert(clamped > 0);
    printf("  ok  GLM's limited SwiGLU differs on %zu/%u elements, so it is "
           "not reusable here\n", clamped, COUNT);
}

static void check_expert_accumulation(void)
{
    /* Eight experts, as top-8 selects, each contributing a BF16 output. */
    uint32_t state = 909u;
    uint16_t outputs[8][64];
    float weights[8];
    float accumulator[64];
    uint16_t final[64];

    float weight_sum = 0.0f;
    for (size_t e = 0; e < 8u; e++) {
        weights[e] = unit_random(&state);
        weight_sum += weights[e];
    }
    for (size_t e = 0; e < 8u; e++) {
        weights[e] /= weight_sum;
        for (size_t i = 0; i < 64u; i++) {
            outputs[e][i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 4.0f);
        }
    }

    memset(accumulator, 0, sizeof accumulator);
    for (size_t e = 0; e < 8u; e++) {
        assert(mimo26_expert_accumulate_f32(accumulator, outputs[e], weights[e],
                                            64u) == MIMO26_OPS_OK);
    }
    assert(mimo26_expert_finalize_bf16(final, accumulator, 64u) ==
           MIMO26_OPS_OK);

    size_t differs_from_bf16_steps = 0;
    for (size_t i = 0; i < 64u; i++) {
        double reference = 0.0;
        for (size_t e = 0; e < 8u; e++) {
            reference += (double)mimo26_bf16_to_f32(outputs[e][i]) *
                         (double)weights[e];
        }
        const uint16_t expected = mimo26_f32_to_bf16((float)reference);
        const int delta = (int)final[i] - (int)expected;
        assert(delta <= 1 && delta >= -1);

        /* The alternative: round to BF16 after every expert. */
        float stepwise = 0.0f;
        for (size_t e = 0; e < 8u; e++) {
            stepwise = mimo26_bf16_to_f32(mimo26_f32_to_bf16(
                stepwise + mimo26_bf16_to_f32(outputs[e][i]) * weights[e]));
        }
        if (mimo26_f32_to_bf16(stepwise) != final[i]) {
            differs_from_bf16_steps++;
        }
    }
    assert(differs_from_bf16_steps > 0);
    printf("  ok  expert sum accumulates in f32 with one final cast; "
           "%zu/64 elements differ from per-expert rounding\n",
           differs_from_bf16_steps);
}

static void check_residual(void)
{
    uint16_t residual[16];
    uint16_t delta[16];
    uint16_t out[16];
    for (size_t i = 0; i < 16u; i++) {
        residual[i] = mimo26_f32_to_bf16(1.0f + (float)i);
        delta[i] = mimo26_f32_to_bf16(0.25f * (float)i);
    }
    assert(mimo26_residual_add_bf16(out, residual, delta, 16u) ==
           MIMO26_OPS_OK);
    for (size_t i = 0; i < 16u; i++) {
        const float expected = mimo26_bf16_to_f32(residual[i]) +
                               mimo26_bf16_to_f32(delta[i]);
        assert(out[i] == mimo26_f32_to_bf16(expected));
    }
    printf("  ok  residual add is BF16 with a single rounding\n");
}

int main(void)
{
    check_bf16_roundtrip();
    check_rmsnorm();
    check_silu_product();
    check_expert_accumulation();
    check_residual();
    printf("test_mimo26_ops: ok\n");
    return 0;
}
