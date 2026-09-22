/*
 * M1 router contract. GLM's glm53_router_topk_f32 implements the same rule
 * from a different lane, so it serves as a cross-check on the selected set
 * and the weight values; it returns descending selection order while MiMo
 * declares ascending expert id, so ordering is compared as a set.
 *
 * The declared ascending accumulation order changes the normalization
 * denominator by a fraction of an ulp. That difference is measured here
 * rather than assumed negligible.
 */
#include "mimo26_router.h"
#include "glm53_arch_math.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPERTS MIMO26_ROUTER_EXPERTS
#define TOPK MIMO26_ROUTER_TOP_K

static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static uint32_t next_random(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float unit_random(uint32_t *state)
{
    return (float)(next_random(state) >> 8) / (float)(1u << 24);
}

int main(void)
{
    /* Grouping: the no-op mask is accepted, anything else is refused. */
    assert(mimo26_router_check_grouping(1u, 1u) == MIMO26_ROUTER_OK);
    assert(mimo26_router_check_grouping(8u, 4u) ==
           MIMO26_ROUTER_UNSUPPORTED_GROUPING);
    assert(mimo26_router_check_grouping(1u, 2u) ==
           MIMO26_ROUTER_UNSUPPORTED_GROUPING);
    assert(mimo26_router_check_grouping(0u, 1u) ==
           MIMO26_ROUTER_INVALID_ARGUMENT);
    printf("  ok  grouping: n_group=topk_group=1 accepted, others refused\n");

    /* Bias steers selection, but weights come from the uncorrected sigmoid. */
    {
        float logits[EXPERTS];
        float bias[EXPERTS];
        for (size_t e = 0; e < EXPERTS; e++) {
            logits[e] = -10.0f;
            bias[e] = 0.0f;
        }
        /* Experts 100..107 have the highest raw scores. */
        for (size_t e = 100; e < 108; e++) {
            logits[e] = 1.0f;
        }
        /* Expert 7 loses on raw score but wins once its bias is added. */
        logits[7] = 0.5f;
        bias[7] = 5.0f;

        uint32_t indices[TOPK];
        float weights[TOPK];
        assert(mimo26_router_top8_256_f32(indices, weights, logits, bias) ==
               MIMO26_ROUTER_OK);
        /* Ascending expert id, and expert 7 displaced the weakest of 100..107. */
        assert(indices[0] == 7u);
        for (size_t j = 1; j < TOPK; j++) {
            assert(indices[j] > indices[j - 1]);
            assert(indices[j] >= 100u && indices[j] < 108u);
        }
        /* Expert 7's weight reflects sigmoid(0.5), not sigmoid(0.5)+5. */
        float sum_raw = 0.0f;
        for (size_t j = 0; j < TOPK; j++) {
            sum_raw += sigmoid(logits[indices[j]]);
        }
        const float expected_7 = sigmoid(0.5f) / (sum_raw + 1.0e-20f);
        assert(fabsf(weights[0] - expected_7) < 1e-6f);
        assert(weights[0] < sigmoid(1.0f) / (sum_raw + 1.0e-20f));
        printf("  ok  bias selects, uncorrected sigmoid weights (expert 7 in "
               "at w=%.6f)\n", (double)weights[0]);
    }

    /* Exact ties resolve to the lower expert id. */
    {
        float logits[EXPERTS];
        float bias[EXPERTS];
        for (size_t e = 0; e < EXPERTS; e++) {
            logits[e] = 0.0f; /* every score identical */
            bias[e] = 0.0f;
        }
        uint32_t indices[TOPK];
        float weights[TOPK];
        assert(mimo26_router_top8_256_f32(indices, weights, logits, bias) ==
               MIMO26_ROUTER_OK);
        for (size_t j = 0; j < TOPK; j++) {
            assert(indices[j] == (uint32_t)j);
            /* All equal, so every weight is exactly 1/8 up to the epsilon. */
            assert(fabsf(weights[j] - 0.125f) < 1e-6f);
        }
        printf("  ok  all-tied scores select experts 0..7 with equal weight\n");
    }

    /* Weights sum to the scale, and scale is applied after normalization. */
    {
        uint32_t state = 20260921u;
        float logits[EXPERTS];
        float bias[EXPERTS];
        for (size_t e = 0; e < EXPERTS; e++) {
            logits[e] = (unit_random(&state) - 0.5f) * 8.0f;
            bias[e] = (unit_random(&state) - 0.5f) * 0.2f;
        }
        uint32_t indices[TOPK];
        float weights[TOPK];
        assert(mimo26_router_top8_256_f32(indices, weights, logits, bias) ==
               MIMO26_ROUTER_OK);
        float total = 0.0f;
        for (size_t j = 0; j < TOPK; j++) {
            total += weights[j];
        }
        assert(fabsf(total - MIMO26_ROUTER_SCALE) < 1e-6f);

        float scaled[TOPK];
        uint32_t scaled_indices[TOPK];
        assert(mimo26_router_select_f32(scaled_indices, scaled, logits, bias,
                                        EXPERTS, TOPK, 2.5f) ==
               MIMO26_ROUTER_OK);
        float scaled_total = 0.0f;
        for (size_t j = 0; j < TOPK; j++) {
            scaled_total += scaled[j];
            assert(scaled_indices[j] == indices[j]);
        }
        assert(fabsf(scaled_total - 2.5f) < 1e-5f);
        printf("  ok  weights normalize to scale (1.0 and 2.5)\n");
    }

    /* Cross-check the selected set and weights against GLM's implementation,
     * and measure the ordering-induced denominator difference. */
    {
        uint32_t state = 7u;
        double worst_weight_delta = 0.0;
        size_t trials = 0;
        size_t set_mismatches = 0;
        for (trials = 0; trials < 2000u; trials++) {
            float logits[EXPERTS];
            float bias[EXPERTS];
            for (size_t e = 0; e < EXPERTS; e++) {
                logits[e] = (unit_random(&state) - 0.5f) * 12.0f;
                bias[e] = (unit_random(&state) - 0.5f) * 1.0f;
            }
            uint32_t mine[TOPK];
            float my_weights[TOPK];
            size_t theirs[TOPK];
            float their_weights[TOPK];
            assert(mimo26_router_select_f32(mine, my_weights, logits, bias,
                                            EXPERTS, TOPK,
                                            MIMO26_ROUTER_SCALE) ==
                   MIMO26_ROUTER_OK);
            assert(glm53_router_topk_f32(theirs, their_weights, logits, bias,
                                          EXPERTS, TOPK,
                                          MIMO26_ROUTER_SCALE) ==
                   GLM53_ARCH_MATH_OK);
            /* Same set, different order. */
            for (size_t j = 0; j < TOPK; j++) {
                bool found = false;
                for (size_t q = 0; q < TOPK; q++) {
                    if ((size_t)mine[j] == theirs[q]) {
                        const double delta =
                            fabs((double)my_weights[j] -
                                 (double)their_weights[q]);
                        if (delta > worst_weight_delta) {
                            worst_weight_delta = delta;
                        }
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    set_mismatches++;
                }
            }
        }
        assert(set_mismatches == 0);
        /* Only the summation order differs, so this must stay at ulp scale --
         * far below BF16 output granularity (2^-8 = 3.9e-3). */
        assert(worst_weight_delta < 1e-6);
        printf("  ok  %zu trials agree with GLM's router on the selected set; "
               "max weight delta %.3e from summation order alone\n",
               trials, worst_weight_delta);
    }

    /* Non-finite input is refused without touching the outputs. */
    {
        float logits[EXPERTS];
        float bias[EXPERTS];
        for (size_t e = 0; e < EXPERTS; e++) {
            logits[e] = 0.0f;
            bias[e] = 0.0f;
        }
        uint32_t indices[TOPK];
        float weights[TOPK];
        for (size_t j = 0; j < TOPK; j++) {
            indices[j] = 0xABCDu;
            weights[j] = -12345.0f;
        }
        logits[42] = NAN;
        assert(mimo26_router_select_f32(indices, weights, logits, bias, EXPERTS,
                                        TOPK, MIMO26_ROUTER_SCALE) ==
               MIMO26_ROUTER_NONFINITE_VALUE);
        logits[42] = INFINITY;
        assert(mimo26_router_select_f32(indices, weights, logits, bias, EXPERTS,
                                        TOPK, MIMO26_ROUTER_SCALE) ==
               MIMO26_ROUTER_NONFINITE_VALUE);
        logits[42] = 0.0f;
        bias[3] = NAN;
        assert(mimo26_router_select_f32(indices, weights, logits, bias, EXPERTS,
                                        TOPK, MIMO26_ROUTER_SCALE) ==
               MIMO26_ROUTER_NONFINITE_VALUE);
        for (size_t j = 0; j < TOPK; j++) {
            assert(indices[j] == 0xABCDu);
            assert(weights[j] == -12345.0f);
        }
        bias[3] = 0.0f;
        assert(mimo26_router_select_f32(NULL, weights, logits, bias, EXPERTS,
                                        TOPK, MIMO26_ROUTER_SCALE) ==
               MIMO26_ROUTER_INVALID_ARGUMENT);
        assert(mimo26_router_select_f32(indices, weights, logits, bias, EXPERTS,
                                        EXPERTS + 1u, MIMO26_ROUTER_SCALE) ==
               MIMO26_ROUTER_INVALID_ARGUMENT);
        assert(mimo26_router_select_f32(indices, weights, logits, bias, EXPERTS,
                                        TOPK, NAN) ==
               MIMO26_ROUTER_NONFINITE_VALUE);
        printf("  ok  non-finite and invalid arguments refused, outputs "
               "untouched\n");
    }

    /* F32 projection: config says bfloat16, the reference forward says F32. */
    {
        const size_t hidden_size = 64u;
        const size_t experts = 4u;
        float weight[4u * 64u];
        float hidden[64u];
        float logits[4u];
        for (size_t i = 0; i < hidden_size; i++) {
            hidden[i] = (float)(i % 5u) * 0.25f - 0.5f;
        }
        for (size_t e = 0; e < experts; e++) {
            for (size_t i = 0; i < hidden_size; i++) {
                weight[e * hidden_size + i] = (float)((e + i) % 7u) * 0.125f;
            }
        }
        assert(mimo26_router_logits_f32(logits, hidden, weight, experts,
                                        hidden_size) == MIMO26_ROUTER_OK);
        for (size_t e = 0; e < experts; e++) {
            double reference = 0.0;
            for (size_t i = 0; i < hidden_size; i++) {
                reference += (double)weight[e * hidden_size + i] *
                             (double)hidden[i];
            }
            assert(fabs((double)logits[e] - reference) < 1e-4);
        }
        hidden[0] = NAN;
        assert(mimo26_router_logits_f32(logits, hidden, weight, experts,
                                        hidden_size) ==
               MIMO26_ROUTER_NONFINITE_VALUE);
        printf("  ok  F32 router projection matches a double reference\n");
    }

    printf("test_mimo26_router: ok\n");
    return 0;
}
