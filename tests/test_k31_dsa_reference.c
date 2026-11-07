#include "k31_dsa_reference.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition, message)                                            \
    do {                                                                     \
        if (!(condition)) {                                                  \
            fprintf(stderr, "FAIL:%d: %s\n", __LINE__, message);           \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static void check_near(float actual, double expected, double tolerance,
                       const char *message) {
    if (!isfinite(actual) || fabs((double)actual - expected) > tolerance) {
        fprintf(stderr, "FAIL: %s: actual=%.9g expected=%.9g\n", message,
                (double)actual, expected);
        failures++;
    }
}

static void test_bias_free_projection(void) {
    const float input[] = {1.0f, 2.0f};
    const float weight[] = {0.5f, -1.0f, 2.0f, 1.0f, 0.0f, 3.0f};
    float output[3] = {0.0f, 0.0f, 0.0f};
    CHECK(k31_dsa_linear(input, 1u, weight, 2u, 3u, output),
          "bias-free projection succeeds");
    check_near(output[0], -1.5, 1e-6, "projection row 0");
    check_near(output[1], 4.0, 1e-6, "projection row 1");
    check_near(output[2], 6.0, 1e-6, "projection row 2");
    CHECK(!k31_dsa_linear(input, 1u, weight, 0u, 3u, output),
          "zero input width is rejected");
}

static void test_key_layernorm_and_nope(void) {
    float keys[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float beta[] = {0.0f, 0.0f, 0.0f, 0.0f};
    CHECK(k31_dsa_normalize_keys(keys, 1u, 4u, 0u, gamma, beta, 0.0, keys),
          "in-place key LayerNorm succeeds");
    const double scale = 1.0 / sqrt(1.25);
    check_near(keys[0], -1.5 * scale, 1e-6, "LayerNorm element 0");
    check_near(keys[1], -0.5 * scale, 1e-6, "LayerNorm element 1");
    check_near(keys[2], 0.5 * scale, 1e-6, "LayerNorm element 2");
    check_near(keys[3], 1.5 * scale, 1e-6, "LayerNorm element 3");

    const float raw[] = {1.0f, 2.0f};
    const float gamma_two[] = {2.0f, 2.0f};
    const float beta_one[] = {1.0f, 1.0f};
    float normalized[2];
    CHECK(k31_dsa_normalize_keys(raw, 1u, 2u, 0u, gamma_two, beta_one,
                                 0.0, normalized),
          "scaled and shifted LayerNorm succeeds");
    check_near(normalized[0], -1.0, 1e-6, "LayerNorm gamma/beta low");
    check_near(normalized[1], 3.0, 1e-6, "LayerNorm gamma/beta high");
    CHECK(!k31_dsa_normalize_keys(raw, 1u, 2u, 1u, gamma_two, beta_one,
                                  0.0, normalized),
          "released NoPE geometry rejects index RoPE");
}

static void test_weighted_relu_scores_and_scales(void) {
    const float queries[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float keys[] = {2.0f, 3.0f, -2.0f, 3.0f};
    const float weights[] = {2.0f, -4.0f};
    const float query_scale[] = {0.5f, 0.25f};
    const float key_scales[] = {2.0f, 3.0f};
    float scores[2];
    CHECK(k31_dsa_scores(queries, 1u, 2u, 2u, keys, 2u, weights,
                         query_scale, key_scales, scores),
          "indexer scores succeed");
    check_near(scores[0], -1.0, 1e-6,
               "ReLU, signed head weights, per-head query scale, and key scale");
    check_near(scores[1], -4.5, 1e-6,
               "negative head dot is clamped before weighting");

    CHECK(k31_dsa_scores(queries, 1u, 2u, 2u, keys, 2u, weights,
                         NULL, NULL, scores),
          "unit scales are the default");
    check_near(scores[0], -4.0, 1e-6, "unit-scale score 0");
    check_near(scores[1], -6.0, 1e-6, "unit-scale score 1");
    CHECK(!k31_dsa_scores(queries, 1u, 0u, 2u, keys, 2u, weights,
                          NULL, NULL, scores),
          "zero index heads are rejected");
}

static void test_topk_boundaries(void) {
    enum { MAX_KEYS = 2050, TOPK = 2048 };
    static float scores[MAX_KEYS];
    static int32_t selected[TOPK];
    static const size_t lengths[] = {0u, 1u, 2047u, 2048u, 2049u, 2050u};

    for (size_t test = 0u; test < sizeof(lengths) / sizeof(lengths[0]); test++) {
        const size_t length = lengths[test];
        for (size_t i = 0u; i < length; i++) scores[i] = (float)i;
        CHECK(k31_dsa_select_topk(scores, 1u, MAX_KEYS, MAX_KEYS, NULL,
                                  &length, TOPK, selected),
              "boundary top-k succeeds");
        const size_t valid = length < TOPK ? length : TOPK;
        for (size_t slot = 0u; slot < valid; slot++) {
            if (selected[slot] != (int32_t)(length - 1u - slot)) {
                fprintf(stderr,
                        "FAIL: length=%zu slot=%zu index=%d expected=%zu\n",
                        length, slot, selected[slot], length - 1u - slot);
                failures++;
                break;
            }
        }
        for (size_t slot = valid; slot < TOPK; slot++) {
            CHECK(selected[slot] == -1, "short rows have trailing -1 padding");
        }
    }
}

static void test_topk_ties_and_infinity_padding(void) {
    const float tied[] = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    int32_t selected[4];
    size_t length = 6u;
    CHECK(k31_dsa_select_topk(tied, 1u, 6u, 6u, NULL, &length, 4u, selected),
          "tied top-k succeeds");
    for (size_t i = 0u; i < 4u; i++) {
        CHECK(selected[i] == (int32_t)i, "ties choose the lower column first");
    }

    const float with_infinity[] = {-INFINITY, 2.0f, -INFINITY, 2.0f};
    length = 4u;
    CHECK(k31_dsa_select_topk(with_infinity, 1u, 4u, 4u, NULL, &length,
                              4u, selected),
          "negative-infinity padding succeeds");
    CHECK(selected[0] == 1 && selected[1] == 3 && selected[2] == -1 &&
              selected[3] == -1,
          "valid negative-infinity scores are not emitted");

    const float with_nan[] = {1.0f, NAN, 2.0f};
    length = 3u;
    CHECK(!k31_dsa_select_topk(with_nan, 1u, 3u, 3u, NULL, &length,
                               2u, selected),
          "NaN candidate scores are rejected");
}

static void test_causal_selection_and_layer_skip_offset(void) {
    const float scores[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    };
    const size_t positions[] = {0u, 1u, 3u, 5u};
    int32_t selected[4][4];
    CHECK(k31_dsa_select_causal(scores, 1u, 6u, 6u, positions, 4u,
                                selected[0]),
          "first token selects itself");
    CHECK(selected[0][0] == 0 && selected[0][1] == -1,
          "causal row zero has one token");
    CHECK(k31_dsa_select_causal(scores, 4u, 6u, 6u, positions, 4u,
                                selected[0]),
          "causal selection succeeds");
    const int32_t expected[4][4] = {
        {0, -1, -1, -1},
        {1, 0, -1, -1},
        {3, 2, 1, 0},
        {5, 4, 3, 2},
    };
    CHECK(memcmp(selected, expected, sizeof(expected)) == 0,
          "causal ranges include self and exclude every future token");

    bool reuse = true;
    for (size_t layer = 0u; layer < 93u; layer++) {
        CHECK(k31_dsa_layer_reuses_topk(layer, 1u, 1u, &reuse) && !reuse,
              "released freq=1 offset=1 runs every layer's indexer");
    }
    CHECK(k31_dsa_layer_reuses_topk(0u, 4u, 1u, &reuse) && !reuse,
          "offset one produces top-k on layer zero");
    CHECK(k31_dsa_layer_reuses_topk(1u, 4u, 1u, &reuse) && reuse,
          "frequency four reuses on layer one");
    CHECK(k31_dsa_layer_reuses_topk(4u, 4u, 1u, &reuse) && !reuse,
          "frequency four produces on layer four");
    CHECK(k31_dsa_layer_reuses_topk(1u, 4u, 2u, &reuse) && !reuse,
          "offset two delays the first shared layer");
    CHECK(!k31_dsa_layer_reuses_topk(0u, 0u, 1u, &reuse),
          "zero top-k frequency is rejected");
}

static void test_divergent_prefix_windows(void) {
    const float scores[] = {
        100.0f, 100.0f, 1.0f, 3.0f, 2.0f, 100.0f,
        100.0f, 4.0f, 3.0f, 2.0f, 1.0f, 100.0f,
    };
    const size_t starts[] = {2u, 1u};
    const size_t lengths[] = {3u, 4u};
    int32_t selected[2][4];
    CHECK(k31_dsa_select_topk(scores, 2u, 6u, 6u, starts, lengths, 4u,
                              selected[0]),
          "divergent prefix windows succeed");
    const int32_t expected[2][4] = {{3, 4, 2, -1}, {1, 2, 3, 4}};
    CHECK(memcmp(selected, expected, sizeof(expected)) == 0,
          "selection is restricted to each row's own prefix");
    const size_t bad_lengths[] = {5u, 4u};
    CHECK(!k31_dsa_select_topk(scores, 2u, 6u, 6u, starts, bad_lengths, 4u,
                               selected[0]),
          "a prefix extending beyond the score row is rejected");
}

static void test_decode_prefill_selection_equivalence(void) {
    enum { ROWS = 5, COLUMNS = 5, TOPK = 4 };
    const float scores[ROWS][COLUMNS] = {
        {1.0f, 9.0f, 9.0f, 9.0f, 9.0f},
        {1.0f, 1.0f, 9.0f, 9.0f, 9.0f},
        {3.0f, 2.0f, 2.0f, 9.0f, 9.0f},
        {0.0f, 4.0f, 4.0f, 3.0f, 9.0f},
        {-1.0f, 2.0f, 5.0f, 5.0f, 4.0f},
    };
    const size_t positions[] = {0u, 1u, 2u, 3u, 4u};
    int32_t prefill[ROWS][TOPK];
    int32_t decode[ROWS][TOPK];
    CHECK(k31_dsa_select_causal(&scores[0][0], ROWS, COLUMNS, COLUMNS,
                                positions, TOPK, &prefill[0][0]),
          "prefill causal top-k succeeds");
    for (size_t row = 0u; row < ROWS; row++) {
        const size_t length = row + 1u;
        CHECK(k31_dsa_select_topk(&scores[row][0], 1u, COLUMNS, COLUMNS,
                                  NULL, &length, TOPK, &decode[row][0]),
              "decode top-k succeeds");
    }
    CHECK(memcmp(prefill, decode, sizeof(prefill)) == 0,
          "decode and prefill select identical ordered rows");
}

static void test_sparse_mla_and_value_projection(void) {
    const float queries[] = {1.0f, 0.0f};
    const float cache[] = {1.0f, 0.0f, 50.0f, 50.0f, 2.0f, 0.0f};
    const int32_t selected[] = {2, 0, -1};
    float latent[2];
    CHECK(k31_dsa_sparse_mla_attention(queries, 1u, 1u, 2u, cache, 3u, 2u,
                                       selected, 3u, 1.0, latent),
          "sparse MLA attention succeeds");
    check_near(latent[0], (2.0 + exp(-1.0)) / (1.0 + exp(-1.0)), 1e-6,
               "sparse softmax uses only selected cache rows");
    check_near(latent[1], 0.0, 1e-6, "latent value column 1");

    const int32_t empty[] = {-1, -1, -1};
    CHECK(!k31_dsa_sparse_mla_attention(queries, 1u, 1u, 2u, cache, 3u, 2u,
                                        empty, 3u, 1.0, latent),
          "empty sparse attention rows are rejected");

    const float two_head_latent[] = {2.0f, 1.0f, 3.0f, 4.0f};
    const float value_weights[] = {1.0f, 2.0f, -1.0f, 1.0f};
    float values[2];
    CHECK(k31_dsa_project_mla_values(two_head_latent, 1u, 2u, 2u,
                                     value_weights, 1u, values),
          "per-head W_UV projection succeeds");
    check_near(values[0], 4.0, 1e-6, "W_UV head zero");
    check_near(values[1], 1.0, 1e-6, "W_UV head one");
}

static void test_output_gate_location_and_value(void) {
    float values[] = {-1.0f, 0.0f, 2.0f};
    const float gate[] = {0.0f, 2.0f, -2.0f};
    CHECK(k31_dsa_apply_output_gate(values, 1u, 1u, 3u, gate, values),
          "in-place output gate succeeds");
    check_near(values[0], -0.5, 1e-6, "sigmoid(0) output gate");
    check_near(values[1], 0.0, 1e-6, "zero gated value");
    check_near(values[2], 2.0 / (1.0 + exp(2.0)), 1e-6,
               "negative sigmoid output gate");
}

static void test_decode_prefill_sparse_attention_equivalence(void) {
    enum { ROWS = 4, QK = 2, LATENT = 2, TOPK = 3 };
    const float scores[ROWS][ROWS] = {
        {1.0f, 4.0f, 3.0f, 2.0f},
        {1.0f, 4.0f, 3.0f, 2.0f},
        {1.0f, 4.0f, 3.0f, 2.0f},
        {1.0f, 4.0f, 3.0f, 2.0f},
    };
    const float queries[ROWS][QK] = {
        {1.0f, 0.0f},
        {0.5f, 0.25f},
        {-0.5f, 1.0f},
        {1.0f, -1.0f},
    };
    const float cache[ROWS][QK] = {
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
        {2.0f, 0.0f},
    };
    const size_t positions[] = {0u, 1u, 2u, 3u};
    int32_t prefill_indices[ROWS][TOPK];
    float prefill[ROWS][LATENT];
    CHECK(k31_dsa_select_causal(&scores[0][0], ROWS, ROWS, ROWS, positions,
                                TOPK, &prefill_indices[0][0]),
          "sparse prefill selection succeeds");
    CHECK(k31_dsa_sparse_mla_attention(
              &queries[0][0], ROWS, 1u, QK, &cache[0][0], ROWS, LATENT,
              &prefill_indices[0][0], TOPK, 1.0, &prefill[0][0]),
          "sparse prefill attention succeeds");

    for (size_t row = 0u; row < ROWS; row++) {
        const size_t length = row + 1u;
        int32_t decode_indices[TOPK];
        float decode[LATENT];
        CHECK(k31_dsa_select_topk(&scores[row][0], 1u, ROWS, ROWS, NULL,
                                  &length, TOPK, decode_indices),
              "sparse decode selection succeeds");
        CHECK(k31_dsa_sparse_mla_attention(
                  &queries[row][0], 1u, 1u, QK, &cache[0][0], ROWS, LATENT,
                  decode_indices, TOPK, 1.0, decode),
              "sparse decode attention succeeds");
        for (size_t value = 0u; value < LATENT; value++) {
            check_near(decode[value], prefill[row][value], 1e-6,
                       "decode and prefill sparse MLA outputs match");
        }
    }
}

int main(void) {
    test_bias_free_projection();
    test_key_layernorm_and_nope();
    test_weighted_relu_scores_and_scales();
    test_topk_boundaries();
    test_topk_ties_and_infinity_padding();
    test_causal_selection_and_layer_skip_offset();
    test_divergent_prefix_windows();
    test_decode_prefill_selection_equivalence();
    test_sparse_mla_and_value_projection();
    test_output_gate_location_and_value();
    test_decode_prefill_sparse_attention_equivalence();
    if (failures != 0) {
        fprintf(stderr, "K3.1 DSA reference: %d failure(s)\n", failures);
        return 1;
    }
    printf("K3.1 DSA reference: PASS\n");
    return 0;
}
