#include "k3_q8_codec.h"
#include "k3_static_store.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", message);                       \
            failures++;                                                     \
        }                                                                   \
    } while (0)

static uint16_t float_to_bf16_exact(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    CHECK((bits & UINT32_C(0xffff)) == 0u,
          "test value must be exactly representable as BF16");
    return (uint16_t)(bits >> 16u);
}

static void test_zero_block(void) {
    uint16_t input[K3_Q8_CODEC_BLOCK] = {0};
    int8_t output[K3_Q8_CODEC_BLOCK];
    float scale = 0.0f;
    memset(output, 1, sizeof(output));
    CHECK(k3_q8_quantize_bf16_block(input, output, &scale),
          "zero block quantizes");
    CHECK(scale == 1.0f, "zero block uses unit scale");
    for (uint32_t index = 0u; index < K3_Q8_CODEC_BLOCK; index++) {
        CHECK(output[index] == 0, "zero block output is zero");
    }
}

static void test_round_to_nearest_even(void) {
    uint16_t input[K3_Q8_CODEC_BLOCK] = {0};
    static const float values[] = {
        127.0f, 0.5f, 1.5f, 2.5f, 3.5f,
        -0.5f, -1.5f, -2.5f, -3.5f,
    };
    static const int8_t expected[] = {
        127, 0, 2, 2, 4, 0, -2, -2, -4,
    };
    for (uint32_t index = 0u;
         index < sizeof(values) / sizeof(values[0]); index++) {
        input[index] = float_to_bf16_exact(values[index]);
    }
    int8_t output[K3_Q8_CODEC_BLOCK];
    float scale = 0.0f;
    CHECK(k3_q8_quantize_bf16_block(input, output, &scale),
          "rounding block quantizes");
    CHECK(scale == 1.0f, "rounding block scale is exact");
    for (uint32_t index = 0u;
         index < sizeof(expected) / sizeof(expected[0]); index++) {
        CHECK(output[index] == expected[index],
              "rounding is nearest-even");
    }
}

static void test_scaled_block(void) {
    uint16_t input[K3_Q8_CODEC_BLOCK] = {0};
    input[0] = float_to_bf16_exact(63.5f);
    input[1] = float_to_bf16_exact(-63.5f);
    input[2] = float_to_bf16_exact(31.75f);
    int8_t output[K3_Q8_CODEC_BLOCK];
    float scale = 0.0f;
    CHECK(k3_q8_quantize_bf16_block(input, output, &scale),
          "scaled block quantizes");
    CHECK(scale == 0.5f, "scaled block scale");
    CHECK(output[0] == 127 && output[1] == -127 && output[2] == 64,
          "scaled block values");
}

static void test_invalid_values(void) {
    uint16_t input[K3_Q8_CODEC_BLOCK] = {0};
    int8_t output[K3_Q8_CODEC_BLOCK];
    float scale;
    input[17] = UINT16_C(0x7f80);
    CHECK(!k3_q8_quantize_bf16_block(input, output, &scale),
          "infinite source is rejected");
    CHECK(!k3_q8_quantize_bf16_block(NULL, output, &scale),
          "null input is rejected");
}

static void test_histogram(void) {
    k3_q8_histogram histogram;
    k3_q8_histogram_reset(&histogram);
    int8_t one_symbol[16] = {0};
    CHECK(k3_q8_histogram_add(&histogram, one_symbol,
                              sizeof(one_symbol)),
          "one-symbol histogram updates");
    CHECK(histogram.total == sizeof(one_symbol),
          "one-symbol histogram total");
    CHECK(k3_q8_histogram_entropy(&histogram) == 0.0,
          "one-symbol entropy is zero");

    k3_q8_histogram_reset(&histogram);
    int8_t two_symbols[16];
    for (uint32_t index = 0u; index < 16u; index++) {
        two_symbols[index] = (int8_t)(index & 1u);
    }
    CHECK(k3_q8_histogram_add(&histogram, two_symbols,
                              sizeof(two_symbols)),
          "two-symbol histogram updates");
    CHECK(fabs(k3_q8_histogram_entropy(&histogram) - 1.0) < 1e-12,
          "balanced two-symbol entropy is one bit");
    CHECK(!k3_q8_histogram_add(NULL, two_symbols, sizeof(two_symbols)),
          "null histogram is rejected");
}

static void test_static_classification(void) {
    k3_st_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.name =
        "language_model.layers.1.self_attn.q_proj.weight";
    tensor.dtype = K3_ST_DTYPE_BF16;
    tensor.ndim = 2u;
    tensor.shape[0] = 128u;
    tensor.shape[1] = 256u;
    CHECK(k3_static_weight_is_text_tensor(&tensor),
          "projection is a static text tensor");
    CHECK(k3_static_weight_is_q8_candidate(&tensor),
          "projection is a Q8 candidate");

    tensor.name =
        "language_model.layers.1.block_sparse_moe.experts.0.weight";
    CHECK(!k3_static_weight_is_text_tensor(&tensor),
          "routed expert is excluded from static text");
    CHECK(!k3_static_weight_is_q8_candidate(&tensor),
          "routed expert is excluded from Q8");

    tensor.name = "language_model.embed_tokens.weight";
    CHECK(!k3_static_weight_is_text_tensor(&tensor),
          "embedding is excluded from static text");
    CHECK(!k3_static_weight_is_q8_candidate(&tensor),
          "embedding is excluded from Q8");

    tensor.name =
        "language_model.layers.1.block_sparse_moe.gate.weight";
    CHECK(k3_static_weight_is_text_tensor(&tensor),
          "router remains in static text");
    CHECK(!k3_static_weight_is_q8_candidate(&tensor),
          "router is excluded from Q8");

    tensor.name =
        "language_model.layers.3.self_attn.kv_b_proj.weight";
    CHECK(!k3_static_weight_is_q8_candidate(&tensor),
          "MLA kv_b is excluded from Q8");
}

int main(void) {
    test_zero_block();
    test_round_to_nearest_even();
    test_scaled_block();
    test_invalid_values();
    test_histogram();
    test_static_classification();
    if (failures != 0) {
        fprintf(stderr, "K3 static Q8 codec: %d failure(s)\n", failures);
        return 1;
    }
    printf("K3 static Q8 codec: PASS\n");
    return 0;
}
