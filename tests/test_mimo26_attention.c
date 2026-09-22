/*
 * M1/M2 attention primitives: layer shapes, fused QKV split with the pre-cache
 * V scale, partial RoPE, GQA head mapping, the windowed visibility rule at its
 * boundaries, and the SWA sink whose probability mass is discarded.
 */
#include "mimo26_attention.h"
#include "mimo26_ops.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t next_random(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float unit_random(uint32_t *state)
{
    return (float)(next_random(state) >> 8) / (float)(1u << 24);
}

static void check_configs(void)
{
    mimo26_attention_config config;

    /* Layer 0 is dense but its attention is global. */
    assert(mimo26_attention_config_for_layer(0u, &config) ==
           MIMO26_ATTENTION_OK);
    assert(!config.is_swa && config.kv_heads == MIMO26_GLOBAL_KV_HEADS);
    assert(config.kv_groups == 16u && config.window == 0u && !config.has_sink);
    assert(config.qkv_width == MIMO26_GLOBAL_QKV_WIDTH);
    assert(config.rope_theta == MIMO26_GLOBAL_ROPE_THETA);

    assert(mimo26_attention_config_for_layer(1u, &config) ==
           MIMO26_ATTENTION_OK);
    assert(config.is_swa && config.kv_heads == MIMO26_SWA_KV_HEADS);
    assert(config.kv_groups == 8u && config.window == MIMO26_SLIDING_WINDOW);
    assert(config.has_sink);
    assert(config.qkv_width == MIMO26_SWA_QKV_WIDTH);
    assert(config.rope_theta == MIMO26_SWA_ROPE_THETA);

    /* Layers 12, 24 and 36 are windowed: the non-periodic pattern again. */
    for (uint32_t layer = 12u; layer <= 36u; layer += 12u) {
        assert(mimo26_attention_config_for_layer(layer, &config) ==
               MIMO26_ATTENTION_OK);
        assert(config.is_swa);
    }
    for (uint32_t layer = 5u; layer <= 47u; layer += 6u) {
        assert(mimo26_attention_config_for_layer(layer, &config) ==
               MIMO26_ATTENTION_OK);
        assert(!config.is_swa && !config.has_sink);
    }
    assert(mimo26_attention_config_for_layer(48u, &config) ==
           MIMO26_ATTENTION_INVALID_ARGUMENT);

    /* Widths must match the checkpoint's fused QKV tensors exactly. */
    assert(MIMO26_QUERY_HEADS * MIMO26_QK_HEAD_DIM +
           MIMO26_GLOBAL_KV_HEADS * MIMO26_QK_HEAD_DIM +
           MIMO26_GLOBAL_KV_HEADS * MIMO26_V_HEAD_DIM ==
           MIMO26_GLOBAL_QKV_WIDTH);
    assert(MIMO26_QUERY_HEADS * MIMO26_QK_HEAD_DIM +
           MIMO26_SWA_KV_HEADS * MIMO26_QK_HEAD_DIM +
           MIMO26_SWA_KV_HEADS * MIMO26_V_HEAD_DIM == MIMO26_SWA_QKV_WIDTH);

    /* The scale comes from the QK dimension, not the V dimension. */
    const float scale = mimo26_attention_scale();
    assert(fabsf(scale - 1.0f / sqrtf(192.0f)) < 1e-9f);
    assert(fabsf(scale - 1.0f / sqrtf(128.0f)) > 1e-3f);
    printf("  ok  layer shapes, fused widths, and a 192^-0.5 softmax scale\n");
}

static void check_visibility(void)
{
    /* q - 128 < kv <= q, inclusive of the current token. */
    assert(mimo26_attention_visible(0u, 0u, MIMO26_SLIDING_WINDOW));
    assert(!mimo26_attention_visible(1u, 0u, MIMO26_SLIDING_WINDOW));
    assert(mimo26_attention_visible(0u, 127u, MIMO26_SLIDING_WINDOW));
    /* Position 128 evicts position 0 -- eviction begins here, not at 127. */
    assert(!mimo26_attention_visible(0u, 128u, MIMO26_SLIDING_WINDOW));
    assert(mimo26_attention_visible(1u, 128u, MIMO26_SLIDING_WINDOW));
    assert(!mimo26_attention_visible(1u, 129u, MIMO26_SLIDING_WINDOW));
    assert(mimo26_attention_visible(2u, 129u, MIMO26_SLIDING_WINDOW));
    assert(mimo26_attention_visible(130u, 257u, MIMO26_SLIDING_WINDOW));
    assert(!mimo26_attention_visible(129u, 257u, MIMO26_SLIDING_WINDOW));

    /* Exactly 128 visible positions once past the first window. */
    for (uint64_t q = 0; q < 600u; q++) {
        size_t count = 0;
        for (uint64_t kv = 0; kv <= q; kv++) {
            if (mimo26_attention_visible(kv, q, MIMO26_SLIDING_WINDOW)) {
                count++;
            }
        }
        const size_t expected = (q + 1u < MIMO26_SLIDING_WINDOW)
                                    ? (size_t)(q + 1u)
                                    : MIMO26_SLIDING_WINDOW;
        assert(count == expected);
    }

    /* Full attention keeps everything causal and nothing else. */
    assert(mimo26_attention_visible(0u, 100000u, 0u));
    assert(!mimo26_attention_visible(100001u, 100000u, 0u));
    printf("  ok  window visibility: 128 inclusive, eviction starts at 128\n");
}

static void check_qkv_split(void)
{
    mimo26_attention_config config;
    assert(mimo26_attention_config_for_layer(1u, &config) ==
           MIMO26_ATTENTION_OK);

    uint16_t *fused = malloc(config.qkv_width * sizeof *fused);
    uint16_t q[MIMO26_QUERY_HEADS * MIMO26_QK_HEAD_DIM];
    uint16_t k[MIMO26_SWA_KV_HEADS * MIMO26_QK_HEAD_DIM];
    uint16_t v[MIMO26_SWA_KV_HEADS * MIMO26_V_HEAD_DIM];
    assert(fused != NULL);

    uint32_t state = 31337u;
    for (size_t i = 0; i < config.qkv_width; i++) {
        fused[i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 4.0f);
    }
    assert(mimo26_attention_split_qkv(fused, &config, q, k, v) ==
           MIMO26_ATTENTION_OK);

    const size_t q_size = MIMO26_QUERY_HEADS * MIMO26_QK_HEAD_DIM;
    const size_t k_size = config.kv_heads * MIMO26_QK_HEAD_DIM;
    /* Q and K are copied verbatim. */
    assert(memcmp(q, fused, q_size * sizeof *q) == 0);
    assert(memcmp(k, fused + q_size, k_size * sizeof *k) == 0);
    /* V is pre-scaled by 0.707 before it can reach a cache. */
    for (size_t i = 0; i < config.kv_heads * MIMO26_V_HEAD_DIM; i++) {
        const float source = mimo26_bf16_to_f32(fused[q_size + k_size + i]);
        assert(v[i] == mimo26_f32_to_bf16(source * MIMO26_VALUE_SCALE));
    }
    /* 0.707 is not 1/sqrt(2): the difference is visible in BF16. */
    size_t differs_from_inv_sqrt2 = 0;
    for (size_t i = 0; i < config.kv_heads * MIMO26_V_HEAD_DIM; i++) {
        const float source = mimo26_bf16_to_f32(fused[q_size + k_size + i]);
        if (v[i] != mimo26_f32_to_bf16(source * 0.70710678f)) {
            differs_from_inv_sqrt2++;
        }
    }
    assert(differs_from_inv_sqrt2 > 0);
    printf("  ok  QKV split copies Q/K and pre-scales V by 0.707 "
           "(%zu values differ from 1/sqrt(2))\n", differs_from_inv_sqrt2);
    free(fused);
}

static void check_rope(void)
{
    uint16_t cos_table[MIMO26_ROPE_DIM];
    uint16_t sin_table[MIMO26_ROPE_DIM];

    /* Position 0 is the identity rotation. */
    assert(mimo26_rope_table(cos_table, sin_table, 0u,
                             MIMO26_SWA_ROPE_THETA) == MIMO26_ATTENTION_OK);
    for (size_t j = 0; j < MIMO26_ROPE_DIM; j++) {
        assert(mimo26_bf16_to_f32(cos_table[j]) == 1.0f);
        assert(mimo26_bf16_to_f32(sin_table[j]) == 0.0f);
    }
    uint16_t head[MIMO26_QK_HEAD_DIM];
    uint16_t original[MIMO26_QK_HEAD_DIM];
    uint32_t state = 5u;
    for (size_t d = 0; d < MIMO26_QK_HEAD_DIM; d++) {
        head[d] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 3.0f);
    }
    memcpy(original, head, sizeof original);
    assert(mimo26_rope_apply(head, cos_table, sin_table) ==
           MIMO26_ATTENTION_OK);
    assert(memcmp(head, original, sizeof original) == 0);
    printf("  ok  position 0 rotation is the identity\n");

    /* The table duplicates 32 angles across 64 entries. */
    assert(mimo26_rope_table(cos_table, sin_table, 4096u,
                             MIMO26_GLOBAL_ROPE_THETA) ==
           MIMO26_ATTENTION_OK);
    for (size_t j = 0; j < MIMO26_ROPE_PAIRS; j++) {
        assert(cos_table[j] == cos_table[j + MIMO26_ROPE_PAIRS]);
        assert(sin_table[j] == sin_table[j + MIMO26_ROPE_PAIRS]);
    }

    /* Only the first 64 coordinates move; 64..191 are the nope half. */
    memcpy(head, original, sizeof original);
    assert(mimo26_rope_apply(head, cos_table, sin_table) ==
           MIMO26_ATTENTION_OK);
    assert(memcmp(head + MIMO26_ROPE_DIM, original + MIMO26_ROPE_DIM,
                  (MIMO26_QK_HEAD_DIM - MIMO26_ROPE_DIM) * sizeof *head) == 0);
    size_t moved = 0;
    for (size_t j = 0; j < MIMO26_ROPE_DIM; j++) {
        if (head[j] != original[j]) {
            moved++;
        }
    }
    assert(moved > MIMO26_ROPE_DIM / 2u);
    printf("  ok  partial RoPE rotates 64 of 192 coordinates, %zu changed\n",
           moved);

    /* Rotation preserves each pair's magnitude, up to BF16 rounding. */
    double worst = 0.0;
    for (size_t j = 0; j < MIMO26_ROPE_PAIRS; j++) {
        const double a0 = mimo26_bf16_to_f32(original[j]);
        const double b0 = mimo26_bf16_to_f32(original[j + MIMO26_ROPE_PAIRS]);
        const double a1 = mimo26_bf16_to_f32(head[j]);
        const double b1 = mimo26_bf16_to_f32(head[j + MIMO26_ROPE_PAIRS]);
        const double before = sqrt(a0 * a0 + b0 * b0);
        const double after = sqrt(a1 * a1 + b1 * b1);
        if (before > 1e-3) {
            const double relative = fabs(after - before) / before;
            if (relative > worst) {
                worst = relative;
            }
        }
    }
    assert(worst < 0.02); /* BF16 cos/sin and outputs */
    printf("  ok  pair magnitudes preserved (max rel %.3e in BF16)\n", worst);

    /* The two thetas give different rotations at the same position. */
    uint16_t swa_cos[MIMO26_ROPE_DIM];
    uint16_t swa_sin[MIMO26_ROPE_DIM];
    assert(mimo26_rope_table(swa_cos, swa_sin, 4096u, MIMO26_SWA_ROPE_THETA) ==
           MIMO26_ATTENTION_OK);
    assert(memcmp(swa_cos, cos_table, sizeof swa_cos) != 0);
    printf("  ok  global theta 1e7 and SWA theta 1e4 differ\n");
}

static void check_decode(void)
{
    mimo26_attention_config global_config;
    mimo26_attention_config swa_config;
    assert(mimo26_attention_config_for_layer(0u, &global_config) ==
           MIMO26_ATTENTION_OK);
    assert(mimo26_attention_config_for_layer(1u, &swa_config) ==
           MIMO26_ATTENTION_OK);

    const size_t history = 200u; /* past the window, so eviction is active */
    uint32_t state = 777u;

    uint16_t query[MIMO26_QUERY_HEADS * MIMO26_QK_HEAD_DIM];
    for (size_t i = 0; i < sizeof query / sizeof query[0]; i++) {
        query[i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 2.0f);
    }
    uint16_t sink[MIMO26_QUERY_HEADS];
    for (size_t h = 0; h < MIMO26_QUERY_HEADS; h++) {
        sink[h] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 2.0f);
    }
    uint16_t out[MIMO26_QUERY_HEADS * MIMO26_V_HEAD_DIM];

    /* Windowed layer with 8 KV heads. */
    {
        const size_t kv = swa_config.kv_heads;
        uint16_t *keys = malloc(history * kv * MIMO26_QK_HEAD_DIM * sizeof *keys);
        uint16_t *values = malloc(history * kv * MIMO26_V_HEAD_DIM * sizeof *values);
        assert(keys && values);
        for (size_t i = 0; i < history * kv * MIMO26_QK_HEAD_DIM; i++) {
            keys[i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 2.0f);
        }
        for (size_t i = 0; i < history * kv * MIMO26_V_HEAD_DIM; i++) {
            values[i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 2.0f);
        }

        assert(mimo26_attention_decode(out, query, keys, values, sink,
                                       &swa_config, history, 0u,
                                       history - 1u) == MIMO26_ATTENTION_OK);

        /* No future leakage: rewriting evicted history must not move the
         * output, because positions 0..(history-129) are outside the window. */
        uint16_t reference[MIMO26_QUERY_HEADS * MIMO26_V_HEAD_DIM];
        memcpy(reference, out, sizeof reference);
        const size_t evicted = history - MIMO26_SLIDING_WINDOW;
        for (size_t t = 0; t < evicted; t++) {
            for (size_t i = 0; i < kv * MIMO26_QK_HEAD_DIM; i++) {
                keys[t * kv * MIMO26_QK_HEAD_DIM + i] =
                    mimo26_f32_to_bf16(99.0f);
            }
            for (size_t i = 0; i < kv * MIMO26_V_HEAD_DIM; i++) {
                values[t * kv * MIMO26_V_HEAD_DIM + i] =
                    mimo26_f32_to_bf16(-99.0f);
            }
        }
        assert(mimo26_attention_decode(out, query, keys, values, sink,
                                       &swa_config, history, 0u,
                                       history - 1u) == MIMO26_ATTENTION_OK);
        assert(memcmp(out, reference, sizeof reference) == 0);
        printf("  ok  windowed decode ignores all %zu evicted positions\n",
               evicted);

        /* The sink absorbs probability mass, so raising it must shrink the
         * output: the surviving probabilities sum to less than one. */
        uint16_t large_sink[MIMO26_QUERY_HEADS];
        for (size_t h = 0; h < MIMO26_QUERY_HEADS; h++) {
            large_sink[h] = mimo26_f32_to_bf16(30.0f);
        }
        uint16_t damped[MIMO26_QUERY_HEADS * MIMO26_V_HEAD_DIM];
        assert(mimo26_attention_decode(damped, query, keys, values, large_sink,
                                       &swa_config, history, 0u,
                                       history - 1u) == MIMO26_ATTENTION_OK);
        double sum_reference = 0.0;
        double sum_damped = 0.0;
        for (size_t i = 0; i < sizeof damped / sizeof damped[0]; i++) {
            sum_reference += fabs((double)mimo26_bf16_to_f32(reference[i]));
            sum_damped += fabs((double)mimo26_bf16_to_f32(damped[i]));
        }
        assert(sum_damped < sum_reference * 0.05);
        printf("  ok  a dominant sink suppresses the output to %.2f%% "
               "(its probability mass is discarded)\n",
               100.0 * sum_damped / sum_reference);

        /* A sink is mandatory on a windowed layer. */
        assert(mimo26_attention_decode(out, query, keys, values, NULL,
                                       &swa_config, history, 0u,
                                       history - 1u) ==
               MIMO26_ATTENTION_INVALID_ARGUMENT);
        free(keys);
        free(values);
    }

    /* Full-attention layer with 4 KV heads: every position stays visible. */
    {
        const size_t kv = global_config.kv_heads;
        uint16_t *keys = malloc(history * kv * MIMO26_QK_HEAD_DIM * sizeof *keys);
        uint16_t *values = malloc(history * kv * MIMO26_V_HEAD_DIM * sizeof *values);
        assert(keys && values);
        for (size_t i = 0; i < history * kv * MIMO26_QK_HEAD_DIM; i++) {
            keys[i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 2.0f);
        }
        for (size_t i = 0; i < history * kv * MIMO26_V_HEAD_DIM; i++) {
            values[i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 2.0f);
        }
        assert(mimo26_attention_decode(out, query, keys, values, NULL,
                                       &global_config, history, 0u,
                                       history - 1u) == MIMO26_ATTENTION_OK);
        uint16_t reference[MIMO26_QUERY_HEADS * MIMO26_V_HEAD_DIM];
        memcpy(reference, out, sizeof reference);

        /* Changing the oldest position must move a global layer's output. */
        for (size_t i = 0; i < kv * MIMO26_QK_HEAD_DIM; i++) {
            keys[i] = mimo26_f32_to_bf16(5.0f);
        }
        for (size_t i = 0; i < kv * MIMO26_V_HEAD_DIM; i++) {
            values[i] = mimo26_f32_to_bf16(5.0f);
        }
        assert(mimo26_attention_decode(out, query, keys, values, NULL,
                                       &global_config, history, 0u,
                                       history - 1u) == MIMO26_ATTENTION_OK);
        assert(memcmp(out, reference, sizeof reference) != 0);
        printf("  ok  full attention still sees position 0 at distance %zu\n",
               history - 1u);

        /* GQA mapping: query head h reads KV head h / kv_groups. Zeroing one
         * KV head's values must change exactly its group of query heads. */
        for (size_t i = 0; i < history * kv * MIMO26_V_HEAD_DIM; i++) {
            values[i] = mimo26_f32_to_bf16(1.0f);
        }
        assert(mimo26_attention_decode(reference, query, keys, values, NULL,
                                       &global_config, history, 0u,
                                       history - 1u) == MIMO26_ATTENTION_OK);
        const size_t target_kv_head = 2u;
        for (size_t t = 0; t < history; t++) {
            for (size_t d = 0; d < MIMO26_V_HEAD_DIM; d++) {
                values[(t * kv + target_kv_head) * MIMO26_V_HEAD_DIM + d] =
                    mimo26_f32_to_bf16(-1.0f);
            }
        }
        assert(mimo26_attention_decode(out, query, keys, values, NULL,
                                       &global_config, history, 0u,
                                       history - 1u) == MIMO26_ATTENTION_OK);
        for (size_t h = 0; h < MIMO26_QUERY_HEADS; h++) {
            const bool affected =
                (h / global_config.kv_groups) == target_kv_head;
            const int same = memcmp(out + h * MIMO26_V_HEAD_DIM,
                                    reference + h * MIMO26_V_HEAD_DIM,
                                    MIMO26_V_HEAD_DIM * sizeof *out) == 0;
            assert(affected ? !same : same);
        }
        printf("  ok  GQA maps query head h to KV head h/%zu "
               "(only heads %zu..%zu moved)\n", global_config.kv_groups,
               target_kv_head * global_config.kv_groups,
               (target_kv_head + 1u) * global_config.kv_groups - 1u);

        /* History must not claim to extend past the query position. */
        assert(mimo26_attention_decode(out, query, keys, values, NULL,
                                       &global_config, history, 0u, 10u) ==
               MIMO26_ATTENTION_INVALID_ARGUMENT);
        assert(mimo26_attention_decode(out, query, keys, values, NULL,
                                       &global_config, history, 50u, 10u) ==
               MIMO26_ATTENTION_INVALID_ARGUMENT);
        free(keys);
        free(values);
    }
}

int main(void)
{
    check_configs();
    check_visibility();
    check_qkv_split();
    check_rope();
    check_decode();
    printf("test_mimo26_attention: ok\n");
    return 0;
}
