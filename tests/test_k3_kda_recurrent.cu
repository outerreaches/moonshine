#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                        \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    K3_HEAD_DIM = 128,
    CORRECTNESS_HEADS = 4,
    CORRECTNESS_STEPS = 7,
    MODEL_HEADS = 96,
    BENCHMARK_STEPS = 200,
    CONV_SEQUENCE_CHANNELS = 257,
    CONV_SEQUENCE_STEPS = 17,
};

static uint32_t rng_state = UINT32_C(0x4b444133);

static float random_unit(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state & UINT32_C(0x00ffffff)) /
           (float)UINT32_C(0x01000000);
}

static float random_symmetric(float scale) {
    return (random_unit() * 2.0f - 1.0f) * scale;
}

static inline float bf16_to_float(hip_bfloat16 value) {
    return (float)value;
}

static inline hip_bfloat16 float_to_bf16(float value) {
    return hip_bfloat16(value);
}

static void fill_step_inputs(hip_bfloat16 *q,
                             hip_bfloat16 *k,
                             hip_bfloat16 *v,
                             hip_bfloat16 *gate,
                             hip_bfloat16 *beta,
                             uint32_t heads) {
    for (uint32_t i = 0; i < heads * K3_HEAD_DIM; i++) {
        q[i] = float_to_bf16(random_symmetric(1.0f));
        k[i] = float_to_bf16(random_symmetric(1.0f));
        v[i] = float_to_bf16(random_symmetric(0.75f));
        gate[i] = float_to_bf16(random_symmetric(0.5f));
    }
    for (uint32_t head = 0; head < heads; head++) {
        beta[head] = float_to_bf16(random_symmetric(2.0f));
    }
}

static void reference_step(hip_bfloat16 *output,
                           float *state,
                           const hip_bfloat16 *q,
                           const hip_bfloat16 *k,
                           const hip_bfloat16 *v,
                           const hip_bfloat16 *raw_gate,
                           const hip_bfloat16 *raw_beta,
                           const float *A_log,
                           const float *dt_bias,
                           uint32_t heads,
                           float lower_bound) {
    const float scale = 1.0f / sqrtf((float)K3_HEAD_DIM);
    for (uint32_t head = 0; head < heads; head++) {
        const uint64_t base = (uint64_t)head * K3_HEAD_DIM;
        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
        for (uint32_t d = 0; d < K3_HEAD_DIM; d++) {
            float q_value = bf16_to_float(q[base + d]);
            float k_value = bf16_to_float(k[base + d]);
            q_norm_sq += q_value * q_value;
            k_norm_sq += k_value * k_value;
        }
        const float q_scale = scale / sqrtf(q_norm_sq + 1e-6f);
        const float k_scale = 1.0f / sqrtf(k_norm_sq + 1e-6f);
        const float beta =
            1.0f / (1.0f + expf(-bf16_to_float(raw_beta[head])));
        const float A = expf(A_log[head]);

        for (uint32_t value = 0; value < K3_HEAD_DIM; value++) {
            const uint64_t state_base =
                ((uint64_t)head * K3_HEAD_DIM + value) * K3_HEAD_DIM;
            float prediction = 0.0f;
            for (uint32_t key = 0; key < K3_HEAD_DIM; key++) {
                const float gate_input =
                    bf16_to_float(raw_gate[base + key]) +
                    dt_bias[base + key];
                const float gate =
                    lower_bound / (1.0f + expf(-A * gate_input));
                state[state_base + key] *= expf(gate);
                prediction += state[state_base + key] *
                              bf16_to_float(k[base + key]) * k_scale;
            }
            const float delta =
                beta * (bf16_to_float(v[base + value]) - prediction);
            float result = 0.0f;
            for (uint32_t key = 0; key < K3_HEAD_DIM; key++) {
                state[state_base + key] +=
                    delta * bf16_to_float(k[base + key]) * k_scale;
                result += state[state_base + key] *
                          bf16_to_float(q[base + key]) * q_scale;
            }
            output[base + value] = float_to_bf16(result);
        }
    }
}

static int test_recurrent_correctness(void) {
    const uint32_t heads = CORRECTNESS_HEADS;
    const size_t vector_elements = (size_t)heads * K3_HEAD_DIM;
    const size_t state_elements =
        (size_t)heads * K3_HEAD_DIM * K3_HEAD_DIM;
    const size_t vector_bytes =
        vector_elements * sizeof(hip_bfloat16);
    const size_t state_bytes = state_elements * sizeof(float);
    const size_t head_bytes = (size_t)heads * sizeof(float);
    const float lower_bound = -5.0f;

    hip_bfloat16 *q = (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *k = (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *v = (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *gate = (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *beta =
        (hip_bfloat16 *)malloc((size_t)heads * sizeof(hip_bfloat16));
    hip_bfloat16 *expected_output =
        (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *got_output =
        (hip_bfloat16 *)malloc(vector_bytes);
    float *A_log = (float *)malloc(head_bytes);
    float *dt_bias = (float *)malloc(vector_elements * sizeof(float));
    float *expected_state = (float *)malloc(state_bytes);
    float *got_state = (float *)malloc(state_bytes);
    CHECK(q && k && v && gate && beta && expected_output && got_output &&
          A_log && dt_bias && expected_state && got_state,
          "KDA correctness host allocation");

    for (uint32_t head = 0; head < heads; head++) {
        A_log[head] = random_symmetric(0.75f);
    }
    for (size_t i = 0; i < vector_elements; i++) {
        dt_bias[i] = random_symmetric(0.4f);
    }
    for (size_t i = 0; i < state_elements; i++) {
        expected_state[i] = random_symmetric(0.02f);
    }

    void *d_q = NULL;
    void *d_k = NULL;
    void *d_v = NULL;
    void *d_gate = NULL;
    void *d_beta = NULL;
    void *d_A_log = NULL;
    void *d_dt_bias = NULL;
    void *d_output = NULL;
    void *d_state = NULL;
    HIP_CHECK(hipMalloc(&d_q, vector_bytes));
    HIP_CHECK(hipMalloc(&d_k, vector_bytes));
    HIP_CHECK(hipMalloc(&d_v, vector_bytes));
    HIP_CHECK(hipMalloc(&d_gate, vector_bytes));
    HIP_CHECK(hipMalloc(&d_beta,
                        (size_t)heads * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_A_log, head_bytes));
    HIP_CHECK(hipMalloc(&d_dt_bias,
                        vector_elements * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, vector_bytes));
    HIP_CHECK(hipMalloc(&d_state, state_bytes));
    HIP_CHECK(hipMemcpy(d_A_log, A_log, head_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dt_bias, dt_bias,
                        vector_elements * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_state, expected_state, state_bytes,
                        hipMemcpyHostToDevice));

    float maximum_output_error = 0.0f;
    for (uint32_t step = 0; step < CORRECTNESS_STEPS; step++) {
        fill_step_inputs(q, k, v, gate, beta, heads);
        reference_step(expected_output, expected_state, q, k, v, gate, beta,
                       A_log, dt_bias, heads, lower_bound);
        HIP_CHECK(hipMemcpy(d_q, q, vector_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_k, k, vector_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_v, v, vector_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_gate, gate, vector_bytes,
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_beta, beta,
                            (size_t)heads * sizeof(hip_bfloat16),
                            hipMemcpyHostToDevice));
        CHECK(k3_rocm_kda_recurrent_bf16_f32_state(
                  d_output, d_state, d_q, d_k, d_v, d_gate, d_beta,
                  d_A_log, d_dt_bias, heads, K3_HEAD_DIM,
                  lower_bound, NULL),
              "KDA recurrent launch");
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(got_output, d_output, vector_bytes,
                            hipMemcpyDeviceToHost));
        for (size_t i = 0; i < vector_elements; i++) {
            float error = fabsf(bf16_to_float(got_output[i]) -
                                bf16_to_float(expected_output[i]));
            if (error > maximum_output_error) {
                maximum_output_error = error;
            }
            if (error > 0.00390625f) {
                fprintf(stderr,
                        "FAIL: KDA output step=%u index=%zu got=%f "
                        "expected=%f error=%f\n",
                        step, i, bf16_to_float(got_output[i]),
                        bf16_to_float(expected_output[i]), error);
                return 1;
            }
        }
    }

    HIP_CHECK(hipMemcpy(got_state, d_state, state_bytes,
                        hipMemcpyDeviceToHost));
    float maximum_state_error = 0.0f;
    for (size_t i = 0; i < state_elements; i++) {
        float error = fabsf(got_state[i] - expected_state[i]);
        if (error > maximum_state_error) maximum_state_error = error;
    }
    CHECK(maximum_state_error <= 2e-5f,
          "KDA persistent state exceeds F32 tolerance");
    printf("  recurrent KDA %u heads x %u, %u steps: "
           "max_output_abs=%.7f max_state_abs=%.7f\n",
           heads, K3_HEAD_DIM, CORRECTNESS_STEPS,
           maximum_output_error, maximum_state_error);

    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_dt_bias));
    HIP_CHECK(hipFree(d_A_log));
    HIP_CHECK(hipFree(d_beta));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_q));
    free(got_state);
    free(expected_state);
    free(dt_bias);
    free(A_log);
    free(got_output);
    free(expected_output);
    free(beta);
    free(gate);
    free(v);
    free(k);
    free(q);
    return 0;
}

static int test_decode_support_ops(void) {
    const uint32_t channels = MODEL_HEADS * K3_HEAD_DIM;
    const size_t vector_bytes =
        (size_t)channels * sizeof(hip_bfloat16);
    const size_t cache_bytes =
        (size_t)channels * 4u * sizeof(hip_bfloat16);
    const size_t conv_weight_bytes =
        (size_t)channels * 4u * sizeof(float);
    const size_t norm_weight_bytes =
        (size_t)K3_HEAD_DIM * sizeof(float);
    hip_bfloat16 *input = (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *gate = (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *cache = (hip_bfloat16 *)malloc(cache_bytes);
    hip_bfloat16 *expected_cache =
        (hip_bfloat16 *)malloc(cache_bytes);
    hip_bfloat16 *conv_output =
        (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *norm_output =
        (hip_bfloat16 *)malloc(vector_bytes);
    float *conv_weight = (float *)malloc(conv_weight_bytes);
    float *norm_weight = (float *)malloc(norm_weight_bytes);
    CHECK(input && gate && cache && expected_cache && conv_output &&
          norm_output && conv_weight && norm_weight,
          "KDA support-op host allocation");
    for (uint32_t i = 0; i < channels; i++) {
        input[i] = float_to_bf16(random_symmetric(1.0f));
        gate[i] = float_to_bf16(random_symmetric(3.0f));
    }
    for (uint32_t i = 0; i < channels * 4u; i++) {
        cache[i] = float_to_bf16(random_symmetric(1.0f));
        conv_weight[i] = random_symmetric(0.5f);
    }
    memcpy(expected_cache, cache, cache_bytes);
    for (uint32_t i = 0; i < K3_HEAD_DIM; i++) {
        norm_weight[i] = 1.0f + random_symmetric(0.1f);
    }

    void *d_input = NULL;
    void *d_gate = NULL;
    void *d_cache = NULL;
    void *d_conv_weight = NULL;
    void *d_norm_weight = NULL;
    void *d_conv_output = NULL;
    void *d_norm_output = NULL;
    HIP_CHECK(hipMalloc(&d_input, vector_bytes));
    HIP_CHECK(hipMalloc(&d_gate, vector_bytes));
    HIP_CHECK(hipMalloc(&d_cache, cache_bytes));
    HIP_CHECK(hipMalloc(&d_conv_weight, conv_weight_bytes));
    HIP_CHECK(hipMalloc(&d_norm_weight, norm_weight_bytes));
    HIP_CHECK(hipMalloc(&d_conv_output, vector_bytes));
    HIP_CHECK(hipMalloc(&d_norm_output, vector_bytes));
    HIP_CHECK(hipMemcpy(d_input, input, vector_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate, gate, vector_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cache, cache, cache_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_conv_weight, conv_weight, conv_weight_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norm_weight, norm_weight, norm_weight_bytes,
                        hipMemcpyHostToDevice));
    CHECK(k3_rocm_short_conv4_silu_bf16_f32_weight(
              d_conv_output, d_cache, d_input, d_conv_weight,
              channels, NULL),
          "short-conv launch");
    CHECK(k3_rocm_rms_norm_sigmoid_gate_bf16_f32_weight(
              d_norm_output, d_input, d_gate, d_norm_weight,
              MODEL_HEADS, K3_HEAD_DIM, 1e-5f, NULL),
          "gated RMSNorm launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(conv_output, d_conv_output, vector_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(norm_output, d_norm_output, vector_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(cache, d_cache, cache_bytes,
                        hipMemcpyDeviceToHost));

    float maximum_conv_error = 0.0f;
    for (uint32_t channel = 0; channel < channels; channel++) {
        const uint64_t base = (uint64_t)channel * 4u;
        hip_bfloat16 current = input[channel];
        hip_bfloat16 x0 = expected_cache[base + 1u];
        hip_bfloat16 x1 = expected_cache[base + 2u];
        hip_bfloat16 x2 = expected_cache[base + 3u];
        float sum =
            bf16_to_float(x0) * conv_weight[base] +
            bf16_to_float(x1) * conv_weight[base + 1u] +
            bf16_to_float(x2) * conv_weight[base + 2u] +
            bf16_to_float(current) * conv_weight[base + 3u];
        float expected =
            bf16_to_float(float_to_bf16(sum / (1.0f + expf(-sum))));
        float error = fabsf(bf16_to_float(conv_output[channel]) - expected);
        if (error > maximum_conv_error) maximum_conv_error = error;
        expected_cache[base] = x0;
        expected_cache[base + 1u] = x1;
        expected_cache[base + 2u] = x2;
        expected_cache[base + 3u] = current;
    }
    CHECK(maximum_conv_error <= 0.00390625f,
          "short-conv output mismatch");
    CHECK(memcmp(cache, expected_cache, cache_bytes) == 0,
          "short-conv cache mismatch");

    float maximum_norm_error = 0.0f;
    for (uint32_t head = 0; head < MODEL_HEADS; head++) {
        const uint64_t base = (uint64_t)head * K3_HEAD_DIM;
        float sum_squares = 0.0f;
        for (uint32_t d = 0; d < K3_HEAD_DIM; d++) {
            float value = bf16_to_float(input[base + d]);
            sum_squares += value * value;
        }
        float reciprocal_std =
            1.0f / sqrtf(sum_squares / (float)K3_HEAD_DIM + 1e-5f);
        for (uint32_t d = 0; d < K3_HEAD_DIM; d++) {
            float sigmoid_gate =
                1.0f / (1.0f + expf(-bf16_to_float(gate[base + d])));
            float expected = bf16_to_float(float_to_bf16(
                bf16_to_float(input[base + d]) * reciprocal_std *
                norm_weight[d] * sigmoid_gate));
            float error =
                fabsf(bf16_to_float(norm_output[base + d]) - expected);
            if (error > maximum_norm_error) maximum_norm_error = error;
        }
    }
    CHECK(maximum_norm_error <= 0.00390625f,
          "gated RMSNorm output mismatch");
    printf("  short-conv4 + gated RMSNorm, 12288 channels: "
           "max_abs=(%.7f, %.7f)\n",
           maximum_conv_error, maximum_norm_error);

    HIP_CHECK(hipFree(d_norm_output));
    HIP_CHECK(hipFree(d_conv_output));
    HIP_CHECK(hipFree(d_norm_weight));
    HIP_CHECK(hipFree(d_conv_weight));
    HIP_CHECK(hipFree(d_cache));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_input));
    free(norm_weight);
    free(conv_weight);
    free(norm_output);
    free(conv_output);
    free(expected_cache);
    free(cache);
    free(gate);
    free(input);
    return 0;
}

static int test_short_conv_sequence_equivalence(void) {
    const uint32_t channels = CONV_SEQUENCE_CHANNELS;
    const uint32_t steps = CONV_SEQUENCE_STEPS;
    const size_t sequence_elements = (size_t)steps * channels;
    const size_t sequence_bytes =
        sequence_elements * sizeof(hip_bfloat16);
    const size_t cache_bytes =
        (size_t)channels * 4u * sizeof(hip_bfloat16);
    const size_t weight_bytes =
        (size_t)channels * 4u * sizeof(float);

    hip_bfloat16 *input =
        (hip_bfloat16 *)malloc(sequence_bytes);
    hip_bfloat16 *initial_cache =
        (hip_bfloat16 *)malloc(cache_bytes);
    hip_bfloat16 *reference_output =
        (hip_bfloat16 *)malloc(sequence_bytes);
    hip_bfloat16 *sequence_output =
        (hip_bfloat16 *)malloc(sequence_bytes);
    hip_bfloat16 *reference_cache =
        (hip_bfloat16 *)malloc(cache_bytes);
    hip_bfloat16 *sequence_cache =
        (hip_bfloat16 *)malloc(cache_bytes);
    float *weight = (float *)malloc(weight_bytes);
    CHECK(input && initial_cache && reference_output && sequence_output &&
          reference_cache && sequence_cache && weight,
          "short-conv sequence host allocation");

    for (size_t i = 0; i < sequence_elements; i++) {
        input[i] = float_to_bf16(random_symmetric(1.0f));
    }
    for (uint32_t i = 0; i < channels * 4u; i++) {
        initial_cache[i] = float_to_bf16(random_symmetric(1.0f));
        weight[i] = random_symmetric(0.5f);
    }

    void *d_input = NULL;
    void *d_weight = NULL;
    void *d_reference_output = NULL;
    void *d_sequence_output = NULL;
    void *d_reference_cache = NULL;
    void *d_sequence_cache = NULL;
    HIP_CHECK(hipMalloc(&d_input, sequence_bytes));
    HIP_CHECK(hipMalloc(&d_weight, weight_bytes));
    HIP_CHECK(hipMalloc(&d_reference_output, sequence_bytes));
    HIP_CHECK(hipMalloc(&d_sequence_output, sequence_bytes));
    HIP_CHECK(hipMalloc(&d_reference_cache, cache_bytes));
    HIP_CHECK(hipMalloc(&d_sequence_cache, cache_bytes));
    HIP_CHECK(hipMemcpy(
        d_input, input, sequence_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_weight, weight, weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_reference_cache, initial_cache, cache_bytes,
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_sequence_cache, initial_cache, cache_bytes,
        hipMemcpyHostToDevice));

    for (uint32_t step = 0; step < steps; step++) {
        const size_t offset = (size_t)step * channels;
        CHECK(k3_rocm_short_conv4_silu_bf16_f32_weight(
                  (hip_bfloat16 *)d_reference_output + offset,
                  d_reference_cache,
                  (const hip_bfloat16 *)d_input + offset,
                  d_weight, channels, NULL),
              "repeated one-token short-conv launch");
    }
    CHECK(k3_rocm_short_conv4_silu_sequence_bf16_f32_weight(
              d_sequence_output, d_sequence_cache, d_input, d_weight,
              steps, channels, NULL),
          "sequence short-conv launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(
        reference_output, d_reference_output, sequence_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        sequence_output, d_sequence_output, sequence_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        reference_cache, d_reference_cache, cache_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        sequence_cache, d_sequence_cache, cache_bytes,
        hipMemcpyDeviceToHost));

    CHECK(memcmp(reference_output, sequence_output, sequence_bytes) == 0,
          "sequence short-conv output differs from one-token launches");
    CHECK(memcmp(reference_cache, sequence_cache, cache_bytes) == 0,
          "sequence short-conv cache differs from one-token launches");
    printf("  short-conv4 sequence %u channels x %u steps: exact\n",
           channels, steps);

    HIP_CHECK(hipFree(d_sequence_cache));
    HIP_CHECK(hipFree(d_reference_cache));
    HIP_CHECK(hipFree(d_sequence_output));
    HIP_CHECK(hipFree(d_reference_output));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_input));
    free(weight);
    free(sequence_cache);
    free(reference_cache);
    free(sequence_output);
    free(reference_output);
    free(initial_cache);
    free(input);
    return 0;
}

static int benchmark_model_shape(void) {
    const uint32_t heads = MODEL_HEADS;
    const size_t vector_elements = (size_t)heads * K3_HEAD_DIM;
    const size_t vector_bytes =
        vector_elements * sizeof(hip_bfloat16);
    const size_t state_bytes =
        (size_t)heads * K3_HEAD_DIM * K3_HEAD_DIM * sizeof(float);
    void *d_q = NULL;
    void *d_k = NULL;
    void *d_v = NULL;
    void *d_gate = NULL;
    void *d_beta = NULL;
    void *d_A_log = NULL;
    void *d_dt_bias = NULL;
    void *d_output = NULL;
    void *d_state = NULL;
    HIP_CHECK(hipMalloc(&d_q, vector_bytes));
    HIP_CHECK(hipMalloc(&d_k, vector_bytes));
    HIP_CHECK(hipMalloc(&d_v, vector_bytes));
    HIP_CHECK(hipMalloc(&d_gate, vector_bytes));
    HIP_CHECK(hipMalloc(&d_beta,
                        (size_t)heads * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_A_log, (size_t)heads * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dt_bias, vector_elements * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, vector_bytes));
    HIP_CHECK(hipMalloc(&d_state, state_bytes));
    HIP_CHECK(hipMemset(d_q, 0, vector_bytes));
    HIP_CHECK(hipMemset(d_k, 0, vector_bytes));
    HIP_CHECK(hipMemset(d_v, 0, vector_bytes));
    HIP_CHECK(hipMemset(d_gate, 0, vector_bytes));
    HIP_CHECK(hipMemset(d_beta, 0,
                        (size_t)heads * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMemset(d_A_log, 0, (size_t)heads * sizeof(float)));
    HIP_CHECK(hipMemset(d_dt_bias, 0, vector_elements * sizeof(float)));
    HIP_CHECK(hipMemset(d_state, 0, state_bytes));

    for (uint32_t i = 0; i < 10; i++) {
        CHECK(k3_rocm_kda_recurrent_bf16_f32_state(
                  d_output, d_state, d_q, d_k, d_v, d_gate, d_beta,
                  d_A_log, d_dt_bias, heads, K3_HEAD_DIM, -5.0f, NULL),
              "KDA benchmark warmup");
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t i = 0; i < BENCHMARK_STEPS; i++) {
        CHECK(k3_rocm_kda_recurrent_bf16_f32_state(
                  d_output, d_state, d_q, d_k, d_v, d_gate, d_beta,
                  d_A_log, d_dt_bias, heads, K3_HEAD_DIM, -5.0f, NULL),
              "KDA benchmark launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("  model-shape KDA 96x128 state=6.000 MiB: "
           "%.3f ms/token/layer (mean of %u)\n",
           elapsed_ms / (float)BENCHMARK_STEPS, BENCHMARK_STEPS);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_dt_bias));
    HIP_CHECK(hipFree(d_A_log));
    HIP_CHECK(hipFree(d_beta));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_q));
    return 0;
}

int main(void) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 recurrent KDA: SKIP (no device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 recurrent KDA on %s (%s)\n",
           properties.name, properties.gcnArchName);
    if (test_recurrent_correctness() != 0) return 1;
    if (test_decode_support_ops() != 0) return 1;
    if (test_short_conv_sequence_equivalence() != 0) return 1;
    if (benchmark_model_shape() != 0) return 1;
    printf("K3 recurrent KDA: PASS\n");
    return 0;
}
