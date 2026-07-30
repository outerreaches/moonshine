#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

static uint32_t rng_state = UINT32_C(0x4b334b33);

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

static int test_situ(void) {
    const uint64_t count = 8192;
    const size_t bytes = (size_t)count * sizeof(hip_bfloat16);
    hip_bfloat16 *gate = (hip_bfloat16 *)malloc(bytes);
    hip_bfloat16 *up = (hip_bfloat16 *)malloc(bytes);
    hip_bfloat16 *got = (hip_bfloat16 *)malloc(bytes);
    CHECK(gate && up && got, "SiTU host allocation");
    for (uint64_t i = 0; i < count; i++) {
        gate[i] = float_to_bf16(random_symmetric(12.0f));
        up[i] = float_to_bf16(random_symmetric(50.0f));
    }

    void *d_gate = NULL;
    void *d_up = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_gate, bytes));
    HIP_CHECK(hipMalloc(&d_up, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMemcpy(d_gate, gate, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_up, up, bytes, hipMemcpyHostToDevice));

    CHECK(k3_rocm_situ_bf16(d_output, d_gate, d_up, count,
                            4.0f, 25.0f, NULL),
          "SiTU launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got, d_output, bytes, hipMemcpyDeviceToHost));

    float maximum_error = 0.0f;
    for (uint64_t i = 0; i < count; i++) {
        float g = bf16_to_float(gate[i]);
        float u = bf16_to_float(up[i]);
        float expected =
            (4.0f * tanhf(g / 4.0f) / (1.0f + expf(-g))) *
            (25.0f * tanhf(u / 25.0f));
        expected = bf16_to_float(float_to_bf16(expected));
        float error = fabsf(bf16_to_float(got[i]) - expected);
        if (error > maximum_error) maximum_error = error;
    }
    CHECK(maximum_error <= 0.0625f, "SiTU exceeds BF16 tolerance");
    printf("  SiTU beta=4 linear_beta=25: max_abs_error=%.6f\n",
           maximum_error);

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_up));
    HIP_CHECK(hipFree(d_gate));
    free(got);
    free(up);
    free(gate);
    return 0;
}

static int test_attn_res(void) {
    const uint32_t tokens = 17;
    const uint32_t block_capacity = 10;
    const uint32_t num_blocks = 8;
    const uint32_t hidden = 7168;
    const float epsilon = 1e-5f;
    const uint64_t vector_elements = (uint64_t)tokens * hidden;
    const uint64_t block_elements =
        (uint64_t)tokens * block_capacity * hidden;
    const size_t vector_bytes =
        (size_t)vector_elements * sizeof(hip_bfloat16);
    const size_t block_bytes =
        (size_t)block_elements * sizeof(hip_bfloat16);
    const size_t weight_bytes = (size_t)hidden * sizeof(hip_bfloat16);

    hip_bfloat16 *prefix =
        (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *blocks =
        (hip_bfloat16 *)malloc(block_bytes);
    hip_bfloat16 *norm =
        (hip_bfloat16 *)malloc(weight_bytes);
    hip_bfloat16 *qk =
        (hip_bfloat16 *)malloc(weight_bytes);
    hip_bfloat16 *got =
        (hip_bfloat16 *)malloc(vector_bytes);
    float *expected = (float *)malloc((size_t)vector_elements * sizeof(float));
    CHECK(prefix && blocks && norm && qk && got && expected,
          "AttnRes host allocation");

    for (uint64_t i = 0; i < vector_elements; i++) {
        prefix[i] = float_to_bf16(random_symmetric(0.75f));
    }
    for (uint64_t i = 0; i < block_elements; i++) {
        blocks[i] = float_to_bf16(random_symmetric(0.75f));
    }
    for (uint32_t d = 0; d < hidden; d++) {
        norm[d] = float_to_bf16(1.0f + random_symmetric(0.1f));
        qk[d] = float_to_bf16(
            random_symmetric(1.0f / sqrtf((float)hidden)));
    }

    for (uint32_t token = 0; token < tokens; token++) {
        float logits[9];
        float probabilities[9];
        for (uint32_t source = 0; source <= num_blocks; source++) {
            bool is_prefix = source == num_blocks;
            uint64_t base = is_prefix ?
                (uint64_t)token * hidden :
                ((uint64_t)token * block_capacity + source) * hidden;
            float sum_squares = 0.0f;
            float dot = 0.0f;
            for (uint32_t d = 0; d < hidden; d++) {
                float value = is_prefix ?
                    bf16_to_float(prefix[base + d]) :
                    bf16_to_float(blocks[base + d]);
                sum_squares += value * value;
                dot += value * bf16_to_float(norm[d]) *
                       bf16_to_float(qk[d]);
            }
            logits[source] =
                dot / sqrtf(sum_squares / (float)hidden + epsilon);
        }
        float maximum = logits[0];
        for (uint32_t source = 1; source <= num_blocks; source++) {
            if (logits[source] > maximum) maximum = logits[source];
        }
        float denominator = 0.0f;
        for (uint32_t source = 0; source <= num_blocks; source++) {
            probabilities[source] = expf(logits[source] - maximum);
            denominator += probabilities[source];
        }
        for (uint32_t d = 0; d < hidden; d++) {
            float mixed = 0.0f;
            for (uint32_t source = 0; source <= num_blocks; source++) {
                bool is_prefix = source == num_blocks;
                uint64_t index = is_prefix ?
                    (uint64_t)token * hidden + d :
                    ((uint64_t)token * block_capacity + source) * hidden + d;
                float value = is_prefix ?
                    bf16_to_float(prefix[index]) :
                    bf16_to_float(blocks[index]);
                mixed += probabilities[source] / denominator * value;
            }
            expected[(uint64_t)token * hidden + d] = mixed;
        }
    }

    void *d_prefix = NULL;
    void *d_blocks = NULL;
    void *d_norm = NULL;
    void *d_qk = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_prefix, vector_bytes));
    HIP_CHECK(hipMalloc(&d_blocks, block_bytes));
    HIP_CHECK(hipMalloc(&d_norm, weight_bytes));
    HIP_CHECK(hipMalloc(&d_qk, weight_bytes));
    HIP_CHECK(hipMalloc(&d_output, vector_bytes));
    HIP_CHECK(hipMemcpy(d_prefix, prefix, vector_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_blocks, blocks, block_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norm, norm, weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qk, qk, weight_bytes, hipMemcpyHostToDevice));

    CHECK(k3_rocm_attn_res_bf16(d_output, d_prefix, d_blocks, d_norm, d_qk,
                                tokens, block_capacity, num_blocks, hidden,
                                epsilon, NULL),
          "AttnRes launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got, d_output, vector_bytes, hipMemcpyDeviceToHost));

    float maximum_error = 0.0f;
    float maximum_relative = 0.0f;
    for (uint64_t i = 0; i < vector_elements; i++) {
        float actual = bf16_to_float(got[i]);
        float error = fabsf(actual - expected[i]);
        float relative = error / fmaxf(fabsf(expected[i]), 1e-3f);
        if (error > maximum_error) maximum_error = error;
        if (relative > maximum_relative) maximum_relative = relative;
        if (error > 0.08f && relative > 0.03f) {
            fprintf(stderr,
                    "FAIL: AttnRes mismatch at %" PRIu64
                    ": got=%f expected=%f abs=%f rel=%f\n",
                    i, actual, expected[i], error, relative);
            return 1;
        }
    }
    printf("  AttnRes tokens=17 blocks=8 hidden=7168: "
           "max_abs_error=%.6f max_rel_error=%.6f\n",
           maximum_error, maximum_relative);

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_qk));
    HIP_CHECK(hipFree(d_norm));
    HIP_CHECK(hipFree(d_blocks));
    HIP_CHECK(hipFree(d_prefix));
    free(expected);
    free(got);
    free(qk);
    free(norm);
    free(blocks);
    free(prefix);
    return 0;
}

static float e8m0_to_float(uint8_t exponent) {
    union {
        uint32_t bits;
        float value;
    } decoded;
    decoded.bits = exponent == 0u ?
        UINT32_C(0x00400000) : (uint32_t)exponent << 23u;
    return decoded.value;
}

static float e2m1_to_float(uint8_t nibble) {
    static const float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    float value = magnitude[nibble & 7u];
    return (nibble & 8u) ? -value : value;
}

static int test_mxfp4_gemv(void) {
    const uint32_t rows = 128;
    const uint32_t columns = 512;
    const size_t packed_bytes = (size_t)rows * columns / 2u;
    const size_t scale_bytes = (size_t)rows * columns / 32u;
    const size_t input_bytes = (size_t)columns * sizeof(hip_bfloat16);
    const size_t output_bytes = (size_t)rows * sizeof(hip_bfloat16);
    uint8_t *packed = (uint8_t *)malloc(packed_bytes);
    uint8_t *scales = (uint8_t *)malloc(scale_bytes);
    hip_bfloat16 *input = (hip_bfloat16 *)malloc(input_bytes);
    hip_bfloat16 *got = (hip_bfloat16 *)malloc(output_bytes);
    CHECK(packed && scales && input && got, "MXFP4 GEMV host allocation");

    for (size_t i = 0; i < packed_bytes; i++) {
        packed[i] = (uint8_t)(random_unit() * 256.0f);
    }
    for (size_t i = 0; i < scale_bytes; i++) {
        /* Keep the fixture in a representative finite range around 2^0. */
        scales[i] = (uint8_t)(122u + (uint32_t)(random_unit() * 11.0f));
    }
    for (uint32_t i = 0; i < columns; i++) {
        input[i] = float_to_bf16(random_symmetric(1.0f));
    }

    void *d_packed = NULL;
    void *d_scales = NULL;
    void *d_input = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_packed, packed_bytes));
    HIP_CHECK(hipMalloc(&d_scales, scale_bytes));
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_output, output_bytes));
    HIP_CHECK(hipMemcpy(d_packed, packed, packed_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scales, scales, scale_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input, input, input_bytes, hipMemcpyHostToDevice));

    CHECK(k3_rocm_mxfp4_gemv_bf16(d_output, d_packed, d_scales, d_input,
                                   rows, columns, NULL),
          "MXFP4 GEMV launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got, d_output, output_bytes, hipMemcpyDeviceToHost));

    float maximum_error = 0.0f;
    for (uint32_t row = 0; row < rows; row++) {
        float expected = 0.0f;
        for (uint32_t column = 0; column < columns; column++) {
            uint8_t byte =
                packed[(size_t)row * columns / 2u + column / 2u];
            uint8_t nibble =
                (column & 1u) ? byte >> 4u : byte & 0x0fu;
            float weight = e2m1_to_float(nibble) *
                e8m0_to_float(
                    scales[(size_t)row * columns / 32u + column / 32u]);
            expected += weight * bf16_to_float(input[column]);
        }
        expected = bf16_to_float(float_to_bf16(expected));
        float error = fabsf(bf16_to_float(got[row]) - expected);
        if (error > maximum_error) maximum_error = error;
    }
    CHECK(maximum_error <= 0.015625f,
          "MXFP4 GEMV exceeds BF16 tolerance");
    printf("  native-HF MXFP4 GEMV 128x512: max_abs_error=%.6f\n",
           maximum_error);

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_scales));
    HIP_CHECK(hipFree(d_packed));
    free(got);
    free(input);
    free(scales);
    free(packed);
    return 0;
}

static int test_dense_router_and_reductions(void) {
    const uint32_t rows = 128;
    const uint32_t columns = 512;
    const uint32_t experts = 896;
    const uint32_t top_k = 16;
    const size_t matrix_bytes =
        (size_t)rows * columns * sizeof(hip_bfloat16);
    const size_t input_bytes = (size_t)columns * sizeof(hip_bfloat16);
    const size_t bf16_output_bytes = (size_t)rows * sizeof(hip_bfloat16);
    const size_t f32_output_bytes = (size_t)rows * sizeof(float);
    hip_bfloat16 *matrix = (hip_bfloat16 *)malloc(matrix_bytes);
    hip_bfloat16 *input = (hip_bfloat16 *)malloc(input_bytes);
    hip_bfloat16 *got_bf16 = (hip_bfloat16 *)malloc(bf16_output_bytes);
    float *got_f32 = (float *)malloc(f32_output_bytes);
    CHECK(matrix && input && got_bf16 && got_f32,
          "dense GEMV host allocation");
    for (uint64_t i = 0; i < (uint64_t)rows * columns; i++) {
        matrix[i] = float_to_bf16(random_symmetric(0.25f));
    }
    for (uint32_t i = 0; i < columns; i++) {
        input[i] = float_to_bf16(random_symmetric(0.5f));
    }

    void *d_matrix = NULL;
    void *d_input = NULL;
    void *d_bf16 = NULL;
    void *d_f32 = NULL;
    HIP_CHECK(hipMalloc(&d_matrix, matrix_bytes));
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_bf16, bf16_output_bytes));
    HIP_CHECK(hipMalloc(&d_f32, f32_output_bytes));
    HIP_CHECK(hipMemcpy(d_matrix, matrix, matrix_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input, input, input_bytes, hipMemcpyHostToDevice));
    CHECK(k3_rocm_bf16_gemv_bf16(d_bf16, d_matrix, d_input,
                                  rows, columns, NULL),
          "BF16 GEMV BF16-output launch");
    CHECK(k3_rocm_bf16_gemv_f32(d_f32, d_matrix, d_input,
                                 rows, columns, NULL),
          "BF16 GEMV F32-output launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got_bf16, d_bf16, bf16_output_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_f32, d_f32, f32_output_bytes,
                        hipMemcpyDeviceToHost));
    float maximum_bf16_error = 0.0f;
    float maximum_f32_error = 0.0f;
    for (uint32_t row = 0; row < rows; row++) {
        float partial[256];
        for (uint32_t tid = 0; tid < 256; tid++) {
            float sum = 0.0f;
            for (uint32_t column = tid;
                 column < columns;
                 column += 256) {
                sum += bf16_to_float(matrix[(uint64_t)row * columns + column]) *
                       bf16_to_float(input[column]);
            }
            partial[tid] = sum;
        }
        for (uint32_t width = 128; width > 0; width /= 2) {
            for (uint32_t tid = 0; tid < width; tid++) {
                partial[tid] += partial[tid + width];
            }
        }
        float bf16_expected = bf16_to_float(float_to_bf16(partial[0]));
        float bf16_error = fabsf(bf16_to_float(got_bf16[row]) - bf16_expected);
        float f32_error = fabsf(got_f32[row] - partial[0]);
        if (bf16_error > maximum_bf16_error) {
            maximum_bf16_error = bf16_error;
        }
        if (f32_error > maximum_f32_error) maximum_f32_error = f32_error;
    }
    CHECK(maximum_bf16_error == 0.0f && maximum_f32_error <= 1e-5f,
          "BF16 dense GEMV mismatch");

    float *logits = (float *)malloc((size_t)experts * sizeof(float));
    float *bias = (float *)malloc((size_t)experts * sizeof(float));
    uint32_t got_ids[top_k];
    float got_weights[top_k];
    CHECK(logits && bias, "router host allocation");
    for (uint32_t i = 0; i < experts; i++) {
        logits[i] = random_symmetric(4.0f);
        bias[i] = random_symmetric(0.15f);
    }
    void *d_logits = NULL;
    void *d_bias = NULL;
    void *d_ids = NULL;
    void *d_weights = NULL;
    HIP_CHECK(hipMalloc(&d_logits, (size_t)experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias, (size_t)experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ids, (size_t)top_k * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_weights, (size_t)top_k * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_logits, logits, (size_t)experts * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias, bias, (size_t)experts * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK(k3_rocm_router_topk_f32(d_ids, d_weights, d_logits, d_bias,
                                  experts, top_k, 1.0f, NULL),
          "router top-k launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got_ids, d_ids, sizeof(got_ids),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_weights, d_weights, sizeof(got_weights),
                        hipMemcpyDeviceToHost));

    float expected_choice[top_k];
    float expected_raw[top_k];
    uint32_t expected_ids[top_k];
    for (uint32_t rank = 0; rank < top_k; rank++) {
        expected_choice[rank] = -INFINITY;
        expected_raw[rank] = 0.0f;
        expected_ids[rank] = UINT32_MAX;
    }
    for (uint32_t expert = 0; expert < experts; expert++) {
        float raw = 1.0f / (1.0f + expf(-logits[expert]));
        float choice = raw + bias[expert];
        uint32_t position = top_k;
        for (uint32_t rank = 0; rank < top_k; rank++) {
            if (choice > expected_choice[rank] ||
                (choice == expected_choice[rank] &&
                 expert < expected_ids[rank])) {
                position = rank;
                break;
            }
        }
        if (position == top_k) continue;
        for (uint32_t rank = top_k - 1u; rank > position; rank--) {
            expected_choice[rank] = expected_choice[rank - 1u];
            expected_raw[rank] = expected_raw[rank - 1u];
            expected_ids[rank] = expected_ids[rank - 1u];
        }
        expected_choice[position] = choice;
        expected_raw[position] = raw;
        expected_ids[position] = expert;
    }
    float denominator = 1e-20f;
    for (uint32_t rank = 0; rank < top_k; rank++) {
        denominator += expected_raw[rank];
    }
    for (uint32_t rank = 0; rank < top_k; rank++) {
        CHECK(got_ids[rank] == expected_ids[rank], "router ID mismatch");
        CHECK(fabsf(got_weights[rank] -
                    expected_raw[rank] / denominator) <= 1e-6f,
              "router weight mismatch");
    }

    const uint32_t vectors = 16;
    hip_bfloat16 *values = (hip_bfloat16 *)malloc(
        (size_t)vectors * columns * sizeof(hip_bfloat16));
    float vector_weights[vectors];
    hip_bfloat16 *weighted = (hip_bfloat16 *)malloc(input_bytes);
    hip_bfloat16 *norm_weight = (hip_bfloat16 *)malloc(input_bytes);
    hip_bfloat16 *normalized = (hip_bfloat16 *)malloc(input_bytes);
    hip_bfloat16 *added = (hip_bfloat16 *)malloc(input_bytes);
    CHECK(values && weighted && norm_weight && normalized && added,
          "reduction host allocation");
    for (uint64_t i = 0; i < (uint64_t)vectors * columns; i++) {
        values[i] = float_to_bf16(random_symmetric(1.0f));
    }
    float weight_sum = 0.0f;
    for (uint32_t i = 0; i < vectors; i++) {
        vector_weights[i] = random_unit();
        weight_sum += vector_weights[i];
    }
    for (uint32_t i = 0; i < vectors; i++) vector_weights[i] /= weight_sum;
    for (uint32_t i = 0; i < columns; i++) {
        norm_weight[i] = float_to_bf16(1.0f + random_symmetric(0.1f));
    }
    void *d_values = NULL;
    void *d_vector_weights = NULL;
    void *d_weighted = NULL;
    void *d_norm_weight = NULL;
    void *d_normalized = NULL;
    void *d_added = NULL;
    HIP_CHECK(hipMalloc(&d_values,
                        (size_t)vectors * columns * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_vector_weights,
                        (size_t)vectors * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weighted, input_bytes));
    HIP_CHECK(hipMalloc(&d_norm_weight, input_bytes));
    HIP_CHECK(hipMalloc(&d_normalized, input_bytes));
    HIP_CHECK(hipMalloc(&d_added, input_bytes));
    HIP_CHECK(hipMemcpy(d_values, values,
                        (size_t)vectors * columns * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vector_weights, vector_weights,
                        (size_t)vectors * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norm_weight, norm_weight, input_bytes,
                        hipMemcpyHostToDevice));
    CHECK(k3_rocm_weighted_sum_bf16(d_weighted, d_values, d_vector_weights,
                                    vectors, columns, NULL),
          "weighted sum launch");
    CHECK(k3_rocm_rms_norm_bf16(d_normalized, d_weighted, d_norm_weight,
                                1u, columns, 1e-5f, NULL),
          "RMSNorm launch");
    CHECK(k3_rocm_add_bf16(d_added, d_weighted, d_normalized,
                           columns, NULL),
          "BF16 add launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(weighted, d_weighted, input_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(normalized, d_normalized, input_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(added, d_added, input_bytes,
                        hipMemcpyDeviceToHost));
    float sum_squares = 0.0f;
    for (uint32_t d = 0; d < columns; d++) {
        float sum = 0.0f;
        for (uint32_t vector = 0; vector < vectors; vector++) {
            sum += vector_weights[vector] *
                   bf16_to_float(values[(uint64_t)vector * columns + d]);
        }
        float expected = bf16_to_float(float_to_bf16(sum));
        CHECK(bf16_to_float(weighted[d]) == expected,
              "weighted sum mismatch");
        sum_squares += expected * expected;
    }
    float reciprocal_std =
        1.0f / sqrtf(sum_squares / (float)columns + 1e-5f);
    for (uint32_t d = 0; d < columns; d++) {
        float expected_norm = bf16_to_float(float_to_bf16(
            bf16_to_float(weighted[d]) * reciprocal_std *
            bf16_to_float(norm_weight[d])));
        CHECK(fabsf(bf16_to_float(normalized[d]) - expected_norm) <= 0.0078125f,
              "RMSNorm mismatch");
        float expected_add = bf16_to_float(float_to_bf16(
            bf16_to_float(weighted[d]) + bf16_to_float(normalized[d])));
        CHECK(bf16_to_float(added[d]) == expected_add, "BF16 add mismatch");
    }
    printf("  BF16 GEMV/router/RMSNorm/reductions: PASS\n");

    HIP_CHECK(hipFree(d_added));
    HIP_CHECK(hipFree(d_normalized));
    HIP_CHECK(hipFree(d_norm_weight));
    HIP_CHECK(hipFree(d_weighted));
    HIP_CHECK(hipFree(d_vector_weights));
    HIP_CHECK(hipFree(d_values));
    free(added);
    free(normalized);
    free(norm_weight);
    free(weighted);
    free(values);
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_ids));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_logits));
    free(bias);
    free(logits);
    HIP_CHECK(hipFree(d_f32));
    HIP_CHECK(hipFree(d_bf16));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_matrix));
    free(got_f32);
    free(got_bf16);
    free(input);
    free(matrix);
    return 0;
}

int main(void) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 ROCm components: SKIP (no device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 ROCm components on %s (%s)\n",
           properties.name, properties.gcnArchName);
    if (test_situ() != 0) return 1;
    if (test_mxfp4_gemv() != 0) return 1;
    if (test_dense_router_and_reductions() != 0) return 1;
    if (test_attn_res() != 0) return 1;
    printf("K3 ROCm components: PASS\n");
    return 0;
}
