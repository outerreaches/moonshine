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

static uint32_t rng_state = UINT32_C(0x4b335046);

static uint32_t random_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float random_symmetric(float scale) {
    return
        ((float)(random_u32() & UINT32_C(0x00ffffff)) /
         (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * scale;
}

static int test_projection_batches(void) {
    enum {
        VECTORS = 2,
        ROWS = 64,
        COLUMNS = 256,
    };
    const size_t weight_bytes =
        (size_t)ROWS * COLUMNS * sizeof(hip_bfloat16);
    const size_t input_bytes =
        (size_t)VECTORS * COLUMNS * sizeof(hip_bfloat16);
    const size_t bf16_output_bytes =
        (size_t)VECTORS * ROWS * sizeof(hip_bfloat16);
    const size_t f32_output_bytes =
        (size_t)VECTORS * ROWS * sizeof(float);
    hip_bfloat16 *weights =
        (hip_bfloat16 *)malloc(weight_bytes);
    hip_bfloat16 *inputs =
        (hip_bfloat16 *)malloc(input_bytes);
    CHECK(weights && inputs, "projection host allocation");
    for (uint64_t index = 0;
         index < (uint64_t)ROWS * COLUMNS; index++) {
        weights[index] =
            hip_bfloat16(random_symmetric(0.25f));
    }
    for (uint64_t index = 0;
         index < (uint64_t)VECTORS * COLUMNS; index++) {
        inputs[index] =
            hip_bfloat16(random_symmetric(0.5f));
    }

    void *d_weights = NULL;
    void *d_inputs = NULL;
    void *d_batch_bf16 = NULL;
    void *d_loop_bf16 = NULL;
    void *d_batch_f32 = NULL;
    void *d_loop_f32 = NULL;
    void *d_quantized = NULL;
    void *d_scales = NULL;
    void *d_dequantized = NULL;
    void *d_blas_bf16 = NULL;
    HIP_CHECK(hipMalloc(&d_weights, weight_bytes));
    HIP_CHECK(hipMalloc(&d_inputs, input_bytes));
    HIP_CHECK(hipMalloc(&d_batch_bf16, bf16_output_bytes));
    HIP_CHECK(hipMalloc(&d_loop_bf16, bf16_output_bytes));
    HIP_CHECK(hipMalloc(&d_batch_f32, f32_output_bytes));
    HIP_CHECK(hipMalloc(&d_loop_f32, f32_output_bytes));
    HIP_CHECK(hipMalloc(
        &d_quantized, (size_t)ROWS * COLUMNS));
    HIP_CHECK(hipMalloc(
        &d_scales,
        (size_t)ROWS * (COLUMNS / 128u) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dequantized, weight_bytes));
    HIP_CHECK(hipMalloc(&d_blas_bf16, bf16_output_bytes));
    HIP_CHECK(hipMemcpy(
        d_weights, weights, weight_bytes,
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_inputs, inputs, input_bytes,
        hipMemcpyHostToDevice));

    CHECK(k3_rocm_bf16_gemm_bf16(
              d_batch_bf16, d_weights, d_inputs,
              VECTORS, ROWS, COLUMNS, NULL),
          "BF16 batch projection");
    CHECK(k3_rocm_bf16_gemm_f32(
              d_batch_f32, d_weights, d_inputs,
              VECTORS, ROWS, COLUMNS, NULL),
          "F32 batch projection");
    for (uint32_t vector = 0; vector < VECTORS; vector++) {
        CHECK(k3_rocm_bf16_gemv_bf16(
                  (uint8_t *)d_loop_bf16 +
                      (size_t)vector * ROWS *
                          sizeof(hip_bfloat16),
                  d_weights,
                  (uint8_t *)d_inputs +
                      (size_t)vector * COLUMNS *
                          sizeof(hip_bfloat16),
                  ROWS, COLUMNS, NULL),
              "BF16 sequential projection");
        CHECK(k3_rocm_bf16_gemv_f32(
                  (uint8_t *)d_loop_f32 +
                      (size_t)vector * ROWS * sizeof(float),
                  d_weights,
                  (uint8_t *)d_inputs +
                      (size_t)vector * COLUMNS *
                          sizeof(hip_bfloat16),
                  ROWS, COLUMNS, NULL),
              "F32 sequential projection");
    }
    HIP_CHECK(hipDeviceSynchronize());
    void *batch = malloc(f32_output_bytes);
    void *loop = malloc(f32_output_bytes);
    CHECK(batch && loop, "projection comparison allocation");
    HIP_CHECK(hipMemcpy(
        batch, d_batch_bf16, bf16_output_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        loop, d_loop_bf16, bf16_output_bytes,
        hipMemcpyDeviceToHost));
    CHECK(memcmp(batch, loop, bf16_output_bytes) == 0,
          "BF16 batch/sequential projection mismatch");
    HIP_CHECK(hipMemcpy(
        batch, d_batch_f32, f32_output_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        loop, d_loop_f32, f32_output_bytes,
        hipMemcpyDeviceToHost));
    CHECK(memcmp(batch, loop, f32_output_bytes) == 0,
          "F32 batch/sequential projection mismatch");

    CHECK(k3_rocm_bf16_quantize_q8_128(
              d_quantized, d_scales, d_weights,
              ROWS, COLUMNS, NULL),
          "Q8 quantization");
    CHECK(k3_rocm_q8_128_gemm_bf16(
              d_batch_bf16, d_quantized, d_scales, d_inputs,
              VECTORS, ROWS, COLUMNS, NULL),
          "Q8 batch projection");
    for (uint32_t vector = 0; vector < VECTORS; vector++) {
        CHECK(k3_rocm_q8_128_gemv_bf16(
                  (uint8_t *)d_loop_bf16 +
                      (size_t)vector * ROWS *
                          sizeof(hip_bfloat16),
                  d_quantized, d_scales,
                  (uint8_t *)d_inputs +
                      (size_t)vector * COLUMNS *
                          sizeof(hip_bfloat16),
                  ROWS, COLUMNS, NULL),
              "Q8 sequential projection");
    }
    HIP_CHECK(hipDeviceSynchronize());

    /*
     * Q8 is the last BF16 result written. It must be byte-identical because
     * the batch kernel changes only the independent grid-y coordinate.
     */
    HIP_CHECK(hipMemcpy(
        batch, d_batch_bf16, bf16_output_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        loop, d_loop_bf16, bf16_output_bytes,
        hipMemcpyDeviceToHost));
    CHECK(memcmp(batch, loop, bf16_output_bytes) == 0,
          "Q8 batch/sequential projection mismatch");

    k3_rocm_blas_context *blas = NULL;
    CHECK(k3_rocm_blas_context_create(&blas),
          "projection hipBLAS context");
    CHECK(k3_rocm_q8_128_dequantize_bf16(
              d_dequantized, d_quantized, d_scales,
              ROWS, COLUMNS, NULL),
          "Q8 BF16 dequantization");
    CHECK(k3_rocm_blas_bf16_gemm_bf16(
              blas, d_blas_bf16, d_dequantized, d_inputs,
              VECTORS, ROWS, COLUMNS, NULL),
          "dequantized hipBLAS projection");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(
        loop, d_blas_bf16, bf16_output_bytes,
        hipMemcpyDeviceToHost));
    float max_abs = 0.0f;
    const hip_bfloat16 *custom =
        (const hip_bfloat16 *)batch;
    const hip_bfloat16 *candidate =
        (const hip_bfloat16 *)loop;
    for (uint64_t index = 0u;
         index < (uint64_t)VECTORS * ROWS; index++) {
        const float difference = fabsf(
            (float)custom[index] - (float)candidate[index]);
        if (difference > max_abs) max_abs = difference;
    }
    CHECK(max_abs <= 0.03125f,
          "dequantized hipBLAS projection drift");
    printf("  Q8 custom vs dequant+hipBLAS max_abs=%.8g\n",
           max_abs);
    k3_rocm_blas_context_destroy(blas);
    free(loop);
    free(batch);
    HIP_CHECK(hipFree(d_blas_bf16));
    HIP_CHECK(hipFree(d_dequantized));
    HIP_CHECK(hipFree(d_scales));
    HIP_CHECK(hipFree(d_quantized));
    HIP_CHECK(hipFree(d_loop_f32));
    HIP_CHECK(hipFree(d_batch_f32));
    HIP_CHECK(hipFree(d_loop_bf16));
    HIP_CHECK(hipFree(d_batch_bf16));
    HIP_CHECK(hipFree(d_inputs));
    HIP_CHECK(hipFree(d_weights));
    free(inputs);
    free(weights);
    return 0;
}

static int test_mxfp4_batch(void) {
    enum {
        VECTORS = 2,
        ROWS = 64,
        COLUMNS = 256,
    };
    const size_t packed_bytes = (size_t)ROWS * COLUMNS / 2u;
    const size_t scale_bytes = (size_t)ROWS * COLUMNS / 32u;
    const size_t input_bytes =
        (size_t)VECTORS * COLUMNS * sizeof(hip_bfloat16);
    const size_t output_bytes =
        (size_t)VECTORS * ROWS * sizeof(hip_bfloat16);
    uint8_t *packed = (uint8_t *)malloc(packed_bytes);
    uint8_t *scales = (uint8_t *)malloc(scale_bytes);
    hip_bfloat16 *inputs =
        (hip_bfloat16 *)malloc(input_bytes);
    CHECK(packed && scales && inputs, "MXFP4 batch host allocation");
    for (size_t index = 0; index < packed_bytes; index++) {
        packed[index] = (uint8_t)random_u32();
    }
    for (size_t index = 0; index < scale_bytes; index++) {
        scales[index] =
            (uint8_t)(122u + random_u32() % 11u);
    }
    for (uint64_t index = 0;
         index < (uint64_t)VECTORS * COLUMNS; index++) {
        inputs[index] =
            hip_bfloat16(random_symmetric(0.5f));
    }

    void *d_packed = NULL;
    void *d_scales = NULL;
    void *d_inputs = NULL;
    void *d_batch = NULL;
    void *d_loop = NULL;
    HIP_CHECK(hipMalloc(&d_packed, packed_bytes));
    HIP_CHECK(hipMalloc(&d_scales, scale_bytes));
    HIP_CHECK(hipMalloc(&d_inputs, input_bytes));
    HIP_CHECK(hipMalloc(&d_batch, output_bytes));
    HIP_CHECK(hipMalloc(&d_loop, output_bytes));
    HIP_CHECK(hipMemcpy(
        d_packed, packed, packed_bytes,
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_scales, scales, scale_bytes,
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_inputs, inputs, input_bytes,
        hipMemcpyHostToDevice));
    CHECK(k3_rocm_mxfp4_gemm_bf16(
              d_batch, d_packed, d_scales, d_inputs,
              VECTORS, ROWS, COLUMNS, NULL),
          "MXFP4 batch projection");
    for (uint32_t vector = 0; vector < VECTORS; vector++) {
        CHECK(k3_rocm_mxfp4_gemv_bf16(
                  (uint8_t *)d_loop +
                      (size_t)vector * ROWS *
                          sizeof(hip_bfloat16),
                  d_packed, d_scales,
                  (uint8_t *)d_inputs +
                      (size_t)vector * COLUMNS *
                          sizeof(hip_bfloat16),
                  ROWS, COLUMNS, NULL),
              "MXFP4 sequential projection");
    }
    HIP_CHECK(hipDeviceSynchronize());
    void *batch = malloc(output_bytes);
    void *loop = malloc(output_bytes);
    CHECK(batch && loop, "MXFP4 comparison allocation");
    HIP_CHECK(hipMemcpy(
        batch, d_batch, output_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        loop, d_loop, output_bytes,
        hipMemcpyDeviceToHost));
    CHECK(memcmp(batch, loop, output_bytes) == 0,
          "MXFP4 batch/sequential mismatch");

    free(loop);
    free(batch);
    HIP_CHECK(hipFree(d_loop));
    HIP_CHECK(hipFree(d_batch));
    HIP_CHECK(hipFree(d_inputs));
    HIP_CHECK(hipFree(d_scales));
    HIP_CHECK(hipFree(d_packed));
    free(inputs);
    free(scales);
    free(packed);
    return 0;
}

static int test_router_and_reduction_batches(void) {
    enum {
        TOKENS = 2,
        EXPERTS = 896,
        TOP_K = 16,
        HIDDEN = 128,
    };
    const size_t logits_bytes =
        (size_t)TOKENS * EXPERTS * sizeof(float);
    const size_t bias_bytes =
        (size_t)EXPERTS * sizeof(float);
    const size_t ids_bytes =
        (size_t)TOKENS * TOP_K * sizeof(uint32_t);
    const size_t weights_bytes =
        (size_t)TOKENS * TOP_K * sizeof(float);
    const size_t values_bytes =
        (size_t)TOKENS * TOP_K * HIDDEN *
        sizeof(hip_bfloat16);
    const size_t output_bytes =
        (size_t)TOKENS * HIDDEN * sizeof(hip_bfloat16);
    float *logits = (float *)malloc(logits_bytes);
    float *bias = (float *)malloc(bias_bytes);
    hip_bfloat16 *values =
        (hip_bfloat16 *)malloc(values_bytes);
    CHECK(logits && bias && values,
          "router/reduction host allocation");
    for (uint64_t index = 0;
         index < (uint64_t)TOKENS * EXPERTS; index++) {
        logits[index] = random_symmetric(4.0f);
    }
    for (uint32_t index = 0; index < EXPERTS; index++) {
        bias[index] = random_symmetric(0.15f);
    }
    for (uint64_t index = 0;
         index < (uint64_t)TOKENS * TOP_K * HIDDEN; index++) {
        values[index] =
            hip_bfloat16(random_symmetric(0.5f));
    }

    void *d_logits = NULL;
    void *d_bias = NULL;
    void *d_batch_ids = NULL;
    void *d_loop_ids = NULL;
    void *d_batch_weights = NULL;
    void *d_loop_weights = NULL;
    void *d_values = NULL;
    void *d_batch_output = NULL;
    void *d_loop_output = NULL;
    HIP_CHECK(hipMalloc(&d_logits, logits_bytes));
    HIP_CHECK(hipMalloc(&d_bias, bias_bytes));
    HIP_CHECK(hipMalloc(&d_batch_ids, ids_bytes));
    HIP_CHECK(hipMalloc(&d_loop_ids, ids_bytes));
    HIP_CHECK(hipMalloc(&d_batch_weights, weights_bytes));
    HIP_CHECK(hipMalloc(&d_loop_weights, weights_bytes));
    HIP_CHECK(hipMalloc(&d_values, values_bytes));
    HIP_CHECK(hipMalloc(&d_batch_output, output_bytes));
    HIP_CHECK(hipMalloc(&d_loop_output, output_bytes));
    HIP_CHECK(hipMemcpy(
        d_logits, logits, logits_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_bias, bias, bias_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_values, values, values_bytes, hipMemcpyHostToDevice));
    CHECK(k3_rocm_router_topk_f32_batch(
              d_batch_ids, d_batch_weights,
              d_logits, d_bias, TOKENS,
              EXPERTS, TOP_K, 1.0f, NULL),
          "batch router");
    for (uint32_t token = 0; token < TOKENS; token++) {
        CHECK(k3_rocm_router_topk_f32(
                  (uint8_t *)d_loop_ids +
                      (size_t)token * TOP_K * sizeof(uint32_t),
                  (uint8_t *)d_loop_weights +
                      (size_t)token * TOP_K * sizeof(float),
                  (uint8_t *)d_logits +
                      (size_t)token * EXPERTS * sizeof(float),
                  d_bias, EXPERTS, TOP_K, 1.0f, NULL),
              "sequential router");
    }
    CHECK(k3_rocm_weighted_sum_bf16_batch(
              d_batch_output, d_values, d_batch_weights,
              TOKENS, TOP_K, HIDDEN, NULL),
          "batch weighted sum");
    for (uint32_t token = 0; token < TOKENS; token++) {
        CHECK(k3_rocm_weighted_sum_bf16(
                  (uint8_t *)d_loop_output +
                      (size_t)token * HIDDEN *
                          sizeof(hip_bfloat16),
                  (uint8_t *)d_values +
                      (size_t)token * TOP_K * HIDDEN *
                          sizeof(hip_bfloat16),
                  (uint8_t *)d_loop_weights +
                      (size_t)token * TOP_K * sizeof(float),
                  TOP_K, HIDDEN, NULL),
              "sequential weighted sum");
    }
    HIP_CHECK(hipDeviceSynchronize());
    void *batch = malloc(ids_bytes + weights_bytes + output_bytes);
    void *loop = malloc(ids_bytes + weights_bytes + output_bytes);
    CHECK(batch && loop, "router/reduction comparison allocation");
    uint8_t *batch_bytes = (uint8_t *)batch;
    uint8_t *loop_bytes = (uint8_t *)loop;
    HIP_CHECK(hipMemcpy(
        batch_bytes, d_batch_ids, ids_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        loop_bytes, d_loop_ids, ids_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        batch_bytes + ids_bytes,
        d_batch_weights, weights_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        loop_bytes + ids_bytes,
        d_loop_weights, weights_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        batch_bytes + ids_bytes + weights_bytes,
        d_batch_output, output_bytes,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        loop_bytes + ids_bytes + weights_bytes,
        d_loop_output, output_bytes,
        hipMemcpyDeviceToHost));
    CHECK(memcmp(
              batch, loop,
              ids_bytes + weights_bytes + output_bytes) == 0,
          "router/reduction batch mismatch");

    free(loop);
    free(batch);
    HIP_CHECK(hipFree(d_loop_output));
    HIP_CHECK(hipFree(d_batch_output));
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_loop_weights));
    HIP_CHECK(hipFree(d_batch_weights));
    HIP_CHECK(hipFree(d_loop_ids));
    HIP_CHECK(hipFree(d_batch_ids));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_logits));
    free(values);
    free(bias);
    free(logits);
    return 0;
}

int main(void) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 prefill ops: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    if (test_projection_batches() != 0) return 1;
    if (test_mxfp4_batch() != 0) return 1;
    if (test_router_and_reduction_batches() != 0) return 1;
    printf("K3 prefill ops: 2-token batch equals sequential: PASS\n");
    return 0;
}
