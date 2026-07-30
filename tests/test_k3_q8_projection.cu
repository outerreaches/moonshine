#include "k3_rocm_ops.h"
#include "k3_safetensors.h"

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
    K3_ROWS = 12288,
    K3_COLUMNS = 7168,
    K3_Q8_BLOCK = 128,
    K3_THREADS = 256,
    K3_ORACLE_ROWS = 128,
    K3_BENCHMARK_STEPS = 20,
};

static uint32_t rng_state = UINT32_C(0x51384b33);

static float random_input(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & UINT32_C(0x00ffffff)) /
            (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * 0.125f;
}

static hip_bfloat16 q8_reference_row(
        const int8_t *quantized,
        const float *scales,
        const hip_bfloat16 *input) {
    float partial[K3_THREADS] = {0.0f};
    const uint32_t octet_count = K3_COLUMNS / 8u;
    for (uint32_t tid = 0; tid < K3_THREADS; tid++) {
        for (uint32_t octet = tid;
             octet < octet_count;
             octet += K3_THREADS) {
            const float scale =
                scales[octet / (K3_Q8_BLOCK / 8u)];
            const uint32_t column = octet * 8u;
            for (uint32_t i = 0; i < 8u; i++) {
                partial[tid] +=
                    (float)quantized[column + i] *
                    scale * (float)input[column + i];
            }
        }
    }
    for (uint32_t width = K3_THREADS / 2u; width > 0; width /= 2u) {
        for (uint32_t tid = 0; tid < width; tid++) {
            partial[tid] += partial[tid + width];
        }
    }
    return hip_bfloat16(partial[0]);
}

static bool benchmark_projection(
        bool q8,
        void *output,
        const void *weights,
        const void *scales,
        const void *input,
        float *mean_ms) {
    hipEvent_t start;
    hipEvent_t stop;
    if (hipEventCreate(&start) != hipSuccess ||
        hipEventCreate(&stop) != hipSuccess) {
        return false;
    }
    bool okay = q8 ?
        k3_rocm_q8_128_gemv_bf16(
            output, weights, scales, input,
            K3_ROWS, K3_COLUMNS, NULL) :
        k3_rocm_bf16_gemv_bf16(
            output, weights, input,
            K3_ROWS, K3_COLUMNS, NULL);
    if (!okay || hipDeviceSynchronize() != hipSuccess ||
        hipEventRecord(start, NULL) != hipSuccess) {
        return false;
    }
    for (uint32_t i = 0; i < K3_BENCHMARK_STEPS; i++) {
        okay = q8 ?
            k3_rocm_q8_128_gemv_bf16(
                output, weights, scales, input,
                K3_ROWS, K3_COLUMNS, NULL) :
            k3_rocm_bf16_gemv_bf16(
                output, weights, input,
                K3_ROWS, K3_COLUMNS, NULL);
        if (!okay) return false;
    }
    if (hipEventRecord(stop, NULL) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) {
        return false;
    }
    float elapsed = 0.0f;
    if (hipEventElapsedTime(&elapsed, start, stop) != hipSuccess) {
        return false;
    }
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    *mean_ms = elapsed / (float)K3_BENCHMARK_STEPS;
    return true;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 Q8 static projection: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 Q8 static projection on %s (%s)\n",
           properties.name, properties.gcnArchName);

    char error[512];
    k3_st_model model;
    CHECK(k3_st_model_open(&model, root, 96u, error, sizeof(error)),
          error);
    const char *tensor_name =
        "language_model.model.layers.0.self_attn.q_proj.weight";
    const k3_st_tensor *tensor = k3_st_find(&model, tensor_name);
    CHECK(tensor, "missing real KDA q_proj");
    CHECK(tensor->ndim == 2u &&
          tensor->shape[0] == K3_ROWS &&
          tensor->shape[1] == K3_COLUMNS,
          "unexpected real KDA q_proj shape");
    k3_st_read read;
    memset(&read, 0, sizeof(read));
    CHECK(k3_st_read_span(
              &model, tensor->shard, tensor->physical_offset,
              tensor->byte_length, 4096u, &read, error, sizeof(error)),
          error);

    const size_t input_bytes =
        (size_t)K3_COLUMNS * sizeof(hip_bfloat16);
    const size_t output_bytes =
        (size_t)K3_ROWS * sizeof(hip_bfloat16);
    const size_t quantized_bytes =
        (size_t)K3_ROWS * K3_COLUMNS;
    const size_t scale_count =
        (size_t)K3_ROWS * (K3_COLUMNS / K3_Q8_BLOCK);
    const size_t scale_bytes = scale_count * sizeof(float);
    hip_bfloat16 *input =
        (hip_bfloat16 *)malloc(input_bytes);
    hip_bfloat16 *reference =
        (hip_bfloat16 *)malloc(output_bytes);
    hip_bfloat16 *compressed =
        (hip_bfloat16 *)malloc(output_bytes);
    int8_t *oracle_quantized =
        (int8_t *)malloc((size_t)K3_ORACLE_ROWS * K3_COLUMNS);
    float *oracle_scales =
        (float *)malloc(
            (size_t)K3_ORACLE_ROWS *
            (K3_COLUMNS / K3_Q8_BLOCK) * sizeof(float));
    CHECK(input && reference && compressed &&
          oracle_quantized && oracle_scales,
          "Q8 projection host allocation");
    for (uint32_t i = 0; i < K3_COLUMNS; i++) {
        input[i] = hip_bfloat16(random_input());
    }

    void *d_weights = NULL;
    void *d_quantized = NULL;
    void *d_scales = NULL;
    void *d_input = NULL;
    void *d_reference = NULL;
    void *d_compressed = NULL;
    HIP_CHECK(hipMalloc(&d_weights, tensor->byte_length));
    HIP_CHECK(hipMalloc(&d_quantized, quantized_bytes));
    HIP_CHECK(hipMalloc(&d_scales, scale_bytes));
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_reference, output_bytes));
    HIP_CHECK(hipMalloc(&d_compressed, output_bytes));
    HIP_CHECK(hipMemcpy(d_weights, read.data, tensor->byte_length,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input, input, input_bytes,
                        hipMemcpyHostToDevice));

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    CHECK(k3_rocm_bf16_quantize_q8_128(
              d_quantized, d_scales, d_weights,
              K3_ROWS, K3_COLUMNS, NULL),
          "real Q8 projection quantization");
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float conversion_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&conversion_ms, start, stop));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    CHECK(k3_rocm_bf16_gemv_bf16(
              d_reference, d_weights, d_input,
              K3_ROWS, K3_COLUMNS, NULL),
          "real BF16 projection");
    CHECK(k3_rocm_q8_128_gemv_bf16(
              d_compressed, d_quantized, d_scales, d_input,
              K3_ROWS, K3_COLUMNS, NULL),
          "real Q8 projection");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(reference, d_reference, output_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(compressed, d_compressed, output_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        oracle_quantized, d_quantized,
        (size_t)K3_ORACLE_ROWS * K3_COLUMNS,
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        oracle_scales, d_scales,
        (size_t)K3_ORACLE_ROWS *
        (K3_COLUMNS / K3_Q8_BLOCK) * sizeof(float),
        hipMemcpyDeviceToHost));

    float q8_oracle_maximum = 0.0f;
    for (uint32_t row = 0; row < K3_ORACLE_ROWS; row++) {
        hip_bfloat16 expected = q8_reference_row(
            oracle_quantized + (uint64_t)row * K3_COLUMNS,
            oracle_scales +
                (uint64_t)row * (K3_COLUMNS / K3_Q8_BLOCK),
            input);
        float error = fabsf((float)compressed[row] - (float)expected);
        if (error > q8_oracle_maximum) q8_oracle_maximum = error;
    }
    CHECK(q8_oracle_maximum <= 0.0009765625f,
          "real Q8 projection disagrees with packed CPU oracle");

    double squared_error = 0.0;
    double squared_reference = 0.0;
    double squared_compressed = 0.0;
    double dot = 0.0;
    float maximum_absolute = 0.0f;
    for (uint32_t row = 0; row < K3_ROWS; row++) {
        float want = (float)reference[row];
        float got = (float)compressed[row];
        float error_value = got - want;
        float absolute = fabsf(error_value);
        if (absolute > maximum_absolute) maximum_absolute = absolute;
        squared_error += (double)error_value * error_value;
        squared_reference += (double)want * want;
        squared_compressed += (double)got * got;
        dot += (double)want * got;
    }
    const double normalized_rmse =
        sqrt(squared_error / squared_reference);
    const double cosine =
        dot / sqrt(squared_reference * squared_compressed);
    CHECK(isfinite(normalized_rmse) && normalized_rmse < 0.02,
          "real Q8 projection error exceeds prototype gate");
    CHECK(isfinite(cosine) && cosine > 0.999,
          "real Q8 projection cosine below prototype gate");

    float bf16_ms = 0.0f;
    float q8_ms = 0.0f;
    CHECK(benchmark_projection(
              false, d_reference, d_weights, NULL, d_input, &bf16_ms),
          "BF16 projection benchmark");
    CHECK(benchmark_projection(
              true, d_compressed, d_quantized, d_scales, d_input, &q8_ms),
          "Q8 projection benchmark");
    printf("  real q_proj: %u x %u, BF16=%.3f MiB, "
           "Q8+scales=%.3f MiB (%.2f%%)\n",
           K3_ROWS, K3_COLUMNS,
           tensor->byte_length / 1048576.0,
           (quantized_bytes + scale_bytes) / 1048576.0,
           100.0 * (double)(quantized_bytes + scale_bytes) /
               (double)tensor->byte_length);
    printf("  one-time quantization: %.3f ms\n", conversion_ms);
    printf("  packed CPU oracle max_abs: %.7f\n",
           q8_oracle_maximum);
    printf("  error vs BF16: max_abs=%.7f nrmse=%.7f cosine=%.9f\n",
           maximum_absolute, normalized_rmse, cosine);
    printf("  GEMV: BF16=%.3f ms, Q8=%.3f ms, speedup=%.3fx\n",
           bf16_ms, q8_ms, bf16_ms / q8_ms);

    HIP_CHECK(hipFree(d_compressed));
    HIP_CHECK(hipFree(d_reference));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_scales));
    HIP_CHECK(hipFree(d_quantized));
    HIP_CHECK(hipFree(d_weights));
    free(oracle_scales);
    free(oracle_quantized);
    free(compressed);
    free(reference);
    free(input);
    k3_st_read_release(&read);
    k3_st_model_close(&model);
    printf("K3 Q8 static projection: PASS\n");
    return 0;
}
