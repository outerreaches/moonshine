#include "k3_rocm_ops.h"
#include "k3_safetensors.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    K3_HIDDEN = 7168,
    K3_VOCAB = 163840,
    K3_EMBEDDING_TOKEN = 42,
    K3_EMBEDDING_BENCHMARK_ROWS = 128,
    K3_BENCHMARK_STEPS = 5,
};

static uint32_t rng_state = UINT32_C(0x4b33454f);

static float random_input(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & UINT32_C(0x00ffffff)) /
            (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * 0.125f;
}

static double elapsed_seconds(struct timespec start,
                              struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static hip_bfloat16 reference_dot(
        const hip_bfloat16 *weights,
        const hip_bfloat16 *input) {
    float partial[256];
    for (uint32_t tid = 0; tid < 256u; tid++) {
        float sum = 0.0f;
        for (uint32_t column = tid;
             column < K3_HIDDEN;
             column += 256u) {
            sum += (float)weights[column] * (float)input[column];
        }
        partial[tid] = sum;
    }
    for (uint32_t width = 128u; width > 0; width /= 2u) {
        for (uint32_t tid = 0; tid < width; tid++) {
            partial[tid] += partial[tid + width];
        }
    }
    return hip_bfloat16(partial[0]);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 embedding/output: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 embedding/output on %s (%s)\n",
           properties.name, properties.gcnArchName);

    char error[512];
    k3_st_model model;
    CHECK(k3_st_model_open(&model, root, 96u, error, sizeof(error)),
          error);
    const k3_st_tensor *embedding = k3_st_find(
        &model, "language_model.model.embed_tokens.weight");
    const k3_st_tensor *norm = k3_st_find(
        &model, "language_model.model.norm.weight");
    const k3_st_tensor *head = k3_st_find(
        &model, "language_model.lm_head.weight");
    CHECK(embedding && norm && head,
          "missing embedding, final norm, or output head");
    CHECK(embedding->ndim == 2u &&
          embedding->shape[0] == K3_VOCAB &&
          embedding->shape[1] == K3_HIDDEN &&
          head->ndim == 2u &&
          head->shape[0] == K3_VOCAB &&
          head->shape[1] == K3_HIDDEN &&
          norm->ndim == 1u && norm->shape[0] == K3_HIDDEN,
          "unexpected embedding/output tensor shape");

    const size_t hidden_bytes =
        (size_t)K3_HIDDEN * sizeof(hip_bfloat16);
    const size_t logits_bytes =
        (size_t)K3_VOCAB * sizeof(hip_bfloat16);
    k3_st_read embedding_row;
    memset(&embedding_row, 0, sizeof(embedding_row));
    CHECK(k3_st_read_span(
              &model, embedding->shard,
              embedding->physical_offset +
                  (uint64_t)K3_EMBEDDING_TOKEN * hidden_bytes,
              hidden_bytes, 4096u, &embedding_row,
              error, sizeof(error)),
          error);
    CHECK(embedding_row.data_bytes == hidden_bytes,
          "streamed embedding row byte count");

    void *embedding_stream_buffer = NULL;
    const size_t embedding_stream_capacity = hidden_bytes + 8192u;
    CHECK(posix_memalign(&embedding_stream_buffer, 4096u,
                         embedding_stream_capacity) == 0,
          "streamed embedding reusable-buffer allocation");
    uint32_t stream_token = K3_EMBEDDING_TOKEN;
    uint64_t streamed_physical_bytes = 0;
    uint32_t direct_rows = 0;
    double maximum_row_seconds = 0.0;
    struct timespec embedding_benchmark_start;
    struct timespec embedding_benchmark_end;
    clock_gettime(CLOCK_MONOTONIC, &embedding_benchmark_start);
    for (uint32_t row = 0;
         row < K3_EMBEDDING_BENCHMARK_ROWS; row++) {
        stream_token =
            (stream_token * UINT32_C(1664525) +
             UINT32_C(1013904223)) % K3_VOCAB;
        k3_st_read_view row_view;
        struct timespec row_start;
        struct timespec row_end;
        clock_gettime(CLOCK_MONOTONIC, &row_start);
        CHECK(k3_st_read_span_into(
                  &model, embedding->shard,
                  embedding->physical_offset +
                      (uint64_t)stream_token * hidden_bytes,
                  hidden_bytes, 4096u,
                  embedding_stream_buffer,
                  embedding_stream_capacity, &row_view,
                  error, sizeof(error)),
              error);
        clock_gettime(CLOCK_MONOTONIC, &row_end);
        CHECK(row_view.data_bytes == hidden_bytes,
              "streamed embedding benchmark row byte count");
        const double row_seconds =
            elapsed_seconds(row_start, row_end);
        if (row_seconds > maximum_row_seconds) {
            maximum_row_seconds = row_seconds;
        }
        streamed_physical_bytes += row_view.allocation_bytes;
        if (row_view.used_direct_io) direct_rows++;
    }
    clock_gettime(CLOCK_MONOTONIC, &embedding_benchmark_end);
    const double embedding_benchmark_seconds =
        elapsed_seconds(embedding_benchmark_start,
                        embedding_benchmark_end);
    free(embedding_stream_buffer);

    k3_st_read norm_read;
    memset(&norm_read, 0, sizeof(norm_read));
    CHECK(k3_st_read_span(
              &model, norm->shard, norm->physical_offset,
              norm->byte_length, 4096u, &norm_read,
              error, sizeof(error)),
          error);
    k3_st_read head_read;
    memset(&head_read, 0, sizeof(head_read));
    struct timespec read_start;
    struct timespec read_end;
    clock_gettime(CLOCK_MONOTONIC, &read_start);
    CHECK(k3_st_read_span(
              &model, head->shard, head->physical_offset,
              head->byte_length, 4096u, &head_read,
              error, sizeof(error)),
          error);
    clock_gettime(CLOCK_MONOTONIC, &read_end);

    hip_bfloat16 *hidden =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *expected_norm =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *got_norm =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *got_embedding =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *logits =
        (hip_bfloat16 *)malloc(logits_bytes);
    CHECK(hidden && expected_norm && got_norm && got_embedding && logits,
          "embedding/output host allocation");
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        hidden[i] = hip_bfloat16(random_input());
    }
    float sum_squares = 0.0f;
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        const float value = (float)hidden[i];
        sum_squares += value * value;
    }
    const float reciprocal_std =
        1.0f / sqrtf(sum_squares / (float)K3_HIDDEN + 1e-5f);
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        expected_norm[i] = hip_bfloat16(
            (float)hidden[i] * reciprocal_std *
            (float)((const hip_bfloat16 *)norm_read.data)[i]);
    }

    void *d_embedding = NULL;
    void *d_hidden = NULL;
    void *d_norm_weight = NULL;
    void *d_normalized = NULL;
    void *d_head = NULL;
    void *d_logits = NULL;
    HIP_CHECK(hipMalloc(&d_embedding, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_hidden, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_norm_weight, norm->byte_length));
    HIP_CHECK(hipMalloc(&d_normalized, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_head, head->byte_length));
    HIP_CHECK(hipMalloc(&d_logits, logits_bytes));
    HIP_CHECK(hipMemcpy(d_embedding, embedding_row.data, hidden_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_hidden, hidden, hidden_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norm_weight, norm_read.data, norm->byte_length,
                        hipMemcpyHostToDevice));
    struct timespec upload_start;
    struct timespec upload_end;
    clock_gettime(CLOCK_MONOTONIC, &upload_start);
    HIP_CHECK(hipMemcpy(d_head, head_read.data, head->byte_length,
                        hipMemcpyHostToDevice));
    clock_gettime(CLOCK_MONOTONIC, &upload_end);

    uint32_t *token_id_host = NULL;
    uint32_t *token_id_device = NULL;
    float *token_value_host = NULL;
    float *token_value_device = NULL;
    HIP_CHECK(hipHostMalloc(
        (void **)&token_id_host, sizeof(uint32_t), hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(
        (void **)&token_value_host, sizeof(float), hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer(
        (void **)&token_id_device, token_id_host, 0));
    HIP_CHECK(hipHostGetDevicePointer(
        (void **)&token_value_device, token_value_host, 0));

    CHECK(k3_rocm_rms_norm_bf16(
              d_normalized, d_hidden, d_norm_weight,
              1u, K3_HIDDEN, 1e-5f, NULL) &&
          k3_rocm_bf16_gemv_bf16(
              d_logits, d_head, d_normalized,
              K3_VOCAB, K3_HIDDEN, NULL) &&
          k3_rocm_argmax_bf16(
              token_id_device, token_value_device,
              d_logits, K3_VOCAB, NULL),
          "embedding/output launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got_embedding, d_embedding, hidden_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(memcmp(got_embedding, embedding_row.data, hidden_bytes) == 0,
          "streamed embedding row changed on upload");
    HIP_CHECK(hipMemcpy(got_norm, d_normalized, hidden_bytes,
                        hipMemcpyDeviceToHost));
    float norm_maximum = 0.0f;
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        const float difference =
            fabsf((float)got_norm[i] - (float)expected_norm[i]);
        if (difference > norm_maximum) norm_maximum = difference;
    }
    CHECK(norm_maximum <= 0.00390625f,
          "final RMSNorm mismatch");
    HIP_CHECK(hipMemcpy(logits, d_logits, logits_bytes,
                        hipMemcpyDeviceToHost));
    uint32_t cpu_token = 0;
    float cpu_value = (float)logits[0];
    for (uint32_t token = 1; token < K3_VOCAB; token++) {
        const float value = (float)logits[token];
        if (value > cpu_value) {
            cpu_value = value;
            cpu_token = token;
        }
    }
    CHECK(*token_id_host == cpu_token &&
          *token_value_host == cpu_value,
          "mapped greedy argmax mismatch");
    const hip_bfloat16 selected_reference = reference_dot(
        (const hip_bfloat16 *)head_read.data +
            (uint64_t)cpu_token * K3_HIDDEN,
        expected_norm);
    CHECK(fabsf((float)selected_reference - cpu_value) <= 0.015625f,
          "selected output-head row mismatch");

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t i = 0; i < K3_BENCHMARK_STEPS; i++) {
        CHECK(k3_rocm_rms_norm_bf16(
                  d_normalized, d_hidden, d_norm_weight,
                  1u, K3_HIDDEN, 1e-5f, NULL) &&
              k3_rocm_bf16_gemv_bf16(
                  d_logits, d_head, d_normalized,
                  K3_VOCAB, K3_HIDDEN, NULL) &&
              k3_rocm_argmax_bf16(
                  token_id_device, token_value_device,
                  d_logits, K3_VOCAB, NULL),
              "timed embedding/output launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("  streamed embedding row: %.3f KiB from %.3f GiB table\n",
           hidden_bytes / 1024.0,
           embedding->byte_length / 1073741824.0);
    printf("  random embedding rows: %.3f ms mean, %.3f ms max, "
           "%.2f KiB physical (%u/%u O_DIRECT)\n",
           embedding_benchmark_seconds * 1000.0 /
               (double)K3_EMBEDDING_BENCHMARK_ROWS,
           maximum_row_seconds * 1000.0,
           (double)streamed_physical_bytes /
               (double)K3_EMBEDDING_BENCHMARK_ROWS / 1024.0,
           direct_rows, K3_EMBEDDING_BENCHMARK_ROWS);
    printf("  output head: %.3f GiB read in %.3f ms, uploaded in %.3f ms\n",
           head->byte_length / 1073741824.0,
           elapsed_seconds(read_start, read_end) * 1000.0,
           elapsed_seconds(upload_start, upload_end) * 1000.0);
    printf("  final norm max_abs=%.7f; greedy token=%u value=%.6f\n",
           norm_maximum, *token_id_host, *token_value_host);
    printf("  final norm + BF16 head + mapped argmax: "
           "%.3f ms/token (mean of %u)\n",
           elapsed_ms / (float)K3_BENCHMARK_STEPS,
           K3_BENCHMARK_STEPS);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    HIP_CHECK(hipHostFree(token_value_host));
    HIP_CHECK(hipHostFree(token_id_host));
    HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_head));
    HIP_CHECK(hipFree(d_normalized));
    HIP_CHECK(hipFree(d_norm_weight));
    HIP_CHECK(hipFree(d_hidden));
    HIP_CHECK(hipFree(d_embedding));
    free(logits);
    free(got_embedding);
    free(got_norm);
    free(expected_norm);
    free(hidden);
    k3_st_read_release(&head_read);
    k3_st_read_release(&norm_read);
    k3_st_read_release(&embedding_row);
    k3_st_model_close(&model);
    printf("K3 embedding/output: PASS\n");
    return 0;
}
