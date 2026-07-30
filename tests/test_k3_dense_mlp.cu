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
    K3_HIDDEN = 7168,
    K3_INTERMEDIATE = 33792,
    K3_TENSOR_COUNT = 3,
    K3_BENCHMARK_STEPS = 5,
};

enum {
    T_GATE,
    T_UP,
    T_DOWN,
};

typedef struct {
    const k3_st_tensor *tensor;
    k3_st_read read;
    void *device;
} loaded_tensor;

typedef struct {
    void *quantized;
    void *scales;
    uint32_t rows;
    uint32_t columns;
    uint64_t storage_bytes;
} q8_tensor;

static uint32_t rng_state = UINT32_C(0x4b334d4c);

static float random_input(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & UINT32_C(0x00ffffff)) /
            (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * 0.125f;
}

static bool load_tensor(k3_st_model *model,
                        const char *name,
                        uint32_t rows,
                        uint32_t columns,
                        loaded_tensor *loaded,
                        char *error,
                        size_t error_size) {
    memset(loaded, 0, sizeof(*loaded));
    loaded->tensor = k3_st_find(model, name);
    if (!loaded->tensor ||
        loaded->tensor->dtype != K3_ST_DTYPE_BF16 ||
        loaded->tensor->ndim != 2u ||
        loaded->tensor->shape[0] != rows ||
        loaded->tensor->shape[1] != columns) {
        snprintf(error, error_size,
                 "missing or invalid BF16 matrix %s", name);
        return false;
    }
    if (!k3_st_read_span(model, loaded->tensor->shard,
                         loaded->tensor->physical_offset,
                         loaded->tensor->byte_length, 4096u,
                         &loaded->read, error, error_size)) {
        return false;
    }
    if (hipMalloc(&loaded->device,
                  loaded->tensor->byte_length) != hipSuccess ||
        hipMemcpy(loaded->device, loaded->read.data,
                  loaded->tensor->byte_length,
                  hipMemcpyHostToDevice) != hipSuccess) {
        snprintf(error, error_size,
                 "ROCm upload failed for %s", name);
        return false;
    }
    return true;
}

static void release_tensor(loaded_tensor *loaded) {
    if (loaded->device) (void)hipFree(loaded->device);
    k3_st_read_release(&loaded->read);
    memset(loaded, 0, sizeof(*loaded));
}

static bool quantize_tensor(const loaded_tensor *loaded,
                            q8_tensor *q8) {
    memset(q8, 0, sizeof(*q8));
    q8->rows = (uint32_t)loaded->tensor->shape[0];
    q8->columns = (uint32_t)loaded->tensor->shape[1];
    if (q8->columns % 128u != 0u) return false;
    const uint64_t quantized_bytes =
        (uint64_t)q8->rows * q8->columns;
    const uint64_t scale_bytes =
        (uint64_t)q8->rows * (q8->columns / 128u) * sizeof(float);
    if (hipMalloc(&q8->quantized, quantized_bytes) != hipSuccess) {
        return false;
    }
    if (hipMalloc(&q8->scales, scale_bytes) != hipSuccess ||
        !k3_rocm_bf16_quantize_q8_128(
            q8->quantized, q8->scales, loaded->device,
            q8->rows, q8->columns, NULL)) {
        if (q8->scales) (void)hipFree(q8->scales);
        (void)hipFree(q8->quantized);
        memset(q8, 0, sizeof(*q8));
        return false;
    }
    q8->storage_bytes = quantized_bytes + scale_bytes;
    return true;
}

static void release_q8_tensor(q8_tensor *q8) {
    if (q8->scales) (void)hipFree(q8->scales);
    if (q8->quantized) (void)hipFree(q8->quantized);
    memset(q8, 0, sizeof(*q8));
}

static bool run_dense_mlp(
        bool use_q8,
        void *output,
        void *gate,
        void *up,
        void *activation,
        const void *input,
        const loaded_tensor tensors[K3_TENSOR_COUNT],
        const q8_tensor q8[K3_TENSOR_COUNT]) {
    const bool gate_ok = use_q8 ?
        k3_rocm_q8_128_gemv_bf16(
            gate, q8[T_GATE].quantized, q8[T_GATE].scales,
            input, K3_INTERMEDIATE, K3_HIDDEN, NULL) :
        k3_rocm_bf16_gemv_bf16(
            gate, tensors[T_GATE].device, input,
            K3_INTERMEDIATE, K3_HIDDEN, NULL);
    const bool up_ok = use_q8 ?
        k3_rocm_q8_128_gemv_bf16(
            up, q8[T_UP].quantized, q8[T_UP].scales,
            input, K3_INTERMEDIATE, K3_HIDDEN, NULL) :
        k3_rocm_bf16_gemv_bf16(
            up, tensors[T_UP].device, input,
            K3_INTERMEDIATE, K3_HIDDEN, NULL);
    if (!gate_ok || !up_ok ||
        !k3_rocm_situ_bf16(
            activation, gate, up, K3_INTERMEDIATE,
            4.0f, 25.0f, NULL)) {
        return false;
    }
    return use_q8 ?
        k3_rocm_q8_128_gemv_bf16(
            output, q8[T_DOWN].quantized, q8[T_DOWN].scales,
            activation, K3_HIDDEN, K3_INTERMEDIATE, NULL) :
        k3_rocm_bf16_gemv_bf16(
            output, tensors[T_DOWN].device, activation,
            K3_HIDDEN, K3_INTERMEDIATE, NULL);
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const uint8_t *values = (const uint8_t *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        hash ^= values[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 dense MLP: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 dense MLP on %s (%s)\n",
           properties.name, properties.gcnArchName);

    char error[512];
    k3_st_model model;
    CHECK(k3_st_model_open(&model, root, 96u,
                           error, sizeof(error)),
          error);
    static const char *names[K3_TENSOR_COUNT] = {
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
    };
    loaded_tensor tensors[K3_TENSOR_COUNT];
    memset(tensors, 0, sizeof(tensors));
    CHECK(load_tensor(&model, names[T_GATE],
                      K3_INTERMEDIATE, K3_HIDDEN,
                      &tensors[T_GATE], error, sizeof(error)),
          error);
    CHECK(load_tensor(&model, names[T_UP],
                      K3_INTERMEDIATE, K3_HIDDEN,
                      &tensors[T_UP], error, sizeof(error)),
          error);
    CHECK(load_tensor(&model, names[T_DOWN],
                      K3_HIDDEN, K3_INTERMEDIATE,
                      &tensors[T_DOWN], error, sizeof(error)),
          error);

    q8_tensor q8[K3_TENSOR_COUNT];
    memset(q8, 0, sizeof(q8));
    uint64_t source_bytes = 0;
    uint64_t q8_bytes = 0;
    for (uint32_t i = 0; i < K3_TENSOR_COUNT; i++) {
        CHECK(quantize_tensor(&tensors[i], &q8[i]),
              "dense MLP Q8 conversion");
        source_bytes += tensors[i].tensor->byte_length;
        q8_bytes += q8[i].storage_bytes;
    }
    HIP_CHECK(hipDeviceSynchronize());

    const size_t hidden_bytes =
        (size_t)K3_HIDDEN * sizeof(hip_bfloat16);
    const size_t intermediate_bytes =
        (size_t)K3_INTERMEDIATE * sizeof(hip_bfloat16);
    hip_bfloat16 *input =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *bf16_output =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *q8_output =
        (hip_bfloat16 *)malloc(hidden_bytes);
    CHECK(input && bf16_output && q8_output,
          "dense MLP host activation allocation");
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        input[i] = hip_bfloat16(random_input());
    }

    void *d_input = NULL;
    void *d_gate = NULL;
    void *d_up = NULL;
    void *d_activation = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_input, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_gate, intermediate_bytes));
    HIP_CHECK(hipMalloc(&d_up, intermediate_bytes));
    HIP_CHECK(hipMalloc(&d_activation, intermediate_bytes));
    HIP_CHECK(hipMalloc(&d_output, hidden_bytes));
    HIP_CHECK(hipMemcpy(d_input, input, hidden_bytes,
                        hipMemcpyHostToDevice));

    CHECK(run_dense_mlp(
              false, d_output, d_gate, d_up, d_activation,
              d_input, tensors, q8),
          "real BF16 dense MLP launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(bf16_output, d_output, hidden_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(run_dense_mlp(
              true, d_output, d_gate, d_up, d_activation,
              d_input, tensors, q8),
          "real Q8 dense MLP launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(q8_output, d_output, hidden_bytes,
                        hipMemcpyDeviceToHost));

    double error_squared = 0.0;
    double reference_squared = 0.0;
    double dot = 0.0;
    double bf16_squared = 0.0;
    double q8_squared = 0.0;
    float maximum_absolute = 0.0f;
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        const float reference = (float)bf16_output[i];
        const float candidate = (float)q8_output[i];
        const float difference = candidate - reference;
        maximum_absolute =
            fmaxf(maximum_absolute, fabsf(difference));
        error_squared += (double)difference * difference;
        reference_squared += (double)reference * reference;
        dot += (double)reference * candidate;
        bf16_squared += (double)reference * reference;
        q8_squared += (double)candidate * candidate;
    }
    const double nrmse =
        sqrt(error_squared / fmax(reference_squared, 1e-30));
    const double cosine =
        dot / fmax(sqrt(bf16_squared * q8_squared), 1e-30);
    CHECK(nrmse < 0.05 && cosine > 0.998,
          "dense MLP Q8 error gate");

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t step = 0; step < K3_BENCHMARK_STEPS; step++) {
        CHECK(run_dense_mlp(
                  false, d_output, d_gate, d_up, d_activation,
                  d_input, tensors, q8),
              "timed BF16 dense MLP launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float bf16_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&bf16_ms, start, stop));
    bf16_ms /= (float)K3_BENCHMARK_STEPS;

    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t step = 0; step < K3_BENCHMARK_STEPS; step++) {
        CHECK(run_dense_mlp(
                  true, d_output, d_gate, d_up, d_activation,
                  d_input, tensors, q8),
              "timed Q8 dense MLP launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float q8_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&q8_ms, start, stop));
    q8_ms /= (float)K3_BENCHMARK_STEPS;

    printf("  weights: %.3f GiB BF16 -> %.3f GiB Q8 "
           "(%.2f%%)\n",
           source_bytes / 1073741824.0,
           q8_bytes / 1073741824.0,
           100.0 * (double)q8_bytes / (double)source_bytes);
    printf("  BF16 %.3f ms; Q8 %.3f ms (%.2fx)\n",
           bf16_ms, q8_ms, bf16_ms / q8_ms);
    printf("  Q8 versus BF16: NRMSE %.9f, cosine %.9f, "
           "max_abs %.7f\n",
           nrmse, cosine, maximum_absolute);
    printf("  output hashes: BF16 0x%016llx, Q8 0x%016llx\n",
           (unsigned long long)fnv1a64(bf16_output, hidden_bytes),
           (unsigned long long)fnv1a64(q8_output, hidden_bytes));

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_activation));
    HIP_CHECK(hipFree(d_up));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_input));
    free(q8_output);
    free(bf16_output);
    free(input);
    for (uint32_t i = 0; i < K3_TENSOR_COUNT; i++) {
        release_q8_tensor(&q8[i]);
        release_tensor(&tensors[i]);
    }
    k3_st_model_close(&model);
    printf("K3 dense MLP: PASS\n");
    return 0;
}
