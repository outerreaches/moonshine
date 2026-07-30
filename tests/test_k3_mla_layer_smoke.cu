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
    K3_HEADS = 96,
    K3_Q_LORA = 1536,
    K3_Q_HEAD = 192,
    K3_Q = K3_HEADS * K3_Q_HEAD,
    K3_LATENT = 512,
    K3_CACHE_DIM = 576,
    K3_VALUE = 128,
    K3_INNER = K3_HEADS * K3_VALUE,
    K3_KV_HEAD_STRIDE = 256,
    K3_REDUCTION_THREADS = 256,
    K3_TENSOR_COUNT = 8,
    K3_CONTEXT = 8192,
    K3_BENCHMARK_STEPS = 10,
};

enum {
    T_Q_A,
    T_Q_A_NORM,
    T_Q_B,
    T_KV_A,
    T_KV_A_NORM,
    T_KV_B,
    T_GATE,
    T_OUTPUT,
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

static const char *tensor_names[K3_TENSOR_COUNT] = {
    "language_model.model.layers.3.self_attn.q_a_proj.weight",
    "language_model.model.layers.3.self_attn.q_a_layernorm.weight",
    "language_model.model.layers.3.self_attn.q_b_proj.weight",
    "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
    "language_model.model.layers.3.self_attn.kv_a_layernorm.weight",
    "language_model.model.layers.3.self_attn.kv_b_proj.weight",
    "language_model.model.layers.3.self_attn.g_proj.weight",
    "language_model.model.layers.3.self_attn.o_proj.weight",
};

static uint32_t rng_state = UINT32_C(0x4b334d33);

static float random_input(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & UINT32_C(0x00ffffff)) /
            (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * 0.125f;
}

static inline float bf16_to_float(hip_bfloat16 value) {
    return (float)value;
}

static inline hip_bfloat16 float_to_bf16(float value) {
    return hip_bfloat16(value);
}

static bool load_tensor(k3_st_model *model,
                        const char *name,
                        loaded_tensor *loaded,
                        char *error,
                        size_t error_size) {
    memset(loaded, 0, sizeof(*loaded));
    loaded->tensor = k3_st_find(model, name);
    if (!loaded->tensor) {
        snprintf(error, error_size, "missing tensor %s", name);
        return false;
    }
    if (!k3_st_read_span(model, loaded->tensor->shard,
                         loaded->tensor->physical_offset,
                         loaded->tensor->byte_length, 4096u,
                         &loaded->read, error, error_size)) {
        return false;
    }
    if (hipMalloc(&loaded->device, loaded->tensor->byte_length) != hipSuccess ||
        hipMemcpy(loaded->device, loaded->read.data,
                  loaded->tensor->byte_length,
                  hipMemcpyHostToDevice) != hipSuccess) {
        snprintf(error, error_size, "ROCm upload failed for %s", name);
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
    if (!loaded || !loaded->tensor ||
        loaded->tensor->dtype != K3_ST_DTYPE_BF16 ||
        loaded->tensor->ndim != 2u ||
        loaded->tensor->shape[1] % 128u != 0) {
        return false;
    }
    q8->rows = (uint32_t)loaded->tensor->shape[0];
    q8->columns = (uint32_t)loaded->tensor->shape[1];
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

static bool launch_gemv(const q8_tensor *q8,
                        void *output,
                        const void *weights,
                        const void *input,
                        uint32_t rows,
                        uint32_t columns) {
    return q8 && q8->quantized ?
        k3_rocm_q8_128_gemv_bf16(
            output, q8->quantized, q8->scales, input,
            rows, columns, NULL) :
        k3_rocm_bf16_gemv_bf16(
            output, weights, input, rows, columns, NULL);
}

static void reference_gemv(hip_bfloat16 *output,
                           const hip_bfloat16 *weights,
                           const hip_bfloat16 *input,
                           uint32_t rows,
                           uint32_t columns) {
    for (uint32_t row = 0; row < rows; row++) {
        float partial[K3_REDUCTION_THREADS];
        for (uint32_t tid = 0; tid < K3_REDUCTION_THREADS; tid++) {
            float sum = 0.0f;
            for (uint32_t column = tid;
                 column < columns;
                 column += K3_REDUCTION_THREADS) {
                sum += bf16_to_float(
                           weights[(uint64_t)row * columns + column]) *
                       bf16_to_float(input[column]);
            }
            partial[tid] = sum;
        }
        for (uint32_t width = K3_REDUCTION_THREADS / 2u;
             width > 0;
             width /= 2u) {
            for (uint32_t tid = 0; tid < width; tid++) {
                partial[tid] += partial[tid + width];
            }
        }
        output[row] = float_to_bf16(partial[0]);
    }
}

static void reference_rms_norm(hip_bfloat16 *output,
                               const hip_bfloat16 *input,
                               const hip_bfloat16 *weight,
                               uint32_t count) {
    float sum_squares = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float value = bf16_to_float(input[i]);
        sum_squares += value * value;
    }
    float reciprocal_std =
        1.0f / sqrtf(sum_squares / (float)count + 1e-5f);
    for (uint32_t i = 0; i < count; i++) {
        output[i] = float_to_bf16(
            bf16_to_float(input[i]) * reciprocal_std *
            bf16_to_float(weight[i]));
    }
}

static void reference_absorb_q(hip_bfloat16 *output,
                               const hip_bfloat16 *q,
                               const hip_bfloat16 *kv_b) {
    for (uint32_t head = 0; head < K3_HEADS; head++) {
        for (uint32_t latent = 0; latent < K3_LATENT; latent++) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < 128u; d++) {
                sum += bf16_to_float(
                           q[(uint64_t)head * K3_Q_HEAD + d]) *
                       bf16_to_float(kv_b[
                           ((uint64_t)head * K3_KV_HEAD_STRIDE + d) *
                           K3_LATENT + latent]);
            }
            output[(uint64_t)head * K3_CACHE_DIM + latent] =
                float_to_bf16(sum);
        }
        for (uint32_t d = 0; d < 64u; d++) {
            output[(uint64_t)head * K3_CACHE_DIM + K3_LATENT + d] =
                q[(uint64_t)head * K3_Q_HEAD + 128u + d];
        }
    }
}

static void reference_decompress_v(hip_bfloat16 *output,
                                   const hip_bfloat16 *latent,
                                   const hip_bfloat16 *kv_b) {
    for (uint32_t head = 0; head < K3_HEADS; head++) {
        for (uint32_t value = 0; value < K3_VALUE; value++) {
            float sum = 0.0f;
            const uint64_t weight_base =
                ((uint64_t)head * K3_KV_HEAD_STRIDE +
                 128u + value) * K3_LATENT;
            for (uint32_t d = 0; d < K3_LATENT; d++) {
                sum += bf16_to_float(kv_b[weight_base + d]) *
                       bf16_to_float(
                           latent[(uint64_t)head * K3_LATENT + d]);
            }
            output[(uint64_t)head * K3_VALUE + value] =
                float_to_bf16(sum);
        }
    }
}

static bool compare_vector(const char *label,
                           const hip_bfloat16 *actual,
                           const hip_bfloat16 *expected,
                           uint32_t count,
                           float absolute_tolerance,
                           float relative_tolerance) {
    float maximum_absolute = 0.0f;
    float maximum_relative = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float got = bf16_to_float(actual[i]);
        float want = bf16_to_float(expected[i]);
        float absolute = fabsf(got - want);
        float relative = absolute / fmaxf(fabsf(want), 1e-3f);
        if (absolute > maximum_absolute) maximum_absolute = absolute;
        if (relative > maximum_relative) maximum_relative = relative;
        if (!isfinite(got) ||
            (absolute > absolute_tolerance &&
             relative > relative_tolerance)) {
            fprintf(stderr,
                    "FAIL: %s[%u] got=%f expected=%f abs=%f rel=%f\n",
                    label, i, got, want, absolute, relative);
            return false;
        }
    }
    printf("  %-12s max_abs=%-10.7f max_rel=%.7f\n",
           label, maximum_absolute, maximum_relative);
    return true;
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool launch_layer(
        k3_rocm_blas_context *blas,
        const loaded_tensor tensors[K3_TENSOR_COUNT],
        const q8_tensor q8_tensors[K3_TENSOR_COUNT],
        void *d_hidden,
        void *d_q_a,
        void *d_q_a_norm,
        void *d_q,
        void *d_kv,
        void *d_cache,
        uint32_t cache_row,
        uint32_t token_count,
        void *d_packed_k,
        void *d_absorbed_q,
        void *d_scores,
        void *d_probabilities,
        void *d_latent,
        void *d_value,
        void *d_gate,
        void *d_gated,
        void *d_output) {
    hip_bfloat16 *cache_row_pointer =
        (hip_bfloat16 *)d_cache + (uint64_t)cache_row * K3_CACHE_DIM;
    return
        launch_gemv(
            q8_tensors ? &q8_tensors[T_Q_A] : NULL,
            d_q_a, tensors[T_Q_A].device, d_hidden,
            K3_Q_LORA, K3_HIDDEN) &&
        k3_rocm_rms_norm_bf16(
            d_q_a_norm, d_q_a, tensors[T_Q_A_NORM].device,
            1u, K3_Q_LORA, 1e-5f, NULL) &&
        launch_gemv(
            q8_tensors ? &q8_tensors[T_Q_B] : NULL,
            d_q, tensors[T_Q_B].device, d_q_a_norm,
            K3_Q, K3_Q_LORA) &&
        launch_gemv(
            q8_tensors ? &q8_tensors[T_KV_A] : NULL,
            d_kv, tensors[T_KV_A].device, d_hidden,
            K3_CACHE_DIM, K3_HIDDEN) &&
        k3_rocm_rms_norm_bf16(
            cache_row_pointer, d_kv, tensors[T_KV_A_NORM].device,
            1u, K3_LATENT, 1e-5f, NULL) &&
        hipMemcpyAsync(
            cache_row_pointer + K3_LATENT,
            (const hip_bfloat16 *)d_kv + K3_LATENT,
            64u * sizeof(hip_bfloat16),
            hipMemcpyDeviceToDevice, NULL) == hipSuccess &&
        k3_rocm_mla_absorb_q_bf16(
            d_absorbed_q, d_q, d_packed_k, K3_HEADS, NULL) &&
        k3_rocm_blas_mla_attention_bf16(
            blas, d_latent, d_scores, d_probabilities,
            d_absorbed_q, d_cache, K3_HEADS, token_count, NULL) &&
        k3_rocm_mla_decompress_v_bf16(
            d_value, d_latent, tensors[T_KV_B].device,
            K3_HEADS, NULL) &&
        launch_gemv(
            q8_tensors ? &q8_tensors[T_GATE] : NULL,
            d_gate, tensors[T_GATE].device, d_hidden,
            K3_INNER, K3_HIDDEN) &&
        k3_rocm_sigmoid_mul_bf16(
            d_gated, d_value, d_gate, K3_INNER, NULL) &&
        launch_gemv(
            q8_tensors ? &q8_tensors[T_OUTPUT] : NULL,
            d_output, tensors[T_OUTPUT].device, d_gated,
            K3_HIDDEN, K3_INNER);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 real MLA layer: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 real MLA layer on %s (%s)\n",
           properties.name, properties.gcnArchName);

    char error[512];
    k3_st_model model;
    if (!k3_st_model_open(&model, root, 96u, error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }
    loaded_tensor tensors[K3_TENSOR_COUNT];
    memset(tensors, 0, sizeof(tensors));
    uint64_t resident_bytes = 0;
    for (uint32_t i = 0; i < K3_TENSOR_COUNT; i++) {
        if (!load_tensor(&model, tensor_names[i], &tensors[i],
                         error, sizeof(error))) {
            fprintf(stderr, "FAIL: %s\n", error);
            return 1;
        }
        resident_bytes += tensors[i].tensor->byte_length;
    }
    printf("  static attention weights: %.3f MiB\n",
           resident_bytes / 1048576.0);
    const uint32_t q8_tensor_ids[] = {
        T_Q_A, T_Q_B, T_KV_A, T_GATE, T_OUTPUT,
    };
    q8_tensor q8_tensors[K3_TENSOR_COUNT];
    memset(q8_tensors, 0, sizeof(q8_tensors));
    uint64_t q8_source_bytes = 0;
    uint64_t q8_storage_bytes = 0;
    for (uint32_t i = 0;
         i < sizeof(q8_tensor_ids) / sizeof(q8_tensor_ids[0]);
         i++) {
        const uint32_t id = q8_tensor_ids[i];
        CHECK(quantize_tensor(&tensors[id], &q8_tensors[id]),
              "real MLA Q8 projection conversion");
        q8_source_bytes += tensors[id].tensor->byte_length;
        q8_storage_bytes += q8_tensors[id].storage_bytes;
    }
    HIP_CHECK(hipDeviceSynchronize());
    printf("  Q8 projection tier: %.3f MiB from %.3f MiB (%.2f%%); "
           "kv_b stays BF16\n",
           q8_storage_bytes / 1048576.0,
           q8_source_bytes / 1048576.0,
           100.0 * (double)q8_storage_bytes /
               (double)q8_source_bytes);

    const size_t hidden_bytes =
        (size_t)K3_HIDDEN * sizeof(hip_bfloat16);
    const size_t q_lora_bytes =
        (size_t)K3_Q_LORA * sizeof(hip_bfloat16);
    const size_t q_bytes =
        (size_t)K3_Q * sizeof(hip_bfloat16);
    const size_t cache_bytes =
        (size_t)K3_CONTEXT * K3_CACHE_DIM * sizeof(hip_bfloat16);
    const size_t packed_bytes =
        (size_t)K3_HEADS * K3_LATENT * 128u * sizeof(hip_bfloat16);
    const size_t absorbed_bytes =
        (size_t)K3_HEADS * K3_CACHE_DIM * sizeof(hip_bfloat16);
    const size_t score_bytes =
        (size_t)K3_HEADS * K3_CONTEXT * sizeof(float);
    const size_t probability_bytes =
        (size_t)K3_HEADS * K3_CONTEXT * sizeof(hip_bfloat16);
    const size_t latent_bytes =
        (size_t)K3_HEADS * K3_LATENT * sizeof(hip_bfloat16);
    const size_t inner_bytes =
        (size_t)K3_INNER * sizeof(hip_bfloat16);

    hip_bfloat16 *hidden =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *expected_q_a =
        (hip_bfloat16 *)malloc(q_lora_bytes);
    hip_bfloat16 *expected_q_a_norm =
        (hip_bfloat16 *)malloc(q_lora_bytes);
    hip_bfloat16 *expected_q =
        (hip_bfloat16 *)malloc(q_bytes);
    hip_bfloat16 *expected_kv =
        (hip_bfloat16 *)malloc(K3_CACHE_DIM * sizeof(hip_bfloat16));
    hip_bfloat16 *expected_cache =
        (hip_bfloat16 *)malloc(K3_CACHE_DIM * sizeof(hip_bfloat16));
    hip_bfloat16 *expected_absorbed =
        (hip_bfloat16 *)malloc(absorbed_bytes);
    hip_bfloat16 *expected_latent =
        (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *expected_value =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_gate =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_gated =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_output =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *got =
        (hip_bfloat16 *)malloc(absorbed_bytes);
    CHECK(hidden && expected_q_a && expected_q_a_norm && expected_q &&
          expected_kv && expected_cache && expected_absorbed &&
          expected_latent && expected_value && expected_gate && expected_gated &&
          expected_output && got,
          "real MLA host allocation");
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        hidden[i] = float_to_bf16(random_input());
    }

    reference_gemv(
        expected_q_a,
        (const hip_bfloat16 *)tensors[T_Q_A].read.data,
        hidden, K3_Q_LORA, K3_HIDDEN);
    reference_rms_norm(
        expected_q_a_norm, expected_q_a,
        (const hip_bfloat16 *)tensors[T_Q_A_NORM].read.data,
        K3_Q_LORA);
    reference_gemv(
        expected_q,
        (const hip_bfloat16 *)tensors[T_Q_B].read.data,
        expected_q_a_norm, K3_Q, K3_Q_LORA);
    reference_gemv(
        expected_kv,
        (const hip_bfloat16 *)tensors[T_KV_A].read.data,
        hidden, K3_CACHE_DIM, K3_HIDDEN);
    reference_rms_norm(
        expected_cache, expected_kv,
        (const hip_bfloat16 *)tensors[T_KV_A_NORM].read.data,
        K3_LATENT);
    memcpy(expected_cache + K3_LATENT, expected_kv + K3_LATENT,
           64u * sizeof(hip_bfloat16));
    reference_absorb_q(
        expected_absorbed, expected_q,
        (const hip_bfloat16 *)tensors[T_KV_B].read.data);
    for (uint32_t head = 0; head < K3_HEADS; head++) {
        memcpy(expected_latent + (uint64_t)head * K3_LATENT,
               expected_cache,
               K3_LATENT * sizeof(hip_bfloat16));
    }
    reference_decompress_v(
        expected_value, expected_latent,
        (const hip_bfloat16 *)tensors[T_KV_B].read.data);
    reference_gemv(
        expected_gate,
        (const hip_bfloat16 *)tensors[T_GATE].read.data,
        hidden, K3_INNER, K3_HIDDEN);
    for (uint32_t i = 0; i < K3_INNER; i++) {
        expected_gated[i] = float_to_bf16(
            bf16_to_float(expected_value[i]) /
            (1.0f + expf(-bf16_to_float(expected_gate[i]))));
    }
    reference_gemv(
        expected_output,
        (const hip_bfloat16 *)tensors[T_OUTPUT].read.data,
        expected_gated, K3_HIDDEN, K3_INNER);

    void *d_hidden = NULL;
    void *d_q_a = NULL;
    void *d_q_a_norm = NULL;
    void *d_q = NULL;
    void *d_kv = NULL;
    void *d_cache = NULL;
    void *d_packed_k = NULL;
    void *d_absorbed_q = NULL;
    void *d_scores = NULL;
    void *d_probabilities = NULL;
    void *d_latent = NULL;
    void *d_value = NULL;
    void *d_gate = NULL;
    void *d_gated = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_hidden, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_q_a, q_lora_bytes));
    HIP_CHECK(hipMalloc(&d_q_a_norm, q_lora_bytes));
    HIP_CHECK(hipMalloc(&d_q, q_bytes));
    HIP_CHECK(hipMalloc(&d_kv,
                        K3_CACHE_DIM * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_cache, cache_bytes));
    HIP_CHECK(hipMalloc(&d_packed_k, packed_bytes));
    HIP_CHECK(hipMalloc(&d_absorbed_q, absorbed_bytes));
    HIP_CHECK(hipMalloc(&d_scores, score_bytes));
    HIP_CHECK(hipMalloc(&d_probabilities, probability_bytes));
    HIP_CHECK(hipMalloc(&d_latent, latent_bytes));
    HIP_CHECK(hipMalloc(&d_value, inner_bytes));
    HIP_CHECK(hipMalloc(&d_gate, inner_bytes));
    HIP_CHECK(hipMalloc(&d_gated, inner_bytes));
    HIP_CHECK(hipMalloc(&d_output, hidden_bytes));
    HIP_CHECK(hipMemcpy(d_hidden, hidden, hidden_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_cache, 0, cache_bytes));
    CHECK(k3_rocm_mla_pack_k_weight_bf16(
              d_packed_k, tensors[T_KV_B].device, K3_HEADS, NULL),
          "real MLA K-weight pack");
    k3_rocm_blas_context *blas = NULL;
    CHECK(k3_rocm_blas_context_create(&blas),
          "real MLA rocBLAS context");
    CHECK(launch_layer(
              blas, tensors, NULL, d_hidden, d_q_a, d_q_a_norm, d_q, d_kv,
              d_cache, 0u, 1u, d_packed_k, d_absorbed_q,
              d_scores, d_probabilities, d_latent, d_value,
              d_gate, d_gated, d_output),
          "real MLA one-token launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got, d_absorbed_q, absorbed_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("absorbed_q", got, expected_absorbed,
                         K3_HEADS * K3_CACHE_DIM, 0.015625f, 0.02f),
          "real MLA absorbed-query mismatch");
    HIP_CHECK(hipMemcpy(got, d_value, inner_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("value", got, expected_value,
                         K3_INNER, 0.015625f, 0.03f),
          "real MLA value mismatch");
    HIP_CHECK(hipMemcpy(got, d_output, hidden_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("attention", got, expected_output,
                         K3_HIDDEN, 0.03125f, 0.03f),
          "real MLA output mismatch");
    printf("  attention output hash: 0x%016llx\n",
           (unsigned long long)fnv1a64(got, hidden_bytes));

    HIP_CHECK(hipMemset(d_cache, 0, cache_bytes));
    CHECK(launch_layer(
              blas, tensors, NULL, d_hidden, d_q_a, d_q_a_norm, d_q, d_kv,
              d_cache, K3_CONTEXT - 1u, K3_CONTEXT,
              d_packed_k, d_absorbed_q,
              d_scores, d_probabilities, d_latent, d_value,
              d_gate, d_gated, d_output),
          "real MLA 8k warmup");
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t i = 0; i < K3_BENCHMARK_STEPS; i++) {
        CHECK(launch_layer(
                  blas, tensors, NULL, d_hidden,
                  d_q_a, d_q_a_norm, d_q, d_kv,
                  d_cache, K3_CONTEXT - 1u, K3_CONTEXT,
                  d_packed_k, d_absorbed_q,
                  d_scores, d_probabilities, d_latent, d_value,
                  d_gate, d_gated, d_output),
              "timed real MLA layer launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("  complete MLA attention, ctx=8192: "
           "%.3f ms/token/layer (mean of %u)\n",
           elapsed_ms / (float)K3_BENCHMARK_STEPS,
           K3_BENCHMARK_STEPS);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    HIP_CHECK(hipMemset(d_cache, 0, cache_bytes));
    CHECK(launch_layer(
              blas, tensors, q8_tensors,
              d_hidden, d_q_a, d_q_a_norm, d_q, d_kv,
              d_cache, 0u, 1u, d_packed_k, d_absorbed_q,
              d_scores, d_probabilities, d_latent, d_value,
              d_gate, d_gated, d_output),
          "Q8 real MLA one-token launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got, d_output, hidden_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("Q8 output", got, expected_output,
                         K3_HIDDEN, 0.03125f, 0.10f),
          "Q8 real MLA output mismatch");
    double q8_squared_error = 0.0;
    double q8_squared_reference = 0.0;
    double q8_squared_output = 0.0;
    double q8_dot = 0.0;
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        const double want = bf16_to_float(expected_output[i]);
        const double actual = bf16_to_float(got[i]);
        const double difference = actual - want;
        q8_squared_error += difference * difference;
        q8_squared_reference += want * want;
        q8_squared_output += actual * actual;
        q8_dot += want * actual;
    }
    const double q8_nrmse =
        sqrt(q8_squared_error / q8_squared_reference);
    const double q8_cosine =
        q8_dot / sqrt(q8_squared_reference * q8_squared_output);
    CHECK(isfinite(q8_nrmse) && q8_nrmse < 0.03,
          "Q8 real MLA NRMSE exceeds prototype gate");
    CHECK(isfinite(q8_cosine) && q8_cosine > 0.999,
          "Q8 real MLA cosine below prototype gate");
    printf("  Q8 attention output: nrmse=%.7f cosine=%.9f "
           "hash=0x%016llx\n",
           q8_nrmse, q8_cosine,
           (unsigned long long)fnv1a64(got, hidden_bytes));

    HIP_CHECK(hipMemset(d_cache, 0, cache_bytes));
    CHECK(launch_layer(
              blas, tensors, q8_tensors,
              d_hidden, d_q_a, d_q_a_norm, d_q, d_kv,
              d_cache, K3_CONTEXT - 1u, K3_CONTEXT,
              d_packed_k, d_absorbed_q,
              d_scores, d_probabilities, d_latent, d_value,
              d_gate, d_gated, d_output),
          "Q8 real MLA 8k warmup");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t i = 0; i < K3_BENCHMARK_STEPS; i++) {
        CHECK(launch_layer(
                  blas, tensors, q8_tensors,
                  d_hidden, d_q_a, d_q_a_norm, d_q, d_kv,
                  d_cache, K3_CONTEXT - 1u, K3_CONTEXT,
                  d_packed_k, d_absorbed_q,
                  d_scores, d_probabilities, d_latent, d_value,
                  d_gate, d_gated, d_output),
              "timed Q8 MLA layer launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("  Q8 complete MLA attention, ctx=8192: "
           "%.3f ms/token/layer (mean of %u)\n",
           elapsed_ms / (float)K3_BENCHMARK_STEPS,
           K3_BENCHMARK_STEPS);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    k3_rocm_blas_context_destroy(blas);

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_gated));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_value));
    HIP_CHECK(hipFree(d_latent));
    HIP_CHECK(hipFree(d_probabilities));
    HIP_CHECK(hipFree(d_scores));
    HIP_CHECK(hipFree(d_absorbed_q));
    HIP_CHECK(hipFree(d_packed_k));
    HIP_CHECK(hipFree(d_cache));
    HIP_CHECK(hipFree(d_kv));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_q_a_norm));
    HIP_CHECK(hipFree(d_q_a));
    HIP_CHECK(hipFree(d_hidden));
    free(got);
    free(expected_output);
    free(expected_gated);
    free(expected_gate);
    free(expected_value);
    free(expected_latent);
    free(expected_absorbed);
    free(expected_cache);
    free(expected_kv);
    free(expected_q);
    free(expected_q_a_norm);
    free(expected_q_a);
    free(hidden);
    for (uint32_t i = 0; i < K3_TENSOR_COUNT; i++) {
        release_q8_tensor(&q8_tensors[i]);
    }
    for (uint32_t i = 0; i < K3_TENSOR_COUNT; i++) {
        release_tensor(&tensors[i]);
    }
    k3_st_model_close(&model);
    printf("K3 real MLA layer: PASS\n");
    return 0;
}
