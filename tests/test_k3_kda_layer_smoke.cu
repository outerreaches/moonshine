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
    K3_HEAD_DIM = 128,
    K3_INNER = K3_HEADS * K3_HEAD_DIM,
    K3_REDUCTION_THREADS = 256,
    K3_TENSOR_COUNT = 14,
    K3_BENCHMARK_STEPS = 10,
};

enum {
    T_Q_PROJ,
    T_K_PROJ,
    T_V_PROJ,
    T_Q_CONV,
    T_K_CONV,
    T_V_CONV,
    T_F_A,
    T_F_B,
    T_B_PROJ,
    T_A_LOG,
    T_DT_BIAS,
    T_G_PROJ,
    T_O_NORM,
    T_O_PROJ,
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
    "language_model.model.layers.0.self_attn.q_proj.weight",
    "language_model.model.layers.0.self_attn.k_proj.weight",
    "language_model.model.layers.0.self_attn.v_proj.weight",
    "language_model.model.layers.0.self_attn.q_conv1d.weight",
    "language_model.model.layers.0.self_attn.k_conv1d.weight",
    "language_model.model.layers.0.self_attn.v_conv1d.weight",
    "language_model.model.layers.0.self_attn.f_a_proj.weight",
    "language_model.model.layers.0.self_attn.f_b_proj.weight",
    "language_model.model.layers.0.self_attn.b_proj.weight",
    "language_model.model.layers.0.self_attn.A_log",
    "language_model.model.layers.0.self_attn.dt_bias",
    "language_model.model.layers.0.self_attn.g_proj.weight",
    "language_model.model.layers.0.self_attn.o_norm.weight",
    "language_model.model.layers.0.self_attn.o_proj.weight",
};

static uint32_t rng_state = UINT32_C(0x4b334c30);

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

/*
 * Reproduce k3_rocm_bf16_gemv_bf16's per-thread accumulation and reduction
 * tree so any layer mismatch points beyond the dense projection primitive.
 */
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

static void reference_conv_zero_state(hip_bfloat16 *output,
                                      const hip_bfloat16 *input,
                                      const float *weight) {
    for (uint32_t channel = 0; channel < K3_INNER; channel++) {
        float convolution =
            bf16_to_float(input[channel]) * weight[channel * 4u + 3u];
        output[channel] = float_to_bf16(
            convolution / (1.0f + expf(-convolution)));
    }
}

static void reference_kda(hip_bfloat16 *output,
                          float *state,
                          const hip_bfloat16 *q,
                          const hip_bfloat16 *k,
                          const hip_bfloat16 *v,
                          const hip_bfloat16 *raw_gate,
                          const hip_bfloat16 *raw_beta,
                          const float *A_log,
                          const float *dt_bias) {
    const float scale = 1.0f / sqrtf((float)K3_HEAD_DIM);
    for (uint32_t head = 0; head < K3_HEADS; head++) {
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
                    -5.0f / (1.0f + expf(-A * gate_input));
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

static void reference_gated_norm(hip_bfloat16 *output,
                                 const hip_bfloat16 *input,
                                 const hip_bfloat16 *gate,
                                 const float *weight) {
    for (uint32_t head = 0; head < K3_HEADS; head++) {
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
            output[base + d] = float_to_bf16(
                bf16_to_float(input[base + d]) *
                reciprocal_std * weight[d] * sigmoid_gate);
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

static bool launch_gemv(k3_rocm_blas_context *blas,
                        const q8_tensor *q8,
                        void *output,
                        const void *weights,
                        const void *input,
                        uint32_t rows,
                        uint32_t columns) {
    return q8 && q8->quantized ?
        k3_rocm_q8_128_gemv_bf16(
            output, q8->quantized, q8->scales, input,
            rows, columns, NULL) :
        blas ?
        k3_rocm_blas_bf16_gemv_bf16(
            blas, output, weights, input, rows, columns, NULL) :
        k3_rocm_bf16_gemv_bf16(
            output, weights, input, rows, columns, NULL);
}

static bool launch_layer(k3_rocm_blas_context *blas,
                         const loaded_tensor tensors[K3_TENSOR_COUNT],
                         const q8_tensor q8_tensors[K3_TENSOR_COUNT],
                         void *d_hidden,
                         void *d_q_projection,
                         void *d_k_projection,
                         void *d_v_projection,
                         void *d_q,
                         void *d_k,
                         void *d_v,
                         void *d_q_cache,
                         void *d_k_cache,
                         void *d_v_cache,
                         void *d_f_a,
                         void *d_raw_gate,
                         void *d_raw_beta,
                         void *d_state,
                         void *d_kda,
                         void *d_output_gate,
                         void *d_gated,
                         void *d_output) {
    return
        launch_gemv(
            blas, q8_tensors ? &q8_tensors[T_Q_PROJ] : NULL,
            d_q_projection, tensors[T_Q_PROJ].device, d_hidden,
            K3_INNER, K3_HIDDEN) &&
        launch_gemv(
            blas, q8_tensors ? &q8_tensors[T_K_PROJ] : NULL,
            d_k_projection, tensors[T_K_PROJ].device, d_hidden,
            K3_INNER, K3_HIDDEN) &&
        launch_gemv(
            blas, q8_tensors ? &q8_tensors[T_V_PROJ] : NULL,
            d_v_projection, tensors[T_V_PROJ].device, d_hidden,
            K3_INNER, K3_HIDDEN) &&
        k3_rocm_short_conv4_silu_bf16_f32_weight(
            d_q, d_q_cache, d_q_projection, tensors[T_Q_CONV].device,
            K3_INNER, NULL) &&
        k3_rocm_short_conv4_silu_bf16_f32_weight(
            d_k, d_k_cache, d_k_projection, tensors[T_K_CONV].device,
            K3_INNER, NULL) &&
        k3_rocm_short_conv4_silu_bf16_f32_weight(
            d_v, d_v_cache, d_v_projection, tensors[T_V_CONV].device,
            K3_INNER, NULL) &&
        launch_gemv(
            blas, q8_tensors ? &q8_tensors[T_F_A] : NULL,
            d_f_a, tensors[T_F_A].device, d_hidden,
            K3_HEAD_DIM, K3_HIDDEN) &&
        launch_gemv(
            blas, q8_tensors ? &q8_tensors[T_F_B] : NULL,
            d_raw_gate, tensors[T_F_B].device, d_f_a,
            K3_INNER, K3_HEAD_DIM) &&
        launch_gemv(
            blas, q8_tensors ? &q8_tensors[T_B_PROJ] : NULL,
            d_raw_beta, tensors[T_B_PROJ].device, d_hidden,
            K3_HEADS, K3_HIDDEN) &&
        k3_rocm_kda_recurrent_bf16_f32_state(
            d_kda, d_state, d_q, d_k, d_v, d_raw_gate, d_raw_beta,
            tensors[T_A_LOG].device, tensors[T_DT_BIAS].device,
            K3_HEADS, K3_HEAD_DIM, -5.0f, NULL) &&
        launch_gemv(
            blas, q8_tensors ? &q8_tensors[T_G_PROJ] : NULL,
            d_output_gate, tensors[T_G_PROJ].device, d_hidden,
            K3_INNER, K3_HIDDEN) &&
        k3_rocm_rms_norm_sigmoid_gate_bf16_f32_weight(
            d_gated, d_kda, d_output_gate, tensors[T_O_NORM].device,
            K3_HEADS, K3_HEAD_DIM, 1e-5f, NULL) &&
        launch_gemv(
            blas, q8_tensors ? &q8_tensors[T_O_PROJ] : NULL,
            d_output, tensors[T_O_PROJ].device, d_gated,
            K3_HIDDEN, K3_INNER);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 real KDA layer: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 real KDA layer on %s (%s)\n",
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
    CHECK(tensors[T_A_LOG].tensor->dtype == K3_ST_DTYPE_F32 &&
          tensors[T_A_LOG].tensor->shape[0] == 128u,
          "A_log must have 96 active values plus 32-value padding");
    const float *A_log =
        (const float *)tensors[T_A_LOG].read.data;
    for (uint32_t i = K3_HEADS; i < 128u; i++) {
        CHECK(A_log[i] == 0.0f, "A_log padding is not zero");
    }
    printf("  static attention weights: %.3f MiB; A_log padding verified\n",
           resident_bytes / 1048576.0);
    const uint32_t q8_tensor_ids[] = {
        T_Q_PROJ, T_K_PROJ, T_V_PROJ, T_F_A,
        T_F_B, T_B_PROJ, T_G_PROJ, T_O_PROJ,
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
              "real KDA Q8 projection conversion");
        q8_source_bytes += tensors[id].tensor->byte_length;
        q8_storage_bytes += q8_tensors[id].storage_bytes;
    }
    HIP_CHECK(hipDeviceSynchronize());
    printf("  Q8 projection tier: %.3f MiB from %.3f MiB (%.2f%%)\n",
           q8_storage_bytes / 1048576.0,
           q8_source_bytes / 1048576.0,
           100.0 * (double)q8_storage_bytes /
               (double)q8_source_bytes);

    const size_t hidden_bytes =
        (size_t)K3_HIDDEN * sizeof(hip_bfloat16);
    const size_t inner_bytes =
        (size_t)K3_INNER * sizeof(hip_bfloat16);
    const size_t cache_bytes =
        (size_t)K3_INNER * 4u * sizeof(hip_bfloat16);
    const size_t state_bytes =
        (size_t)K3_HEADS * K3_HEAD_DIM * K3_HEAD_DIM * sizeof(float);
    hip_bfloat16 *hidden =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *expected_q_projection =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_k_projection =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_v_projection =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_q = (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_k = (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_v = (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_f_a =
        (hip_bfloat16 *)malloc(K3_HEAD_DIM * sizeof(hip_bfloat16));
    hip_bfloat16 *expected_raw_gate =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_raw_beta =
        (hip_bfloat16 *)malloc(K3_HEADS * sizeof(hip_bfloat16));
    hip_bfloat16 *expected_kda =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_output_gate =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_gated =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *expected_output =
        (hip_bfloat16 *)malloc(hidden_bytes);
    float *expected_state = (float *)calloc(1, state_bytes);
    hip_bfloat16 *got_inner =
        (hip_bfloat16 *)malloc(inner_bytes);
    hip_bfloat16 *got_output =
        (hip_bfloat16 *)malloc(hidden_bytes);
    CHECK(hidden && expected_q_projection && expected_k_projection &&
          expected_v_projection && expected_q && expected_k && expected_v &&
          expected_f_a && expected_raw_gate && expected_raw_beta &&
          expected_kda && expected_output_gate && expected_gated &&
          expected_output && expected_state && got_inner && got_output,
          "real KDA host activation allocation");
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        hidden[i] = float_to_bf16(random_input());
    }

    reference_gemv(
        expected_q_projection,
        (const hip_bfloat16 *)tensors[T_Q_PROJ].read.data,
        hidden, K3_INNER, K3_HIDDEN);
    reference_gemv(
        expected_k_projection,
        (const hip_bfloat16 *)tensors[T_K_PROJ].read.data,
        hidden, K3_INNER, K3_HIDDEN);
    reference_gemv(
        expected_v_projection,
        (const hip_bfloat16 *)tensors[T_V_PROJ].read.data,
        hidden, K3_INNER, K3_HIDDEN);
    reference_conv_zero_state(
        expected_q, expected_q_projection,
        (const float *)tensors[T_Q_CONV].read.data);
    reference_conv_zero_state(
        expected_k, expected_k_projection,
        (const float *)tensors[T_K_CONV].read.data);
    reference_conv_zero_state(
        expected_v, expected_v_projection,
        (const float *)tensors[T_V_CONV].read.data);
    reference_gemv(
        expected_f_a,
        (const hip_bfloat16 *)tensors[T_F_A].read.data,
        hidden, K3_HEAD_DIM, K3_HIDDEN);
    reference_gemv(
        expected_raw_gate,
        (const hip_bfloat16 *)tensors[T_F_B].read.data,
        expected_f_a, K3_INNER, K3_HEAD_DIM);
    reference_gemv(
        expected_raw_beta,
        (const hip_bfloat16 *)tensors[T_B_PROJ].read.data,
        hidden, K3_HEADS, K3_HIDDEN);
    reference_kda(
        expected_kda, expected_state,
        expected_q, expected_k, expected_v,
        expected_raw_gate, expected_raw_beta,
        A_log, (const float *)tensors[T_DT_BIAS].read.data);
    reference_gemv(
        expected_output_gate,
        (const hip_bfloat16 *)tensors[T_G_PROJ].read.data,
        hidden, K3_INNER, K3_HIDDEN);
    reference_gated_norm(
        expected_gated, expected_kda, expected_output_gate,
        (const float *)tensors[T_O_NORM].read.data);
    reference_gemv(
        expected_output,
        (const hip_bfloat16 *)tensors[T_O_PROJ].read.data,
        expected_gated, K3_HIDDEN, K3_INNER);

    void *d_hidden = NULL;
    void *d_q_projection = NULL;
    void *d_k_projection = NULL;
    void *d_v_projection = NULL;
    void *d_q = NULL;
    void *d_k = NULL;
    void *d_v = NULL;
    void *d_q_cache = NULL;
    void *d_k_cache = NULL;
    void *d_v_cache = NULL;
    void *d_f_a = NULL;
    void *d_raw_gate = NULL;
    void *d_raw_beta = NULL;
    void *d_state = NULL;
    void *d_kda = NULL;
    void *d_output_gate = NULL;
    void *d_gated = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_hidden, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_q_projection, inner_bytes));
    HIP_CHECK(hipMalloc(&d_k_projection, inner_bytes));
    HIP_CHECK(hipMalloc(&d_v_projection, inner_bytes));
    HIP_CHECK(hipMalloc(&d_q, inner_bytes));
    HIP_CHECK(hipMalloc(&d_k, inner_bytes));
    HIP_CHECK(hipMalloc(&d_v, inner_bytes));
    HIP_CHECK(hipMalloc(&d_q_cache, cache_bytes));
    HIP_CHECK(hipMalloc(&d_k_cache, cache_bytes));
    HIP_CHECK(hipMalloc(&d_v_cache, cache_bytes));
    HIP_CHECK(hipMalloc(&d_f_a,
                        K3_HEAD_DIM * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_raw_gate, inner_bytes));
    HIP_CHECK(hipMalloc(&d_raw_beta,
                        K3_HEADS * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_state, state_bytes));
    HIP_CHECK(hipMalloc(&d_kda, inner_bytes));
    HIP_CHECK(hipMalloc(&d_output_gate, inner_bytes));
    HIP_CHECK(hipMalloc(&d_gated, inner_bytes));
    HIP_CHECK(hipMalloc(&d_output, hidden_bytes));
    HIP_CHECK(hipMemcpy(d_hidden, hidden, hidden_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_q_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_k_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_v_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_state, 0, state_bytes));
    CHECK(launch_layer(
              NULL,
              tensors, NULL, d_hidden,
              d_q_projection, d_k_projection, d_v_projection,
              d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
              d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
              d_output_gate, d_gated, d_output),
          "real KDA layer launch");
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(got_inner, d_q, inner_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("q_conv", got_inner, expected_q,
                         K3_INNER, 0.00390625f, 0.001f),
          "real q-conv mismatch");
    HIP_CHECK(hipMemcpy(got_inner, d_raw_gate, inner_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("raw_gate", got_inner, expected_raw_gate,
                         K3_INNER, 0.015625f, 0.001f),
          "real raw-gate mismatch");
    HIP_CHECK(hipMemcpy(got_inner, d_kda, inner_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("recurrence", got_inner, expected_kda,
                         K3_INNER, 0.00390625f, 0.01f),
          "real recurrence mismatch");
    HIP_CHECK(hipMemcpy(got_inner, d_gated, inner_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("gated_norm", got_inner, expected_gated,
                         K3_INNER, 0.00390625f, 0.01f),
          "real gated-norm mismatch");
    HIP_CHECK(hipMemcpy(got_output, d_output, hidden_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("attention", got_output, expected_output,
                         K3_HIDDEN, 0.015625f, 0.01f),
          "real attention mismatch");
    printf("  attention output hash: 0x%016llx\n",
           (unsigned long long)fnv1a64(got_output, hidden_bytes));

    /* Reset recurrent state before measuring repeat-token decode. */
    HIP_CHECK(hipMemset(d_q_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_k_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_v_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_state, 0, state_bytes));
    CHECK(launch_layer(
              NULL,
              tensors, NULL, d_hidden,
              d_q_projection, d_k_projection, d_v_projection,
              d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
              d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
              d_output_gate, d_gated, d_output),
          "real KDA warmup");
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t i = 0; i < K3_BENCHMARK_STEPS; i++) {
        CHECK(launch_layer(
                  NULL,
                  tensors, NULL, d_hidden,
                  d_q_projection, d_k_projection, d_v_projection,
                  d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
                  d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
                  d_output_gate, d_gated, d_output),
              "timed real KDA layer launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("  naive complete KDA attention: %.3f ms/token/layer "
           "(mean of %u)\n",
           elapsed_ms / (float)K3_BENCHMARK_STEPS,
           K3_BENCHMARK_STEPS);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    k3_rocm_blas_context *blas = NULL;
    CHECK(k3_rocm_blas_context_create(&blas),
          "rocBLAS context creation");
    HIP_CHECK(hipMemset(d_q_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_k_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_v_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_state, 0, state_bytes));
    CHECK(launch_layer(
              blas,
              tensors, NULL, d_hidden,
              d_q_projection, d_k_projection, d_v_projection,
              d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
              d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
              d_output_gate, d_gated, d_output),
          "rocBLAS real KDA layer launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got_output, d_output, hidden_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("rocBLAS out", got_output, expected_output,
                         K3_HIDDEN, 0.125f, 0.03f),
          "rocBLAS real attention mismatch");

    HIP_CHECK(hipMemset(d_q_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_k_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_v_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_state, 0, state_bytes));
    CHECK(launch_layer(
              blas,
              tensors, NULL, d_hidden,
              d_q_projection, d_k_projection, d_v_projection,
              d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
              d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
              d_output_gate, d_gated, d_output),
          "rocBLAS real KDA warmup");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t i = 0; i < K3_BENCHMARK_STEPS; i++) {
        CHECK(launch_layer(
                  blas,
                  tensors, NULL, d_hidden,
                  d_q_projection, d_k_projection, d_v_projection,
                  d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
                  d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
                  d_output_gate, d_gated, d_output),
              "timed rocBLAS KDA layer launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("  rocBLAS complete KDA attention: %.3f ms/token/layer "
           "(mean of %u)\n",
           elapsed_ms / (float)K3_BENCHMARK_STEPS,
           K3_BENCHMARK_STEPS);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    k3_rocm_blas_context_destroy(blas);

    HIP_CHECK(hipMemset(d_q_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_k_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_v_cache, 0, cache_bytes));
    HIP_CHECK(hipMemset(d_state, 0, state_bytes));
    CHECK(launch_layer(
              NULL, tensors, q8_tensors, d_hidden,
              d_q_projection, d_k_projection, d_v_projection,
              d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
              d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
              d_output_gate, d_gated, d_output),
          "Q8 real KDA layer launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got_output, d_output, hidden_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(compare_vector("Q8 output", got_output, expected_output,
                         K3_HIDDEN, 0.015625f, 0.10f),
          "Q8 real attention mismatch");
    double q8_squared_error = 0.0;
    double q8_squared_reference = 0.0;
    double q8_squared_output = 0.0;
    double q8_dot = 0.0;
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        const double want = bf16_to_float(expected_output[i]);
        const double got = bf16_to_float(got_output[i]);
        const double difference = got - want;
        q8_squared_error += difference * difference;
        q8_squared_reference += want * want;
        q8_squared_output += got * got;
        q8_dot += want * got;
    }
    const double q8_nrmse =
        sqrt(q8_squared_error / q8_squared_reference);
    const double q8_cosine =
        q8_dot / sqrt(q8_squared_reference * q8_squared_output);
    CHECK(isfinite(q8_nrmse) && q8_nrmse < 0.03,
          "Q8 real KDA NRMSE exceeds prototype gate");
    CHECK(isfinite(q8_cosine) && q8_cosine > 0.999,
          "Q8 real KDA cosine below prototype gate");
    printf("  Q8 attention output: nrmse=%.7f cosine=%.9f "
           "hash=0x%016llx\n",
           q8_nrmse, q8_cosine,
           (unsigned long long)fnv1a64(got_output, hidden_bytes));

    CHECK(launch_layer(
              NULL, tensors, q8_tensors, d_hidden,
              d_q_projection, d_k_projection, d_v_projection,
              d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
              d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
              d_output_gate, d_gated, d_output),
          "Q8 real KDA warmup");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t i = 0; i < K3_BENCHMARK_STEPS; i++) {
        CHECK(launch_layer(
                  NULL, tensors, q8_tensors, d_hidden,
                  d_q_projection, d_k_projection, d_v_projection,
                  d_q, d_k, d_v, d_q_cache, d_k_cache, d_v_cache,
                  d_f_a, d_raw_gate, d_raw_beta, d_state, d_kda,
                  d_output_gate, d_gated, d_output),
              "timed Q8 KDA layer launch");
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("  Q8 complete KDA attention: %.3f ms/token/layer "
           "(mean of %u)\n",
           elapsed_ms / (float)K3_BENCHMARK_STEPS,
           K3_BENCHMARK_STEPS);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_gated));
    HIP_CHECK(hipFree(d_output_gate));
    HIP_CHECK(hipFree(d_kda));
    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_raw_beta));
    HIP_CHECK(hipFree(d_raw_gate));
    HIP_CHECK(hipFree(d_f_a));
    HIP_CHECK(hipFree(d_v_cache));
    HIP_CHECK(hipFree(d_k_cache));
    HIP_CHECK(hipFree(d_q_cache));
    HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_v_projection));
    HIP_CHECK(hipFree(d_k_projection));
    HIP_CHECK(hipFree(d_q_projection));
    HIP_CHECK(hipFree(d_hidden));
    free(got_output);
    free(got_inner);
    free(expected_state);
    free(expected_output);
    free(expected_gated);
    free(expected_output_gate);
    free(expected_kda);
    free(expected_raw_beta);
    free(expected_raw_gate);
    free(expected_f_a);
    free(expected_v);
    free(expected_k);
    free(expected_q);
    free(expected_v_projection);
    free(expected_k_projection);
    free(expected_q_projection);
    free(hidden);
    for (uint32_t i = 0; i < K3_TENSOR_COUNT; i++) {
        release_q8_tensor(&q8_tensors[i]);
    }
    for (uint32_t i = 0; i < K3_TENSOR_COUNT; i++) {
        release_tensor(&tensors[i]);
    }
    k3_st_model_close(&model);
    printf("K3 real KDA layer: PASS\n");
    return 0;
}
