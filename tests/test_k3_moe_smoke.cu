#include "k3_rocm_ops.h"
#include "k3_safetensors.h"
#include "k3_io_uring.h"
#include "k3_expert_cache.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <inttypes.h>
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
    K3_LATENT = 3584,
    K3_EXPERT_HIDDEN = 3072,
    K3_SHARED_HIDDEN = 6144,
    K3_EXPERT_COUNT = 896,
    K3_TOP_K = 16,
    K3_REDUCTION_THREADS = 256,
    K3_STATIC_COUNT = 8,
    K3_EXPERT_TENSOR_COUNT = 6,
    K3_STREAM_QD = 2,
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

typedef struct {
    uint64_t relative[K3_EXPERT_TENSOR_COUNT];
    uint64_t physical_start;
    uint64_t aligned_start;
    uint32_t aligned_bytes;
    uint16_t shard;
} selected_layout;

static uint32_t rng_state = UINT32_C(0x4b334d4f);

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

static void reference_bf16_gemv(float *output_f32,
                                hip_bfloat16 *output_bf16,
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
        if (output_f32) output_f32[row] = partial[0];
        if (output_bf16) output_bf16[row] = float_to_bf16(partial[0]);
    }
}

static void reference_mxfp4_gemv(hip_bfloat16 *output,
                                 const uint8_t *packed,
                                 const uint8_t *scales,
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
                uint8_t byte =
                    packed[(uint64_t)row * columns / 2u + column / 2u];
                uint8_t nibble =
                    (column & 1u) ? byte >> 4u : byte & 0x0fu;
                float weight = e2m1_to_float(nibble) *
                    e8m0_to_float(
                        scales[(uint64_t)row * columns / 32u +
                               column / 32u]);
                sum += weight * bf16_to_float(input[column]);
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

static void reference_situ(hip_bfloat16 *output,
                           const hip_bfloat16 *gate,
                           const hip_bfloat16 *up,
                           uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        float g = bf16_to_float(gate[i]);
        float u = bf16_to_float(up[i]);
        output[i] = float_to_bf16(
            (4.0f * tanhf(g / 4.0f) / (1.0f + expf(-g))) *
            (25.0f * tanhf(u / 25.0f)));
    }
}

static void reference_rms_norm(hip_bfloat16 *output,
                               const hip_bfloat16 *input,
                               const hip_bfloat16 *weight,
                               uint32_t count,
                               float epsilon) {
    float partial[K3_REDUCTION_THREADS];
    for (uint32_t tid = 0; tid < K3_REDUCTION_THREADS; tid++) {
        float sum = 0.0f;
        for (uint32_t i = tid; i < count; i += K3_REDUCTION_THREADS) {
            float value = bf16_to_float(input[i]);
            sum += value * value;
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
    float reciprocal_std = 1.0f / sqrtf(partial[0] / count + epsilon);
    for (uint32_t i = 0; i < count; i++) {
        output[i] = float_to_bf16(
            bf16_to_float(input[i]) * reciprocal_std *
            bf16_to_float(weight[i]));
    }
}

static void reference_router(uint32_t ids[K3_TOP_K],
                             float weights[K3_TOP_K],
                             const float logits[K3_EXPERT_COUNT],
                             const float bias[K3_EXPERT_COUNT]) {
    float choices[K3_TOP_K];
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        choices[rank] = -INFINITY;
        weights[rank] = 0.0f;
        ids[rank] = UINT32_MAX;
    }
    for (uint32_t expert = 0; expert < K3_EXPERT_COUNT; expert++) {
        float raw = 1.0f / (1.0f + expf(-logits[expert]));
        float choice = raw + bias[expert];
        uint32_t position = K3_TOP_K;
        for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
            if (choice > choices[rank] ||
                (choice == choices[rank] && expert < ids[rank])) {
                position = rank;
                break;
            }
        }
        if (position == K3_TOP_K) continue;
        for (uint32_t rank = K3_TOP_K - 1u; rank > position; rank--) {
            choices[rank] = choices[rank - 1u];
            weights[rank] = weights[rank - 1u];
            ids[rank] = ids[rank - 1u];
        }
        choices[position] = choice;
        weights[position] = raw;
        ids[position] = expert;
    }
    float denominator = 1e-20f;
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        denominator += weights[rank];
    }
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        weights[rank] /= denominator;
    }
}

static bool load_static_tensor(k3_st_model *model,
                               const char *name,
                               loaded_tensor *loaded,
                               uint64_t *read_bytes,
                               double *read_seconds,
                               char *error,
                               size_t error_size) {
    memset(loaded, 0, sizeof(*loaded));
    loaded->tensor = k3_st_find(model, name);
    if (!loaded->tensor) {
        snprintf(error, error_size, "missing tensor %s", name);
        return false;
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    if (!k3_st_read_span(model, loaded->tensor->shard,
                         loaded->tensor->physical_offset,
                         loaded->tensor->byte_length, 4096u,
                         &loaded->read, error, error_size)) {
        return false;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    *read_seconds += (double)(end.tv_sec - start.tv_sec) +
                     (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    *read_bytes += loaded->read.allocation_bytes;
    if (hipMalloc(&loaded->device, loaded->tensor->byte_length) != hipSuccess ||
        hipMemcpy(loaded->device, loaded->read.data,
                  loaded->tensor->byte_length,
                  hipMemcpyHostToDevice) != hipSuccess) {
        snprintf(error, error_size, "ROCm upload failed for %s", name);
        return false;
    }
    return true;
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

static bool launch_static_gemv(
        bool use_q8,
        const q8_tensor *q8,
        void *output,
        const loaded_tensor *weights,
        const void *input,
        uint32_t rows,
        uint32_t columns) {
    return use_q8 ?
        k3_rocm_q8_128_gemv_bf16(
            output, q8->quantized, q8->scales, input,
            rows, columns, NULL) :
        k3_rocm_bf16_gemv_bf16(
            output, weights->device, input, rows, columns, NULL);
}

static bool compare_output(const hip_bfloat16 *actual,
                           const hip_bfloat16 *expected,
                           uint32_t count) {
    float maximum_absolute = 0.0f;
    float maximum_relative = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float got = bf16_to_float(actual[i]);
        float want = bf16_to_float(expected[i]);
        float absolute = fabsf(got - want);
        float relative = absolute / fmaxf(fabsf(want), 1e-3f);
        if (absolute > maximum_absolute) maximum_absolute = absolute;
        if (relative > maximum_relative) maximum_relative = relative;
        if (!isfinite(got) || (absolute > 0.5f && relative > 0.05f)) {
            fprintf(stderr,
                    "FAIL: output[%u] got=%f expected=%f abs=%f rel=%f\n",
                    i, got, want, absolute, relative);
            return false;
        }
    }
    printf("  final output: max_abs=%.6f max_rel=%.6f\n",
           maximum_absolute, maximum_relative);
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

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 full MoE smoke: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));

    char error[512];
    k3_st_model model;
    CHECK(k3_st_model_open(&model, root, 96u, error, sizeof(error)), error);

    static const char *static_names[K3_STATIC_COUNT] = {
        "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias",
        "language_model.model.layers.1.block_sparse_moe.gate.weight",
        "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight",
        "language_model.model.layers.1.block_sparse_moe.routed_expert_norm.weight",
        "language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj.weight",
        "language_model.model.layers.1.block_sparse_moe.shared_experts.down_proj.weight",
        "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
        "language_model.model.layers.1.block_sparse_moe.shared_experts.up_proj.weight",
    };
    loaded_tensor static_tensor[K3_STATIC_COUNT];
    memset(static_tensor, 0, sizeof(static_tensor));
    uint64_t static_read_bytes = 0;
    double static_read_seconds = 0.0;
    for (uint32_t i = 0; i < K3_STATIC_COUNT; i++) {
        CHECK(load_static_tensor(&model, static_names[i], &static_tensor[i],
                                 &static_read_bytes, &static_read_seconds,
                                 error, sizeof(error)),
              error);
    }
    const uint32_t q8_tensor_ids[] = { 2u, 4u, 5u, 6u, 7u };
    q8_tensor q8_tensors[K3_STATIC_COUNT];
    memset(q8_tensors, 0, sizeof(q8_tensors));
    uint64_t q8_source_bytes = 0;
    uint64_t q8_storage_bytes = 0;
    for (uint32_t i = 0;
         i < sizeof(q8_tensor_ids) / sizeof(q8_tensor_ids[0]);
         i++) {
        const uint32_t id = q8_tensor_ids[i];
        CHECK(quantize_tensor(&static_tensor[id], &q8_tensors[id]),
              "real MoE Q8 static-tail conversion");
        q8_source_bytes += static_tensor[id].tensor->byte_length;
        q8_storage_bytes += q8_tensors[id].storage_bytes;
    }
    HIP_CHECK(hipDeviceSynchronize());

    const size_t hidden_bytes = K3_HIDDEN * sizeof(hip_bfloat16);
    const size_t latent_bytes = K3_LATENT * sizeof(hip_bfloat16);
    const size_t expert_hidden_bytes =
        K3_EXPERT_HIDDEN * sizeof(hip_bfloat16);
    const size_t shared_hidden_bytes =
        K3_SHARED_HIDDEN * sizeof(hip_bfloat16);
    hip_bfloat16 *input = (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *cpu_latent = (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *cpu_expert_gate =
        (hip_bfloat16 *)malloc(expert_hidden_bytes);
    hip_bfloat16 *cpu_expert_up =
        (hip_bfloat16 *)malloc(expert_hidden_bytes);
    hip_bfloat16 *cpu_expert_activation =
        (hip_bfloat16 *)malloc(expert_hidden_bytes);
    hip_bfloat16 *cpu_expert_output =
        (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *cpu_routed = (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *cpu_routed_norm = (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *cpu_routed_full = (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *cpu_shared_gate =
        (hip_bfloat16 *)malloc(shared_hidden_bytes);
    hip_bfloat16 *cpu_shared_up =
        (hip_bfloat16 *)malloc(shared_hidden_bytes);
    hip_bfloat16 *cpu_shared_activation =
        (hip_bfloat16 *)malloc(shared_hidden_bytes);
    hip_bfloat16 *cpu_shared_output =
        (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *cpu_final = (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *gpu_final = (hip_bfloat16 *)malloc(hidden_bytes);
    hip_bfloat16 *q8_final = (hip_bfloat16 *)malloc(hidden_bytes);
    float *cpu_logits = (float *)malloc(K3_EXPERT_COUNT * sizeof(float));
    float *cpu_accumulator = (float *)calloc(K3_LATENT, sizeof(float));
    CHECK(input && cpu_latent && cpu_expert_gate && cpu_expert_up &&
          cpu_expert_activation && cpu_expert_output && cpu_routed &&
          cpu_routed_norm && cpu_routed_full && cpu_shared_gate &&
          cpu_shared_up && cpu_shared_activation && cpu_shared_output &&
          cpu_final && gpu_final && q8_final && cpu_logits && cpu_accumulator,
          "host activation allocation");
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        input[i] = float_to_bf16(random_input());
    }

    hip_bfloat16 *d_input = NULL;
    float *d_logits = NULL;
    uint32_t *d_ids = NULL;
    float *d_route_weights = NULL;
    hip_bfloat16 *d_latent = NULL;
    hip_bfloat16 *d_expert_gate = NULL;
    hip_bfloat16 *d_expert_up = NULL;
    hip_bfloat16 *d_expert_activation = NULL;
    hip_bfloat16 *d_expert_outputs = NULL;
    hip_bfloat16 *d_routed = NULL;
    hip_bfloat16 *d_routed_norm = NULL;
    hip_bfloat16 *d_routed_full = NULL;
    hip_bfloat16 *d_shared_gate = NULL;
    hip_bfloat16 *d_shared_up = NULL;
    hip_bfloat16 *d_shared_activation = NULL;
    hip_bfloat16 *d_shared_output = NULL;
    hip_bfloat16 *d_final = NULL;
    HIP_CHECK(hipMalloc(&d_input, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_logits, K3_EXPERT_COUNT * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ids, K3_TOP_K * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_route_weights, K3_TOP_K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_latent, latent_bytes));
    HIP_CHECK(hipMalloc(&d_expert_gate, expert_hidden_bytes));
    HIP_CHECK(hipMalloc(&d_expert_up, expert_hidden_bytes));
    HIP_CHECK(hipMalloc(&d_expert_activation, expert_hidden_bytes));
    HIP_CHECK(hipMalloc(&d_expert_outputs, K3_TOP_K * latent_bytes));
    HIP_CHECK(hipMalloc(&d_routed, latent_bytes));
    HIP_CHECK(hipMalloc(&d_routed_norm, latent_bytes));
    HIP_CHECK(hipMalloc(&d_routed_full, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_shared_gate, shared_hidden_bytes));
    HIP_CHECK(hipMalloc(&d_shared_up, shared_hidden_bytes));
    HIP_CHECK(hipMalloc(&d_shared_activation, shared_hidden_bytes));
    HIP_CHECK(hipMalloc(&d_shared_output, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_final, hidden_bytes));
    HIP_CHECK(hipMemcpy(d_input, input, hidden_bytes, hipMemcpyHostToDevice));

    CHECK(k3_rocm_bf16_gemv_f32(d_logits, static_tensor[1].device, d_input,
                                 K3_EXPERT_COUNT, K3_HIDDEN, NULL),
          "real router GEMV");
    CHECK(k3_rocm_router_topk_f32(d_ids, d_route_weights, d_logits,
                                  static_tensor[0].device,
                                  K3_EXPERT_COUNT, K3_TOP_K, 1.0f, NULL),
          "real router top-k");
    HIP_CHECK(hipDeviceSynchronize());
    uint32_t selected_ids[K3_TOP_K];
    float selected_weights[K3_TOP_K];
    HIP_CHECK(hipMemcpy(selected_ids, d_ids, sizeof(selected_ids),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(selected_weights, d_route_weights,
                        sizeof(selected_weights), hipMemcpyDeviceToHost));

    reference_bf16_gemv(cpu_logits, NULL,
                        (const hip_bfloat16 *)static_tensor[1].read.data,
                        input, K3_EXPERT_COUNT, K3_HIDDEN);
    uint32_t cpu_ids[K3_TOP_K];
    float cpu_weights[K3_TOP_K];
    reference_router(cpu_ids, cpu_weights, cpu_logits,
                     (const float *)static_tensor[0].read.data);
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        CHECK(selected_ids[rank] == cpu_ids[rank], "real router ID mismatch");
        CHECK(fabsf(selected_weights[rank] - cpu_weights[rank]) <= 1e-6f,
              "real router weight mismatch");
    }

    CHECK(k3_rocm_bf16_gemv_bf16(d_latent, static_tensor[2].device,
                                  d_input, K3_LATENT, K3_HIDDEN, NULL),
          "latent down projection");
    reference_bf16_gemv(NULL, cpu_latent,
                        (const hip_bfloat16 *)static_tensor[2].read.data,
                        input, K3_LATENT, K3_HIDDEN);

    const uint64_t expert_span_bytes = UINT64_C(17547264);
    uint8_t *d_selected_experts = NULL;
    HIP_CHECK(hipMalloc(&d_selected_experts,
                        (uint64_t)K3_TOP_K * expert_span_bytes));
    selected_layout layouts[K3_TOP_K];
    uint64_t expert_read_bytes = 0;
    double expert_read_seconds = 0.0;

    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        char names[K3_EXPERT_TENSOR_COUNT][160];
        static const char *suffix[K3_EXPERT_TENSOR_COUNT] = {
            "w1.weight_packed", "w1.weight_scale",
            "w2.weight_packed", "w2.weight_scale",
            "w3.weight_packed", "w3.weight_scale",
        };
        const k3_st_tensor *expert_tensor[K3_EXPERT_TENSOR_COUNT];
        for (uint32_t i = 0; i < K3_EXPERT_TENSOR_COUNT; i++) {
            snprintf(names[i], sizeof(names[i]),
                     "language_model.model.layers.1.block_sparse_moe."
                     "experts.%u.%s", selected_ids[rank], suffix[i]);
            expert_tensor[i] = k3_st_find(&model, names[i]);
            CHECK(expert_tensor[i] != NULL, "selected expert tensor lookup");
        }
        uint64_t start = expert_tensor[0]->physical_offset;
        uint64_t end = expert_tensor[5]->physical_offset +
                       expert_tensor[5]->byte_length;
        CHECK(end - start == expert_span_bytes,
              "selected expert physical span");
        layouts[rank].physical_start = start;
        layouts[rank].aligned_start = start & ~UINT64_C(4095);
        uint64_t aligned_end =
            (end + UINT64_C(4095)) & ~UINT64_C(4095);
        CHECK(aligned_end - layouts[rank].aligned_start <= UINT32_MAX,
              "selected expert aligned span");
        layouts[rank].aligned_bytes =
            (uint32_t)(aligned_end - layouts[rank].aligned_start);
        layouts[rank].shard = expert_tensor[0]->shard;
        k3_st_read expert_read;
        struct timespec read_start, read_end;
        clock_gettime(CLOCK_MONOTONIC, &read_start);
        CHECK(k3_st_read_span(&model, expert_tensor[0]->shard,
                              start, end - start, 4096u,
                              &expert_read, error, sizeof(error)),
              error);
        clock_gettime(CLOCK_MONOTONIC, &read_end);
        expert_read_seconds +=
            (double)(read_end.tv_sec - read_start.tv_sec) +
            (double)(read_end.tv_nsec - read_start.tv_nsec) / 1e9;
        expert_read_bytes += expert_read.allocation_bytes;
        uint8_t *device_base =
            d_selected_experts + (uint64_t)rank * expert_span_bytes;
        HIP_CHECK(hipMemcpy(device_base, expert_read.data,
                            expert_span_bytes, hipMemcpyHostToDevice));
        const uint8_t *host[K3_EXPERT_TENSOR_COUNT];
        for (uint32_t i = 0; i < K3_EXPERT_TENSOR_COUNT; i++) {
            layouts[rank].relative[i] =
                expert_tensor[i]->physical_offset - start;
            host[i] = expert_read.data + layouts[rank].relative[i];
        }

        reference_mxfp4_gemv(cpu_expert_gate, host[0], host[1],
                              cpu_latent, K3_EXPERT_HIDDEN, K3_LATENT);
        reference_mxfp4_gemv(cpu_expert_up, host[4], host[5],
                              cpu_latent, K3_EXPERT_HIDDEN, K3_LATENT);
        reference_situ(cpu_expert_activation, cpu_expert_gate,
                       cpu_expert_up, K3_EXPERT_HIDDEN);
        reference_mxfp4_gemv(cpu_expert_output, host[2], host[3],
                              cpu_expert_activation,
                              K3_LATENT, K3_EXPERT_HIDDEN);
        for (uint32_t d = 0; d < K3_LATENT; d++) {
            cpu_accumulator[d] +=
                cpu_weights[rank] * bf16_to_float(cpu_expert_output[d]);
        }
        k3_st_read_release(&expert_read);
    }
    for (uint32_t d = 0; d < K3_LATENT; d++) {
        cpu_routed[d] = float_to_bf16(cpu_accumulator[d]);
    }

    auto launch_full_moe = [&](bool use_q8) -> bool {
        if (!k3_rocm_bf16_gemv_f32(
                d_logits, static_tensor[1].device, d_input,
                K3_EXPERT_COUNT, K3_HIDDEN, NULL) ||
            !k3_rocm_router_topk_f32(
                d_ids, d_route_weights, d_logits, static_tensor[0].device,
                K3_EXPERT_COUNT, K3_TOP_K, 1.0f, NULL) ||
            !launch_static_gemv(
                use_q8, &q8_tensors[2],
                d_latent, &static_tensor[2], d_input,
                K3_LATENT, K3_HIDDEN)) {
            return false;
        }
        for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
            uint8_t *base =
                d_selected_experts + (uint64_t)rank * expert_span_bytes;
            if (!k3_rocm_mxfp4_gemv_bf16(
                    d_expert_gate, base + layouts[rank].relative[0],
                    base + layouts[rank].relative[1], d_latent,
                    K3_EXPERT_HIDDEN, K3_LATENT, NULL) ||
                !k3_rocm_mxfp4_gemv_bf16(
                    d_expert_up, base + layouts[rank].relative[4],
                    base + layouts[rank].relative[5], d_latent,
                    K3_EXPERT_HIDDEN, K3_LATENT, NULL) ||
                !k3_rocm_situ_bf16(
                    d_expert_activation, d_expert_gate, d_expert_up,
                    K3_EXPERT_HIDDEN, 4.0f, 25.0f, NULL) ||
                !k3_rocm_mxfp4_gemv_bf16(
                    d_expert_outputs + (uint64_t)rank * K3_LATENT,
                    base + layouts[rank].relative[2],
                    base + layouts[rank].relative[3],
                    d_expert_activation, K3_LATENT,
                    K3_EXPERT_HIDDEN, NULL)) {
                return false;
            }
        }
        return
            k3_rocm_weighted_sum_bf16(
                d_routed, d_expert_outputs, d_route_weights,
                K3_TOP_K, K3_LATENT, NULL) &&
            k3_rocm_rms_norm_bf16(
                d_routed_norm, d_routed, static_tensor[3].device,
                1u, K3_LATENT, 1e-5f, NULL) &&
            launch_static_gemv(
                use_q8, &q8_tensors[4],
                d_routed_full, &static_tensor[4], d_routed_norm,
                K3_HIDDEN, K3_LATENT) &&
            launch_static_gemv(
                use_q8, &q8_tensors[6],
                d_shared_gate, &static_tensor[6], d_input,
                K3_SHARED_HIDDEN, K3_HIDDEN) &&
            launch_static_gemv(
                use_q8, &q8_tensors[7],
                d_shared_up, &static_tensor[7], d_input,
                K3_SHARED_HIDDEN, K3_HIDDEN) &&
            k3_rocm_situ_bf16(
                d_shared_activation, d_shared_gate, d_shared_up,
                K3_SHARED_HIDDEN, 4.0f, 25.0f, NULL) &&
            launch_static_gemv(
                use_q8, &q8_tensors[5],
                d_shared_output, &static_tensor[5],
                d_shared_activation, K3_HIDDEN, K3_SHARED_HIDDEN) &&
            k3_rocm_add_bf16(
                d_final, d_routed_full, d_shared_output,
                K3_HIDDEN, NULL);
    };

    CHECK(launch_full_moe(false), "full real MoE warmup launch");
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start_event, end_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&end_event));
    const uint32_t iterations = 10;
    HIP_CHECK(hipEventRecord(start_event, NULL));
    for (uint32_t iteration = 0; iteration < iterations; iteration++) {
        CHECK(launch_full_moe(false), "full real MoE timed launch");
    }
    HIP_CHECK(hipEventRecord(end_event, NULL));
    HIP_CHECK(hipEventSynchronize(end_event));
    float gpu_elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&gpu_elapsed_ms,
                                  start_event, end_event));
    HIP_CHECK(hipMemcpy(gpu_final, d_final, hidden_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(launch_full_moe(true), "Q8 full real MoE warmup launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(start_event, NULL));
    for (uint32_t iteration = 0; iteration < iterations; iteration++) {
        CHECK(launch_full_moe(true), "Q8 full real MoE timed launch");
    }
    HIP_CHECK(hipEventRecord(end_event, NULL));
    HIP_CHECK(hipEventSynchronize(end_event));
    float q8_gpu_elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&q8_gpu_elapsed_ms,
                                  start_event, end_event));
    HIP_CHECK(hipMemcpy(q8_final, d_final, hidden_bytes,
                        hipMemcpyDeviceToHost));
    const uint32_t router_schedule_iterations = 100u;
    uint32_t router_probe_ids[K3_TOP_K];
    struct timespec router_start, router_end;
    clock_gettime(CLOCK_MONOTONIC, &router_start);
    for (uint32_t iteration = 0;
         iteration < router_schedule_iterations;
         iteration++) {
        CHECK(k3_rocm_bf16_gemv_f32(
                  d_logits, static_tensor[1].device, d_input,
                  K3_EXPERT_COUNT, K3_HIDDEN, NULL) &&
              k3_rocm_router_topk_f32(
                  d_ids, d_route_weights, d_logits,
                  static_tensor[0].device,
                  K3_EXPERT_COUNT, K3_TOP_K, 1.0f, NULL),
              "scheduled router launch");
        HIP_CHECK(hipMemcpy(router_probe_ids, d_ids,
                            sizeof(router_probe_ids),
                            hipMemcpyDeviceToHost));
    }
    clock_gettime(CLOCK_MONOTONIC, &router_end);
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        CHECK(router_probe_ids[rank] == selected_ids[rank],
              "scheduled router changed expert IDs");
    }
    const double router_schedule_ms =
        ((double)(router_end.tv_sec - router_start.tv_sec) +
         (double)(router_end.tv_nsec - router_start.tv_nsec) / 1e9) *
        1000.0 / router_schedule_iterations;
    uint32_t *mapped_ids_host = NULL;
    uint32_t *mapped_ids_device = NULL;
    float *mapped_weights_host = NULL;
    float *mapped_weights_device = NULL;
    HIP_CHECK(hipHostMalloc(
        (void **)&mapped_ids_host, K3_TOP_K * sizeof(uint32_t),
        hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(
        (void **)&mapped_weights_host, K3_TOP_K * sizeof(float),
        hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer(
        (void **)&mapped_ids_device, mapped_ids_host, 0));
    HIP_CHECK(hipHostGetDevicePointer(
        (void **)&mapped_weights_device, mapped_weights_host, 0));
    clock_gettime(CLOCK_MONOTONIC, &router_start);
    for (uint32_t iteration = 0;
         iteration < router_schedule_iterations;
         iteration++) {
        CHECK(k3_rocm_bf16_gemv_f32(
                  d_logits, static_tensor[1].device, d_input,
                  K3_EXPERT_COUNT, K3_HIDDEN, NULL) &&
              k3_rocm_router_topk_f32(
                  mapped_ids_device, mapped_weights_device, d_logits,
                  static_tensor[0].device,
                  K3_EXPERT_COUNT, K3_TOP_K, 1.0f, NULL),
              "mapped scheduled router launch");
        HIP_CHECK(hipStreamSynchronize(NULL));
    }
    clock_gettime(CLOCK_MONOTONIC, &router_end);
    const double mapped_router_schedule_ms =
        ((double)(router_end.tv_sec - router_start.tv_sec) +
         (double)(router_end.tv_nsec - router_start.tv_nsec) / 1e9) *
        1000.0 / router_schedule_iterations;
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        CHECK(mapped_ids_host[rank] == selected_ids[rank],
              "mapped scheduled router changed expert IDs");
        CHECK(fabsf(mapped_weights_host[rank] - selected_weights[rank]) <=
                  1e-6f,
              "mapped scheduled router changed expert weights");
    }

    /*
     * Production-shaped miss path: keep two fixed O_DIRECT reads in flight,
     * launch each expert directly from its GPU-visible completion buffer, and
     * refill the storage queue before doing that expert's arithmetic.
     */
    void *stream_host[K3_TOP_K] = { 0 };
    void *stream_device[K3_TOP_K] = { 0 };
    struct iovec stream_iov[K3_TOP_K];
    uint32_t stream_buffer_bytes = 0;
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        if (layouts[rank].aligned_bytes > stream_buffer_bytes) {
            stream_buffer_bytes = layouts[rank].aligned_bytes;
        }
    }
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        HIP_CHECK(hipHostMalloc(&stream_host[rank], stream_buffer_bytes,
                                hipHostMallocMapped));
        CHECK((uintptr_t)stream_host[rank] % 4096u == 0,
              "stream buffer is not 4 KiB aligned");
        HIP_CHECK(hipHostGetDevicePointer(&stream_device[rank],
                                          stream_host[rank], 0));
        stream_iov[rank].iov_base = stream_host[rank];
        stream_iov[rank].iov_len = stream_buffer_bytes;
    }
    k3_io_uring *stream_ring = NULL;
    CHECK(k3_io_uring_create(&stream_ring, stream_iov, K3_TOP_K,
                             error, sizeof(error)),
          error);
    hipStream_t expert_stream = NULL;
    hipStream_t shared_stream = NULL;
    HIP_CHECK(hipStreamCreateWithFlags(&expert_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&shared_stream, hipStreamNonBlocking));

    auto make_stream_request = [&](uint32_t rank) -> k3_io_request {
        k3_io_request request;
        request.fd = model.shards[layouts[rank].shard].direct_fd;
        request.offset = layouts[rank].aligned_start;
        request.bytes = layouts[rank].aligned_bytes;
        request.buffer_index = (uint16_t)rank;
        request.user_data = rank;
        return request;
    };
    k3_io_request initial_requests[K3_STREAM_QD];
    for (uint32_t rank = 0; rank < K3_STREAM_QD; rank++) {
        initial_requests[rank] = make_stream_request(rank);
    }
    struct timespec stream_start, stream_selected_end, stream_full_end;
    clock_gettime(CLOCK_MONOTONIC, &stream_start);
    CHECK(k3_io_uring_submit(stream_ring, initial_requests, K3_STREAM_QD,
                             error, sizeof(error)),
          error);
    /*
     * The shared expert depends only on the post-attention layer input. Start
     * its Q8 path after the QD2 reads so storage and GPU static work overlap.
     */
    CHECK(k3_rocm_q8_128_gemv_bf16(
              d_shared_gate, q8_tensors[6].quantized,
              q8_tensors[6].scales, d_input,
              K3_SHARED_HIDDEN, K3_HIDDEN, shared_stream) &&
          k3_rocm_q8_128_gemv_bf16(
              d_shared_up, q8_tensors[7].quantized,
              q8_tensors[7].scales, d_input,
              K3_SHARED_HIDDEN, K3_HIDDEN, shared_stream) &&
          k3_rocm_situ_bf16(
              d_shared_activation, d_shared_gate, d_shared_up,
              K3_SHARED_HIDDEN, 4.0f, 25.0f, shared_stream) &&
          k3_rocm_q8_128_gemv_bf16(
              d_shared_output, q8_tensors[5].quantized,
              q8_tensors[5].scales, d_shared_activation,
              K3_HIDDEN, K3_SHARED_HIDDEN, shared_stream),
          "overlapped Q8 shared-expert launch");
    CHECK(k3_rocm_q8_128_gemv_bf16(
              d_latent, q8_tensors[2].quantized,
              q8_tensors[2].scales, d_input,
              K3_LATENT, K3_HIDDEN, expert_stream),
          "overlapped Q8 latent projection");
    uint32_t next_rank = K3_STREAM_QD;
    uint32_t streamed = 0u;
    while (streamed < K3_TOP_K) {
        k3_io_completion completions[K3_STREAM_QD];
        uint16_t completion_count = 0;
        CHECK(k3_io_uring_wait(stream_ring, completions, K3_STREAM_QD,
                               &completion_count, error, sizeof(error)),
              error);
        k3_io_request refill[K3_STREAM_QD];
        uint16_t refill_count = 0;
        while (next_rank < K3_TOP_K &&
               refill_count < completion_count) {
            refill[refill_count++] = make_stream_request(next_rank++);
        }
        if (refill_count) {
            CHECK(k3_io_uring_submit(stream_ring, refill, refill_count,
                                     error, sizeof(error)),
                  error);
        }

        for (uint16_t i = 0; i < completion_count; i++) {
            uint32_t rank = (uint32_t)completions[i].user_data;
            CHECK(rank < K3_TOP_K &&
                  completions[i].result ==
                      (int32_t)layouts[rank].aligned_bytes,
                  "streamed selected-expert read failed");
            uint8_t *base = (uint8_t *)stream_device[rank] +
                (layouts[rank].physical_start -
                 layouts[rank].aligned_start);
            CHECK(k3_rocm_mxfp4_gemv_bf16(
                      d_expert_gate, base + layouts[rank].relative[0],
                      base + layouts[rank].relative[1], d_latent,
                      K3_EXPERT_HIDDEN, K3_LATENT, expert_stream) &&
                  k3_rocm_mxfp4_gemv_bf16(
                      d_expert_up, base + layouts[rank].relative[4],
                      base + layouts[rank].relative[5], d_latent,
                      K3_EXPERT_HIDDEN, K3_LATENT, expert_stream) &&
                  k3_rocm_situ_bf16(
                      d_expert_activation, d_expert_gate, d_expert_up,
                      K3_EXPERT_HIDDEN, 4.0f, 25.0f, expert_stream) &&
                  k3_rocm_mxfp4_gemv_bf16(
                      d_expert_outputs + (uint64_t)rank * K3_LATENT,
                      base + layouts[rank].relative[2],
                      base + layouts[rank].relative[3],
                      d_expert_activation, K3_LATENT,
                      K3_EXPERT_HIDDEN, expert_stream),
                  "completion-ordered expert launch");
            /*
             * A single ordered nonblocking stream safely reuses the scratch
             * tensors. Expert compute is much faster than storage arrivals,
             * so another GPU lane would add complexity without useful work.
             */
            streamed++;
        }
    }
    CHECK(k3_io_uring_outstanding(stream_ring) == 0,
          "stream ring retained outstanding reads");
    HIP_CHECK(hipStreamSynchronize(expert_stream));
    CHECK(k3_rocm_weighted_sum_bf16(
              d_routed, d_expert_outputs, mapped_weights_device,
              K3_TOP_K, K3_LATENT, NULL),
          "completion-ordered weighted sum");
    HIP_CHECK(hipStreamSynchronize(NULL));
    clock_gettime(CLOCK_MONOTONIC, &stream_selected_end);
    CHECK(k3_rocm_rms_norm_bf16(
              d_routed_norm, d_routed, static_tensor[3].device,
              1u, K3_LATENT, 1e-5f, NULL) &&
          k3_rocm_q8_128_gemv_bf16(
              d_routed_full, q8_tensors[4].quantized,
              q8_tensors[4].scales, d_routed_norm,
              K3_HIDDEN, K3_LATENT, NULL),
          "streamed Q8 routed tail");
    HIP_CHECK(hipStreamSynchronize(shared_stream));
    CHECK(k3_rocm_add_bf16(
              d_final, d_routed_full, d_shared_output,
              K3_HIDDEN, NULL),
          "streamed Q8 final add");
    HIP_CHECK(hipDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &stream_full_end);
    hip_bfloat16 *streamed_routed =
        (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *streamed_final =
        (hip_bfloat16 *)malloc(hidden_bytes);
    CHECK(streamed_routed && streamed_final,
          "streamed output allocation");
    HIP_CHECK(hipMemcpy(streamed_routed, d_routed, latent_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(streamed_final, d_final, hidden_bytes,
                        hipMemcpyDeviceToHost));

    /*
     * Admit the just-read experts into a device-resident one-layer cache and
     * exercise the real hit path. Production keeps only the small staging ring
     * fixed-registered: registering all 32/layer cache slots exceeds this
     * host's memlock limit. The cache planner keeps logical slots independent
     * of the route buffer that supplied each admission.
     */
    k3_expert_cache *expert_cache = NULL;
    CHECK(k3_expert_cache_create(&expert_cache, 1u, K3_TOP_K,
                                 error, sizeof(error)),
          error);
    uint16_t cache_ids[K3_TOP_K];
    k3_expert_cache_access cache_misses[K3_TOP_K];
    void *cache_device[K3_TOP_K] = { 0 };
    uint8_t *d_cache_storage = NULL;
    HIP_CHECK(hipMalloc(
        &d_cache_storage,
        (uint64_t)K3_TOP_K * expert_span_bytes));
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        CHECK(selected_ids[rank] <= UINT16_MAX,
              "selected expert does not fit cache ID");
        cache_ids[rank] = (uint16_t)selected_ids[rank];
    }
    CHECK(k3_expert_cache_plan(expert_cache, 0u, cache_ids, K3_TOP_K,
                               cache_misses, error, sizeof(error)),
          error);
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        CHECK(!cache_misses[rank].hit && cache_misses[rank].admit &&
                  cache_misses[rank].destination_slot < K3_TOP_K,
              "first cache route must admit every expert");
        uint8_t *source =
            (uint8_t *)stream_device[rank] +
            (layouts[rank].physical_start -
             layouts[rank].aligned_start);
        uint8_t *destination =
            d_cache_storage +
            (uint64_t)cache_misses[rank].destination_slot *
                expert_span_bytes;
        HIP_CHECK(hipMemcpyAsync(
            destination, source, expert_span_bytes,
            hipMemcpyDeviceToDevice, expert_stream));
        cache_device[cache_misses[rank].destination_slot] =
            destination;
    }
    HIP_CHECK(hipStreamSynchronize(expert_stream));
    CHECK(k3_expert_cache_commit(expert_cache, 0u,
                                 error, sizeof(error)),
          error);

    k3_expert_cache_access cache_hits[K3_TOP_K];
    CHECK(k3_expert_cache_plan(expert_cache, 0u, cache_ids, K3_TOP_K,
                               cache_hits, error, sizeof(error)),
          error);
    const uint32_t cache_iterations = 10u;
    struct timespec cache_start, cache_end;
    clock_gettime(CLOCK_MONOTONIC, &cache_start);
    for (uint32_t iteration = 0; iteration < cache_iterations; iteration++) {
        for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
            CHECK(cache_hits[rank].hit && !cache_hits[rank].admit &&
                      cache_hits[rank].source_slot < K3_TOP_K,
                  "second cache route must hit every expert");
            uint8_t *base =
                (uint8_t *)cache_device[cache_hits[rank].source_slot];
            CHECK(base != NULL, "cache hit has no mapped expert buffer");
            CHECK(k3_rocm_mxfp4_gemv_bf16(
                      d_expert_gate, base + layouts[rank].relative[0],
                      base + layouts[rank].relative[1], d_latent,
                      K3_EXPERT_HIDDEN, K3_LATENT, expert_stream) &&
                  k3_rocm_mxfp4_gemv_bf16(
                      d_expert_up, base + layouts[rank].relative[4],
                      base + layouts[rank].relative[5], d_latent,
                      K3_EXPERT_HIDDEN, K3_LATENT, expert_stream) &&
                  k3_rocm_situ_bf16(
                      d_expert_activation, d_expert_gate, d_expert_up,
                      K3_EXPERT_HIDDEN, 4.0f, 25.0f, expert_stream) &&
                  k3_rocm_mxfp4_gemv_bf16(
                      d_expert_outputs + (uint64_t)rank * K3_LATENT,
                      base + layouts[rank].relative[2],
                      base + layouts[rank].relative[3],
                      d_expert_activation, K3_LATENT,
                      K3_EXPERT_HIDDEN, expert_stream),
                  "cache-hit expert launch");
        }
        HIP_CHECK(hipStreamSynchronize(expert_stream));
        CHECK(k3_rocm_weighted_sum_bf16(
                  d_routed, d_expert_outputs, d_route_weights,
                  K3_TOP_K, K3_LATENT, NULL),
              "cache-hit weighted sum");
        HIP_CHECK(hipDeviceSynchronize());
    }
    clock_gettime(CLOCK_MONOTONIC, &cache_end);
    hip_bfloat16 *cached_routed =
        (hip_bfloat16 *)malloc(latent_bytes);
    CHECK(cached_routed != NULL, "cached output allocation");
    HIP_CHECK(hipMemcpy(cached_routed, d_routed, latent_bytes,
                        hipMemcpyDeviceToHost));
    CHECK(k3_expert_cache_commit(expert_cache, 0u,
                                 error, sizeof(error)),
          error);
    k3_expert_cache_stats cache_stats;
    k3_expert_cache_get_stats(expert_cache, &cache_stats);
    CHECK(cache_stats.accesses == 2u * K3_TOP_K &&
              cache_stats.hits == K3_TOP_K &&
              cache_stats.misses == K3_TOP_K &&
              cache_stats.admissions == K3_TOP_K,
          "cache telemetry mismatch");

    /*
     * The accepted 32-expert/layer traces predict about 45% hits. Exercise a
     * representative seven-hit/nine-miss layer with hits launched immediately
     * and QD2 continuously refilled for misses.
     */
    const uint32_t mixed_hit_count = 7u;
    const uint32_t mixed_miss_count = K3_TOP_K - mixed_hit_count;
    const uint32_t mixed_iterations = 3u;
    struct timespec mixed_start, mixed_end;
    clock_gettime(CLOCK_MONOTONIC, &mixed_start);
    for (uint32_t iteration = 0; iteration < mixed_iterations; iteration++) {
        CHECK(k3_rocm_bf16_gemv_f32(
                  d_logits, static_tensor[1].device, d_input,
                  K3_EXPERT_COUNT, K3_HIDDEN, NULL) &&
              k3_rocm_router_topk_f32(
                  mapped_ids_device, mapped_weights_device, d_logits,
                  static_tensor[0].device,
                  K3_EXPERT_COUNT, K3_TOP_K, 1.0f, NULL),
              "mixed scheduled router launch");
        HIP_CHECK(hipStreamSynchronize(NULL));

        k3_io_request mixed_initial[K3_STREAM_QD];
        for (uint32_t i = 0; i < K3_STREAM_QD; i++) {
            mixed_initial[i] =
                make_stream_request(mixed_hit_count + i);
        }
        CHECK(k3_io_uring_submit(
                  stream_ring, mixed_initial, K3_STREAM_QD,
                  error, sizeof(error)),
              error);
        CHECK(k3_rocm_q8_128_gemv_bf16(
                  d_latent, q8_tensors[2].quantized,
                  q8_tensors[2].scales, d_input,
                  K3_LATENT, K3_HIDDEN, expert_stream) &&
              k3_rocm_q8_128_gemv_bf16(
                  d_shared_gate, q8_tensors[6].quantized,
                  q8_tensors[6].scales, d_input,
                  K3_SHARED_HIDDEN, K3_HIDDEN, shared_stream) &&
              k3_rocm_q8_128_gemv_bf16(
                  d_shared_up, q8_tensors[7].quantized,
                  q8_tensors[7].scales, d_input,
                  K3_SHARED_HIDDEN, K3_HIDDEN, shared_stream) &&
              k3_rocm_situ_bf16(
                  d_shared_activation, d_shared_gate, d_shared_up,
                  K3_SHARED_HIDDEN, 4.0f, 25.0f, shared_stream) &&
              k3_rocm_q8_128_gemv_bf16(
                  d_shared_output, q8_tensors[5].quantized,
                  q8_tensors[5].scales, d_shared_activation,
                  K3_HIDDEN, K3_SHARED_HIDDEN, shared_stream),
              "mixed overlapped static launch");

        for (uint32_t rank = 0; rank < mixed_hit_count; rank++) {
            uint8_t *base =
                (uint8_t *)cache_device[cache_hits[rank].source_slot];
            CHECK(k3_rocm_mxfp4_gemv_bf16(
                      d_expert_gate, base + layouts[rank].relative[0],
                      base + layouts[rank].relative[1], d_latent,
                      K3_EXPERT_HIDDEN, K3_LATENT, expert_stream) &&
                  k3_rocm_mxfp4_gemv_bf16(
                      d_expert_up, base + layouts[rank].relative[4],
                      base + layouts[rank].relative[5], d_latent,
                      K3_EXPERT_HIDDEN, K3_LATENT, expert_stream) &&
                  k3_rocm_situ_bf16(
                      d_expert_activation, d_expert_gate, d_expert_up,
                      K3_EXPERT_HIDDEN, 4.0f, 25.0f, expert_stream) &&
                  k3_rocm_mxfp4_gemv_bf16(
                      d_expert_outputs + (uint64_t)rank * K3_LATENT,
                      base + layouts[rank].relative[2],
                      base + layouts[rank].relative[3],
                      d_expert_activation, K3_LATENT,
                      K3_EXPERT_HIDDEN, expert_stream),
                  "mixed cache-hit expert launch");
        }

        uint32_t mixed_next_rank = mixed_hit_count + K3_STREAM_QD;
        uint32_t mixed_streamed = 0;
        while (mixed_streamed < mixed_miss_count) {
            k3_io_completion completions[K3_STREAM_QD];
            uint16_t completion_count = 0;
            CHECK(k3_io_uring_wait(
                      stream_ring, completions, K3_STREAM_QD,
                      &completion_count, error, sizeof(error)),
                  error);
            k3_io_request refill[K3_STREAM_QD];
            uint16_t refill_count = 0;
            while (mixed_next_rank < K3_TOP_K &&
                   refill_count < completion_count) {
                refill[refill_count++] =
                    make_stream_request(mixed_next_rank++);
            }
            if (refill_count) {
                CHECK(k3_io_uring_submit(
                          stream_ring, refill, refill_count,
                          error, sizeof(error)),
                      error);
            }
            for (uint16_t i = 0; i < completion_count; i++) {
                const uint32_t rank =
                    (uint32_t)completions[i].user_data;
                CHECK(rank >= mixed_hit_count && rank < K3_TOP_K &&
                          completions[i].result ==
                              (int32_t)layouts[rank].aligned_bytes,
                      "mixed streamed expert read failed");
                uint8_t *base =
                    (uint8_t *)stream_device[rank] +
                    (layouts[rank].physical_start -
                     layouts[rank].aligned_start);
                CHECK(k3_rocm_mxfp4_gemv_bf16(
                          d_expert_gate,
                          base + layouts[rank].relative[0],
                          base + layouts[rank].relative[1], d_latent,
                          K3_EXPERT_HIDDEN, K3_LATENT, expert_stream) &&
                      k3_rocm_mxfp4_gemv_bf16(
                          d_expert_up,
                          base + layouts[rank].relative[4],
                          base + layouts[rank].relative[5], d_latent,
                          K3_EXPERT_HIDDEN, K3_LATENT, expert_stream) &&
                      k3_rocm_situ_bf16(
                          d_expert_activation, d_expert_gate, d_expert_up,
                          K3_EXPERT_HIDDEN, 4.0f, 25.0f,
                          expert_stream) &&
                      k3_rocm_mxfp4_gemv_bf16(
                          d_expert_outputs +
                              (uint64_t)rank * K3_LATENT,
                          base + layouts[rank].relative[2],
                          base + layouts[rank].relative[3],
                          d_expert_activation, K3_LATENT,
                          K3_EXPERT_HIDDEN, expert_stream),
                      "mixed completion expert launch");
                /*
                 * Admission follows this expert's reads on the same ordered
                 * stream, so its staging slot cannot be reused too early.
                 * The copy normally overlaps the next 2.6 ms SSD arrival.
                 */
                HIP_CHECK(hipMemcpyAsync(
                    d_cache_storage +
                        (uint64_t)rank * expert_span_bytes,
                    base, expert_span_bytes,
                    hipMemcpyDeviceToDevice, expert_stream));
                mixed_streamed++;
            }
        }
        CHECK(k3_io_uring_outstanding(stream_ring) == 0,
              "mixed scheduler retained outstanding reads");
        HIP_CHECK(hipStreamSynchronize(expert_stream));
        CHECK(k3_rocm_weighted_sum_bf16(
                  d_routed, d_expert_outputs, mapped_weights_device,
                  K3_TOP_K, K3_LATENT, NULL) &&
              k3_rocm_rms_norm_bf16(
                  d_routed_norm, d_routed, static_tensor[3].device,
                  1u, K3_LATENT, 1e-5f, NULL) &&
              k3_rocm_q8_128_gemv_bf16(
                  d_routed_full, q8_tensors[4].quantized,
                  q8_tensors[4].scales, d_routed_norm,
                  K3_HIDDEN, K3_LATENT, NULL),
              "mixed Q8 routed tail");
        HIP_CHECK(hipStreamSynchronize(shared_stream));
        CHECK(k3_rocm_add_bf16(
                  d_final, d_routed_full, d_shared_output,
                  K3_HIDDEN, NULL),
              "mixed Q8 final add");
        HIP_CHECK(hipDeviceSynchronize());
    }
    clock_gettime(CLOCK_MONOTONIC, &mixed_end);
    const double mixed_layer_ms =
        ((double)(mixed_end.tv_sec - mixed_start.tv_sec) +
         (double)(mixed_end.tv_nsec - mixed_start.tv_nsec) / 1e9) *
        1000.0 / mixed_iterations;
    hip_bfloat16 *mixed_final =
        (hip_bfloat16 *)malloc(hidden_bytes);
    CHECK(mixed_final != NULL, "mixed output allocation");
    HIP_CHECK(hipMemcpy(mixed_final, d_final, hidden_bytes,
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipStreamDestroy(expert_stream));
    HIP_CHECK(hipStreamDestroy(shared_stream));
    k3_io_uring_destroy(stream_ring);
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        HIP_CHECK(hipHostFree(stream_host[rank]));
    }

    struct timespec cpu_tail_start, cpu_tail_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_tail_start);
    reference_rms_norm(
        cpu_routed_norm, cpu_routed,
        (const hip_bfloat16 *)static_tensor[3].read.data,
        K3_LATENT, 1e-5f);
    reference_bf16_gemv(
        NULL, cpu_routed_full,
        (const hip_bfloat16 *)static_tensor[4].read.data,
        cpu_routed_norm, K3_HIDDEN, K3_LATENT);
    reference_bf16_gemv(
        NULL, cpu_shared_gate,
        (const hip_bfloat16 *)static_tensor[6].read.data,
        input, K3_SHARED_HIDDEN, K3_HIDDEN);
    reference_bf16_gemv(
        NULL, cpu_shared_up,
        (const hip_bfloat16 *)static_tensor[7].read.data,
        input, K3_SHARED_HIDDEN, K3_HIDDEN);
    reference_situ(cpu_shared_activation, cpu_shared_gate,
                   cpu_shared_up, K3_SHARED_HIDDEN);
    reference_bf16_gemv(
        NULL, cpu_shared_output,
        (const hip_bfloat16 *)static_tensor[5].read.data,
        cpu_shared_activation, K3_HIDDEN, K3_SHARED_HIDDEN);
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        cpu_final[i] = float_to_bf16(
            bf16_to_float(cpu_routed_full[i]) +
            bf16_to_float(cpu_shared_output[i]));
    }
    clock_gettime(CLOCK_MONOTONIC, &cpu_tail_end);

    printf("K3 full real MoE smoke: layer=1 top_k=16\n");
    printf("  selected experts:");
    for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
        printf(" %u", selected_ids[rank]);
    }
    printf("\n");
    printf("  route weight sum: %.9f\n", ({
        float sum = 0.0f;
        for (uint32_t rank = 0; rank < K3_TOP_K; rank++) {
            sum += selected_weights[rank];
        }
        sum;
    }));
    printf("  static reads: %.2f MiB in %.3f ms\n",
           (double)static_read_bytes / (1024.0 * 1024.0),
           static_read_seconds * 1000.0);
    printf("  Q8 static-tail tier: %.2f MiB from %.2f MiB (%.2f%%); "
           "router stays BF16\n",
           q8_storage_bytes / 1048576.0,
           q8_source_bytes / 1048576.0,
           100.0 * (double)q8_storage_bytes /
               (double)q8_source_bytes);
    printf("  selected-expert reads: %.2f MiB in %.3f ms "
           "(%.3f ms/expert)\n",
           (double)expert_read_bytes / (1024.0 * 1024.0),
           expert_read_seconds * 1000.0,
           expert_read_seconds * 1000.0 / K3_TOP_K);
    printf("  resident ROCm MoE compute: %.3f ms/token "
           "(10-run mean)\n",
           gpu_elapsed_ms / iterations);
    printf("  Q8 resident ROCm MoE compute: %.3f ms/token "
           "(10-run mean)\n",
           q8_gpu_elapsed_ms / iterations);
    printf("  router GEMV/top-k/device-to-host IDs: %.3f ms/layer "
           "(%u-run mean)\n",
           router_schedule_ms, router_schedule_iterations);
    printf("  router GEMV/top-k/mapped coherent IDs: %.3f ms/layer "
           "(%u-run mean)\n",
           mapped_router_schedule_ms, router_schedule_iterations);
    printf("  fixed-buffer QD%u selected I/O + completion-ordered compute: "
           "%.3f ms/layer\n", K3_STREAM_QD,
           ((double)(stream_selected_end.tv_sec - stream_start.tv_sec) +
            (double)(stream_selected_end.tv_nsec - stream_start.tv_nsec) /
                1e9) *
               1000.0);
    printf("  fixed-buffer QD%u + overlapped shared + Q8 routed tail: "
           "%.3f ms/layer\n", K3_STREAM_QD,
           ((double)(stream_full_end.tv_sec - stream_start.tv_sec) +
            (double)(stream_full_end.tv_nsec - stream_start.tv_nsec) /
               1e9) *
               1000.0);
    printf("  scheduled Q8 full MoE including router: %.3f ms/layer\n",
           mapped_router_schedule_ms +
           ((double)(stream_full_end.tv_sec - stream_start.tv_sec) +
            (double)(stream_full_end.tv_nsec - stream_start.tv_nsec) /
                1e9) *
               1000.0);
    printf("  device-cache top-16 hit compute: %.3f ms/layer "
           "(10-run mean)\n",
           ((double)(cache_end.tv_sec - cache_start.tv_sec) +
            (double)(cache_end.tv_nsec - cache_start.tv_nsec) / 1e9) *
               1000.0 / cache_iterations);
    printf("  mixed scheduler, 7 hits + 9 QD2 misses + admissions, "
           "full Q8 MoE: "
           "%.3f ms/layer (%u-run mean)\n",
           mixed_layer_ms, mixed_iterations);
    printf("  CPU static-tail oracle: %.3f ms\n",
           ((double)(cpu_tail_end.tv_sec - cpu_tail_start.tv_sec) +
            (double)(cpu_tail_end.tv_nsec - cpu_tail_start.tv_nsec) / 1e9) *
               1000.0);
    CHECK(compare_output(gpu_final, cpu_final, K3_HIDDEN),
          "full MoE output correctness");
    CHECK(compare_output(q8_final, cpu_final, K3_HIDDEN),
          "Q8 full MoE output correctness");
    double q8_squared_error = 0.0;
    double q8_squared_reference = 0.0;
    double q8_squared_output = 0.0;
    double q8_dot = 0.0;
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        const double want = bf16_to_float(cpu_final[i]);
        const double actual = bf16_to_float(q8_final[i]);
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
          "Q8 full MoE NRMSE exceeds prototype gate");
    CHECK(isfinite(q8_cosine) && q8_cosine > 0.999,
          "Q8 full MoE cosine below prototype gate");
    printf("  Q8 final output: nrmse=%.7f cosine=%.9f "
           "hash=0x%016" PRIx64 "\n",
           q8_nrmse, q8_cosine, fnv1a64(q8_final, hidden_bytes));
    CHECK(compare_output(streamed_routed, cpu_routed, K3_LATENT),
          "completion-ordered routed output correctness");
    CHECK(compare_output(streamed_final, cpu_final, K3_HIDDEN),
          "streamed full Q8 MoE output correctness");
    CHECK(compare_output(mixed_final, cpu_final, K3_HIDDEN),
          "mixed full Q8 MoE output correctness");
    CHECK(compare_output(cached_routed, cpu_routed, K3_LATENT),
          "device-cache routed output correctness");
    printf("  output FNV-1a64: 0x%016" PRIx64 "\n",
           fnv1a64(gpu_final, hidden_bytes));
    printf("K3 full real MoE smoke: PASS\n");

    HIP_CHECK(hipEventDestroy(end_event));
    HIP_CHECK(hipEventDestroy(start_event));
    k3_expert_cache_destroy(expert_cache);
    HIP_CHECK(hipFree(d_cache_storage));
    HIP_CHECK(hipFree(d_selected_experts));
    HIP_CHECK(hipFree(d_final));
    HIP_CHECK(hipFree(d_shared_output));
    HIP_CHECK(hipFree(d_shared_activation));
    HIP_CHECK(hipFree(d_shared_up));
    HIP_CHECK(hipFree(d_shared_gate));
    HIP_CHECK(hipFree(d_routed_full));
    HIP_CHECK(hipFree(d_routed_norm));
    HIP_CHECK(hipFree(d_routed));
    HIP_CHECK(hipFree(d_expert_outputs));
    HIP_CHECK(hipFree(d_expert_activation));
    HIP_CHECK(hipFree(d_expert_up));
    HIP_CHECK(hipFree(d_expert_gate));
    HIP_CHECK(hipFree(d_latent));
    HIP_CHECK(hipFree(d_route_weights));
    HIP_CHECK(hipFree(d_ids));
    HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipHostFree(mapped_weights_host));
    HIP_CHECK(hipHostFree(mapped_ids_host));
    free(cpu_accumulator);
    free(mixed_final);
    free(streamed_final);
    free(cached_routed);
    free(streamed_routed);
    free(cpu_logits);
    free(q8_final);
    free(gpu_final);
    free(cpu_final);
    free(cpu_shared_output);
    free(cpu_shared_activation);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_routed_full);
    free(cpu_routed_norm);
    free(cpu_routed);
    free(cpu_expert_output);
    free(cpu_expert_activation);
    free(cpu_expert_up);
    free(cpu_expert_gate);
    free(cpu_latent);
    free(input);
    for (uint32_t i = 0; i < K3_STATIC_COUNT; i++) {
        release_q8_tensor(&q8_tensors[i]);
        if (static_tensor[i].device) HIP_CHECK(hipFree(static_tensor[i].device));
        k3_st_read_release(&static_tensor[i].read);
    }
    k3_st_model_close(&model);
    return 0;
}
