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
    K3_LAYERS = 93,
    K3_BLOCK_CAPACITY = 8,
    K3_ATTN_RES_CALLS = 186,
    K3_NORM_CALLS = 186,
    K3_ADD_CALLS = 178,
    K3_BLOCK_SAVES = 8,
    K3_BENCHMARK_STEPS = 10,
    K3_REAL_WEIGHT_PAIRS = 3,
};

typedef struct {
    const k3_st_tensor *tensor;
    k3_st_read read;
    void *device;
} loaded_tensor;

typedef struct {
    loaded_tensor norm;
    loaded_tensor projection;
    uint32_t blocks;
    const char *label;
} real_weight_pair;

static uint32_t rng_state = UINT32_C(0x4b335253);

static float random_input(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & UINT32_C(0x00ffffff)) /
            (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * 0.125f;
}

static bool expected_vector(const k3_st_tensor *tensor) {
    if (!tensor || tensor->dtype != K3_ST_DTYPE_BF16) return false;
    if (tensor->ndim == 1u) {
        return tensor->shape[0] == K3_HIDDEN;
    }
    return tensor->ndim == 2u &&
           tensor->shape[0] == 1u &&
           tensor->shape[1] == K3_HIDDEN;
}

static bool audit_tensor(const k3_st_model *model,
                         const char *name,
                         uint64_t *bytes,
                         char *error,
                         size_t error_size) {
    const k3_st_tensor *tensor = k3_st_find(model, name);
    if (!expected_vector(tensor)) {
        snprintf(error, error_size,
                 "missing or invalid BF16 vector %s", name);
        return false;
    }
    *bytes += tensor->byte_length;
    return true;
}

static bool load_tensor(k3_st_model *model,
                        const char *name,
                        loaded_tensor *loaded,
                        char *error,
                        size_t error_size) {
    memset(loaded, 0, sizeof(*loaded));
    loaded->tensor = k3_st_find(model, name);
    if (!expected_vector(loaded->tensor)) {
        snprintf(error, error_size,
                 "missing or invalid BF16 vector %s", name);
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

static void reference_attn_res(
        float *output,
        const hip_bfloat16 *prefix,
        const hip_bfloat16 *blocks,
        const hip_bfloat16 *norm,
        const hip_bfloat16 *projection,
        uint32_t num_blocks) {
    float logits[K3_BLOCK_CAPACITY + 1u];
    float probabilities[K3_BLOCK_CAPACITY + 1u];
    for (uint32_t source = 0; source <= num_blocks; source++) {
        const hip_bfloat16 *value = source == num_blocks ?
            prefix : blocks + (uint64_t)source * K3_HIDDEN;
        float sum_squares = 0.0f;
        float dot = 0.0f;
        for (uint32_t d = 0; d < K3_HIDDEN; d++) {
            const float element = (float)value[d];
            sum_squares += element * element;
            dot += element * (float)norm[d] * (float)projection[d];
        }
        logits[source] =
            dot / sqrtf(sum_squares / (float)K3_HIDDEN + 1e-5f);
    }
    float maximum = logits[0];
    for (uint32_t source = 1; source <= num_blocks; source++) {
        maximum = fmaxf(maximum, logits[source]);
    }
    float denominator = 0.0f;
    for (uint32_t source = 0; source <= num_blocks; source++) {
        probabilities[source] = expf(logits[source] - maximum);
        denominator += probabilities[source];
    }
    for (uint32_t d = 0; d < K3_HIDDEN; d++) {
        float value = 0.0f;
        for (uint32_t source = 0; source <= num_blocks; source++) {
            const hip_bfloat16 *vector = source == num_blocks ?
                prefix : blocks + (uint64_t)source * K3_HIDDEN;
            value += probabilities[source] / denominator *
                     (float)vector[d];
        }
        output[d] = value;
    }
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 residual spine: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 residual spine on %s (%s)\n",
           properties.name, properties.gcnArchName);

    char error[512];
    char name[256];
    k3_st_model model;
    CHECK(k3_st_model_open(&model, root, 96u, error, sizeof(error)),
          error);

    uint64_t audited_bytes = 0;
    uint32_t audited_tensors = 0;
    for (uint32_t layer = 0; layer < K3_LAYERS; layer++) {
        static const char *suffixes[] = {
            "self_attention_res_norm.weight",
            "self_attention_res_proj.weight",
            "mlp_res_norm.weight",
            "mlp_res_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        };
        for (uint32_t i = 0;
             i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
            snprintf(name, sizeof(name),
                     "language_model.model.layers.%u.%s",
                     layer, suffixes[i]);
            CHECK(audit_tensor(&model, name, &audited_bytes,
                               error, sizeof(error)),
                  error);
            audited_tensors++;
        }
    }
    static const char *output_names[] = {
        "language_model.model.output_attn_res_norm.weight",
        "language_model.model.output_attn_res_proj.weight",
    };
    for (uint32_t i = 0; i < 2u; i++) {
        CHECK(audit_tensor(&model, output_names[i], &audited_bytes,
                           error, sizeof(error)),
              error);
        audited_tensors++;
    }
    CHECK(audited_tensors == 560u,
          "residual-spine tensor inventory count");

    real_weight_pair pairs[K3_REAL_WEIGHT_PAIRS];
    memset(pairs, 0, sizeof(pairs));
    pairs[0].blocks = 1u;
    pairs[0].label = "layer 0 MLP";
    pairs[1].blocks = 8u;
    pairs[1].label = "layer 92 attention";
    pairs[2].blocks = 8u;
    pairs[2].label = "final output";
    static const char *pair_names[K3_REAL_WEIGHT_PAIRS][2] = {
        {
            "language_model.model.layers.0.mlp_res_norm.weight",
            "language_model.model.layers.0.mlp_res_proj.weight",
        },
        {
            "language_model.model.layers.92.self_attention_res_norm.weight",
            "language_model.model.layers.92.self_attention_res_proj.weight",
        },
        {
            "language_model.model.output_attn_res_norm.weight",
            "language_model.model.output_attn_res_proj.weight",
        },
    };
    for (uint32_t i = 0; i < K3_REAL_WEIGHT_PAIRS; i++) {
        CHECK(load_tensor(&model, pair_names[i][0], &pairs[i].norm,
                          error, sizeof(error)),
              error);
        CHECK(load_tensor(&model, pair_names[i][1],
                          &pairs[i].projection,
                          error, sizeof(error)),
              error);
    }
    loaded_tensor input_norm;
    loaded_tensor post_norm;
    CHECK(load_tensor(
              &model,
              "language_model.model.layers.0.input_layernorm.weight",
              &input_norm, error, sizeof(error)),
          error);
    CHECK(load_tensor(
              &model,
              "language_model.model.layers.0.post_attention_layernorm.weight",
              &post_norm, error, sizeof(error)),
          error);

    const size_t vector_bytes =
        (size_t)K3_HIDDEN * sizeof(hip_bfloat16);
    const size_t block_bytes =
        (size_t)K3_BLOCK_CAPACITY * vector_bytes;
    hip_bfloat16 *prefix =
        (hip_bfloat16 *)malloc(vector_bytes);
    hip_bfloat16 *blocks =
        (hip_bfloat16 *)malloc(block_bytes);
    hip_bfloat16 *got =
        (hip_bfloat16 *)malloc(vector_bytes);
    float *expected =
        (float *)malloc((size_t)K3_HIDDEN * sizeof(float));
    CHECK(prefix && blocks && got && expected,
          "residual-spine host allocation");
    for (uint32_t d = 0; d < K3_HIDDEN; d++) {
        prefix[d] = hip_bfloat16(random_input());
    }
    for (uint64_t i = 0;
         i < (uint64_t)K3_BLOCK_CAPACITY * K3_HIDDEN; i++) {
        blocks[i] = hip_bfloat16(random_input());
    }

    void *d_prefix = NULL;
    void *d_blocks = NULL;
    void *d_output = NULL;
    void *d_scratch = NULL;
    HIP_CHECK(hipMalloc(&d_prefix, vector_bytes));
    HIP_CHECK(hipMalloc(&d_blocks, block_bytes));
    HIP_CHECK(hipMalloc(&d_output, vector_bytes));
    HIP_CHECK(hipMalloc(&d_scratch, vector_bytes));
    HIP_CHECK(hipMemcpy(d_prefix, prefix, vector_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_blocks, blocks, block_bytes,
                        hipMemcpyHostToDevice));

    for (uint32_t i = 0; i < K3_REAL_WEIGHT_PAIRS; i++) {
        reference_attn_res(
            expected, prefix, blocks,
            (const hip_bfloat16 *)pairs[i].norm.read.data,
            (const hip_bfloat16 *)pairs[i].projection.read.data,
            pairs[i].blocks);
        CHECK(k3_rocm_attn_res_bf16(
                  d_output, d_prefix, d_blocks,
                  pairs[i].norm.device, pairs[i].projection.device,
                  1u, K3_BLOCK_CAPACITY, pairs[i].blocks,
                  K3_HIDDEN, 1e-5f, NULL),
              "real AttnRes launch");
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(got, d_output, vector_bytes,
                            hipMemcpyDeviceToHost));
        float maximum_absolute = 0.0f;
        float maximum_relative = 0.0f;
        for (uint32_t d = 0; d < K3_HIDDEN; d++) {
            const float actual = (float)got[d];
            const float absolute = fabsf(actual - expected[d]);
            const float relative =
                absolute / fmaxf(fabsf(expected[d]), 1e-3f);
            maximum_absolute = fmaxf(maximum_absolute, absolute);
            maximum_relative = fmaxf(maximum_relative, relative);
            if (absolute > 0.08f && relative > 0.03f) {
                fprintf(stderr,
                        "FAIL: %s AttnRes mismatch at %u: "
                        "got=%f expected=%f abs=%f rel=%f\n",
                        pairs[i].label, d, actual, expected[d],
                        absolute, relative);
                return 1;
            }
        }
        printf("  real %-18s blocks=%u: max_abs=%.7f max_rel=%.7f\n",
               pairs[i].label, pairs[i].blocks,
               maximum_absolute, maximum_relative);
    }

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t step = 0; step < K3_BENCHMARK_STEPS; step++) {
        for (uint32_t num_blocks = 1u;
             num_blocks <= K3_BLOCK_CAPACITY; num_blocks++) {
            const uint32_t calls = num_blocks < 8u ? 24u : 18u;
            for (uint32_t call = 0; call < calls; call++) {
                CHECK(k3_rocm_attn_res_bf16(
                          d_output, d_prefix, d_blocks,
                          pairs[2].norm.device,
                          pairs[2].projection.device,
                          1u, K3_BLOCK_CAPACITY, num_blocks,
                          K3_HIDDEN, 1e-5f, NULL),
                      "timed AttnRes launch");
            }
        }
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float attn_res_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&attn_res_ms, start, stop));
    attn_res_ms /= (float)K3_BENCHMARK_STEPS;

    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t step = 0; step < K3_BENCHMARK_STEPS; step++) {
        for (uint32_t layer = 0; layer < K3_LAYERS; layer++) {
            const void *post_input = d_prefix;
            if (layer % 12u == 0u) {
                HIP_CHECK(hipMemcpyAsync(
                    (uint8_t *)d_blocks +
                        (uint64_t)(layer / 12u) * vector_bytes,
                    d_prefix, vector_bytes,
                    hipMemcpyDeviceToDevice, NULL));
            }
            CHECK(k3_rocm_rms_norm_bf16(
                      d_scratch, d_prefix, input_norm.device,
                      1u, K3_HIDDEN, 1e-5f, NULL),
                  "timed input norm launch");
            if (layer % 12u != 0u) {
                CHECK(k3_rocm_add_bf16(
                          d_output, d_prefix, d_scratch,
                          K3_HIDDEN, NULL),
                      "timed attention residual launch");
                post_input = d_output;
            }
            CHECK(k3_rocm_rms_norm_bf16(
                      d_scratch, post_input, post_norm.device,
                      1u, K3_HIDDEN, 1e-5f, NULL) &&
                  k3_rocm_add_bf16(
                      d_output, post_input, d_scratch,
                      K3_HIDDEN, NULL),
                  "timed norm/add launch");
        }
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float norm_add_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&norm_add_ms, start, stop));
    norm_add_ms /= (float)K3_BENCHMARK_STEPS;

    printf("  audited glue tensors: %u, %.3f MiB\n",
           audited_tensors,
           (double)audited_bytes / (1024.0 * 1024.0));
    printf("  exact AttnRes schedule: %u calls, %.3f ms/token\n",
           K3_ATTN_RES_CALLS, attn_res_ms);
    printf("  layer glue: %u RMSNorm + %u add + %u block save, "
           "%.3f ms/token\n",
           K3_NORM_CALLS, K3_ADD_CALLS, K3_BLOCK_SAVES,
           norm_add_ms);
    printf("  complete residual glue: %.3f ms/token\n",
           attn_res_ms + norm_add_ms);

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipFree(d_scratch));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_blocks));
    HIP_CHECK(hipFree(d_prefix));
    free(expected);
    free(got);
    free(blocks);
    free(prefix);
    release_tensor(&post_norm);
    release_tensor(&input_norm);
    for (uint32_t i = 0; i < K3_REAL_WEIGHT_PAIRS; i++) {
        release_tensor(&pairs[i].projection);
        release_tensor(&pairs[i].norm);
    }
    k3_st_model_close(&model);
    printf("K3 residual spine: PASS\n");
    return 0;
}
