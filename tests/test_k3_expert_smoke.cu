#include "k3_rocm_ops.h"
#include "k3_safetensors.h"
#include "k3_io_uring.h"

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
    K3_LATENT_SIZE = 3584,
    K3_EXPERT_SIZE = 3072,
    K3_MXFP4_GROUP = 32,
    K3_REDUCTION_THREADS = 256,
};

static uint32_t rng_state = UINT32_C(0x4b334558);

static float random_input(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & UINT32_C(0x00ffffff)) /
            (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * 0.25f;
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

/*
 * Independent scalar correctness oracle for the native SafeTensors layout.
 * Its 256-way reduction deliberately remains separate from the optimized GPU
 * work assignment; comparisons use the established BF16 component tolerance.
 */
static void reference_gemv(hip_bfloat16 *output,
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
                        scales[(uint64_t)row * columns / K3_MXFP4_GROUP +
                               column / K3_MXFP4_GROUP]);
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

static bool compare_vectors(const char *label,
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
    printf("  %-10s max_abs=%-10.6f max_rel=%.6f\n",
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

static double seconds_between(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 real expert smoke: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));

    char error[512];
    k3_st_model model;
    if (!k3_st_model_open(&model, root, 96u, error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }

    static const char *names[] = {
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight_packed",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight_scale",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight_packed",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight_scale",
    };
    const k3_st_tensor *tensor[6];
    for (size_t i = 0; i < 6; i++) {
        tensor[i] = k3_st_find(&model, names[i]);
        CHECK(tensor[i] != NULL, "real expert tensor lookup");
    }
    const uint64_t physical_start = tensor[0]->physical_offset;
    const uint64_t physical_end =
        tensor[5]->physical_offset + tensor[5]->byte_length;
    CHECK(physical_end - physical_start == UINT64_C(17547264),
          "unexpected real expert span");

    struct timespec read_start, read_end;
    k3_st_read read;
    clock_gettime(CLOCK_MONOTONIC, &read_start);
    CHECK(k3_st_read_span(&model, tensor[0]->shard, physical_start,
                          physical_end - physical_start, 4096u,
                          &read, error, sizeof(error)),
          error);
    clock_gettime(CLOCK_MONOTONIC, &read_end);

    const uint8_t *host_weight[6];
    for (size_t i = 0; i < 6; i++) {
        host_weight[i] =
            read.data + (tensor[i]->physical_offset - physical_start);
    }
    for (size_t scale_index = 1; scale_index < 6; scale_index += 2) {
        for (uint64_t i = 0; i < tensor[scale_index]->byte_length; i++) {
            CHECK(host_weight[scale_index][i] != UINT8_C(0xff),
                  "MXFP4 source contains invalid E8M0 NaN scale");
        }
    }

    const size_t latent_bytes =
        K3_LATENT_SIZE * sizeof(hip_bfloat16);
    const size_t expert_bytes =
        K3_EXPERT_SIZE * sizeof(hip_bfloat16);
    hip_bfloat16 *input = (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *gpu_w1 = (hip_bfloat16 *)malloc(expert_bytes);
    hip_bfloat16 *gpu_w3 = (hip_bfloat16 *)malloc(expert_bytes);
    hip_bfloat16 *gpu_activation = (hip_bfloat16 *)malloc(expert_bytes);
    hip_bfloat16 *gpu_output = (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *mapped_output = (hip_bfloat16 *)malloc(latent_bytes);
    hip_bfloat16 *cpu_w1 = (hip_bfloat16 *)malloc(expert_bytes);
    hip_bfloat16 *cpu_w3 = (hip_bfloat16 *)malloc(expert_bytes);
    hip_bfloat16 *cpu_activation = (hip_bfloat16 *)malloc(expert_bytes);
    hip_bfloat16 *cpu_output = (hip_bfloat16 *)malloc(latent_bytes);
    CHECK(input && gpu_w1 && gpu_w3 && gpu_activation && gpu_output &&
          mapped_output &&
          cpu_w1 && cpu_w3 && cpu_activation && cpu_output,
          "expert activation allocation");
    for (uint32_t i = 0; i < K3_LATENT_SIZE; i++) {
        input[i] = float_to_bf16(random_input());
    }

    uint8_t *d_expert = NULL;
    hip_bfloat16 *d_input = NULL;
    hip_bfloat16 *d_w1 = NULL;
    hip_bfloat16 *d_w3 = NULL;
    hip_bfloat16 *d_activation = NULL;
    hip_bfloat16 *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_expert, read.data_bytes));
    HIP_CHECK(hipMalloc(&d_input, latent_bytes));
    HIP_CHECK(hipMalloc(&d_w1, expert_bytes));
    HIP_CHECK(hipMalloc(&d_w3, expert_bytes));
    HIP_CHECK(hipMalloc(&d_activation, expert_bytes));
    HIP_CHECK(hipMalloc(&d_output, latent_bytes));
    hipEvent_t copy_start_event, copy_end_event;
    HIP_CHECK(hipEventCreate(&copy_start_event));
    HIP_CHECK(hipEventCreate(&copy_end_event));
    HIP_CHECK(hipEventRecord(copy_start_event, NULL));
    HIP_CHECK(hipMemcpyAsync(d_expert, read.data, read.data_bytes,
                             hipMemcpyHostToDevice, NULL));
    HIP_CHECK(hipEventRecord(copy_end_event, NULL));
    HIP_CHECK(hipEventSynchronize(copy_end_event));
    float copy_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&copy_ms, copy_start_event, copy_end_event));
    HIP_CHECK(hipMemcpy(d_input, input, latent_bytes,
                        hipMemcpyHostToDevice));

    const uint8_t *device_weight[6];
    for (size_t i = 0; i < 6; i++) {
        device_weight[i] =
            d_expert + (tensor[i]->physical_offset - physical_start);
    }

    /* Warm up code objects and caches before recording steady-state latency. */
    CHECK(k3_rocm_mxfp4_gemv_bf16(d_w1, device_weight[0],
                                   device_weight[1], d_input,
                                   K3_EXPERT_SIZE, K3_LATENT_SIZE, NULL),
          "real w1 launch");
    CHECK(k3_rocm_mxfp4_gemv_bf16(d_w3, device_weight[4],
                                   device_weight[5], d_input,
                                   K3_EXPERT_SIZE, K3_LATENT_SIZE, NULL),
          "real w3 launch");
    CHECK(k3_rocm_situ_bf16(d_activation, d_w1, d_w3,
                            K3_EXPERT_SIZE, 4.0f, 25.0f, NULL),
          "real SiTU launch");
    CHECK(k3_rocm_mxfp4_gemv_bf16(d_output, device_weight[2],
                                   device_weight[3], d_activation,
                                   K3_LATENT_SIZE, K3_EXPERT_SIZE, NULL),
          "real w2 launch");
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start_event, end_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&end_event));
    const unsigned iterations = 20;
    HIP_CHECK(hipEventRecord(start_event, NULL));
    for (unsigned iteration = 0; iteration < iterations; iteration++) {
        CHECK(k3_rocm_mxfp4_gemv_bf16(d_w1, device_weight[0],
                                       device_weight[1], d_input,
                                       K3_EXPERT_SIZE, K3_LATENT_SIZE, NULL),
              "timed w1 launch");
        CHECK(k3_rocm_mxfp4_gemv_bf16(d_w3, device_weight[4],
                                       device_weight[5], d_input,
                                       K3_EXPERT_SIZE, K3_LATENT_SIZE, NULL),
              "timed w3 launch");
        CHECK(k3_rocm_situ_bf16(d_activation, d_w1, d_w3,
                                K3_EXPERT_SIZE, 4.0f, 25.0f, NULL),
              "timed SiTU launch");
        CHECK(k3_rocm_mxfp4_gemv_bf16(d_output, device_weight[2],
                                       device_weight[3], d_activation,
                                       K3_LATENT_SIZE, K3_EXPERT_SIZE, NULL),
              "timed w2 launch");
    }
    HIP_CHECK(hipEventRecord(end_event, NULL));
    HIP_CHECK(hipEventSynchronize(end_event));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event, end_event));

    HIP_CHECK(hipMemcpy(gpu_w1, d_w1, expert_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_w3, d_w3, expert_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_activation, d_activation, expert_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_output, d_output, latent_bytes,
                        hipMemcpyDeviceToHost));

    const uint64_t mapped_start =
        physical_start & ~UINT64_C(4095);
    const uint64_t mapped_end =
        (physical_end + UINT64_C(4095)) & ~UINT64_C(4095);
    const uint64_t mapped_bytes = mapped_end - mapped_start;
    void *mapped_allocation = NULL;
    void *mapped_allocation_second = NULL;
    HIP_CHECK(hipHostMalloc(&mapped_allocation, (size_t)mapped_bytes,
                            hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(&mapped_allocation_second, (size_t)mapped_bytes,
                            hipHostMallocMapped));
    CHECK((uintptr_t)mapped_allocation % 4096u == 0,
          "GPU-visible allocation is not 4 KiB aligned");
    k3_st_read_view mapped_read;
    struct timespec mapped_read_start, mapped_read_end;
    clock_gettime(CLOCK_MONOTONIC, &mapped_read_start);
    CHECK(k3_st_read_span_into(&model, tensor[0]->shard, physical_start,
                               physical_end - physical_start, 4096u,
                               mapped_allocation, mapped_bytes, &mapped_read,
                               error, sizeof(error)),
          error);
    clock_gettime(CLOCK_MONOTONIC, &mapped_read_end);
    CHECK(mapped_read.used_direct_io,
          "GPU-visible expert read did not use O_DIRECT");

    struct iovec fixed_buffers[2] = {
        { mapped_allocation, (size_t)mapped_bytes },
        { mapped_allocation_second, (size_t)mapped_bytes },
    };
    k3_io_uring *fixed_ring = NULL;
    CHECK(k3_io_uring_create(&fixed_ring, fixed_buffers, 2u,
                             error, sizeof(error)),
          error);
    k3_io_request fixed_requests[2] = {
        {
            model.shards[tensor[0]->shard].direct_fd,
            mapped_start,
            (uint32_t)mapped_bytes,
            0u,
            UINT64_C(0x4b330000),
        },
        {
            model.shards[tensor[0]->shard].direct_fd,
            mapped_start,
            (uint32_t)mapped_bytes,
            1u,
            UINT64_C(0x4b330001),
        },
    };
    struct timespec fixed_start, fixed_end;
    clock_gettime(CLOCK_MONOTONIC, &fixed_start);
    CHECK(k3_io_uring_submit(fixed_ring, fixed_requests, 2u,
                             error, sizeof(error)),
          error);
    unsigned fixed_completed = 0;
    while (fixed_completed < 2u) {
        k3_io_completion completions[2];
        uint16_t completion_count = 0;
        CHECK(k3_io_uring_wait(fixed_ring, completions, 2u,
                               &completion_count, error, sizeof(error)),
              error);
        for (uint16_t i = 0; i < completion_count; i++) {
            CHECK(completions[i].result == (int32_t)mapped_bytes,
                  "fixed GPU-visible expert read failed");
        }
        fixed_completed += completion_count;
    }
    clock_gettime(CLOCK_MONOTONIC, &fixed_end);
    k3_io_uring_destroy(fixed_ring);
    void *mapped_device_allocation = NULL;
    HIP_CHECK(hipHostGetDevicePointer(&mapped_device_allocation,
                                      mapped_allocation, 0));
    uint8_t *mapped_device_data =
        (uint8_t *)mapped_device_allocation +
        (mapped_read.data - (uint8_t *)mapped_allocation);
    const uint8_t *mapped_device_weight[6];
    for (size_t i = 0; i < 6; i++) {
        mapped_device_weight[i] =
            mapped_device_data +
            (tensor[i]->physical_offset - physical_start);
    }

    CHECK(k3_rocm_mxfp4_gemv_bf16(
              d_w1, mapped_device_weight[0], mapped_device_weight[1], d_input,
              K3_EXPERT_SIZE, K3_LATENT_SIZE, NULL),
          "mapped real w1 launch");
    CHECK(k3_rocm_mxfp4_gemv_bf16(
              d_w3, mapped_device_weight[4], mapped_device_weight[5], d_input,
              K3_EXPERT_SIZE, K3_LATENT_SIZE, NULL),
          "mapped real w3 launch");
    CHECK(k3_rocm_situ_bf16(d_activation, d_w1, d_w3,
                            K3_EXPERT_SIZE, 4.0f, 25.0f, NULL),
          "mapped real SiTU launch");
    CHECK(k3_rocm_mxfp4_gemv_bf16(
              d_output, mapped_device_weight[2], mapped_device_weight[3],
              d_activation, K3_LATENT_SIZE, K3_EXPERT_SIZE, NULL),
          "mapped real w2 launch");
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t mapped_start_event, mapped_end_event;
    HIP_CHECK(hipEventCreate(&mapped_start_event));
    HIP_CHECK(hipEventCreate(&mapped_end_event));
    HIP_CHECK(hipEventRecord(mapped_start_event, NULL));
    for (unsigned iteration = 0; iteration < iterations; iteration++) {
        CHECK(k3_rocm_mxfp4_gemv_bf16(
                  d_w1, mapped_device_weight[0], mapped_device_weight[1],
                  d_input, K3_EXPERT_SIZE, K3_LATENT_SIZE, NULL),
              "timed mapped w1 launch");
        CHECK(k3_rocm_mxfp4_gemv_bf16(
                  d_w3, mapped_device_weight[4], mapped_device_weight[5],
                  d_input, K3_EXPERT_SIZE, K3_LATENT_SIZE, NULL),
              "timed mapped w3 launch");
        CHECK(k3_rocm_situ_bf16(d_activation, d_w1, d_w3,
                                K3_EXPERT_SIZE, 4.0f, 25.0f, NULL),
              "timed mapped SiTU launch");
        CHECK(k3_rocm_mxfp4_gemv_bf16(
                  d_output, mapped_device_weight[2], mapped_device_weight[3],
                  d_activation, K3_LATENT_SIZE, K3_EXPERT_SIZE, NULL),
              "timed mapped w2 launch");
    }
    HIP_CHECK(hipEventRecord(mapped_end_event, NULL));
    HIP_CHECK(hipEventSynchronize(mapped_end_event));
    float mapped_elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&mapped_elapsed_ms,
                                  mapped_start_event, mapped_end_event));
    HIP_CHECK(hipMemcpy(mapped_output, d_output, latent_bytes,
                        hipMemcpyDeviceToHost));

    struct timespec oracle_start, oracle_end;
    clock_gettime(CLOCK_MONOTONIC, &oracle_start);
    reference_gemv(cpu_w1, host_weight[0], host_weight[1], input,
                   K3_EXPERT_SIZE, K3_LATENT_SIZE);
    reference_gemv(cpu_w3, host_weight[4], host_weight[5], input,
                   K3_EXPERT_SIZE, K3_LATENT_SIZE);
    for (uint32_t i = 0; i < K3_EXPERT_SIZE; i++) {
        float gate = bf16_to_float(cpu_w1[i]);
        float up = bf16_to_float(cpu_w3[i]);
        float activated =
            (4.0f * tanhf(gate / 4.0f) / (1.0f + expf(-gate))) *
            (25.0f * tanhf(up / 25.0f));
        cpu_activation[i] = float_to_bf16(activated);
    }
    reference_gemv(cpu_output, host_weight[2], host_weight[3],
                   cpu_activation, K3_LATENT_SIZE, K3_EXPERT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &oracle_end);

    printf("K3 real routed-expert smoke: layer=1 expert=0\n");
    printf("  SSD read: %.2f MiB in %.3f ms (%s)\n",
           (double)read.data_bytes / (1024.0 * 1024.0),
           seconds_between(read_start, read_end) * 1000.0,
           read.used_direct_io ? "O_DIRECT" : "buffered");
    printf("  explicit expert upload: %.3f ms\n", copy_ms);
    const double source_gib =
        (double)(physical_end - physical_start) /
        (1024.0 * 1024.0 * 1024.0);
    const double resident_ms =
        elapsed_ms / (double)iterations;
    const double mapped_ms =
        mapped_elapsed_ms / (double)iterations;
    printf("  ROCm forward: %.3f ms/expert, %.1f GiB/s source "
           "(20-run mean)\n",
           resident_ms, source_gib / (resident_ms / 1000.0));
    printf("  GPU-visible O_DIRECT read: %.3f ms\n",
           seconds_between(mapped_read_start, mapped_read_end) * 1000.0);
    printf("  GPU-visible fixed-buffer QD2: %.3f ms for two reads\n",
           seconds_between(fixed_start, fixed_end) * 1000.0);
    printf("  GPU-visible forward: %.3f ms/expert, %.1f GiB/s source "
           "(20-run mean)\n",
           mapped_ms, source_gib / (mapped_ms / 1000.0));
    printf("  CPU oracle: %.3f ms\n",
           seconds_between(oracle_start, oracle_end) * 1000.0);
    CHECK(compare_vectors("w1", gpu_w1, cpu_w1, K3_EXPERT_SIZE,
                          0.0625f, 0.02f),
          "real w1 correctness");
    CHECK(compare_vectors("w3", gpu_w3, cpu_w3, K3_EXPERT_SIZE,
                          0.0625f, 0.02f),
          "real w3 correctness");
    CHECK(compare_vectors("SiTU", gpu_activation, cpu_activation,
                          K3_EXPERT_SIZE, 0.125f, 0.03f),
          "real SiTU correctness");
    CHECK(compare_vectors("w2", gpu_output, cpu_output, K3_LATENT_SIZE,
                          0.25f, 0.04f),
          "real w2 correctness");
    CHECK(compare_vectors("mapped w2", mapped_output, cpu_output,
                          K3_LATENT_SIZE, 0.25f, 0.04f),
          "GPU-visible real w2 correctness");
    printf("  output FNV-1a64: 0x%016" PRIx64 "\n",
           fnv1a64(gpu_output, latent_bytes));
    printf("K3 real routed-expert smoke: PASS\n");

    HIP_CHECK(hipEventDestroy(end_event));
    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(mapped_end_event));
    HIP_CHECK(hipEventDestroy(mapped_start_event));
    HIP_CHECK(hipHostFree(mapped_allocation));
    HIP_CHECK(hipHostFree(mapped_allocation_second));
    HIP_CHECK(hipEventDestroy(copy_end_event));
    HIP_CHECK(hipEventDestroy(copy_start_event));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_activation));
    HIP_CHECK(hipFree(d_w3));
    HIP_CHECK(hipFree(d_w1));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_expert));
    free(cpu_output);
    free(cpu_activation);
    free(cpu_w3);
    free(cpu_w1);
    free(gpu_output);
    free(mapped_output);
    free(gpu_activation);
    free(gpu_w3);
    free(gpu_w1);
    free(input);
    k3_st_read_release(&read);
    k3_st_model_close(&model);
    return 0;
}
