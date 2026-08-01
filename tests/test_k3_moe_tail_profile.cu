#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <stdint.h>
#include <stdio.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        const hipError_t status_ = (call);                                  \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                     \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                    \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    K3_PROFILE_THREADS = 256,
    K3_PROFILE_TOP_K = 16,
    K3_PROFILE_LATENT = 3584,
    K3_PROFILE_HIDDEN = 7168,
    K3_PROFILE_MAX_TOKENS = 32768,
    K3_PROFILE_Q8_BLOCK = 128,
    K3_PROFILE_STEPS = 3,
};

static const uint32_t k_token_counts[] = {8192u, 16384u, 32768u};
static const uint32_t k_tiles[] = {16u, 32u, 64u};

static uint32_t blocks_for(uint64_t count) {
    uint64_t blocks =
        (count + K3_PROFILE_THREADS - 1u) / K3_PROFILE_THREADS;
    if (blocks > 65535u) blocks = 65535u;
    return (uint32_t)blocks;
}

__global__ static void fill_q8(int8_t *values, uint64_t count) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        values[index] =
            (int8_t)((int32_t)(index % 15u) - 7);
    }
}

__global__ static void fill_f32(float *values, uint64_t count) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        values[index] =
            (float)(1u + index % 4u) / 256.0f;
    }
}

__global__ static void fill_bf16(
        hip_bfloat16 *values,
        uint64_t count,
        uint32_t seed) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        uint32_t bits = (uint32_t)index ^ seed;
        bits ^= bits << 13u;
        bits ^= bits >> 17u;
        bits ^= bits << 5u;
        values[index] = hip_bfloat16(
            (float)((int32_t)(bits % 33u) - 16) / 512.0f);
    }
}

__global__ static void compare_bf16(
        const hip_bfloat16 *reference,
        const hip_bfloat16 *candidate,
        uint64_t count,
        unsigned long long *mismatches,
        uint32_t *max_abs_bits) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    unsigned long long local_mismatches = 0u;
    float local_max_abs = 0.0f;
    for (; index < count; index += stride) {
        const float left = (float)reference[index];
        const float right = (float)candidate[index];
        if (left != right) local_mismatches++;
        const float difference = fabsf(left - right);
        if (difference > local_max_abs) local_max_abs = difference;
    }
    if (local_mismatches != 0u) {
        atomicAdd(mismatches, local_mismatches);
    }
    atomicMax(max_abs_bits, __float_as_uint(local_max_abs));
}

static bool event_mean(
        hipEvent_t start,
        hipEvent_t stop,
        uint32_t steps,
        float *mean_ms) {
    if (hipEventRecord(stop, NULL) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) {
        return false;
    }
    float elapsed_ms = 0.0f;
    if (hipEventElapsedTime(&elapsed_ms, start, stop) != hipSuccess) {
        return false;
    }
    *mean_ms = elapsed_ms / (float)steps;
    return true;
}

int main(void) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 MoE-tail profile: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 MoE-tail model-shape profile on %s (%s)\n",
           properties.name, properties.gcnArchName);

    const uint64_t weight_count =
        (uint64_t)K3_PROFILE_HIDDEN * K3_PROFILE_LATENT;
    const uint64_t scale_count =
        weight_count / K3_PROFILE_Q8_BLOCK;
    const uint64_t input_count =
        (uint64_t)K3_PROFILE_MAX_TOKENS * K3_PROFILE_LATENT;
    const uint64_t output_count =
        (uint64_t)K3_PROFILE_MAX_TOKENS * K3_PROFILE_HIDDEN;
    const uint64_t expert_output_count =
        input_count * K3_PROFILE_TOP_K;
    const uint64_t route_weight_count =
        (uint64_t)K3_PROFILE_MAX_TOKENS * K3_PROFILE_TOP_K;

    void *quantized = NULL;
    void *scales = NULL;
    void *dequantized = NULL;
    void *expert_outputs = NULL;
    void *route_weights = NULL;
    void *routed = NULL;
    void *routed_norm = NULL;
    void *norm_weight = NULL;
    void *routed_full = NULL;
    void *shared_output = NULL;
    void *residual_output = NULL;
    void *difference_mismatches = NULL;
    void *difference_max_abs = NULL;
    HIP_CHECK(hipMalloc(&quantized, weight_count));
    HIP_CHECK(hipMalloc(&scales, scale_count * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &dequantized, weight_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &expert_outputs,
        expert_output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &route_weights,
        route_weight_count * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &routed, input_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &routed_norm, input_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &norm_weight,
        K3_PROFILE_LATENT * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &routed_full, output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &shared_output, output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &residual_output, output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &difference_mismatches, sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(
        &difference_max_abs, sizeof(uint32_t)));

    hipLaunchKernelGGL(
        fill_q8, dim3(blocks_for(weight_count)),
        dim3(K3_PROFILE_THREADS), 0, 0,
        (int8_t *)quantized, weight_count);
    hipLaunchKernelGGL(
        fill_f32, dim3(blocks_for(scale_count)),
        dim3(K3_PROFILE_THREADS), 0, 0,
        (float *)scales, scale_count);
    hipLaunchKernelGGL(
        fill_f32, dim3(blocks_for(route_weight_count)),
        dim3(K3_PROFILE_THREADS), 0, 0,
        (float *)route_weights, route_weight_count);
    hipLaunchKernelGGL(
        fill_bf16, dim3(blocks_for(expert_output_count)),
        dim3(K3_PROFILE_THREADS), 0, 0,
        (hip_bfloat16 *)expert_outputs,
        expert_output_count, UINT32_C(0x3af18821));
    hipLaunchKernelGGL(
        fill_bf16, dim3(blocks_for(K3_PROFILE_LATENT)),
        dim3(K3_PROFILE_THREADS), 0, 0,
        (hip_bfloat16 *)norm_weight,
        K3_PROFILE_LATENT, UINT32_C(0x8772c10d));
    hipLaunchKernelGGL(
        fill_bf16, dim3(blocks_for(output_count)),
        dim3(K3_PROFILE_THREADS), 0, 0,
        (hip_bfloat16 *)shared_output,
        output_count, UINT32_C(0x2db49fa1));
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    k3_rocm_blas_context *blas = NULL;
    CHECK(k3_rocm_blas_context_create(&blas),
          "creating hipBLAS context");

    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t step = 0u; step < K3_PROFILE_STEPS; step++) {
        CHECK(k3_rocm_q8_128_dequantize_bf16(
                  dequantized, quantized, scales,
                  K3_PROFILE_HIDDEN, K3_PROFILE_LATENT, NULL),
              "routed-up Q8 dequantization");
    }
    float dequant_ms = 0.0f;
    CHECK(event_mean(
              start, stop, K3_PROFILE_STEPS, &dequant_ms),
          "routed-up dequantization timing");

    for (uint32_t count_index = 0u;
         count_index <
             sizeof(k_token_counts) / sizeof(k_token_counts[0]);
         count_index++) {
        const uint32_t tokens = k_token_counts[count_index];
        printf("  tokens=%u\n", tokens);

        CHECK(k3_rocm_weighted_sum_bf16_batch(
                  routed, expert_outputs, route_weights,
                  tokens, K3_PROFILE_TOP_K,
                  K3_PROFILE_LATENT, NULL),
              "weighted-sum warmup");
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(start, NULL));
        for (uint32_t step = 0u; step < K3_PROFILE_STEPS; step++) {
            CHECK(k3_rocm_weighted_sum_bf16_batch(
                      routed, expert_outputs, route_weights,
                      tokens, K3_PROFILE_TOP_K,
                      K3_PROFILE_LATENT, NULL),
                  "weighted-sum timed launch");
        }
        float weighted_ms = 0.0f;
        CHECK(event_mean(
                  start, stop, K3_PROFILE_STEPS, &weighted_ms),
              "weighted-sum timing");

        CHECK(k3_rocm_rms_norm_bf16(
                  routed_norm, routed, norm_weight,
                  tokens, K3_PROFILE_LATENT, 1e-5f, NULL),
              "routed RMSNorm warmup");
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(start, NULL));
        for (uint32_t step = 0u; step < K3_PROFILE_STEPS; step++) {
            CHECK(k3_rocm_rms_norm_bf16(
                      routed_norm, routed, norm_weight,
                      tokens, K3_PROFILE_LATENT, 1e-5f, NULL),
                  "routed RMSNorm timed launch");
        }
        float norm_ms = 0.0f;
        CHECK(event_mean(
                  start, stop, K3_PROFILE_STEPS, &norm_ms),
              "routed RMSNorm timing");

        float tile_ms[3] = {0.0f, 0.0f, 0.0f};
        for (uint32_t tile_index = 0u;
             tile_index < sizeof(k_tiles) / sizeof(k_tiles[0]);
             tile_index++) {
            CHECK(k3_rocm_q8_128_gemm_tiled_bf16(
                      routed_full, quantized, scales, routed_norm,
                      tokens, K3_PROFILE_HIDDEN,
                      K3_PROFILE_LATENT, k_tiles[tile_index], NULL),
                  "routed-up Q8 warmup");
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipEventRecord(start, NULL));
            for (uint32_t step = 0u;
                 step < K3_PROFILE_STEPS; step++) {
                CHECK(k3_rocm_q8_128_gemm_tiled_bf16(
                          routed_full, quantized, scales, routed_norm,
                          tokens, K3_PROFILE_HIDDEN,
                          K3_PROFILE_LATENT, k_tiles[tile_index], NULL),
                      "routed-up Q8 timed launch");
            }
            CHECK(event_mean(
                      start, stop, K3_PROFILE_STEPS,
                      &tile_ms[tile_index]),
                  "routed-up Q8 timing");
        }

        const uint64_t current_output_count =
            (uint64_t)tokens * K3_PROFILE_HIDDEN;
        HIP_CHECK(hipMemcpy(
            shared_output, routed_full,
            current_output_count * sizeof(hip_bfloat16),
            hipMemcpyDeviceToDevice));

        CHECK(k3_rocm_blas_bf16_gemm_bf16(
                  blas, routed_full, dequantized, routed_norm,
                  tokens, K3_PROFILE_HIDDEN,
                  K3_PROFILE_LATENT, NULL),
              "routed-up hipBLAS warmup");
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(start, NULL));
        for (uint32_t step = 0u; step < K3_PROFILE_STEPS; step++) {
            CHECK(k3_rocm_blas_bf16_gemm_bf16(
                      blas, routed_full, dequantized, routed_norm,
                      tokens, K3_PROFILE_HIDDEN,
                      K3_PROFILE_LATENT, NULL),
                  "routed-up hipBLAS timed launch");
        }
        float blas_ms = 0.0f;
        CHECK(event_mean(
                  start, stop, K3_PROFILE_STEPS, &blas_ms),
              "routed-up hipBLAS timing");

        HIP_CHECK(hipMemset(
            difference_mismatches, 0,
            sizeof(unsigned long long)));
        HIP_CHECK(hipMemset(
            difference_max_abs, 0, sizeof(uint32_t)));
        hipLaunchKernelGGL(
            compare_bf16,
            dim3(blocks_for(current_output_count)),
            dim3(K3_PROFILE_THREADS), 0, 0,
            (const hip_bfloat16 *)shared_output,
            (const hip_bfloat16 *)routed_full,
            current_output_count,
            (unsigned long long *)difference_mismatches,
            (uint32_t *)difference_max_abs);
        HIP_CHECK(hipGetLastError());
        unsigned long long mismatches = 0u;
        uint32_t max_abs_bits = 0u;
        HIP_CHECK(hipMemcpy(
            &mismatches, difference_mismatches,
            sizeof(mismatches), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(
            &max_abs_bits, difference_max_abs,
            sizeof(max_abs_bits), hipMemcpyDeviceToHost));
        union {
            uint32_t bits;
            float value;
        } max_abs = { max_abs_bits };
        CHECK(max_abs.value <= 0.0001220703125f &&
                  mismatches * 1000u < current_output_count,
              "routed-up hipBLAS drift exceeded the profile envelope");

        const uint64_t hidden_elements =
            (uint64_t)tokens * K3_PROFILE_HIDDEN;
        CHECK(k3_rocm_add_bf16(
                  residual_output, routed_full, shared_output,
                  hidden_elements, NULL),
              "residual-add warmup");
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(start, NULL));
        for (uint32_t step = 0u; step < K3_PROFILE_STEPS; step++) {
            CHECK(k3_rocm_add_bf16(
                      residual_output, routed_full, shared_output,
                      hidden_elements, NULL) &&
                  k3_rocm_add_bf16(
                      routed_full, residual_output, shared_output,
                      hidden_elements, NULL),
                  "residual adds timed launch");
        }
        float adds_ms = 0.0f;
        CHECK(event_mean(
                  start, stop, K3_PROFILE_STEPS, &adds_ms),
              "residual-add timing");

        printf("    weighted=%.3f ms norm=%.3f ms adds(2)=%.3f ms\n",
               weighted_ms, norm_ms, adds_ms);
        printf("    routed-up Q8 tile16/32/64="
               "%.3f/%.3f/%.3f ms\n",
               tile_ms[0], tile_ms[1], tile_ms[2]);
        printf("    dequant+hipBLAS=%.3f+%.3f=%.3f ms\n",
               dequant_ms, blas_ms, dequant_ms + blas_ms);
        printf("    hipBLAS drift: mismatches=%llu/%llu "
               "max_abs=%.8g\n",
               mismatches,
               (unsigned long long)current_output_count,
               max_abs.value);
        printf("    isolated production tail=%.3f ms/layer; "
               "hipBLAS candidate=%.3f ms/layer\n",
               weighted_ms + norm_ms + tile_ms[0] + adds_ms,
               weighted_ms + norm_ms + dequant_ms +
                   blas_ms + adds_ms);
    }

    k3_rocm_blas_context_destroy(blas);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipFree(difference_max_abs));
    HIP_CHECK(hipFree(difference_mismatches));
    HIP_CHECK(hipFree(residual_output));
    HIP_CHECK(hipFree(shared_output));
    HIP_CHECK(hipFree(routed_full));
    HIP_CHECK(hipFree(norm_weight));
    HIP_CHECK(hipFree(routed_norm));
    HIP_CHECK(hipFree(routed));
    HIP_CHECK(hipFree(route_weights));
    HIP_CHECK(hipFree(expert_outputs));
    HIP_CHECK(hipFree(dequantized));
    HIP_CHECK(hipFree(scales));
    HIP_CHECK(hipFree(quantized));
    printf("K3 MoE-tail model-shape profile: PASS\n");
    return 0;
}
