#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                        \
                    hipGetErrorString(status_));                            \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                       \
            return false;                                                   \
        }                                                                   \
    } while (0)

enum {
    K3_THREADS = 256,
    K3_STATIC_ROWS = 12288,
    K3_STATIC_COLUMNS = 7168,
    K3_STATIC_TOKENS = 8192,
    K3_ROUTER_ROWS = 896,
    K3_ROUTER_COLUMNS = 7168,
    K3_ROUTER_TOKENS = 8192,
    K3_EXPERT_GATE_ROWS = 3072,
    K3_EXPERT_GATE_COLUMNS = 3584,
    K3_EXPERT_DOWN_ROWS = 3584,
    K3_EXPERT_DOWN_COLUMNS = 3072,
    K3_EXPERT_MAX_TOKENS = 1170,
    K3_Q8_BLOCK = 128,
    K3_Q8_STEPS = 3,
    K3_ROUTER_STEPS = 5,
    K3_EXPERT_STEPS = 20,
};

static const uint32_t k_tiles[] = {16u, 32u, 64u};
static const uint32_t k_expert_counts[] = {
    9u, 32u, 64u, 128u, 146u, 192u, 585u, 1170u,
};

__global__ static void fill_bf16(
        hip_bfloat16 *values, uint64_t count, uint32_t seed) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        uint32_t bits = (uint32_t)index ^ seed;
        bits ^= bits << 13u;
        bits ^= bits >> 17u;
        bits ^= bits << 5u;
        const int32_t centered =
            (int32_t)(bits % 33u) - 16;
        values[index] =
            hip_bfloat16((float)centered / 512.0f);
    }
}

__global__ static void fill_q8(
        int8_t *values, uint64_t count) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        values[index] =
            (int8_t)((int32_t)(index % 15u) - 7);
    }
}

__global__ static void fill_f32_scales(
        float *values, uint64_t count) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        values[index] =
            (float)(1u + index % 4u) / 256.0f;
    }
}

__global__ static void fill_mxfp4(
        uint8_t *packed,
        uint64_t packed_count,
        uint8_t *scales,
        uint64_t scale_count) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = index; i < packed_count; i += stride) {
        const uint8_t low = (uint8_t)(i % 15u);
        const uint8_t high = (uint8_t)((i * 7u + 3u) % 15u);
        packed[i] = (uint8_t)(low | (high << 4u));
    }
    for (uint64_t i = index; i < scale_count; i += stride) {
        scales[i] = (uint8_t)(124u + i % 4u);
    }
}

__global__ static void dequant_q8_bf16(
        hip_bfloat16 *output,
        const int8_t *quantized,
        const float *scales,
        uint64_t count) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        output[index] = hip_bfloat16(
            (float)quantized[index] *
            scales[index / K3_Q8_BLOCK]);
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
    float local_max = 0.0f;
    for (; index < count; index += stride) {
        const uint16_t left =
            ((const uint16_t *)reference)[index];
        const uint16_t right =
            ((const uint16_t *)candidate)[index];
        local_mismatches += left != right;
        const float difference = fabsf(
            (float)reference[index] -
            (float)candidate[index]);
        local_max = fmaxf(local_max, difference);
    }
    if (local_mismatches != 0u) {
        atomicAdd(mismatches, local_mismatches);
    }
    atomicMax(max_abs_bits, __float_as_uint(local_max));
}

__global__ static void compare_f32(
        const float *reference,
        const float *candidate,
        uint64_t count,
        uint32_t *max_abs_bits) {
    uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride =
        (uint64_t)gridDim.x * blockDim.x;
    float local_max = 0.0f;
    for (; index < count; index += stride) {
        local_max = fmaxf(
            local_max,
            fabsf(reference[index] - candidate[index]));
    }
    atomicMax(max_abs_bits, __float_as_uint(local_max));
}

static uint32_t blocks_for(uint64_t count) {
    uint64_t blocks =
        (count + K3_THREADS - 1u) / K3_THREADS;
    if (blocks > 65535u) blocks = 65535u;
    return (uint32_t)blocks;
}

static float float_from_bits(uint32_t bits) {
    union {
        uint32_t bits;
        float value;
    } converted;
    converted.bits = bits;
    return converted.value;
}

static void probe_mxfp4_hipblaslt(void) {
    hipblasLtHandle_t handle = NULL;
    hipblasLtMatmulDesc_t operation = NULL;
    hipblasLtMatrixLayout_t weight_layout = NULL;
    hipblasLtMatrixLayout_t input_layout = NULL;
    hipblasLtMatrixLayout_t output_layout = NULL;
    hipblasLtMatmulPreference_t preference = NULL;
    hipblasStatus_t status = hipblasLtCreate(&handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        printf("  hipBLASLt MXFP4 probe: handle status=%d\n",
               (int)status);
        return;
    }
    status = hipblasLtMatmulDescCreate(
        &operation, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    const hipblasOperation_t transpose = HIPBLAS_OP_T;
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSA,
            &transpose, sizeof(transpose));
    }
    const hipblasLtMatmulMatrixScale_t block_scale =
        HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
            &block_scale, sizeof(block_scale));
    }
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatrixLayoutCreate(
            &weight_layout,
            (hipDataType)HIP_R_4F_E2M1_EXT,
            K3_EXPERT_GATE_COLUMNS,
            K3_EXPERT_GATE_ROWS,
            K3_EXPERT_GATE_COLUMNS);
    }
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatrixLayoutCreate(
            &input_layout, HIP_R_16BF,
            K3_EXPERT_GATE_COLUMNS,
            146u, K3_EXPERT_GATE_COLUMNS);
    }
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatrixLayoutCreate(
            &output_layout, HIP_R_16BF,
            K3_EXPERT_GATE_ROWS,
            146u, K3_EXPERT_GATE_ROWS);
    }
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatmulPreferenceCreate(&preference);
    }
    uint64_t workspace_bytes = UINT64_C(64) << 20u;
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatmulPreferenceSetAttribute(
            preference,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes));
    }
    hipblasLtMatmulHeuristicResult_t result;
    int returned = 0;
    if (status == HIPBLAS_STATUS_SUCCESS) {
        fflush(stderr);
        const int saved_stderr = dup(STDERR_FILENO);
        const int null_stderr =
            open("/dev/null", O_WRONLY);
        if (saved_stderr >= 0 && null_stderr >= 0) {
            (void)dup2(null_stderr, STDERR_FILENO);
        }
        status = hipblasLtMatmulAlgoGetHeuristic(
            handle, operation,
            weight_layout, input_layout,
            output_layout, output_layout,
            preference, 1, &result, &returned);
        fflush(stderr);
        if (saved_stderr >= 0) {
            (void)dup2(saved_stderr, STDERR_FILENO);
            (void)close(saved_stderr);
        }
        if (null_stderr >= 0) {
            (void)close(null_stderr);
        }
    }
    printf("  hipBLASLt MXFP4/BF16 blockscale probe: "
           "status=%d algorithms=%d%s\n",
           (int)status, returned,
           status == HIPBLAS_STATUS_SUCCESS && returned > 0 ?
               " (execution validation required)" :
               " (no usable candidate)");
    if (preference) {
        (void)hipblasLtMatmulPreferenceDestroy(preference);
    }
    if (output_layout) {
        (void)hipblasLtMatrixLayoutDestroy(output_layout);
    }
    if (input_layout) {
        (void)hipblasLtMatrixLayoutDestroy(input_layout);
    }
    if (weight_layout) {
        (void)hipblasLtMatrixLayoutDestroy(weight_layout);
    }
    if (operation) {
        (void)hipblasLtMatmulDescDestroy(operation);
    }
    (void)hipblasLtDestroy(handle);
}

static bool bf16_difference(
        const void *reference,
        const void *candidate,
        uint64_t count,
        uint64_t *mismatches,
        float *max_abs) {
    unsigned long long *d_mismatches = NULL;
    uint32_t *d_max_abs = NULL;
    HIP_CHECK(hipMalloc(&d_mismatches, sizeof(*d_mismatches)));
    HIP_CHECK(hipMalloc(&d_max_abs, sizeof(*d_max_abs)));
    HIP_CHECK(hipMemset(d_mismatches, 0, sizeof(*d_mismatches)));
    HIP_CHECK(hipMemset(d_max_abs, 0, sizeof(*d_max_abs)));
    hipLaunchKernelGGL(
        compare_bf16, dim3(blocks_for(count)),
        dim3(K3_THREADS), 0, 0,
        (const hip_bfloat16 *)reference,
        (const hip_bfloat16 *)candidate,
        count, d_mismatches, d_max_abs);
    HIP_CHECK(hipGetLastError());
    unsigned long long host_mismatches = 0u;
    uint32_t host_max_abs = 0u;
    HIP_CHECK(hipMemcpy(
        &host_mismatches, d_mismatches,
        sizeof(host_mismatches), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        &host_max_abs, d_max_abs,
        sizeof(host_max_abs), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_max_abs));
    HIP_CHECK(hipFree(d_mismatches));
    *mismatches = (uint64_t)host_mismatches;
    *max_abs = float_from_bits(host_max_abs);
    return true;
}

static bool f32_difference(
        const void *reference,
        const void *candidate,
        uint64_t count,
        float *max_abs) {
    uint32_t *d_max_abs = NULL;
    HIP_CHECK(hipMalloc(&d_max_abs, sizeof(*d_max_abs)));
    HIP_CHECK(hipMemset(d_max_abs, 0, sizeof(*d_max_abs)));
    hipLaunchKernelGGL(
        compare_f32, dim3(blocks_for(count)),
        dim3(K3_THREADS), 0, 0,
        (const float *)reference,
        (const float *)candidate,
        count, d_max_abs);
    HIP_CHECK(hipGetLastError());
    uint32_t host_max_abs = 0u;
    HIP_CHECK(hipMemcpy(
        &host_max_abs, d_max_abs,
        sizeof(host_max_abs), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_max_abs));
    *max_abs = float_from_bits(host_max_abs);
    return true;
}

static bool elapsed_events(
        hipEvent_t start, hipEvent_t stop,
        uint32_t steps, float *mean_ms) {
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    *mean_ms = elapsed_ms / (float)steps;
    return true;
}

static bool benchmark_q8_tile(
        void *output, const void *quantized,
        const void *scales, const void *input,
        uint32_t tile, float *mean_ms) {
    CHECK(k3_rocm_q8_128_gemm_tiled_bf16(
              output, quantized, scales, input,
              K3_STATIC_TOKENS, K3_STATIC_ROWS,
              K3_STATIC_COLUMNS, tile, NULL),
          "Q8 warmup launch");
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t step = 0u; step < K3_Q8_STEPS; step++) {
        CHECK(k3_rocm_q8_128_gemm_tiled_bf16(
                  output, quantized, scales, input,
                  K3_STATIC_TOKENS, K3_STATIC_ROWS,
                  K3_STATIC_COLUMNS, tile, NULL),
              "Q8 timed launch");
    }
    CHECK(elapsed_events(
              start, stop, K3_Q8_STEPS, mean_ms),
          "Q8 event timing");
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    return true;
}

static bool benchmark_q8(void) {
    const uint64_t weight_count =
        (uint64_t)K3_STATIC_ROWS * K3_STATIC_COLUMNS;
    const uint64_t scale_count =
        weight_count / K3_Q8_BLOCK;
    const uint64_t input_count =
        (uint64_t)K3_STATIC_TOKENS * K3_STATIC_COLUMNS;
    const uint64_t output_count =
        (uint64_t)K3_STATIC_TOKENS * K3_STATIC_ROWS;
    void *quantized = NULL;
    void *scales = NULL;
    void *input = NULL;
    void *reference = NULL;
    void *output = NULL;
    void *dequantized = NULL;
    HIP_CHECK(hipMalloc(&quantized, weight_count));
    HIP_CHECK(hipMalloc(&scales, scale_count * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &input, input_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &reference, output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &output, output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &dequantized, weight_count * sizeof(hip_bfloat16)));
    hipLaunchKernelGGL(
        fill_q8, dim3(blocks_for(weight_count)),
        dim3(K3_THREADS), 0, 0,
        (int8_t *)quantized, weight_count);
    hipLaunchKernelGGL(
        fill_f32_scales, dim3(blocks_for(scale_count)),
        dim3(K3_THREADS), 0, 0,
        (float *)scales, scale_count);
    hipLaunchKernelGGL(
        fill_bf16, dim3(blocks_for(input_count)),
        dim3(K3_THREADS), 0, 0,
        (hip_bfloat16 *)input, input_count,
        UINT32_C(0x41ac783d));
    HIP_CHECK(hipDeviceSynchronize());

    printf("  Q8-128 static projection: tokens=%u rows=%u cols=%u\n",
           K3_STATIC_TOKENS, K3_STATIC_ROWS, K3_STATIC_COLUMNS);
    for (uint32_t index = 0u;
         index < sizeof(k_tiles) / sizeof(k_tiles[0]); index++) {
        const uint32_t tile = k_tiles[index];
        float mean_ms = 0.0f;
        CHECK(benchmark_q8_tile(
                  output, quantized, scales, input,
                  tile, &mean_ms),
              "Q8 tile benchmark");
        if (tile == 16u) {
            HIP_CHECK(hipMemcpy(
                reference, output,
                output_count * sizeof(hip_bfloat16),
                hipMemcpyDeviceToDevice));
        } else {
            uint64_t mismatches = 0u;
            float max_abs = 0.0f;
            CHECK(bf16_difference(
                      reference, output, output_count,
                      &mismatches, &max_abs),
                  "Q8 tile comparison");
            CHECK(mismatches == 0u,
                  "Q8 tile changed exact output");
        }
        printf("    tile=%u mean=%.3f ms\n", tile, mean_ms);
    }

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    hipLaunchKernelGGL(
        dequant_q8_bf16, dim3(blocks_for(weight_count)),
        dim3(K3_THREADS), 0, 0,
        (hip_bfloat16 *)dequantized,
        (const int8_t *)quantized,
        (const float *)scales, weight_count);
    float dequant_ms = 0.0f;
    CHECK(elapsed_events(start, stop, 1u, &dequant_ms),
          "Q8 dequant timing");
    k3_rocm_blas_context *blas = NULL;
    CHECK(k3_rocm_blas_context_create(&blas),
          "BF16 BLAS context");
    CHECK(k3_rocm_blas_bf16_gemm_bf16(
              blas, output, dequantized, input,
              K3_STATIC_TOKENS, K3_STATIC_ROWS,
              K3_STATIC_COLUMNS, NULL),
          "dequantized BF16 BLAS warmup");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t step = 0u; step < K3_Q8_STEPS; step++) {
        CHECK(k3_rocm_blas_bf16_gemm_bf16(
                  blas, output, dequantized, input,
                  K3_STATIC_TOKENS, K3_STATIC_ROWS,
                  K3_STATIC_COLUMNS, NULL),
              "dequantized BF16 BLAS timed launch");
    }
    float blas_ms = 0.0f;
    CHECK(elapsed_events(
              start, stop, K3_Q8_STEPS, &blas_ms),
          "dequantized BF16 BLAS timing");
    uint64_t mismatches = 0u;
    float max_abs = 0.0f;
    CHECK(bf16_difference(
              reference, output, output_count,
              &mismatches, &max_abs),
          "Q8 versus dequantized BLAS comparison");
    CHECK(isfinite(max_abs) && max_abs <= 0.25f,
          "dequantized BF16 BLAS error");
    printf("    dequant=%.3f ms hipBLAS=%.3f ms "
           "combined=%.3f ms max_abs=%.6f\n",
           dequant_ms, blas_ms, dequant_ms + blas_ms, max_abs);
    k3_rocm_blas_context_destroy(blas);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipFree(dequantized));
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(reference));
    HIP_CHECK(hipFree(input));
    HIP_CHECK(hipFree(scales));
    HIP_CHECK(hipFree(quantized));
    return true;
}

static bool benchmark_router(void) {
    const uint64_t weight_count =
        (uint64_t)K3_ROUTER_ROWS * K3_ROUTER_COLUMNS;
    const uint64_t input_count =
        (uint64_t)K3_ROUTER_TOKENS * K3_ROUTER_COLUMNS;
    const uint64_t output_count =
        (uint64_t)K3_ROUTER_TOKENS * K3_ROUTER_ROWS;
    void *weights = NULL;
    void *input = NULL;
    void *reference = NULL;
    void *output = NULL;
    HIP_CHECK(hipMalloc(
        &weights, weight_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &input, input_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &reference, output_count * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &output, output_count * sizeof(float)));
    hipLaunchKernelGGL(
        fill_bf16, dim3(blocks_for(weight_count)),
        dim3(K3_THREADS), 0, 0,
        (hip_bfloat16 *)weights, weight_count,
        UINT32_C(0x128a5f11));
    hipLaunchKernelGGL(
        fill_bf16, dim3(blocks_for(input_count)),
        dim3(K3_THREADS), 0, 0,
        (hip_bfloat16 *)input, input_count,
        UINT32_C(0x80930de5));
    HIP_CHECK(hipDeviceSynchronize());

    printf("  BF16 router: tokens=%u rows=%u cols=%u\n",
           K3_ROUTER_TOKENS, K3_ROUTER_ROWS, K3_ROUTER_COLUMNS);
    for (uint32_t index = 0u;
         index < sizeof(k_tiles) / sizeof(k_tiles[0]); index++) {
        const uint32_t tile = k_tiles[index];
        CHECK(k3_rocm_bf16_gemm_tiled_f32(
                  output, weights, input,
                  K3_ROUTER_TOKENS, K3_ROUTER_ROWS,
                  K3_ROUTER_COLUMNS, tile, NULL),
              "BF16 router warmup");
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t start;
        hipEvent_t stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start, NULL));
        for (uint32_t step = 0u; step < K3_ROUTER_STEPS; step++) {
            CHECK(k3_rocm_bf16_gemm_tiled_f32(
                      output, weights, input,
                      K3_ROUTER_TOKENS, K3_ROUTER_ROWS,
                      K3_ROUTER_COLUMNS, tile, NULL),
                  "BF16 router timed launch");
        }
        float mean_ms = 0.0f;
        CHECK(elapsed_events(
                  start, stop, K3_ROUTER_STEPS, &mean_ms),
              "BF16 router timing");
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipEventDestroy(start));
        if (tile == 16u) {
            HIP_CHECK(hipMemcpy(
                reference, output,
                output_count * sizeof(float),
                hipMemcpyDeviceToDevice));
        } else {
            float max_abs = 0.0f;
            CHECK(f32_difference(
                      reference, output,
                      output_count, &max_abs),
                  "BF16 router tile comparison");
            CHECK(max_abs == 0.0f,
                  "BF16 router tile changed exact output");
        }
        printf("    tile=%u mean=%.3f ms\n", tile, mean_ms);
    }

    k3_rocm_blas_context *blas = NULL;
    CHECK(k3_rocm_blas_context_create(&blas),
          "router BLAS context");
    CHECK(k3_rocm_blas_bf16_gemm_f32(
              blas, output, weights, input,
              K3_ROUTER_TOKENS, K3_ROUTER_ROWS,
              K3_ROUTER_COLUMNS, NULL),
          "router BLAS warmup");
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (uint32_t step = 0u; step < K3_ROUTER_STEPS; step++) {
        CHECK(k3_rocm_blas_bf16_gemm_f32(
                  blas, output, weights, input,
                  K3_ROUTER_TOKENS, K3_ROUTER_ROWS,
                  K3_ROUTER_COLUMNS, NULL),
              "router BLAS timed launch");
    }
    float blas_ms = 0.0f;
    CHECK(elapsed_events(
              start, stop, K3_ROUTER_STEPS, &blas_ms),
          "router BLAS timing");
    float max_abs = 0.0f;
    CHECK(f32_difference(
              reference, output,
              output_count, &max_abs),
          "router BLAS comparison");
    CHECK(isfinite(max_abs) && max_abs <= 0.05f,
          "router BLAS error");
    printf("    hipBLAS mean=%.3f ms max_abs=%.6f\n",
           blas_ms, max_abs);
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    k3_rocm_blas_context_destroy(blas);
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(reference));
    HIP_CHECK(hipFree(input));
    HIP_CHECK(hipFree(weights));
    return true;
}

static bool benchmark_expert_shape(
        const char *label,
        uint32_t rows,
        uint32_t columns) {
    const uint64_t packed_count =
        (uint64_t)rows * columns / 2u;
    const uint64_t scale_count =
        (uint64_t)rows * columns / 32u;
    const uint64_t input_count =
        (uint64_t)K3_EXPERT_MAX_TOKENS * columns;
    const uint64_t output_count =
        (uint64_t)K3_EXPERT_MAX_TOKENS * rows;
    void *packed = NULL;
    void *scales = NULL;
    void *input = NULL;
    void *reference = NULL;
    void *output = NULL;
    HIP_CHECK(hipMalloc(&packed, packed_count));
    HIP_CHECK(hipMalloc(&scales, scale_count));
    HIP_CHECK(hipMalloc(
        &input, input_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &reference, output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &output, output_count * sizeof(hip_bfloat16)));
    hipLaunchKernelGGL(
        fill_mxfp4,
        dim3(blocks_for(
            packed_count > scale_count ?
                packed_count : scale_count)),
        dim3(K3_THREADS), 0, 0,
        (uint8_t *)packed, packed_count,
        (uint8_t *)scales, scale_count);
    hipLaunchKernelGGL(
        fill_bf16, dim3(blocks_for(input_count)),
        dim3(K3_THREADS), 0, 0,
        (hip_bfloat16 *)input, input_count,
        UINT32_C(0x7c299423));
    HIP_CHECK(hipDeviceSynchronize());

    printf("  MXFP4 expert %s: rows=%u cols=%u\n",
           label, rows, columns);
    for (uint32_t count_index = 0u;
         count_index <
             sizeof(k_expert_counts) /
             sizeof(k_expert_counts[0]);
         count_index++) {
        const uint32_t tokens =
            k_expert_counts[count_index];
        printf("    selected=%u", tokens);
        for (uint32_t tile_index = 0u;
             tile_index <
                 sizeof(k_tiles) / sizeof(k_tiles[0]);
             tile_index++) {
            const uint32_t tile = k_tiles[tile_index];
            CHECK(k3_rocm_mxfp4_gemm_tiled_bf16(
                      output, packed, scales, input,
                      tokens, rows, columns, tile, NULL),
                  "MXFP4 expert warmup");
            HIP_CHECK(hipDeviceSynchronize());
            hipEvent_t start;
            hipEvent_t stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));
            HIP_CHECK(hipEventRecord(start, NULL));
            for (uint32_t step = 0u;
                 step < K3_EXPERT_STEPS; step++) {
                CHECK(k3_rocm_mxfp4_gemm_tiled_bf16(
                          output, packed, scales, input,
                          tokens, rows, columns, tile, NULL),
                      "MXFP4 expert timed launch");
            }
            float mean_ms = 0.0f;
            CHECK(elapsed_events(
                      start, stop, K3_EXPERT_STEPS, &mean_ms),
                  "MXFP4 expert timing");
            HIP_CHECK(hipEventDestroy(stop));
            HIP_CHECK(hipEventDestroy(start));
            const uint64_t active_output_count =
                (uint64_t)tokens * rows;
            if (tile == 16u) {
                HIP_CHECK(hipMemcpy(
                    reference, output,
                    active_output_count *
                        sizeof(hip_bfloat16),
                    hipMemcpyDeviceToDevice));
            } else {
                uint64_t mismatches = 0u;
                float max_abs = 0.0f;
                CHECK(bf16_difference(
                          reference, output,
                          active_output_count,
                          &mismatches, &max_abs),
                      "MXFP4 expert tile comparison");
                CHECK(mismatches == 0u,
                      "MXFP4 tile changed exact output");
            }
            printf(" tile%u=%.3f", tile, mean_ms);
        }
        printf(" ms\n");
    }
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(reference));
    HIP_CHECK(hipFree(input));
    HIP_CHECK(hipFree(scales));
    HIP_CHECK(hipFree(packed));
    return true;
}

int main(void) {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess ||
        device_count == 0) {
        printf("K3 prefill GEMM shapes: SKIP (no ROCm device)\n");
        return 0;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipDeviceProp_t properties;
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        return 1;
    }
    printf("K3 prefill GEMM shapes on %s (%s)\n",
           properties.name, properties.gcnArchName);
    probe_mxfp4_hipblaslt();
    if (!benchmark_q8() ||
        !benchmark_router() ||
        !benchmark_expert_shape(
            "gate/up",
            K3_EXPERT_GATE_ROWS,
            K3_EXPERT_GATE_COLUMNS) ||
        !benchmark_expert_shape(
            "down",
            K3_EXPERT_DOWN_ROWS,
            K3_EXPERT_DOWN_COLUMNS)) {
        return 1;
    }
    printf("K3 prefill GEMM shapes: PASS\n");
    return 0;
}
