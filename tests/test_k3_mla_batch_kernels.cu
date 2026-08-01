#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                      \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                     \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    HEADS = 96,
    Q_DIM = 192,
    LATENT = 512,
    CACHE_DIM = 576,
    VALUE_DIM = 128,
    KV_HEAD_STRIDE = 256,
    QUERIES = 32,
    THREADS = 256,
    REPEATS = 20,
};

__global__ static void fill_bf16(hip_bfloat16 *output,
                                 uint64_t count,
                                 uint32_t salt) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t bits = (uint32_t)index ^ salt;
    bits ^= bits << 13;
    bits ^= bits >> 17;
    bits ^= bits << 5;
    const float value =
        ((float)(bits & UINT32_C(0xffff)) / 32768.0f - 1.0f) * 0.08f;
    output[index] = hip_bfloat16(value);
}

static bool launch_absorb_loop(void *output,
                               const void *q,
                               const void *packed) {
    const uint64_t q_stride = (uint64_t)HEADS * Q_DIM;
    const uint64_t output_stride = (uint64_t)HEADS * CACHE_DIM;
    for (uint32_t query = 0; query < QUERIES; query++) {
        if (!k3_rocm_mla_absorb_q_bf16(
                (hip_bfloat16 *)output + query * output_stride,
                (const hip_bfloat16 *)q + query * q_stride,
                packed, HEADS, NULL)) {
            return false;
        }
    }
    return true;
}

static bool launch_decompress_loop(void *output,
                                   const void *latent,
                                   const void *kv_b) {
    const uint64_t latent_stride = (uint64_t)HEADS * LATENT;
    const uint64_t output_stride = (uint64_t)HEADS * VALUE_DIM;
    for (uint32_t query = 0; query < QUERIES; query++) {
        if (!k3_rocm_mla_decompress_v_bf16(
                (hip_bfloat16 *)output + query * output_stride,
                (const hip_bfloat16 *)latent + query * latent_stride,
                kv_b, HEADS, NULL)) {
            return false;
        }
    }
    return true;
}

static float time_absorb(bool batched,
                         void *output,
                         const void *q,
                         const void *packed) {
    hipEvent_t start;
    hipEvent_t stop;
    if (hipEventCreate(&start) != hipSuccess ||
        hipEventCreate(&stop) != hipSuccess) return -1.0f;
    bool ok = batched ?
        k3_rocm_mla_absorb_q_batch_bf16(
            output, q, packed, HEADS, QUERIES, NULL) :
        launch_absorb_loop(output, q, packed);
    if (!ok || hipDeviceSynchronize() != hipSuccess ||
        hipEventRecord(start, NULL) != hipSuccess) return -1.0f;
    for (uint32_t repeat = 0; repeat < REPEATS; repeat++) {
        ok = batched ?
            k3_rocm_mla_absorb_q_batch_bf16(
                output, q, packed, HEADS, QUERIES, NULL) :
            launch_absorb_loop(output, q, packed);
        if (!ok) return -1.0f;
    }
    if (hipEventRecord(stop, NULL) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) return -1.0f;
    float milliseconds = 0.0f;
    if (hipEventElapsedTime(&milliseconds, start, stop) != hipSuccess) {
        return -1.0f;
    }
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    return milliseconds / REPEATS;
}

static float time_decompress(bool batched,
                             void *output,
                             const void *latent,
                             const void *kv_b) {
    hipEvent_t start;
    hipEvent_t stop;
    if (hipEventCreate(&start) != hipSuccess ||
        hipEventCreate(&stop) != hipSuccess) return -1.0f;
    bool ok = batched ?
        k3_rocm_mla_decompress_v_batch_bf16(
            output, latent, kv_b, HEADS, QUERIES, NULL) :
        launch_decompress_loop(output, latent, kv_b);
    if (!ok || hipDeviceSynchronize() != hipSuccess ||
        hipEventRecord(start, NULL) != hipSuccess) return -1.0f;
    for (uint32_t repeat = 0; repeat < REPEATS; repeat++) {
        ok = batched ?
            k3_rocm_mla_decompress_v_batch_bf16(
                output, latent, kv_b, HEADS, QUERIES, NULL) :
            launch_decompress_loop(output, latent, kv_b);
        if (!ok) return -1.0f;
    }
    if (hipEventRecord(stop, NULL) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) return -1.0f;
    float milliseconds = 0.0f;
    if (hipEventElapsedTime(&milliseconds, start, stop) != hipSuccess) {
        return -1.0f;
    }
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    return milliseconds / REPEATS;
}

int main(void) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 MLA batch kernels: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 MLA batch kernels on %s (%s)\n",
           properties.name, properties.gcnArchName);

    const size_t q_count = (size_t)QUERIES * HEADS * Q_DIM;
    const size_t absorbed_count =
        (size_t)QUERIES * HEADS * CACHE_DIM;
    const size_t latent_count = (size_t)QUERIES * HEADS * LATENT;
    const size_t output_count =
        (size_t)QUERIES * HEADS * VALUE_DIM;
    const size_t kv_count =
        (size_t)HEADS * KV_HEAD_STRIDE * LATENT;
    const size_t packed_count = (size_t)HEADS * LATENT * 128u;
    void *d_q = NULL;
    void *d_loop_absorbed = NULL;
    void *d_batch_absorbed = NULL;
    void *d_latent = NULL;
    void *d_loop_output = NULL;
    void *d_batch_output = NULL;
    void *d_kv_b = NULL;
    void *d_packed = NULL;
    HIP_CHECK(hipMalloc(&d_q, q_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_loop_absorbed,
                        absorbed_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_batch_absorbed,
                        absorbed_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_latent,
                        latent_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_loop_output,
                        output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_batch_output,
                        output_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_kv_b, kv_count * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_packed,
                        packed_count * sizeof(hip_bfloat16)));
    hipLaunchKernelGGL(fill_bf16,
                       dim3((q_count + THREADS - 1u) / THREADS),
                       dim3(THREADS), 0, NULL,
                       (hip_bfloat16 *)d_q, q_count,
                       UINT32_C(0x71626174));
    hipLaunchKernelGGL(fill_bf16,
                       dim3((latent_count + THREADS - 1u) / THREADS),
                       dim3(THREADS), 0, NULL,
                       (hip_bfloat16 *)d_latent, latent_count,
                       UINT32_C(0x6c626174));
    hipLaunchKernelGGL(fill_bf16,
                       dim3((kv_count + THREADS - 1u) / THREADS),
                       dim3(THREADS), 0, NULL,
                       (hip_bfloat16 *)d_kv_b, kv_count,
                       UINT32_C(0x77626174));
    HIP_CHECK(hipGetLastError());
    CHECK(k3_rocm_mla_pack_k_weight_bf16(
              d_packed, d_kv_b, HEADS, NULL),
          "batch fixture key pack");
    CHECK(launch_absorb_loop(d_loop_absorbed, d_q, d_packed),
          "looped query absorption");
    CHECK(k3_rocm_mla_absorb_q_batch_bf16(
              d_batch_absorbed, d_q, d_packed,
              HEADS, QUERIES, NULL),
          "batched query absorption");
    CHECK(launch_decompress_loop(d_loop_output, d_latent, d_kv_b),
          "looped value decompression");
    CHECK(k3_rocm_mla_decompress_v_batch_bf16(
              d_batch_output, d_latent, d_kv_b,
              HEADS, QUERIES, NULL),
          "batched value decompression");
    HIP_CHECK(hipDeviceSynchronize());

    hip_bfloat16 *looped = (hip_bfloat16 *)malloc(
        absorbed_count * sizeof(hip_bfloat16));
    hip_bfloat16 *batched = (hip_bfloat16 *)malloc(
        absorbed_count * sizeof(hip_bfloat16));
    CHECK(looped && batched, "host comparison allocation");
    HIP_CHECK(hipMemcpy(looped, d_loop_absorbed,
                        absorbed_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(batched, d_batch_absorbed,
                        absorbed_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    CHECK(memcmp(looped, batched,
                 absorbed_count * sizeof(hip_bfloat16)) == 0,
          "batched query absorption is not bitwise exact");
    HIP_CHECK(hipMemcpy(looped, d_loop_output,
                        output_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(batched, d_batch_output,
                        output_count * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));
    CHECK(memcmp(looped, batched,
                 output_count * sizeof(hip_bfloat16)) == 0,
          "batched value decompression is not bitwise exact");

    const float absorb_loop_ms = time_absorb(
        false, d_loop_absorbed, d_q, d_packed);
    const float absorb_batch_ms = time_absorb(
        true, d_batch_absorbed, d_q, d_packed);
    const float decompress_loop_ms = time_decompress(
        false, d_loop_output, d_latent, d_kv_b);
    const float decompress_batch_ms = time_decompress(
        true, d_batch_output, d_latent, d_kv_b);
    CHECK(absorb_loop_ms > 0.0f && absorb_batch_ms > 0.0f &&
          decompress_loop_ms > 0.0f && decompress_batch_ms > 0.0f,
          "batch primitive timing");
    printf("  absorb 32 queries: loop %.3f ms, batch %.3f ms, %.3fx\n",
           absorb_loop_ms, absorb_batch_ms,
           absorb_loop_ms / absorb_batch_ms);
    printf("  decompress 32 queries: loop %.3f ms, batch %.3f ms, %.3fx\n",
           decompress_loop_ms, decompress_batch_ms,
           decompress_loop_ms / decompress_batch_ms);
    printf("  exact outputs: absorb %zu, decompress %zu BF16 values\n",
           absorbed_count, output_count);

    free(batched);
    free(looped);
    HIP_CHECK(hipFree(d_packed));
    HIP_CHECK(hipFree(d_kv_b));
    HIP_CHECK(hipFree(d_batch_output));
    HIP_CHECK(hipFree(d_loop_output));
    HIP_CHECK(hipFree(d_latent));
    HIP_CHECK(hipFree(d_batch_absorbed));
    HIP_CHECK(hipFree(d_loop_absorbed));
    HIP_CHECK(hipFree(d_q));
    printf("K3 MLA batch kernels: PASS\n");
    return 0;
}
