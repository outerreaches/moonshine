// Does k3_rocm_mxfp4_gemm_bf16 match the per-vector GEMV loop at the widths
// expert-major execution actually uses?
//
// The existing equality assertion in test_k3_prefill_ops.cu uses VECTORS = 2,
// which fits inside a single K3_ROCM_BATCH_TILE (16). The header promises the
// batch variants "preserve the GEMV reduction order independently for every
// row", and expert-major relies on that at up to `prefill_chunk` vectors, so
// the promise needs checking past one tile and on the production shapes.
#include "k3_rocm_ops.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(expr)                                                       \
    do {                                                                      \
        const hipError_t status = (expr);                                     \
        if (status != hipSuccess) {                                           \
            std::fprintf(stderr, "%s: %s\n", #expr,                           \
                         hipGetErrorString(status));                          \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main(void)
{
    /* Production expert shapes: gate/up are [2048, 4096]. */
    const uint32_t rows = 2048u;
    const uint32_t columns = 4096u;
    const uint32_t widths[] = {1u, 2u, 8u, 15u, 16u, 17u, 32u, 64u, 128u};
    const uint32_t max_vectors = 128u;

    const size_t packed_bytes = (size_t)rows * (columns / 2u);
    const size_t scale_bytes = (size_t)rows * (columns / 32u);
    std::vector<uint8_t> packed(packed_bytes), scales(scale_bytes);
    std::vector<uint16_t> inputs((size_t)max_vectors * columns);
    uint64_t state = 0x9E3779B97F4A7C15ull;
    auto next = [&state]() {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return state;
    };
    for (size_t i = 0; i < packed_bytes; i++) packed[i] = (uint8_t)(next() >> 24);
    /* E8M0 exponents near 1.0 so products stay in a sane range. */
    for (size_t i = 0; i < scale_bytes; i++)
        scales[i] = (uint8_t)(120u + (next() >> 59) % 12u);
    for (size_t i = 0; i < inputs.size(); i++)
        inputs[i] = (uint16_t)(0x3C00u | ((next() >> 50) & 0x03FFu));

    void *d_packed = nullptr, *d_scales = nullptr, *d_inputs = nullptr;
    void *d_batch = nullptr, *d_loop = nullptr;
    HIP_CHECK(hipMalloc(&d_packed, packed_bytes));
    HIP_CHECK(hipMalloc(&d_scales, scale_bytes));
    HIP_CHECK(hipMalloc(&d_inputs, inputs.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_batch, (size_t)max_vectors * rows * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_loop, (size_t)max_vectors * rows * sizeof(uint16_t)));
    HIP_CHECK(hipMemcpy(d_packed, packed.data(), packed_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scales, scales.data(), scale_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_inputs, inputs.data(),
                        inputs.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice));

    int failures = 0;
    for (uint32_t width : widths) {
        const size_t out_bytes = (size_t)width * rows * sizeof(uint16_t);
        HIP_CHECK(hipMemset(d_batch, 0, out_bytes));
        HIP_CHECK(hipMemset(d_loop, 0, out_bytes));
        if (!k3_rocm_mxfp4_gemm_bf16(d_batch, d_packed, d_scales, d_inputs,
                                     width, rows, columns, nullptr)) {
            std::fprintf(stderr, "gemm failed at width %u\n", width);
            return 1;
        }
        for (uint32_t v = 0; v < width; v++) {
            if (!k3_rocm_mxfp4_gemv_bf16(
                    (uint8_t *)d_loop + (size_t)v * rows * sizeof(uint16_t),
                    d_packed, d_scales,
                    (const uint8_t *)d_inputs +
                        (size_t)v * columns * sizeof(uint16_t),
                    rows, columns, nullptr)) {
                std::fprintf(stderr, "gemv failed at width %u\n", width);
                return 1;
            }
        }
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<uint8_t> batch(out_bytes), loop(out_bytes);
        HIP_CHECK(hipMemcpy(batch.data(), d_batch, out_bytes,
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(loop.data(), d_loop, out_bytes,
                            hipMemcpyDeviceToHost));
        size_t differing = 0;
        for (size_t i = 0; i < out_bytes; i += 2) {
            if (std::memcmp(batch.data() + i, loop.data() + i, 2) != 0) {
                differing++;
            }
        }
        const size_t total = out_bytes / 2u;
        std::printf("  width %3u: %s (%zu/%zu elements differ)\n", width,
                    differing == 0 ? "IDENTICAL" : "DIFFER", differing, total);
        if (differing != 0) failures++;
    }
    std::printf("%s\n", failures == 0
        ? "PASS: batch GEMM matches the GEMV loop at every tested width"
        : "FAIL: the batch/GEMV equality does not hold at all widths");
    return failures == 0 ? 0 : 1;
}
