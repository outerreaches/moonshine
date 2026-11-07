#include "glm53_rocm_ops.h"
#include "glm53_fp8_oracle.h"
#include "k3_rocm_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)
#define HIP_CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, \
                 #c, hipGetErrorString(e_)); return false; } } while (0)

static uint16_t bf16_bits(hip_bfloat16 v) {
    uint16_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static float decode(uint8_t code) {
    float value = 0.0f;
    glm53_fp8_oracle_status status = glm53_fp8_e4m3fn_decode(code, &value);
    if (code == UINT8_C(0x7f) || code == UINT8_C(0xff)) {
        CHECK(status == GLM53_FP8_ORACLE_NAN_ENCODING);
    } else {
        CHECK(status == GLM53_FP8_ORACLE_OK);
    }
    return value;
}

static bool test_exhaustive_dequant(void) {
    std::vector<uint8_t> host_codes;
    for (unsigned i = 0; i < 256; ++i)
        if (i != 0x7f && i != 0xff) host_codes.push_back((uint8_t)i);
    const float one = 1.0f;
    uint8_t *codes = nullptr;
    float *scales = nullptr;
    hip_bfloat16 *output = nullptr;
    HIP_CHECK(hipMalloc(&codes, host_codes.size()));
    HIP_CHECK(hipMalloc(&scales, sizeof(float) * 2u));
    HIP_CHECK(hipMalloc(&output, sizeof(*output) * host_codes.size()));
    HIP_CHECK(hipMemcpy(codes, host_codes.data(), host_codes.size(),
                        hipMemcpyHostToDevice));
    float scale_values[2] = {one, one};
    HIP_CHECK(hipMemcpy(scales, scale_values, sizeof(scale_values),
                        hipMemcpyHostToDevice));
    CHECK(glm53_rocm_fp8_dequantize_bf16(output, codes, scales,
                                         1u, (uint32_t)host_codes.size(),
                                         nullptr));
    std::vector<hip_bfloat16> got(host_codes.size());
    HIP_CHECK(hipMemcpy(got.data(), output, sizeof(*output) * got.size(),
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
        hip_bfloat16 want(decode(host_codes[i]));
        CHECK(bf16_bits(got[i]) == bf16_bits(want));
    }
    HIP_CHECK(hipFree(output)); HIP_CHECK(hipFree(scales));
    HIP_CHECK(hipFree(codes));
    return true;
}

static bool test_nan_dequant(void) {
    const uint8_t host_codes[2] = {UINT8_C(0x7f), UINT8_C(0xff)};
    const float one = 1.0f;
    uint8_t *codes = nullptr; float *scales = nullptr;
    hip_bfloat16 *output = nullptr;
    HIP_CHECK(hipMalloc(&codes, sizeof(host_codes)));
    HIP_CHECK(hipMalloc(&scales, sizeof(one)));
    HIP_CHECK(hipMalloc(&output, sizeof(*output) * 2u));
    HIP_CHECK(hipMemcpy(codes, host_codes, sizeof(host_codes), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(scales, &one, sizeof(one), hipMemcpyHostToDevice));
    CHECK(glm53_rocm_fp8_dequantize_bf16(output, codes, scales, 1u, 2u, nullptr));
    hip_bfloat16 got[2];
    HIP_CHECK(hipMemcpy(got, output, sizeof(got), hipMemcpyDeviceToHost));
    CHECK(std::isnan((float)got[0])); CHECK(std::isnan((float)got[1]));
    HIP_CHECK(hipFree(output)); HIP_CHECK(hipFree(scales)); HIP_CHECK(hipFree(codes));
    return true;
}

struct Fixture {
    static constexpr uint32_t rows = 129u, cols = 257u;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    std::vector<hip_bfloat16> input;
    Fixture() : codes((size_t)rows * cols), scales{0.25f, -0.5f, 1.25f,
                                                     2.0f, 0.75f, -1.5f},
                input(cols) {
        const uint8_t patterns[] = {0x00,0x80,0x01,0x81,0x08,0x88,0x38,0xb8,
                                    0x7e,0xfe,0x77,0xf7,0x20,0xa0,0x5d,0xdd};
        for (size_t i = 0; i < codes.size(); ++i)
            codes[i] = patterns[(i * 13u + i / cols * 3u) %
                                (sizeof(patterns) / sizeof(patterns[0]))];
        for (uint32_t c = 0; c < cols; ++c) {
            float value = 0.125f * (float)((int)(c % 17u) - 8) +
                          0.03125f * (float)((int)(c % 5u) - 2);
            input[c] = hip_bfloat16(value);
        }
    }
};

static bool test_fused_gemv(void) {
    Fixture f;
    uint8_t *codes = nullptr; float *scales = nullptr; float *output = nullptr;
    hip_bfloat16 *input = nullptr; hipStream_t stream = nullptr;
    HIP_CHECK(hipMalloc(&codes, f.codes.size()));
    HIP_CHECK(hipMalloc(&scales, sizeof(float) * f.scales.size()));
    HIP_CHECK(hipMalloc(&input, sizeof(*input) * f.input.size()));
    HIP_CHECK(hipMalloc(&output, sizeof(*output) * Fixture::rows));
    HIP_CHECK(hipMemcpy(codes, f.codes.data(), f.codes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(scales, f.scales.data(), sizeof(float)*f.scales.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(input, f.input.data(), sizeof(*input)*f.input.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipStreamCreate(&stream));
    CHECK(glm53_rocm_fp8_gemv_f32(output, codes, scales, input,
                                  Fixture::rows, Fixture::cols, stream));
    std::vector<float> got(Fixture::rows);
    HIP_CHECK(hipMemcpyAsync(got.data(), output, sizeof(float)*got.size(),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    double max_abs = 0.0, max_bound_ratio = 0.0;
    for (uint32_t r = 0; r < Fixture::rows; ++r) {
        float want = 0.0f, magnitude = 0.0f;
        for (uint32_t c = 0; c < Fixture::cols; ++c) {
            float term = decode(f.codes[(size_t)r*Fixture::cols+c]) *
                         f.scales[(r/128u)*3u + c/128u] * (float)f.input[c];
            want += term; magnitude += std::fabs(term);
        }
        double error = std::fabs((double)got[r] - want);
        double tolerance = 2.0e-6 * (double)magnitude + 2.0e-5;
        max_abs = std::fmax(max_abs, error);
        max_bound_ratio = std::fmax(max_bound_ratio, error / tolerance);
        CHECK(error <= tolerance);
    }
    std::printf("fused GEMV max_abs=%.9g max_tolerance_ratio=%.6g\n",
                max_abs, max_bound_ratio);
    HIP_CHECK(hipStreamDestroy(stream)); HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(input)); HIP_CHECK(hipFree(scales)); HIP_CHECK(hipFree(codes));
    return true;
}

static bool test_prefill_baseline(void) {
    Fixture f;
    constexpr uint32_t tokens = 17u;
    std::vector<hip_bfloat16> inputs((size_t)tokens * Fixture::cols);
    for (uint32_t t = 0; t < tokens; ++t)
        for (uint32_t c = 0; c < Fixture::cols; ++c)
            inputs[(size_t)t*Fixture::cols+c] = hip_bfloat16(
                (float)((int)((t*7u+c*3u)%29u)-14) * 0.03125f);
    uint8_t *codes = nullptr; float *scales = nullptr; float *output = nullptr;
    hip_bfloat16 *weights = nullptr, *device_inputs = nullptr;
    HIP_CHECK(hipMalloc(&codes, f.codes.size()));
    HIP_CHECK(hipMalloc(&scales, sizeof(float)*f.scales.size()));
    HIP_CHECK(hipMalloc(&weights, sizeof(*weights)*f.codes.size()));
    HIP_CHECK(hipMalloc(&device_inputs, sizeof(*device_inputs)*inputs.size()));
    HIP_CHECK(hipMalloc(&output, sizeof(*output)*tokens*Fixture::rows));
    HIP_CHECK(hipMemcpy(codes, f.codes.data(), f.codes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(scales, f.scales.data(), sizeof(float)*f.scales.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_inputs, inputs.data(), sizeof(*device_inputs)*inputs.size(), hipMemcpyHostToDevice));
    CHECK(glm53_rocm_fp8_dequantize_bf16(weights, codes, scales,
                                         Fixture::rows, Fixture::cols, nullptr));
    k3_rocm_blas_context *context = nullptr;
    CHECK(k3_rocm_blas_context_create(&context));
    CHECK(k3_rocm_blas_bf16_gemm_f32(context, output, weights, device_inputs,
                                     tokens, Fixture::rows, Fixture::cols, nullptr));
    std::vector<float> got((size_t)tokens*Fixture::rows);
    std::vector<hip_bfloat16> rounded_weights(f.codes.size());
    HIP_CHECK(hipMemcpy(rounded_weights.data(), weights,
                        sizeof(*weights)*rounded_weights.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got.data(), output, sizeof(float)*got.size(), hipMemcpyDeviceToHost));
    double max_abs = 0.0, max_bound_ratio = 0.0;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t r = 0; r < Fixture::rows; ++r) {
            float want = 0.0f, magnitude = 0.0f;
            for (uint32_t c = 0; c < Fixture::cols; ++c) {
                float term = (float)rounded_weights[(size_t)r*Fixture::cols+c] *
                             (float)inputs[(size_t)token*Fixture::cols+c];
                want += term; magnitude += std::fabs(term);
            }
            double error = std::fabs((double)got[(size_t)token*Fixture::rows+r]-want);
            double tolerance = 3.0e-6*(double)magnitude + 3.0e-5;
            max_abs = std::fmax(max_abs, error);
            max_bound_ratio = std::fmax(max_bound_ratio, error/tolerance);
            CHECK(error <= tolerance);
        }
    }
    std::printf("dequant+hipBLAS prefill max_abs=%.9g max_tolerance_ratio=%.6g\n",
                max_abs, max_bound_ratio);
    k3_rocm_blas_context_destroy(context);
    HIP_CHECK(hipFree(output)); HIP_CHECK(hipFree(device_inputs));
    HIP_CHECK(hipFree(weights)); HIP_CHECK(hipFree(scales)); HIP_CHECK(hipFree(codes));
    return true;
}

static bool test_invalid_arguments(void) {
    void *p = (void *)(uintptr_t)0x1000;
    CHECK(!glm53_rocm_fp8_gemv_f32(nullptr,p,p,p,1,1,nullptr));
    CHECK(!glm53_rocm_fp8_gemv_f32(p,nullptr,p,p,1,1,nullptr));
    CHECK(!glm53_rocm_fp8_gemv_f32(p,p,nullptr,p,1,1,nullptr));
    CHECK(!glm53_rocm_fp8_gemv_f32(p,p,p,nullptr,1,1,nullptr));
    CHECK(!glm53_rocm_fp8_gemv_f32(p,p,p,p,0,1,nullptr));
    CHECK(!glm53_rocm_fp8_gemv_f32(p,p,p,p,1,0,nullptr));
    CHECK(!glm53_rocm_fp8_dequantize_bf16(nullptr,p,p,1,1,nullptr));
    CHECK(!glm53_rocm_fp8_dequantize_bf16(p,nullptr,p,1,1,nullptr));
    CHECK(!glm53_rocm_fp8_dequantize_bf16(p,p,nullptr,1,1,nullptr));
    CHECK(!glm53_rocm_fp8_dequantize_bf16(p,p,p,0,1,nullptr));
    CHECK(!glm53_rocm_fp8_dequantize_bf16(p,p,p,UINT32_MAX,UINT32_MAX,nullptr));
    return true;
}

int main(void) {
    int devices = 0;
    hipError_t status = hipGetDeviceCount(&devices);
    if (status != hipSuccess || devices == 0) {
        std::printf("SKIP: no HIP device (%s)\n", hipGetErrorString(status));
        return 0;
    }
    if (!test_invalid_arguments() || !test_exhaustive_dequant() ||
        !test_nan_dequant() || !test_fused_gemv() ||
        !test_prefill_baseline()) return 1;
    std::puts("PASS test_glm53_rocm_ops");
    return 0;
}
