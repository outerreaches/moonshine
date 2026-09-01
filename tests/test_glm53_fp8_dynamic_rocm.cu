#include "glm53_rocm_ops.h"
#include "glm53_fp8_dynamic.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)
#define HIP_CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, \
                 #c, hipGetErrorString(e_)); return false; } } while (0)

__global__ static void poison_launch_status_kernel(void) {}

static bool same_float_bits(float a, float b) {
    uint32_t x, y;
    std::memcpy(&x, &a, sizeof(x));
    std::memcpy(&y, &b, sizeof(y));
    return x == y;
}

static bool test_dynamic_quantize_and_gemv(void) {
    constexpr uint32_t rows = 257u;
    constexpr uint32_t columns = 512u;
    constexpr size_t groups = columns / 128u;
    std::vector<float> input(columns, 0.0f);
    const float floor_scale = GLM53_FP8_DYNAMIC_SCALE_FLOOR;

    /* A floor-scaled group: signed zero, exact midpoint ties, subnormals. */
    input[0] = 0.0f;
    input[1] = -0.0f;
    input[2] = floor_scale * (1.0f / 1024.0f);  /* code 0/1 tie -> 0 */
    input[3] = floor_scale * (3.0f / 1024.0f);  /* code 1/2 tie -> 2 */
    input[4] = -floor_scale * (3.0f / 1024.0f);
    input[5] = floor_scale * (1.0f / 512.0f);
    input[127] = 0.0f; /* the group absmax remains below eps=1e-10 */

    /* Exact scale 1 makes normal ties and max codes easy to audit. */
    for (uint32_t c = 128u; c < 256u; ++c)
        input[c] = (float)((int)(c % 19u) - 9) * 0.125f;
    input[128] = 448.0f;
    input[129] = -448.0f;
    input[130] = 1.0625f; /* midpoint between codes 0x38 and 0x39 */
    input[131] = -1.1875f;

    /* The absmax values exercise dynamic endpoint clamp/saturation. */
    for (uint32_t c = 256u; c < 384u; ++c)
        input[c] = (float)((int)(c % 31u) - 15) * 7.25f;
    input[256] = 1000.0f;
    input[257] = -1000.0f;
    for (uint32_t c = 384u; c < columns; ++c)
        input[c] = std::sin((float)c * 0.17f) * 37.0f;
    input[511] = -731.0f;

    std::vector<uint8_t> want_q(columns);
    std::vector<float> want_xs(groups);
    CHECK(glm53_fp8_dynamic_quantize_f32(
        want_q.data(), want_q.size(), columns,
        want_xs.data(), want_xs.size(), groups,
        input.data(), input.size(), columns, 1u, columns) ==
        GLM53_FP8_DYNAMIC_OK);
    CHECK(want_q[0] == 0x00u && want_q[1] == 0x80u);
    CHECK(want_q[2] == 0x00u && want_q[3] == 0x02u && want_q[4] == 0x82u);
    CHECK(want_q[128] == 0x7eu && want_q[129] == 0xfeu);
    CHECK(same_float_bits(want_xs[0], floor_scale));

    std::vector<uint8_t> weights((size_t)rows * columns);
    std::vector<float> weight_scales(((rows + 127u) / 128u) * groups);
    for (size_t i = 0; i < weight_scales.size(); ++i)
        weight_scales[i] = 0.00390625f * (float)(i + 1u);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < columns; ++c) {
            float value = (float)((int)((r * 17u + c * 29u) % 65u) - 32) *
                          0.25f;
            uint8_t code = 0;
            CHECK(glm53_fp8_e4m3fn_encode(value, &code) ==
                  GLM53_FP8_DYNAMIC_OK);
            weights[(size_t)r * columns + c] = code;
        }
    }

    float *d_input = nullptr, *d_xs = nullptr, *d_ws = nullptr,
          *d_output = nullptr;
    uint8_t *d_q = nullptr, *d_w = nullptr;
    hipStream_t stream = nullptr;
    HIP_CHECK(hipMalloc(&d_input, sizeof(float) * input.size()));
    HIP_CHECK(hipMalloc(&d_q, want_q.size()));
    HIP_CHECK(hipMalloc(&d_xs, sizeof(float) * want_xs.size()));
    HIP_CHECK(hipMalloc(&d_w, weights.size()));
    HIP_CHECK(hipMalloc(&d_ws, sizeof(float) * weight_scales.size()));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float) * rows));
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyAsync(d_input, input.data(), sizeof(float)*input.size(),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_w, weights.data(), weights.size(),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_ws, weight_scales.data(),
                             sizeof(float)*weight_scales.size(),
                             hipMemcpyHostToDevice, stream));
    hipLaunchKernelGGL(poison_launch_status_kernel, dim3(0u), dim3(1u), 0,
                       stream);
    CHECK(hipPeekAtLastError() != hipSuccess);
    CHECK(glm53_rocm_fp8_dynamic_gemv_f32(
        d_output, rows, d_q, columns, d_xs, groups,
        d_w, weights.size(), d_ws, weight_scales.size(),
        d_input, columns, rows, columns, stream));
    std::vector<uint8_t> got_q(columns);
    std::vector<float> got_xs(groups), got(rows);
    HIP_CHECK(hipMemcpyAsync(got_q.data(), d_q, got_q.size(),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(got_xs.data(), d_xs,
                             sizeof(float)*got_xs.size(),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(got.data(), d_output, sizeof(float)*got.size(),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    CHECK(got_q == want_q);
    for (size_t i = 0; i < groups; ++i)
        CHECK(same_float_bits(got_xs[i], want_xs[i]));

    double worst_ratio = 0.0;
    for (uint32_t r = 0; r < rows; ++r) {
        float want = 0.0f;
        double magnitude = 0.0;
        for (uint32_t c = 0; c < columns; ++c) {
            float w, x;
            CHECK(glm53_fp8_dynamic_decode(weights[(size_t)r*columns+c], &w)
                  == GLM53_FP8_DYNAMIC_OK);
            CHECK(glm53_fp8_dynamic_decode(want_q[c], &x)
                  == GLM53_FP8_DYNAMIC_OK);
            const float term =
                (w * weight_scales[(r/128u)*groups+c/128u]) *
                (x * want_xs[c/128u]);
            want += term;
            magnitude += std::fabs((double)term);
        }
        const double error = std::fabs((double)got[r] - (double)want);
        const double tolerance = 3.0e-6 * magnitude + 2.0e-5;
        worst_ratio = std::fmax(worst_ratio, error / tolerance);
        CHECK(error <= tolerance);
    }
    std::printf("dynamic W8A8 GEMV max_tolerance_ratio=%.6g\n", worst_ratio);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_output)); HIP_CHECK(hipFree(d_ws));
    HIP_CHECK(hipFree(d_w)); HIP_CHECK(hipFree(d_xs));
    HIP_CHECK(hipFree(d_q)); HIP_CHECK(hipFree(d_input));
    return true;
}

static bool test_rejected_geometry_and_buffers(void) {
    void *a = (void *)(uintptr_t)0x100000u;
    void *b = (void *)(uintptr_t)0x200000u;
    void *c = (void *)(uintptr_t)0x300000u;
    void *d = (void *)(uintptr_t)0x400000u;
    void *e = (void *)(uintptr_t)0x500000u;
    void *f = (void *)(uintptr_t)0x600000u;
#define CALL(o,oc,q,qc,xs,xsc,w,wc,ws,wsc,x,xc,r,col) \
    glm53_rocm_fp8_dynamic_gemv_f32(o,oc,q,qc,xs,xsc,w,wc,ws,wsc, \
                                     x,xc,r,col,nullptr)
    CHECK(!CALL(nullptr,128,b,128,c,1,d,16384,e,1,f,128,128,128));
    CHECK(!CALL(a,128,b,128,c,1,d,16384,e,1,f,128,0,128));
    CHECK(!CALL(a,128,b,128,c,1,d,16384,e,1,f,128,128,0));
    CHECK(!CALL(a,128,b,128,c,1,d,16384,e,1,f,128,UINT32_MAX,128));
    CHECK(!CALL(a,128,b,129,c,2,d,16512,e,2,f,129,128,129));
    CHECK(!CALL(a,128,b,128,c,1,d,16383,e,1,f,128,128,128));
    CHECK(!CALL(a,127,b,128,c,1,d,16384,e,1,f,128,128,128));
    CHECK(!CALL(a,128,a,128,c,1,d,16384,e,1,f,128,128,128));
    CHECK(!CALL((void *)(UINTPTR_MAX-1u),128,b,128,c,1,d,16384,e,1,f,128,
                128,128));
#undef CALL
    return true;
}

int main(void) {
    int device = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&device) != hipSuccess ||
        hipGetDeviceProperties(&prop, device) != hipSuccess) {
        std::fprintf(stderr, "FAIL: no HIP device\n"); return 1;
    }
    if (std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0) {
        std::fprintf(stderr, "FAIL: test requires gfx1151, got %s\n",
                     prop.gcnArchName); return 1;
    }
    if (!test_dynamic_quantize_and_gemv() ||
        !test_rejected_geometry_and_buffers()) return 1;
    std::printf("glm53 dynamic ROCm tests passed on %s\n", prop.gcnArchName);
    return 0;
}
