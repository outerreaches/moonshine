#include "glm53_fp8_oracle.h"
#include "glm53_manifest.h"
#include "glm53_official_tensor.h"
#include "glm53_rocm_ops.h"
#include "k3_safetensors.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    goto done; } } while (0)
#define HIP_CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, \
                 #c, hipGetErrorString(e_)); goto done; } } while (0)

static const char *weight_name =
    "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight";
static const float first_anchor[8] = {
    0.16060075163841248f, -0.012741511687636375f,
   -0.24690984189510345f,  0.2515947222709656f,
   -0.5886113047599792f,  -0.20694327354431152f,
   -0.3180915415287018f, -0.4408954679965973f
};
static const float last_anchor[8] = {
   -0.45440906286239624f, 0.2125268578529358f,
    0.18755406141281128f, 0.13979649543762207f,
   -0.29989394545555115f, 0.4025675654411316f,
    0.5775325298309326f,  0.3584621250629425f
};

int main(int argc, char **argv) {
    constexpr uint32_t rows = 512u, cols = 4096u;
    char error[512] = {0};
    glm53_manifest manifest = {};
    k3_st_model model = {};
    glm53_official_fp8_matrix matrix = {};
    k3_st_read weight_read = {}, scale_read = {};
    std::vector<uint8_t> weights;
    std::vector<float> scales, x, cpu, golden, gpu, sum_abs;
    std::vector<hip_bfloat16> x_bf16;
    uint8_t *d_weight = nullptr;
    float *d_scale = nullptr, *d_output = nullptr;
    hip_bfloat16 *d_input = nullptr;
    int result = 1, devices = 0;
    std::FILE *reference = nullptr;
    hipDeviceProp_t properties = {};

    if (argc != 3) {
        std::fprintf(stderr, "usage: %s OFFICIAL_ROOT REFERENCE_F32\n", argv[0]);
        return 2;
    }
    CHECK(glm53_manifest_load(&manifest, argv[1], error, sizeof(error)));
    CHECK(k3_st_model_open_5digit_total(&model, argv[1], GLM53_SHARD_COUNT,
                                        error, sizeof(error)));
    CHECK(glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    CHECK(glm53_official_bind_fp8_matrix(&model, weight_name, rows, cols,
                                          &matrix, error, sizeof(error)));
    CHECK(matrix.weight->byte_length == (uint64_t)rows * cols);
    CHECK(matrix.scale_inv->byte_length ==
          (uint64_t)matrix.scale_rows * matrix.scale_columns * sizeof(float));
    CHECK(k3_st_read_span(&model, matrix.weight->shard,
                          matrix.weight->physical_offset,
                          matrix.weight->byte_length, 4096u,
                          &weight_read, error, sizeof(error)));
    CHECK(k3_st_read_span(&model, matrix.scale_inv->shard,
                          matrix.scale_inv->physical_offset,
                          matrix.scale_inv->byte_length, 4096u,
                          &scale_read, error, sizeof(error)));

    weights.assign(weight_read.data,
                   weight_read.data + matrix.weight->byte_length);
    scales.resize((size_t)matrix.scale_rows * matrix.scale_columns);
    std::memcpy(scales.data(), scale_read.data,
                scales.size() * sizeof(scales[0]));
    x.resize(cols); x_bf16.resize(cols);
    for (uint32_t c = 0; c < cols; ++c) {
        x[c] = (float)((int)((17u * c) % 31u) - 15) / 32.0f;
        x_bf16[c] = hip_bfloat16(x[c]);
        CHECK((float)x_bf16[c] == x[c]);
    }
    golden.resize(rows);
    reference = std::fopen(argv[2], "rb");
    CHECK(reference != nullptr);
    CHECK(std::fread(golden.data(), sizeof(float), rows, reference) == rows);
    CHECK(std::fgetc(reference) == EOF);
    CHECK(std::fclose(reference) == 0);
    reference = nullptr;
    cpu.assign(rows, 0.0f);
    CHECK(glm53_fp8_project_f32(
              cpu.data(), cpu.size(), 1u,
              weights.data(), weights.size(), cols,
              scales.data(), scales.size(), matrix.scale_columns,
              x.data(), x.size(), 1u, rows, cols) == GLM53_FP8_ORACLE_OK);
    for (uint32_t r = 0; r < rows; ++r)
        CHECK(std::fabs(cpu[r] - golden[r]) <= 2.0e-6f);
    for (uint32_t i = 0; i < 8u; ++i) {
        CHECK(std::fabs(golden[i] - first_anchor[i]) <= 2.0e-7f);
        CHECK(std::fabs(golden[rows - 8u + i] - last_anchor[i]) <= 2.0e-7f);
    }

    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0);
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    CHECK(std::strncmp(properties.gcnArchName, "gfx1151", 7u) == 0);
    HIP_CHECK(hipMalloc(&d_weight, weights.size()));
    HIP_CHECK(hipMalloc(&d_scale, scales.size() * sizeof(scales[0])));
    HIP_CHECK(hipMalloc(&d_input, x_bf16.size() * sizeof(x_bf16[0])));
    HIP_CHECK(hipMalloc(&d_output, rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_weight, weights.data(), weights.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scale, scales.data(), scales.size()*sizeof(scales[0]),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input, x_bf16.data(), x_bf16.size()*sizeof(x_bf16[0]),
                        hipMemcpyHostToDevice));
    CHECK(glm53_rocm_fp8_gemv_f32(d_output, d_weight, d_scale, d_input,
                                   rows, cols, nullptr));
    gpu.resize(rows);
    HIP_CHECK(hipMemcpy(gpu.data(), d_output, rows*sizeof(float),
                        hipMemcpyDeviceToHost));

    sum_abs.assign(rows, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            float decoded = 0.0f;
            CHECK(glm53_fp8_e4m3fn_decode(weights[(size_t)r*cols+c], &decoded) ==
                  GLM53_FP8_ORACLE_OK);
            sum_abs[r] += std::fabs(decoded *
                scales[(size_t)(r/128u)*matrix.scale_columns+c/128u] * x[c]);
        }
        CHECK(std::isfinite(gpu[r]));
        CHECK(std::fabs(gpu[r]-cpu[r]) <= 2.0e-6f*sum_abs[r] + 2.0e-5f);
    }
    std::printf("PASS official FP8 projection arch=%s rows=%u cols=%u first=%g last=%g\n",
                properties.gcnArchName, rows, cols, gpu.front(), gpu.back());
    result = 0;

done:
    if (reference) (void)std::fclose(reference);
    if (result != 0 && error[0]) std::fprintf(stderr, "%s\n", error);
    if (d_output) (void)hipFree(d_output);
    if (d_input) (void)hipFree(d_input);
    if (d_scale) (void)hipFree(d_scale);
    if (d_weight) (void)hipFree(d_weight);
    k3_st_read_release(&scale_read);
    k3_st_read_release(&weight_read);
    k3_st_model_close(&model);
    glm53_manifest_free(&manifest);
    return result;
}
