#include "../glm53_expert_stream.h"
#include "../glm53_architecture.h"
#include "../glm53_fp8_dynamic.h"
#include "../glm53_manifest.h"
#include "../glm53_rocm_ops.h"
#include "../glm53_weights.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    goto done; \
} } while (0)
#define HIP_CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, \
                 #c, hipGetErrorString(e_)); goto done; \
} } while (0)

static bool same_float_bits(float a, float b) {
    uint32_t x = 0u, y = 0u;
    std::memcpy(&x, &a, sizeof(x));
    std::memcpy(&y, &b, sizeof(y));
    return x == y;
}

/* Deliberately independent of the production/oracle decoder. */
static bool decode_e4m3fn(uint8_t code, float *out) {
    const uint32_t magnitude = code & 0x7fu;
    const uint32_t exponent = magnitude >> 3u;
    const uint32_t mantissa = magnitude & 7u;
    float value;
    if (magnitude == 0x7fu) return false;
    if (exponent == 0u)
        value = std::ldexp((float)mantissa, -9);
    else
        value = std::ldexp(1.0f + (float)mantissa / 8.0f,
                           (int)exponent - 7);
    *out = (code & 0x80u) ? -value : value;
    return true;
}

struct matrix_view {
    const char *label;
    const k3_st_tensor *weight;
    const k3_st_tensor *scale;
    const uint8_t *weight_data;
    const float *scale_data;
    uint32_t rows;
    uint32_t columns;
    size_t weight_count;
    size_t scale_count;
};

static bool bind_matrix(matrix_view *m, const char *label,
                        const k3_st_tensor *weight,
                        const k3_st_tensor *scale,
                        uint32_t rows, uint32_t columns,
                        const glm53_expert_extent *weight_extent,
                        const uint8_t *weight_bytes,
                        const glm53_expert_extent *scale_extent,
                        const uint8_t *scale_bytes) {
    if (m == nullptr || weight == nullptr || scale == nullptr ||
        weight_extent == nullptr || scale_extent == nullptr) return false;
    const uint64_t weight_count = (uint64_t)rows * columns;
    const uint64_t scale_count =
        (uint64_t)((rows + 127u) / 128u) * ((columns + 127u) / 128u);
    const uint64_t weight_end = weight->physical_offset + weight->byte_length;
    const uint64_t scale_end = scale->physical_offset + scale->byte_length;
    if (weight->dtype != K3_ST_DTYPE_F8_E4M3 || weight->ndim != 2u ||
        weight->shape[0] != rows || weight->shape[1] != columns ||
        weight->byte_length != weight_count ||
        weight->shard != weight_extent->shard ||
        weight->physical_offset < weight_extent->offset ||
        weight_end < weight->physical_offset ||
        weight_end > weight_extent->offset + weight_extent->length ||
        scale->dtype != K3_ST_DTYPE_F32 || scale->ndim != 2u ||
        scale->shape[0] != (rows + 127u) / 128u ||
        scale->shape[1] != (columns + 127u) / 128u ||
        scale->byte_length != scale_count * sizeof(float) ||
        scale->shard != scale_extent->shard ||
        scale->physical_offset < scale_extent->offset ||
        scale_end < scale->physical_offset ||
        scale_end > scale_extent->offset + scale_extent->length)
        return false;
    m->label = label;
    m->weight = weight;
    m->scale = scale;
    m->weight_data = weight_bytes +
        (size_t)(weight->physical_offset - weight_extent->offset);
    m->scale_data = reinterpret_cast<const float *>(scale_bytes +
        (size_t)(scale->physical_offset - scale_extent->offset));
    m->rows = rows;
    m->columns = columns;
    m->weight_count = (size_t)weight_count;
    m->scale_count = (size_t)scale_count;
    return true;
}

static bool validate_official(const matrix_view &m) {
    for (size_t i = 0; i < m.weight_count; ++i) {
        if ((m.weight_data[i] & 0x7fu) == 0x7fu) {
            std::fprintf(stderr, "FAIL: %s official NaN code at %zu: 0x%02x\n",
                         m.label, i, (unsigned)m.weight_data[i]);
            return false;
        }
    }
    for (size_t i = 0; i < m.scale_count; ++i) {
        if (!std::isfinite(m.scale_data[i]) || m.scale_data[i] <= 0.0f) {
            std::fprintf(stderr, "FAIL: %s bad official scale at %zu: %.9g\n",
                         m.label, i, m.scale_data[i]);
            return false;
        }
    }
    return true;
}

/* Serial F32 dynamic-W8A8 reference. Accumulation order is column-major. */
static bool reference_gemv(const matrix_view &m,
                           const uint8_t *q, const float *q_scales,
                           std::vector<float> *output,
                           std::vector<double> *sum_abs) {
    output->assign(m.rows, 0.0f);
    sum_abs->assign(m.rows, 0.0);
    const uint32_t groups = m.columns / 128u;
    for (uint32_t r = 0; r < m.rows; ++r) {
        float sum = 0.0f;
        double magnitude = 0.0;
        for (uint32_t c = 0; c < m.columns; ++c) {
            float w = 0.0f, x = 0.0f;
            if (!decode_e4m3fn(m.weight_data[(size_t)r*m.columns+c], &w) ||
                !decode_e4m3fn(q[c], &x)) return false;
            const float term =
                (w * m.scale_data[(size_t)(r/128u)*groups+c/128u]) *
                (x * q_scales[c/128u]);
            sum += term;
            magnitude += std::fabs((double)term);
        }
        (*output)[r] = sum;
        (*sum_abs)[r] = magnitude;
    }
    return true;
}

static bool compare_output(const char *label, const std::vector<float> &got,
                           const std::vector<float> &want,
                           const std::vector<double> &sum_abs) {
    double worst_ratio = -1.0;
    size_t worst = 0u;
    for (size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs((double)got[i] - want[i]);
        const double bound = 2.0e-6 * sum_abs[i] + 2.0e-5;
        const double ratio = error / bound;
        if (!std::isfinite(got[i]) || ratio > worst_ratio) {
            worst_ratio = ratio;
            worst = i;
        }
    }
    const double error = std::fabs((double)got[worst] - want[worst]);
    const double bound = 2.0e-6 * sum_abs[worst] + 2.0e-5;
    std::printf("  %-4s max_ratio=%.8g argmax=%zu got=%.9g ref=%.9g "
                "abs=%.8g bound=%.8g sum_abs=%.8g\n",
                label, worst_ratio, worst, got[worst], want[worst], error,
                bound, sum_abs[worst]);
    if (!std::isfinite(got[worst]) || error > bound) {
        std::fprintf(stderr, "FAIL: %s output exceeds bound at %zu\n",
                     label, worst);
        return false;
    }
    return true;
}

static bool compare_quant(const char *label,
                          const std::vector<uint8_t> &got_q,
                          const std::vector<float> &got_s,
                          const std::vector<uint8_t> &want_q,
                          const std::vector<float> &want_s) {
    if (got_q != want_q) {
        size_t i = 0u;
        while (i < got_q.size() && got_q[i] == want_q[i]) ++i;
        std::fprintf(stderr, "FAIL: %s activation q[%zu] got=0x%02x want=0x%02x\n",
                     label, i, (unsigned)got_q[i], (unsigned)want_q[i]);
        return false;
    }
    for (size_t i = 0; i < got_s.size(); ++i) {
        if (!same_float_bits(got_s[i], want_s[i])) {
            std::fprintf(stderr, "FAIL: %s activation scale[%zu] got=%.9g "
                         "want=%.9g\n", label, i, got_s[i], want_s[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    constexpr uint32_t hidden = 4096u, intermediate = 2048u;
    char error[512] = {0};
    glm53_manifest manifest = {};
    k3_st_model all = {}, main_model = {};
    glm53_architecture_report architecture = {};
    glm53_weight_plan weights = {};
    glm53_expert_stream_stage stage = {};
    glm53_expert_stream_ledger ledger = {};
    k3_st_read weight_read = {}, scale_read = {};
    std::vector<uint8_t> staged_weights, resident_scales;
    std::vector<float> x, h, gate_ref, up_ref, down_ref;
    std::vector<double> gate_abs, up_abs, down_abs;
    std::vector<uint8_t> xq, hq, gpu_q;
    std::vector<float> xs, hs, gpu_s, gpu_output;
    matrix_view down = {}, gate = {}, up = {};
    const glm53_expert_plan *expert = nullptr;
    size_t main_count = 0u, weight_reads = 0u, weight_copies = 0u;
    uint8_t *d_weights = nullptr, *d_q = nullptr;
    float *d_scales = nullptr, *d_input = nullptr, *d_qscales = nullptr,
          *d_output = nullptr;
    hipStream_t stream = nullptr;
    hipDeviceProp_t properties = {};
    int devices = 0, result = 1;
    const auto started = std::chrono::steady_clock::now();

    if (argc != 2) {
        std::fprintf(stderr, "usage: %s OFFICIAL_ROOT\n", argv[0]);
        return 2;
    }
    CHECK(glm53_manifest_load(&manifest, argv[1], error, sizeof(error)));
    CHECK(k3_st_model_open_5digit_total(&all, argv[1], GLM53_SHARD_COUNT,
                                        error, sizeof(error)));
    CHECK(glm53_manifest_reconcile(&manifest, &all, error, sizeof(error)));
    CHECK(glm53_architecture_validate(&all, &architecture,
                                      error, sizeof(error)));
    main_model.tensors = (k3_st_tensor *)std::calloc(
        GLM53_MAIN_TENSOR_COUNT, sizeof(*main_model.tensors));
    CHECK(main_model.tensors != nullptr);
    main_model.shards = all.shards;
    main_model.shard_count = all.shard_count;
    main_model.routed_span = all.routed_span;
    main_model.routed_span_context = all.routed_span_context;
    for (size_t i = 0u; i < all.tensor_count; ++i) {
        if (glm53_architecture_validate_main_tensor(&all.tensors[i], nullptr)) {
            CHECK(main_count < GLM53_MAIN_TENSOR_COUNT);
            main_model.tensors[main_count++] = all.tensors[i];
        }
    }
    CHECK(main_count == GLM53_MAIN_TENSOR_COUNT);
    main_model.tensor_count = main_count;
    main_model.tensor_capacity = main_count;
    CHECK(glm53_architecture_validate_main(&main_model, &architecture,
                                           error, sizeof(error)));
    CHECK(glm53_weight_plan_build_manifest(&weights, &manifest, &main_model,
                                            error, sizeof(error)));
    CHECK(weights.routed_experts.expert_count ==
          (size_t)GLM53_EXPERT_LAYER_COUNT * GLM53_EXPERTS_PER_LAYER);
    expert = &weights.routed_experts.experts[0];
    CHECK(expert->layer == 3u && expert->expert == 0u);
    CHECK(expert->logical[0].length == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(expert->logical[1].length == GLM53_EXPERT_SCALE_BYTES);
    CHECK(expert->logical_bytes == GLM53_EXPERT_LOGICAL_BYTES);
    CHECK(glm53_expert_stream_stage_build(&stage, expert, &all,
                                           error, sizeof(error)) ==
          GLM53_EXPERT_STREAM_OK);
    CHECK(stage.request_count == 1u && stage.copy_count == 1u);
    CHECK(stage.logical_bytes == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(stage.copy.byte_count == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(stage.copy.slot_offset == 0u);
    CHECK(stage.request.shard == expert->logical[0].shard);
    CHECK(stage.request.file_offset <= expert->logical[0].offset);
    CHECK(stage.copy.request_offset ==
          expert->logical[0].offset - stage.request.file_offset);
    CHECK(stage.copy.request_offset + stage.copy.byte_count <=
          stage.request.byte_count);
    CHECK(stage.request.file_offset <= UINT64_MAX - stage.request.byte_count);
    CHECK(expert->logical[1].offset <=
          UINT64_MAX - expert->logical[1].length);
    CHECK(stage.request.shard != expert->logical[1].shard ||
          stage.request.file_offset + stage.request.byte_count <=
              expert->logical[1].offset ||
          stage.request.file_offset >=
              expert->logical[1].offset + expert->logical[1].length);
    CHECK(glm53_expert_stream_ledger_from_stage(&ledger, &stage) ==
          GLM53_EXPERT_STREAM_OK);
    CHECK(ledger.request_count == 1u && ledger.copy_count == 1u);
    CHECK(ledger.copied_bytes == GLM53_EXPERT_WEIGHT_BYTES);

    /* The sole streamed-weight payload read. Scales are a separate resident
     * test read below and are not covered by this exact logical span. */
    CHECK(k3_st_read_span(&all, expert->logical[0].shard,
                          expert->logical[0].offset,
                          GLM53_EXPERT_WEIGHT_BYTES,
                          GLM53_EXPERT_IO_ALIGNMENT, &weight_read,
                          error, sizeof(error)));
    ++weight_reads;
    CHECK(weight_read.data_bytes == GLM53_EXPERT_WEIGHT_BYTES);
    CHECK(weight_read.physical_offset == stage.request.file_offset);
    CHECK(weight_read.allocation_bytes == stage.request.byte_count);
    CHECK(weight_read.data == static_cast<uint8_t *>(weight_read.allocation) +
          stage.copy.request_offset);
    staged_weights.resize((size_t)GLM53_EXPERT_WEIGHT_BYTES);
    std::memcpy(staged_weights.data(), weight_read.data,
                staged_weights.size());
    ++weight_copies;
    CHECK(k3_st_read_span(&all, expert->logical[1].shard,
                          expert->logical[1].offset,
                          GLM53_EXPERT_SCALE_BYTES,
                          GLM53_EXPERT_IO_ALIGNMENT, &scale_read,
                          error, sizeof(error)));
    CHECK(scale_read.data_bytes == GLM53_EXPERT_SCALE_BYTES);
    resident_scales.assign(scale_read.data,
                           scale_read.data + GLM53_EXPERT_SCALE_BYTES);
    CHECK(weight_reads == 1u && weight_copies == 1u);

    CHECK(bind_matrix(&down, "down",
              expert->tensors[GLM53_EXPERT_DOWN_WEIGHT],
              expert->tensors[GLM53_EXPERT_DOWN_SCALE], hidden, intermediate,
              &expert->logical[0], staged_weights.data(),
              &expert->logical[1], resident_scales.data()));
    CHECK(bind_matrix(&gate, "gate",
              expert->tensors[GLM53_EXPERT_GATE_WEIGHT],
              expert->tensors[GLM53_EXPERT_GATE_SCALE], intermediate, hidden,
              &expert->logical[0], staged_weights.data(),
              &expert->logical[1], resident_scales.data()));
    CHECK(bind_matrix(&up, "up",
              expert->tensors[GLM53_EXPERT_UP_WEIGHT],
              expert->tensors[GLM53_EXPERT_UP_SCALE], intermediate, hidden,
              &expert->logical[0], staged_weights.data(),
              &expert->logical[1], resident_scales.data()));
    CHECK(validate_official(down) && validate_official(gate) &&
          validate_official(up));

    x.resize(hidden);
    for (uint32_t i = 0; i < hidden; ++i)
        x[i] = (float)((int)((i * 37u + 11u) % 257u) - 128) / 256.0f;
    xq.resize(hidden); xs.resize(hidden / 128u);
    CHECK(glm53_fp8_dynamic_quantize_f32(
        xq.data(), xq.size(), hidden, xs.data(), xs.size(), xs.size(),
        x.data(), x.size(), hidden, 1u, hidden) == GLM53_FP8_DYNAMIC_OK);
    CHECK(reference_gemv(gate, xq.data(), xs.data(), &gate_ref, &gate_abs));
    CHECK(reference_gemv(up, xq.data(), xs.data(), &up_ref, &up_abs));
    h.resize(intermediate);
    for (uint32_t i = 0; i < intermediate; ++i) {
        h[i] = (gate_ref[i] / (1.0f + std::exp(-gate_ref[i]))) * up_ref[i];
        CHECK(std::isfinite(h[i]));
    }
    hq.resize(intermediate); hs.resize(intermediate / 128u);
    CHECK(glm53_fp8_dynamic_quantize_f32(
        hq.data(), hq.size(), intermediate, hs.data(), hs.size(), hs.size(),
        h.data(), h.size(), intermediate, 1u, intermediate) ==
          GLM53_FP8_DYNAMIC_OK);
    CHECK(reference_gemv(down, hq.data(), hs.data(), &down_ref, &down_abs));

    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0);
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    CHECK(std::strncmp(properties.gcnArchName, "gfx1151", 7u) == 0);
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMalloc(&d_weights, staged_weights.size()));
    HIP_CHECK(hipMalloc(&d_scales, resident_scales.size()));
    HIP_CHECK(hipMalloc(&d_input, hidden * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_q, hidden));
    HIP_CHECK(hipMalloc(&d_qscales, (hidden / 128u) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, hidden * sizeof(float)));
    HIP_CHECK(hipMemcpyAsync(d_weights, staged_weights.data(),
                             staged_weights.size(), hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(d_scales, resident_scales.data(),
                             resident_scales.size(), hipMemcpyHostToDevice,
                             stream));

    for (const matrix_view *m : {&gate, &up}) {
        const size_t wo = (size_t)(m->weight->physical_offset -
                                   expert->logical[0].offset);
        const size_t so = (size_t)(m->scale->physical_offset -
                                   expert->logical[1].offset) / sizeof(float);
        HIP_CHECK(hipMemcpyAsync(d_input, x.data(), hidden*sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK(glm53_rocm_fp8_dynamic_gemv_f32(
            d_output, m->rows, d_q, m->columns, d_qscales, m->columns/128u,
            d_weights + wo, m->weight_count, d_scales + so, m->scale_count,
            d_input, m->columns, m->rows, m->columns, stream));
        gpu_q.resize(m->columns); gpu_s.resize(m->columns/128u);
        gpu_output.resize(m->rows);
        HIP_CHECK(hipMemcpyAsync(gpu_q.data(), d_q, gpu_q.size(),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(gpu_s.data(), d_qscales,
                                 gpu_s.size()*sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(gpu_output.data(), d_output,
                                 gpu_output.size()*sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        CHECK(compare_quant(m->label, gpu_q, gpu_s, xq, xs));
        CHECK(compare_output(m->label, gpu_output,
              m == &gate ? gate_ref : up_ref,
              m == &gate ? gate_abs : up_abs));
    }
    {
        const matrix_view *m = &down;
        const size_t wo = (size_t)(m->weight->physical_offset -
                                   expert->logical[0].offset);
        const size_t so = (size_t)(m->scale->physical_offset -
                                   expert->logical[1].offset) / sizeof(float);
        HIP_CHECK(hipMemcpyAsync(d_input, h.data(), h.size()*sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK(glm53_rocm_fp8_dynamic_gemv_f32(
            d_output, m->rows, d_q, m->columns, d_qscales, m->columns/128u,
            d_weights + wo, m->weight_count, d_scales + so, m->scale_count,
            d_input, m->columns, m->rows, m->columns, stream));
        gpu_q.resize(m->columns); gpu_s.resize(m->columns/128u);
        gpu_output.resize(m->rows);
        HIP_CHECK(hipMemcpyAsync(gpu_q.data(), d_q, gpu_q.size(),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(gpu_s.data(), d_qscales,
                                 gpu_s.size()*sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(gpu_output.data(), d_output,
                                 gpu_output.size()*sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        CHECK(compare_quant(m->label, gpu_q, gpu_s, hq, hs));
        CHECK(compare_output(m->label, gpu_output, down_ref, down_abs));
    }
    result = 0;

done:
    if (stream != nullptr) {
        hipError_t e = hipStreamSynchronize(stream);
        if (e != hipSuccess) result = 1;
    }
    if (d_output != nullptr && hipFree(d_output) != hipSuccess) result = 1;
    if (d_qscales != nullptr && hipFree(d_qscales) != hipSuccess) result = 1;
    if (d_input != nullptr && hipFree(d_input) != hipSuccess) result = 1;
    if (d_q != nullptr && hipFree(d_q) != hipSuccess) result = 1;
    if (d_scales != nullptr && hipFree(d_scales) != hipSuccess) result = 1;
    if (d_weights != nullptr && hipFree(d_weights) != hipSuccess) result = 1;
    if (stream != nullptr && hipStreamDestroy(stream) != hipSuccess) result = 1;
    k3_st_read_release(&scale_read);
    k3_st_read_release(&weight_read);
    glm53_weight_plan_free(&weights);
    std::free(main_model.tensors);
    k3_st_model_close(&all);
    glm53_manifest_free(&manifest);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (result == 0)
        std::printf("PASS phase5b official layer=3 expert=0 weight_reads=%zu "
                    "weight_copies=%zu elapsed=%.3fs\n",
                    weight_reads, weight_copies, seconds);
    else if (error[0] != '\0')
        std::fprintf(stderr, "detail: %s\n", error);
    return result;
}
