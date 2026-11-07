#include "glm53_dense_ops.h"

#include "glm53_rocm_ops.h"
#include "glm53_vector_ops.h"

#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstdint>

namespace {
struct Span { const void *pointer; size_t count; size_t element_size; };

static bool checked_product(size_t a, size_t b, size_t *result) {
    if (a != 0u && b > SIZE_MAX / a) return false;
    *result = a * b;
    return true;
}

static bool pairwise_disjoint(const Span *spans, size_t count) {
    if (spans == nullptr || count == 0u || count > 15u) return false;
    uintptr_t begin[15], end[15];
    for (size_t i = 0u; i < count; ++i) {
        if (spans[i].pointer == nullptr || spans[i].count == 0u ||
            spans[i].element_size == 0u ||
            spans[i].count > SIZE_MAX / spans[i].element_size) return false;
        const size_t bytes = spans[i].count * spans[i].element_size;
        begin[i] = reinterpret_cast<uintptr_t>(spans[i].pointer);
        if (begin[i] > UINTPTR_MAX - bytes) return false;
        end[i] = begin[i] + bytes;
    }
    for (size_t i = 0u; i < count; ++i)
        for (size_t j = i + 1u; j < count; ++j)
            if (begin[i] < end[j] && begin[j] < end[i]) return false;
    return true;
}
} // namespace

extern "C" bool glm53_dense_mlp_fp8_bf16(
    void *output, size_t output_count,
    const void *input, size_t input_count,
    const glm53_dense_fp8_matrix *gate,
    const glm53_dense_fp8_matrix *up,
    const glm53_dense_fp8_matrix *down,
    uint32_t hidden, uint32_t intermediate, float limit,
    glm53_dense_scratch *scratch, void *stream) {
    if (gate == nullptr || up == nullptr || down == nullptr ||
        scratch == nullptr || hidden == 0u || intermediate == 0u ||
        hidden % 128u != 0u || intermediate % 128u != 0u ||
        !std::isfinite(limit) || limit <= 0.0f) return false;

    const size_t h = static_cast<size_t>(hidden);
    const size_t m = static_cast<size_t>(intermediate);
    size_t gate_weight_count, down_weight_count;
    if (!checked_product(m, h, &gate_weight_count) ||
        !checked_product(h, m, &down_weight_count)) return false;
    const size_t h_groups = h / 128u, m_groups = m / 128u;
    size_t scale_count;
    if (!checked_product(h_groups, m_groups, &scale_count)) return false;
    const size_t maximum = h > m ? h : m;
    const size_t maximum_groups = maximum / 128u;

    if (output_count < h || input_count < h ||
        gate->weights_count < gate_weight_count ||
        up->weights_count < gate_weight_count ||
        down->weights_count < down_weight_count ||
        gate->inverse_scales_count < scale_count ||
        up->inverse_scales_count < scale_count ||
        down->inverse_scales_count < scale_count ||
        scratch->input_f32_count < h || scratch->gate_f32_count < m ||
        scratch->up_f32_count < m || scratch->activation_f32_count < m ||
        scratch->down_f32_count < h || scratch->q8_count < maximum ||
        scratch->dynamic_inverse_scale_count < maximum_groups) return false;

    /* Validate complete accessible spans, not merely the required prefixes.
     * This also proves every nested primitive's non-overlap contract. */
    const Span spans[] = {
        {output, output_count, sizeof(hip_bfloat16)},
        {input, input_count, sizeof(hip_bfloat16)},
        {gate->weights, gate->weights_count, sizeof(uint8_t)},
        {gate->inverse_scales, gate->inverse_scales_count, sizeof(float)},
        {up->weights, up->weights_count, sizeof(uint8_t)},
        {up->inverse_scales, up->inverse_scales_count, sizeof(float)},
        {down->weights, down->weights_count, sizeof(uint8_t)},
        {down->inverse_scales, down->inverse_scales_count, sizeof(float)},
        {scratch->input_f32, scratch->input_f32_count, sizeof(float)},
        {scratch->gate_f32, scratch->gate_f32_count, sizeof(float)},
        {scratch->up_f32, scratch->up_f32_count, sizeof(float)},
        {scratch->activation_f32, scratch->activation_f32_count, sizeof(float)},
        {scratch->down_f32, scratch->down_f32_count, sizeof(float)},
        {scratch->q8, scratch->q8_count, sizeof(uint8_t)},
        {scratch->dynamic_inverse_scales,
         scratch->dynamic_inverse_scale_count, sizeof(float)},
    };
    if (!pairwise_disjoint(spans, sizeof(spans) / sizeof(spans[0]))) return false;

    if (!glm53_vector_cast_bf16_f32(
            scratch->input_f32, scratch->input_f32_count,
            input, input_count, h, stream)) return false;
    if (!glm53_rocm_fp8_dynamic_gemv_f32(
            scratch->gate_f32, scratch->gate_f32_count,
            scratch->q8, scratch->q8_count,
            scratch->dynamic_inverse_scales,
            scratch->dynamic_inverse_scale_count,
            gate->weights, gate->weights_count,
            gate->inverse_scales, gate->inverse_scales_count,
            scratch->input_f32, scratch->input_f32_count,
            intermediate, hidden, stream)) return false;
    if (!glm53_rocm_fp8_dynamic_gemv_f32(
            scratch->up_f32, scratch->up_f32_count,
            scratch->q8, scratch->q8_count,
            scratch->dynamic_inverse_scales,
            scratch->dynamic_inverse_scale_count,
            up->weights, up->weights_count,
            up->inverse_scales, up->inverse_scales_count,
            scratch->input_f32, scratch->input_f32_count,
            intermediate, hidden, stream)) return false;
    if (!glm53_vector_swiglu_f32(
            scratch->activation_f32, scratch->activation_f32_count,
            scratch->gate_f32, scratch->gate_f32_count,
            scratch->up_f32, scratch->up_f32_count, m, limit, stream)) return false;
    if (!glm53_rocm_fp8_dynamic_gemv_f32(
            scratch->down_f32, scratch->down_f32_count,
            scratch->q8, scratch->q8_count,
            scratch->dynamic_inverse_scales,
            scratch->dynamic_inverse_scale_count,
            down->weights, down->weights_count,
            down->inverse_scales, down->inverse_scales_count,
            scratch->activation_f32, scratch->activation_f32_count,
            hidden, intermediate, stream)) return false;
    return glm53_vector_cast_f32_bf16(
        output, output_count, scratch->down_f32, scratch->down_f32_count,
        h, stream);
}
