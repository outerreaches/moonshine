#include "mimo26_ops.h"

#include <math.h>
#include <string.h>

float mimo26_bf16_to_f32(uint16_t value)
{
    const uint32_t bits = (uint32_t)value << 16u;
    float result;
    memcpy(&result, &bits, sizeof result);
    return result;
}

uint16_t mimo26_f32_to_bf16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);

    /* Keep NaN a NaN: truncating the mantissa can otherwise produce an
     * infinity when every retained mantissa bit is zero. */
    if (((bits >> 23u) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0u) {
        return (uint16_t)((bits >> 16u) | 0x0040u);
    }

    const uint32_t lower = bits & 0x0000FFFFu;
    const uint32_t upper = bits >> 16u;
    const uint32_t round_up =
        (lower > 0x8000u || (lower == 0x8000u && (upper & 1u) == 1u)) ? 1u : 0u;
    return (uint16_t)(upper + round_up);
}

mimo26_ops_status mimo26_rmsnorm_bf16(uint16_t *out, const uint16_t *in,
                                      const uint16_t *weight, size_t count,
                                      float epsilon)
{
    if (out == NULL || in == NULL || weight == NULL || count == 0u) {
        return MIMO26_OPS_INVALID_ARGUMENT;
    }
    if (!isfinite(epsilon) || epsilon < 0.0f) {
        return MIMO26_OPS_NONFINITE_VALUE;
    }

    /* Validate before writing so a rejected call leaves out untouched. */
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(mimo26_bf16_to_f32(in[i])) ||
            !isfinite(mimo26_bf16_to_f32(weight[i]))) {
            return MIMO26_OPS_NONFINITE_VALUE;
        }
    }

    /* mean of squares in f32, summed in index order for reproducibility. */
    float sum_squares = 0.0f;
    for (size_t i = 0; i < count; i++) {
        const float x = mimo26_bf16_to_f32(in[i]);
        sum_squares += x * x;
    }
    const float variance = sum_squares / (float)count;
    if (!isfinite(variance)) {
        return MIMO26_OPS_NONFINITE_VALUE;
    }
    const float inverse = 1.0f / sqrtf(variance + epsilon);
    if (!isfinite(inverse)) {
        return MIMO26_OPS_NONFINITE_VALUE;
    }

    for (size_t i = 0; i < count; i++) {
        const float scaled = mimo26_bf16_to_f32(in[i]) * inverse;
        /* Round to BF16 first, then multiply by the BF16 weight. */
        const float rounded = mimo26_bf16_to_f32(mimo26_f32_to_bf16(scaled));
        const float product = mimo26_bf16_to_f32(weight[i]) * rounded;
        out[i] = mimo26_f32_to_bf16(product);
    }
    return MIMO26_OPS_OK;
}

mimo26_ops_status mimo26_silu_product_f32(float *out, const float *gate,
                                          const float *up, size_t count)
{
    if (out == NULL || gate == NULL || up == NULL || count == 0u) {
        return MIMO26_OPS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(gate[i]) || !isfinite(up[i])) {
            return MIMO26_OPS_NONFINITE_VALUE;
        }
    }
    for (size_t i = 0; i < count; i++) {
        const float sigmoid = 1.0f / (1.0f + expf(-gate[i]));
        out[i] = (gate[i] * sigmoid) * up[i];
    }
    return MIMO26_OPS_OK;
}

mimo26_ops_status mimo26_expert_accumulate_f32(float *accumulator,
                                               const uint16_t *expert_output,
                                               float weight, size_t count)
{
    if (accumulator == NULL || expert_output == NULL || count == 0u) {
        return MIMO26_OPS_INVALID_ARGUMENT;
    }
    if (!isfinite(weight)) {
        return MIMO26_OPS_NONFINITE_VALUE;
    }
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(mimo26_bf16_to_f32(expert_output[i]))) {
            return MIMO26_OPS_NONFINITE_VALUE;
        }
    }
    /* BF16 output promoted to f32, multiplied by the f32 router weight, and
     * added into the f32 accumulator -- no intermediate BF16 rounding. */
    for (size_t i = 0; i < count; i++) {
        accumulator[i] += mimo26_bf16_to_f32(expert_output[i]) * weight;
    }
    return MIMO26_OPS_OK;
}

mimo26_ops_status mimo26_expert_finalize_bf16(uint16_t *out,
                                              const float *accumulator,
                                              size_t count)
{
    if (out == NULL || accumulator == NULL || count == 0u) {
        return MIMO26_OPS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(accumulator[i])) {
            return MIMO26_OPS_NONFINITE_VALUE;
        }
    }
    for (size_t i = 0; i < count; i++) {
        out[i] = mimo26_f32_to_bf16(accumulator[i]);
    }
    return MIMO26_OPS_OK;
}

mimo26_ops_status mimo26_residual_add_bf16(uint16_t *out, const uint16_t *residual,
                                           const uint16_t *delta, size_t count)
{
    if (out == NULL || residual == NULL || delta == NULL || count == 0u) {
        return MIMO26_OPS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(mimo26_bf16_to_f32(residual[i])) ||
            !isfinite(mimo26_bf16_to_f32(delta[i]))) {
            return MIMO26_OPS_NONFINITE_VALUE;
        }
    }
    for (size_t i = 0; i < count; i++) {
        out[i] = mimo26_f32_to_bf16(mimo26_bf16_to_f32(residual[i]) +
                                    mimo26_bf16_to_f32(delta[i]));
    }
    return MIMO26_OPS_OK;
}

mimo26_ops_status mimo26_silu_product_bf16(uint16_t *out, const uint16_t *gate,
                                           const uint16_t *up, size_t count)
{
    if (out == NULL || gate == NULL || up == NULL || count == 0u) {
        return MIMO26_OPS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; i++) {
        const float g = mimo26_bf16_to_f32(gate[i]);
        const float u = mimo26_bf16_to_f32(up[i]);
        if (!isfinite(g) || !isfinite(u)) {
            return MIMO26_OPS_NONFINITE_VALUE;
        }
        /* silu rounds to BF16 first, then the product rounds again. */
        const float activated = g * (1.0f / (1.0f + expf(-g)));
        const float rounded = mimo26_bf16_to_f32(mimo26_f32_to_bf16(activated));
        out[i] = mimo26_f32_to_bf16(rounded * u);
    }
    return MIMO26_OPS_OK;
}

mimo26_ops_status mimo26_matmul_bf16(uint16_t *y, const uint16_t *weights,
                                     const uint16_t *x, size_t rows,
                                     size_t cols)
{
    if (y == NULL || weights == NULL || x == NULL || rows == 0u ||
        cols == 0u) {
        return MIMO26_OPS_INVALID_ARGUMENT;
    }
    for (size_t r = 0; r < rows; r++) {
        const uint16_t *row = weights + r * cols;
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            sum += mimo26_bf16_to_f32(row[c]) * mimo26_bf16_to_f32(x[c]);
        }
        if (!isfinite(sum)) {
            return MIMO26_OPS_NONFINITE_VALUE;
        }
        y[r] = mimo26_f32_to_bf16(sum);
    }
    return MIMO26_OPS_OK;
}
