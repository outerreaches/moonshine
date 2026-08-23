#include "k3_q8_codec.h"

#include <math.h>
#include <string.h>

static float bf16_to_float(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16u;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static long round_to_nearest_even(float value) {
    const float lower_value = floorf(value);
    long lower = (long)lower_value;
    const float fraction = value - lower_value;
    if (fraction > 0.5f ||
        (fraction == 0.5f && lower % 2 != 0)) {
        lower++;
    }
    return lower;
}

bool k3_q8_quantize_bf16_block(
        const uint16_t input[K3_Q8_CODEC_BLOCK],
        int8_t output[K3_Q8_CODEC_BLOCK],
        float *scale) {
    if (!input || !output || !scale) return false;

    float maximum = 0.0f;
    float values[K3_Q8_CODEC_BLOCK];
    for (uint32_t index = 0u; index < K3_Q8_CODEC_BLOCK; index++) {
        values[index] = bf16_to_float(input[index]);
        if (!isfinite(values[index])) return false;
        maximum = fmaxf(maximum, fabsf(values[index]));
    }

    *scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    for (uint32_t index = 0u; index < K3_Q8_CODEC_BLOCK; index++) {
        const float quotient = values[index] / *scale;
        long quantized = round_to_nearest_even(quotient);
        if (quantized < -127) quantized = -127;
        if (quantized > 127) quantized = 127;
        output[index] = (int8_t)quantized;
    }
    return true;
}

void k3_q8_histogram_reset(k3_q8_histogram *histogram) {
    if (histogram) memset(histogram, 0, sizeof(*histogram));
}

bool k3_q8_histogram_add(k3_q8_histogram *histogram,
                          const int8_t *values,
                          size_t count) {
    if (!histogram || (!values && count != 0u) ||
        count > UINT64_MAX - histogram->total) {
        return false;
    }
    for (size_t index = 0u; index < count; index++) {
        histogram->counts[(uint8_t)values[index]]++;
    }
    histogram->total += (uint64_t)count;
    return true;
}

double k3_q8_histogram_entropy(const k3_q8_histogram *histogram) {
    if (!histogram || histogram->total == 0u) return 0.0;
    const double total = (double)histogram->total;
    double entropy = 0.0;
    for (uint32_t symbol = 0u; symbol < K3_Q8_CODEC_SYMBOLS; symbol++) {
        if (histogram->counts[symbol] == 0u) continue;
        const double probability =
            (double)histogram->counts[symbol] / total;
        entropy -= probability * log2(probability);
    }
    return entropy;
}
