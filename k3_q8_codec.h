#ifndef K3_Q8_CODEC_H
#define K3_Q8_CODEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    K3_Q8_CODEC_BLOCK = 128,
    K3_Q8_CODEC_SYMBOLS = 256,
};

typedef struct {
    uint64_t counts[K3_Q8_CODEC_SYMBOLS];
    uint64_t total;
} k3_q8_histogram;

/*
 * CPU reference for Moonshine's Q8/128 kernel. INPUT contains one little-endian
 * BF16 block. SCALE is max(abs(input))/127, or 1 for an all-zero block.
 * Quantization uses round-to-nearest-even and clamps to [-127, 127].
 */
bool k3_q8_quantize_bf16_block(const uint16_t input[K3_Q8_CODEC_BLOCK],
                               int8_t output[K3_Q8_CODEC_BLOCK],
                               float *scale);

void k3_q8_histogram_reset(k3_q8_histogram *histogram);
bool k3_q8_histogram_add(k3_q8_histogram *histogram,
                          const int8_t *values,
                          size_t count);
double k3_q8_histogram_entropy(const k3_q8_histogram *histogram);

#ifdef __cplusplus
}
#endif

#endif
