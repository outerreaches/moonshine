#ifndef GLM53_FP8_DYNAMIC_H
#define GLM53_FP8_DYNAMIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_FP8_DYNAMIC_GROUP_COLS 128u
#define GLM53_FP8_DYNAMIC_MAX 448.0f
#define GLM53_FP8_DYNAMIC_EPS 1.0e-10f
#define GLM53_FP8_DYNAMIC_INV_MAX (1.0f / 448.0f)
#define GLM53_FP8_DYNAMIC_SCALE_FLOOR \
    (GLM53_FP8_DYNAMIC_EPS * GLM53_FP8_DYNAMIC_INV_MAX)

typedef enum glm53_fp8_dynamic_status {
    GLM53_FP8_DYNAMIC_OK = 0,
    GLM53_FP8_DYNAMIC_INVALID_ARGUMENT,
    GLM53_FP8_DYNAMIC_DIMENSION_OVERFLOW,
    GLM53_FP8_DYNAMIC_BUFFER_TOO_SMALL,
    GLM53_FP8_DYNAMIC_NONFINITE_VALUE,
    GLM53_FP8_DYNAMIC_NAN_ENCODING
} glm53_fp8_dynamic_status;

/*
 * Cast one finite binary32 value to OCP E4M3FN with round-to-nearest-even.
 * This has torch.float8_e4m3fn cast semantics: signed zero is retained,
 * subnormals are produced, and a finite magnitude above the 448/NaN midpoint
 * (464) returns GLM53_FP8_DYNAMIC_NAN_ENCODING without changing *encoded.
 * NaN and infinity return GLM53_FP8_DYNAMIC_NONFINITE_VALUE.  On every error,
 * *encoded is unchanged.
 */
glm53_fp8_dynamic_status glm53_fp8_e4m3fn_encode(float value,
                                                  uint8_t *encoded);

/* Decode all non-NaN OCP E4M3FN codes exactly. */
glm53_fp8_dynamic_status glm53_fp8_dynamic_decode(uint8_t encoded,
                                                   float *value);

/*
 * Model-free oracle for activation_scheme=dynamic with a [1, 128]
 * activation group paired with [128, 128] weight blocks.
 *
 * input and quantized are row-major [tokens, columns].  inverse_scales is
 * row-major [tokens, columns / 128].  "inverse" means the dequant multiplier:
 * decoded(quantized[t,c]) * inverse_scales[t,c/128] approximates input[t,c].
 * Strides and counts are in elements and describe complete accessible spans.
 * columns must be a nonzero multiple of 128.  Input/output buffers must not
 * overlap.
 *
 * For each token/group, in binary32:
 *   absmax = max(abs(x));
 *   scale = max(absmax, 1e-10) * (1 / 448);
 *   q = E4M3FN_RNE(clamp(x / scale, -448, 448)).
 *
 * The function allocates no memory.  It validates all geometry and all input
 * values before writing either output.  Therefore any reported error leaves
 * both output buffers unchanged.
 */
glm53_fp8_dynamic_status glm53_fp8_dynamic_quantize_f32(
    uint8_t *quantized, size_t quantized_count, size_t quantized_row_stride,
    float *inverse_scales, size_t inverse_scale_count,
    size_t inverse_scale_row_stride,
    const float *input, size_t input_count, size_t input_row_stride,
    size_t tokens, size_t columns);

#ifdef __cplusplus
}
#endif

#endif
