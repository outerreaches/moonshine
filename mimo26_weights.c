#include "mimo26_weights.h"

#include "glm53_fp8_oracle.h"
#include "mimo26_architecture.h"
#include "mimo26_ops.h"
#include "mimo26_router.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIMO26_FP8_BLOCK 128u
#define MIMO26_MXFP4_BLOCK 32u

static mimo26_weights_status fail(char *error, size_t size,
                                  mimo26_weights_status status,
                                  const char *format, ...)
{
    if (error != NULL && size > 0) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return status;
}

/* E2M1 magnitudes derived from the bit fields rather than copied. */
static float e2m1_value(uint8_t code)
{
    const uint32_t exponent = (uint32_t)((code >> 1) & 0x3u);
    const uint32_t mantissa = (uint32_t)(code & 0x1u);
    float magnitude;
    if (exponent == 0u) {
        magnitude = (float)mantissa * 0.5f;
    } else {
        magnitude = (1.0f + 0.5f * (float)mantissa) *
                    ldexpf(1.0f, (int)exponent - 1);
    }
    return (code & 0x8u) ? -magnitude : magnitude;
}

/* OCP E8M0: 2^(byte-127), with byte 0 meaning 2^-127 and 255 meaning NaN. */
static bool e8m0_value(uint8_t byte, float *value)
{
    if (byte == 0xFFu) {
        return false;
    }
    *value = byte == 0u ? ldexpf(1.0f, -127) : ldexpf(1.0f, (int)byte - 127);
    return true;
}

mimo26_weights_status mimo26_dequantize_fp8_block(
    uint16_t *out, const uint8_t *codes, const float *scales,
    size_t scale_stride, size_t rows, size_t cols, char *error,
    size_t error_size)
{
    if (out == NULL || codes == NULL || scales == NULL || rows == 0u ||
        cols == 0u || scale_stride == 0u) {
        return fail(error, error_size, MIMO26_WEIGHTS_INVALID_ARGUMENT,
                    "invalid fp8 dequantize arguments");
    }
    if (scale_stride < (cols + MIMO26_FP8_BLOCK - 1u) / MIMO26_FP8_BLOCK) {
        return fail(error, error_size, MIMO26_WEIGHTS_UNEXPECTED_LAYOUT,
                    "fp8 scale stride %zu too small for %zu columns",
                    scale_stride, cols);
    }
    for (size_t r = 0; r < rows; r++) {
        const size_t scale_row = (r / MIMO26_FP8_BLOCK) * scale_stride;
        const uint8_t *code_row = codes + r * cols;
        uint16_t *out_row = out + r * cols;
        for (size_t c = 0; c < cols; c++) {
            float decoded = 0.0f;
            const glm53_fp8_oracle_status status =
                glm53_fp8_e4m3fn_decode(code_row[c], &decoded);
            if (status != GLM53_FP8_ORACLE_OK) {
                return fail(error, error_size, MIMO26_WEIGHTS_NONFINITE_VALUE,
                            "fp8 NaN encoding at row %zu column %zu", r, c);
            }
            const float scale = scales[scale_row + c / MIMO26_FP8_BLOCK];
            if (!isfinite(scale)) {
                return fail(error, error_size, MIMO26_WEIGHTS_NONFINITE_VALUE,
                            "fp8 scale is not finite at row %zu", r);
            }
            out_row[c] = mimo26_f32_to_bf16(decoded * scale);
        }
    }
    return MIMO26_WEIGHTS_OK;
}

mimo26_weights_status mimo26_dequantize_mxfp4(
    uint16_t *out, const uint8_t *packed, const uint8_t *scales, size_t rows,
    size_t cols, char *error, size_t error_size)
{
    if (out == NULL || packed == NULL || scales == NULL || rows == 0u ||
        cols == 0u || (cols % MIMO26_MXFP4_BLOCK) != 0u) {
        return fail(error, error_size, MIMO26_WEIGHTS_INVALID_ARGUMENT,
                    "invalid mxfp4 dequantize arguments");
    }
    const size_t blocks = cols / MIMO26_MXFP4_BLOCK;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *packed_row = packed + r * (cols / 2u);
        const uint8_t *scale_row = scales + r * blocks;
        uint16_t *out_row = out + r * cols;
        for (size_t block = 0; block < blocks; block++) {
            float scale = 0.0f;
            if (!e8m0_value(scale_row[block], &scale)) {
                return fail(error, error_size, MIMO26_WEIGHTS_NONFINITE_VALUE,
                            "mxfp4 E8M0 NaN scale at row %zu block %zu", r,
                            block);
            }
            for (size_t i = 0; i < MIMO26_MXFP4_BLOCK; i++) {
                const size_t column = block * MIMO26_MXFP4_BLOCK + i;
                const uint8_t byte = packed_row[column / 2u];
                /* Element 2k is the low nibble of byte k. */
                const uint8_t code = (column & 1u) ? (uint8_t)(byte >> 4)
                                                   : (uint8_t)(byte & 0x0Fu);
                out_row[column] = mimo26_f32_to_bf16(e2m1_value(code) * scale);
            }
        }
    }
    return MIMO26_WEIGHTS_OK;
}

static const k3_st_tensor *require(const k3_st_model *model, const char *name,
                                   k3_st_dtype dtype, size_t ndim,
                                   const uint64_t *shape, char *error,
                                   size_t error_size,
                                   mimo26_weights_status *status)
{
    const k3_st_tensor *tensor = k3_st_find(model, name);
    if (tensor == NULL) {
        *status = fail(error, error_size, MIMO26_WEIGHTS_MISSING_TENSOR,
                       "missing tensor %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        *status = fail(error, error_size, MIMO26_WEIGHTS_UNEXPECTED_LAYOUT,
                       "%s has dtype %d ndim %u", name, (int)tensor->dtype,
                       (unsigned)tensor->ndim);
        return NULL;
    }
    for (size_t d = 0; d < ndim; d++) {
        if (tensor->shape[d] != shape[d]) {
            *status = fail(error, error_size, MIMO26_WEIGHTS_UNEXPECTED_LAYOUT,
                           "%s dimension %zu is %llu, expected %llu", name, d,
                           (unsigned long long)tensor->shape[d],
                           (unsigned long long)shape[d]);
            return NULL;
        }
    }
    *status = MIMO26_WEIGHTS_OK;
    return tensor;
}

/* Read a tensor's bytes into a fresh buffer the caller owns. */
static mimo26_weights_status read_tensor(const k3_st_model *model,
                                         const k3_st_tensor *tensor,
                                         void **out, char *error,
                                         size_t error_size)
{
    *out = NULL;
    k3_st_read read;
    memset(&read, 0, sizeof read);
    if (!k3_st_read_span(model, tensor->shard, tensor->physical_offset,
                         tensor->byte_length, 4096u, &read, error,
                         error_size)) {
        return MIMO26_WEIGHTS_READ_FAILED;
    }
    void *buffer = malloc(tensor->byte_length);
    if (buffer == NULL) {
        k3_st_read_release(&read);
        return fail(error, error_size, MIMO26_WEIGHTS_OUT_OF_MEMORY,
                    "out of memory reading %s", tensor->name);
    }
    memcpy(buffer, read.data, tensor->byte_length);
    k3_st_read_release(&read);
    *out = buffer;
    return MIMO26_WEIGHTS_OK;
}

/* Copy a natively-BF16 tensor. */
static mimo26_weights_status load_bf16(const k3_st_model *model,
                                       const char *name, size_t ndim,
                                       const uint64_t *shape,
                                       uint16_t **out, size_t *bytes,
                                       char *error, size_t error_size)
{
    mimo26_weights_status status = MIMO26_WEIGHTS_OK;
    const k3_st_tensor *tensor = require(model, name, K3_ST_DTYPE_BF16, ndim,
                                         shape, error, error_size, &status);
    if (tensor == NULL) {
        return status;
    }
    void *buffer = NULL;
    status = read_tensor(model, tensor, &buffer, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) {
        return status;
    }
    *out = (uint16_t *)buffer;
    *bytes += tensor->byte_length;
    return MIMO26_WEIGHTS_OK;
}

static mimo26_weights_status load_f32(const k3_st_model *model,
                                      const char *name, size_t ndim,
                                      const uint64_t *shape, float **out,
                                      size_t *bytes, char *error,
                                      size_t error_size)
{
    mimo26_weights_status status = MIMO26_WEIGHTS_OK;
    const k3_st_tensor *tensor = require(model, name, K3_ST_DTYPE_F32, ndim,
                                         shape, error, error_size, &status);
    if (tensor == NULL) {
        return status;
    }
    void *buffer = NULL;
    status = read_tensor(model, tensor, &buffer, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) {
        return status;
    }
    *out = (float *)buffer;
    *bytes += tensor->byte_length;
    return MIMO26_WEIGHTS_OK;
}

/*
 * Read the fused QKV projection, dequantize, and de-interleave it.
 *
 * The checkpoint stores fused QKV as kv_heads contiguous groups, each ordered
 * [Q_g | K_g | V_g], not as [all Q | all K | all V]. Per group Q has
 * (heads/kv_heads)*qk_dim rows, K has qk_dim, V has v_dim. The shipped
 * modeling_mimo_v2.py splits it the plain way, which scrambles every head --
 * confirmed against vLLM's _shard_fp8_qkv_proj, which documents the real
 * layout.
 *
 * Scale indexing differs by layer kind. SWA groups are 1856 rows = 14.5
 * blocks, so groups straddle block boundaries and the grid is flat
 * (8 x 14.5 = 116 = ceil(14848/128)). Global groups are 3392 rows = 26.5
 * blocks and the grid is 108 rather than 106, which is per-group padding to
 * 27 blocks. Using flat indexing on a global layer misassociates the scales
 * of every group after the first.
 */
static mimo26_weights_status load_fused_qkv(const k3_st_model *model,
                                            const char *name, uint64_t rows,
                                            uint64_t cols, size_t heads,
                                            size_t kv_heads, size_t qk_dim,
                                            size_t v_dim, uint16_t **out,
                                            size_t *bytes, char *error,
                                            size_t error_size)
{
    const uint64_t shape[2] = {rows, cols};
    mimo26_weights_status status = MIMO26_WEIGHTS_OK;
    const k3_st_tensor *tensor = require(model, name, K3_ST_DTYPE_F8_E4M3, 2u,
                                         shape, error, error_size, &status);
    if (tensor == NULL) {
        return status;
    }
    char scale_name[320];
    snprintf(scale_name, sizeof scale_name, "%s_scale_inv", name);
    const k3_st_tensor *scale_tensor = k3_st_find(model, scale_name);
    if (scale_tensor == NULL || scale_tensor->dtype != K3_ST_DTYPE_F32 ||
        scale_tensor->ndim != 2u) {
        return fail(error, error_size, MIMO26_WEIGHTS_UNEXPECTED_LAYOUT,
                    "%s missing or not a 2-D F32 grid", scale_name);
    }

    const size_t q_per_group = (heads / kv_heads) * qk_dim;
    const size_t rows_per_group = q_per_group + qk_dim + v_dim;
    if (rows_per_group * kv_heads != (size_t)rows) {
        return fail(error, error_size, MIMO26_WEIGHTS_UNEXPECTED_LAYOUT,
                    "%s: %llu rows is not %zu groups of %zu", name,
                    (unsigned long long)rows, kv_heads, rows_per_group);
    }
    const size_t scale_rows = (size_t)scale_tensor->shape[0];
    const size_t scale_stride = (size_t)scale_tensor->shape[1];
    const bool grouped_scales = (scale_rows % kv_heads) == 0u &&
                                (scale_rows / kv_heads) * MIMO26_FP8_BLOCK >=
                                    rows_per_group;
    const size_t scale_per_group = grouped_scales ? scale_rows / kv_heads : 0u;

    void *codes = NULL;
    void *scales = NULL;
    status = read_tensor(model, tensor, &codes, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) {
        return status;
    }
    status = read_tensor(model, scale_tensor, &scales, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) {
        free(codes);
        return status;
    }
    uint16_t *decoded = malloc((size_t)rows * (size_t)cols * sizeof *decoded);
    if (decoded == NULL) {
        free(codes);
        free(scales);
        return fail(error, error_size, MIMO26_WEIGHTS_OUT_OF_MEMORY,
                    "out of memory dequantizing %s", name);
    }

    const uint8_t *code_bytes = codes;
    const float *scale_values = scales;
    /* Destination offsets for the de-interleaved [Q | K | V] layout. */
    const size_t q_total = heads * qk_dim;
    const size_t k_total = kv_heads * qk_dim;
    for (size_t g = 0; g < kv_heads; g++) {
        for (size_t r = 0; r < rows_per_group; r++) {
            const size_t source_row = g * rows_per_group + r;
            size_t scale_row;
            if (grouped_scales) {
                scale_row = g * scale_per_group + r / MIMO26_FP8_BLOCK;
            } else {
                scale_row = source_row / MIMO26_FP8_BLOCK;
            }
            size_t destination_row;
            if (r < q_per_group) {
                destination_row = g * q_per_group + r;
            } else if (r < q_per_group + qk_dim) {
                destination_row = q_total + g * qk_dim + (r - q_per_group);
            } else {
                destination_row = q_total + k_total + g * v_dim +
                                  (r - q_per_group - qk_dim);
            }
            const uint8_t *source = code_bytes + source_row * (size_t)cols;
            uint16_t *target = decoded + destination_row * (size_t)cols;
            for (size_t c = 0; c < (size_t)cols; c++) {
                float value = 0.0f;
                if (glm53_fp8_e4m3fn_decode(source[c], &value) !=
                    GLM53_FP8_ORACLE_OK) {
                    free(codes);
                    free(scales);
                    free(decoded);
                    return fail(error, error_size,
                                MIMO26_WEIGHTS_NONFINITE_VALUE,
                                "%s: FP8 NaN at row %zu", name, source_row);
                }
                const float scale =
                    scale_values[scale_row * scale_stride +
                                 c / MIMO26_FP8_BLOCK];
                target[c] = mimo26_f32_to_bf16(value * scale);
            }
        }
    }
    free(codes);
    free(scales);
    *out = decoded;
    *bytes += (size_t)rows * (size_t)cols * sizeof *decoded;
    return MIMO26_WEIGHTS_OK;
}

/* Read an F8_E4M3 matrix plus its scale grid and dequantize to BF16. */
static mimo26_weights_status load_fp8(const k3_st_model *model,
                                      const char *name, uint64_t rows,
                                      uint64_t cols, uint16_t **out,
                                      size_t *bytes, char *error,
                                      size_t error_size)
{
    const uint64_t shape[2] = {rows, cols};
    mimo26_weights_status status = MIMO26_WEIGHTS_OK;
    const k3_st_tensor *tensor = require(model, name, K3_ST_DTYPE_F8_E4M3, 2u,
                                         shape, error, error_size, &status);
    if (tensor == NULL) {
        return status;
    }
    char scale_name[256];
    snprintf(scale_name, sizeof scale_name, "%s_scale_inv", name);
    const k3_st_tensor *scale_tensor = k3_st_find(model, scale_name);
    if (scale_tensor == NULL) {
        return fail(error, error_size, MIMO26_WEIGHTS_MISSING_TENSOR,
                    "missing %s", scale_name);
    }
    if (scale_tensor->dtype != K3_ST_DTYPE_F32 || scale_tensor->ndim != 2u) {
        return fail(error, error_size, MIMO26_WEIGHTS_UNEXPECTED_LAYOUT,
                    "%s is not a 2-D F32 grid", scale_name);
    }
    /* Surplus scale rows are allowed: the global QKV grid is [108,32] where
     * ceil(13568/128) is 106. Too few is still refused. */
    const uint64_t needed_rows = (rows + MIMO26_FP8_BLOCK - 1u) /
                                 MIMO26_FP8_BLOCK;
    const uint64_t needed_cols = (cols + MIMO26_FP8_BLOCK - 1u) /
                                 MIMO26_FP8_BLOCK;
    if (scale_tensor->shape[0] < needed_rows ||
        scale_tensor->shape[1] < needed_cols) {
        return fail(error, error_size, MIMO26_WEIGHTS_UNEXPECTED_LAYOUT,
                    "%s is %llux%llu, needs at least %llux%llu", scale_name,
                    (unsigned long long)scale_tensor->shape[0],
                    (unsigned long long)scale_tensor->shape[1],
                    (unsigned long long)needed_rows,
                    (unsigned long long)needed_cols);
    }

    void *codes = NULL;
    void *scales = NULL;
    status = read_tensor(model, tensor, &codes, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) {
        return status;
    }
    status = read_tensor(model, scale_tensor, &scales, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) {
        free(codes);
        return status;
    }
    uint16_t *decoded = malloc((size_t)rows * (size_t)cols * sizeof *decoded);
    if (decoded == NULL) {
        free(codes);
        free(scales);
        return fail(error, error_size, MIMO26_WEIGHTS_OUT_OF_MEMORY,
                    "out of memory dequantizing %s", name);
    }
    status = mimo26_dequantize_fp8_block(
        decoded, (const uint8_t *)codes, (const float *)scales,
        (size_t)scale_tensor->shape[1], (size_t)rows, (size_t)cols, error,
        error_size);
    free(codes);
    free(scales);
    if (status != MIMO26_WEIGHTS_OK) {
        free(decoded);
        return status;
    }
    *out = decoded;
    *bytes += (size_t)rows * (size_t)cols * sizeof *decoded;
    return MIMO26_WEIGHTS_OK;
}

void mimo26_layer_weights_free(mimo26_layer_weights *weights)
{
    if (weights == NULL) {
        return;
    }
    free(weights->input_layernorm);
    free(weights->post_attention_layernorm);
    free(weights->qkv_proj);
    free(weights->o_proj);
    free(weights->sink_bias);
    free(weights->gate_weight);
    free(weights->gate_bias);
    free(weights->dense_gate);
    free(weights->dense_up);
    free(weights->dense_down);
    memset(weights, 0, sizeof *weights);
}

mimo26_weights_status mimo26_layer_weights_load(
    mimo26_layer_weights *weights, const k3_st_model *model, uint32_t layer,
    char *error, size_t error_size)
{
    if (weights == NULL || model == NULL) {
        return fail(error, error_size, MIMO26_WEIGHTS_INVALID_ARGUMENT,
                    "invalid layer weight arguments");
    }
    memset(weights, 0, sizeof *weights);
    if (mimo26_attention_config_for_layer(layer, &weights->attention) !=
        MIMO26_ATTENTION_OK) {
        return fail(error, error_size, MIMO26_WEIGHTS_INVALID_ARGUMENT,
                    "layer %u is out of range", layer);
    }
    weights->layer = layer;
    weights->is_moe = mimo26_architecture_layer_kind(layer) !=
                      MIMO26_LAYER_DENSE_GLOBAL;

    char name[256];
    mimo26_weights_status status;
    const uint64_t hidden[1] = {MIMO26_HIDDEN_SIZE};

    snprintf(name, sizeof name, "model.layers.%u.input_layernorm.weight", layer);
    status = load_bf16(model, name, 1u, hidden, &weights->input_layernorm,
                       &weights->bytes, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }

    snprintf(name, sizeof name,
             "model.layers.%u.post_attention_layernorm.weight", layer);
    status = load_bf16(model, name, 1u, hidden,
                       &weights->post_attention_layernorm, &weights->bytes,
                       error, error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }

    snprintf(name, sizeof name, "model.layers.%u.self_attn.qkv_proj.weight",
             layer);
    status = load_fused_qkv(model, name, weights->attention.qkv_width,
                            MIMO26_HIDDEN_SIZE, MIMO26_QUERY_HEADS,
                            weights->attention.kv_heads, MIMO26_QK_HEAD_DIM,
                            MIMO26_V_HEAD_DIM, &weights->qkv_proj,
                            &weights->bytes, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }

    {
        const uint64_t o_shape[2] = {
            MIMO26_HIDDEN_SIZE,
            MIMO26_QUERY_HEADS * MIMO26_V_HEAD_DIM,
        };
        snprintf(name, sizeof name, "model.layers.%u.self_attn.o_proj.weight",
                 layer);
        status = load_bf16(model, name, 2u, o_shape, &weights->o_proj,
                           &weights->bytes, error, error_size);
        if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    }

    if (weights->attention.has_sink) {
        const uint64_t sink_shape[1] = {MIMO26_QUERY_HEADS};
        snprintf(name, sizeof name,
                 "model.layers.%u.self_attn.attention_sink_bias", layer);
        status = load_bf16(model, name, 1u, sink_shape, &weights->sink_bias,
                           &weights->bytes, error, error_size);
        if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    }

    if (weights->is_moe) {
        const uint64_t gate_shape[2] = {MIMO26_ROUTER_EXPERTS,
                                        MIMO26_HIDDEN_SIZE};
        snprintf(name, sizeof name, "model.layers.%u.mlp.gate.weight", layer);
        status = load_bf16(model, name, 2u, gate_shape, &weights->gate_weight,
                           &weights->bytes, error, error_size);
        if (status != MIMO26_WEIGHTS_OK) { goto failed; }

        const uint64_t bias_shape[1] = {MIMO26_ROUTER_EXPERTS};
        snprintf(name, sizeof name,
                 "model.layers.%u.mlp.gate.e_score_correction_bias", layer);
        status = load_f32(model, name, 1u, bias_shape, &weights->gate_bias,
                          &weights->bytes, error, error_size);
        if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    } else {
        snprintf(name, sizeof name, "model.layers.%u.mlp.gate_proj.weight",
                 layer);
        status = load_fp8(model, name, 16384u, MIMO26_HIDDEN_SIZE,
                          &weights->dense_gate, &weights->bytes, error,
                          error_size);
        if (status != MIMO26_WEIGHTS_OK) { goto failed; }
        snprintf(name, sizeof name, "model.layers.%u.mlp.up_proj.weight",
                 layer);
        status = load_fp8(model, name, 16384u, MIMO26_HIDDEN_SIZE,
                          &weights->dense_up, &weights->bytes, error,
                          error_size);
        if (status != MIMO26_WEIGHTS_OK) { goto failed; }
        snprintf(name, sizeof name, "model.layers.%u.mlp.down_proj.weight",
                 layer);
        status = load_fp8(model, name, MIMO26_HIDDEN_SIZE, 16384u,
                          &weights->dense_down, &weights->bytes, error,
                          error_size);
        if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    }
    return MIMO26_WEIGHTS_OK;

failed:
    mimo26_layer_weights_free(weights);
    return status;
}

void mimo26_expert_weights_free(mimo26_expert_weights *weights)
{
    if (weights == NULL) {
        return;
    }
    free(weights->gate);
    free(weights->up);
    free(weights->down);
    memset(weights, 0, sizeof *weights);
}

static mimo26_weights_status load_expert_projection(
    const k3_st_model *model, uint32_t layer, uint32_t expert,
    const char *projection, uint64_t rows, uint64_t cols, uint16_t **out,
    size_t *bytes, char *error, size_t error_size)
{
    char name[256];
    char scale_name[256];
    snprintf(name, sizeof name,
             "model.layers.%u.mlp.experts.%u.%s.weight", layer, expert,
             projection);
    snprintf(scale_name, sizeof scale_name,
             "model.layers.%u.mlp.experts.%u.%s.weight_scale", layer, expert,
             projection);

    const uint64_t packed_shape[2] = {rows, cols / 2u};
    mimo26_weights_status status = MIMO26_WEIGHTS_OK;
    const k3_st_tensor *tensor = require(model, name, K3_ST_DTYPE_U8, 2u,
                                         packed_shape, error, error_size,
                                         &status);
    if (tensor == NULL) {
        return status;
    }
    const uint64_t scale_shape[2] = {rows, cols / MIMO26_MXFP4_BLOCK};
    const k3_st_tensor *scale_tensor = require(model, scale_name,
                                               K3_ST_DTYPE_U8, 2u, scale_shape,
                                               error, error_size, &status);
    if (scale_tensor == NULL) {
        return status;
    }

    void *packed = NULL;
    void *scales = NULL;
    status = read_tensor(model, tensor, &packed, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) {
        return status;
    }
    status = read_tensor(model, scale_tensor, &scales, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) {
        free(packed);
        return status;
    }
    uint16_t *decoded = malloc((size_t)rows * (size_t)cols * sizeof *decoded);
    if (decoded == NULL) {
        free(packed);
        free(scales);
        return fail(error, error_size, MIMO26_WEIGHTS_OUT_OF_MEMORY,
                    "out of memory dequantizing %s", name);
    }
    status = mimo26_dequantize_mxfp4(decoded, (const uint8_t *)packed,
                                     (const uint8_t *)scales, (size_t)rows,
                                     (size_t)cols, error, error_size);
    free(packed);
    free(scales);
    if (status != MIMO26_WEIGHTS_OK) {
        free(decoded);
        return status;
    }
    *out = decoded;
    *bytes += (size_t)rows * (size_t)cols * sizeof *decoded;
    return MIMO26_WEIGHTS_OK;
}

mimo26_weights_status mimo26_expert_weights_load(
    mimo26_expert_weights *weights, const k3_st_model *model, uint32_t layer,
    uint32_t expert, char *error, size_t error_size)
{
    if (weights == NULL || model == NULL) {
        return fail(error, error_size, MIMO26_WEIGHTS_INVALID_ARGUMENT,
                    "invalid expert weight arguments");
    }
    memset(weights, 0, sizeof *weights);
    const mimo26_layer_kind kind = mimo26_architecture_layer_kind(layer);
    if (kind != MIMO26_LAYER_MOE_SWA && kind != MIMO26_LAYER_MOE_GLOBAL) {
        return fail(error, error_size, MIMO26_WEIGHTS_INVALID_ARGUMENT,
                    "layer %u has no routed experts", layer);
    }
    if (expert >= MIMO26_ROUTER_EXPERTS) {
        return fail(error, error_size, MIMO26_WEIGHTS_INVALID_ARGUMENT,
                    "expert %u is out of range", expert);
    }
    weights->layer = layer;
    weights->expert = expert;

    mimo26_weights_status status = load_expert_projection(
        model, layer, expert, "gate_proj", 2048u, 4096u, &weights->gate,
        &weights->bytes, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    status = load_expert_projection(model, layer, expert, "up_proj", 2048u,
                                    4096u, &weights->up, &weights->bytes,
                                    error, error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    status = load_expert_projection(model, layer, expert, "down_proj", 4096u,
                                    2048u, &weights->down, &weights->bytes,
                                    error, error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    return MIMO26_WEIGHTS_OK;

failed:
    mimo26_expert_weights_free(weights);
    return status;
}

void mimo26_static_weights_free(mimo26_static_weights *weights)
{
    if (weights == NULL) {
        return;
    }
    free(weights->embed_tokens);
    free(weights->norm);
    free(weights->lm_head);
    memset(weights, 0, sizeof *weights);
}

mimo26_weights_status mimo26_static_weights_load(
    mimo26_static_weights *weights, const k3_st_model *model, char *error,
    size_t error_size)
{
    if (weights == NULL || model == NULL) {
        return fail(error, error_size, MIMO26_WEIGHTS_INVALID_ARGUMENT,
                    "invalid static weight arguments");
    }
    memset(weights, 0, sizeof *weights);
    const uint64_t vocab[2] = {MIMO26_VOCAB_SIZE, MIMO26_HIDDEN_SIZE};
    const uint64_t hidden[1] = {MIMO26_HIDDEN_SIZE};

    mimo26_weights_status status = load_bf16(model, "model.embed_tokens.weight",
                                             2u, vocab, &weights->embed_tokens,
                                             &weights->bytes, error,
                                             error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    status = load_bf16(model, "model.norm.weight", 1u, hidden, &weights->norm,
                       &weights->bytes, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    status = load_bf16(model, "lm_head.weight", 2u, vocab, &weights->lm_head,
                       &weights->bytes, error, error_size);
    if (status != MIMO26_WEIGHTS_OK) { goto failed; }
    return MIMO26_WEIGHTS_OK;

failed:
    mimo26_static_weights_free(weights);
    return status;
}
