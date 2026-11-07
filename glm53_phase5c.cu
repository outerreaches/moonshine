#include "glm53_phase5c.h"

#include "glm53_dense_ops.h"
#include "glm53_mhc_ops.h"
#include "k3_rocm_ops.h"

#include <hip/hip_runtime.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

enum cached_binding {
    C_LM_HEAD = 0, C_EMBED, C_FINAL_NORM,
    C_HC_ATTN_BASE, C_HC_ATTN_SCALE, C_HC_ATTN_FN,
    C_HC_FFN_BASE, C_HC_FFN_SCALE, C_HC_FFN_FN,
    C_INPUT_NORM, C_POST_ATTN_NORM,
    C_DOWN, C_DOWN_SCALE, C_GATE, C_GATE_SCALE, C_UP, C_UP_SCALE
};

struct binding_contract {
    bool global;
    unsigned role;
    k3_st_dtype dtype;
    uint8_t ndim;
    uint64_t a, b;
};

static const binding_contract kContracts[] = {
    {true, GLM53_GLOBAL_LM_HEAD, K3_ST_DTYPE_BF16, 2, 154880, 4096},
    {true, GLM53_GLOBAL_EMBED_TOKENS, K3_ST_DTYPE_BF16, 2, 154880, 4096},
    {true, GLM53_GLOBAL_FINAL_NORM, K3_ST_DTYPE_BF16, 1, 4096, 0},
    {false, GLM53_ROLE_HC_ATTN_BASE, K3_ST_DTYPE_F32, 1, 24, 0},
    {false, GLM53_ROLE_HC_ATTN_SCALE, K3_ST_DTYPE_F32, 1, 3, 0},
    {false, GLM53_ROLE_HC_ATTN_FN, K3_ST_DTYPE_BF16, 2, 24, 16384},
    {false, GLM53_ROLE_HC_FFN_BASE, K3_ST_DTYPE_F32, 1, 24, 0},
    {false, GLM53_ROLE_HC_FFN_SCALE, K3_ST_DTYPE_F32, 1, 3, 0},
    {false, GLM53_ROLE_HC_FFN_FN, K3_ST_DTYPE_BF16, 2, 24, 16384},
    {false, GLM53_ROLE_INPUT_NORM, K3_ST_DTYPE_BF16, 1, 4096, 0},
    {false, GLM53_ROLE_POST_ATTN_NORM, K3_ST_DTYPE_BF16, 1, 4096, 0},
    {false, GLM53_ROLE_MLP_DOWN, K3_ST_DTYPE_F8_E4M3, 2, 4096, 12288},
    {false, GLM53_ROLE_MLP_DOWN_SCALE, K3_ST_DTYPE_F32, 2, 32, 96},
    {false, GLM53_ROLE_MLP_GATE, K3_ST_DTYPE_F8_E4M3, 2, 12288, 4096},
    {false, GLM53_ROLE_MLP_GATE_SCALE, K3_ST_DTYPE_F32, 2, 96, 32},
    {false, GLM53_ROLE_MLP_UP, K3_ST_DTYPE_F8_E4M3, 2, 12288, 4096},
    {false, GLM53_ROLE_MLP_UP_SCALE, K3_ST_DTYPE_F32, 2, 96, 32}
};

static_assert(sizeof(kContracts) / sizeof(kContracts[0]) ==
              GLM53_PHASE5C_LAYER0_EARLY_HEAD_DIAGNOSTIC_BINDING_COUNT, "binding count");
static const char *const kBindingNames[] = {
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "model.language_model.layers.0.hc_attn_base",
    "model.language_model.layers.0.hc_attn_scale",
    "model.language_model.layers.0.hc_attn_fn",
    "model.language_model.layers.0.hc_ffn_base",
    "model.language_model.layers.0.hc_ffn_scale",
    "model.language_model.layers.0.hc_ffn_fn",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.post_attention_layernorm.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight_scale_inv",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.layers.0.mlp.up_proj.weight_scale_inv"
};
static_assert(sizeof(kBindingNames) / sizeof(kBindingNames[0]) ==
              GLM53_PHASE5C_LAYER0_EARLY_HEAD_DIAGNOSTIC_BINDING_COUNT, "binding names");

static void set_error(char *error, size_t size, const char *fmt, ...) {
    if (!error || !size) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, size, fmt, ap);
    va_end(ap);
}

static uint64_t dtype_bytes(k3_st_dtype d) {
    return d == K3_ST_DTYPE_F32 ? 4u : 2u; /* FP8 is handled below. */
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static const glm53_static_binding *contract_binding(
        const glm53_static_bindings *b, const binding_contract &c) {
    return c.global ? &b->globals[c.role] : &b->layers[0][c.role];
}

static bool binding_matches(const glm53_static_binding *b,
                            const binding_contract &c, const char *name) {
    if (!b || !b->runtime || !b->runtime->name || !b->device ||
        strcmp(b->runtime->name, name) != 0 || b->runtime->dtype != b->dtype ||
        b->runtime->logical_bytes != b->logical_bytes || b->role != c.role ||
        b->dtype != c.dtype || b->ndim != c.ndim || b->shape[0] != c.a ||
        (c.ndim == 2 && b->shape[1] != c.b)) return false;
    if (b->kind != (c.global ? GLM53_STATIC_BINDING_GLOBAL : GLM53_STATIC_BINDING_LAYER))
        return false;
    if (c.global) {
        if (b->layer != UINT16_MAX || b->expert != UINT16_MAX) return false;
    } else if (b->layer != 0u || b->expert != UINT16_MAX) return false;
    for (unsigned i = c.ndim; i < K3_ST_MAX_DIMS; ++i)
        if (b->shape[i] != 0u) return false;
    uint64_t elements = c.a * (c.ndim == 2 ? c.b : 1u);
    uint64_t bytes = elements * (c.dtype == K3_ST_DTYPE_F8_E4M3 ? 1u : dtype_bytes(c.dtype));
    return b->logical_bytes == bytes;
}

} // namespace

__global__ static void phase5c_zero_bf16_kernel(uint16_t *output) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < GLM53_PHASE5C_LAYER0_HIDDEN) output[i] = 0u;
}

struct glm53_phase5c_session {
    glm53_phase5c_session_state state;
    hipStream_t stream;
    void *workspace;
    glm53_phase5c_workspace_layout layout;
    glm53_static_binding cached[GLM53_PHASE5C_LAYER0_EARLY_HEAD_DIAGNOSTIC_BINDING_COUNT];
    bool diagnostic_available;
    unsigned published_slot;
    uint64_t published_position;
};

extern "C" bool glm53_phase5c_workspace_layout_build(glm53_phase5c_workspace_layout *out) {
    if (!out) return false;
    const uint64_t H = GLM53_PHASE5C_LAYER0_HIDDEN;
    const uint64_t I = GLM53_PHASE5C_LAYER0_INTERMEDIATE;
    const uint64_t V = GLM53_PHASE5C_LAYER0_VOCAB;
    const uint64_t sizes[GLM53_PHASE5C_WS_REGION_COUNT] = {
        H*2, 4*H*2, 4*H*2, 24*4, 4*4, 4*2, 16*2, H*2, H*2, H*2,
        H*4, I*4, I*4, I*4, H*4, I, (I/128)*4, V*2, V*2
    };
    glm53_phase5c_workspace_layout result{};
    result.alignment = GLM53_PHASE5C_WORKSPACE_ALIGNMENT;
    uint64_t cursor = 0;
    for (unsigned i = 0; i < GLM53_PHASE5C_WS_REGION_COUNT; ++i) {
        cursor = align_up(cursor, result.alignment);
        result.regions[i].offset = cursor;
        result.regions[i].bytes = sizes[i];
        result.accounted_bytes += sizes[i];
        cursor += sizes[i];
    }
    result.workspace_bytes = align_up(cursor, result.alignment);
    *out = result;
    return true;
}

extern "C" glm53_phase5c_status
 glm53_phase5c_validate_layer0_early_head_diagnostic_bindings(
        const glm53_static_bindings *bindings, char *error, size_t error_size) {
    if (!bindings) {
        set_error(error, error_size, "bindings is null");
        return GLM53_PHASE5C_INVALID_ARGUMENT;
    }
    if (!bindings->built) {
        set_error(error, error_size, "bindings is not built");
        return GLM53_PHASE5C_BAD_BINDINGS;
    }
    for (unsigned i = 0; i < GLM53_PHASE5C_LAYER0_EARLY_HEAD_DIAGNOSTIC_BINDING_COUNT; ++i) {
        if (!binding_matches(contract_binding(bindings, kContracts[i]), kContracts[i], kBindingNames[i])) {
            set_error(error, error_size, "layer0 early-head diagnostic binding %u violates its exact schema", i);
            return GLM53_PHASE5C_BAD_BINDINGS;
        }
    }
    if (error && error_size) error[0] = '\0';
    return GLM53_PHASE5C_OK;
}

extern "C" glm53_phase5c_status glm53_phase5c_session_create(
        glm53_phase5c_session **out, const glm53_static_bindings *bindings,
        char *error, size_t error_size) {
    if (!out) return GLM53_PHASE5C_INVALID_ARGUMENT;
    glm53_phase5c_status validation =
        glm53_phase5c_validate_layer0_early_head_diagnostic_bindings(bindings, error, error_size);
    if (validation != GLM53_PHASE5C_OK) return validation;
    glm53_phase5c_session *s = (glm53_phase5c_session *)calloc(1, sizeof(*s));
    if (!s) return GLM53_PHASE5C_ALLOCATION_FAILED;
    glm53_phase5c_workspace_layout_build(&s->layout);
    hipError_t h = hipStreamCreateWithFlags(&s->stream, hipStreamNonBlocking);
    if (h != hipSuccess) {
        free(s);
        set_error(error, error_size, "nonblocking stream creation failed: %s", hipGetErrorString(h));
        return GLM53_PHASE5C_ALLOCATION_FAILED;
    }
    h = hipMalloc(&s->workspace, (size_t)s->layout.workspace_bytes);
    if (h != hipSuccess || ((uintptr_t)s->workspace % s->layout.alignment) != 0u) {
        if (s->workspace) { hipError_t ignored = hipFree(s->workspace); (void)ignored; }
        { hipError_t ignored = hipStreamDestroy(s->stream); (void)ignored; }
        free(s);
        set_error(error, error_size, "aligned workspace allocation failed");
        return GLM53_PHASE5C_ALLOCATION_FAILED;
    }
    for (unsigned i = 0; i < GLM53_PHASE5C_LAYER0_EARLY_HEAD_DIAGNOSTIC_BINDING_COUNT; ++i)
        s->cached[i] = *contract_binding(bindings, kContracts[i]);
    s->state = GLM53_PHASE5C_SESSION_READY;
    s->published_slot = 0;
    *out = s;
    if (error && error_size) error[0] = '\0';
    return GLM53_PHASE5C_OK;
}

extern "C" void glm53_phase5c_session_destroy(glm53_phase5c_session *s) {
    if (!s) return;
    if (s->stream) { hipError_t ignored = hipStreamSynchronize(s->stream); (void)ignored; }
    if (s->workspace) { hipError_t ignored = hipFree(s->workspace); (void)ignored; }
    if (s->stream) { hipError_t ignored = hipStreamDestroy(s->stream); (void)ignored; }
    memset(s, 0, sizeof(*s));
    free(s);
}

extern "C" glm53_phase5c_status glm53_phase5c_session_reset(glm53_phase5c_session *s) {
    if (!s) return GLM53_PHASE5C_INVALID_ARGUMENT;
    if (s->state != GLM53_PHASE5C_SESSION_READY) return GLM53_PHASE5C_INVALID_STATE;
    hipError_t h = hipStreamSynchronize(s->stream);
    if (h != hipSuccess) {
        s->state = GLM53_PHASE5C_SESSION_FAULT;
        return GLM53_PHASE5C_DEVICE_SYNC_FAILED;
    }
    s->diagnostic_available = false;
    s->published_position = 0;
    s->published_slot = 0;
    s->state = GLM53_PHASE5C_SESSION_READY;
    return GLM53_PHASE5C_OK;
}

extern "C" glm53_phase5c_session_state glm53_phase5c_session_get_state(
        const glm53_phase5c_session *s) {
    return s ? s->state : GLM53_PHASE5C_SESSION_FAULT;
}

static void *region(glm53_phase5c_session *s, glm53_phase5c_workspace_region r) {
    return (void *)((unsigned char *)s->workspace + s->layout.regions[r].offset);
}

extern "C" bool glm53_phase5c_session_get_layer0_early_head_diagnostic(
        const glm53_phase5c_session *s,
        glm53_phase5c_layer0_early_head_diagnostic *out) {
    if (!s || !out) return false;
    glm53_phase5c_layer0_early_head_diagnostic value{};
    value.available = s->diagnostic_available;
    if (value.available) {
        value.position = s->published_position;
        glm53_phase5c_workspace_region r = s->published_slot == 0
            ? GLM53_PHASE5C_WS_LAYER0_EARLY_HEAD_DIAGNOSTIC_LOGITS_0
            : GLM53_PHASE5C_WS_LAYER0_EARLY_HEAD_DIAGNOSTIC_LOGITS_1;
        value.layer0_early_head_diagnostic_logits =
            (const unsigned char *)s->workspace + s->layout.regions[r].offset;
        value.layer0_early_head_diagnostic_logits_count = GLM53_PHASE5C_LAYER0_VOCAB;
    }
    *out = value;
    return true;
}

static glm53_phase5c_status drain_failure(glm53_phase5c_session *s,
                                          glm53_phase5c_status original) {
    hipError_t h = hipStreamSynchronize(s->stream);
    if (h != hipSuccess) {
        s->state = GLM53_PHASE5C_SESSION_FAULT;
        return GLM53_PHASE5C_DEVICE_SYNC_FAILED;
    }
    s->state = GLM53_PHASE5C_SESSION_READY;
    return original;
}

extern "C" glm53_phase5c_status glm53_phase5c_layer0_early_head_diagnostic_step(
        glm53_phase5c_session *s, uint32_t token, uint64_t position,
        const glm53_phase5c_attention_provider *provider) {
    if (!s) return GLM53_PHASE5C_INVALID_ARGUMENT;
    if (s->state != GLM53_PHASE5C_SESSION_READY) return GLM53_PHASE5C_INVALID_STATE;
    s->state = GLM53_PHASE5C_SESSION_BUSY;
    if (token >= GLM53_PHASE5C_LAYER0_VOCAB)
        return drain_failure(s, GLM53_PHASE5C_INVALID_TOKEN);
    if (!provider || !provider->run)
        return drain_failure(s, GLM53_PHASE5C_INVALID_ARGUMENT);

    void *embedding = region(s, GLM53_PHASE5C_WS_EMBEDDING);
    void *streams_a = region(s, GLM53_PHASE5C_WS_STREAMS_A);
    void *streams_b = region(s, GLM53_PHASE5C_WS_STREAMS_B);
    void *mix = region(s, GLM53_PHASE5C_WS_MHC_MIX);
    void *pre = region(s, GLM53_PHASE5C_WS_MHC_PRE);
    void *post = region(s, GLM53_PHASE5C_WS_MHC_POST);
    void *comb = region(s, GLM53_PHASE5C_WS_MHC_COMB);
    void *collapsed = region(s, GLM53_PHASE5C_WS_COLLAPSED);
    void *normalized = region(s, GLM53_PHASE5C_WS_NORMALIZED);
    void *branch = region(s, GLM53_PHASE5C_WS_BRANCH);
    const size_t H = GLM53_PHASE5C_LAYER0_HIDDEN;
    const size_t I = GLM53_PHASE5C_LAYER0_INTERMEDIATE;

    const unsigned char *table = (const unsigned char *)s->cached[C_EMBED].device;
    hipError_t h = hipMemcpyAsync(embedding, table + (size_t)token * H * 2u,
                                  H * 2u, hipMemcpyDeviceToDevice, s->stream);
    bool ok = h == hipSuccess;
    if (ok) ok = glm53_mhc_replicate_bf16(streams_a, 4*H, embedding, H, H, s->stream);
    if (ok) ok = glm53_mhc_prepare_bf16(
        mix,24,pre,4,post,4,comb,16,collapsed,H,streams_a,4*H,
        s->cached[C_HC_ATTN_FN].device,24*4*H,
        s->cached[C_HC_ATTN_BASE].device,24,
        s->cached[C_HC_ATTN_SCALE].device,3,H,s->stream);
    if (ok) ok = k3_rocm_rms_norm_bf16(normalized, collapsed,
        s->cached[C_INPUT_NORM].device,1,H,1.0e-5f,s->stream);
    if (!ok) return drain_failure(s, GLM53_PHASE5C_LAUNCH_FAILED);
    if (!provider->run(branch,H,normalized,H,position,s->stream,provider->context))
        return drain_failure(s, GLM53_PHASE5C_PROVIDER_FAILED);
    ok = glm53_mhc_expand_bf16(streams_b,4*H,branch,H,streams_a,4*H,post,4,comb,16,H,s->stream);
    if (ok) ok = glm53_mhc_prepare_bf16(
        mix,24,pre,4,post,4,comb,16,collapsed,H,streams_b,4*H,
        s->cached[C_HC_FFN_FN].device,24*4*H,
        s->cached[C_HC_FFN_BASE].device,24,
        s->cached[C_HC_FFN_SCALE].device,3,H,s->stream);
    if (ok) ok = k3_rocm_rms_norm_bf16(normalized, collapsed,
        s->cached[C_POST_ATTN_NORM].device,1,H,1.0e-5f,s->stream);

    glm53_dense_fp8_matrix gate = {s->cached[C_GATE].device,
        (size_t)s->cached[C_GATE].logical_bytes,
        s->cached[C_GATE_SCALE].device,(size_t)(s->cached[C_GATE_SCALE].logical_bytes/4)};
    glm53_dense_fp8_matrix up = {s->cached[C_UP].device,
        (size_t)s->cached[C_UP].logical_bytes,
        s->cached[C_UP_SCALE].device,(size_t)(s->cached[C_UP_SCALE].logical_bytes/4)};
    glm53_dense_fp8_matrix down = {s->cached[C_DOWN].device,
        (size_t)s->cached[C_DOWN].logical_bytes,
        s->cached[C_DOWN_SCALE].device,(size_t)(s->cached[C_DOWN_SCALE].logical_bytes/4)};
    glm53_dense_scratch scratch = {
        region(s,GLM53_PHASE5C_WS_DENSE_INPUT_F32),H,
        region(s,GLM53_PHASE5C_WS_DENSE_GATE_F32),I,
        region(s,GLM53_PHASE5C_WS_DENSE_UP_F32),I,
        region(s,GLM53_PHASE5C_WS_DENSE_ACTIVATION_F32),I,
        region(s,GLM53_PHASE5C_WS_DENSE_DOWN_F32),H,
        region(s,GLM53_PHASE5C_WS_DENSE_Q8),I,
        region(s,GLM53_PHASE5C_WS_DENSE_DYNAMIC_SCALE),I/128
    };
    if (ok) ok = glm53_dense_mlp_fp8_bf16(branch,H,normalized,H,&gate,&up,&down,
        H,I,10.0f,&scratch,s->stream);
    if (ok) ok = glm53_mhc_expand_bf16(streams_a,4*H,branch,H,streams_b,4*H,
                                        post,4,comb,16,H,s->stream);
    if (ok) ok = glm53_mhc_hyper_mean_bf16(collapsed,H,streams_a,4*H,H,s->stream);
    if (ok) ok = k3_rocm_rms_norm_bf16(normalized,collapsed,
        s->cached[C_FINAL_NORM].device,1,H,1.0e-5f,s->stream);
    unsigned next_slot = s->diagnostic_available ? 1u - s->published_slot : 0u;
    void *diagnostic_logits = region(s, next_slot == 0
        ? GLM53_PHASE5C_WS_LAYER0_EARLY_HEAD_DIAGNOSTIC_LOGITS_0
        : GLM53_PHASE5C_WS_LAYER0_EARLY_HEAD_DIAGNOSTIC_LOGITS_1);
    if (ok) ok = k3_rocm_bf16_gemv_bf16(diagnostic_logits,
        s->cached[C_LM_HEAD].device,normalized,
        GLM53_PHASE5C_LAYER0_VOCAB,H,s->stream);
    if (!ok) return drain_failure(s, GLM53_PHASE5C_LAUNCH_FAILED);

    h = hipStreamSynchronize(s->stream); /* the successful step's sole sync */
    if (h != hipSuccess) {
        s->state = GLM53_PHASE5C_SESSION_FAULT;
        return GLM53_PHASE5C_DEVICE_SYNC_FAILED;
    }
    s->published_slot = next_slot;
    s->published_position = position;
    s->diagnostic_available = true;
    s->state = GLM53_PHASE5C_SESSION_READY;
    return GLM53_PHASE5C_OK;
}

extern "C" bool glm53_phase5c_zero_attention_provider(
        void *output, size_t output_count, const void *input, size_t input_count,
        uint64_t position, void *stream, void *context) {
    (void)position;
    (void)context;
    if (!output || !input || output_count < GLM53_PHASE5C_LAYER0_HIDDEN ||
        input_count < GLM53_PHASE5C_LAYER0_HIDDEN || output == input) return false;
    (void)hipGetLastError();
    hipLaunchKernelGGL(phase5c_zero_bf16_kernel, dim3(16), dim3(256), 0,
                       (hipStream_t)stream, (uint16_t *)output);
    return hipGetLastError() == hipSuccess;
}

extern "C" const char *glm53_phase5c_status_string(glm53_phase5c_status s) {
    switch (s) {
    case GLM53_PHASE5C_OK:return "ok";
    case GLM53_PHASE5C_INVALID_ARGUMENT:return "invalid argument";
    case GLM53_PHASE5C_BAD_BINDINGS:return "bad bindings";
    case GLM53_PHASE5C_ALLOCATION_FAILED:return "allocation failed";
    case GLM53_PHASE5C_INVALID_STATE:return "invalid state";
    case GLM53_PHASE5C_INVALID_TOKEN:return "invalid token";
    case GLM53_PHASE5C_LAUNCH_FAILED:return "launch failed";
    case GLM53_PHASE5C_PROVIDER_FAILED:return "provider failed";
    case GLM53_PHASE5C_DEVICE_SYNC_FAILED:return "device sync failed";
    default:return "unknown";
    }
}
