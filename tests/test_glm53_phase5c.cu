#include "glm53_phase5c.h"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t bytes(k3_st_dtype d, uint64_t a, uint64_t b = 1) {
    return a * b * (d == K3_ST_DTYPE_F8_E4M3 ? 1u : d == K3_ST_DTYPE_F32 ? 4u : 2u);
}

static void set_binding(glm53_static_binding &x, bool global, unsigned role,
                        k3_st_dtype dtype, uint64_t a, uint64_t b,
                        uintptr_t address) {
    std::memset(&x, 0, sizeof(x));
    x.device = (void *)address;
    x.kind = global ? GLM53_STATIC_BINDING_GLOBAL : GLM53_STATIC_BINDING_LAYER;
    x.dtype = dtype;
    x.ndim = b ? 2 : 1;
    x.shape[0] = a;
    x.shape[1] = b;
    x.logical_bytes = bytes(dtype, a, b ? b : 1);
    x.layer = global ? UINT16_MAX : 0;
    x.expert = UINT16_MAX;
    x.role = (uint16_t)role;
}

static glm53_static_bindings valid_bindings() {
    glm53_static_bindings b{};
    b.built = true;
    uintptr_t p = 0x100000;
#define G(role,d,a,z) set_binding(b.globals[role],true,role,d,a,z,p+=0x1000)
#define L(role,d,a,z) set_binding(b.layers[0][role],false,role,d,a,z,p+=0x1000)
    G(GLM53_GLOBAL_LM_HEAD,K3_ST_DTYPE_BF16,154880,4096);
    G(GLM53_GLOBAL_EMBED_TOKENS,K3_ST_DTYPE_BF16,154880,4096);
    G(GLM53_GLOBAL_FINAL_NORM,K3_ST_DTYPE_BF16,4096,0);
    L(GLM53_ROLE_HC_ATTN_BASE,K3_ST_DTYPE_F32,24,0);
    L(GLM53_ROLE_HC_ATTN_SCALE,K3_ST_DTYPE_F32,3,0);
    L(GLM53_ROLE_HC_ATTN_FN,K3_ST_DTYPE_BF16,24,16384);
    L(GLM53_ROLE_HC_FFN_BASE,K3_ST_DTYPE_F32,24,0);
    L(GLM53_ROLE_HC_FFN_SCALE,K3_ST_DTYPE_F32,3,0);
    L(GLM53_ROLE_HC_FFN_FN,K3_ST_DTYPE_BF16,24,16384);
    L(GLM53_ROLE_INPUT_NORM,K3_ST_DTYPE_BF16,4096,0);
    L(GLM53_ROLE_POST_ATTN_NORM,K3_ST_DTYPE_BF16,4096,0);
    L(GLM53_ROLE_MLP_DOWN,K3_ST_DTYPE_F8_E4M3,4096,12288);
    L(GLM53_ROLE_MLP_DOWN_SCALE,K3_ST_DTYPE_F32,32,96);
    L(GLM53_ROLE_MLP_GATE,K3_ST_DTYPE_F8_E4M3,12288,4096);
    L(GLM53_ROLE_MLP_GATE_SCALE,K3_ST_DTYPE_F32,96,32);
    L(GLM53_ROLE_MLP_UP,K3_ST_DTYPE_F8_E4M3,12288,4096);
    L(GLM53_ROLE_MLP_UP_SCALE,K3_ST_DTYPE_F32,96,32);
#undef G
#undef L
    static const char *names[] = {
        "lm_head.weight", "model.language_model.embed_tokens.weight",
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
    glm53_static_binding *selected[] = {
        &b.globals[GLM53_GLOBAL_LM_HEAD], &b.globals[GLM53_GLOBAL_EMBED_TOKENS],
        &b.globals[GLM53_GLOBAL_FINAL_NORM],
        &b.layers[0][GLM53_ROLE_HC_ATTN_BASE], &b.layers[0][GLM53_ROLE_HC_ATTN_SCALE],
        &b.layers[0][GLM53_ROLE_HC_ATTN_FN], &b.layers[0][GLM53_ROLE_HC_FFN_BASE],
        &b.layers[0][GLM53_ROLE_HC_FFN_SCALE], &b.layers[0][GLM53_ROLE_HC_FFN_FN],
        &b.layers[0][GLM53_ROLE_INPUT_NORM], &b.layers[0][GLM53_ROLE_POST_ATTN_NORM],
        &b.layers[0][GLM53_ROLE_MLP_DOWN], &b.layers[0][GLM53_ROLE_MLP_DOWN_SCALE],
        &b.layers[0][GLM53_ROLE_MLP_GATE], &b.layers[0][GLM53_ROLE_MLP_GATE_SCALE],
        &b.layers[0][GLM53_ROLE_MLP_UP], &b.layers[0][GLM53_ROLE_MLP_UP_SCALE]
    };
    static glm53_static_runtime_entry runtime[17];
    for (unsigned i = 0; i < 17; ++i) {
        runtime[i].name = const_cast<char *>(names[i]);
        runtime[i].dtype = selected[i]->dtype;
        runtime[i].logical_bytes = selected[i]->logical_bytes;
        selected[i]->runtime = &runtime[i];
    }
    return b;
}

static void test_layout_ledger() {
    glm53_phase5c_workspace_layout l{};
    assert(glm53_phase5c_workspace_layout_build(&l));
    assert(l.alignment == 256);
    assert(l.accounted_bytes == 910872);
    assert(l.workspace_bytes == 911872);
    uint64_t accounted = 0;
    uint64_t prior_end = 0;
    for (unsigned i = 0; i < GLM53_PHASE5C_WS_REGION_COUNT; ++i) {
        assert(l.regions[i].offset % l.alignment == 0);
        assert(l.regions[i].bytes != 0);
        assert(l.regions[i].offset >= prior_end);
        prior_end = l.regions[i].offset + l.regions[i].bytes;
        accounted += l.regions[i].bytes;
    }
    assert(accounted == l.accounted_bytes);
    assert(prior_end <= l.workspace_bytes);
    assert(l.regions[GLM53_PHASE5C_WS_LAYER0_EARLY_HEAD_DIAGNOSTIC_LOGITS_0].bytes == 154880u*2u);
}

static void test_binding_schema() {
    char error[160];
    glm53_static_bindings b = valid_bindings();
    assert(glm53_phase5c_validate_layer0_early_head_diagnostic_bindings(&b,error,sizeof(error)) == GLM53_PHASE5C_OK);
    b.layers[0][GLM53_ROLE_MLP_GATE_SCALE].shape[1] = 31;
    assert(glm53_phase5c_validate_layer0_early_head_diagnostic_bindings(&b,error,sizeof(error)) == GLM53_PHASE5C_BAD_BINDINGS);
    assert(error[0]);
    b = valid_bindings();
    b.globals[GLM53_GLOBAL_FINAL_NORM].dtype = K3_ST_DTYPE_F32;
    assert(glm53_phase5c_validate_layer0_early_head_diagnostic_bindings(&b,nullptr,0) == GLM53_PHASE5C_BAD_BINDINGS);
    b = valid_bindings();
    b.layers[0][GLM53_ROLE_HC_FFN_BASE].runtime = b.layers[0][GLM53_ROLE_HC_ATTN_BASE].runtime;
    assert(glm53_phase5c_validate_layer0_early_head_diagnostic_bindings(&b,nullptr,0) == GLM53_PHASE5C_BAD_BINDINGS);
    b = valid_bindings(); b.built = false;
    assert(glm53_phase5c_validate_layer0_early_head_diagnostic_bindings(&b,nullptr,0) == GLM53_PHASE5C_BAD_BINDINGS);
}

static void test_zero_provider() {
    void *input = nullptr, *output = nullptr;
    hipStream_t stream = nullptr;
    assert(hipMalloc(&input, 4096u*2u) == hipSuccess);
    assert(hipMalloc(&output, 4096u*2u) == hipSuccess);
    assert(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess);
    assert(hipMemset(output, 0x7f, 4096u*2u) == hipSuccess);
    assert(glm53_phase5c_zero_attention_provider(output,4096,input,4096,77,stream,nullptr));
    assert(hipStreamSynchronize(stream) == hipSuccess);
    std::vector<unsigned char> host(4096u*2u, 1);
    assert(hipMemcpy(host.data(),output,host.size(),hipMemcpyDeviceToHost) == hipSuccess);
    for (unsigned char v : host) assert(v == 0);
    assert(!glm53_phase5c_zero_attention_provider(output,4095,input,4096,0,stream,nullptr));
    assert(!glm53_phase5c_zero_attention_provider(input,4096,input,4096,0,stream,nullptr));
    assert(hipStreamDestroy(stream) == hipSuccess);
    assert(hipFree(output) == hipSuccess);
    assert(hipFree(input) == hipSuccess);
}

static void test_invalid_token_and_states_without_payload() {
    glm53_static_bindings b = valid_bindings();
    glm53_phase5c_session *s = nullptr;
    char error[160];
    assert(glm53_phase5c_session_create(&s,&b,error,sizeof(error)) == GLM53_PHASE5C_OK);
    assert(s);
    assert(glm53_phase5c_session_get_state(s) == GLM53_PHASE5C_SESSION_READY);
    glm53_phase5c_layer0_early_head_diagnostic d{};
    assert(glm53_phase5c_session_get_layer0_early_head_diagnostic(s,&d));
    assert(!d.available);
    glm53_phase5c_attention_provider zero{glm53_phase5c_zero_attention_provider,nullptr};
    assert(glm53_phase5c_layer0_early_head_diagnostic_step(s,154880,5,&zero) == GLM53_PHASE5C_INVALID_TOKEN);
    assert(glm53_phase5c_session_get_state(s) == GLM53_PHASE5C_SESSION_READY);
    assert(glm53_phase5c_session_get_layer0_early_head_diagnostic(s,&d) && !d.available);
    assert(glm53_phase5c_layer0_early_head_diagnostic_step(s,0,5,nullptr) == GLM53_PHASE5C_INVALID_ARGUMENT);
    assert(glm53_phase5c_session_get_state(s) == GLM53_PHASE5C_SESSION_READY);
    assert(glm53_phase5c_session_reset(s) == GLM53_PHASE5C_OK);
    assert(glm53_phase5c_session_get_layer0_early_head_diagnostic(s,&d) && !d.available);
    assert(glm53_phase5c_session_get_state(s) == GLM53_PHASE5C_SESSION_READY);
    glm53_phase5c_session_destroy(s);
    assert(glm53_phase5c_layer0_early_head_diagnostic_step(nullptr,0,0,&zero) == GLM53_PHASE5C_INVALID_ARGUMENT);
}

int main() {
    test_layout_ledger();
    test_binding_schema();
    test_zero_provider();
    test_invalid_token_and_states_without_payload();
    std::puts("glm53 Phase5C layer0 early-head diagnostic model-free: PASS");
    return 0;
}
