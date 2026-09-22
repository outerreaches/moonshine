#include "mimo26_architecture.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static k3_st_tensor tensor(const char *name, k3_st_dtype dtype, uint8_t ndim,
                           uint64_t a, uint64_t b)
{
    k3_st_tensor t;
    memset(&t, 0, sizeof t);
    t.name = (char *)name;
    t.dtype = dtype;
    t.ndim = ndim;
    t.shape[0] = a;
    t.shape[1] = b;
    return t;
}

int main(void)
{
    /* Layer kinds follow config's explicit pattern, which is not periodic. */
    assert(mimo26_architecture_layer_kind(0) == MIMO26_LAYER_DENSE_GLOBAL);
    assert(mimo26_architecture_layer_kind(1) == MIMO26_LAYER_MOE_SWA);
    assert(mimo26_architecture_layer_kind(5) == MIMO26_LAYER_MOE_GLOBAL);
    assert(mimo26_architecture_layer_kind(11) == MIMO26_LAYER_MOE_GLOBAL);
    assert(mimo26_architecture_layer_kind(17) == MIMO26_LAYER_MOE_GLOBAL);
    assert(mimo26_architecture_layer_kind(47) == MIMO26_LAYER_MOE_GLOBAL);
    assert(mimo26_architecture_layer_kind(48) == MIMO26_LAYER_INVALID);
    /* A period-12 reading would wrongly call these full attention. */
    assert(mimo26_architecture_layer_kind(12) == MIMO26_LAYER_MOE_SWA);
    assert(mimo26_architecture_layer_kind(24) == MIMO26_LAYER_MOE_SWA);
    assert(mimo26_architecture_layer_kind(36) == MIMO26_LAYER_MOE_SWA);
    assert(mimo26_architecture_layer_is_swa(12));
    assert(!mimo26_architecture_layer_is_swa(11));

    size_t swa = 0;
    for (uint32_t layer = 0; layer < MIMO26_TEXT_LAYER_COUNT; layer++) {
        if (mimo26_architecture_layer_is_swa(layer)) {
            swa++;
        }
    }
    assert(swa == 39);

    /* Routed expert contracts are packed byte extents, not logical shapes. */
    k3_st_tensor t = tensor("model.layers.1.mlp.experts.0.gate_proj.weight",
                            K3_ST_DTYPE_U8, 2, 2048, 2048);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.shape[1] = 4096; /* logical width, not the packed extent */
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t = tensor("model.layers.47.mlp.experts.255.down_proj.weight",
               K3_ST_DTYPE_U8, 2, 4096, 1024);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.dtype = K3_ST_DTYPE_F8_E4M3; /* storage is MXFP4 in U8, not FP8 */
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t = tensor("model.layers.1.mlp.experts.0.down_proj.weight_scale",
               K3_ST_DTYPE_U8, 2, 4096, 64);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.layers.1.mlp.experts.0.down_proj.weight_scale_inv";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    /* Expert index out of range, and the dense layer has no experts. */
    t = tensor("model.layers.1.mlp.experts.256.gate_proj.weight",
               K3_ST_DTYPE_U8, 2, 2048, 2048);
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.layers.0.mlp.experts.0.gate_proj.weight";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));

    /* Global QKV keeps its over-provisioned [108,32] scale grid: 108, not the
     * 106 an exact ceil(13568/128) would demand. */
    t = tensor("model.layers.5.self_attn.qkv_proj.weight_scale_inv",
               K3_ST_DTYPE_F32, 2, 108, 32);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.shape[0] = 106;
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t.shape[0] = 116; /* the SWA grid does not belong on a global layer */
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t = tensor("model.layers.1.self_attn.qkv_proj.weight_scale_inv",
               K3_ST_DTYPE_F32, 2, 116, 32);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));

    /* Fused QKV widths are per-kind and must not be interchangeable. */
    mimo26_layer_kind kind = MIMO26_LAYER_INVALID;
    t = tensor("model.layers.5.self_attn.qkv_proj.weight",
               K3_ST_DTYPE_F8_E4M3, 2, 13568, 4096);
    assert(mimo26_architecture_validate_main_tensor(&t, &kind));
    assert(kind == MIMO26_LAYER_MOE_GLOBAL);
    t.shape[0] = 14848;
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t = tensor("model.layers.1.self_attn.qkv_proj.weight",
               K3_ST_DTYPE_F8_E4M3, 2, 14848, 4096);
    assert(mimo26_architecture_validate_main_tensor(&t, &kind));
    assert(kind == MIMO26_LAYER_MOE_SWA);

    /* The learned sink exists only on sliding-window layers. */
    t = tensor("model.layers.1.self_attn.attention_sink_bias",
               K3_ST_DTYPE_BF16, 1, 64, 0);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.layers.5.self_attn.attention_sink_bias";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));

    /* Dense MLP belongs to layer 0 only; the router belongs to layers 1..47. */
    t = tensor("model.layers.0.mlp.gate_proj.weight", K3_ST_DTYPE_F8_E4M3, 2,
               16384, 4096);
    assert(mimo26_architecture_validate_main_tensor(&t, &kind));
    assert(kind == MIMO26_LAYER_DENSE_GLOBAL);
    t.name = "model.layers.1.mlp.gate_proj.weight";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t = tensor("model.layers.1.mlp.gate.weight", K3_ST_DTYPE_BF16, 2, 256,
               4096);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.layers.0.mlp.gate.weight";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));

    /* Untied embedding and head, vocab 152576. */
    t = tensor("model.embed_tokens.weight", K3_ST_DTYPE_BF16, 2, 152576, 4096);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "lm_head.weight";
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.shape[0] = 151936;
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));

    /* Non-canonical indices are rejected, never normalized. */
    t = tensor("model.layers.01.input_layernorm.weight", K3_ST_DTYPE_BF16, 1,
               4096, 0);
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.layers.048.input_layernorm.weight";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.layers.48.input_layernorm.weight";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.layers.1.unknown.weight";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.layers..input_layernorm.weight";
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    /* Extra trailing dimensions must not be ignored. */
    t = tensor("model.layers.1.input_layernorm.weight", K3_ST_DTYPE_BF16, 1,
               4096, 0);
    assert(mimo26_architecture_validate_main_tensor(&t, NULL));
    t.shape[1] = 1;
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));

    /* MTP is a separate namespace numbered 0..2 -- it does not continue the
     * text layer numbering, unlike GLM's. Never accepted as text. */
    t = tensor("model.mtp.layers.0.self_attn.qkv_proj.weight",
               K3_ST_DTYPE_F8_E4M3, 2, 14848, 4096);
    assert(mimo26_architecture_validate_mtp_tensor(&t));
    assert(!mimo26_architecture_validate_main_tensor(&t, NULL));
    t.name = "model.mtp.layers.2.self_attn.qkv_proj.weight";
    assert(mimo26_architecture_validate_mtp_tensor(&t));
    t.name = "model.mtp.layers.3.self_attn.qkv_proj.weight";
    assert(!mimo26_architecture_validate_mtp_tensor(&t));
    t.name = "model.mtp.layers.48.self_attn.qkv_proj.weight";
    assert(!mimo26_architecture_validate_mtp_tensor(&t));
    t.name = "model.mtp.layers.00.self_attn.qkv_proj.weight";
    assert(!mimo26_architecture_validate_mtp_tensor(&t));
    t = tensor("model.mtp.layers.0.eh_proj.weight", K3_ST_DTYPE_BF16, 2, 4096,
               8192);
    assert(mimo26_architecture_validate_mtp_tensor(&t));
    t.shape[1] = 4096;
    assert(!mimo26_architecture_validate_mtp_tensor(&t));
    t = tensor("model.layers.1.input_layernorm.weight", K3_ST_DTYPE_BF16, 1,
               4096, 0);
    assert(!mimo26_architecture_validate_mtp_tensor(&t));

    /* Empty metadata fails and leaves the report zeroed. */
    k3_st_model empty;
    memset(&empty, 0, sizeof empty);
    mimo26_architecture_report report;
    memset(&report, 0x7f, sizeof report);
    char error[192];
    assert(!mimo26_architecture_validate_main(&empty, &report, error,
                                              sizeof error));
    mimo26_architecture_report zero = {0};
    assert(memcmp(&report, &zero, sizeof zero) == 0);
    memset(&report, 0x7f, sizeof report);
    assert(!mimo26_architecture_validate(&empty, &report, error, sizeof error));
    assert(memcmp(&report, &zero, sizeof zero) == 0);

    /* A short but well-formed model still fails the exact counts. */
    k3_st_tensor one = tensor("model.norm.weight", K3_ST_DTYPE_BF16, 1, 4096, 0);
    k3_st_model tiny;
    memset(&tiny, 0, sizeof tiny);
    tiny.tensors = &one;
    tiny.tensor_count = 1;
    memset(&report, 0x7f, sizeof report);
    assert(!mimo26_architecture_validate_main(&tiny, &report, error,
                                              sizeof error));
    assert(strstr(error, "main tensor count") != NULL);
    assert(memcmp(&report, &zero, sizeof zero) == 0);

    /* An unnamed tensor is reported rather than skipped. */
    k3_st_tensor unnamed;
    memset(&unnamed, 0, sizeof unnamed);
    tiny.tensors = &unnamed;
    assert(!mimo26_architecture_validate_main(&tiny, &report, error,
                                              sizeof error));
    assert(strstr(error, "no name") != NULL);

    printf("test_mimo26_architecture: ok\n");
    return 0;
}
