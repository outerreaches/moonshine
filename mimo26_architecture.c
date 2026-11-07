#include "mimo26_architecture.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *suffix;
    k3_st_dtype dtype;
    uint8_t ndim;
    uint64_t shape[4];
} contract;

#define C1(n,d,a) {n,d,1,{a,0,0,0}}
#define C2(n,d,a,b) {n,d,2,{a,b,0,0}}

static const contract globals[] = {
 C2("lm_head.weight",K3_ST_DTYPE_BF16,MIMO26_VOCAB_SIZE,MIMO26_HIDDEN_SIZE),
 C2("model.embed_tokens.weight",K3_ST_DTYPE_BF16,MIMO26_VOCAB_SIZE,MIMO26_HIDDEN_SIZE),
 C1("model.norm.weight",K3_ST_DTYPE_BF16,MIMO26_HIDDEN_SIZE),
};

/* Present on every text layer regardless of kind. */
static const contract common[] = {
 C1("input_layernorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("post_attention_layernorm.weight",K3_ST_DTYPE_BF16,4096),
};

/* Layer 0 only. intermediate_size is 16384. */
static const contract dense_mlp[] = {
 C2("mlp.down_proj.weight",K3_ST_DTYPE_F8_E4M3,4096,16384),
 C2("mlp.down_proj.weight_scale_inv",K3_ST_DTYPE_F32,32,128),
 C2("mlp.gate_proj.weight",K3_ST_DTYPE_F8_E4M3,16384,4096),
 C2("mlp.gate_proj.weight_scale_inv",K3_ST_DTYPE_F32,128,32),
 C2("mlp.up_proj.weight",K3_ST_DTYPE_F8_E4M3,16384,4096),
 C2("mlp.up_proj.weight_scale_inv",K3_ST_DTYPE_F32,128,32),
};

/* Layers 1..47. One group, top-8, no shared expert. */
static const contract moe_gate[] = {
 C1("mlp.gate.e_score_correction_bias",K3_ST_DTYPE_F32,MIMO26_ROUTED_EXPERTS_PER_LAYER),
 C2("mlp.gate.weight",K3_ST_DTYPE_BF16,MIMO26_ROUTED_EXPERTS_PER_LAYER,MIMO26_HIDDEN_SIZE),
};

/*
 * Routed experts are U8-packed MXFP4, two codes per byte, with U8 E8M0 scales
 * per 32-element block. Shapes are the packed byte extents, not logical
 * parameter counts: gate/up are [2048,4096] logical, down is [4096,2048].
 */
static const contract expert_proj[] = {
 C2("down_proj.weight",K3_ST_DTYPE_U8,4096,1024),
 C2("down_proj.weight_scale",K3_ST_DTYPE_U8,4096,64),
 C2("gate_proj.weight",K3_ST_DTYPE_U8,2048,2048),
 C2("gate_proj.weight_scale",K3_ST_DTYPE_U8,2048,128),
 C2("up_proj.weight",K3_ST_DTYPE_U8,2048,2048),
 C2("up_proj.weight_scale",K3_ST_DTYPE_U8,2048,128),
};

/*
 * Fused QKV. Global is 64*192 + 4*192 + 4*128 = 13568; SWA is
 * 64*192 + 8*192 + 8*128 = 14848. o_proj input is 64*128 = 8192 and is BF16,
 * being listed in quantization_config.ignored_layers.
 *
 * The global FP8 scale grid is [108,32], but ceil(13568/128) is 106. The two
 * extra block rows hold live values: the grid was built for a 13824-row layout
 * (V sized as if head_dim were 192) and the weights were later sliced. Flat
 * row/128 indexing stays correct because every Q/K/V segment boundary lands on
 * a block boundary. Requiring an exact ceil() grid here rejects the real
 * checkpoint, so 108 is the contract.
 */
static const contract attn_global[] = {
 C2("self_attn.o_proj.weight",K3_ST_DTYPE_BF16,4096,8192),
 C2("self_attn.qkv_proj.weight",K3_ST_DTYPE_F8_E4M3,13568,4096),
 C2("self_attn.qkv_proj.weight_scale_inv",K3_ST_DTYPE_F32,108,32),
};
static const contract attn_swa[] = {
 C1("self_attn.attention_sink_bias",K3_ST_DTYPE_BF16,64),
 C2("self_attn.o_proj.weight",K3_ST_DTYPE_BF16,4096,8192),
 C2("self_attn.qkv_proj.weight",K3_ST_DTYPE_F8_E4M3,14848,4096),
 C2("self_attn.qkv_proj.weight_scale_inv",K3_ST_DTYPE_F32,116,32),
};

/* Three indexed MTP layers. Attention is SWA-shaped and carries a sink. */
static const contract mtp[] = {
 C2("eh_proj.weight",K3_ST_DTYPE_BF16,4096,8192),
 C1("enorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("final_layernorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("hnorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("input_layernorm.weight",K3_ST_DTYPE_BF16,4096),
 C2("mlp.down_proj.weight",K3_ST_DTYPE_F8_E4M3,4096,16384),
 C2("mlp.down_proj.weight_scale_inv",K3_ST_DTYPE_F32,32,128),
 C2("mlp.gate_proj.weight",K3_ST_DTYPE_F8_E4M3,16384,4096),
 C2("mlp.gate_proj.weight_scale_inv",K3_ST_DTYPE_F32,128,32),
 C2("mlp.up_proj.weight",K3_ST_DTYPE_F8_E4M3,16384,4096),
 C2("mlp.up_proj.weight_scale_inv",K3_ST_DTYPE_F32,128,32),
 C1("pre_mlp_layernorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("self_attn.attention_sink_bias",K3_ST_DTYPE_BF16,64),
 C2("self_attn.o_proj.weight",K3_ST_DTYPE_BF16,4096,8192),
 C2("self_attn.qkv_proj.weight",K3_ST_DTYPE_F8_E4M3,14848,4096),
 C2("self_attn.qkv_proj.weight_scale_inv",K3_ST_DTYPE_F32,116,32),
};

/*
 * MTP layers are numbered 0..2 in their own namespace -- they do NOT continue
 * the text layer numbering the way GLM's do. Assuming continuation rejects
 * every MTP tensor in the real checkpoint.
 */
#define MIMO26_MTP_LAYER_COUNT 3u

/* config hybrid_layer_pattern: 1 is sliding window, 0 is full attention. */
static const uint8_t swa_pattern[MIMO26_TEXT_LAYER_COUNT] = {
 0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,
 1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,
};

static bool fail(char *error, size_t size, const char *format, ...)
{
    if (error != NULL && size > 0) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return false;
}

/* Strict decimal: no sign, no leading zero unless the value is exactly "0". */
static bool parse_index(const char *text, const char **end, uint32_t limit,
                        uint32_t *value)
{
    if (text[0] < '0' || text[0] > '9') {
        return false;
    }
    if (text[0] == '0' && text[1] >= '0' && text[1] <= '9') {
        return false;
    }
    uint64_t parsed = 0;
    const char *cursor = text;
    while (*cursor >= '0' && *cursor <= '9') {
        parsed = parsed * 10u + (uint64_t)(*cursor - '0');
        if (parsed > limit) {
            return false;
        }
        cursor++;
    }
    *value = (uint32_t)parsed;
    *end = cursor;
    return true;
}

static bool matches(const k3_st_tensor *tensor, const contract *entry)
{
    if (tensor->dtype != entry->dtype || tensor->ndim != entry->ndim) {
        return false;
    }
    for (uint8_t dim = 0; dim < entry->ndim; dim++) {
        if (tensor->shape[dim] != entry->shape[dim]) {
            return false;
        }
    }
    for (uint8_t dim = entry->ndim; dim < K3_ST_MAX_DIMS; dim++) {
        if (tensor->shape[dim] != 0) {
            return false;
        }
    }
    return true;
}

static const contract *lookup(const contract *table, size_t count,
                              const char *suffix)
{
    for (size_t index = 0; index < count; index++) {
        if (strcmp(table[index].suffix, suffix) == 0) {
            return &table[index];
        }
    }
    return NULL;
}

mimo26_layer_kind mimo26_architecture_layer_kind(uint32_t layer)
{
    if (layer >= MIMO26_TEXT_LAYER_COUNT) {
        return MIMO26_LAYER_INVALID;
    }
    if (layer == 0) {
        return MIMO26_LAYER_DENSE_GLOBAL;
    }
    return swa_pattern[layer] ? MIMO26_LAYER_MOE_SWA : MIMO26_LAYER_MOE_GLOBAL;
}

bool mimo26_architecture_layer_is_swa(uint32_t layer)
{
    return mimo26_architecture_layer_kind(layer) == MIMO26_LAYER_MOE_SWA;
}

bool mimo26_architecture_validate_main_tensor(const k3_st_tensor *tensor,
                                              mimo26_layer_kind *kind)
{
    if (kind != NULL) {
        *kind = MIMO26_LAYER_INVALID;
    }
    if (tensor == NULL || tensor->name == NULL) {
        return false;
    }

    const contract *entry = lookup(globals, sizeof globals / sizeof *globals,
                                   tensor->name);
    if (entry != NULL) {
        return matches(tensor, entry);
    }

    static const char prefix[] = "model.layers.";
    if (strncmp(tensor->name, prefix, sizeof prefix - 1) != 0) {
        return false;
    }
    const char *cursor = tensor->name + sizeof prefix - 1;
    uint32_t layer = 0;
    if (!parse_index(cursor, &cursor, MIMO26_TEXT_LAYER_COUNT - 1u, &layer)) {
        return false;
    }
    if (*cursor != '.') {
        return false;
    }
    cursor++;

    const mimo26_layer_kind layer_kind = mimo26_architecture_layer_kind(layer);
    if (layer_kind == MIMO26_LAYER_INVALID) {
        return false;
    }

    /* Routed experts carry their own index before the projection suffix. */
    static const char experts[] = "mlp.experts.";
    if (strncmp(cursor, experts, sizeof experts - 1) == 0) {
        if (layer_kind == MIMO26_LAYER_DENSE_GLOBAL) {
            return false;
        }
        cursor += sizeof experts - 1;
        uint32_t expert = 0;
        if (!parse_index(cursor, &cursor,
                         MIMO26_ROUTED_EXPERTS_PER_LAYER - 1u, &expert)) {
            return false;
        }
        if (*cursor != '.') {
            return false;
        }
        cursor++;
        entry = lookup(expert_proj, sizeof expert_proj / sizeof *expert_proj,
                       cursor);
        if (entry == NULL || !matches(tensor, entry)) {
            return false;
        }
        if (kind != NULL) {
            *kind = layer_kind;
        }
        return true;
    }

    entry = lookup(common, sizeof common / sizeof *common, cursor);
    if (entry == NULL) {
        if (layer_kind == MIMO26_LAYER_DENSE_GLOBAL) {
            entry = lookup(dense_mlp, sizeof dense_mlp / sizeof *dense_mlp,
                           cursor);
        } else {
            entry = lookup(moe_gate, sizeof moe_gate / sizeof *moe_gate,
                           cursor);
        }
    }
    if (entry == NULL) {
        if (layer_kind == MIMO26_LAYER_MOE_SWA) {
            entry = lookup(attn_swa, sizeof attn_swa / sizeof *attn_swa,
                           cursor);
        } else {
            entry = lookup(attn_global,
                           sizeof attn_global / sizeof *attn_global, cursor);
        }
    }
    if (entry == NULL || !matches(tensor, entry)) {
        return false;
    }
    if (kind != NULL) {
        *kind = layer_kind;
    }
    return true;
}

bool mimo26_architecture_validate_mtp_tensor(const k3_st_tensor *tensor)
{
    if (tensor == NULL || tensor->name == NULL) {
        return false;
    }
    static const char prefix[] = "model.mtp.layers.";
    if (strncmp(tensor->name, prefix, sizeof prefix - 1) != 0) {
        return false;
    }
    const char *cursor = tensor->name + sizeof prefix - 1;
    uint32_t layer = 0;
    if (!parse_index(cursor, &cursor, MIMO26_MTP_LAYER_COUNT - 1u, &layer)) {
        return false;
    }
    if (*cursor != '.') {
        return false;
    }
    cursor++;
    const contract *entry = lookup(mtp, sizeof mtp / sizeof *mtp, cursor);
    return entry != NULL && matches(tensor, entry);
}

static bool is_modality_name(const char *name, bool *audio)
{
    static const char speech[] = "speech_embeddings.";
    static const char encoder[] = "audio_encoder.";
    if (strncmp(name, speech, sizeof speech - 1) == 0 ||
        strncmp(name, encoder, sizeof encoder - 1) == 0) {
        *audio = true;
        return true;
    }
    if (strstr(name, "vision") != NULL || strstr(name, "visual") != NULL) {
        *audio = false;
        return true;
    }
    return false;
}

bool mimo26_architecture_validate_main(const k3_st_model *model,
                                       mimo26_architecture_report *report,
                                       char *error, size_t error_size)
{
    if (report != NULL) {
        memset(report, 0, sizeof *report);
    }
    if (model == NULL || model->tensors == NULL || model->tensor_count == 0) {
        return fail(error, error_size, "model metadata is empty");
    }

    mimo26_architecture_report local;
    memset(&local, 0, sizeof local);

    for (size_t index = 0; index < model->tensor_count; index++) {
        const k3_st_tensor *tensor = &model->tensors[index];
        if (tensor->name == NULL) {
            return fail(error, error_size, "tensor %zu has no name", index);
        }
        bool audio = false;
        if (is_modality_name(tensor->name, &audio)) {
            continue;
        }
        if (strncmp(tensor->name, "model.mtp.", 10) == 0) {
            continue;
        }
        mimo26_layer_kind kind = MIMO26_LAYER_INVALID;
        if (!mimo26_architecture_validate_main_tensor(tensor, &kind)) {
            return fail(error, error_size, "unexpected text tensor %s",
                        tensor->name);
        }
        if (strstr(tensor->name, ".mlp.experts.") != NULL) {
            local.expert_count++;
        } else {
            local.main_count++;
            if (kind == MIMO26_LAYER_DENSE_GLOBAL) {
                local.dense_count++;
            }
            if (strstr(tensor->name, ".self_attn.") != NULL) {
                if (kind == MIMO26_LAYER_MOE_SWA) {
                    local.swa_count++;
                } else {
                    local.global_count++;
                }
            }
        }
    }

    if (local.main_count != MIMO26_MAIN_TENSOR_COUNT) {
        return fail(error, error_size, "main tensor count %zu, expected %u",
                    local.main_count, MIMO26_MAIN_TENSOR_COUNT);
    }
    if (local.expert_count != MIMO26_EXPERT_TENSOR_COUNT) {
        return fail(error, error_size, "expert tensor count %zu, expected %u",
                    local.expert_count, MIMO26_EXPERT_TENSOR_COUNT);
    }
    if (report != NULL) {
        *report = local;
    }
    return true;
}

bool mimo26_architecture_validate(const k3_st_model *model,
                                  mimo26_architecture_report *report,
                                  char *error, size_t error_size)
{
    if (report != NULL) {
        memset(report, 0, sizeof *report);
    }
    mimo26_architecture_report local;
    if (!mimo26_architecture_validate_main(model, &local, error, error_size)) {
        return false;
    }

    for (size_t index = 0; index < model->tensor_count; index++) {
        const k3_st_tensor *tensor = &model->tensors[index];
        bool audio = false;
        if (is_modality_name(tensor->name, &audio)) {
            if (audio) {
                local.audio_count++;
            } else {
                local.vision_count++;
            }
            continue;
        }
        if (strncmp(tensor->name, "model.mtp.", 10) != 0) {
            continue;
        }
        if (!mimo26_architecture_validate_mtp_tensor(tensor)) {
            return fail(error, error_size, "unexpected MTP tensor %s",
                        tensor->name);
        }
        local.mtp_count++;
    }

    if (local.mtp_count != MIMO26_MTP_TENSOR_COUNT) {
        return fail(error, error_size, "MTP tensor count %zu, expected %u",
                    local.mtp_count, MIMO26_MTP_TENSOR_COUNT);
    }
    if (local.vision_count != MIMO26_VISION_TENSOR_COUNT) {
        return fail(error, error_size, "vision tensor count %zu, expected %u",
                    local.vision_count, MIMO26_VISION_TENSOR_COUNT);
    }
    const size_t expect_audio = MIMO26_AUDIO_ENCODER_TENSOR_COUNT +
                                MIMO26_SPEECH_EMBEDDING_COUNT;
    if (local.audio_count != expect_audio) {
        return fail(error, error_size, "audio tensor count %zu, expected %zu",
                    local.audio_count, expect_audio);
    }
    if (report != NULL) {
        *report = local;
    }
    return true;
}
