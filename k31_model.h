#ifndef K31_MODEL_H
#define K31_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define K31_MODEL_ID "moonshotai/Kimi-K3.1-Pro"
#define K31_MODEL_REVISION "6d0a0202956796543364d666ab19c6ad1a494ee1"
#define K31_MODEL_ARCHITECTURE "KimiK31ForConditionalGeneration"
#define K31_MODEL_TYPE "kimi_k31"

#define K31_MODEL_LAYER_COUNT 93u
#define K31_MODEL_KDA_LAYER_COUNT 69u
#define K31_MODEL_DSA_LAYER_COUNT 24u

typedef enum {
    K31_ATTENTION_UNKNOWN = 0,
    K31_ATTENTION_KDA = 1,
    K31_ATTENTION_DSA = 2,
} k31_attention_type;

typedef enum {
    K31_INDEX_K_NORM_LAYERNORM = 1,
} k31_index_k_norm_type;

typedef struct {
    uint32_t index_topk;
    uint32_t index_head_dim;
    uint32_t index_num_heads;
    uint32_t index_rope_dim;
    k31_index_k_norm_type index_k_norm_type;
    uint32_t index_topk_freq;
    uint32_t index_skip_topk_offset;
} k31_dsa_geometry;

typedef struct {
    const char *model_id;
    const char *model_revision;
    uint32_t layer_count;
    uint32_t kda_layer_count;
    uint32_t dsa_layer_count;
    /* Upstream config layer N is stored at attention_by_layer[N - 1]. */
    k31_attention_type attention_by_layer[K31_MODEL_LAYER_COUNT];
    k31_dsa_geometry dsa;
} k31_model_profile;

/*
 * Parse and validate an in-memory official config.json. The source need not be
 * NUL-terminated. On success, profile owns no allocations and requires no
 * destruction. On failure, profile is left unchanged.
 */
bool k31_model_profile_parse(k31_model_profile *profile,
                             const char *json,
                             size_t json_size,
                             char *error,
                             size_t error_size);

/* Read and validate a config.json file. On failure, profile is unchanged. */
bool k31_model_profile_load(k31_model_profile *profile,
                            const char *path,
                            char *error,
                            size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
