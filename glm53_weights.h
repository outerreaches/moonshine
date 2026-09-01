#ifndef GLM53_WEIGHTS_H
#define GLM53_WEIGHTS_H

#include "glm53_expert_plan.h"
#include "glm53_manifest.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_WEIGHT_GLOBAL_COUNT 3u
#define GLM53_WEIGHT_LAYER_COUNT 45u
#define GLM53_WEIGHT_MAIN_BYTES UINT64_C(319706118392)
#define GLM53_WEIGHT_MAIN_F8_BYTES UINT64_C(307023052800)
#define GLM53_WEIGHT_MAIN_F32_BYTES UINT64_C(76137720)
#define GLM53_WEIGHT_MAIN_BF16_BYTES UINT64_C(12606927872)
#define GLM53_WEIGHT_STREAMED_BYTES UINT64_C(304405807104)
#define GLM53_WEIGHT_ROUTED_SCALE_BYTES UINT64_C(74317824)
#define GLM53_WEIGHT_RESIDENT_STATIC_BYTES UINT64_C(15225993464)
#define GLM53_WEIGHT_ENGINE_RESIDENT_BYTES UINT64_C(15300311288)
#define GLM53_WEIGHT_RESIDENT_STATIC_COUNT 1425u
#define GLM53_WEIGHT_STREAMED_COUNT 36288u
#define GLM53_WEIGHT_ROUTED_SCALE_COUNT 36288u

typedef enum {
    GLM53_WEIGHT_RESIDENT_STATIC = 0,
    GLM53_WEIGHT_STREAMED_ROUTED_EXPERT,
    GLM53_WEIGHT_RESIDENT_ROUTED_SCALE
} glm53_weight_class;

typedef enum {
    GLM53_GLOBAL_LM_HEAD = 0,
    GLM53_GLOBAL_EMBED_TOKENS,
    GLM53_GLOBAL_FINAL_NORM
} glm53_global_role;

/* A role is present only when it applies to the layer kind. */
typedef enum {
    GLM53_ROLE_HC_ATTN_BASE = 0, GLM53_ROLE_HC_ATTN_SCALE,
    GLM53_ROLE_HC_FFN_BASE, GLM53_ROLE_HC_FFN_SCALE,
    GLM53_ROLE_HC_ATTN_FN, GLM53_ROLE_HC_FFN_FN,
    GLM53_ROLE_INPUT_NORM, GLM53_ROLE_POST_ATTN_NORM,
    GLM53_ROLE_MLP_DOWN, GLM53_ROLE_MLP_DOWN_SCALE,
    GLM53_ROLE_MLP_GATE, GLM53_ROLE_MLP_GATE_SCALE,
    GLM53_ROLE_MLP_UP, GLM53_ROLE_MLP_UP_SCALE,
    GLM53_ROLE_ROUTER_BIAS, GLM53_ROLE_ROUTER_WEIGHT,
    GLM53_ROLE_SHARED_DOWN, GLM53_ROLE_SHARED_DOWN_SCALE,
    GLM53_ROLE_SHARED_GATE, GLM53_ROLE_SHARED_GATE_SCALE,
    GLM53_ROLE_SHARED_UP, GLM53_ROLE_SHARED_UP_SCALE,
    GLM53_ROLE_ATTN_A_LOG, GLM53_ROLE_ATTN_DT_BIAS,
    GLM53_ROLE_ATTN_B_PROJ, GLM53_ROLE_ATTN_F_A_PROJ,
    GLM53_ROLE_ATTN_F_B_PROJ, GLM53_ROLE_ATTN_G_A_PROJ,
    GLM53_ROLE_ATTN_G_B_PROJ, GLM53_ROLE_ATTN_K_CONV,
    GLM53_ROLE_ATTN_K_PROJ, GLM53_ROLE_ATTN_O_NORM,
    GLM53_ROLE_ATTN_O_PROJ, GLM53_ROLE_ATTN_Q_CONV,
    GLM53_ROLE_ATTN_Q_PROJ, GLM53_ROLE_ATTN_V_CONV,
    GLM53_ROLE_ATTN_V_PROJ,
    GLM53_ROLE_DSA_KV_A, GLM53_ROLE_DSA_KV_A_SCALE,
    GLM53_ROLE_DSA_KV_A_NORM, GLM53_ROLE_DSA_KV_B,
    GLM53_ROLE_DSA_O, GLM53_ROLE_DSA_O_SCALE,
    GLM53_ROLE_DSA_Q_A, GLM53_ROLE_DSA_Q_A_SCALE,
    GLM53_ROLE_DSA_Q_A_NORM, GLM53_ROLE_DSA_Q_B,
    GLM53_ROLE_DSA_Q_B_SCALE,
    GLM53_ROLE_INDEX_APE, GLM53_ROLE_INDEX_GATE,
    GLM53_ROLE_INDEX_K_NORM_BIAS, GLM53_ROLE_INDEX_K_NORM,
    GLM53_ROLE_INDEX_WEIGHTS, GLM53_ROLE_INDEX_WK,
    GLM53_ROLE_INDEX_WQ_B,
    GLM53_WEIGHT_LAYER_ROLE_COUNT
} glm53_weight_layer_role;

typedef struct {
    uint64_t f8_bytes;
    uint64_t f32_bytes;
    uint64_t bf16_bytes;
    uint64_t total_bytes;
    size_t tensor_count;
} glm53_weight_ledger;

typedef enum {
    GLM53_WEIGHT_LAYER_INVALID = 0,
    GLM53_WEIGHT_LAYER_DENSE_KDA,
    GLM53_WEIGHT_LAYER_ROUTED_KDA,
    GLM53_WEIGHT_LAYER_ROUTED_DSA
} glm53_weight_layer_kind;

typedef struct {
    glm53_weight_layer_kind kind;
    const k3_st_tensor *roles[GLM53_WEIGHT_LAYER_ROLE_COUNT];
} glm53_weight_layer;

typedef struct {
    bool built;
    const k3_st_tensor *globals[GLM53_WEIGHT_GLOBAL_COUNT];
    glm53_weight_layer layers[GLM53_WEIGHT_LAYER_COUNT];
    glm53_expert_model_plan routed_experts;
    glm53_weight_ledger resident_static;
    glm53_weight_ledger streamed_routed_experts;
    glm53_weight_ledger resident_routed_scales;
    glm53_weight_ledger total;
} glm53_weight_plan;

/* Classify one exact main-text tensor. Unknown, MTP, vision, and malformed
 * tensors return false. */
bool glm53_weight_classify(const k3_st_tensor *tensor,
                           glm53_weight_class *class_out);

/* Build a payload-free plan. The model must contain exactly the 74001 official
 * main-text metadata records. Input order is irrelevant. Output is changed
 * only on success. */
bool glm53_weight_plan_build(glm53_weight_plan *plan,
                             const k3_st_model *model,
                             char *error, size_t error_size);

/* As above, and require every input tensor to agree with its manifest entry.
 * Extra manifest entries (for MTP/vision) do not enter the plan. */
bool glm53_weight_plan_build_manifest(glm53_weight_plan *plan,
                                      const glm53_manifest *manifest,
                                      const k3_st_model *model,
                                      char *error, size_t error_size);

/* Return resident static tensors plus resident routed FP8 scales. Streamed
 * routed expert values are deliberately excluded. Output is zero on failure. */
bool glm53_weight_plan_engine_resident_bytes(const glm53_weight_plan *plan,
                                              uint64_t *bytes);

void glm53_weight_plan_free(glm53_weight_plan *plan);

#ifdef __cplusplus
}
#endif
#endif
