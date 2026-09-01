#ifndef GLM53_ARCHITECTURE_H
#define GLM53_ARCHITECTURE_H

#include "k3_safetensors.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_TEXT_LAYER_COUNT 45u
#define GLM53_EXPERT_COUNT 288u
#define GLM53_MAIN_TENSOR_COUNT 74001u
#define GLM53_MTP_TENSOR_COUNT 1760u
#define GLM53_VISION_TENSOR_COUNT 347u

typedef enum {
    GLM53_LAYER_INVALID = 0,
    GLM53_LAYER_DENSE_KDA,
    GLM53_LAYER_ROUTED_KDA,
    GLM53_LAYER_ROUTED_DSA
} glm53_layer_kind;

typedef struct {
    /* Tensor counts. Dense/routed and KDA/DSA exclude the three globals. */
    size_t dense_count;
    size_t routed_count;
    size_t kda_count;
    size_t dsa_count;
    size_t main_count;
    size_t mtp_count;
    size_t vision_count;
} glm53_architecture_report;

/* Layers 0..2 are dense. Every layer 3,7,...,43 is DSA; all others are KDA. */
glm53_layer_kind glm53_architecture_layer_kind(uint32_t layer);

/* Validate one required main-text tensor contract. This helper does not accept
 * MTP or vision tensors and is useful for bounded schema tests. */
bool glm53_architecture_validate_main_tensor(const k3_st_tensor *tensor,
                                              glm53_layer_kind *kind);
bool glm53_architecture_validate_mtp_tensor(const k3_st_tensor *tensor);
bool glm53_architecture_validate_vision_tensor(const k3_st_tensor *tensor);

/* Exact validators for the three independent metadata namespaces.  Each
 * result object is cleared on entry and is populated only after success. */
bool glm53_architecture_validate_main(const k3_st_model *model,
                                      glm53_architecture_report *report,
                                      char *error, size_t error_size);
bool glm53_architecture_validate_mtp_metadata(const k3_st_model *model,
                                              size_t *count,
                                              char *error, size_t error_size);
bool glm53_architecture_validate_vision_metadata(const k3_st_model *model,
                                                 size_t *count,
                                                 char *error, size_t error_size);

/* Strict, payload-free validation of the complete official HF metadata. */
bool glm53_architecture_validate(const k3_st_model *model,
                                 glm53_architecture_report *report,
                                 char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
