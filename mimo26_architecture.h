#ifndef MIMO26_ARCHITECTURE_H
#define MIMO26_ARCHITECTURE_H

#include "k3_safetensors.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIMO26_TEXT_LAYER_COUNT 48u
#define MIMO26_MOE_LAYER_COUNT 47u
#define MIMO26_ROUTED_EXPERTS_PER_LAYER 256u
#define MIMO26_EXPERTS_PER_TOKEN 8u
#define MIMO26_HIDDEN_SIZE 4096u
#define MIMO26_VOCAB_SIZE 152576u

/* Observed exactly in revision 3b38d063180c3e4aed9691fdc735f3d10b266ee4. */
#define MIMO26_MAIN_TENSOR_COUNT 382u
#define MIMO26_EXPERT_TENSOR_COUNT 72192u
#define MIMO26_MTP_TENSOR_COUNT 48u
#define MIMO26_VISION_TENSOR_COUNT 364u
#define MIMO26_AUDIO_ENCODER_TENSOR_COUNT 75u
#define MIMO26_SPEECH_EMBEDDING_COUNT 20u
#define MIMO26_TOTAL_TENSOR_COUNT 73081u

typedef enum {
    MIMO26_LAYER_INVALID = 0,
    MIMO26_LAYER_DENSE_GLOBAL, /* layer 0: dense MLP, global attention */
    MIMO26_LAYER_MOE_GLOBAL,
    MIMO26_LAYER_MOE_SWA
} mimo26_layer_kind;

typedef struct {
    size_t main_count;    /* text tensors excluding routed experts */
    size_t expert_count;  /* routed expert weights and scales */
    size_t mtp_count;
    size_t vision_count;
    size_t audio_count;   /* audio encoder plus speech embeddings */
    size_t dense_count;   /* tensors belonging to the dense layer */
    size_t swa_count;     /* attention tensors on sliding-window layers */
    size_t global_count;  /* attention tensors on full-attention layers */
} mimo26_architecture_report;

/*
 * Sliding-window layers are taken from config's explicit 48-entry
 * hybrid_layer_pattern. Full-attention layers are 0, 5, 11, 17, 23, 29, 35,
 * 41 and 47 -- note this is NOT periodic: the first gap is five, every later
 * gap is six. Deriving the pattern from a period misclassifies layers 12, 24
 * and 36, whose tensors then fail geometry validation.
 */
mimo26_layer_kind mimo26_architecture_layer_kind(uint32_t layer);

/* True when the layer uses sliding-window attention with a learned sink. */
bool mimo26_architecture_layer_is_swa(uint32_t layer);

/*
 * Validate one required main-text tensor contract, routed experts included.
 * Rejects MTP, vision and audio names. Non-canonical decimal layer or expert
 * indices are rejected rather than normalized.
 */
bool mimo26_architecture_validate_main_tensor(const k3_st_tensor *tensor,
                                              mimo26_layer_kind *kind);
bool mimo26_architecture_validate_mtp_tensor(const k3_st_tensor *tensor);

/* Exact, payload-free validation of the complete official HF metadata. Each
 * report is cleared on entry and populated only on success. */
bool mimo26_architecture_validate_main(const k3_st_model *model,
                                       mimo26_architecture_report *report,
                                       char *error, size_t error_size);
bool mimo26_architecture_validate(const k3_st_model *model,
                                  mimo26_architecture_report *report,
                                  char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
