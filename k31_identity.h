#ifndef K31_IDENTITY_H
#define K31_IDENTITY_H

#include "k31_model.h"
#include "k3_safetensors.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Complete tensor-layout plus pinned model/DSA profile identity. */
uint64_t k31_model_layout_crc64(
    const k3_st_model *model,
    const k31_model_profile *profile);

#ifdef __cplusplus
}
#endif

#endif
