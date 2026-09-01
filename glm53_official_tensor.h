#ifndef GLM53_OFFICIAL_TENSOR_H
#define GLM53_OFFICIAL_TENSOR_H

#include "k3_safetensors.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_OFFICIAL_TENSOR_NAME_CAPACITY 512u

typedef struct {
    const k3_st_tensor *weight;
    const k3_st_tensor *scale_inv;
    uint64_t rows;
    uint64_t columns;
    uint64_t scale_rows;
    uint64_t scale_columns;
} glm53_official_fp8_matrix;

/* Require one tensor with exactly the requested dtype, rank, and dimensions. */
bool glm53_official_require_tensor(const k3_st_model *model,
                                   const char *name,
                                   k3_st_dtype dtype,
                                   uint8_t ndim,
                                   const uint64_t *shape,
                                   const k3_st_tensor **out,
                                   char *error,
                                   size_t error_size);

/* Bind an F8 matrix and its blockwise F32 <weight_name>_scale_inv tensor. */
bool glm53_official_bind_fp8_matrix(const k3_st_model *model,
                                    const char *weight_name,
                                    uint64_t rows,
                                    uint64_t cols,
                                    glm53_official_fp8_matrix *out,
                                    char *error,
                                    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
