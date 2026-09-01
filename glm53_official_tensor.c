#include "glm53_official_tensor.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void glm53_tensor_error(char *error, size_t error_size,
                               const char *format, ...)
{
    va_list arguments;

    if (error == NULL || error_size == 0u) {
        return;
    }
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

bool glm53_official_require_tensor(const k3_st_model *model,
                                   const char *name,
                                   k3_st_dtype dtype,
                                   uint8_t ndim,
                                   const uint64_t *shape,
                                   const k3_st_tensor **out,
                                   char *error,
                                   size_t error_size)
{
    const k3_st_tensor *tensor;
    uint8_t dimension;

    if (out != NULL) {
        *out = NULL;
    }
    if (model == NULL || name == NULL || name[0] == '\0' || out == NULL ||
        ndim > K3_ST_MAX_DIMS || (ndim != 0u && shape == NULL) ||
        (error == NULL && error_size != 0u)) {
        glm53_tensor_error(error, error_size, "invalid tensor requirement arguments");
        return false;
    }

    tensor = k3_st_find(model, name);
    if (tensor == NULL) {
        glm53_tensor_error(error, error_size, "required tensor '%s' is missing", name);
        return false;
    }
    if (tensor->dtype != dtype) {
        glm53_tensor_error(error, error_size, "tensor '%s' has the wrong dtype", name);
        return false;
    }
    if (tensor->ndim != ndim) {
        glm53_tensor_error(error, error_size, "tensor '%s' has the wrong rank", name);
        return false;
    }
    for (dimension = 0u; dimension < ndim; ++dimension) {
        if (tensor->shape[dimension] != shape[dimension]) {
            glm53_tensor_error(error, error_size,
                               "tensor '%s' has the wrong dimension %u",
                               name, (unsigned)dimension);
            return false;
        }
    }

    *out = tensor;
    return true;
}

bool glm53_official_bind_fp8_matrix(const k3_st_model *model,
                                    const char *weight_name,
                                    uint64_t rows,
                                    uint64_t cols,
                                    glm53_official_fp8_matrix *out,
                                    char *error,
                                    size_t error_size)
{
    static const char suffix[] = "_scale_inv";
    char scale_name[GLM53_OFFICIAL_TENSOR_NAME_CAPACITY];
    uint64_t weight_shape[2];
    uint64_t scale_shape[2];
    const k3_st_tensor *weight;
    const k3_st_tensor *scale_inv;
    size_t name_length;

    if (out != NULL) {
        out->weight = NULL;
        out->scale_inv = NULL;
        out->rows = 0u;
        out->columns = 0u;
        out->scale_rows = 0u;
        out->scale_columns = 0u;
    }
    if (model == NULL || weight_name == NULL || weight_name[0] == '\0' ||
        rows == 0u || cols == 0u || out == NULL ||
        (error == NULL && error_size != 0u)) {
        glm53_tensor_error(error, error_size, "invalid FP8 matrix arguments");
        return false;
    }

    name_length = strlen(weight_name);
    if (name_length > SIZE_MAX - sizeof(suffix) ||
        name_length + sizeof(suffix) > sizeof(scale_name)) {
        glm53_tensor_error(error, error_size, "FP8 matrix name is too long");
        return false;
    }
    memcpy(scale_name, weight_name, name_length);
    memcpy(scale_name + name_length, suffix, sizeof(suffix));

    weight_shape[0] = rows;
    weight_shape[1] = cols;
    scale_shape[0] = rows / 128u + (rows % 128u != 0u);
    scale_shape[1] = cols / 128u + (cols % 128u != 0u);

    if (!glm53_official_require_tensor(model, weight_name,
                                       K3_ST_DTYPE_F8_E4M3, 2u,
                                       weight_shape, &weight,
                                       error, error_size) ||
        !glm53_official_require_tensor(model, scale_name,
                                       K3_ST_DTYPE_F32, 2u,
                                       scale_shape, &scale_inv,
                                       error, error_size)) {
        return false;
    }

    out->weight = weight;
    out->scale_inv = scale_inv;
    out->rows = rows;
    out->columns = cols;
    out->scale_rows = scale_shape[0];
    out->scale_columns = scale_shape[1];
    return true;
}
