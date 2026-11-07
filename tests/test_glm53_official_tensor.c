#include "glm53_official_tensor.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

const k3_st_tensor *k3_st_find(const k3_st_model *model, const char *name)
{
    size_t index;
    if (model == NULL || name == NULL) return NULL;
    for (index = 0u; index < model->tensor_count; ++index) {
        if (model->tensors[index].name != NULL &&
            strcmp(model->tensors[index].name, name) == 0) {
            return &model->tensors[index];
        }
    }
    return NULL;
}

static k3_st_tensor tensor(char *name, k3_st_dtype dtype, uint8_t ndim,
                           uint64_t d0, uint64_t d1)
{
    k3_st_tensor value;
    memset(&value, 0, sizeof(value));
    value.name = name;
    value.dtype = dtype;
    value.ndim = ndim;
    value.shape[0] = d0;
    value.shape[1] = d1;
    return value;
}

static void check_positive(void)
{
    k3_st_tensor tensors[4];
    k3_st_model model;
    glm53_official_fp8_matrix bound;
    char error[128];

    tensors[0] = tensor("exact", K3_ST_DTYPE_F8_E4M3, 2u, 256u, 384u);
    tensors[1] = tensor("exact_scale_inv", K3_ST_DTYPE_F32, 2u, 2u, 3u);
    tensors[2] = tensor("partial", K3_ST_DTYPE_F8_E4M3, 2u, 129u, 257u);
    tensors[3] = tensor("partial_scale_inv", K3_ST_DTYPE_F32, 2u, 2u, 3u);
    memset(&model, 0, sizeof(model));
    model.tensors = tensors;
    model.tensor_count = 4u;

    assert(glm53_official_bind_fp8_matrix(&model, "exact", 256u, 384u,
                                           &bound, error, sizeof(error)));
    assert(bound.weight == &tensors[0] && bound.scale_inv == &tensors[1]);
    assert(bound.rows == 256u && bound.columns == 384u);
    assert(bound.scale_rows == 2u && bound.scale_columns == 3u);

    assert(glm53_official_bind_fp8_matrix(&model, "partial", 129u, 257u,
                                           &bound, NULL, 0u));
    assert(bound.weight == &tensors[2] && bound.scale_inv == &tensors[3]);
    assert(bound.scale_rows == 2u && bound.scale_columns == 3u);
}

static void check_require_failures(void)
{
    k3_st_tensor item = tensor("x", K3_ST_DTYPE_F32, 2u, 2u, 3u);
    k3_st_model model;
    const k3_st_tensor *out = &item;
    uint64_t shape[2] = {2u, 3u};
    char error[128];

    memset(&model, 0, sizeof(model));
    model.tensors = &item;
    model.tensor_count = 1u;
    assert(!glm53_official_require_tensor(NULL, "x", K3_ST_DTYPE_F32, 2u,
                                           shape, &out, error, sizeof(error)));
    assert(out == NULL);
    assert(!glm53_official_require_tensor(&model, NULL, K3_ST_DTYPE_F32, 2u,
                                           shape, &out, error, sizeof(error)));
    assert(!glm53_official_require_tensor(&model, "", K3_ST_DTYPE_F32, 2u,
                                           shape, &out, error, sizeof(error)));
    assert(!glm53_official_require_tensor(&model, "x", K3_ST_DTYPE_F32, 2u,
                                           NULL, &out, error, sizeof(error)));
    assert(!glm53_official_require_tensor(&model, "x", K3_ST_DTYPE_F32,
                                           K3_ST_MAX_DIMS + 1u, shape, &out,
                                           error, sizeof(error)));
    assert(!glm53_official_require_tensor(&model, "x", K3_ST_DTYPE_F32, 2u,
                                           shape, NULL, error, sizeof(error)));
    assert(!glm53_official_require_tensor(&model, "missing", K3_ST_DTYPE_F32,
                                           2u, shape, &out, error, sizeof(error)));
    assert(!glm53_official_require_tensor(&model, "x", K3_ST_DTYPE_F8_E4M3,
                                           2u, shape, &out, error, sizeof(error)));
    item.dtype = K3_ST_DTYPE_F32;
    item.ndim = 1u;
    assert(!glm53_official_require_tensor(&model, "x", K3_ST_DTYPE_F32,
                                           2u, shape, &out, error, sizeof(error)));
    item.ndim = 2u;
    item.shape[1] = 4u;
    assert(!glm53_official_require_tensor(&model, "x", K3_ST_DTYPE_F32,
                                           2u, shape, &out, error, sizeof(error)));
}

static void check_bind_failures(void)
{
    k3_st_tensor tensors[2];
    k3_st_model model;
    glm53_official_fp8_matrix out;
    char error[128];
    char long_name[GLM53_OFFICIAL_TENSOR_NAME_CAPACITY];

    tensors[0] = tensor("w", K3_ST_DTYPE_F8_E4M3, 2u, 129u, 257u);
    tensors[1] = tensor("w_scale_inv", K3_ST_DTYPE_F32, 2u, 2u, 3u);
    memset(&model, 0, sizeof(model));
    model.tensors = tensors;
    model.tensor_count = 2u;

    assert(!glm53_official_bind_fp8_matrix(NULL, "w", 129u, 257u,
                                            &out, error, sizeof(error)));
    assert(!glm53_official_bind_fp8_matrix(&model, NULL, 129u, 257u,
                                            &out, error, sizeof(error)));
    assert(!glm53_official_bind_fp8_matrix(&model, "w", 0u, 257u,
                                            &out, error, sizeof(error)));
    assert(!glm53_official_bind_fp8_matrix(&model, "w", 129u, 0u,
                                            &out, error, sizeof(error)));
    assert(!glm53_official_bind_fp8_matrix(&model, "w", 129u, 257u,
                                            NULL, error, sizeof(error)));

    model.tensor_count = 1u;
    assert(!glm53_official_bind_fp8_matrix(&model, "w", 129u, 257u,
                                            &out, error, sizeof(error)));
    assert(out.weight == NULL && out.scale_inv == NULL && out.rows == 0u);
    model.tensor_count = 2u;

    tensors[0].dtype = K3_ST_DTYPE_F32;
    assert(!glm53_official_bind_fp8_matrix(&model, "w", 129u, 257u,
                                            &out, error, sizeof(error)));
    tensors[0].dtype = K3_ST_DTYPE_F8_E4M3;
    tensors[0].ndim = 1u;
    assert(!glm53_official_bind_fp8_matrix(&model, "w", 129u, 257u,
                                            &out, error, sizeof(error)));
    tensors[0].ndim = 2u;
    tensors[0].shape[0] = 130u;
    assert(!glm53_official_bind_fp8_matrix(&model, "w", 129u, 257u,
                                            &out, error, sizeof(error)));
    tensors[0].shape[0] = 129u;
    tensors[1].shape[0] = 1u;
    assert(!glm53_official_bind_fp8_matrix(&model, "w", 129u, 257u,
                                            &out, error, sizeof(error)));

    memset(long_name, 'a', sizeof(long_name));
    long_name[sizeof(long_name) - 1u] = '\0';
    assert(!glm53_official_bind_fp8_matrix(&model, long_name, 1u, 1u,
                                            &out, error, sizeof(error)));
}

int main(void)
{
    check_positive();
    check_require_failures();
    check_bind_failures();
    puts("glm53 official tensor tests passed");
    return 0;
}
