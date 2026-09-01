#include "glm53_architecture.h"
#include "glm53_manifest.h"
#include "k3_safetensors.h"

#include <stdio.h>
#include <string.h>

static k3_st_tensor *find_mutable(k3_st_model *model, const char *name) {
    size_t i;
    for (i = 0; i < model->tensor_count; ++i)
        if (strcmp(model->tensors[i].name, name) == 0) return &model->tensors[i];
    return NULL;
}

int main(int argc, char **argv) {
    char error[512] = {0};
    glm53_manifest manifest = {0};
    k3_st_model model = {0};
    glm53_architecture_report report = {0};
    int result = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s OFFICIAL_ROOT\n", argv[0]);
        return 2;
    }
    if (!glm53_manifest_load(&manifest, argv[1], error, sizeof(error))) {
        fprintf(stderr, "manifest: %s\n", error);
        goto done;
    }
    if (!k3_st_model_open_5digit_total(&model, argv[1], GLM53_SHARD_COUNT,
                                       error, sizeof(error))) {
        fprintf(stderr, "headers: %s\n", error);
        goto done;
    }
    if (!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error))) {
        fprintf(stderr, "reconcile: %s\n", error);
        goto done;
    }
    if (!glm53_architecture_validate(&model, &report, error, sizeof(error))) {
        fprintf(stderr, "architecture: %s\n", error);
        goto done;
    }
    {
        k3_st_tensor *mtp = find_mutable(
            &model, "model.language_model.layers.45.input_layernorm.weight");
        k3_st_tensor *vision = find_mutable(
            &model, "model.visual.patch_embed.proj.weight");
        size_t count = 123u;
        glm53_architecture_report failed;
        k3_st_dtype saved;
        if (mtp == NULL || vision == NULL) {
            fprintf(stderr, "architecture: mutation tensor missing\n");
            goto done;
        }
        saved = mtp->dtype;
        mtp->dtype = K3_ST_DTYPE_F32;
        if (glm53_architecture_validate_mtp_metadata(
                &model, &count, error, sizeof(error)) || count != 0u) {
            fprintf(stderr, "architecture: MTP substitution accepted/published\n");
            mtp->dtype = saved;
            goto done;
        }
        memset(&failed, 0x7f, sizeof(failed));
        if (glm53_architecture_validate(&model, &failed, error, sizeof(error)) ||
            memcmp(&failed, &(glm53_architecture_report){0}, sizeof(failed)) != 0) {
            fprintf(stderr, "architecture: MTP substitution report published\n");
            mtp->dtype = saved;
            goto done;
        }
        mtp->dtype = saved;

        saved = vision->dtype;
        vision->dtype = K3_ST_DTYPE_F32;
        count = 123u;
        if (glm53_architecture_validate_vision_metadata(
                &model, &count, error, sizeof(error)) || count != 0u) {
            fprintf(stderr, "architecture: vision substitution accepted/published\n");
            vision->dtype = saved;
            goto done;
        }
        memset(&failed, 0x7f, sizeof(failed));
        if (glm53_architecture_validate(&model, &failed, error, sizeof(error)) ||
            memcmp(&failed, &(glm53_architecture_report){0}, sizeof(failed)) != 0) {
            fprintf(stderr, "architecture: vision substitution report published\n");
            vision->dtype = saved;
            goto done;
        }
        vision->dtype = saved;
    }
    printf("PASS main=%zu dense=%zu routed=%zu kda=%zu dsa=%zu "
           "mtp=%zu vision=%zu mutations=2\n",
           report.main_count, report.dense_count, report.routed_count,
           report.kda_count, report.dsa_count, report.mtp_count,
           report.vision_count);
    result = 0;

done:
    k3_st_model_close(&model);
    glm53_manifest_free(&manifest);
    return result;
}
