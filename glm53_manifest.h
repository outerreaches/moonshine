#ifndef GLM53_MANIFEST_H
#define GLM53_MANIFEST_H

#include "k3_safetensors.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_SHARD_COUNT 62u
#define GLM53_INDEX_TENSOR_COUNT 76108u
#define GLM53_MODULE_EXCLUSION_COUNT 1509u
#define GLM53_INDEX_TOTAL_BYTES UINT64_C(328326771576)

typedef struct {
    char *name;
    uint16_t shard; /* zero based */
} glm53_index_entry;

typedef struct {
    char **modules_to_not_convert;
    size_t module_count;
    glm53_index_entry *entries;
    size_t entry_count;
    uint64_t total_size;
} glm53_manifest;

/* Strictly validate the official GLM-5.3-Flash config and index contracts. */
bool glm53_manifest_parse(glm53_manifest *manifest,
                          const char *config_json,
                          size_t config_size,
                          const char *index_json,
                          size_t index_size,
                          char *error,
                          size_t error_size);

bool glm53_manifest_load(glm53_manifest *manifest,
                         const char *root,
                         char *error,
                         size_t error_size);

void glm53_manifest_free(glm53_manifest *manifest);

/* Apply config namespace normalization and dot-component-boundary matching. */
bool glm53_manifest_tensor_excluded(const glm53_manifest *manifest,
                                    const char *tensor_name);

/* Validate the already parsed, payload-free SafeTensors metadata directory. */
bool glm53_manifest_reconcile(const glm53_manifest *manifest,
                              const k3_st_model *model,
                              char *error,
                              size_t error_size);

/* Pure count gate used by reconciliation for a complete official manifest. */
bool glm53_manifest_official_dtype_counts_valid(size_t tensor_count,
                                                 size_t f8_count,
                                                 size_t f32_count,
                                                 size_t bf16_count,
                                                 size_t scale_count);

#ifdef __cplusplus
}
#endif

#endif
