#include "k31_model.h"

#include "k3_json.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error == NULL || error_size == 0u) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static int32_t required_value(const k3_json_document *document,
                              int32_t object,
                              const char *context,
                              const char *key,
                              char *error,
                              size_t error_size) {
    int32_t found = -1;
    int32_t child = document->tokens[object].first_child;
    while (child >= 0) {
        const int32_t value = document->tokens[child].next_sibling;
        if (k3_json_string_equal(document, child, key)) {
            if (found >= 0) {
                set_error(error, error_size, "%s.%s is duplicated",
                          context, key);
                return -1;
            }
            found = value;
        }
        child = document->tokens[value].next_sibling;
    }
    if (found < 0) {
        set_error(error, error_size, "%s.%s is required", context, key);
    }
    return found;
}

static bool required_object(const k3_json_document *document,
                            int32_t parent,
                            const char *parent_context,
                            const char *key,
                            int32_t *object,
                            char *error,
                            size_t error_size) {
    const int32_t value = required_value(document, parent, parent_context,
                                         key, error, error_size);
    if (value < 0) {
        return false;
    }
    if (document->tokens[value].type != K3_JSON_OBJECT) {
        set_error(error, error_size, "%s.%s must be an object",
                  parent_context, key);
        return false;
    }
    *object = value;
    return true;
}

static bool required_u32(const k3_json_document *document,
                         int32_t object,
                         const char *context,
                         const char *key,
                         uint32_t expected,
                         uint32_t *output,
                         char *error,
                         size_t error_size) {
    const int32_t value = required_value(document, object, context, key,
                                         error, error_size);
    uint32_t parsed = 0u;
    if (value < 0) {
        return false;
    }
    if (!k3_json_u32(document, value, &parsed) || parsed != expected) {
        set_error(error, error_size, "%s.%s must be %u",
                  context, key, expected);
        return false;
    }
    *output = parsed;
    return true;
}

static bool parse_layer_array(const k3_json_document *document,
                              int32_t object,
                              const char *context,
                              const char *key,
                              size_t expected_count,
                              bool selected[K31_MODEL_LAYER_COUNT + 1u],
                              char *error,
                              size_t error_size) {
    const int32_t array = required_value(document, object, context, key,
                                         error, error_size);
    if (array < 0) {
        return false;
    }
    if (document->tokens[array].type != K3_JSON_ARRAY ||
        document->tokens[array].size != expected_count) {
        set_error(error, error_size, "%s.%s must contain exactly %zu layers",
                  context, key, expected_count);
        return false;
    }
    memset(selected, 0,
           sizeof(selected[0]) * (K31_MODEL_LAYER_COUNT + 1u));
    for (size_t index = 0u; index < expected_count; index++) {
        const int32_t value = k3_json_array_get(document, array, index);
        uint32_t layer = 0u;
        if (!k3_json_u32(document, value, &layer) ||
            layer == 0u || layer > K31_MODEL_LAYER_COUNT) {
            set_error(error, error_size,
                      "%s.%s contains an unknown attention layer",
                      context, key);
            return false;
        }
        if (selected[layer]) {
            set_error(error, error_size,
                      "%s.%s contains duplicate layer %u",
                      context, key, layer);
            return false;
        }
        selected[layer] = true;
    }
    return true;
}

static bool parse_dsa_geometry(const k3_json_document *document,
                               int32_t dsa_config,
                               k31_dsa_geometry *geometry,
                               char *error,
                               size_t error_size) {
    static const char context[] = "config.text_config.dsa_config";
    if (!required_u32(document, dsa_config, context, "index_topk",
                      2048u, &geometry->index_topk, error, error_size) ||
        !required_u32(document, dsa_config, context, "index_head_dim",
                      128u, &geometry->index_head_dim, error, error_size) ||
        !required_u32(document, dsa_config, context, "index_num_heads",
                      32u, &geometry->index_num_heads, error, error_size) ||
        !required_u32(document, dsa_config, context, "index_rope_dim",
                      0u, &geometry->index_rope_dim, error, error_size) ||
        !required_u32(document, dsa_config, context, "index_topk_freq",
                      1u, &geometry->index_topk_freq, error, error_size) ||
        !required_u32(document, dsa_config, context,
                      "index_skip_topk_offset", 1u,
                      &geometry->index_skip_topk_offset,
                      error, error_size)) {
        return false;
    }
    const int32_t norm = required_value(document, dsa_config, context,
                                        "index_k_norm_type",
                                        error, error_size);
    if (norm < 0) {
        return false;
    }
    if (!k3_json_string_equal(document, norm, "layernorm")) {
        set_error(error, error_size,
                  "%s.index_k_norm_type must be layernorm", context);
        return false;
    }
    geometry->index_k_norm_type = K31_INDEX_K_NORM_LAYERNORM;
    return true;
}

bool k31_model_profile_parse(k31_model_profile *profile,
                             const char *json,
                             size_t json_size,
                             char *error,
                             size_t error_size) {
    if (error != NULL && error_size != 0u) {
        error[0] = '\0';
    }
    if (profile == NULL || json == NULL) {
        set_error(error, error_size, "model profile arguments are invalid");
        return false;
    }

    k3_json_document document;
    memset(&document, 0, sizeof(document));
    if (!k3_json_parse(&document, json, json_size, error, error_size)) {
        return false;
    }

    k31_model_profile candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.model_id = K31_MODEL_ID;
    candidate.model_revision = K31_MODEL_REVISION;
    candidate.kda_layer_count = K31_MODEL_KDA_LAYER_COUNT;
    candidate.dsa_layer_count = K31_MODEL_DSA_LAYER_COUNT;
    bool ok = false;

    if (document.tokens[document.root].type != K3_JSON_OBJECT) {
        set_error(error, error_size, "config root must be an object");
        goto cleanup;
    }

    const int32_t architectures = required_value(
        &document, document.root, "config", "architectures",
        error, error_size);
    if (architectures < 0) {
        goto cleanup;
    }
    if (document.tokens[architectures].type != K3_JSON_ARRAY ||
        document.tokens[architectures].size != 1u ||
        !k3_json_string_equal(&document,
                              k3_json_array_get(&document, architectures, 0u),
                              K31_MODEL_ARCHITECTURE)) {
        set_error(error, error_size,
                  "config.architectures must contain only %s",
                  K31_MODEL_ARCHITECTURE);
        goto cleanup;
    }

    const int32_t model_type = required_value(
        &document, document.root, "config", "model_type", error, error_size);
    if (model_type < 0) {
        goto cleanup;
    }
    if (!k3_json_string_equal(&document, model_type, K31_MODEL_TYPE)) {
        set_error(error, error_size, "config.model_type must be %s",
                  K31_MODEL_TYPE);
        goto cleanup;
    }

    int32_t text_config = -1;
    int32_t linear_config = -1;
    int32_t dsa_config = -1;
    if (!required_object(&document, document.root, "config", "text_config",
                         &text_config, error, error_size) ||
        !required_u32(&document, text_config, "config.text_config",
                      "num_hidden_layers", K31_MODEL_LAYER_COUNT,
                      &candidate.layer_count, error, error_size) ||
        !required_object(&document, text_config, "config.text_config",
                         "linear_attn_config", &linear_config,
                         error, error_size) ||
        !required_object(&document, text_config, "config.text_config",
                         "dsa_config", &dsa_config, error, error_size) ||
        !parse_dsa_geometry(&document, dsa_config, &candidate.dsa,
                            error, error_size)) {
        goto cleanup;
    }

    bool kda[K31_MODEL_LAYER_COUNT + 1u];
    bool dsa[K31_MODEL_LAYER_COUNT + 1u];
    bool full[K31_MODEL_LAYER_COUNT + 1u];
    if (!parse_layer_array(&document, linear_config,
                           "config.text_config.linear_attn_config",
                           "kda_layers", K31_MODEL_KDA_LAYER_COUNT,
                           kda, error, error_size) ||
        !parse_layer_array(&document, linear_config,
                           "config.text_config.linear_attn_config",
                           "full_attn_layers", K31_MODEL_DSA_LAYER_COUNT,
                           full, error, error_size) ||
        !parse_layer_array(&document, dsa_config,
                           "config.text_config.dsa_config", "dsa_layers",
                           K31_MODEL_DSA_LAYER_COUNT,
                           dsa, error, error_size)) {
        goto cleanup;
    }

    for (uint32_t layer = 1u; layer <= K31_MODEL_LAYER_COUNT; layer++) {
        const bool expected_dsa = layer % 4u == 0u || layer == 93u;
        if (dsa[layer] != expected_dsa ||
            full[layer] != expected_dsa ||
            kda[layer] == expected_dsa) {
            set_error(error, error_size,
                      "conflicting or missing attention placement at layer %u",
                      layer);
            goto cleanup;
        }
        candidate.attention_by_layer[layer - 1u] = expected_dsa ?
            K31_ATTENTION_DSA : K31_ATTENTION_KDA;
    }

    *profile = candidate;
    ok = true;

cleanup:
    k3_json_document_free(&document);
    return ok;
}

static bool read_config_file(const char *path,
                             char **contents,
                             size_t *contents_size,
                             char *error,
                             size_t error_size) {
    *contents = NULL;
    *contents_size = 0u;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        set_error(error, error_size, "opening %s failed: %s",
                  path, strerror(errno));
        return false;
    }

    size_t size = 0u;
    size_t capacity = 4096u;
    char *data = (char *)malloc(capacity);
    bool ok = data != NULL;
    if (!ok) {
        set_error(error, error_size, "allocating config buffer failed");
    }
    while (ok) {
        if (size == capacity) {
            if (capacity > SIZE_MAX / 2u) {
                set_error(error, error_size, "config file is too large");
                ok = false;
                break;
            }
            capacity *= 2u;
            char *grown = (char *)realloc(data, capacity);
            if (grown == NULL) {
                set_error(error, error_size,
                          "growing config buffer failed");
                ok = false;
                break;
            }
            data = grown;
        }
        const size_t count = fread(data + size, 1u, capacity - size, file);
        size += count;
        if (count == 0u) {
            if (ferror(file)) {
                set_error(error, error_size, "reading %s failed: %s",
                          path, strerror(errno));
                ok = false;
            } else if (feof(file)) {
                break;
            } else {
                set_error(error, error_size, "reading %s made no progress",
                          path);
                ok = false;
            }
        }
    }
    if (fclose(file) != 0 && ok) {
        set_error(error, error_size, "closing %s failed: %s",
                  path, strerror(errno));
        ok = false;
    }
    if (!ok) {
        free(data);
        return false;
    }
    *contents = data;
    *contents_size = size;
    return true;
}

bool k31_model_profile_load(k31_model_profile *profile,
                            const char *path,
                            char *error,
                            size_t error_size) {
    if (error != NULL && error_size != 0u) {
        error[0] = '\0';
    }
    if (profile == NULL || path == NULL) {
        set_error(error, error_size, "model profile arguments are invalid");
        return false;
    }
    char *contents = NULL;
    size_t contents_size = 0u;
    if (!read_config_file(path, &contents, &contents_size,
                          error, error_size)) {
        return false;
    }
    k31_model_profile candidate;
    const bool ok = k31_model_profile_parse(
        &candidate, contents, contents_size, error, error_size);
    free(contents);
    if (ok) {
        *profile = candidate;
    }
    return ok;
}
