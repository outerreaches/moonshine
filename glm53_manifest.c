#include "glm53_manifest.h"
#include "k3_json.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void fail(char *error, size_t size, const char *format, ...) {
    if (!error || size == 0u) return;
    va_list ap;
    va_start(ap, format);
    vsnprintf(error, size, format, ap);
    va_end(ap);
}

static bool token_type(const k3_json_document *d, int32_t t,
                       k3_json_type type) {
    return t >= 0 && (size_t)t < d->token_count && d->tokens[t].type == type;
}

static bool string_is(const k3_json_document *d, int32_t object,
                      const char *key, const char *expected) {
    return k3_json_string_equal(d, k3_json_object_get(d, object, key), expected);
}

static bool u32_is(const k3_json_document *d, int32_t object,
                   const char *key, uint32_t expected) {
    uint32_t value = 0u;
    return k3_json_u32(d, k3_json_object_get(d, object, key), &value) &&
           value == expected;
}

static bool object_keys_unique(const k3_json_document *d, int32_t object,
                               char *error, size_t error_size) {
    if (!token_type(d, object, K3_JSON_OBJECT)) return false;
    for (int32_t a = d->tokens[object].first_child; a >= 0;) {
        const int32_t av = d->tokens[a].next_sibling;
        if (av < 0) return false;
        for (int32_t b = d->tokens[av].next_sibling; b >= 0;) {
            const int32_t bv = d->tokens[b].next_sibling;
            if (bv < 0) return false;
            char *ak = NULL;
            char *bk = NULL;
            const bool ok = k3_json_string_dup(d, a, &ak, error, error_size) &&
                            k3_json_string_dup(d, b, &bk, error, error_size);
            const bool same = ok && strcmp(ak, bk) == 0;
            free(ak);
            free(bk);
            if (!ok) return false;
            if (same) {
                fail(error, error_size, "duplicate JSON object key");
                return false;
            }
            b = d->tokens[bv].next_sibling;
        }
        a = d->tokens[av].next_sibling;
    }
    return true;
}

static int compare_strings(const void *left, const void *right) {
    const char *const *a = (const char *const *)left;
    const char *const *b = (const char *const *)right;
    return strcmp(*a, *b);
}

static int compare_entries(const void *left, const void *right) {
    const glm53_index_entry *a = (const glm53_index_entry *)left;
    const glm53_index_entry *b = (const glm53_index_entry *)right;
    return strcmp(a->name, b->name);
}

static bool validate_layer_array(const k3_json_document *d, int32_t text,
                                 const char *key, bool mlp,
                                 char *error, size_t error_size) {
    const int32_t array = k3_json_object_get(d, text, key);
    if (!token_type(d, array, K3_JSON_ARRAY) || d->tokens[array].size != 45u) {
        fail(error, error_size, "%s must contain 45 entries", key);
        return false;
    }
    for (size_t i = 0; i < 45u; i++) {
        const char *expected = mlp ? (i < 3u ? "dense" : "sparse") :
            (((i + 1u) % 4u) == 0u ? "deepseek_sparse_attention" :
                                      "linear_attention");
        if (!k3_json_string_equal(d, k3_json_array_get(d, array, i), expected)) {
            fail(error, error_size, "%s entry %zu is invalid", key, i);
            return false;
        }
    }
    return true;
}

static bool validate_attention_partition(const k3_json_document *d,
                                         int32_t text,
                                         char *error, size_t error_size) {
    const int32_t linear = k3_json_object_get(d, text, "linear_attn_config");
    if (!token_type(d, linear, K3_JSON_OBJECT) ||
        !object_keys_unique(d, linear, error, error_size)) return false;
    const int32_t kda = k3_json_object_get(d, linear, "kda_layers");
    const int32_t full = k3_json_object_get(d, linear, "full_attn_layers");
    if (!token_type(d, kda, K3_JSON_ARRAY) || d->tokens[kda].size != 34u ||
        !token_type(d, full, K3_JSON_ARRAY) || d->tokens[full].size != 11u) {
        fail(error, error_size, "attention layer partition has wrong size");
        return false;
    }
    bool seen[45] = { false };
    for (unsigned which = 0u; which < 2u; which++) {
        const int32_t array = which == 0u ? kda : full;
        for (size_t i = 0u; i < d->tokens[array].size; i++) {
            uint32_t layer = 0u;
            if (!k3_json_u32(d, k3_json_array_get(d, array, i), &layer) ||
                layer >= 45u || seen[layer] ||
                (((layer + 1u) % 4u) == 0u) != (which == 1u)) {
                fail(error, error_size, "invalid attention layer partition");
                return false;
            }
            seen[layer] = true;
        }
    }
    for (size_t i = 0u; i < 45u; i++) if (!seen[i]) return false;
    return true;
}

static bool parse_config(glm53_manifest *manifest, const char *source,
                         size_t source_size, char *error, size_t error_size) {
    k3_json_document d = { 0 };
    bool ok = false;
    if (!k3_json_parse(&d, source, source_size, error, error_size) ||
        !token_type(&d, d.root, K3_JSON_OBJECT) ||
        !object_keys_unique(&d, d.root, error, error_size)) goto done;
    const int32_t architectures = k3_json_object_get(&d, d.root, "architectures");
    const int32_t text = k3_json_object_get(&d, d.root, "text_config");
    const int32_t quant = k3_json_object_get(&d, d.root, "quantization_config");
    if (!string_is(&d, d.root, "model_type", "glm5_next") ||
        !token_type(&d, architectures, K3_JSON_ARRAY) ||
        d.tokens[architectures].size != 1u ||
        !k3_json_string_equal(&d, k3_json_array_get(&d, architectures, 0u),
                              "Glm5NextForConditionalGeneration") ||
        !token_type(&d, text, K3_JSON_OBJECT) ||
        !token_type(&d, quant, K3_JSON_OBJECT) ||
        !object_keys_unique(&d, text, error, error_size) ||
        !object_keys_unique(&d, quant, error, error_size)) goto done;
    if (!string_is(&d, text, "model_type", "glm5_next_text") ||
        !u32_is(&d, text, "hidden_size", 4096u) ||
        !u32_is(&d, text, "num_hidden_layers", 45u) ||
        !u32_is(&d, text, "num_nextn_predict_layers", 1u) ||
        !u32_is(&d, text, "first_k_dense_replace", 3u) ||
        !u32_is(&d, text, "n_routed_experts", 288u) ||
        !u32_is(&d, text, "n_shared_experts", 1u) ||
        !u32_is(&d, text, "num_experts_per_tok", 8u) ||
        !u32_is(&d, text, "moe_intermediate_size", 2048u) ||
        !validate_layer_array(&d, text, "mlp_layer_types", true,
                              error, error_size) ||
        !validate_layer_array(&d, text, "layer_types", false,
                              error, error_size) ||
        !validate_attention_partition(&d, text, error, error_size)) goto done;
    const int32_t block = k3_json_object_get(&d, quant, "weight_block_size");
    uint32_t b0 = 0u, b1 = 0u;
    if (!string_is(&d, quant, "quant_method", "fp8") ||
        !string_is(&d, quant, "fmt", "e4m3") ||
        !string_is(&d, quant, "activation_scheme", "dynamic") ||
        !token_type(&d, block, K3_JSON_ARRAY) || d.tokens[block].size != 2u ||
        !k3_json_u32(&d, k3_json_array_get(&d, block, 0u), &b0) || b0 != 128u ||
        !k3_json_u32(&d, k3_json_array_get(&d, block, 1u), &b1) || b1 != 128u) {
        fail(error, error_size, "invalid FP8 quantization contract");
        goto done;
    }
    const int32_t modules = k3_json_object_get(&d, quant, "modules_to_not_convert");
    if (!token_type(&d, modules, K3_JSON_ARRAY) ||
        d.tokens[modules].size != GLM53_MODULE_EXCLUSION_COUNT) {
        fail(error, error_size, "modules_to_not_convert must contain 1509 entries");
        goto done;
    }
    manifest->modules_to_not_convert = (char **)calloc(
        GLM53_MODULE_EXCLUSION_COUNT, sizeof(char *));
    if (!manifest->modules_to_not_convert) goto done;
    manifest->module_count = GLM53_MODULE_EXCLUSION_COUNT;
    for (size_t i = 0u; i < manifest->module_count; i++) {
        if (!k3_json_string_dup(&d, k3_json_array_get(&d, modules, i),
                                &manifest->modules_to_not_convert[i],
                                error, error_size) ||
            manifest->modules_to_not_convert[i][0] == '\0') goto done;
    }
    qsort(manifest->modules_to_not_convert, manifest->module_count,
          sizeof(char *), compare_strings);
    for (size_t i = 1u; i < manifest->module_count; i++) {
        if (strcmp(manifest->modules_to_not_convert[i - 1u],
                   manifest->modules_to_not_convert[i]) == 0) {
            fail(error, error_size, "duplicate modules_to_not_convert entry");
            goto done;
        }
    }
    ok = true;
done:
    k3_json_document_free(&d);
    if (!ok && (!error || !error[0])) fail(error, error_size, "invalid GLM config");
    return ok;
}

static bool shard_name(const char *name, uint16_t *shard) {
    static const char prefix[] = "model-";
    static const char middle[] = "-of-";
    static const char suffix[] = ".safetensors";
    if (!name || strlen(name) != 32u || memcmp(name, prefix, 6u) != 0 ||
        memcmp(name + 11u, middle, 4u) != 0 ||
        memcmp(name + 15u, "00062", 5u) != 0 ||
        memcmp(name + 20u, suffix, 12u) != 0) return false;
    unsigned value = 0u;
    for (size_t i = 6u; i < 11u; i++) {
        if (name[i] < '0' || name[i] > '9') return false;
        value = value * 10u + (unsigned)(name[i] - '0');
    }
    if (value == 0u || value > GLM53_SHARD_COUNT) return false;
    *shard = (uint16_t)(value - 1u);
    return true;
}

static bool parse_index(glm53_manifest *manifest, const char *source,
                        size_t source_size, char *error, size_t error_size) {
    k3_json_document d = { 0 };
    bool ok = false;
    if (!k3_json_parse(&d, source, source_size, error, error_size) ||
        !token_type(&d, d.root, K3_JSON_OBJECT) || d.tokens[d.root].size != 2u ||
        !object_keys_unique(&d, d.root, error, error_size)) goto done;
    const int32_t metadata = k3_json_object_get(&d, d.root, "metadata");
    const int32_t weights = k3_json_object_get(&d, d.root, "weight_map");
    uint64_t total = 0u;
    if (!token_type(&d, metadata, K3_JSON_OBJECT) ||
        d.tokens[metadata].size != 1u ||
        !object_keys_unique(&d, metadata, error, error_size) ||
        !k3_json_u64(&d, k3_json_object_get(&d, metadata, "total_size"), &total) ||
        total != GLM53_INDEX_TOTAL_BYTES ||
        !token_type(&d, weights, K3_JSON_OBJECT) ||
        d.tokens[weights].size != GLM53_INDEX_TENSOR_COUNT) {
        fail(error, error_size, "invalid GLM index dimensions");
        goto done;
    }
    manifest->entries = (glm53_index_entry *)calloc(
        GLM53_INDEX_TENSOR_COUNT, sizeof(*manifest->entries));
    if (!manifest->entries) goto done;
    manifest->entry_count = GLM53_INDEX_TENSOR_COUNT;
    manifest->total_size = total;
    bool seen[GLM53_SHARD_COUNT] = { false };
    int32_t key = d.tokens[weights].first_child;
    for (size_t i = 0u; i < manifest->entry_count; i++) {
        if (key < 0) goto done;
        const int32_t value = d.tokens[key].next_sibling;
        if (value < 0 ||
            !k3_json_string_dup(&d, key, &manifest->entries[i].name,
                                error, error_size) ||
            manifest->entries[i].name[0] == '\0') goto done;
        char *file = NULL;
        if (!k3_json_string_dup(&d, value, &file, error, error_size) ||
            !shard_name(file, &manifest->entries[i].shard)) {
            free(file);
            fail(error, error_size, "unsafe or malformed shard name");
            goto done;
        }
        free(file);
        seen[manifest->entries[i].shard] = true;
        key = d.tokens[value].next_sibling;
    }
    if (key >= 0) goto done;
    for (size_t i = 0u; i < GLM53_SHARD_COUNT; i++) {
        if (!seen[i]) { fail(error, error_size, "index shard family has a gap"); goto done; }
    }
    qsort(manifest->entries, manifest->entry_count,
          sizeof(*manifest->entries), compare_entries);
    for (size_t i = 1u; i < manifest->entry_count; i++) {
        if (strcmp(manifest->entries[i - 1u].name,
                   manifest->entries[i].name) == 0) {
            fail(error, error_size, "duplicate tensor name in index");
            goto done;
        }
    }
    ok = true;
done:
    k3_json_document_free(&d);
    if (!ok && (!error || !error[0])) fail(error, error_size, "invalid GLM index");
    return ok;
}

void glm53_manifest_free(glm53_manifest *manifest) {
    if (!manifest) return;
    for (size_t i = 0u; i < manifest->module_count; i++)
        free(manifest->modules_to_not_convert[i]);
    for (size_t i = 0u; i < manifest->entry_count; i++)
        free(manifest->entries[i].name);
    free(manifest->modules_to_not_convert);
    free(manifest->entries);
    memset(manifest, 0, sizeof(*manifest));
}

bool glm53_manifest_parse(glm53_manifest *manifest,
                          const char *config_json, size_t config_size,
                          const char *index_json, size_t index_size,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!manifest || !config_json || !index_json) {
        fail(error, error_size, "invalid manifest parse arguments");
        return false;
    }
    memset(manifest, 0, sizeof(*manifest));
    if (!parse_config(manifest, config_json, config_size, error, error_size) ||
        !parse_index(manifest, index_json, index_size, error, error_size)) {
        glm53_manifest_free(manifest);
        return false;
    }
    return true;
}

static bool read_file(const char *path, char **data, size_t *size,
                      char *error, size_t error_size) {
    *data = NULL; *size = 0u;
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < 0 ||
        (uint64_t)st.st_size > SIZE_MAX || st.st_size > 128 * 1024 * 1024) {
        fail(error, error_size, "%s: invalid file size", path);
        return false;
    }
    FILE *file = fopen(path, "rb");
    if (!file) { fail(error, error_size, "%s: %s", path, strerror(errno)); return false; }
    *size = (size_t)st.st_size;
    *data = (char *)malloc(*size ? *size : 1u);
    const bool ok = *data && fread(*data, 1u, *size, file) == *size &&
                    fgetc(file) == EOF && !ferror(file);
    fclose(file);
    if (!ok) { free(*data); *data = NULL; fail(error, error_size, "%s: read failed", path); }
    return ok;
}

bool glm53_manifest_load(glm53_manifest *manifest, const char *root,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!manifest || !root || root[0] == '\0') return false;
    const size_t bytes = strlen(root) + 40u;
    char *config_path = (char *)malloc(bytes);
    char *index_path = (char *)malloc(bytes);
    char *config = NULL, *index = NULL;
    size_t config_size = 0u, index_size = 0u;
    bool ok = config_path && index_path;
    if (ok) {
        snprintf(config_path, bytes, "%s/config.json", root);
        snprintf(index_path, bytes, "%s/model.safetensors.index.json", root);
        ok = read_file(config_path, &config, &config_size, error, error_size) &&
             read_file(index_path, &index, &index_size, error, error_size) &&
             glm53_manifest_parse(manifest, config, config_size,
                                  index, index_size, error, error_size);
    }
    if (!ok && (!error || !error[0])) fail(error, error_size, "manifest load failed");
    free(config_path); free(index_path); free(config); free(index);
    return ok;
}

static bool component_match(const char *candidate, size_t candidate_size,
                            const char *selector) {
    const size_t n = strlen(selector);
    if (n == 0u || n > candidate_size) return false;
    for (size_t at = 0u; at + n <= candidate_size; at++) {
        if ((at == 0u || candidate[at - 1u] == '.') &&
            (at + n == candidate_size || candidate[at + n] == '.') &&
            memcmp(candidate + at, selector, n) == 0) return true;
    }
    return false;
}

bool glm53_manifest_tensor_excluded(const glm53_manifest *manifest,
                                    const char *tensor_name) {
    if (!manifest || !tensor_name) return false;
    const char *source = tensor_name;
    const char *prefix = "";
    if (strncmp(source, "model.language_model.", 21u) == 0) {
        prefix = "model."; source += 21u;
    } else if (strncmp(source, "model.visual.", 13u) == 0) {
        prefix = "visual."; source += 13u;
    }
    const size_t full_size = strlen(prefix) + strlen(source);
    char *normalized = (char *)malloc(full_size + 1u);
    if (!normalized) return false;
    strcpy(normalized, prefix); strcat(normalized, source);
    size_t owner_size = full_size;
    static const char *terminal[] = { ".weight_scale_inv", ".weight", ".bias" };
    for (size_t i = 0u; i < 3u; i++) {
        const size_t n = strlen(terminal[i]);
        if (owner_size >= n && strcmp(normalized + owner_size - n, terminal[i]) == 0) {
            owner_size -= n; break;
        }
    }
    bool matched = false;
    for (size_t i = 0u; i < manifest->module_count && !matched; i++) {
        matched = component_match(normalized, full_size,
                                  manifest->modules_to_not_convert[i]) ||
                  component_match(normalized, owner_size,
                                  manifest->modules_to_not_convert[i]);
    }
    free(normalized);
    return matched;
}

static const glm53_index_entry *find_entry(const glm53_manifest *manifest,
                                           const char *name) {
    glm53_index_entry key = { (char *)name, 0u };
    return (const glm53_index_entry *)bsearch(
        &key, manifest->entries, manifest->entry_count,
        sizeof(*manifest->entries), compare_entries);
}

static uint64_t ceil_div_128(uint64_t value) {
    return value / UINT64_C(128) + (value % UINT64_C(128) != 0u);
}

static bool scale_shape(const k3_st_tensor *weight,
                        const k3_st_tensor *scale) {
    return weight->ndim == 2u && scale->ndim == 2u &&
        scale->shape[0] == ceil_div_128(weight->shape[0]) &&
        scale->shape[1] == ceil_div_128(weight->shape[1]);
}

typedef struct {
    const k3_st_tensor *tensor;
    uint64_t end;
} tensor_range;

static int compare_tensor_pointers(const void *left, const void *right) {
    const k3_st_tensor *const *a = (const k3_st_tensor *const *)left;
    const k3_st_tensor *const *b = (const k3_st_tensor *const *)right;
    return strcmp((*a)->name, (*b)->name);
}

static int compare_ranges(const void *left, const void *right) {
    const tensor_range *a = (const tensor_range *)left;
    const tensor_range *b = (const tensor_range *)right;
    if (a->tensor->shard != b->tensor->shard)
        return a->tensor->shard < b->tensor->shard ? -1 : 1;
    if (a->tensor->physical_offset != b->tensor->physical_offset)
        return a->tensor->physical_offset < b->tensor->physical_offset ? -1 : 1;
    if (a->end != b->end) return a->end < b->end ? -1 : 1;
    return strcmp(a->tensor->name, b->tensor->name);
}

static const k3_st_tensor *find_sorted_tensor(
        const k3_st_tensor *const *sorted, size_t count, const char *name) {
    size_t low = 0u, high = count;
    while (low < high) {
        const size_t middle = low + (high - low) / 2u;
        const int order = strcmp(sorted[middle]->name, name);
        if (order < 0) low = middle + 1u;
        else high = middle;
    }
    return low < count && strcmp(sorted[low]->name, name) == 0 ?
        sorted[low] : NULL;
}

bool glm53_manifest_official_dtype_counts_valid(size_t tensor_count,
                                                 size_t f8_count,
                                                 size_t f32_count,
                                                 size_t bf16_count,
                                                 size_t scale_count) {
    return tensor_count == GLM53_INDEX_TENSOR_COUNT &&
           f8_count == 37338u && f32_count == 37629u &&
           bf16_count == 1141u && scale_count == 37338u;
}

bool glm53_manifest_reconcile(const glm53_manifest *manifest,
                              const k3_st_model *model,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!manifest || !model || model->shard_count != GLM53_SHARD_COUNT ||
        model->tensor_count != manifest->entry_count ||
        (model->tensor_count != 0u && !model->tensors) || !model->shards) {
        fail(error, error_size, "header/index tensor or shard count mismatch");
        return false;
    }
    const size_t count = model->tensor_count;
    if (count > SIZE_MAX / sizeof(const k3_st_tensor *) ||
        count > SIZE_MAX / sizeof(tensor_range)) {
        fail(error, error_size, "reconciliation view size overflow");
        return false;
    }
    const k3_st_tensor **by_name = count ?
        (const k3_st_tensor **)malloc(count * sizeof(*by_name)) : NULL;
    tensor_range *ranges = count ?
        (tensor_range *)malloc(count * sizeof(*ranges)) : NULL;
    if (count && (!by_name || !ranges)) {
        free(by_name); free(ranges);
        fail(error, error_size, "allocating reconciliation views failed");
        return false;
    }
    bool ok = false;
    size_t f8_count = 0u, f32_count = 0u, bf16_count = 0u, scale_count = 0u;
    for (size_t i = 0u; i < count; i++) {
        const k3_st_tensor *tensor = &model->tensors[i];
        if (!tensor->name || tensor->name[0] == '\0' ||
            tensor->shard >= model->shard_count || tensor->byte_length == 0u ||
            tensor->physical_offset > UINT64_MAX - tensor->byte_length) {
            fail(error, error_size, "invalid tensor physical range");
            goto done;
        }
        const uint64_t end = tensor->physical_offset + tensor->byte_length;
        if (tensor->physical_offset <
                model->shards[tensor->shard].data_offset ||
            end > model->shards[tensor->shard].file_bytes) {
            fail(error, error_size, "tensor physical range exceeds shard");
            goto done;
        }
        switch (tensor->dtype) {
        case K3_ST_DTYPE_F8_E4M3: f8_count++; break;
        case K3_ST_DTYPE_F32: f32_count++; break;
        case K3_ST_DTYPE_BF16: bf16_count++; break;
        default:
            fail(error, error_size, "unsupported GLM tensor dtype");
            goto done;
        }
        by_name[i] = tensor;
        ranges[i].tensor = tensor;
        ranges[i].end = end;
    }
    qsort(by_name, count, sizeof(*by_name), compare_tensor_pointers);
    for (size_t i = 1u; i < count; i++) {
        if (strcmp(by_name[i - 1u]->name, by_name[i]->name) == 0) {
            fail(error, error_size, "duplicate tensor name in model metadata");
            goto done;
        }
    }
    qsort(ranges, count, sizeof(*ranges), compare_ranges);
    for (size_t i = 1u; i < count; i++) {
        if (ranges[i - 1u].tensor->shard == ranges[i].tensor->shard &&
            ranges[i].tensor->physical_offset < ranges[i - 1u].end) {
            fail(error, error_size, "overlapping tensor physical ranges");
            goto done;
        }
    }

    uint64_t total = 0u;
    for (size_t i = 0u; i < count; i++) {
        const k3_st_tensor *tensor = by_name[i];
        const glm53_index_entry *entry = find_entry(manifest, tensor->name);
        if (!entry || entry->shard != tensor->shard ||
            total > UINT64_MAX - tensor->byte_length) {
            fail(error, error_size, "header/index mismatch at %s", tensor->name);
            goto done;
        }
        total += tensor->byte_length;
        const size_t name_size = strlen(tensor->name);
        const char scale_suffix[] = "_scale_inv";
        const size_t suffix_size = sizeof(scale_suffix) - 1u;
        if (name_size >= suffix_size &&
            strcmp(tensor->name + name_size - suffix_size, scale_suffix) == 0) {
            scale_count++;
            char *base = strdup(tensor->name);
            if (!base) goto done;
            base[name_size - suffix_size] = '\0';
            const k3_st_tensor *weight = find_sorted_tensor(
                by_name, count, base);
            const bool valid = tensor->dtype == K3_ST_DTYPE_F32 && weight &&
                weight->dtype == K3_ST_DTYPE_F8_E4M3 &&
                weight->shard == tensor->shard && scale_shape(weight, tensor) &&
                !glm53_manifest_tensor_excluded(manifest, base);
            free(base);
            if (!valid) {
                fail(error, error_size, "invalid FP8 scale companion");
                goto done;
            }
        } else if (glm53_manifest_tensor_excluded(manifest, tensor->name)) {
            if (tensor->dtype != K3_ST_DTYPE_BF16 &&
                tensor->dtype != K3_ST_DTYPE_F32) {
                fail(error, error_size,
                     "excluded tensor is not source precision");
                goto done;
            }
        } else {
            const char suffix[] = "_scale_inv";
            char *scale_name = (char *)malloc(name_size + sizeof(suffix));
            if (!scale_name) goto done;
            memcpy(scale_name, tensor->name, name_size);
            memcpy(scale_name + name_size, suffix, sizeof(suffix));
            const k3_st_tensor *scale = find_sorted_tensor(
                by_name, count, scale_name);
            const bool valid = tensor->dtype == K3_ST_DTYPE_F8_E4M3 && scale &&
                scale->dtype == K3_ST_DTYPE_F32 &&
                scale->shard == tensor->shard && scale_shape(tensor, scale);
            free(scale_name);
            if (!valid) {
                fail(error, error_size, "invalid FP8 weight/scale pair");
                goto done;
            }
        }
    }
    if (total != manifest->total_size) {
        fail(error, error_size,
             "header byte sum does not match index total_size");
        goto done;
    }
    if (count == GLM53_INDEX_TENSOR_COUNT &&
        !glm53_manifest_official_dtype_counts_valid(
            count, f8_count, f32_count, bf16_count, scale_count)) {
        fail(error, error_size, "official GLM dtype counts do not match");
        goto done;
    }
    ok = true;
done:
    free(by_name);
    free(ranges);
    return ok;
}
