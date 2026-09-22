#include "mimo26_manifest.h"

#include "k3_json.h"
#include "mimo26_architecture.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool fail(char *error, size_t size, const char *format, ...)
{
    if (error != NULL && size > 0) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return false;
}

static bool token_is(const k3_json_document *document, int32_t token,
                     k3_json_type type)
{
    return token >= 0 && (size_t)token < document->token_count &&
           document->tokens[token].type == type;
}

/*
 * A shard name must be a plain basename inside the checkpoint directory.
 * Anything with a separator, a parent reference, a leading dash or a wrong
 * suffix is rejected rather than sanitized.
 */
static bool shard_name_safe(const char *name)
{
    if (name == NULL || name[0] == '\0' || name[0] == '/' || name[0] == '-') {
        return false;
    }
    if (strchr(name, '/') != NULL || strchr(name, '\\') != NULL) {
        return false;
    }
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) {
        return false;
    }
    const size_t length = strlen(name);
    static const char suffix[] = ".safetensors";
    const size_t suffix_length = sizeof suffix - 1u;
    if (length <= suffix_length ||
        strcmp(name + length - suffix_length, suffix) != 0) {
        return false;
    }
    return true;
}

static int compare_strings(const void *left, const void *right)
{
    return strcmp(*(const char *const *)left, *(const char *const *)right);
}

static int compare_entries(const void *left, const void *right)
{
    const mimo26_index_entry *a = left;
    const mimo26_index_entry *b = right;
    return strcmp(a->name, b->name);
}

static int32_t find_shard(char **names, size_t count, const char *name)
{
    size_t low = 0;
    size_t high = count;
    while (low < high) {
        const size_t mid = low + (high - low) / 2u;
        const int order = strcmp(names[mid], name);
        if (order == 0) {
            return (int32_t)mid;
        }
        if (order < 0) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return -1;
}

void mimo26_manifest_free(mimo26_manifest *manifest)
{
    if (manifest == NULL) {
        return;
    }
    for (size_t i = 0; i < manifest->shard_count; i++) {
        free(manifest->shard_names[i]);
    }
    free(manifest->shard_names);
    for (size_t i = 0; i < manifest->entry_count; i++) {
        free(manifest->entries[i].name);
    }
    free(manifest->entries);
    free(manifest->save_format);
    memset(manifest, 0, sizeof *manifest);
}

bool mimo26_manifest_parse(mimo26_manifest *manifest,
                           const char *index_json,
                           size_t index_size,
                           char *error,
                           size_t error_size)
{
    if (manifest == NULL || index_json == NULL) {
        return fail(error, error_size, "invalid manifest parse arguments");
    }
    memset(manifest, 0, sizeof *manifest);

    k3_json_document document;
    memset(&document, 0, sizeof document);
    if (!k3_json_parse(&document, index_json, index_size, error, error_size)) {
        return false;
    }

    bool ok = false;
    const int32_t metadata = k3_json_object_get(&document, document.root,
                                                "metadata");
    const int32_t weights = k3_json_object_get(&document, document.root,
                                               "weight_map");
    if (!token_is(&document, document.root, K3_JSON_OBJECT) ||
        !token_is(&document, metadata, K3_JSON_OBJECT) ||
        !token_is(&document, weights, K3_JSON_OBJECT)) {
        fail(error, error_size, "index is missing metadata or weight_map");
        goto done;
    }

    if (!k3_json_u64(&document,
                     k3_json_object_get(&document, metadata, "total_size"),
                     &manifest->total_size)) {
        fail(error, error_size, "index metadata.total_size is not an integer");
        goto done;
    }
    if (!k3_json_u32(&document,
                     k3_json_object_get(&document, metadata, "tp_size"),
                     &manifest->tp_size)) {
        fail(error, error_size, "index metadata.tp_size is not an integer");
        goto done;
    }
    if (!k3_json_string_dup(&document,
                            k3_json_object_get(&document, metadata,
                                               "save_format"),
                            &manifest->save_format, error, error_size)) {
        goto done;
    }
    if (strcmp(manifest->save_format, MIMO26_INDEX_SAVE_FORMAT) != 0) {
        fail(error, error_size, "index save_format is %s, expected %s",
             manifest->save_format, MIMO26_INDEX_SAVE_FORMAT);
        goto done;
    }

    const size_t count = document.tokens[weights].size;
    if (count == 0) {
        fail(error, error_size, "weight_map is empty");
        goto done;
    }
    manifest->entries = calloc(count, sizeof *manifest->entries);
    manifest->shard_names = calloc(count, sizeof *manifest->shard_names);
    if (manifest->entries == NULL || manifest->shard_names == NULL) {
        fail(error, error_size, "out of memory allocating manifest");
        goto done;
    }

    /*
     * Object children alternate key and value as siblings. Shard names are
     * appended in first-seen order and bound immediately; they are sorted at
     * the end and the bindings remapped, so shard indices are deterministic
     * without walking the document twice.
     */
    for (int32_t key = document.tokens[weights].first_child; key >= 0;) {
        const int32_t value = document.tokens[key].next_sibling;
        if (!token_is(&document, key, K3_JSON_STRING) ||
            !token_is(&document, value, K3_JSON_STRING)) {
            fail(error, error_size, "weight_map entry is not a string pair");
            goto done;
        }
        if (manifest->entry_count >= count) {
            fail(error, error_size, "weight_map has more entries than declared");
            goto done;
        }
        char *tensor = NULL;
        char *shard = NULL;
        if (!k3_json_string_dup(&document, key, &tensor, error, error_size)) {
            goto done;
        }
        if (tensor[0] == '\0') {
            fail(error, error_size, "weight_map has an empty tensor name");
            free(tensor);
            goto done;
        }
        if (!k3_json_string_dup(&document, value, &shard, error, error_size)) {
            free(tensor);
            goto done;
        }
        if (!shard_name_safe(shard)) {
            fail(error, error_size, "unsafe shard name for %s", tensor);
            free(tensor);
            free(shard);
            goto done;
        }

        size_t which = manifest->shard_count;
        for (size_t i = 0; i < manifest->shard_count; i++) {
            if (strcmp(manifest->shard_names[i], shard) == 0) {
                which = i;
                break;
            }
        }
        if (which == manifest->shard_count) {
            manifest->shard_names[manifest->shard_count++] = shard;
        } else {
            free(shard);
        }
        manifest->entries[manifest->entry_count].name = tensor;
        manifest->entries[manifest->entry_count].shard = (uint16_t)which;
        manifest->entry_count++;

        key = document.tokens[value].next_sibling;
    }
    if (manifest->entry_count != count) {
        fail(error, error_size, "weight_map declared %zu entries, walked %zu",
             count, manifest->entry_count);
        goto done;
    }

    /* Sort shard names, then remap every binding through the permutation. */
    {
        char **ordered = calloc(manifest->shard_count, sizeof *ordered);
        uint16_t *remap = calloc(manifest->shard_count, sizeof *remap);
        if (ordered == NULL || remap == NULL) {
            free(ordered);
            free(remap);
            fail(error, error_size, "out of memory ordering shard names");
            goto done;
        }
        memcpy(ordered, manifest->shard_names,
               manifest->shard_count * sizeof *ordered);
        qsort(ordered, manifest->shard_count, sizeof *ordered,
              compare_strings);
        for (size_t i = 0; i < manifest->shard_count; i++) {
            const int32_t position = find_shard(ordered, manifest->shard_count,
                                                manifest->shard_names[i]);
            if (position < 0) {
                free(ordered);
                free(remap);
                fail(error, error_size, "shard name lost while ordering");
                goto done;
            }
            remap[i] = (uint16_t)position;
        }
        for (size_t i = 0; i < manifest->entry_count; i++) {
            manifest->entries[i].shard = remap[manifest->entries[i].shard];
        }
        memcpy(manifest->shard_names, ordered,
               manifest->shard_count * sizeof *ordered);
        free(ordered);
        free(remap);
    }

    qsort(manifest->entries, manifest->entry_count,
          sizeof *manifest->entries, compare_entries);
    for (size_t i = 1; i < manifest->entry_count; i++) {
        if (strcmp(manifest->entries[i - 1u].name,
                   manifest->entries[i].name) == 0) {
            fail(error, error_size, "duplicate tensor name in index: %s",
                 manifest->entries[i].name);
            goto done;
        }
    }

    ok = true;
done:
    k3_json_document_free(&document);
    if (!ok) {
        mimo26_manifest_free(manifest);
    }
    return ok;
}

static bool read_file(const char *path, char **data, size_t *size,
                      char *error, size_t error_size)
{
    *data = NULL;
    *size = 0;
    FILE *handle = fopen(path, "rb");
    if (handle == NULL) {
        return fail(error, error_size, "cannot open %s", path);
    }
    if (fseek(handle, 0, SEEK_END) != 0) {
        fclose(handle);
        return fail(error, error_size, "cannot seek %s", path);
    }
    const long length = ftell(handle);
    if (length < 0) {
        fclose(handle);
        return fail(error, error_size, "cannot size %s", path);
    }
    rewind(handle);
    char *buffer = malloc((size_t)length + 1u);
    if (buffer == NULL) {
        fclose(handle);
        return fail(error, error_size, "out of memory reading %s", path);
    }
    if (fread(buffer, 1u, (size_t)length, handle) != (size_t)length) {
        free(buffer);
        fclose(handle);
        return fail(error, error_size, "short read of %s", path);
    }
    fclose(handle);
    buffer[length] = '\0';
    *data = buffer;
    *size = (size_t)length;
    return true;
}

bool mimo26_manifest_load(mimo26_manifest *manifest,
                          const char *root,
                          char *error,
                          size_t error_size)
{
    if (manifest == NULL || root == NULL) {
        return fail(error, error_size, "invalid manifest load arguments");
    }
    const size_t path_bytes = strlen(root) + 64u;
    char *path = malloc(path_bytes);
    if (path == NULL) {
        return fail(error, error_size, "out of memory allocating path");
    }
    snprintf(path, path_bytes, "%s/model.safetensors.index.json", root);

    char *json = NULL;
    size_t size = 0;
    bool ok = read_file(path, &json, &size, error, error_size) &&
              mimo26_manifest_parse(manifest, json, size, error, error_size);
    free(json);
    free(path);
    return ok;
}

int32_t mimo26_manifest_shard_of(const mimo26_manifest *manifest,
                                 const char *tensor_name)
{
    if (manifest == NULL || tensor_name == NULL) {
        return -1;
    }
    size_t low = 0;
    size_t high = manifest->entry_count;
    while (low < high) {
        const size_t mid = low + (high - low) / 2u;
        const int order = strcmp(manifest->entries[mid].name, tensor_name);
        if (order == 0) {
            return (int32_t)manifest->entries[mid].shard;
        }
        if (order < 0) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return -1;
}

bool mimo26_manifest_open_model(const mimo26_manifest *manifest,
                                const char *root,
                                k3_st_model *model,
                                char *error,
                                size_t error_size)
{
    if (manifest == NULL || root == NULL || model == NULL ||
        manifest->shard_count == 0) {
        return fail(error, error_size, "invalid manifest open arguments");
    }
    char **paths = calloc(manifest->shard_count, sizeof *paths);
    if (paths == NULL) {
        return fail(error, error_size, "out of memory allocating shard paths");
    }
    bool ok = true;
    for (size_t i = 0; i < manifest->shard_count && ok; i++) {
        const size_t bytes = strlen(root) + strlen(manifest->shard_names[i]) +
                             2u;
        paths[i] = malloc(bytes);
        if (paths[i] == NULL) {
            ok = fail(error, error_size, "out of memory allocating shard path");
            break;
        }
        snprintf(paths[i], bytes, "%s/%s", root, manifest->shard_names[i]);
    }
    if (ok) {
        ok = k3_st_model_open_paths(model, (const char *const *)paths,
                                    manifest->shard_count, error, error_size);
    }
    for (size_t i = 0; i < manifest->shard_count; i++) {
        free(paths[i]);
    }
    free(paths);
    return ok;
}

bool mimo26_manifest_reconcile(const mimo26_manifest *manifest,
                               const k3_st_model *model,
                               char *error,
                               size_t error_size)
{
    if (manifest == NULL || model == NULL || model->tensors == NULL) {
        return fail(error, error_size, "invalid reconcile arguments");
    }
    if (model->tensor_count != manifest->entry_count) {
        return fail(error, error_size,
                    "header tensor count %zu, index lists %zu",
                    model->tensor_count, manifest->entry_count);
    }

    uint64_t payload = 0;
    for (size_t i = 0; i < model->tensor_count; i++) {
        const k3_st_tensor *tensor = &model->tensors[i];
        const int32_t shard = mimo26_manifest_shard_of(manifest, tensor->name);
        if (shard < 0) {
            return fail(error, error_size, "header tensor %s is not indexed",
                        tensor->name);
        }
        if ((uint16_t)shard != tensor->shard) {
            return fail(error, error_size,
                        "%s: index says shard %d, header says %u",
                        tensor->name, shard, (unsigned)tensor->shard);
        }
        if (payload > UINT64_MAX - tensor->byte_length) {
            return fail(error, error_size, "payload byte sum overflows");
        }
        payload += tensor->byte_length;
    }
    if (payload != manifest->total_size) {
        return fail(error, error_size,
                    "payload byte sum %llu, index total_size %llu",
                    (unsigned long long)payload,
                    (unsigned long long)manifest->total_size);
    }
    return true;
}

static const char *const projection_names[MIMO26_EXPERT_PROJECTION_COUNT] = {
    "gate_proj", "up_proj", "down_proj",
};

bool mimo26_expert_spans(const k3_st_model *model,
                         uint32_t layer,
                         uint32_t expert,
                         mimo26_expert_span spans[MIMO26_EXPERT_PROJECTION_COUNT],
                         char *error,
                         size_t error_size)
{
    if (model == NULL || spans == NULL) {
        return fail(error, error_size, "invalid expert span arguments");
    }
    memset(spans, 0, sizeof *spans * MIMO26_EXPERT_PROJECTION_COUNT);

    const mimo26_layer_kind kind = mimo26_architecture_layer_kind(layer);
    if (kind != MIMO26_LAYER_MOE_SWA && kind != MIMO26_LAYER_MOE_GLOBAL) {
        return fail(error, error_size, "layer %u has no routed experts", layer);
    }
    if (expert >= MIMO26_ROUTED_EXPERTS_PER_LAYER) {
        return fail(error, error_size, "expert %u out of range", expert);
    }

    for (size_t i = 0; i < MIMO26_EXPERT_PROJECTION_COUNT; i++) {
        char name[160];
        snprintf(name, sizeof name,
                 "model.layers.%u.mlp.experts.%u.%s.weight",
                 layer, expert, projection_names[i]);
        const k3_st_tensor *weight = k3_st_find(model, name);
        if (weight == NULL) {
            return fail(error, error_size, "missing %s", name);
        }
        snprintf(name, sizeof name,
                 "model.layers.%u.mlp.experts.%u.%s.weight_scale",
                 layer, expert, projection_names[i]);
        const k3_st_tensor *scale = k3_st_find(model, name);
        if (scale == NULL) {
            return fail(error, error_size, "missing %s", name);
        }
        /* Weight and scale are one cache identity, so they must be colocated
         * for a single-shard lease to be possible. */
        if (weight->shard != scale->shard) {
            return fail(error, error_size,
                        "layer %u expert %u %s spans shards %u and %u",
                        layer, expert, projection_names[i],
                        (unsigned)weight->shard, (unsigned)scale->shard);
        }
        spans[i].shard = weight->shard;
        spans[i].weight_offset = weight->physical_offset;
        spans[i].weight_bytes = weight->byte_length;
        spans[i].scale_offset = scale->physical_offset;
        spans[i].scale_bytes = scale->byte_length;
    }
    return true;
}

uint64_t mimo26_expert_bytes(void)
{
    /* gate and up: 2048*2048 packed + 2048*128 scales; down: 4096*1024 packed
     * + 4096*64 scales. 12.75 MiB in total. */
    const uint64_t gate_up = (2048u * 2048u) + (2048u * 128u);
    const uint64_t down = (4096u * 1024u) + (4096u * 64u);
    return 2u * gate_up + down;
}
