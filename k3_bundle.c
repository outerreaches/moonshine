#include "k3_bundle.h"

#include "k3_json.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

enum { K3_BUNDLE_MANIFEST_MAX = 1024 * 1024 };

static const char *K3_BUNDLE_SCHEMA = "moonshine-k3-mzg2-bundle-v1";
static const char *K3_BUNDLE_BASE_MODEL = "moonshotai/Kimi-K3";
static const char *K3_BUNDLE_BASE_REVISION =
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569";
static const char *K3_BUNDLE_SOURCE_MANIFEST =
    "476fa0ba64e3233cbb9ca0642327361a73f6807e751edb071c92fa2216b202a4";
static const char *K3_BUNDLE_EXPERT_NAMES[K3_BUNDLE_EXPERT_TENSORS] = {
    "w1.weight_packed", "w1.weight_scale",
    "w2.weight_packed", "w2.weight_scale",
    "w3.weight_packed", "w3.weight_scale",
};
static const uint64_t K3_BUNDLE_EXPERT_OFFSETS[K3_BUNDLE_EXPERT_TENSORS] = {
    UINT64_C(0), UINT64_C(5505024), UINT64_C(5849088),
    UINT64_C(11354112), UINT64_C(11698176), UINT64_C(17203200),
};
static const uint64_t K3_BUNDLE_EXPERT_SIZES[K3_BUNDLE_EXPERT_TENSORS] = {
    UINT64_C(5505024), UINT64_C(344064), UINT64_C(5505024),
    UINT64_C(344064), UINT64_C(5505024), UINT64_C(344064),
};

static void bundle_error(char *error, size_t error_size,
                         const char *format, ...) {
    if (!error || error_size == 0u) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static bool path_join(char output[PATH_MAX], const char *root,
                      const char *relative) {
    if (!root || !relative || relative[0] == '\0' || relative[0] == '/' ||
        strstr(relative, "//") != NULL) {
        return false;
    }
    const char *cursor = relative;
    while (*cursor) {
        const char *end = strchr(cursor, '/');
        const size_t size = end ? (size_t)(end - cursor) : strlen(cursor);
        if (size == 0u || (size == 1u && cursor[0] == '.') ||
            (size == 2u && cursor[0] == '.' && cursor[1] == '.')) {
            return false;
        }
        if (!end) break;
        cursor = end + 1u;
    }
    const int length = snprintf(output, PATH_MAX, "%s/%s", root, relative);
    return length > 0 && length < PATH_MAX;
}

static bool read_manifest(const char *path, char **text, size_t *text_size,
                          char *error, size_t error_size) {
    *text = NULL;
    *text_size = 0u;
    const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        bundle_error(error, error_size, "open bundle manifest: %s",
                     strerror(errno));
        return false;
    }
    struct stat status;
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size <= 0 || status.st_size > K3_BUNDLE_MANIFEST_MAX) {
        close(fd);
        bundle_error(error, error_size, "invalid bundle manifest file");
        return false;
    }
    char *buffer = (char *)malloc((size_t)status.st_size + 1u);
    if (!buffer) {
        close(fd);
        bundle_error(error, error_size, "allocate bundle manifest");
        return false;
    }
    size_t used = 0u;
    while (used < (size_t)status.st_size) {
        const ssize_t count = read(fd, buffer + used,
                                   (size_t)status.st_size - used);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            free(buffer);
            close(fd);
            bundle_error(error, error_size, "read bundle manifest");
            return false;
        }
        used += (size_t)count;
    }
    close(fd);
    buffer[used] = '\0';
    *text = buffer;
    *text_size = used;
    return true;
}

static bool required_string(const k3_json_document *document, int32_t object,
                            const char *key, char **value,
                            char *error, size_t error_size) {
    const int32_t token = k3_json_object_get(document, object, key);
    if (!k3_json_string_dup(document, token, value, error, error_size)) {
        bundle_error(error, error_size, "bundle field %s must be a string", key);
        return false;
    }
    return true;
}

static bool required_u32(const k3_json_document *document, int32_t object,
                         const char *key, uint32_t *value) {
    return k3_json_u32(document, k3_json_object_get(document, object, key), value);
}

static bool required_u64(const k3_json_document *document, int32_t object,
                         const char *key, uint64_t *value) {
    return k3_json_u64(document, k3_json_object_get(document, object, key), value);
}

static bool parse_hex_u64(const char *text, uint64_t *value) {
    if (!text || strlen(text) != 16u || !value) return false;
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 16);
    if (errno != 0 || end != text + 16u) return false;
    *value = (uint64_t)parsed;
    return true;
}

static bool is_sha256(const char *text) {
    if (!text || strlen(text) != 64u) return false;
    for (size_t index = 0u; index < 64u; index++) {
        const char value = text[index];
        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    return true;
}

static bool regular_file_size(const char *path, uint64_t expected) {
    struct stat status;
    return lstat(path, &status) == 0 && S_ISREG(status.st_mode) &&
        (uint64_t)status.st_size == expected;
}

bool k3_bundle_detect(const char *root, bool *present,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!root || !present) {
        bundle_error(error, error_size, "invalid bundle detection arguments");
        return false;
    }
    char path[PATH_MAX];
    if (!path_join(path, root, "moonshine-bundle.json")) {
        bundle_error(error, error_size, "bundle manifest path overflow");
        return false;
    }
    struct stat status;
    if (lstat(path, &status) != 0) {
        if (errno == ENOENT) {
            *present = false;
            return true;
        }
        bundle_error(error, error_size, "stat bundle manifest: %s",
                     strerror(errno));
        return false;
    }
    if (!S_ISREG(status.st_mode)) {
        bundle_error(error, error_size, "bundle manifest is not regular");
        return false;
    }
    *present = true;
    return true;
}

bool k3_bundle_load(k3_bundle *bundle, const char *root,
                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!bundle || !root) {
        bundle_error(error, error_size, "invalid bundle load arguments");
        return false;
    }
    memset(bundle, 0, sizeof(*bundle));
    char manifest_path[PATH_MAX];
    if (!path_join(manifest_path, root, "moonshine-bundle.json")) {
        bundle_error(error, error_size, "bundle manifest path overflow");
        return false;
    }
    char *text = NULL;
    size_t text_size = 0u;
    k3_json_document document;
    memset(&document, 0, sizeof(document));
    char *schema = NULL, *base_model = NULL, *base_revision = NULL;
    char *source_manifest = NULL, *layout_crc = NULL;
    char *static_path = NULL, *static_sha = NULL;
    char *routed_format = NULL, *routed_path = NULL, *routed_sha = NULL;
    bool ok = read_manifest(manifest_path, &text, &text_size, error, error_size) &&
        k3_json_parse(&document, text, text_size, error, error_size);
    if (!ok || document.root < 0 ||
        document.tokens[document.root].type != K3_JSON_OBJECT) goto cleanup;
    uint32_t version = 0u;
    ok = required_string(&document, document.root, "schema", &schema, error, error_size) &&
        required_u32(&document, document.root, "version", &version) &&
        required_string(&document, document.root, "base_model", &base_model, error, error_size) &&
        required_string(&document, document.root, "base_revision", &base_revision, error, error_size) &&
        required_string(&document, document.root, "source_manifest_sha256", &source_manifest, error, error_size) &&
        required_string(&document, document.root, "source_model_layout_crc64", &layout_crc, error, error_size) &&
        strcmp(schema, K3_BUNDLE_SCHEMA) == 0 && version == 1u &&
        strcmp(base_model, K3_BUNDLE_BASE_MODEL) == 0 &&
        strcmp(base_revision, K3_BUNDLE_BASE_REVISION) == 0 &&
        strcmp(source_manifest, K3_BUNDLE_SOURCE_MANIFEST) == 0 &&
        parse_hex_u64(layout_crc, &bundle->source_model_layout_crc64) &&
        bundle->source_model_layout_crc64 ==
            K3_BUNDLE_MODEL_LAYOUT_CRC64;
    const int32_t static_store = k3_json_object_get(
        &document, document.root, "static_store");
    ok = ok && static_store >= 0 &&
        document.tokens[static_store].type == K3_JSON_OBJECT &&
        required_string(&document, static_store, "path", &static_path, error, error_size) &&
        required_u32(&document, static_store, "tensor_count", &bundle->static_tensor_count) &&
        required_u64(&document, static_store, "payload_bytes", &bundle->static_payload_bytes) &&
        required_u64(&document, static_store, "file_bytes", &bundle->static_file_bytes) &&
        required_string(&document, static_store, "sha256", &static_sha, error, error_size) &&
        bundle->static_tensor_count == K3_BUNDLE_STATIC_TENSORS &&
        bundle->static_payload_bytes == K3_BUNDLE_STATIC_PAYLOAD_BYTES &&
        bundle->static_file_bytes > bundle->static_payload_bytes &&
        is_sha256(static_sha) &&
        path_join(bundle->static_path, root, static_path) &&
        regular_file_size(bundle->static_path, bundle->static_file_bytes);
    const int32_t routed_store = k3_json_object_get(
        &document, document.root, "routed_store");
    uint32_t layers = 0u, experts_per_layer = 0u, experts = 0u, tile_bytes = 0u;
    ok = ok && routed_store >= 0 &&
        document.tokens[routed_store].type == K3_JSON_OBJECT &&
        required_string(&document, routed_store, "format", &routed_format, error, error_size) &&
        required_string(&document, routed_store, "path", &routed_path, error, error_size) &&
        required_u32(&document, routed_store, "layers", &layers) &&
        required_u32(&document, routed_store, "experts_per_layer", &experts_per_layer) &&
        required_u32(&document, routed_store, "experts", &experts) &&
        required_u32(&document, routed_store, "tile_bytes", &tile_bytes) &&
        required_string(&document, routed_store, "manifest_sha256", &routed_sha, error, error_size) &&
        strcmp(routed_format, "mzg2") == 0 && layers == K3_BUNDLE_LAYERS &&
        experts_per_layer == K3_BUNDLE_EXPERTS_PER_LAYER &&
        experts == K3_BUNDLE_EXPERTS && tile_bytes == K3_BUNDLE_TILE_BYTES &&
        is_sha256(routed_sha) &&
        path_join(bundle->mzg2_path, root, routed_path);
    struct stat routed_status;
    ok = ok && lstat(bundle->mzg2_path, &routed_status) == 0 &&
        S_ISDIR(routed_status.st_mode);
    const int32_t expert_layout = k3_json_object_get(
        &document, document.root, "expert_layout");
    uint64_t expert_bytes = 0u;
    const int32_t tensors = expert_layout >= 0 ?
        k3_json_object_get(&document, expert_layout, "tensors") : -1;
    ok = ok && expert_layout >= 0 &&
        document.tokens[expert_layout].type == K3_JSON_OBJECT &&
        required_u64(&document, expert_layout, "bytes", &expert_bytes) &&
        expert_bytes == K3_BUNDLE_EXPERT_BYTES && tensors >= 0 &&
        document.tokens[tensors].type == K3_JSON_ARRAY &&
        document.tokens[tensors].size == K3_BUNDLE_EXPERT_TENSORS;
    for (uint32_t index = 0u; ok && index < K3_BUNDLE_EXPERT_TENSORS; index++) {
        const int32_t item = k3_json_array_get(&document, tensors, index);
        char *name = NULL;
        uint64_t offset = 0u, bytes = 0u;
        ok = item >= 0 && document.tokens[item].type == K3_JSON_OBJECT &&
            required_string(&document, item, "name", &name, error, error_size) &&
            required_u64(&document, item, "offset", &offset) &&
            required_u64(&document, item, "bytes", &bytes) &&
            strcmp(name, K3_BUNDLE_EXPERT_NAMES[index]) == 0 &&
            offset == K3_BUNDLE_EXPERT_OFFSETS[index] &&
            bytes == K3_BUNDLE_EXPERT_SIZES[index];
        free(name);
    }
    if (!ok) bundle_error(error, error_size, "invalid standalone bundle manifest");

cleanup:
    free(schema); free(base_model); free(base_revision); free(source_manifest);
    free(layout_crc); free(static_path); free(static_sha); free(routed_format);
    free(routed_path); free(routed_sha);
    k3_json_document_free(&document);
    free(text);
    return ok;
}
