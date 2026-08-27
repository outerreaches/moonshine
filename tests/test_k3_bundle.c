#include "k3_bundle.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        result = 1; \
        goto cleanup; \
    } \
} while (0)

static bool write_text(const char *path, const char *text) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    const size_t size = strlen(text);
    const bool ok = write(fd, text, size) == (ssize_t)size && fsync(fd) == 0;
    close(fd);
    return ok;
}

int main(void) {
    int result = 0;
    char root[] = "/tmp/k3-bundle-XXXXXX";
    char manifest[4096];
    char static_path[4096];
    char mzg2_path[4096];
    char error[512] = {0};
    k3_bundle bundle;
    bool present = false;
    CHECK(mkdtemp(root) != NULL, "create bundle root");
    snprintf(manifest, sizeof(manifest), "%s/moonshine-bundle.json", root);
    snprintf(static_path, sizeof(static_path), "%s/model-static.safetensors", root);
    snprintf(mzg2_path, sizeof(mzg2_path), "%s/expert-store-mzg2", root);
    CHECK(mkdir(mzg2_path, 0700) == 0, "create MZG2 root");
    const int static_fd = open(static_path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    CHECK(static_fd >= 0, "create sparse static store");
    CHECK(ftruncate(static_fd, (off_t)(K3_BUNDLE_STATIC_PAYLOAD_BYTES + 4096u)) == 0,
          "size sparse static store");
    close(static_fd);

    static const char *valid =
        "{"
        "\"schema\":\"moonshine-k3-mzg2-bundle-v1\","
        "\"version\":1,"
        "\"base_model\":\"moonshotai/Kimi-K3\","
        "\"base_revision\":\"9f62e4e9fffbd0a83ddd60e1c209d828994b3569\","
        "\"source_manifest_sha256\":\"476fa0ba64e3233cbb9ca0642327361a73f6807e751edb071c92fa2216b202a4\","
        "\"source_model_layout_crc64\":\"d17f7f2aad23c9c9\","
        "\"static_store\":{"
          "\"path\":\"model-static.safetensors\","
          "\"tensor_count\":2460,"
          "\"payload_bytes\":113509540864,"
          "\"file_bytes\":113509544960,"
          "\"sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"},"
        "\"routed_store\":{"
          "\"format\":\"mzg2\",\"path\":\"expert-store-mzg2\","
          "\"layers\":92,\"experts_per_layer\":896,\"experts\":82432,"
          "\"tile_bytes\":16384,"
          "\"manifest_sha256\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"},"
        "\"expert_layout\":{\"bytes\":17547264,\"tensors\":["
          "{\"name\":\"w1.weight_packed\",\"offset\":0,\"bytes\":5505024},"
          "{\"name\":\"w1.weight_scale\",\"offset\":5505024,\"bytes\":344064},"
          "{\"name\":\"w2.weight_packed\",\"offset\":5849088,\"bytes\":5505024},"
          "{\"name\":\"w2.weight_scale\",\"offset\":11354112,\"bytes\":344064},"
          "{\"name\":\"w3.weight_packed\",\"offset\":11698176,\"bytes\":5505024},"
          "{\"name\":\"w3.weight_scale\",\"offset\":17203200,\"bytes\":344064}"
        "]}}";
    CHECK(k3_bundle_detect(root, &present, error, sizeof(error)) && !present,
          "absent bundle detection");
    CHECK(write_text(manifest, valid), "write valid bundle manifest");
    CHECK(k3_bundle_detect(root, &present, error, sizeof(error)) && present,
          "present bundle detection");
    CHECK(k3_bundle_load(&bundle, root, error, sizeof(error)), error);
    CHECK(bundle.source_model_layout_crc64 == UINT64_C(0xd17f7f2aad23c9c9) &&
              bundle.static_tensor_count == K3_BUNDLE_STATIC_TENSORS &&
              bundle.static_payload_bytes == K3_BUNDLE_STATIC_PAYLOAD_BYTES &&
              strcmp(bundle.static_path, static_path) == 0 &&
              strcmp(bundle.mzg2_path, mzg2_path) == 0,
          "parsed bundle contract");

    char *invalid = strdup(valid);
    CHECK(invalid != NULL, "allocate invalid identity manifest");
    char *needle = strstr(invalid, "d17f7f2aad23c9c9");
    CHECK(needle != NULL, "locate model identity");
    memset(needle, '0', 16u);
    CHECK(write_text(manifest, invalid), "write wrong-identity manifest");
    CHECK(!k3_bundle_load(&bundle, root, error, sizeof(error)),
          "wrong model identity accepted");
    free(invalid);

    invalid = strdup(valid);
    CHECK(invalid != NULL, "allocate invalid manifest");
    needle = strstr(invalid, "model-static.safetensors");
    CHECK(needle != NULL, "locate static path");
    memcpy(needle, "../bad-static.safetensors", 24u);
    CHECK(write_text(manifest, invalid), "write traversal manifest");
    CHECK(!k3_bundle_load(&bundle, root, error, sizeof(error)),
          "bundle path traversal accepted");
    free(invalid);

    CHECK(write_text(manifest, "{\"schema\":\"wrong\"}"),
          "write invalid bundle manifest");
    CHECK(!k3_bundle_load(&bundle, root, error, sizeof(error)),
          "invalid bundle schema accepted");
    printf("K3 standalone bundle manifest: PASS\n");

cleanup:
    unlink(manifest);
    unlink(static_path);
    rmdir(mzg2_path);
    rmdir(root);
    return result;
}
