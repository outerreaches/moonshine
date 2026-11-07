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

#define FIXTURE_STATIC_TENSORS 2460u
#define FIXTURE_STATIC_PAYLOAD_BYTES UINT64_C(4096)

static bool write_text(const char *path, const char *text) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    const size_t size = strlen(text);
    const bool ok = write(fd, text, size) == (ssize_t)size && fsync(fd) == 0;
    close(fd);
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 2) {
        k3_bundle candidate;
        char error[512] = {0};
        if (!k3_bundle_load(&candidate, argv[1], error, sizeof(error))) {
            fprintf(stderr, "FAIL: %s\n", error);
            return 1;
        }
        int static_fd = -1;
        if (!k3_bundle_open_verified_static(
                &candidate, &static_fd, error, sizeof(error))) {
            fprintf(stderr, "FAIL: %s\n", error);
            return 1;
        }
        close(static_fd);
        puts("K3.1 generated bundle native load: PASS");
        return 0;
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [BUNDLE_ROOT]\n", argv[0]);
        return 2;
    }
    int result = 0;
    char root[] = "/tmp/k3-bundle-XXXXXX";
    char manifest[4096];
    char static_path[4096];
    char mzg2_path[4096];
    char mzg2_manifest_path[4096];
    char gate_static_path[4096];
    char gate_manifest_path[4096];
    int gate_fd = -1;
    char error[512] = {0};
    k3_bundle bundle;
    bool present = false;
    CHECK(mkdtemp(root) != NULL, "create bundle root");
    snprintf(manifest, sizeof(manifest), "%s/moonshine-bundle.json", root);
    snprintf(static_path, sizeof(static_path), "%s/model-static.safetensors", root);
    snprintf(mzg2_path, sizeof(mzg2_path), "%s/expert-store-mzg2", root);
    snprintf(mzg2_manifest_path, sizeof(mzg2_manifest_path),
             "%s/expert-store-mzg2/expert-store.json", root);
    snprintf(gate_static_path, sizeof(gate_static_path),
             "%s/gate-static", root);
    snprintf(gate_manifest_path, sizeof(gate_manifest_path),
             "%s/gate-manifest", root);
    CHECK(mkdir(mzg2_path, 0700) == 0, "create MZG2 root");
    const int static_fd = open(static_path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    CHECK(static_fd >= 0, "create sparse static store");
    CHECK(ftruncate(static_fd, (off_t)(FIXTURE_STATIC_PAYLOAD_BYTES + 4096u)) == 0,
          "size sparse static store");
    close(static_fd);

    static const char *valid =
        "{"
        "\"schema\":\"moonshine-k31-mzg2-bundle-v1\","
        "\"version\":1,"
        "\"base_model\":\"moonshotai/Kimi-K3.1-Pro\","
        "\"base_revision\":\"6d0a0202956796543364d666ab19c6ad1a494ee1\","
        "\"source_manifest_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
        "\"source_model_layout_crc64\":\"d17f7f2aad23c9c9\","
        "\"static_store\":{"
          "\"path\":\"model-static.safetensors\","
          "\"tensor_count\":2460,"
          "\"payload_bytes\":4096,"
          "\"file_bytes\":8192,"
          "\"sha256\":\"f5fd96195f68ce60b2bf87ebf0ef4ae2637cc5320026907fda36d7dbb725249f\"},"
        "\"routed_store\":{"
          "\"format\":\"mzg2\",\"path\":\"expert-store-mzg2\","
          "\"layers\":92,\"experts_per_layer\":896,\"experts\":82432,"
          "\"tile_bytes\":16384,"
          "\"manifest_sha256\":\"274c006b34378256bb70e86a15c07f3c813cf0827feca62ba8d08f1d6979727b\"},"
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
              bundle.static_tensor_count == FIXTURE_STATIC_TENSORS &&
              bundle.static_payload_bytes == FIXTURE_STATIC_PAYLOAD_BYTES &&
              strcmp(bundle.static_path, static_path) == 0 &&
              strcmp(bundle.static_sha256,
                     "f5fd96195f68ce60b2bf87ebf0ef4ae2637cc5320026907fda36d7dbb725249f") == 0 &&
              strcmp(bundle.source_manifest_sha256,
                     "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") == 0 &&
              strcmp(bundle.mzg2_path, mzg2_path) == 0 &&
              strcmp(bundle.mzg2_manifest_path, mzg2_manifest_path) == 0 &&
              strcmp(bundle.mzg2_manifest_sha256,
                     "274c006b34378256bb70e86a15c07f3c813cf0827feca62ba8d08f1d6979727b") == 0,
          "parsed bundle contract");

    CHECK(write_text(gate_static_path, "abc") &&
              write_text(gate_manifest_path, "abc"),
          "write runtime artifact fixtures");
    k3_bundle gate;
    memset(&gate, 0, sizeof(gate));
    snprintf(gate.static_path, sizeof(gate.static_path), "%s", gate_static_path);
    snprintf(gate.mzg2_manifest_path, sizeof(gate.mzg2_manifest_path),
             "%s", gate_manifest_path);
    snprintf(gate.static_sha256, sizeof(gate.static_sha256),
             "%s", "ba7816bf8f01cfea414140de5dae2223"
                    "b00361a396177a9cb410ff61f20015ad");
    snprintf(gate.mzg2_manifest_sha256, sizeof(gate.mzg2_manifest_sha256),
             "%s", "ba7816bf8f01cfea414140de5dae2223"
                    "b00361a396177a9cb410ff61f20015ad");
    gate.static_file_bytes = 3u;
    CHECK(k3_bundle_open_verified_static(
              &gate, &gate_fd, error, sizeof(error)), error);
    close(gate_fd);
    gate_fd = -1;
    CHECK(write_text(gate_static_path, "abd"),
          "mutate runtime static fixture");
    CHECK(!k3_bundle_open_verified_static(
              &gate, &gate_fd, error, sizeof(error)) && gate_fd == -1,
          "same-size runtime static corruption accepted");
    CHECK(write_text(gate_static_path, "abc") &&
              write_text(gate_manifest_path, "abd"),
          "mutate runtime MZG2 manifest fixture");
    CHECK(!k3_bundle_open_verified_static(
              &gate, &gate_fd, error, sizeof(error)) && gate_fd == -1,
          "runtime MZG2 manifest corruption accepted");

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
    CHECK(invalid != NULL, "allocate wrong static digest manifest");
    needle = strstr(invalid,
                    "f5fd96195f68ce60b2bf87ebf0ef4ae2637cc5320026907fda36d7dbb725249f");
    CHECK(needle != NULL, "locate static digest");
    needle[0] = 'g';
    CHECK(write_text(manifest, invalid), "write malformed static digest manifest");
    CHECK(!k3_bundle_load(&bundle, root, error, sizeof(error)),
          "wrong static digest accepted");
    free(invalid);

    invalid = strdup(valid);
    CHECK(invalid != NULL, "allocate wrong MZG2 digest manifest");
    needle = strstr(invalid,
                    "274c006b34378256bb70e86a15c07f3c813cf0827feca62ba8d08f1d6979727b");
    CHECK(needle != NULL, "locate MZG2 digest");
    needle[0] = 'g';
    CHECK(write_text(manifest, invalid), "write malformed MZG2 digest manifest");
    CHECK(!k3_bundle_load(&bundle, root, error, sizeof(error)),
          "wrong MZG2 manifest digest accepted");
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
    printf("K3.1 standalone bundle manifest: PASS\n");

cleanup:
    if (gate_fd >= 0) close(gate_fd);
    unlink(gate_static_path);
    unlink(gate_manifest_path);
    unlink(manifest);
    unlink(static_path);
    rmdir(mzg2_path);
    rmdir(root);
    return result;
}
