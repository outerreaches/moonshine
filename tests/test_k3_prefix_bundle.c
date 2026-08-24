#include "k3_prefix_bundle.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                     \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static bool fake_state(const char *path, uint64_t bytes) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return false;
    const bool ok = fchmod(fd, 0600) == 0 &&
        ftruncate(fd, (off_t)bytes) == 0 && close(fd) == 0;
    return ok;
}

static bool publish_entry(k3_prefix_bundle *bundle, uint32_t base,
                          char id[33], char state_path[4096],
                          char *error, size_t error_size) {
    uint32_t tokens[] = {base, base + 1u, base + 2u, base + 3u};
    const k3_tool_choice_marker choices[] = {{1u, K3_TOOL_CHOICE_REQUIRED}};
    const k3_single_tool_call_marker singles[] = {{2u}};
    const k3_response_format_marker formats[] = {
        {3u, K3_RESPONSE_FORMAT_JSON_SCHEMA, "{\"type\":\"object\"}"},
    };
    const k3_prefix_bundle_snapshot snapshot = {
        .tokens = tokens,
        .token_count = 4u,
        .tool_choices = choices,
        .tool_choice_count = 1u,
        .single_tool_calls = singles,
        .single_tool_call_count = 1u,
        .response_formats = formats,
        .response_format_count = 1u,
    };
    const k3_engine_state_file_info info = {
        .format_version = 1u,
        .context = 8192u,
        .token_position = 4u,
        .model_layout_crc64 = UINT64_C(0x12345678),
        .payload_bytes = 2048u,
        .file_bytes = 4096u,
        .payload_crc64 = base + UINT64_C(0xabcdef00),
        .q8_projections = true,
    };
    return k3_prefix_bundle_allocate_state_path(
               bundle, id, 33u, state_path, 4096u,
               error, error_size) &&
        fake_state(state_path, info.file_bytes) &&
        k3_prefix_bundle_publish(
            bundle, id, state_path, &snapshot, &info,
            error, error_size);
}

int main(void) {
    char root[] = "/tmp/k3-prefix-bundle-XXXXXX";
    CHECK(mkdtemp(root) != NULL, "temporary root");
    const k3_prefix_bundle_identity identity = {
        .format_version = 1u,
        .context = 8192u,
        .model_layout_crc64 = UINT64_C(0x12345678),
        .q8_projections = true,
    };
    char error[512];
    k3_prefix_bundle *bundle = NULL;
    CHECK(k3_prefix_bundle_open(
              &bundle, root, &identity, 2u, 20000u,
              error, sizeof(error)),
          error);
    char id[3][33];
    char state[3][4096];
    CHECK(publish_entry(bundle, 10u, id[0], state[0], error, sizeof(error)),
          error);
    CHECK(publish_entry(bundle, 20u, id[1], state[1], error, sizeof(error)),
          error);
    CHECK(publish_entry(bundle, 30u, id[2], state[2], error, sizeof(error)),
          error);
    CHECK(k3_prefix_bundle_count(bundle) == 2u,
          "entry limit did not evict oldest");
    const uint32_t evicted[] = {10u, 11u, 12u, 13u};
    const uint32_t retained[] = {20u, 21u, 22u, 23u};
    size_t index = SIZE_MAX;
    CHECK(!k3_prefix_bundle_find_exact(bundle, evicted, 4u, &index),
          "oldest entry was not evicted");
    CHECK(k3_prefix_bundle_find_exact(bundle, retained, 4u, &index),
          "retained entry lookup");
    CHECK(access(state[0], F_OK) != 0,
          "evicted state file remains");
    k3_prefix_bundle_destroy(bundle);
    bundle = NULL;

    CHECK(k3_prefix_bundle_open(
              &bundle, root, &identity, 2u, 20000u,
              error, sizeof(error)),
          error);
    CHECK(k3_prefix_bundle_count(bundle) == 2u,
          "manifest reload count");
    k3_prefix_bundle_entry entry;
    CHECK(k3_prefix_bundle_entry_at(bundle, 0u, &entry),
          "manifest entry view");
    CHECK(entry.token_count == 4u &&
              entry.tool_choice_count == 1u &&
              entry.single_tool_call_count == 1u &&
              entry.response_format_count == 1u &&
              strcmp(entry.response_formats[0].response_schema_json,
                     "{\"type\":\"object\"}") == 0,
          "metadata round trip");

    k3_prefix_bundle_identity wrong = identity;
    wrong.context = 32768u;
    k3_prefix_bundle *rejected = NULL;
    CHECK(!k3_prefix_bundle_open(
              &rejected, root, &wrong, 2u, 20000u,
              error, sizeof(error)) && rejected == NULL,
          "wrong manifest identity was accepted");

    CHECK(k3_prefix_bundle_remove(bundle, 0u, error, sizeof(error)), error);
    CHECK(k3_prefix_bundle_count(bundle) == 1u, "entry removal");
    k3_prefix_bundle_entry remaining;
    CHECK(k3_prefix_bundle_entry_at(bundle, 0u, &remaining),
          "remaining entry");
    char remaining_state[4096];
    snprintf(remaining_state, sizeof(remaining_state), "%s", remaining.state_path);
    char remaining_meta[4096];
    snprintf(remaining_meta, sizeof(remaining_meta), "%s", remaining.state_path);
    char *suffix = strrchr(remaining_meta, '.');
    CHECK(suffix != NULL, "state suffix");
    strcpy(suffix, ".meta");
    k3_prefix_bundle_destroy(bundle);

    char manifest[4096];
    snprintf(manifest, sizeof(manifest), "%s/manifest.bin", root);
    int fd = open(manifest, O_RDWR);
    CHECK(fd >= 0, "manifest corruption open");
    CHECK(lseek(fd, -1, SEEK_END) >= 0, "manifest corruption seek");
    unsigned char byte = 0xffu;
    CHECK(write(fd, &byte, 1u) == 1 && close(fd) == 0,
          "manifest corruption write");
    CHECK(!k3_prefix_bundle_open(
              &rejected, root, &identity, 2u, 20000u,
              error, sizeof(error)) && rejected == NULL,
          "corrupt manifest was accepted");

    (void)unlink(remaining_state);
    (void)unlink(remaining_meta);
    (void)unlink(manifest);
    char entries[4096];
    snprintf(entries, sizeof(entries), "%s/entries", root);
    CHECK(rmdir(entries) == 0 && rmdir(root) == 0, "temporary cleanup");
    printf("K3 exact-prefix bundle: PASS\n");
    return 0;
}
