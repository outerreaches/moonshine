#define _XOPEN_SOURCE 700

#include "k3_mzg2.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", message); \
            result = 1; \
            goto cleanup; \
        } \
    } while (0)

static const char expected[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
static const char stale[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
static const char k3[] =
    "476fa0ba64e3233cbb9ca0642327361a73f6807e751edb071c92fa2216b202a4";

int main(void) {
    int result = 0;
    int fd = -1;
    k3_mzg2_store *store = NULL;
    char path[128];
    char error[512] = {0};
    snprintf(path, sizeof(path), "/tmp/k31-mzg2-identity-%ld",
             (long)getpid());
    fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
    CHECK(fd >= 0, "create MZG2 identity fixture");
    k3_mzg2_file_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "K3MZG2", 7);
    header.version = K3_MZG2_VERSION;
    header.header_bytes = K3_MZG2_FILE_HEADER_BYTES;
    header.alignment = K3_MZG2_ALIGNMENT;
    header.layer = 1;
    header.expert_count = 1;
    header.index_entry_bytes = sizeof(k3_mzg2_index_entry);
    header.tile_bytes = 16384;
    header.index_offset = K3_MZG2_FILE_HEADER_BYTES;
    header.index_bytes = sizeof(k3_mzg2_index_entry);
    header.data_offset = 8192;
    header.max_block_bytes = 4096;
    memset(header.source_manifest_sha256, 0x11, 32);
    k3_mzg2_index_entry entry;
    memset(&entry, 0, sizeof(entry));
    entry.layer = 1;
    entry.expert = 0;
    entry.block_bytes = 4096;
    entry.block_offset = 8192;
    CHECK(ftruncate(fd, 12288) == 0 &&
          pwrite(fd, &header, sizeof(header), 0) == sizeof(header) &&
          pwrite(fd, &entry, sizeof(entry), 4096) == sizeof(entry),
          "write MZG2 identity fixture");

    CHECK(!k3_mzg2_store_open_path(&store, path, NULL, error, sizeof(error)) &&
          !store,
          "missing source manifest accepted");
    CHECK(!k3_mzg2_store_open_path(&store, path, "XYZ", error, sizeof(error)) &&
          !store,
          "malformed source manifest accepted");
    CHECK(!k3_mzg2_store_open_path(&store, path, stale, error, sizeof(error)) &&
          !store,
          "stale source manifest accepted");
    CHECK(!k3_mzg2_store_open_path(&store, path, k3, error, sizeof(error)) &&
          !store,
          "K3 source manifest accepted");
    CHECK(k3_mzg2_store_open_path(&store, path, expected, error, sizeof(error)),
          error);
    k3_mzg2_span span;
    CHECK(k3_mzg2_store_span(store, 1, 0, &span) &&
          span.offset == 8192 && span.bytes == 4096,
          "validated MZG2 span mismatch");
    k3_mzg2_store_destroy(store);
    store = NULL;
    puts("PASS: K3.1 native MZG2 source identity");

cleanup:
    if (store) k3_mzg2_store_destroy(store);
    if (fd >= 0) close(fd);
    unlink(path);
    return result;
}
