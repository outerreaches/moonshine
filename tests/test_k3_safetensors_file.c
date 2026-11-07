#include "k3_safetensors.h"

#include <fcntl.h>
#include <stdint.h>
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

static bool write_full(int fd, const void *data, size_t bytes) {
    const uint8_t *cursor = (const uint8_t *)data;
    while (bytes != 0u) {
        const ssize_t written = write(fd, cursor, bytes);
        if (written <= 0) return false;
        cursor += (size_t)written;
        bytes -= (size_t)written;
    }
    return true;
}

int main(void) {
    int result = 0;
    char path[] = "/tmp/k3-static-safetensors-XXXXXX";
    char link_path[128] = {0};
    char fifo_path[128] = {0};
    int fd = mkstemp(path);
    k3_st_model model;
    memset(&model, 0, sizeof(model));
    k3_st_read read = {0};
    char error[512] = {0};
    static const char header_json[] =
        "{\"language_model.model.norm.weight\":{"
        "\"data_offsets\":[0,4],\"dtype\":\"F32\",\"shape\":[1]}}";
    const uint64_t header_bytes = sizeof(header_json) - 1u;
    uint8_t prefix[8];
    for (uint32_t index = 0u; index < 8u; index++) {
        prefix[index] = (uint8_t)(header_bytes >> (index * 8u));
    }
    static const uint8_t payload[4] = {1u, 2u, 3u, 4u};
    CHECK(fd >= 0, "create synthetic SafeTensors file");
    CHECK(write_full(fd, prefix, sizeof(prefix)) &&
              write_full(fd, header_json, header_bytes) &&
              write_full(fd, payload, sizeof(payload)) &&
              fsync(fd) == 0,
          "write synthetic SafeTensors file");
    close(fd);
    fd = -1;

    CHECK(k3_st_model_open_file(&model, path, error, sizeof(error)), error);
    CHECK(model.shard_count == 1u && model.tensor_count == 1u,
          "single-file directory dimensions");
    const k3_st_tensor *tensor = k3_st_find(
        &model, "language_model.model.norm.weight");
    CHECK(tensor != NULL && tensor->dtype == K3_ST_DTYPE_F32 &&
              tensor->byte_length == sizeof(payload) &&
              tensor->physical_offset == 8u + header_bytes,
          "single-file tensor metadata");
    CHECK(k3_st_read_span(
              &model, tensor->shard, tensor->physical_offset,
              tensor->byte_length, 8u, &read,
              error, sizeof(error)),
          error);
    CHECK(read.data_bytes == sizeof(payload) &&
              memcmp(read.data, payload, sizeof(payload)) == 0,
          "single-file tensor payload");
    k3_st_read_release(&read);
    k3_st_model_close(&model);

    snprintf(link_path, sizeof(link_path), "%s.link", path);
    CHECK(symlink(path, link_path) == 0,
          "create single-file symlink fixture");
    CHECK(!k3_st_model_open_file(&model, link_path, error, sizeof(error)),
          "single-file symlink accepted");
    unlink(link_path);
    snprintf(fifo_path, sizeof(fifo_path), "%s.fifo", path);
    CHECK(mkfifo(fifo_path, 0600) == 0,
          "create single-file FIFO fixture");
    CHECK(!k3_st_model_open_file(&model, fifo_path, error, sizeof(error)),
          "single-file FIFO accepted");
    unlink(fifo_path);
    CHECK(!k3_st_model_open_file(&model, "/no/such/static.safetensors",
                                 error, sizeof(error)),
          "missing single-file model accepted");
    printf("K3 single-file SafeTensors: PASS\n");

cleanup:
    if (fd >= 0) close(fd);
    k3_st_read_release(&read);
    k3_st_model_close(&model);
    if (link_path[0]) unlink(link_path);
    if (fifo_path[0]) unlink(fifo_path);
    unlink(path);
    return result;
}
