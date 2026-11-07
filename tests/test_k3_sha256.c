#include "k3_sha256.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", (message)); \
            return 1; \
        } \
    } while (0)

int main(void) {
    char path[] = "/tmp/k3-sha256-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "temporary SHA-256 fixture");
    CHECK(close(fd) == 0, "close empty SHA-256 fixture");
    char error[512];
    static const char empty_expected[] =
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855";
    CHECK(k3_sha256_file_matches(
              path, empty_expected, error, sizeof(error)), error);
    fd = open(path, O_WRONLY);
    CHECK(fd >= 0 && write(fd, "abc", 3u) == 3 && close(fd) == 0,
          "write SHA-256 fixture");
    static const char expected[] =
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad";
    CHECK(k3_sha256_file_matches(path, expected, error, sizeof(error)), error);
    fd = open(path, O_WRONLY | O_TRUNC);
    CHECK(fd >= 0, "open multi-chunk SHA-256 fixture");
    char block[65536];
    memset(block, 'a', sizeof(block));
    size_t remaining = 1024u * 1024u + 3u;
    while (remaining > 0u) {
        const size_t chunk = remaining < sizeof(block) ? remaining : sizeof(block);
        CHECK(write(fd, block, chunk) == (ssize_t)chunk,
              "write multi-chunk SHA-256 fixture");
        remaining -= chunk;
    }
    CHECK(close(fd) == 0, "close multi-chunk SHA-256 fixture");
    static const char multi_expected[] =
        "f5e25b6b994188fdb721357d459785dc"
        "479934edfd40f95658a45ab2ce2d8027";
    CHECK(k3_sha256_file_matches(
              path, multi_expected, error, sizeof(error)), error);
    fd = open(path, O_WRONLY);
    CHECK(fd >= 0 && write(fd, "b", 1u) == 1 && close(fd) == 0,
          "mutate SHA-256 fixture");
    CHECK(!k3_sha256_file_matches(
              path, multi_expected, error, sizeof(error)) &&
              strstr(error, "mismatch") != NULL,
          "same-size payload corruption accepted");
    CHECK(!k3_sha256_file_matches(
              path,
              "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
              error, sizeof(error)),
          "invalid SHA-256 hex accepted");

    fd = open(path, O_WRONLY | O_TRUNC);
    CHECK(fd >= 0 && write(fd, "abc", 3u) == 3 && close(fd) == 0,
          "restore SHA-256 fixture");
    const int verified_fd = open(path, O_RDONLY);
    CHECK(verified_fd >= 0, "open retained SHA-256 descriptor");
    char swap_path[128];
    snprintf(swap_path, sizeof(swap_path), "%s.swap", path);
    CHECK(rename(path, swap_path) == 0, "rename SHA-256 fixture");
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    CHECK(fd >= 0 && write(fd, "abd", 3u) == 3 && close(fd) == 0,
          "write replacement SHA-256 fixture");
    CHECK(k3_sha256_fd_matches(
              verified_fd, swap_path, expected, error, sizeof(error)),
          "retained SHA-256 descriptor followed replacement path");
    CHECK(close(verified_fd) == 0, "close retained SHA-256 descriptor");
    unlink(path);
    CHECK(rename(swap_path, path) == 0, "restore SHA-256 fixture path");

    char link_path[128];
    snprintf(link_path, sizeof(link_path), "%s.link", path);
    CHECK(symlink(path, link_path) == 0, "create SHA-256 symlink fixture");
    CHECK(!k3_sha256_file_matches(link_path, expected, error, sizeof(error)),
          "SHA-256 symlink input accepted");
    unlink(link_path);
    char fifo_path[128];
    snprintf(fifo_path, sizeof(fifo_path), "%s.fifo", path);
    CHECK(mkfifo(fifo_path, 0600) == 0, "create SHA-256 FIFO fixture");
    CHECK(!k3_sha256_file_matches(fifo_path, expected, error, sizeof(error)),
          "SHA-256 FIFO input accepted");
    unlink(fifo_path);
    unlink(path);
    printf("K3 SHA-256 file gate: PASS\n");
    return 0;
}
