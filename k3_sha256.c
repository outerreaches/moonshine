#include "k3_sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/if_alg.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

static void sha_error(char *error, size_t error_size,
                      const char *format, ...) {
    if (!error || error_size == 0u) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static bool send_all_more(int fd, const uint8_t *data, size_t bytes) {
    size_t sent = 0u;
    while (sent < bytes) {
        const ssize_t result = send(fd, data + sent, bytes - sent, MSG_MORE);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return false;
        sent += (size_t)result;
    }
    return true;
}

static bool receive_full(int fd, uint8_t *data, size_t bytes) {
    size_t received = 0u;
    while (received < bytes) {
        const ssize_t result = recv(fd, data + received, bytes - received, 0);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return false;
        received += (size_t)result;
    }
    return true;
}

static bool same_file_version(const struct stat *before,
                              const struct stat *after) {
    return before->st_dev == after->st_dev &&
        before->st_ino == after->st_ino &&
        before->st_size == after->st_size &&
        before->st_mtim.tv_sec == after->st_mtim.tv_sec &&
        before->st_mtim.tv_nsec == after->st_mtim.tv_nsec &&
        before->st_ctim.tv_sec == after->st_ctim.tv_sec &&
        before->st_ctim.tv_nsec == after->st_ctim.tv_nsec;
}

static bool sha256_fd(int file, uint8_t digest[32], int *failure_errno) {
    struct stat before;
    if (fstat(file, &before) != 0) {
        *failure_errno = errno;
        return false;
    }
    if (!S_ISREG(before.st_mode) || before.st_size < 0) {
        *failure_errno = EINVAL;
        return false;
    }
    int algorithm = socket(AF_ALG, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    int operation = -1;
    uint8_t *buffer = NULL;
    bool ok = algorithm >= 0;
    struct sockaddr_alg address;
    memset(&address, 0, sizeof(address));
    address.salg_family = AF_ALG;
    memcpy(address.salg_type, "hash", 5u);
    memcpy(address.salg_name, "sha256", 7u);
    if (ok) ok = bind(algorithm, (struct sockaddr *)&address,
                      sizeof(address)) == 0;
    if (ok) {
        operation = accept4(algorithm, NULL, NULL, SOCK_CLOEXEC);
        ok = operation >= 0;
    }
    if (ok) {
        buffer = (uint8_t *)malloc(1024u * 1024u);
        if (!buffer) errno = ENOMEM;
        ok = buffer != NULL;
    }
    off_t offset = 0;
    while (ok && offset < before.st_size) {
        size_t request = 1024u * 1024u;
        if ((off_t)request > before.st_size - offset) {
            request = (size_t)(before.st_size - offset);
        }
        const ssize_t count = pread(file, buffer, request, offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            if (count == 0) errno = EIO;
            ok = false;
            break;
        }
        ok = send_all_more(operation, buffer, (size_t)count);
        offset += count;
    }
    if (ok) {
        ssize_t result;
        do {
            result = send(operation, NULL, 0u, 0);
        } while (result < 0 && errno == EINTR);
        ok = result == 0 && receive_full(operation, digest, 32u);
    }
    struct stat after;
    if (ok && (fstat(file, &after) != 0 ||
               !same_file_version(&before, &after))) {
        errno = ESTALE;
        ok = false;
    }
    if (!ok) *failure_errno = errno ? errno : EIO;
    free(buffer);
    if (operation >= 0) close(operation);
    if (algorithm >= 0) close(algorithm);
    return ok;
}

static int hex_value(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

static bool parse_digest(const char *expected_hex, uint8_t expected[32],
                         char *error, size_t error_size) {
    if (!expected_hex || strlen(expected_hex) != 64u) {
        sha_error(error, error_size, "invalid expected SHA-256");
        return false;
    }
    for (size_t index = 0u; index < 32u; ++index) {
        const int high = hex_value(expected_hex[index * 2u]);
        const int low = hex_value(expected_hex[index * 2u + 1u]);
        if (high < 0 || low < 0) {
            sha_error(error, error_size, "invalid expected SHA-256");
            return false;
        }
        expected[index] = (uint8_t)((high << 4) | low);
    }
    return true;
}

bool k3_sha256_fd_matches(int fd,
                          const char *label,
                          const char *expected_hex,
                          char *error,
                          size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (fd < 0) {
        sha_error(error, error_size, "invalid SHA-256 file descriptor");
        return false;
    }
    uint8_t expected[32];
    if (!parse_digest(expected_hex, expected, error, error_size)) return false;
    uint8_t actual[32];
    int failure_errno = 0;
    if (!sha256_fd(fd, actual, &failure_errno)) {
        sha_error(error, error_size, "SHA-256 failed for %s: %s",
                  label ? label : "file descriptor", strerror(failure_errno));
        return false;
    }
    if (memcmp(actual, expected, sizeof(actual)) != 0) {
        sha_error(error, error_size, "SHA-256 mismatch: %s",
                  label ? label : "file descriptor");
        return false;
    }
    return true;
}

bool k3_sha256_file_matches(const char *path,
                            const char *expected_hex,
                            char *error,
                            size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) {
        sha_error(error, error_size, "invalid SHA-256 path");
        return false;
    }
    const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) {
        sha_error(error, error_size, "open SHA-256 input %s: %s",
                  path, strerror(errno));
        return false;
    }
    const bool ok = k3_sha256_fd_matches(
        fd, path, expected_hex, error, error_size);
    close(fd);
    return ok;
}
