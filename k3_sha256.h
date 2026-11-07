#ifndef K3_SHA256_H
#define K3_SHA256_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool k3_sha256_fd_matches(int fd,
                          const char *label,
                          const char *expected_hex,
                          char *error,
                          size_t error_size);

bool k3_sha256_file_matches(const char *path,
                            const char *expected_hex,
                            char *error,
                            size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
