#define _XOPEN_SOURCE 700

#include "k3_engine.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(void) {
    char root[128];
    snprintf(root, sizeof(root), "/tmp/k31-engine-profile-%ld",
             (long)getpid());
    if (mkdir(root, 0700) != 0) {
        perror("mkdtemp");
        return 1;
    }
    char path[256];
    snprintf(path, sizeof(path), "%s/config.json", root);
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        perror("open config");
        unlink(path);
        rmdir(root);
        return 1;
    }
    const char config[] = "{}";
    const bool written = write(fd, config, 2) == 2;
    close(fd);
    if (!written) {
        unlink(path);
        rmdir(root);
        return 1;
    }
    k3_engine *engine = NULL;
    k3_engine_stats stats;
    char error[512] = {0};
    const bool created = k3_engine_create(
        &engine, root, 64, 16, 16, true, &stats, error, sizeof(error));
    int result = 0;
    if (created || engine || strstr(error, "architectures") == NULL ||
        strstr(error, "invalid K3.1 engine configuration or model profile")) {
        fprintf(stderr, "FAIL: profile diagnostic was clobbered: %s\n", error);
        result = 1;
    }
    k3_engine_destroy(engine);
    unlink(path);
    rmdir(root);
    if (!result) puts("PASS: K3.1 engine profile diagnostic preservation");
    return result;
}
