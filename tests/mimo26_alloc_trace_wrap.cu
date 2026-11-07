// Link-only hipMalloc instrumentation. Stop loading at first sampled swap.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>
extern "C" hipError_t __real_hipMalloc(void **, size_t);
static unsigned long long read_number(const char *path) {
    FILE *f = fopen(path, "r"); unsigned long long value = 0;
    if (f) { if (fscanf(f, "%llu", &value) != 1) value = 0; fclose(f); }
    return value;
}
static unsigned long long status_number(const char *key) {
    FILE *f = fopen("/proc/self/status", "r"); char line[256];
    unsigned long long value = 0;
    if (!f) return 0;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, key, strlen(key))) {
        if (sscanf(line + strlen(key), "%llu", &value) != 1) value = 0;
        break;
    }
    fclose(f); return value;
}
extern "C" hipError_t __wrap_hipMalloc(void **out, size_t bytes) {
    static unsigned long long requested = 0, calls = 0, experts = 0;
    timespec before{}, after{}; clock_gettime(CLOCK_MONOTONIC, &before);
    auto result = __real_hipMalloc(out, bytes);
    clock_gettime(CLOCK_MONOTONIC, &after);
    if (result == hipSuccess) requested += bytes;
    ++calls;
    if (bytes == 13369344) ++experts;
    auto swap = status_number("VmSwap:");
    if (swap || calls == 1 || experts % 64 == 0) {
        fprintf(stderr, "ALLOC {\"calls\":%llu,\"experts\":%llu,\"requested_bytes\":%llu,"
                "\"last_bytes\":%zu,\"seconds\":%.9f,\"last_seconds\":%.9f,\"swap_KiB\":%llu,"
                "\"gtt_used\":%llu,\"vram_used\":%llu,\"result\":%d}\n",
                calls, experts, requested, bytes, double(after.tv_sec) + after.tv_nsec / 1e9,
                double(after.tv_sec - before.tv_sec) + (after.tv_nsec - before.tv_nsec) / 1e9,
                swap, read_number("/sys/class/drm/card0/device/mem_info_gtt_used"),
                read_number("/sys/class/drm/card0/device/mem_info_vram_used"), int(result));
    }
    if (swap) {
        fprintf(stderr, "SWAP_GUARD stopping allocations; caller will destroy worker\n");
        return hipErrorOutOfMemory; // out remains owned by worker cleanup.
    }
    // Diagnostic-only pacing; never enabled in the server or baseline worker.
    if (bytes == 13369344 && getenv("MIMO26_PROBE_PACE_1MS")) {
        timespec pause{0, 1000000}; nanosleep(&pause, nullptr);
    }
    return result;
}
