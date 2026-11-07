#include <hip/hip_runtime.h>
#include <cassert>
#include <cstring>
extern "C" void mimo26_test_allocation_guards();
extern "C" void mimo26_test_allocations_empty() __attribute__((weak));
int main(int argc, char **argv) {
    assert(argc == 2);
    unsigned char host[8192]{}; void *device = nullptr;
    assert(hipMalloc(&device, sizeof host) == hipSuccess);
    assert(hipMemcpy(device, host, sizeof host, hipMemcpyHostToDevice) == hipSuccess);
    if (!strcmp(argv[1], "copy-overflow")) {
        // Same shape as the former second embedding write into one-token hidden.
        hipMemcpy((unsigned char *)device + sizeof host, host, sizeof host, hipMemcpyHostToDevice);
    } else if (!strcmp(argv[1], "red-zone")) {
        // Bypasses the explicit-copy wrapper; the end sentinel must catch it.
        assert(hipMemset((unsigned char *)device + sizeof host, 0, 1) == hipSuccess);
    } else if (!strcmp(argv[1], "leak")) {
        assert(mimo26_test_allocations_empty);
        mimo26_test_allocations_empty();  // Must refuse a live allocation.
    } else assert(!strcmp(argv[1], "healthy"));
    mimo26_test_allocation_guards();
    assert(hipFree(device) == hipSuccess);
    if (mimo26_test_allocations_empty) mimo26_test_allocations_empty();
}
