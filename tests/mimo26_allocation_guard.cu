// Test-only red zones and explicit-copy bounds. Not a GPU memory sanitizer:
// in-bounds cross-buffer writes and transient write/restore errors can escape.
#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
struct Allocation { void *base; size_t bytes; };
static std::map<uintptr_t, Allocation> allocations;
static constexpr size_t guard_bytes = 256;
static size_t checked = 0;
extern "C" hipError_t __real_hipMalloc(void **, size_t);
extern "C" hipError_t __real_hipFree(void *);
extern "C" hipError_t __real_hipMemcpy(void *, const void *, size_t, hipMemcpyKind);
static void check_one(uintptr_t pointer, const Allocation &a) {
    unsigned char guard[guard_bytes];
    for (const void *edge : {a.base, (void *)(pointer + a.bytes)}) {
        assert(__real_hipMemcpy(guard, edge, sizeof guard, hipMemcpyDeviceToHost) == hipSuccess);
        for (unsigned char byte : guard) assert(byte == 0xa5 && "device allocation red zone corrupted");
    }
    ++checked;
}
extern "C" void mimo26_test_allocation_guards() {
    assert(hipDeviceSynchronize() == hipSuccess);
    for (const auto &entry : allocations) check_one(entry.first, entry.second);
    fprintf(stderr, "TEST_ALLOCATION_GUARDS live=%zu checked=%zu\n", allocations.size(), checked);
}
extern "C" hipError_t __wrap_hipMalloc(void **out, size_t bytes) {
    assert(bytes <= SIZE_MAX - 2 * guard_bytes);
    void *base = nullptr;
    hipError_t status = __real_hipMalloc(&base, bytes + 2 * guard_bytes);
    if (status != hipSuccess) return status;
    *out = (unsigned char *)base + guard_bytes;
    assert(hipMemset(base, 0xa5, guard_bytes) == hipSuccess);
    assert(hipMemset((unsigned char *)*out + bytes, 0xa5, guard_bytes) == hipSuccess);
    allocations.emplace((uintptr_t)*out, Allocation{base, bytes});
    return hipSuccess;
}
extern "C" hipError_t __wrap_hipFree(void *pointer) {
    auto it = allocations.find((uintptr_t)pointer);
    if (it == allocations.end()) return __real_hipFree(pointer);
    check_one(it->first, it->second);
    void *base = it->second.base;
    allocations.erase(it);
    return __real_hipFree(base);
}
static void copy_bounds(const void *pointer, size_t bytes) {
    auto it = allocations.upper_bound((uintptr_t)pointer);
    if (it == allocations.begin()) return;  // Not one of our allocations.
    --it;
    size_t offset = (uintptr_t)pointer - it->first;
    if (offset > it->second.bytes) return;
    assert(bytes <= it->second.bytes - offset && "explicit HIP copy exceeds allocation");
}
extern "C" hipError_t __wrap_hipMemcpy(void *dst, const void *src, size_t bytes, hipMemcpyKind kind) {
    if (kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToDevice) copy_bounds(dst, bytes);
    if (kind == hipMemcpyDeviceToHost || kind == hipMemcpyDeviceToDevice) copy_bounds(src, bytes);
    return __real_hipMemcpy(dst, src, bytes, kind);
}
