// Link-only diagnostic: compare every expert-major projection with the
// one-vector kernel on the SAME resident weights and gathered activations.
// Stops at the first mismatch; never link this into a performance binary.
#include "k3_rocm_ops.h"
#include "mimo26_rocm_layer.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cerrno>
#include <sys/stat.h>
#include <string>
#include <vector>

static unsigned current_layer;
static uint64_t current_position;
static unsigned projection;
static bool dump_device(const char *, const char *, const void *, size_t);

extern "C" mimo26_rocm_layer_status __real_mimo26_rocm_layer_prefill(
    const mimo26_rocm_layer *, mimo26_rocm_layer_scratch *, void *, void *,
    void *, const void *, const void *, uint64_t, uint64_t, uint64_t,
    uint32_t, uint32_t *, void *);

extern "C" mimo26_rocm_layer_status __wrap_mimo26_rocm_layer_prefill(
    const mimo26_rocm_layer *layer, mimo26_rocm_layer_scratch *scratch,
    void *hidden, void *keys, void *values, const void *cos_tables,
    const void *sin_tables, uint64_t history, uint64_t first_position,
    uint64_t position, uint32_t count, uint32_t *routes, void *stream)
{
    current_layer = layer->weights->layer;
    current_position = position;
    projection = 0;
    auto status = __real_mimo26_rocm_layer_prefill(
        layer, scratch, hidden, keys, values, cos_tables, sin_tables,
        history, first_position, position, count, routes, stream);
    const char *trace = std::getenv("MIMO26_LAYER_TRACE");
    if (trace && status == MIMO26_ROCM_LAYER_OK) {
        std::string prefix = std::string(trace) + "-p" + std::to_string(position) +
                             "-l" + std::to_string(current_layer);
        bool saved = dump_device(prefix.c_str(), "-hidden.bin", hidden, (size_t)count * 4096 * 2);
        saved = dump_device(prefix.c_str(), "-key.bin", scratch->key,
                            (size_t)count * layer->weights->kv_heads * 192 * 2) && saved;
        saved = dump_device(prefix.c_str(), "-value.bin", scratch->value,
                            (size_t)count * layer->weights->kv_heads * 128 * 2) && saved;
        if (!saved) status = MIMO26_ROCM_LAYER_SYNC_FAILED;
    }
    std::fprintf(stderr, "projection_guard layer=%u position=%llu count=%u "
                 "checked=%u status=%d\n", current_layer,
                 (unsigned long long)position, count, projection, (int)status);
    return status;
}

extern "C" bool __real_k3_rocm_mxfp4_gemm_bf16(
    void *, const void *, const void *, const void *, uint32_t, uint32_t,
    uint32_t, void *);
extern "C" bool __real_k3_rocm_mxfp4_gemv_rows_bf16(
    void *, const void *, const void *, const void *, uint32_t, uint32_t,
    uint32_t, void *);

static bool dump_device(const char *prefix, const char *suffix,
                        const void *device, size_t size)
{
    std::vector<unsigned char> data(size);
    if (hipMemcpy(data.data(), device, size, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    FILE *file = std::fopen((std::string(prefix) + suffix).c_str(), "wbx");
    if (!file) return false;
    bool ok = std::fwrite(data.data(), 1, size, file) == size;
    return std::fclose(file) == 0 && ok;
}

static unsigned env_unsigned(const char *name, unsigned fallback)
{
    const char *value = std::getenv(name);
    if (value == NULL || *value == '\0') return fallback;
    char *end = NULL;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return (end != NULL && *end == '\0' && parsed > 0ul) ? (unsigned)parsed : fallback;
}

/*
 * Capture real projections for the float64 oracle
 * (Scripts/mimo26-qualification/mxfp4_oracle.py).
 *
 * The mismatch dump above cannot do this job: it fires only where the kernels
 * already disagree and then aborts the prefill, so it yields exactly one
 * projection of one shape. The oracle needs the shapes where they AGREE too --
 * agreement on real data is evidence, and the gate/up shape (2048x4096) had
 * none at all. So this dumps inputs unconditionally on a stride, and capture
 * mode makes a mismatch non-fatal so the run walks the whole model.
 *
 * Only the inputs are written. tests/mimo26_mxfp4_oracle_sweep runs every
 * kernel over them offline and names each output after the kernel that
 * produced it, which is what keeps the labels trustworthy -- see
 * [[capture-filenames-do-not-name-implementations]].
 */
static unsigned capture_seen, capture_written;

static bool capturing(void) { return std::getenv("MIMO26_PROJECTION_CAPTURE") != NULL; }

static void maybe_capture(const void *packed, const void *scales, const void *input,
                          uint32_t vectors, uint32_t rows, uint32_t columns)
{
    const char *root = std::getenv("MIMO26_PROJECTION_CAPTURE");
    if (root == NULL) return;
    if (capture_written >= env_unsigned("MIMO26_PROJECTION_CAPTURE_MAX", 12u)) return;
    /* A prime stride spreads captures over layers, experts and both shapes
     * instead of taking the first N, which would all be layer 0. */
    if (capture_seen++ % env_unsigned("MIMO26_PROJECTION_CAPTURE_STRIDE", 97u) != 0u) return;

    char directory[PATH_MAX];
    std::snprintf(directory, sizeof directory, "%s/l%u-n%u-%ux%ux%u", root,
                  current_layer, capture_seen, vectors, rows, columns);
    if (mkdir(directory, 0755) != 0 && errno != EEXIST) return;

    const std::string prefix = std::string(directory) + "/projection";
    bool ok = dump_device(prefix.c_str(), "-packed.bin", packed, (size_t)rows * columns / 2);
    ok = dump_device(prefix.c_str(), "-scales.bin", scales, (size_t)rows * columns / 32) && ok;
    ok = dump_device(prefix.c_str(), "-input.bin", input, (size_t)vectors * columns * 2) && ok;
    capture_written += ok ? 1u : 0u;
    std::fprintf(stderr, "projection_capture layer=%u seen=%u written=%u saved=%d "
                 "vectors=%u rows=%u columns=%u dir=%s\n", current_layer, capture_seen,
                 capture_written, (int)ok, vectors, rows, columns, directory);
}

static bool check_projection(
    void *output, const void *packed, const void *scales, const void *input,
    uint32_t vectors, uint32_t rows, uint32_t columns, void *stream, bool exact_rows)
{
    auto implementation = exact_rows ? __real_k3_rocm_mxfp4_gemv_rows_bf16
                                     : __real_k3_rocm_mxfp4_gemm_bf16;
    if (!implementation(
            output, packed, scales, input, vectors, rows, columns, stream))
        return false;
    if (vectors == 1) return true;
    projection++;
    maybe_capture(packed, scales, input, vectors, rows, columns);
    size_t bytes = (size_t)vectors * rows * sizeof(uint16_t);
    uint16_t *reference = nullptr;
    if (hipMalloc(&reference, bytes) != hipSuccess) return false;
    bool ok = true;
    for (uint32_t v = 0; v < vectors && ok; v++)
        ok = __real_k3_rocm_mxfp4_gemm_bf16(
            reference + (size_t)v * rows, packed, scales,
            (const uint16_t *)input + (size_t)v * columns,
            1, rows, columns, stream);
    std::vector<uint16_t> batch(bytes / 2), loop(bytes / 2);
    ok = ok && hipStreamSynchronize((hipStream_t)stream) == hipSuccess &&
        hipMemcpy(batch.data(), output, bytes, hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(loop.data(), reference, bytes, hipMemcpyDeviceToHost) == hipSuccess;
    if (ok && std::memcmp(batch.data(), loop.data(), bytes) != 0) {
        size_t differences = 0;
        for (size_t i = 0; i < batch.size(); i++) {
            if (batch[i] == loop[i]) continue;
            if (differences++ < 8)
                std::fprintf(stderr, "projection_mismatch layer=%u position=%llu "
                    "projection=%u vectors=%u rows=%u columns=%u vector=%zu "
                    "row=%zu batch=%04x loop=%04x\n", current_layer,
                    (unsigned long long)current_position, projection,
                    vectors, rows, columns, i / rows, i % rows,
                    (unsigned)batch[i], (unsigned)loop[i]);
        }
        std::fprintf(stderr, "projection_mismatch elements=%zu/%zu\n",
                     differences, batch.size());
        const char *prefix = std::getenv("MIMO26_PROJECTION_DUMP");
        if (prefix) {
            bool saved = dump_device(prefix, "-packed.bin", packed, (size_t)rows * columns / 2);
            saved = dump_device(prefix, "-scales.bin", scales, (size_t)rows * columns / 32) && saved;
            saved = dump_device(prefix, "-input.bin", input, (size_t)vectors * columns * 2) && saved;
            saved = dump_device(prefix, "-batch.bin", output, bytes) && saved;
            saved = dump_device(prefix, "-loop.bin", reference, bytes) && saved;
            std::fprintf(stderr, "projection_dump saved=%d prefix=%s\n", saved, prefix);
        }
        /* In capture mode a mismatch is the expected finding, not a reason to
         * stop: aborting here is what limited the 2026-09-24 evidence to one
         * projection of one shape. The oracle decides who is right afterwards. */
        ok = capturing();
    }
    hipFree(reference);
    return ok;
}

extern "C" bool __wrap_k3_rocm_mxfp4_gemm_bf16(
    void *output, const void *packed, const void *scales, const void *input,
    uint32_t vectors, uint32_t rows, uint32_t columns, void *stream)
{
    return check_projection(output, packed, scales, input, vectors, rows, columns, stream, false);
}

extern "C" bool __wrap_k3_rocm_mxfp4_gemv_rows_bf16(
    void *output, const void *packed, const void *scales, const void *input,
    uint32_t vectors, uint32_t rows, uint32_t columns, void *stream)
{
    return check_projection(output, packed, scales, input, vectors, rows, columns, stream, true);
}
