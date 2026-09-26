/*
 * Sweep the MXFP4 expert kernels across vector counts on REAL captured weights
 * and REAL captured activations, dumping every kernel's output for the float64
 * oracle (Scripts/mimo26-qualification/mxfp4_oracle.py) to judge.
 *
 * Why this exists. The tiled weight-reuse kernel is gated off because it
 * differs from the baseline by one BF16 step on one captured down projection.
 * Two things were missing from that evidence:
 *
 *   1. Nothing said which kernel was RIGHT. Only an oracle can.
 *   2. The capture is 8 vectors, and K3_ROCM_BATCH_TILE is 16 -- so it
 *      exercises a PARTIAL tile in a single blockIdx.y and never a full tile,
 *      multiple tiles, or a tail. tests/test_k3_mxfp4_gemm_widths.cu sweeps
 *      widths 1..128 but on synthetic inputs, and synthetic inputs are exactly
 *      what missed this difference in the first place.
 *
 * So: real data, swept widths, all three kernels dumped side by side. Inputs
 * for width W cycle the fixture's captured vectors (i % captured), which keeps
 * every value a real activation while reaching widths the capture cannot.
 *
 * Usage: sweep FIXTURE_PREFIX CAPTURED_VECTORS ROWS COLUMNS OUT_DIR [W...]
 * Writes OUT_DIR/w<W>/{projection-input.bin,out-*.bin} plus symlinks to the
 * shared weights, one directory per width, each self-describing for the oracle.
 */
#include "k3_rocm_ops.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr, "failed: %s (line %d)\n", #x, __LINE__); return 1; } } while (0)

static double seconds_now(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static std::vector<unsigned char> read_exact(const std::string &path, size_t size)
{
    std::vector<unsigned char> data(size);
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file || std::fread(data.data(), 1, size, file) != size || std::fgetc(file) != EOF) {
        std::fprintf(stderr, "invalid fixture: %s\n", path.c_str());
        std::exit(2);
    }
    std::fclose(file);
    return data;
}

static bool write_exact(const std::string &path, const void *data, size_t size)
{
    FILE *file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    const bool wrote = std::fwrite(data, 1, size, file) == size;
    return std::fclose(file) == 0 && wrote;
}

/* Absolute, so the symlink resolves from wherever the oracle is invoked. */
static std::string absolute(const std::string &path)
{
    if (!path.empty() && path[0] == '/') return path;
    char buffer[4096];
    if (!getcwd(buffer, sizeof buffer)) return path;
    return std::string(buffer) + "/" + path;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s PREFIX CAPTURED ROWS COLUMNS OUT_DIR [WIDTH...]\n", argv[0]);
        return 2;
    }
    const std::string prefix = argv[1], out_root = argv[5];
    const unsigned captured = std::strtoul(argv[2], nullptr, 10);
    const unsigned rows = std::strtoul(argv[3], nullptr, 10);
    const unsigned columns = std::strtoul(argv[4], nullptr, 10);
    REQUIRE(captured > 0 && captured <= 256 && rows > 0 && rows <= 16384 &&
            columns > 0 && columns <= 16384 && columns % 32 == 0);

    std::vector<unsigned> widths;
    for (int i = 6; i < argc; i++) {
        const unsigned width = std::strtoul(argv[i], nullptr, 10);
        REQUIRE(width > 0 && width <= 256);
        widths.push_back(width);
    }
    if (widths.empty()) {
        /* Around the tile boundary (16) and its multiples, plus the production
         * prefill chunk. These are where a tiled kernel's tail logic lives. */
        for (unsigned width : {1u, 2u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u,
                               48u, 63u, 64u, 65u, 96u, 127u, 128u})
            widths.push_back(width);
    }

    const auto packed = read_exact(prefix + "-packed.bin", (size_t)rows * columns / 2);
    const auto scales = read_exact(prefix + "-scales.bin", (size_t)rows * columns / 32);
    const auto captured_input = read_exact(prefix + "-input.bin", (size_t)captured * columns * 2);

    void *device_packed = nullptr, *device_scales = nullptr;
    REQUIRE(hipMalloc(&device_packed, packed.size()) == hipSuccess);
    REQUIRE(hipMalloc(&device_scales, scales.size()) == hipSuccess);
    REQUIRE(hipMemcpy(device_packed, packed.data(), packed.size(), hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(device_scales, scales.data(), scales.size(), hipMemcpyHostToDevice) == hipSuccess);

    REQUIRE(mkdir(out_root.c_str(), 0755) == 0 || errno == EEXIST);
    const std::string packed_source = absolute(prefix + "-packed.bin");
    const std::string scales_source = absolute(prefix + "-scales.bin");

    std::printf("sweep rows=%u columns=%u captured=%u tile=%u\n",
                rows, columns, captured, (unsigned)16);
    for (const unsigned width : widths) {
        const size_t input_bytes = (size_t)width * columns * 2;
        const size_t output_bytes = (size_t)width * rows * 2;

        /* Cycle the captured activations so every value stays real. */
        std::vector<unsigned char> input(input_bytes);
        for (unsigned v = 0; v < width; v++)
            std::memcpy(input.data() + (size_t)v * columns * 2,
                        captured_input.data() + (size_t)(v % captured) * columns * 2,
                        (size_t)columns * 2);

        void *device_input = nullptr, *baseline = nullptr, *grouped = nullptr, *tiled = nullptr;
        REQUIRE(hipMalloc(&device_input, input_bytes) == hipSuccess);
        REQUIRE(hipMalloc(&baseline, output_bytes) == hipSuccess);
        REQUIRE(hipMalloc(&grouped, output_bytes) == hipSuccess);
        REQUIRE(hipMalloc(&tiled, output_bytes) == hipSuccess);
        REQUIRE(hipMemcpy(device_input, input.data(), input_bytes, hipMemcpyHostToDevice) == hipSuccess);

        for (unsigned v = 0; v < width; v++)
            REQUIRE(k3_rocm_mxfp4_gemv_bf16((uint16_t *)baseline + (size_t)v * rows,
                device_packed, device_scales, (uint16_t *)device_input + (size_t)v * columns,
                rows, columns, nullptr));
        REQUIRE(k3_rocm_mxfp4_gemv_rows_bf16(grouped, device_packed, device_scales,
                                             device_input, width, rows, columns, nullptr));
        REQUIRE(k3_rocm_mxfp4_gemm_bf16(tiled, device_packed, device_scales,
                                        device_input, width, rows, columns, nullptr));
        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        std::vector<unsigned char> host_baseline(output_bytes), host_grouped(output_bytes),
                                   host_tiled(output_bytes);
        REQUIRE(hipMemcpy(host_baseline.data(), baseline, output_bytes, hipMemcpyDeviceToHost) == hipSuccess);
        REQUIRE(hipMemcpy(host_grouped.data(), grouped, output_bytes, hipMemcpyDeviceToHost) == hipSuccess);
        REQUIRE(hipMemcpy(host_tiled.data(), tiled, output_bytes, hipMemcpyDeviceToHost) == hipSuccess);

        size_t grouped_differing = 0, tiled_differing = 0;
        for (size_t i = 0; i < output_bytes; i += 2) {
            grouped_differing += std::memcmp(host_grouped.data() + i, host_baseline.data() + i, 2) != 0;
            tiled_differing += std::memcmp(host_tiled.data() + i, host_baseline.data() + i, 2) != 0;
        }

        const std::string directory = out_root + "/w" + std::to_string(width);
        REQUIRE(mkdir(directory.c_str(), 0755) == 0 || errno == EEXIST);
        /* Weights are identical for every width; link rather than copy so a
         * 17-width sweep costs 17 inputs, not 17 copies of a 4 MiB matrix. */
        symlink(packed_source.c_str(), (directory + "/projection-packed.bin").c_str());
        symlink(scales_source.c_str(), (directory + "/projection-scales.bin").c_str());
        REQUIRE(write_exact(directory + "/projection-input.bin", input.data(), input_bytes));
        REQUIRE(write_exact(directory + "/out-baseline-gemv.bin", host_baseline.data(), output_bytes));
        REQUIRE(write_exact(directory + "/out-gemv-rows.bin", host_grouped.data(), output_bytes));
        REQUIRE(write_exact(directory + "/out-gemm-tiled.bin", host_tiled.data(), output_bytes));

        std::printf("width %3u outputs %7zu  gemv_rows_differing %6zu  tiled_differing %6zu\n",
                    width, output_bytes / 2, grouped_differing, tiled_differing);
        std::fflush(stdout);

        /*
         * Optional interleaved timing. The whole reason to want the tiled
         * kernel is weight reuse: gemv_rows re-reads the expert's weights once
         * per vector block, so the traffic should fall roughly with the tile.
         * Interleaved ABBA rather than sequential blocks, because this box
         * drifts ~2.6% and warms up 13-21% -- a sequential A-then-B would
         * attribute that drift to the kernel.
         */
        if (const char *bench = std::getenv("MIMO26_SWEEP_BENCH")) {
            const unsigned repeats = std::strtoul(bench, nullptr, 10);
            double rows_seconds = 0.0, tiled_seconds = 0.0;
            for (unsigned pass = 0; pass < repeats + 1u; pass++) {
                /* ABBA: rows, tiled, tiled, rows -- symmetric in time. */
                double elapsed[4] = {0, 0, 0, 0};
                const bool order[4] = {true, false, false, true};
                for (unsigned step = 0; step < 4u; step++) {
                    REQUIRE(hipDeviceSynchronize() == hipSuccess);
                    const double start = seconds_now();
                    if (order[step])
                        REQUIRE(k3_rocm_mxfp4_gemv_rows_bf16(grouped, device_packed,
                            device_scales, device_input, width, rows, columns, nullptr));
                    else
                        REQUIRE(k3_rocm_mxfp4_gemm_bf16(tiled, device_packed,
                            device_scales, device_input, width, rows, columns, nullptr));
                    REQUIRE(hipDeviceSynchronize() == hipSuccess);
                    elapsed[step] = seconds_now() - start;
                }
                if (pass == 0u) continue;   /* discard the warm-up pass */
                rows_seconds += elapsed[0] + elapsed[3];
                tiled_seconds += elapsed[1] + elapsed[2];
            }
            const double rows_mean = rows_seconds / (2.0 * repeats);
            const double tiled_mean = tiled_seconds / (2.0 * repeats);
            std::printf("          bench gemv_rows %.6f s  tiled %.6f s  speedup %.2fx\n",
                        rows_mean, tiled_mean,
                        tiled_mean > 0.0 ? rows_mean / tiled_mean : 0.0);
            std::fflush(stdout);
        }

        hipFree(device_input); hipFree(baseline); hipFree(grouped); hipFree(tiled);
    }
    hipFree(device_packed); hipFree(device_scales);
    std::printf("sweep complete out=%s\n", out_root.c_str());
    return 0;
}
