// Replay a real projection captured by mimo26_expert_major_projection_guard.
// The fixture is external evidence, not silently downloaded by this test.
// Usage: replay PREFIX VECTORS ROWS COLUMNS
#include "k3_rocm_ops.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static std::vector<unsigned char> read(const std::string &path, size_t size)
{
    std::vector<unsigned char> data(size);
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f || std::fread(data.data(), 1, size, f) != size || std::fgetc(f) != EOF) {
        std::fprintf(stderr, "invalid fixture: %s\n", path.c_str());
        std::exit(2);
    }
    std::fclose(f);
    return data;
}

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "failed: %s\n", #x); return 1; } } while (0)

int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    const std::string prefix = argv[1];
    const unsigned vectors = std::strtoul(argv[2], nullptr, 10);
    const unsigned rows = std::strtoul(argv[3], nullptr, 10);
    const unsigned columns = std::strtoul(argv[4], nullptr, 10);
    REQUIRE(vectors > 1 && vectors <= 256 && rows > 0 && rows <= 16384 &&
            columns > 0 && columns <= 16384 && columns % 32 == 0);
    auto packed = read(prefix + "-packed.bin", (size_t)rows * columns / 2);
    auto scales = read(prefix + "-scales.bin", (size_t)rows * columns / 32);
    auto inputs = read(prefix + "-input.bin", (size_t)vectors * columns * 2);
    const size_t bytes = (size_t)vectors * rows * 2;
    auto expected = read(prefix + "-loop.bin", bytes);
    void *dp = nullptr, *ds = nullptr, *di = nullptr, *dloop = nullptr;
    void *drows = nullptr, *dtiled = nullptr;
    REQUIRE(hipMalloc(&dp, packed.size()) == hipSuccess);
    REQUIRE(hipMalloc(&ds, scales.size()) == hipSuccess);
    REQUIRE(hipMalloc(&di, inputs.size()) == hipSuccess);
    REQUIRE(hipMalloc(&dloop, bytes) == hipSuccess);
    REQUIRE(hipMalloc(&drows, bytes) == hipSuccess);
    REQUIRE(hipMalloc(&dtiled, bytes) == hipSuccess);
    REQUIRE(hipMemcpy(dp, packed.data(), packed.size(), hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(ds, scales.data(), scales.size(), hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(di, inputs.data(), inputs.size(), hipMemcpyHostToDevice) == hipSuccess);
    for (unsigned v = 0; v < vectors; v++)
        REQUIRE(k3_rocm_mxfp4_gemv_bf16((uint16_t *)dloop + (size_t)v * rows,
            dp, ds, (uint16_t *)di + (size_t)v * columns, rows, columns, nullptr));
    REQUIRE(k3_rocm_mxfp4_gemv_rows_bf16(drows, dp, ds, di, vectors, rows, columns, nullptr));
    REQUIRE(k3_rocm_mxfp4_gemm_bf16(dtiled, dp, ds, di, vectors, rows, columns, nullptr));
    REQUIRE(hipDeviceSynchronize() == hipSuccess);
    std::vector<unsigned char> loop(bytes), batch(bytes), tiled(bytes);
    REQUIRE(hipMemcpy(loop.data(), dloop, bytes, hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpy(batch.data(), drows, bytes, hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpy(tiled.data(), dtiled, bytes, hipMemcpyDeviceToHost) == hipSuccess);
    const bool unchanged = loop == expected, exact = batch == loop;
    size_t differing = 0;
    for (size_t i = 0; i < bytes; i += 2)
        differing += std::memcmp(tiled.data() + i, loop.data() + i, 2) != 0;
    std::printf("baseline_unchanged=%d grouped_gemv_exact=%d tiled_differing=%zu/%zu\n",
                unchanged, exact, differing, bytes / 2);
    /*
     * Optionally re-emit all three outputs so the float64 oracle
     * (Scripts/mimo26-qualification/mxfp4_oracle.py) can judge each kernel
     * against the true value rather than against its sibling. The guard's own
     * capture holds only two of the three, and which kernel wrote its
     * "-batch.bin" depends on the call site that tripped it -- dumping here
     * names each file after the kernel that produced it, so the labelling
     * cannot be lost. Writes are refused if the file exists ("wbx").
     */
    if (const char *dump = std::getenv("MIMO26_REPLAY_DUMP")) {
        const std::pair<const char *, const std::vector<unsigned char> *> files[] = {
            {"-gemv-loop.bin", &loop}, {"-gemv-rows.bin", &batch}, {"-gemm-tiled.bin", &tiled}};
        for (const auto &entry : files) {
            FILE *f = std::fopen((std::string(dump) + entry.first).c_str(), "wbx");
            REQUIRE(f != nullptr);
            const bool wrote = std::fwrite(entry.second->data(), 1, bytes, f) == bytes;
            REQUIRE(std::fclose(f) == 0 && wrote);
        }
        std::printf("replay_dump prefix=%s\n", dump);
    }
    hipFree(dp); hipFree(ds); hipFree(di); hipFree(dloop); hipFree(drows); hipFree(dtiled);
    REQUIRE(unchanged && exact);
    return 0;
}
