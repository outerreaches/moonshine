// Exact batched-launch contract: both production expert shapes, signed BF16
// inputs, widths across tile boundaries, and native 8-mod-16 packed alignment.
#include "k3_rocm_ops.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "failed: %s\n", #x); return 1; } } while (0)

int main()
{
    const unsigned widths[] = {1, 2, 8, 15, 16, 17, 31, 32, 64, 128};
    uint64_t state = 0x9E3779B97F4A7C15ull;
    auto random = [&]() { state ^= state << 13; state ^= state >> 7;
                         state ^= state << 17; return state; };
    for (unsigned shape = 0; shape < 2; shape++) {
        const unsigned rows = shape ? 4096 : 2048, columns = shape ? 2048 : 4096;
        const size_t n = (size_t)rows * columns;
        std::vector<uint8_t> packed(n / 2), scales(n / 32);
        std::vector<uint16_t> inputs((size_t)128 * columns);
        for (auto &v : packed) v = (uint8_t)(random() >> 24);
        for (auto &v : scales) v = (uint8_t)(116 + random() % 20);
        for (auto &v : inputs) {
            uint64_t x = random();
            v = (uint16_t)((x & 0x807f) | ((110 + (x >> 32) % 30) << 7));
        }
        void *dp = nullptr, *ds = nullptr, *di = nullptr, *db = nullptr, *dl = nullptr;
        REQUIRE(hipMalloc(&dp, packed.size() + 8) == hipSuccess);
        REQUIRE(hipMalloc(&ds, scales.size()) == hipSuccess);
        REQUIRE(hipMalloc(&di, inputs.size() * 2) == hipSuccess);
        REQUIRE(hipMalloc(&db, (size_t)128 * rows * 2) == hipSuccess);
        REQUIRE(hipMalloc(&dl, (size_t)128 * rows * 2) == hipSuccess);
        REQUIRE(hipMemcpy(ds, scales.data(), scales.size(), hipMemcpyHostToDevice) == hipSuccess);
        REQUIRE(hipMemcpy(di, inputs.data(), inputs.size() * 2, hipMemcpyHostToDevice) == hipSuccess);
        REQUIRE(!k3_rocm_mxfp4_gemv_rows_bf16(db, dp, ds, di, 0, rows, columns, nullptr));
        REQUIRE(!k3_rocm_mxfp4_gemv_rows_bf16(db, dp, ds, di, 65536, rows, columns, nullptr));
        for (unsigned offset : {0u, 8u}) {
            void *weights = (uint8_t *)dp + offset;
            REQUIRE(hipMemcpy(weights, packed.data(), packed.size(), hipMemcpyHostToDevice) == hipSuccess);
            for (unsigned width : widths) {
                REQUIRE(k3_rocm_mxfp4_gemv_rows_bf16(db, weights, ds, di,
                                                  width, rows, columns, nullptr));
                for (unsigned v = 0; v < width; v++)
                    REQUIRE(k3_rocm_mxfp4_gemv_bf16((uint16_t *)dl + (size_t)v * rows,
                        weights, ds, (uint16_t *)di + (size_t)v * columns, rows, columns, nullptr));
                REQUIRE(hipDeviceSynchronize() == hipSuccess);
                std::vector<uint16_t> batch((size_t)width * rows), loop(batch.size());
                REQUIRE(hipMemcpy(batch.data(), db, batch.size() * 2, hipMemcpyDeviceToHost) == hipSuccess);
                REQUIRE(hipMemcpy(loop.data(), dl, loop.size() * 2, hipMemcpyDeviceToHost) == hipSuccess);
                REQUIRE(batch == loop);
                std::printf("PASS rows=%u columns=%u width=%u packed_offset=%u\n",
                            rows, columns, width, offset);
            }
        }
        hipFree(dp); hipFree(ds); hipFree(di); hipFree(db); hipFree(dl);
    }
    return 0;
}
