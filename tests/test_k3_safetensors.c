#include "k3_safetensors.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed_seconds(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

static uint64_t fnv1a64(const uint8_t *data, uint64_t bytes) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int fail(const char *message) {
    fprintf(stderr, "test_k3_safetensors: FAIL: %s\n", message);
    return 1;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    char error[512];
    k3_st_model model;
    struct timespec open_start, open_end, read_start, read_end;
    clock_gettime(CLOCK_MONOTONIC, &open_start);
    if (!k3_st_model_open(&model, root, 96, error, sizeof(error))) {
        fprintf(stderr, "test_k3_safetensors: FAIL: %s\n", error);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &open_end);

    if (model.tensor_count != 497220u) {
        k3_st_model_close(&model);
        return fail("expected exactly 497,220 tensors");
    }

    static const char *names[] = {
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight_packed",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight_scale",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight_packed",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight_scale",
    };
    const k3_st_tensor *tensors[6];
    for (size_t i = 0; i < 6; i++) {
        tensors[i] = k3_st_find(&model, names[i]);
        if (!tensors[i]) {
            k3_st_model_close(&model);
            return fail("expert tensor lookup failed");
        }
        if (tensors[i]->shard != 1u || tensors[i]->dtype != K3_ST_DTYPE_U8) {
            k3_st_model_close(&model);
            return fail("expert tensor source or dtype is wrong");
        }
        if (i > 0 &&
            tensors[i - 1u]->physical_offset +
                tensors[i - 1u]->byte_length != tensors[i]->physical_offset) {
            k3_st_model_close(&model);
            return fail("expert tensors are not physically contiguous");
        }
    }

    const uint64_t physical_start = UINT64_C(1268562960);
    const uint64_t physical_bytes = UINT64_C(17547264);
    if (tensors[0]->physical_offset != physical_start ||
        tensors[5]->physical_offset + tensors[5]->byte_length !=
            physical_start + physical_bytes) {
        k3_st_model_close(&model);
        return fail("expert physical range disagrees with the layout oracle");
    }

    k3_st_read read;
    clock_gettime(CLOCK_MONOTONIC, &read_start);
    if (!k3_st_read_span(&model, 1u, physical_start, physical_bytes, 4096u,
                         &read, error, sizeof(error))) {
        k3_st_model_close(&model);
        fprintf(stderr, "test_k3_safetensors: FAIL: %s\n", error);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &read_end);

    uint64_t hash = fnv1a64(read.data, read.data_bytes);
    if (hash != UINT64_C(0xeb0eff9b9d8ae0ef)) {
        k3_st_read_release(&read);
        k3_st_model_close(&model);
        return fail("expert bytes disagree with the staged-file oracle");
    }

    printf("K3 SafeTensors pager: PASS\n");
    printf("  tensors: %zu across %zu shards\n",
           model.tensor_count, model.shard_count);
    printf("  directory build: %.3f s\n",
           elapsed_seconds(open_start, open_end));
    printf("  expert read: %.2f MiB in %.3f ms (%.2f GiB/s, %s)\n",
           (double)read.data_bytes / (1024.0 * 1024.0),
           elapsed_seconds(read_start, read_end) * 1000.0,
           (double)read.allocation_bytes /
               elapsed_seconds(read_start, read_end) /
               (1024.0 * 1024.0 * 1024.0),
           read.used_direct_io ? "O_DIRECT" : "buffered");
    printf("  aligned request: offset=%" PRIu64 " bytes=%" PRIu64 "\n",
           read.physical_offset, read.allocation_bytes);
    printf("  expert FNV-1a64: 0x%016" PRIx64 "\n", hash);

    k3_st_read_release(&read);
    k3_st_model_close(&model);
    return 0;
}
