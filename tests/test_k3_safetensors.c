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
        "/srv/modelstore/models/moonshotai__Kimi-K3.1-Pro";
    char error[512];
    k3_st_model model;
    struct timespec open_start, open_end, read_start, read_end;
    clock_gettime(CLOCK_MONOTONIC, &open_start);
    if (!k3_st_model_open(&model, root, 96, error, sizeof(error))) {
        fprintf(stderr, "test_k3_safetensors: FAIL: %s\n", error);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &open_end);

    size_t expert_tensor_count = 0u;
    for (size_t index = 0u; index < model.tensor_count; index++) {
        if (strstr(model.tensors[index].name,
                   ".block_sparse_moe.experts.") != NULL) {
            expert_tensor_count++;
        }
    }
    if (expert_tensor_count != 92u * 896u * 6u) {
        k3_st_model_close(&model);
        return fail("routed expert tensor matrix is incomplete");
    }

    static const struct {
        const char *suffix;
        uint64_t rows;
        uint64_t columns;
    } dsa_tensors[] = {
        { "indexer.wq_b.weight", 4096, 1536 },
        { "indexer.wk.weight", 128, 7168 },
        { "indexer.weights_proj.weight", 32, 7168 },
        { "indexer.k_norm.weight", 128, 0 },
        { "indexer.k_norm.bias", 128, 0 },
    };
    for (size_t index = 0u;
         index < sizeof(dsa_tensors) / sizeof(dsa_tensors[0]); index++) {
        char name[256];
        snprintf(name, sizeof(name),
                 "language_model.model.layers.3.self_attn.%s",
                 dsa_tensors[index].suffix);
        const k3_st_tensor *tensor = k3_st_find(&model, name);
        if (!tensor || tensor->dtype != K3_ST_DTYPE_BF16 ||
            (dsa_tensors[index].columns == 0u ?
                (tensor->ndim != 1u || tensor->shape[0] != dsa_tensors[index].rows) :
                (tensor->ndim != 2u || tensor->shape[0] != dsa_tensors[index].rows ||
                 tensor->shape[1] != dsa_tensors[index].columns))) {
            k3_st_model_close(&model);
            return fail("DSA indexer tensor name, dtype, or shape mismatch");
        }
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

    const uint64_t physical_start = tensors[0]->physical_offset;
    const uint64_t physical_bytes = UINT64_C(17547264);
    if (tensors[0]->physical_offset != physical_start ||
        tensors[5]->physical_offset + tensors[5]->byte_length !=
            physical_start + physical_bytes) {
        k3_st_model_close(&model);
        return fail("expert physical range is not one six-tensor span");
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

    printf("K3.1 SafeTensors pager: PASS\n");
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
