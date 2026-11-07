#include "k31_identity.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", message); \
            return 1; \
        } \
    } while (0)

static void tensor(k3_st_tensor *output,
                   const char *name,
                   uint64_t offset,
                   uint64_t bytes,
                   uint16_t shard,
                   uint32_t dimension,
                   k3_st_dtype dtype) {
    memset(output, 0, sizeof(*output));
    output->name = (char *)name;
    output->physical_offset = offset;
    output->byte_length = bytes;
    output->shard = shard;
    output->ndim = 1;
    output->shape[0] = dimension;
    output->dtype = dtype;
}

int main(void) {
    k3_st_shard shards[2];
    memset(shards, 0, sizeof(shards));
    shards[0].file_bytes = 17548228;
    shards[0].data_offset = 960;
    shards[1].file_bytes = 112;
    shards[1].data_offset = 104;
    k3_st_tensor tensors[8];
    tensor(&tensors[0], "language_model.model.embed_tokens.weight",
           104, 8, 1, 4, K3_ST_DTYPE_BF16);
    const char *prefix =
        "language_model.model.layers.1.block_sparse_moe.experts.0.";
    const char *suffix[] = {
        "w1.weight_packed", "w1.weight_scale",
        "w2.weight_packed", "w2.weight_scale",
        "w3.weight_packed", "w3.weight_scale",
    };
    const uint64_t sizes[] = {
        5505024, 344064, 5505024, 344064, 5505024, 344064,
    };
    char names[6][128];
    uint64_t offset = 960;
    for (uint32_t index = 0; index < 6; index++) {
        snprintf(names[index], sizeof(names[index]), "%s%s", prefix, suffix[index]);
        tensor(&tensors[index + 1], names[index], offset, sizes[index], 0,
               sizes[index], K3_ST_DTYPE_U8);
        offset += sizes[index];
    }
    tensor(&tensors[7], "language_model.model.norm.weight",
           offset, 4, 0, 1, K3_ST_DTYPE_F32);
    k3_st_model model;
    memset(&model, 0, sizeof(model));
    model.shards = shards;
    model.shard_count = 2;
    model.tensors = tensors;
    model.tensor_count = 8;

    k31_model_profile profile;
    memset(&profile, 0, sizeof(profile));
    profile.model_id = K31_MODEL_ID;
    profile.model_revision = K31_MODEL_REVISION;
    profile.dsa.index_topk = 2048;
    profile.dsa.index_head_dim = 128;
    profile.dsa.index_num_heads = 32;
    profile.dsa.index_rope_dim = 0;
    profile.dsa.index_k_norm_type = K31_INDEX_K_NORM_LAYERNORM;
    profile.dsa.index_topk_freq = 1;
    profile.dsa.index_skip_topk_offset = 1;
    const uint64_t identity = k31_model_layout_crc64(&model, &profile);
    CHECK(identity == UINT64_C(0x0e85db34387f4e0e),
          "Python/native source layout CRC64 mismatch");
    profile.model_revision = "different-revision";
    CHECK(k31_model_layout_crc64(&model, &profile) != identity,
          "revision is absent from layout identity");
    CHECK(k31_model_layout_crc64(&model, NULL) == 0,
          "missing profile accepted");
    puts("PASS: K3.1 shared source layout CRC64");
    return 0;
}
