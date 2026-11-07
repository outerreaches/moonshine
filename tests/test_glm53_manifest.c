#include "../glm53_manifest.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

typedef struct { char *p; size_t n, cap; } buffer;

static void append(buffer *b, const char *s) {
    const size_t n = strlen(s);
    if (b->n + n + 1u > b->cap) {
        size_t cap = b->cap ? b->cap : 4096u;
        while (cap < b->n + n + 1u) cap *= 2u;
        b->p = (char *)realloc(b->p, cap);
        if (!b->p) { perror("realloc"); abort(); }
        b->cap = cap;
    }
    memcpy(b->p + b->n, s, n + 1u);
    b->n += n;
}

static char *make_config(bool duplicate_selector, size_t *size) {
    buffer b = { 0 };
    append(&b, "{\"architectures\":[\"Glm5NextForConditionalGeneration\"],\"model_type\":\"glm5_next\",\"text_config\":{");
    append(&b, "\"model_type\":\"glm5_next_text\",\"hidden_size\":4096,\"num_hidden_layers\":45,\"num_nextn_predict_layers\":1,\"first_k_dense_replace\":3,\"n_routed_experts\":288,\"n_shared_experts\":1,\"num_experts_per_tok\":8,\"moe_intermediate_size\":2048,");
    append(&b, "\"mlp_layer_types\":[");
    for (unsigned i = 0; i < 45u; i++) { if (i) append(&b, ","); append(&b, i < 3u ? "\"dense\"" : "\"sparse\""); }
    append(&b, "],\"layer_types\":[");
    for (unsigned i = 0; i < 45u; i++) { if (i) append(&b, ","); append(&b, ((i + 1u) % 4u) == 0u ? "\"deepseek_sparse_attention\"" : "\"linear_attention\""); }
    append(&b, "],\"linear_attn_config\":{\"kda_layers\":[");
    bool comma = false;
    for (unsigned i = 0; i < 45u; i++) if (((i + 1u) % 4u) != 0u) { if (comma) append(&b, ","); char x[16]; snprintf(x, sizeof(x), "%u", i); append(&b, x); comma = true; }
    append(&b, "],\"full_attn_layers\":["); comma = false;
    for (unsigned i = 0; i < 45u; i++) if (((i + 1u) % 4u) == 0u) { if (comma) append(&b, ","); char x[16]; snprintf(x, sizeof(x), "%u", i); append(&b, x); comma = true; }
    append(&b, "]}},\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"activation_scheme\":\"dynamic\",\"weight_block_size\":[128,128],\"modules_to_not_convert\":[");
    const char *special[] = { "model.layers.1.foo", "router", "visual", "dt_bias" };
    for (size_t i = 0u; i < GLM53_MODULE_EXCLUSION_COUNT; i++) {
        if (i) append(&b, ",");
        char x[64];
        if (i < 4u) snprintf(x, sizeof(x), "\"%s\"", special[i]);
        else if (duplicate_selector && i == 4u) snprintf(x, sizeof(x), "\"router\"");
        else snprintf(x, sizeof(x), "\"unused_%04zu\"", i);
        append(&b, x);
    }
    append(&b, "]}}");
    *size = b.n;
    return b.p;
}

static char *make_index(size_t *size) {
    buffer b = { 0 };
    append(&b, "{\"metadata\":{\"total_size\":328326771576},\"weight_map\":{");
    for (size_t i = 0u; i < GLM53_INDEX_TENSOR_COUNT; i++) {
        char x[128];
        snprintf(x, sizeof(x), "%s\"tensor.%05zu\":\"model-%05zu-of-00062.safetensors\"",
                 i ? "," : "", i, i % GLM53_SHARD_COUNT + 1u);
        append(&b, x);
    }
    append(&b, "}}");
    *size = b.n;
    return b.p;
}

static int test_manifest(void) {
    size_t cs = 0u, is = 0u;
    char *config = make_config(false, &cs);
    char *index = make_index(&is);
    glm53_manifest manifest = { 0 };
    char error[256];
    CHECK(glm53_manifest_parse(&manifest, config, cs, index, is, error, sizeof(error)));
    CHECK(manifest.module_count == 1509u && manifest.entry_count == 76108u);
    CHECK(glm53_manifest_tensor_excluded(&manifest, "model.language_model.layers.1.foo.weight"));
    CHECK(!glm53_manifest_tensor_excluded(&manifest, "model.language_model.layers.10.foo.weight"));
    CHECK(glm53_manifest_tensor_excluded(&manifest, "x.router.weight"));
    CHECK(!glm53_manifest_tensor_excluded(&manifest, "x.routerish.weight"));
    CHECK(glm53_manifest_tensor_excluded(&manifest, "model.visual.blocks.0.attn.weight"));
    CHECK(glm53_manifest_tensor_excluded(&manifest, "x.dt_bias"));
    glm53_manifest_free(&manifest);

    char *bad = strdup(index);
    CHECK(bad != NULL);
    char *total = strstr(bad, "328326771576"); CHECK(total != NULL); total[0] = '9';
    CHECK(!glm53_manifest_parse(&manifest, config, cs, bad, is, error, sizeof(error)));
    free(bad);

    bad = strdup(index); CHECK(bad != NULL);
    char *shard = strstr(bad, "model-00001-of-00062.safetensors"); CHECK(shard != NULL); shard[0] = '/';
    CHECK(!glm53_manifest_parse(&manifest, config, cs, bad, is, error, sizeof(error)));
    free(bad);

    bad = strdup(index); CHECK(bad != NULL);
    char *last_name = strstr(bad, "tensor.76107"); CHECK(last_name != NULL);
    memcpy(last_name, "tensor.00000", strlen("tensor.00000"));
    CHECK(!glm53_manifest_parse(&manifest, config, cs, bad, is, error, sizeof(error)));
    free(bad);

    CHECK(!glm53_manifest_parse(&manifest, config, cs - 1u, index, is,
                                error, sizeof(error)));

    size_t dcs = 0u;
    char *duplicate = make_config(true, &dcs);
    CHECK(!glm53_manifest_parse(&manifest, duplicate, dcs, index, is, error, sizeof(error)));
    free(duplicate);

    bad = strdup(config); CHECK(bad != NULL);
    char *layers = strstr(bad, "\"mlp_layer_types\":[\"dense\""); CHECK(layers != NULL);
    char *dense = strstr(layers, "dense"); dense[0] = 'x';
    CHECK(!glm53_manifest_parse(&manifest, bad, cs, index, is, error, sizeof(error)));
    free(bad);
    free(config); free(index);
    return 0;
}

static int write_f8_file(const char *path) {
    const char json[] = "{\"x.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    size_t h = strlen(json);
    while (h % 8u) h++;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint64_t length = h;
    if (fwrite(&length, 1u, 8u, f) != 8u || fwrite(json, 1u, strlen(json), f) != strlen(json)) return -1;
    for (size_t i = strlen(json); i < h; i++) fputc(' ', f);
    fputc(0, f);
    return fclose(f);
}

static int test_safetensors_api(void) {
    char root[] = "/tmp/glm53-st-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char path[256];
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors", root);
    CHECK(write_f8_file(path) == 0);
    k3_st_model model = { 0 };
    char error[256];
    CHECK(k3_st_model_open_5digit_total(&model, root, 1u, error, sizeof(error)));
    CHECK(model.tensor_count == 1u && model.tensors[0].dtype == K3_ST_DTYPE_F8_E4M3 && model.tensors[0].byte_length == 1u);
    k3_st_model_close(&model);
    CHECK(!k3_st_model_open(&model, root, 1u, error, sizeof(error)));
    CHECK(unlink(path) == 0 && rmdir(root) == 0);
    return 0;
}

static int test_reconcile(void) {
    char *selectors[] = { "source_precision" };
    glm53_index_entry entries[] = {
        { "foo.weight", 0u }, { "foo.weight_scale_inv", 0u }
    };
    glm53_manifest manifest = { selectors, 1u, entries, 2u, 16388u };
    k3_st_shard shards[GLM53_SHARD_COUNT] = { 0 };
    shards[0].file_bytes = 20000u;
    k3_st_tensor tensors[2] = {
        { "foo.weight", 0u, 16384u, {128u,128u}, 0u, 2u,
          K3_ST_DTYPE_F8_E4M3 },
        { "foo.weight_scale_inv", 16384u, 4u, {1u,1u}, 0u, 2u,
          K3_ST_DTYPE_F32 },
    };
    k3_st_model model = {
        shards, GLM53_SHARD_COUNT, tensors, 2u, 2u, NULL, NULL
    };
    char error[256];
    CHECK(glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));

    tensors[1].dtype = K3_ST_DTYPE_BF16;
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    tensors[1].dtype = K3_ST_DTYPE_F32;

    entries[1].shard = 1u;
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    entries[1].shard = 0u;

    tensors[1].name = "foo.weight";
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    tensors[1].name = "foo.weight_scale_inv";

    tensors[1].physical_offset = 16000u;
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    tensors[1].physical_offset = 16384u;

    shards[0].file_bytes = 16387u;
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    shards[0].file_bytes = 20000u;

    shards[0].data_offset = 1u;
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    shards[0].data_offset = 0u;

    tensors[0].physical_offset = UINT64_MAX - 10u;
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    tensors[0].physical_offset = 0u;

    tensors[1].byte_length = 0u;
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    tensors[1].byte_length = 4u;

    tensors[0].shape[0] = UINT64_MAX;
    tensors[1].shape[0] = 0u; /* would match an overflowing (n+127)/128 */
    CHECK(!glm53_manifest_reconcile(&manifest, &model, error, sizeof(error)));
    tensors[0].shape[0] = 128u;
    tensors[1].shape[0] = 1u;

    CHECK(glm53_manifest_official_dtype_counts_valid(
        GLM53_INDEX_TENSOR_COUNT, 37338u, 37629u, 1141u, 37338u));
    CHECK(!glm53_manifest_official_dtype_counts_valid(
        GLM53_INDEX_TENSOR_COUNT, 37337u, 37629u, 1141u, 37338u));
    CHECK(!glm53_manifest_official_dtype_counts_valid(
        GLM53_INDEX_TENSOR_COUNT, 37338u, 37628u, 1141u, 37338u));
    CHECK(!glm53_manifest_official_dtype_counts_valid(
        GLM53_INDEX_TENSOR_COUNT, 37338u, 37629u, 1140u, 37338u));
    CHECK(!glm53_manifest_official_dtype_counts_valid(
        GLM53_INDEX_TENSOR_COUNT, 37338u, 37629u, 1141u, 37337u));
    CHECK(!glm53_manifest_official_dtype_counts_valid(
        2u, 1u, 1u, 0u, 1u));
    return 0;
}

int main(void) {
    CHECK(test_manifest() == 0);
    CHECK(test_safetensors_api() == 0);
    CHECK(test_reconcile() == 0);
    puts("test_glm53_manifest: ok");
    return 0;
}
