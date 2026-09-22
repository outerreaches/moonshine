#include "mimo26_manifest.h"

#include "mimo26_architecture.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static const char *const good_index =
    "{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":12288,"
    "\"tp_size\":4},"
    "\"weight_map\":{"
    "\"model.norm.weight\":\"model_pp0_ep0_shard0.safetensors\","
    "\"lm_head.weight\":\"model_pp0_ep1_shard0.safetensors\","
    "\"model.embed_tokens.weight\":\"model_pp0_ep0_shard0.safetensors\"}}";

static bool parse(const char *json, mimo26_manifest *manifest, char *error,
                  size_t error_size)
{
    return mimo26_manifest_parse(manifest, json, strlen(json), error,
                                 error_size);
}

static void expect_reject(const char *json, const char *label)
{
    mimo26_manifest manifest;
    char error[256] = {0};
    memset(&manifest, 0x7f, sizeof manifest);
    assert(!parse(json, &manifest, error, sizeof error));
    /* A rejected parse must leave nothing allocated and nothing to free. */
    mimo26_manifest zero = {0};
    assert(memcmp(&manifest, &zero, sizeof zero) == 0);
    assert(error[0] != '\0');
    printf("  ok  reject %s (%s)\n", label, error);
}

int main(void)
{
    mimo26_manifest manifest;
    char error[256] = {0};

    assert(parse(good_index, &manifest, error, sizeof error));
    assert(manifest.entry_count == 3);
    assert(manifest.shard_count == 2);
    assert(manifest.total_size == 12288);
    assert(manifest.tp_size == MIMO26_INDEX_TP_SIZE);
    assert(strcmp(manifest.save_format, MIMO26_INDEX_SAVE_FORMAT) == 0);
    /* Shard names are sorted, so indices are deterministic across runs. */
    assert(strcmp(manifest.shard_names[0],
                  "model_pp0_ep0_shard0.safetensors") == 0);
    assert(strcmp(manifest.shard_names[1],
                  "model_pp0_ep1_shard0.safetensors") == 0);
    /* Entries are sorted by name, and each keeps its own shard binding. */
    assert(strcmp(manifest.entries[0].name, "lm_head.weight") == 0);
    assert(manifest.entries[0].shard == 1);
    assert(mimo26_manifest_shard_of(&manifest, "model.norm.weight") == 0);
    assert(mimo26_manifest_shard_of(&manifest, "lm_head.weight") == 1);
    assert(mimo26_manifest_shard_of(&manifest, "model.embed_tokens.weight") == 0);
    assert(mimo26_manifest_shard_of(&manifest, "absent.weight") == -1);
    assert(mimo26_manifest_shard_of(&manifest, NULL) == -1);
    mimo26_manifest_free(&manifest);
    mimo26_manifest zero = {0};
    assert(memcmp(&manifest, &zero, sizeof zero) == 0);
    printf("  ok  well-formed index\n");

    /* Shard names must stay inside the checkpoint directory. */
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{\"a\":"
                  "\"../escape.safetensors\"}}", "parent traversal");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{\"a\":"
                  "\"/etc/passwd.safetensors\"}}", "absolute path");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{\"a\":"
                  "\"sub/dir.safetensors\"}}", "nested path");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{\"a\":\"\"}}",
                  "empty shard name");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{\"a\":\"weights.bin\"}}",
                  "wrong suffix");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{\"a\":\".safetensors\"}}",
                  "suffix only");

    /* Storage format is part of the identity, not a hint. */
    expect_reject("{\"metadata\":{\"save_format\":\"fp8\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{\"a\":\"s.safetensors\"}}",
                  "unexpected save_format");

    /* Required metadata must be present and integral. */
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"tp_size\":4},"
                  "\"weight_map\":{\"a\":\"s.safetensors\"}}",
                  "missing total_size");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\","
                  "\"total_size\":\"12288\",\"tp_size\":4},"
                  "\"weight_map\":{\"a\":\"s.safetensors\"}}",
                  "total_size as string");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1},"
                  "\"weight_map\":{\"a\":\"s.safetensors\"}}",
                  "missing tp_size");
    expect_reject("{\"weight_map\":{\"a\":\"s.safetensors\"}}",
                  "missing metadata");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4}}", "missing weight_map");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{}}", "empty weight_map");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":[]}", "weight_map not object");
    expect_reject("{\"metadata\":{\"save_format\":\"mxfp4\",\"total_size\":1,"
                  "\"tp_size\":4},\"weight_map\":{\"a\":5}}",
                  "shard value not a string");
    expect_reject("not json at all", "malformed JSON");

    /* Reconciliation rejects a directory that disagrees with the index. */
    assert(parse(good_index, &manifest, error, sizeof error));
    k3_st_tensor tensors[3];
    memset(tensors, 0, sizeof tensors);
    tensors[0].name = (char *)"lm_head.weight";
    tensors[0].shard = 1;
    tensors[0].byte_length = 4096;
    tensors[1].name = (char *)"model.embed_tokens.weight";
    tensors[1].shard = 0;
    tensors[1].byte_length = 4096;
    tensors[2].name = (char *)"model.norm.weight";
    tensors[2].shard = 0;
    tensors[2].byte_length = 4096;

    k3_st_model model;
    memset(&model, 0, sizeof model);
    model.tensors = tensors;
    model.tensor_count = 3;
    assert(mimo26_manifest_reconcile(&manifest, &model, error, sizeof error));
    printf("  ok  reconcile agrees\n");

    tensors[0].shard = 0; /* index says shard 1 */
    assert(!mimo26_manifest_reconcile(&manifest, &model, error, sizeof error));
    assert(strstr(error, "index says shard") != NULL);
    printf("  ok  reject shard disagreement\n");
    tensors[0].shard = 1;

    tensors[2].byte_length = 4095; /* payload sum no longer matches */
    assert(!mimo26_manifest_reconcile(&manifest, &model, error, sizeof error));
    assert(strstr(error, "payload byte sum") != NULL);
    printf("  ok  reject payload sum mismatch\n");
    tensors[2].byte_length = 4096;

    tensors[2].name = (char *)"model.unlisted.weight";
    assert(!mimo26_manifest_reconcile(&manifest, &model, error, sizeof error));
    assert(strstr(error, "not indexed") != NULL);
    printf("  ok  reject unindexed header tensor\n");
    tensors[2].name = (char *)"model.norm.weight";

    model.tensor_count = 2; /* index lists three */
    assert(!mimo26_manifest_reconcile(&manifest, &model, error, sizeof error));
    assert(strstr(error, "index lists") != NULL);
    printf("  ok  reject tensor count mismatch\n");
    mimo26_manifest_free(&manifest);

    /* Span planner argument validation, without a real checkpoint. */
    mimo26_expert_span spans[MIMO26_EXPERT_PROJECTION_COUNT];
    memset(&model, 0, sizeof model);
    assert(!mimo26_expert_spans(&model, 0, 0, spans, error, sizeof error));
    assert(strstr(error, "no routed experts") != NULL);
    assert(!mimo26_expert_spans(&model, 48, 0, spans, error, sizeof error));
    assert(strstr(error, "no routed experts") != NULL);
    assert(!mimo26_expert_spans(&model, 1, MIMO26_ROUTED_EXPERTS_PER_LAYER,
                                spans, error, sizeof error));
    assert(strstr(error, "out of range") != NULL);
    printf("  ok  span planner rejects layer 0, layer 48 and expert 256\n");

    assert(mimo26_expert_bytes() == 13369344u); /* 12.75 MiB */
    printf("  ok  expert footprint is 12.75 MiB\n");

    printf("test_mimo26_manifest: ok\n");
    return 0;
}
