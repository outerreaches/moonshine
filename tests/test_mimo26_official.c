/*
 * M0 exit evidence against a real MiMo-V2.6-Flash checkpoint: load the index,
 * open every shard it names, reconcile the directory, validate the full
 * schema, and resolve read spans for every routed expert.
 *
 * Payload-free. Only SafeTensors headers and the index are read; no tensor
 * bytes are touched and nothing is written.
 */
#include "mimo26_architecture.h"
#include "mimo26_manifest.h"
#include "k3_safetensors.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s CHECKPOINT_ROOT\n", argv[0]);
        return 2;
    }
    const char *root = argv[1];
    char error[512] = {0};
    mimo26_manifest manifest;
    memset(&manifest, 0, sizeof manifest);
    k3_st_model model;
    memset(&model, 0, sizeof model);
    int result = 1;

    if (!mimo26_manifest_load(&manifest, root, error, sizeof error)) {
        fprintf(stderr, "manifest: %s\n", error);
        goto done;
    }
    printf("index          %zu tensors across %zu shards\n",
           manifest.entry_count, manifest.shard_count);
    printf("save_format    %s (tp_size %u)\n", manifest.save_format,
           (unsigned)manifest.tp_size);
    printf("total_size     %llu payload bytes\n",
           (unsigned long long)manifest.total_size);

    if (manifest.shard_count != MIMO26_SHARD_COUNT) {
        fprintf(stderr, "expected %u shards, index names %zu\n",
                MIMO26_SHARD_COUNT, manifest.shard_count);
        goto done;
    }
    if (manifest.entry_count != MIMO26_INDEX_TENSOR_COUNT) {
        fprintf(stderr, "expected %u tensors, index names %zu\n",
                MIMO26_INDEX_TENSOR_COUNT, manifest.entry_count);
        goto done;
    }
    if (manifest.total_size != MIMO26_INDEX_TOTAL_BYTES) {
        fprintf(stderr, "unexpected total_size %llu\n",
                (unsigned long long)manifest.total_size);
        goto done;
    }
    if (manifest.tp_size != MIMO26_INDEX_TP_SIZE) {
        fprintf(stderr, "unexpected tp_size %u\n",
                (unsigned)manifest.tp_size);
        goto done;
    }

    if (!mimo26_manifest_open_model(&manifest, root, &model, error,
                                    sizeof error)) {
        fprintf(stderr, "open: %s\n", error);
        goto done;
    }
    printf("headers        %zu tensors parsed\n", model.tensor_count);

    if (!mimo26_manifest_reconcile(&manifest, &model, error, sizeof error)) {
        fprintf(stderr, "reconcile: %s\n", error);
        goto done;
    }
    printf("reconcile      index and headers agree\n");

    mimo26_architecture_report report;
    memset(&report, 0, sizeof report);
    if (!mimo26_architecture_validate(&model, &report, error, sizeof error)) {
        fprintf(stderr, "architecture: %s\n", error);
        goto done;
    }
    printf("architecture   main %zu, expert %zu, mtp %zu, vision %zu, "
           "audio %zu\n", report.main_count, report.expert_count,
           report.mtp_count, report.vision_count, report.audio_count);
    printf("attention      %zu swa tensors, %zu global tensors, "
           "%zu dense-layer tensors\n", report.swa_count, report.global_count,
           report.dense_count);

    /* Every routed expert must resolve, with weight and scale colocated. */
    uint64_t expert_bytes = 0;
    size_t resolved = 0;
    for (uint32_t layer = 0; layer < MIMO26_TEXT_LAYER_COUNT; layer++) {
        if (mimo26_architecture_layer_kind(layer) ==
            MIMO26_LAYER_DENSE_GLOBAL) {
            continue;
        }
        for (uint32_t expert = 0; expert < MIMO26_ROUTED_EXPERTS_PER_LAYER;
             expert++) {
            mimo26_expert_span spans[MIMO26_EXPERT_PROJECTION_COUNT];
            if (!mimo26_expert_spans(&model, layer, expert, spans, error,
                                     sizeof error)) {
                fprintf(stderr, "spans: %s\n", error);
                goto done;
            }
            uint64_t total = 0;
            for (size_t i = 0; i < MIMO26_EXPERT_PROJECTION_COUNT; i++) {
                total += spans[i].weight_bytes + spans[i].scale_bytes;
            }
            if (total != mimo26_expert_bytes()) {
                fprintf(stderr,
                        "layer %u expert %u is %llu bytes, expected %llu\n",
                        layer, expert, (unsigned long long)total,
                        (unsigned long long)mimo26_expert_bytes());
                goto done;
            }
            expert_bytes += total;
            resolved++;
        }
    }
    printf("expert spans   %zu experts resolved, %.4f GiB total\n", resolved,
           (double)expert_bytes / (double)(1u << 30));

    if (resolved != (size_t)MIMO26_MOE_LAYER_COUNT *
                    MIMO26_ROUTED_EXPERTS_PER_LAYER) {
        fprintf(stderr, "resolved %zu experts, expected %u\n", resolved,
                MIMO26_MOE_LAYER_COUNT * MIMO26_ROUTED_EXPERTS_PER_LAYER);
        goto done;
    }

    /* Static text footprint: everything that is not a routed expert or a
     * deferred modality or speculation group. */
    uint64_t static_text = 0;
    for (size_t i = 0; i < model.tensor_count; i++) {
        const char *name = model.tensors[i].name;
        if (strstr(name, ".mlp.experts.") != NULL ||
            strncmp(name, "model.mtp.", 10) == 0 ||
            strncmp(name, "speech_embeddings.", 18) == 0 ||
            strncmp(name, "audio_encoder.", 14) == 0 ||
            strstr(name, "vision") != NULL || strstr(name, "visual") != NULL) {
            continue;
        }
        static_text += model.tensors[i].byte_length;
    }
    printf("static text    %.4f GiB resident before cache and KV\n",
           (double)static_text / (double)(1u << 30));

    printf("\ntest_mimo26_official: ok\n");
    result = 0;
done:
    k3_st_model_close(&model);
    mimo26_manifest_free(&manifest);
    return result;
}
