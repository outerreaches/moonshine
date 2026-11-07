/*
 * Dequantize real layer and expert weights and print a checksum per tensor,
 * so an independent dequantizer can be compared against this one.
 *
 *   mimo26_dump_weights CHECKPOINT_ROOT LAYER [EXPERT]
 *
 * The checksum is FNV-1a over the BF16 bytes. Both sides round F32 to BF16
 * with round-to-nearest-even, so agreement should be bit-exact rather than
 * approximate -- a tolerance here would hide a real disagreement.
 */
#include "mimo26_architecture.h"
#include "mimo26_manifest.h"
#include "mimo26_router.h"
#include "mimo26_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv1a(const void *data, size_t bytes)
{
    const unsigned char *cursor = data;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= cursor[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void report(const char *name, const uint16_t *values, size_t count)
{
    if (values == NULL) {
        printf("  %-28s absent\n", name);
        return;
    }
    double sum = 0.0;
    double maximum = 0.0;
    for (size_t i = 0; i < count; i++) {
        const uint32_t bits = (uint32_t)values[i] << 16;
        float value;
        memcpy(&value, &bits, sizeof value);
        sum += (double)value * (double)value;
        const double magnitude = value < 0.0f ? -(double)value : (double)value;
        if (magnitude > maximum) {
            maximum = magnitude;
        }
    }
    printf("  %-28s count=%-11zu fnv1a=%016llx rms=%.8e absmax=%.6e\n", name,
           count, (unsigned long long)fnv1a(values, count * sizeof *values),
           sum > 0.0 ? sqrt(sum / (double)count) : 0.0, maximum);
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s CHECKPOINT_ROOT LAYER [EXPERT]\n", argv[0]);
        return 2;
    }
    const char *root = argv[1];
    const uint32_t layer = (uint32_t)strtoul(argv[2], NULL, 10);
    const bool want_expert = (argc == 4);
    const uint32_t expert = want_expert
                                ? (uint32_t)strtoul(argv[3], NULL, 10)
                                : 0u;

    char error[512] = {0};
    mimo26_manifest manifest;
    memset(&manifest, 0, sizeof manifest);
    k3_st_model model;
    memset(&model, 0, sizeof model);
    mimo26_layer_weights weights;
    memset(&weights, 0, sizeof weights);
    mimo26_expert_weights expert_weights;
    memset(&expert_weights, 0, sizeof expert_weights);
    int result = 1;

    if (!mimo26_manifest_load(&manifest, root, error, sizeof error)) {
        fprintf(stderr, "manifest: %s\n", error);
        goto done;
    }
    if (!mimo26_manifest_open_model(&manifest, root, &model, error,
                                    sizeof error)) {
        fprintf(stderr, "open: %s\n", error);
        goto done;
    }
    if (mimo26_layer_weights_load(&weights, &model, layer, error,
                                  sizeof error) != MIMO26_WEIGHTS_OK) {
        fprintf(stderr, "layer weights: %s\n", error);
        goto done;
    }

    printf("layer %u: %s attention, %s MLP, qkv_width %zu, %.3f GiB resident\n",
           layer, weights.attention.is_swa ? "SWA" : "GLOBAL",
           weights.is_moe ? "MoE" : "dense", weights.attention.qkv_width,
           (double)weights.bytes / (double)(1u << 30));

    report("input_layernorm", weights.input_layernorm, MIMO26_HIDDEN_SIZE);
    report("post_attention_layernorm", weights.post_attention_layernorm,
           MIMO26_HIDDEN_SIZE);
    report("qkv_proj", weights.qkv_proj,
           weights.attention.qkv_width * MIMO26_HIDDEN_SIZE);
    report("o_proj", weights.o_proj,
           (size_t)MIMO26_HIDDEN_SIZE * MIMO26_QUERY_HEADS *
               MIMO26_V_HEAD_DIM);
    report("attention_sink_bias", weights.sink_bias, MIMO26_QUERY_HEADS);
    if (weights.is_moe) {
        report("mlp.gate.weight", weights.gate_weight,
               (size_t)MIMO26_ROUTER_EXPERTS * MIMO26_HIDDEN_SIZE);
        if (weights.gate_bias != NULL) {
            printf("  %-28s count=%-11u fnv1a=%016llx\n",
                   "mlp.gate.bias(f32)", MIMO26_ROUTER_EXPERTS,
                   (unsigned long long)fnv1a(
                       weights.gate_bias,
                       MIMO26_ROUTER_EXPERTS * sizeof *weights.gate_bias));
        }
    } else {
        report("mlp.gate_proj", weights.dense_gate, 16384u * 4096u);
        report("mlp.up_proj", weights.dense_up, 16384u * 4096u);
        report("mlp.down_proj", weights.dense_down, 4096u * 16384u);
    }

    if (want_expert) {
        if (mimo26_expert_weights_load(&expert_weights, &model, layer, expert,
                                       error, sizeof error) !=
            MIMO26_WEIGHTS_OK) {
            fprintf(stderr, "expert weights: %s\n", error);
            goto done;
        }
        printf("\nexpert %u of layer %u: %.3f MiB resident\n", expert, layer,
               (double)expert_weights.bytes / (double)(1u << 20));
        report("expert.gate_proj", expert_weights.gate, 2048u * 4096u);
        report("expert.up_proj", expert_weights.up, 2048u * 4096u);
        report("expert.down_proj", expert_weights.down, 4096u * 2048u);
    }
    result = 0;
done:
    mimo26_expert_weights_free(&expert_weights);
    mimo26_layer_weights_free(&weights);
    k3_st_model_close(&model);
    mimo26_manifest_free(&manifest);
    return result;
}
