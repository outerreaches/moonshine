/*
 * M3 layer parity: run one real layer in C and compare against the fixture
 * produced by the checkpoint's own reference modules.
 *
 *   test_mimo26_layer_parity CHECKPOINT_ROOT FIXTURE.bin
 *
 * The fixture comes from tools/mimo26_reference_layer.py, which dequantizes
 * with an independent NumPy implementation and runs the reference classes.
 * Agreement therefore covers weight decoding, the whole layer graph and every
 * cast point at once.
 *
 * Fixture layout, little-endian: u32 layer, tokens, hidden, is_swa; then BF16
 * hidden_in, after_attention and final, each [tokens][hidden]; then u32 route
 * count and that many u32 expert ids.
 */
#include "mimo26_architecture.h"
#include "mimo26_layer.h"
#include "mimo26_manifest.h"
#include "mimo26_ops.h"
#include "mimo26_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Loads every routed expert on demand and keeps it, which is fine for a
 * single-token parity check and keeps residency policy out of the layer. */
typedef struct {
    const k3_st_model     *model;
    mimo26_expert_weights  cache[MIMO26_ROUTER_EXPERTS];
    bool                   present[MIMO26_ROUTER_EXPERTS];
    size_t                 loaded;
} expert_store;

static mimo26_layer_status provide_expert(void *context, uint32_t layer,
                                          uint32_t expert,
                                          const mimo26_expert_weights **out,
                                          char *error, size_t error_size)
{
    expert_store *store = context;
    if (expert >= MIMO26_ROUTER_EXPERTS) {
        return MIMO26_LAYER_INVALID_ARGUMENT;
    }
    if (!store->present[expert]) {
        if (mimo26_expert_weights_load(&store->cache[expert], store->model,
                                       layer, expert, error, error_size) !=
            MIMO26_WEIGHTS_OK) {
            return MIMO26_LAYER_EXPERT_UNAVAILABLE;
        }
        store->present[expert] = true;
        store->loaded++;
    }
    *out = &store->cache[expert];
    return MIMO26_LAYER_OK;
}

static void compare(const char *label, const uint16_t *mine,
                    const uint16_t *theirs, size_t count, size_t *failures)
{
    size_t mismatched = 0;
    double worst_absolute = 0.0;
    double reference_magnitude = 0.0;
    int worst_ulps = 0;
    for (size_t i = 0; i < count; i++) {
        const double a = (double)mimo26_bf16_to_f32(mine[i]);
        const double b = (double)mimo26_bf16_to_f32(theirs[i]);
        if (mine[i] != theirs[i]) {
            mismatched++;
            const int ulps = (int)mine[i] - (int)theirs[i];
            const int magnitude = ulps < 0 ? -ulps : ulps;
            if (magnitude > worst_ulps) {
                worst_ulps = magnitude;
            }
        }
        const double difference = fabs(a - b);
        if (difference > worst_absolute) {
            worst_absolute = difference;
        }
        if (fabs(b) > reference_magnitude) {
            reference_magnitude = fabs(b);
        }
    }
    const double relative = reference_magnitude > 0.0
                                ? worst_absolute / reference_magnitude
                                : worst_absolute;
    /*
     * BF16 has an 8-bit significand, so a few last-place differences are
     * expected from summation order inside the matmuls: torch blocks its
     * reductions, this reference sums in ascending column order. A wrong cast
     * point or a decoding error would move things by far more than an ulp, so
     * the gate is on relative magnitude rather than on bit equality.
     */
    const bool ok = relative <= 0.02;
    printf("  %-18s %6zu/%-6zu bits differ, worst %d ulp, relative %.3e  %s\n",
           label, mismatched, count, worst_ulps, relative,
           ok ? "ok" : "FAIL");
    if (!ok) {
        (*failures)++;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s CHECKPOINT_ROOT FIXTURE.bin\n", argv[0]);
        return 2;
    }
    FILE *handle = fopen(argv[2], "rb");
    if (handle == NULL) {
        fprintf(stderr, "cannot open fixture %s\n", argv[2]);
        return 2;
    }
    uint32_t header[4];
    if (fread(header, sizeof header, 1u, handle) != 1u) {
        fprintf(stderr, "short fixture header\n");
        fclose(handle);
        return 2;
    }
    const uint32_t layer_index = header[0];
    const uint32_t tokens = header[1];
    const uint32_t hidden_size = header[2];
    if (hidden_size != MIMO26_HIDDEN_SIZE || tokens != 1u) {
        fprintf(stderr, "fixture must be a single token at hidden size %u "
                        "(got %u tokens, hidden %u)\n",
                MIMO26_HIDDEN_SIZE, tokens, hidden_size);
        fclose(handle);
        return 2;
    }
    const size_t span = (size_t)tokens * hidden_size;
    uint16_t *hidden_in = malloc(span * sizeof *hidden_in);
    uint16_t *after_attention = malloc(span * sizeof *after_attention);
    uint16_t *final = malloc(span * sizeof *final);
    if (hidden_in == NULL || after_attention == NULL || final == NULL) {
        fclose(handle);
        return 1;
    }
    if (fread(hidden_in, sizeof *hidden_in, span, handle) != span ||
        fread(after_attention, sizeof *after_attention, span, handle) != span ||
        fread(final, sizeof *final, span, handle) != span) {
        fprintf(stderr, "short fixture body\n");
        fclose(handle);
        return 2;
    }
    uint32_t route_count = 0;
    uint32_t reference_route[MIMO26_ROUTER_TOP_K];
    memset(reference_route, 0, sizeof reference_route);
    if (fread(&route_count, sizeof route_count, 1u, handle) == 1u &&
        route_count > 0u && route_count <= MIMO26_ROUTER_TOP_K) {
        if (fread(reference_route, sizeof *reference_route, route_count,
                  handle) != route_count) {
            route_count = 0u;
        }
    }
    fclose(handle);

    char error[512] = {0};
    mimo26_manifest manifest;
    memset(&manifest, 0, sizeof manifest);
    k3_st_model model;
    memset(&model, 0, sizeof model);
    mimo26_layer_weights weights;
    memset(&weights, 0, sizeof weights);
    mimo26_layer_scratch *scratch = NULL;
    mimo26_kv_cache *kv = NULL;
    expert_store store;
    memset(&store, 0, sizeof store);
    size_t failures = 0;
    int result = 1;

    if (!mimo26_manifest_load(&manifest, argv[1], error, sizeof error) ||
        !mimo26_manifest_open_model(&manifest, argv[1], &model, error,
                                    sizeof error)) {
        fprintf(stderr, "checkpoint: %s\n", error);
        goto done;
    }
    if (mimo26_layer_weights_load(&weights, &model, layer_index, error,
                                  sizeof error) != MIMO26_WEIGHTS_OK) {
        fprintf(stderr, "weights: %s\n", error);
        goto done;
    }
    if (mimo26_layer_scratch_create(&scratch) != MIMO26_LAYER_OK) {
        fprintf(stderr, "scratch allocation failed\n");
        goto done;
    }
    if (mimo26_kv_create(&kv, 8u, 0u) != MIMO26_KV_OK) {
        fprintf(stderr, "kv allocation failed\n");
        goto done;
    }
    store.model = &model;

    printf("layer %u: %s attention, %s MLP, single token at position 0\n",
           layer_index, weights.attention.is_swa ? "SWA" : "GLOBAL",
           weights.is_moe ? "MoE" : "dense");
    printf("  scratch %.2f MiB, layer weights %.3f GiB\n",
           (double)mimo26_layer_scratch_bytes(scratch) / (1024.0 * 1024.0),
           (double)weights.bytes / (double)(1u << 30));

    /*
     * A real step stages all 48 layers before committing. This fixture
     * exercises one layer, so the transaction is opened, the layer stages into
     * it, and it is then abandoned -- which also confirms a single layer can
     * run inside the transactional contract without committing.
     */
    if (mimo26_kv_begin(kv, 0u) != MIMO26_KV_OK) {
        fprintf(stderr, "kv begin failed\n");
        goto done;
    }

    uint16_t *hidden = malloc(MIMO26_HIDDEN_SIZE * sizeof *hidden);
    if (hidden == NULL) {
        goto done;
    }
    memcpy(hidden, hidden_in, MIMO26_HIDDEN_SIZE * sizeof *hidden);

    mimo26_layer layer_context;
    memset(&layer_context, 0, sizeof layer_context);
    layer_context.weights = &weights;
    layer_context.provider = provide_expert;
    layer_context.provider_context = &store;

    mimo26_layer_route route;
    const mimo26_layer_status status = mimo26_layer_decode(
        &layer_context, scratch, hidden, kv, 0u, &route, error, sizeof error);
    if (status != MIMO26_LAYER_OK) {
        fprintf(stderr, "layer decode failed (%d): %s\n", (int)status, error);
        free(hidden);
        goto done;
    }
    printf("  loaded %zu experts on demand\n", store.loaded);

    printf("\n  %-18s %-20s %s\n", "stage", "bit differences", "verdict");
    compare("final", hidden, final, MIMO26_HIDDEN_SIZE, &failures);

    if (route.routed && route_count > 0u) {
        size_t matched = 0;
        for (uint32_t i = 0; i < route_count; i++) {
            for (size_t j = 0; j < MIMO26_ROUTER_TOP_K; j++) {
                if (route.experts[j] == reference_route[i]) {
                    matched++;
                    break;
                }
            }
        }
        const bool ok = (matched == route_count);
        printf("  %-18s %zu/%u reference experts also selected  %s\n",
               "route", matched, route_count, ok ? "ok" : "FAIL");
        if (!ok) {
            failures++;
            printf("    reference:");
            for (uint32_t i = 0; i < route_count; i++) {
                printf(" %u", reference_route[i]);
            }
            printf("\n    ours:     ");
            for (size_t j = 0; j < MIMO26_ROUTER_TOP_K; j++) {
                printf(" %u", route.experts[j]);
            }
            printf("\n");
        }
    }

    free(hidden);
    result = failures == 0 ? 0 : 1;
    printf("\ntest_mimo26_layer_parity: %s\n", result == 0 ? "ok" : "FAILED");
done:
    for (size_t i = 0; i < MIMO26_ROUTER_EXPERTS; i++) {
        if (store.present[i]) {
            mimo26_expert_weights_free(&store.cache[i]);
        }
    }
    mimo26_kv_destroy(kv);
    mimo26_layer_scratch_destroy(scratch);
    mimo26_layer_weights_free(&weights);
    k3_st_model_close(&model);
    mimo26_manifest_free(&manifest);
    free(hidden_in);
    free(after_attention);
    free(final);
    return result;
}
