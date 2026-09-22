/*
 * Confirm k3_expert_cache is reusable for MiMo's routing shape, and measure
 * what hit rate a given resident share actually produces.
 *
 * The memory budget reports that about 67% of experts fit at 8K context. That
 * is a resident *share*, not a hit *rate*: the two differ by however skewed
 * routing is, which the budget cannot know. The distributions here are
 * synthetic and are labelled as such -- they bracket the answer rather than
 * predict it. A real trace needs a real forward pass, which is M4.
 */
#include "k3_expert_cache.h"
#include "mimo26_architecture.h"
#include "mimo26_manifest.h"
#include "mimo26_router.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOE_LAYERS 47u
#define TOKENS 4000u

static uint32_t next_random(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double unit_random(uint32_t *state)
{
    return (double)(next_random(state) >> 8) / (double)(1u << 24);
}

/*
 * Draw an expert id with a Zipf-like bias: exponent 0 is uniform, larger
 * values concentrate mass on the low ids.
 *
 * Always derive from the high bits. An LCG's low bits have period 2^k, so
 * `next_random() % 256` cycles with period 256 and recurs every 32 batches --
 * just outside a 64-slot LRU, which produced exactly zero hits and looked
 * like a cache bug rather than a bad generator.
 */
static uint16_t draw_expert(uint32_t *state, double exponent)
{
    if (exponent <= 0.0) {
        return (uint16_t)((next_random(state) >> 8) % MIMO26_ROUTER_EXPERTS);
    }
    const double u = unit_random(state);
    const double scaled = pow(u, exponent + 1.0);
    uint32_t id = (uint32_t)(scaled * (double)MIMO26_ROUTER_EXPERTS);
    if (id >= MIMO26_ROUTER_EXPERTS) {
        id = MIMO26_ROUTER_EXPERTS - 1u;
    }
    return (uint16_t)id;
}

static void draw_batch(uint32_t *state, double exponent, uint16_t *ids)
{
    for (size_t i = 0; i < MIMO26_ROUTER_TOP_K; i++) {
        bool duplicate;
        uint16_t candidate;
        do {
            candidate = draw_expert(state, exponent);
            duplicate = false;
            for (size_t j = 0; j < i; j++) {
                if (ids[j] == candidate) {
                    duplicate = true;
                    break;
                }
            }
        } while (duplicate);
        ids[i] = candidate;
    }
}

static double measure_hit_rate(uint16_t slots_per_layer, double exponent,
                               uint32_t seed)
{
    char error[256];
    k3_expert_cache *cache = NULL;
    assert(k3_expert_cache_create(&cache, (uint16_t)MOE_LAYERS,
                                  slots_per_layer, error, sizeof error));
    uint32_t state = seed;
    for (uint32_t token = 0; token < TOKENS; token++) {
        for (uint16_t layer = 0; layer < MOE_LAYERS; layer++) {
            uint16_t ids[MIMO26_ROUTER_TOP_K];
            k3_expert_cache_access accesses[MIMO26_ROUTER_TOP_K];
            draw_batch(&state, exponent, ids);
            assert(k3_expert_cache_plan(cache, layer, ids,
                                        MIMO26_ROUTER_TOP_K, accesses, error,
                                        sizeof error));
            /* A hit must name a readable source slot; an admission must name a
             * destination distinct from any hit's source, which is what lets a
             * copy proceed while ROCm still reads the hits. */
            for (size_t i = 0; i < MIMO26_ROUTER_TOP_K; i++) {
                if (accesses[i].hit) {
                    assert(accesses[i].source_slot !=
                           K3_EXPERT_CACHE_NO_SLOT);
                } else if (accesses[i].admit) {
                    assert(accesses[i].destination_slot !=
                           K3_EXPERT_CACHE_NO_SLOT);
                    for (size_t j = 0; j < MIMO26_ROUTER_TOP_K; j++) {
                        if (accesses[j].hit) {
                            assert(accesses[j].source_slot !=
                                   accesses[i].destination_slot);
                        }
                    }
                }
            }
            assert(k3_expert_cache_commit(cache, layer, error, sizeof error));
        }
    }
    k3_expert_cache_stats stats;
    k3_expert_cache_get_stats(cache, &stats);
    const double rate = (double)stats.hits / (double)stats.accesses;
    k3_expert_cache_destroy(cache);
    return rate;
}

int main(void)
{
    char error[256];

    /* MiMo's shape: 47 MoE layers, 256 experts each, top-8. */
    k3_expert_cache *cache = NULL;
    assert(k3_expert_cache_create(&cache, (uint16_t)MOE_LAYERS, 64u, error,
                                  sizeof error));
    assert(k3_expert_cache_slot_count(cache) == MOE_LAYERS * 64u);
    const uint64_t bytes = k3_expert_cache_storage_bytes(
        cache, mimo26_expert_bytes());
    printf("  ok  cache accepts MiMo's shape: %u layers x 64 slots, "
           "%.3f GiB of expert storage\n", MOE_LAYERS,
           (double)bytes / (double)(1u << 30));

    /* Two-phase contract: a plan is visible but not yet committed, and an
     * abort leaves residency unchanged. */
    uint16_t ids[MIMO26_ROUTER_TOP_K] = {5u, 9u, 11u, 40u, 77u, 120u, 200u, 255u};
    k3_expert_cache_access accesses[MIMO26_ROUTER_TOP_K];
    assert(k3_expert_cache_plan(cache, 0u, ids, MIMO26_ROUTER_TOP_K, accesses,
                                error, sizeof error));
    for (size_t i = 0; i < MIMO26_ROUTER_TOP_K; i++) {
        assert(!accesses[i].hit); /* cold cache */
        assert(accesses[i].admit);
    }
    /* Only one plan may be pending per layer. */
    assert(!k3_expert_cache_plan(cache, 0u, ids, MIMO26_ROUTER_TOP_K, accesses,
                                 error, sizeof error));
    k3_expert_cache_abort(cache, 0u);
    assert(k3_expert_cache_plan(cache, 0u, ids, MIMO26_ROUTER_TOP_K, accesses,
                                error, sizeof error));
    for (size_t i = 0; i < MIMO26_ROUTER_TOP_K; i++) {
        assert(!accesses[i].hit); /* the abort admitted nothing */
    }
    assert(k3_expert_cache_commit(cache, 0u, error, sizeof error));
    /* Replaying the same batch now hits everywhere. */
    assert(k3_expert_cache_plan(cache, 0u, ids, MIMO26_ROUTER_TOP_K, accesses,
                                error, sizeof error));
    for (size_t i = 0; i < MIMO26_ROUTER_TOP_K; i++) {
        assert(accesses[i].hit);
        assert(!accesses[i].admit);
    }
    assert(k3_expert_cache_commit(cache, 0u, error, sizeof error));
    printf("  ok  plan/commit is two-phase; abort admits nothing; one plan "
           "per layer\n");

    /* Layers are independent: layer 1 must not see layer 0's residents. */
    assert(k3_expert_cache_plan(cache, 1u, ids, MIMO26_ROUTER_TOP_K, accesses,
                                error, sizeof error));
    for (size_t i = 0; i < MIMO26_ROUTER_TOP_K; i++) {
        assert(!accesses[i].hit);
    }
    k3_expert_cache_abort(cache, 1u);
    printf("  ok  residency is per layer\n");

    k3_expert_cache_stats stats;
    k3_expert_cache_get_stats(cache, &stats);
    assert(stats.accesses > 0u && stats.hits > 0u && stats.misses > 0u);
    assert(k3_expert_cache_reset(cache, error, sizeof error));
    k3_expert_cache_get_stats(cache, &stats);
    assert(stats.accesses == 0u && stats.hits == 0u);
    printf("  ok  reset clears residency and telemetry\n");
    k3_expert_cache_destroy(cache);

    /*
     * Hit rate against resident share, synthetic routing. The budget's ~67%
     * resident share at 8K context corresponds to about 171 of 256 slots per
     * layer.
     */
    printf("\n  synthetic hit rate by slots per layer and routing skew\n");
    printf("  %-8s %-10s %10s %10s %10s\n", "slots", "share", "uniform",
           "skew 1.0", "skew 2.5");
    const uint16_t ladder[] = {32u, 64u, 128u, 171u, 224u};
    for (size_t i = 0; i < sizeof ladder / sizeof ladder[0]; i++) {
        const uint16_t slots = ladder[i];
        const double share = (double)slots / (double)MIMO26_ROUTER_EXPERTS;
        const double uniform = measure_hit_rate(slots, 0.0, 11u);
        const double mild = measure_hit_rate(slots, 1.0, 12u);
        const double heavy = measure_hit_rate(slots, 2.5, 13u);
        printf("  %-8u %-10.1f%% %9.1f%% %9.1f%% %9.1f%%\n", slots,
               100.0 * share, 100.0 * uniform, 100.0 * mild, 100.0 * heavy);
        /* Uniform routing cannot beat the resident share by much: that is the
         * floor the budget's share figure would imply. */
        assert(uniform <= share + 0.05);
        /* Skew must help, or the experiment says nothing. */
        assert(heavy > uniform);
    }
    printf("\n  Resident share is a floor, not the hit rate: skewed routing\n"
           "  raises it well above the share, uniform routing does not. A real\n"
           "  trace needs a real forward pass (M4), so these bracket rather\n"
           "  than predict.\n");

    printf("\ntest_mimo26_expert_cache: ok\n");
    return 0;
}
