/* No GPU. Demonstrate why aborted admission forbids retaining old mappings.
 * Cache metadata owns no payload and cannot undo caller-owned overwrites. */
#include "../k3_expert_cache.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    k3_expert_cache *cache = NULL;
    char error[256];
    uint16_t first[] = {1, 2}, next[] = {3, 4};
    k3_expert_cache_access access[2];
    unsigned payload[2] = {0};
    assert(k3_expert_cache_create(&cache, 1, 2, error, sizeof error));
    assert(k3_expert_cache_plan(cache, 0, first, 2, access, error, sizeof error));
    for (unsigned i = 0; i < 2; ++i) payload[access[i].destination_slot] = first[i];
    assert(k3_expert_cache_commit(cache, 0, error, sizeof error));
    assert(k3_expert_cache_plan(cache, 0, next, 2, access, error, sizeof error));
    /* First copy lands; second copy fails. The worker aborts its plan. */
    payload[access[0].destination_slot] = next[0];
    k3_expert_cache_abort(cache, 0);
    assert(k3_expert_cache_plan(cache, 0, first, 2, access, error, sizeof error));
    assert(access[0].hit && access[1].hit);
    assert(payload[access[0].source_slot] != first[0] ||
           payload[access[1].source_slot] != first[1]);
    k3_expert_cache_abort(cache, 0);
    assert(k3_expert_cache_reset(cache, error, sizeof error));
    assert(k3_expert_cache_plan(cache, 0, first, 2, access, error, sizeof error));
    assert(!access[0].hit && !access[1].hit);
    for (unsigned i = 0; i < 2; ++i) payload[access[i].destination_slot] = first[i];
    assert(k3_expert_cache_commit(cache, 0, error, sizeof error));
    assert(k3_expert_cache_plan(cache, 0, first, 2, access, error, sizeof error));
    for (unsigned i = 0; i < 2; ++i)
        assert(access[i].hit && payload[access[i].source_slot] == first[i]);
    k3_expert_cache_abort(cache, 0);
    k3_expert_cache_destroy(cache);
    puts("PASS aborted payload overwrite invalidates retention; cold reset forces correct reload");
    return 0;
}
