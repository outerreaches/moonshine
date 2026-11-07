#include "../mimo26_expert_group_plan.h"
#include "../k3_expert_cache.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
    uint32_t next[256], future[] = {0, 1}, duplicate[] = {0, 0};
    uint32_t current[] = {3}, invalid[] = {256};
    assert(mimo26_expert_group_next_use(3, future, 2, next));
    assert(next[0] == 0 && next[1] == 1 && next[2] == UINT32_MAX);
    assert(!mimo26_expert_group_next_use(3, duplicate, 2, next));
    assert(!mimo26_expert_group_next_use(3, current, 1, next));
    assert(!mimo26_expert_group_next_use(3, invalid, 1, next));
    assert(!mimo26_expert_group_next_use(256, NULL, 0, next));
    assert(!mimo26_expert_group_next_use(3, NULL, 1, next));
    assert(!mimo26_expert_group_next_use(3, NULL, 0, NULL));
    assert(!mimo26_expert_group_next_use(3, future, 256, next));
    assert(mimo26_expert_group_next_use(3, NULL, 0, next));
    for (unsigned i = 0; i < 256; i++) assert(next[i] == UINT32_MAX);

    k3_expert_cache *cache = NULL;
    char error[256];
    assert(k3_expert_cache_create(&cache, 1, 3, error, sizeof error));
    uint16_t seed[] = {0, 1, 2};
    k3_expert_cache_access access[3];
    assert(k3_expert_cache_plan(cache, 0, seed, 3, access, error, sizeof error));
    assert(k3_expert_cache_commit(cache, 0, error, sizeof error));
    uint16_t incoming[] = {3};
    assert(mimo26_expert_group_next_use(3, future, 2, next));
    assert(k3_expert_cache_plan_next_use(cache, 0, incoming, 1, next, 256,
                                       access, error, sizeof error));
    // Oldest entries 0/1 are still needed; evict the newer already-dead 2.
    assert(access[0].admit && access[0].destination_slot == 2);
    assert(k3_expert_cache_commit(cache, 0, error, sizeof error));
    for (unsigned i = 0; i < 2; i++) {
        uint16_t id = (uint16_t)future[i];
        assert(mimo26_expert_group_next_use(id, future + i + 1, 1 - i, next));
        assert(k3_expert_cache_plan_next_use(cache, 0, &id, 1, next, 256,
                                           access, error, sizeof error));
        assert(access[0].hit && !access[0].admit && access[0].source_slot == id);
        assert(k3_expert_cache_commit(cache, 0, error, sizeof error));
    }
    k3_expert_cache_destroy(cache);
    puts("PASS group next-use validation and protection of remaining resident groups");
}
