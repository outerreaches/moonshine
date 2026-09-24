#ifndef MIMO26_EXPERT_GROUP_PLAN_H
#define MIMO26_EXPERT_GROUP_PLAN_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// A group appears once in this chunk's schedule. Only already-routed groups
// are considered: this is not a prediction across requests or chunks.
static inline bool mimo26_expert_group_next_use(
    uint32_t expert, const uint32_t *remaining, size_t count,
    uint32_t next_use[256])
{
    if (expert >= 256 || count >= 256 || (count && !remaining) || !next_use)
        return false;
    for (unsigned e = 0; e < 256; e++) next_use[e] = UINT32_MAX;
    for (size_t i = 0; i < count; i++) {
        uint32_t e = remaining[i];
        if (e >= 256 || e == expert || next_use[e] != UINT32_MAX) return false;
        next_use[e] = (uint32_t)i;
    }
    return true;
}
#endif
