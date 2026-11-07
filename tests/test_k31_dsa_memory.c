#include "k31_dsa_memory.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", message); \
            return 1; \
        } \
    } while (0)

int main(void) {
    k31_dsa_memory_plan plan;
    memset(&plan, 0, sizeof(plan));
    CHECK(k31_dsa_memory_plan_build(128, 24, 576, 128, 32, 96, 512, 2048, &plan),
          "build valid DSA memory plan");
    CHECK(plan.mla_layer_bytes == 147456u &&
          plan.index_layer_bytes == 32768u &&
          plan.mla_cache_bytes == 3538944u &&
          plan.index_cache_bytes == 786432u &&
          plan.gathered_cache_bytes == 2359296u &&
          plan.index_dot_bytes == 16384u &&
          plan.attention_workspace_bytes == 3662336u,
          "DSA cache/workspace arithmetic");

    uint64_t mla_offset = 0;
    uint64_t index_offset = 0;
    CHECK(k31_dsa_cache_offsets(&plan, 2, 3, &mla_offset, &index_offset) &&
          mla_offset == 298368u && index_offset == 66304u,
          "shared cache addressing");
    CHECK(!k31_dsa_cache_offsets(&plan, 24, 3, &mla_offset, &index_offset) &&
          !k31_dsa_cache_offsets(&plan, 2, 128, &mla_offset, &index_offset),
          "out-of-range cache addressing");

    uint64_t state_bytes = 0;
    CHECK(k31_dsa_occupied_state_bytes(&plan, 0, &state_bytes) &&
          state_bytes == 0u,
          "empty state size");
    CHECK(k31_dsa_occupied_state_bytes(&plan, 3, &state_bytes) &&
          state_bytes == 101376u,
          "occupied MLA plus indexer state size");
    CHECK(!k31_dsa_occupied_state_bytes(&plan, 129, &state_bytes),
          "state beyond context accepted");

    k31_dsa_memory_plan unchanged;
    memset(&unchanged, 0xa5, sizeof(unchanged));
    k31_dsa_memory_plan before = unchanged;
    CHECK(!k31_dsa_memory_plan_build(0, 24, 576, 128, 32, 96, 512, 2048,
                                     &unchanged) &&
          !k31_dsa_memory_plan_build(128, 0, 576, 128, 32, 96, 512, 2048,
                                     &unchanged) &&
          !k31_dsa_memory_plan_build(128, 24, 0, 128, 32, 96, 512, 2048,
                                     &unchanged) &&
          !k31_dsa_memory_plan_build(128, 24, 576, 0, 32, 96, 512, 2048,
                                     &unchanged) &&
          !k31_dsa_memory_plan_build(128, 24, 576, 128, 0, 32, 512, 2048,
                                     &unchanged) &&
          !k31_dsa_memory_plan_build(128, 24, 576, 128, 32, 96, 0, 2048,
                                     &unchanged) &&
          !k31_dsa_memory_plan_build(128, 24, 576, 128, 32, 96, 512, 0,
                                     &unchanged) &&
          !k31_dsa_memory_plan_build(128, 24, 512, 128, 32, 96, 513, 2048,
                                     &unchanged),
          "invalid zero or inconsistent geometry accepted");
    CHECK(memcmp(&unchanged, &before, sizeof(unchanged)) == 0,
          "failed planning changed output");
    CHECK(!k31_dsa_memory_plan_build(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                     UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                     UINT32_MAX, &unchanged),
          "overflowing geometry accepted");
    CHECK(!k31_dsa_memory_plan_build(128, 24, 576, 128, 32, 96, 512, 2048,
                                     NULL),
          "null output accepted");
    puts("PASS: K3.1 DSA memory and state layout");
    return 0;
}
