#include "k3_prefill.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                     \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int main(void) {
    k3_prefill_ring_timeline timeline;
    memset(&timeline, 0xa5, sizeof(timeline));
    CHECK(k3_prefill_ring_timeline_begin(&timeline, 100u, 0u),
          "timeline begin");
    CHECK(k3_prefill_ring_timeline_update(&timeline, 110u, 2u) &&
              k3_prefill_ring_timeline_update(&timeline, 140u, 1u) &&
              k3_prefill_ring_timeline_update(&timeline, 155u, 2u) &&
              k3_prefill_ring_timeline_finish(&timeline, 175u),
          "timeline updates");
    CHECK(timeline.depth_nanoseconds[0] == 10u &&
              timeline.depth_nanoseconds[1] == 15u &&
              timeline.depth_nanoseconds[2] == 50u,
          "time-weighted occupancy");
    CHECK(timeline.transitions == 3u && timeline.max_depth == 2u &&
              timeline.current_depth == 2u,
          "timeline transition ledger");

    const k3_prefill_ring_timeline accepted = timeline;
    CHECK(!k3_prefill_ring_timeline_update(&timeline, 174u, 1u) &&
              memcmp(&timeline, &accepted, sizeof(timeline)) == 0,
          "non-monotonic update changed timeline");
    CHECK(!k3_prefill_ring_timeline_update(&timeline, 180u, 3u) &&
              memcmp(&timeline, &accepted, sizeof(timeline)) == 0,
          "out-of-range depth changed timeline");
    CHECK(!k3_prefill_ring_timeline_begin(NULL, 0u, 0u) &&
              !k3_prefill_ring_timeline_finish(NULL, 0u),
          "invalid timeline API accepted null");

    printf("K3 prefill ring timeline: PASS\n");
    return 0;
}
