#include "k3_server_slot.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        return 1; \
    } \
} while (0)

int main(void) {
    k3_server_slot_state state;
    k3_server_slot_init(&state);
    CHECK(state.phase == K3_SERVER_SLOT_IDLE && !state.queued,
          "initial idle state");
    CHECK(!k3_server_slot_busy(&state), "idle is not busy");
    CHECK(strcmp(k3_server_slot_phase_name(state.phase), "idle") == 0,
          "idle phase name");
    CHECK(!k3_server_slot_begin_checkpoint_export(&state),
          "idle cannot begin checkpoint export");

    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_ADMIT_DIRECT,
          "idle request admitted directly");
    CHECK(state.phase == K3_SERVER_SLOT_INFERENCE &&
              k3_server_slot_busy(&state),
          "direct admission enters inference");
    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_REJECT_BUSY,
          "inference rejects contender");

    CHECK(k3_server_slot_begin_checkpoint_export(&state),
          "inference enters checkpoint export");
    CHECK(state.phase == K3_SERVER_SLOT_CHECKPOINT_EXPORT && !state.queued,
          "checkpoint export starts empty");
    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_ADMIT_QUEUED,
          "first export contender queues");
    CHECK(state.queued, "queue occupancy recorded");
    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_REJECT_BUSY,
          "second export contender rejected");
    CHECK(!k3_server_slot_begin_checkpoint_export(&state),
          "checkpoint export cannot reenter");

    CHECK(k3_server_slot_finish_current(&state, false),
          "queued request handed to worker");
    CHECK(state.phase == K3_SERVER_SLOT_INFERENCE && !state.queued,
          "handoff has no idle gap");
    CHECK(!k3_server_slot_finish_current(&state, false),
          "completed direct request returns idle");
    CHECK(state.phase == K3_SERVER_SLOT_IDLE,
          "idle after request without queued successor");

    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_ADMIT_DIRECT,
          "second direct admission");
    CHECK(k3_server_slot_begin_checkpoint_export(&state),
          "second checkpoint export");
    CHECK(!k3_server_slot_finish_current(&state, false),
          "empty checkpoint export returns idle");
    CHECK(state.phase == K3_SERVER_SLOT_IDLE,
          "empty export idle state");

    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_ADMIT_DIRECT,
          "shutdown fixture direct admission");
    CHECK(k3_server_slot_begin_checkpoint_export(&state),
          "shutdown fixture export");
    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_ADMIT_QUEUED,
          "shutdown fixture queue");
    CHECK(!k3_server_slot_finish_current(&state, true),
          "shutdown prevents queued handoff");
    CHECK(state.phase == K3_SERVER_SLOT_STOPPING && state.queued,
          "shutdown retains queued ownership for rejection");
    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_REJECT_STOPPING,
          "stopping rejects new request");
    CHECK(strcmp(k3_server_slot_phase_name(state.phase), "stopping") == 0,
          "stopping phase name");

    k3_server_slot_init(&state);
    CHECK(k3_server_slot_admit(&state) == K3_SERVER_SLOT_ADMIT_DIRECT,
          "explicit stop fixture admission");
    k3_server_slot_stop(&state);
    CHECK(state.phase == K3_SERVER_SLOT_STOPPING,
          "explicit stop transition");
    CHECK(k3_server_slot_busy(&state), "stopping remains busy");
    CHECK(strcmp(k3_server_slot_phase_name(
                     (k3_server_slot_phase)99), "unknown") == 0,
          "unknown phase name");

    printf("K3 server slot state machine: PASS\n");
    return 0;
}
