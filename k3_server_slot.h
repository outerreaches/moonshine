#ifndef K3_SERVER_SLOT_H
#define K3_SERVER_SLOT_H

#include <stdbool.h>

typedef enum {
    K3_SERVER_SLOT_IDLE = 0,
    K3_SERVER_SLOT_INFERENCE = 1,
    K3_SERVER_SLOT_CHECKPOINT_EXPORT = 2,
    K3_SERVER_SLOT_STOPPING = 3,
} k3_server_slot_phase;

typedef enum {
    K3_SERVER_SLOT_ADMIT_DIRECT = 0,
    K3_SERVER_SLOT_ADMIT_QUEUED = 1,
    K3_SERVER_SLOT_REJECT_BUSY = 2,
    K3_SERVER_SLOT_REJECT_STOPPING = 3,
} k3_server_slot_admission;

typedef struct {
    k3_server_slot_phase phase;
    bool queued;
} k3_server_slot_state;

void k3_server_slot_init(k3_server_slot_state *state);
k3_server_slot_admission k3_server_slot_admit(
    k3_server_slot_state *state);
bool k3_server_slot_begin_checkpoint_export(
    k3_server_slot_state *state);
bool k3_server_slot_finish_current(
    k3_server_slot_state *state, bool stopping);
void k3_server_slot_stop(k3_server_slot_state *state);
bool k3_server_slot_busy(const k3_server_slot_state *state);
const char *k3_server_slot_phase_name(k3_server_slot_phase phase);

#endif
