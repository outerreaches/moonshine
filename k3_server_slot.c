#include "k3_server_slot.h"

#include <stddef.h>

void k3_server_slot_init(k3_server_slot_state *state) {
    if (state == NULL) return;
    state->phase = K3_SERVER_SLOT_IDLE;
    state->queued = false;
}

k3_server_slot_admission k3_server_slot_admit(
        k3_server_slot_state *state) {
    if (state == NULL) return K3_SERVER_SLOT_REJECT_STOPPING;
    switch (state->phase) {
        case K3_SERVER_SLOT_IDLE:
            if (state->queued) return K3_SERVER_SLOT_REJECT_BUSY;
            state->phase = K3_SERVER_SLOT_INFERENCE;
            return K3_SERVER_SLOT_ADMIT_DIRECT;
        case K3_SERVER_SLOT_CHECKPOINT_EXPORT:
            if (state->queued) return K3_SERVER_SLOT_REJECT_BUSY;
            state->queued = true;
            return K3_SERVER_SLOT_ADMIT_QUEUED;
        case K3_SERVER_SLOT_INFERENCE:
            return K3_SERVER_SLOT_REJECT_BUSY;
        case K3_SERVER_SLOT_STOPPING:
            return K3_SERVER_SLOT_REJECT_STOPPING;
    }
    return K3_SERVER_SLOT_REJECT_STOPPING;
}

bool k3_server_slot_begin_checkpoint_export(
        k3_server_slot_state *state) {
    if (state == NULL ||
        state->phase != K3_SERVER_SLOT_INFERENCE ||
        state->queued) {
        return false;
    }
    state->phase = K3_SERVER_SLOT_CHECKPOINT_EXPORT;
    return true;
}

bool k3_server_slot_finish_current(
        k3_server_slot_state *state, bool stopping) {
    if (state == NULL) return false;
    if (stopping) {
        state->phase = K3_SERVER_SLOT_STOPPING;
        return false;
    }
    if (state->phase != K3_SERVER_SLOT_INFERENCE &&
        state->phase != K3_SERVER_SLOT_CHECKPOINT_EXPORT) {
        return false;
    }
    if (state->queued) {
        state->queued = false;
        state->phase = K3_SERVER_SLOT_INFERENCE;
        return true;
    }
    state->phase = K3_SERVER_SLOT_IDLE;
    return false;
}

void k3_server_slot_stop(k3_server_slot_state *state) {
    if (state == NULL) return;
    state->phase = K3_SERVER_SLOT_STOPPING;
}

bool k3_server_slot_busy(const k3_server_slot_state *state) {
    return state != NULL && state->phase != K3_SERVER_SLOT_IDLE;
}

const char *k3_server_slot_phase_name(k3_server_slot_phase phase) {
    switch (phase) {
        case K3_SERVER_SLOT_IDLE:
            return "idle";
        case K3_SERVER_SLOT_INFERENCE:
            return "inference";
        case K3_SERVER_SLOT_CHECKPOINT_EXPORT:
            return "checkpoint_export";
        case K3_SERVER_SLOT_STOPPING:
            return "stopping";
    }
    return "unknown";
}
