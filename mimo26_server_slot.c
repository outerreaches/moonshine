#include "mimo26_server_slot.h"

#include <string.h>

void mimo26_slot_init(mimo26_slot *slot)
{
    if (slot == NULL) {
        return;
    }
    memset(slot, 0, sizeof *slot);
    slot->phase = MIMO26_SLOT_IDLE;
}

mimo26_slot_admission mimo26_slot_admit(mimo26_slot *slot, double now_seconds,
                                        double deadline_seconds,
                                        uint32_t max_tokens)
{
    if (slot == NULL) {
        return MIMO26_SLOT_REJECT_BUSY;
    }
    /*
     * Order matters. Quarantine and draining are conditions of the service,
     * busy is a condition of the moment, and a caller can act on the
     * difference: retry shortly for busy, but not for a worker that is
     * unhealthy or a process that is going away.
     */
    if (slot->phase == MIMO26_SLOT_QUARANTINED) {
        slot->rejected_quarantined++;
        return MIMO26_SLOT_REJECT_QUARANTINED;
    }
    if (slot->phase == MIMO26_SLOT_DRAINING ||
        slot->phase == MIMO26_SLOT_STOPPED) {
        slot->rejected_draining++;
        return MIMO26_SLOT_REJECT_DRAINING;
    }
    if (slot->phase != MIMO26_SLOT_IDLE) {
        slot->rejected_busy++;
        return MIMO26_SLOT_REJECT_BUSY;
    }

    slot->phase = MIMO26_SLOT_RUNNING;
    slot->request_id++;
    slot->started_seconds = now_seconds;
    slot->deadline_seconds = deadline_seconds;
    slot->max_tokens = max_tokens;
    slot->produced = 0u;
    slot->cancel_requested = false;
    slot->admitted++;
    return MIMO26_SLOT_ADMIT_OK;
}

void mimo26_slot_cancel(mimo26_slot *slot)
{
    if (slot == NULL) {
        return;
    }
    /*
     * Recorded as a flag rather than acted on here. A disconnect can arrive
     * at any instant, including while a decode step is mid-flight, and the
     * only safe place to unwind is between tokens.
     */
    slot->cancel_requested = true;
    if (slot->phase == MIMO26_SLOT_RUNNING) {
        slot->phase = MIMO26_SLOT_CANCELLING;
    }
}

mimo26_slot_step mimo26_slot_step_check(mimo26_slot *slot, double now_seconds)
{
    if (slot == NULL) {
        return MIMO26_SLOT_STOP_CANCELLED;
    }
    if (slot->cancel_requested) {
        return MIMO26_SLOT_STOP_CANCELLED;
    }
    if (slot->deadline_seconds > 0.0 &&
        now_seconds - slot->started_seconds >= slot->deadline_seconds) {
        return MIMO26_SLOT_STOP_DEADLINE;
    }
    if (slot->max_tokens > 0u && slot->produced >= slot->max_tokens) {
        return MIMO26_SLOT_STOP_LENGTH;
    }
    slot->produced++;
    return MIMO26_SLOT_CONTINUE;
}

void mimo26_slot_finish(mimo26_slot *slot)
{
    if (slot == NULL) {
        return;
    }
    if (slot->phase == MIMO26_SLOT_CANCELLING || slot->cancel_requested) {
        slot->cancelled++;
    }
    /* A drain that was waiting on this request is now complete. */
    if (slot->phase == MIMO26_SLOT_DRAINING) {
        slot->phase = MIMO26_SLOT_STOPPED;
    } else if (slot->phase != MIMO26_SLOT_STOPPED &&
               slot->phase != MIMO26_SLOT_QUARANTINED) {
        slot->phase = MIMO26_SLOT_IDLE;
    }
    slot->cancel_requested = false;
}

void mimo26_slot_fault(mimo26_slot *slot)
{
    if (slot == NULL) {
        return;
    }
    slot->faults++;
    slot->cancel_requested = false;
    /*
     * Quarantine even while draining. The process is going away, but a
     * shutdown that keeps serving from a worker whose KV history may be
     * half-written would publish partially valid state, which the plan
     * forbids for exactly this case.
     */
    slot->phase = MIMO26_SLOT_QUARANTINED;
}

bool mimo26_slot_recover(mimo26_slot *slot)
{
    if (slot == NULL || slot->phase != MIMO26_SLOT_QUARANTINED) {
        return false;
    }
    slot->phase = MIMO26_SLOT_IDLE;
    slot->recoveries++;
    return true;
}

bool mimo26_slot_drain(mimo26_slot *slot)
{
    if (slot == NULL) {
        return true;
    }
    if (slot->phase == MIMO26_SLOT_IDLE ||
        slot->phase == MIMO26_SLOT_STOPPED) {
        slot->phase = MIMO26_SLOT_STOPPED;
        return true;
    }
    if (slot->phase == MIMO26_SLOT_QUARANTINED) {
        /* Nothing is running, so there is nothing to wait for -- but the
         * quarantine stands rather than being cleared by a shutdown. */
        return true;
    }
    slot->phase = MIMO26_SLOT_DRAINING;
    return false;
}

void mimo26_slot_count_rejection(mimo26_slot *slot,
                                 mimo26_slot_admission reason)
{
    if (slot == NULL) {
        return;
    }
    switch (reason) {
    case MIMO26_SLOT_REJECT_BUSY:        slot->rejected_busy++; break;
    case MIMO26_SLOT_REJECT_QUARANTINED: slot->rejected_quarantined++; break;
    case MIMO26_SLOT_REJECT_DRAINING:    slot->rejected_draining++; break;
    case MIMO26_SLOT_ADMIT_OK:           break;
    }
}

bool mimo26_slot_busy(const mimo26_slot *slot)
{
    if (slot == NULL) {
        return true;
    }
    return slot->phase == MIMO26_SLOT_RUNNING ||
           slot->phase == MIMO26_SLOT_CANCELLING ||
           slot->phase == MIMO26_SLOT_DRAINING;
}

bool mimo26_slot_ready(const mimo26_slot *slot)
{
    return slot != NULL && slot->phase == MIMO26_SLOT_IDLE;
}

const char *mimo26_slot_phase_name(mimo26_slot_phase phase)
{
    switch (phase) {
    case MIMO26_SLOT_IDLE:        return "idle";
    case MIMO26_SLOT_RUNNING:     return "running";
    case MIMO26_SLOT_CANCELLING:  return "cancelling";
    case MIMO26_SLOT_QUARANTINED: return "quarantined";
    case MIMO26_SLOT_DRAINING:    return "draining";
    case MIMO26_SLOT_STOPPED:     return "stopped";
    }
    return "unknown";
}

const char *mimo26_slot_finish_reason(mimo26_slot_step step)
{
    switch (step) {
    case MIMO26_SLOT_CONTINUE:       return "stop";
    case MIMO26_SLOT_STOP_CANCELLED: return "cancelled";
    case MIMO26_SLOT_STOP_DEADLINE:  return "deadline";
    case MIMO26_SLOT_STOP_LENGTH:    return "length";
    }
    return "stop";
}
