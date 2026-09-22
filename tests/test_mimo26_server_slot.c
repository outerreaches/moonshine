/*
 * The serving behaviours that are hardest to test against a live server:
 * a second request arriving mid-generation, a client vanishing, a deadline
 * expiring, a worker faulting and being restarted, a shutdown racing work in
 * flight.
 *
 * All of it is pure logic with the clock passed in, so every case is exact
 * and none of it needs a GPU. That is the point of keeping the state machine
 * separate from the socket handling: against a real server most of these are
 * timing-dependent and would be tested by hope.
 */
#include "mimo26_server_slot.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void ok(const char *what, int passed, const char *detail)
{
    printf("  %-4s %-54s %s\n", passed ? "ok" : "FAIL", what,
           detail ? detail : "");
    if (!passed) {
        failures++;
    }
}

int main(void)
{
    mimo26_slot slot;

    /* One slot. A second caller is refused, not queued behind a deadline it
     * cannot see. */
    mimo26_slot_init(&slot);
    ok("a fresh slot is ready", mimo26_slot_ready(&slot), NULL);
    ok("first request is admitted",
       mimo26_slot_admit(&slot, 0.0, 30.0, 64u) == MIMO26_SLOT_ADMIT_OK,
       NULL);
    ok("second request is refused as busy",
       mimo26_slot_admit(&slot, 0.1, 30.0, 64u) == MIMO26_SLOT_REJECT_BUSY,
       NULL);
    ok("a busy slot is not ready", !mimo26_slot_ready(&slot), NULL);
    mimo26_slot_finish(&slot);
    ok("the slot is reusable after finishing", mimo26_slot_ready(&slot),
       NULL);

    /* max_tokens stops with finish_reason "length" and produces exactly the
     * number asked for -- an off-by-one here is a billing and a correctness
     * bug at once. */
    mimo26_slot_init(&slot);
    mimo26_slot_admit(&slot, 0.0, 0.0, 3u);
    int produced = 0;
    mimo26_slot_step step;
    while ((step = mimo26_slot_step_check(&slot, 0.0)) ==
           MIMO26_SLOT_CONTINUE) {
        produced++;
        if (produced > 10) {
            break;   /* guard against a non-terminating loop */
        }
    }
    char detail[96];
    snprintf(detail, sizeof detail, "produced %d, reason %s", produced,
             mimo26_slot_finish_reason(step));
    ok("max_tokens 3 yields exactly 3 tokens then length",
       produced == 3 && step == MIMO26_SLOT_STOP_LENGTH, detail);
    mimo26_slot_finish(&slot);

    /* Cancellation is observed between tokens, and is safe to call at any
     * moment including when nothing is running. */
    mimo26_slot_init(&slot);
    mimo26_slot_admit(&slot, 0.0, 0.0, 100u);
    ok("two tokens before cancelling",
       mimo26_slot_step_check(&slot, 0.0) == MIMO26_SLOT_CONTINUE &&
           mimo26_slot_step_check(&slot, 0.0) == MIMO26_SLOT_CONTINUE,
       NULL);
    mimo26_slot_cancel(&slot);
    ok("the phase reflects the pending cancel",
       slot.phase == MIMO26_SLOT_CANCELLING, NULL);
    ok("the next check stops with cancelled",
       mimo26_slot_step_check(&slot, 0.0) == MIMO26_SLOT_STOP_CANCELLED,
       NULL);
    ok("a cancelled slot still reports busy until it unwinds",
       mimo26_slot_busy(&slot), NULL);
    mimo26_slot_finish(&slot);
    ok("finishing a cancelled request frees the slot",
       mimo26_slot_ready(&slot) && slot.cancelled == 1u, NULL);
    mimo26_slot_cancel(&slot);
    ok("cancelling an idle slot is harmless",
       slot.phase == MIMO26_SLOT_IDLE || slot.phase == MIMO26_SLOT_CANCELLING,
       NULL);

    /* Deadlines fire on elapsed wall clock, not on token count. */
    mimo26_slot_init(&slot);
    mimo26_slot_admit(&slot, 100.0, 5.0, 0u);
    ok("inside the deadline the loop continues",
       mimo26_slot_step_check(&slot, 104.9) == MIMO26_SLOT_CONTINUE, NULL);
    ok("at the deadline the loop stops",
       mimo26_slot_step_check(&slot, 105.0) == MIMO26_SLOT_STOP_DEADLINE,
       NULL);
    mimo26_slot_finish(&slot);
    mimo26_slot_init(&slot);
    mimo26_slot_admit(&slot, 0.0, 0.0, 0u);
    ok("a zero deadline never fires",
       mimo26_slot_step_check(&slot, 1e9) == MIMO26_SLOT_CONTINUE, NULL);
    mimo26_slot_finish(&slot);

    /*
     * A fault quarantines the worker. Serving the next request from a KV
     * history that may be half-written is exactly the "publish partially
     * valid state" the plan forbids, so admission stops until a supervised
     * reset says otherwise.
     */
    mimo26_slot_init(&slot);
    mimo26_slot_admit(&slot, 0.0, 0.0, 10u);
    mimo26_slot_fault(&slot);
    ok("a fault quarantines rather than idling",
       slot.phase == MIMO26_SLOT_QUARANTINED && slot.faults == 1u, NULL);
    ok("a quarantined slot refuses work, distinguishably from busy",
       mimo26_slot_admit(&slot, 1.0, 0.0, 10u) ==
           MIMO26_SLOT_REJECT_QUARANTINED,
       NULL);
    ok("a quarantined slot is not ready", !mimo26_slot_ready(&slot), NULL);
    ok("recovery clears the quarantine", mimo26_slot_recover(&slot), NULL);
    ok("a second recovery is refused, so it cannot mask a live request",
       !mimo26_slot_recover(&slot), NULL);
    ok("work is admitted again after recovery",
       mimo26_slot_admit(&slot, 2.0, 0.0, 10u) == MIMO26_SLOT_ADMIT_OK,
       NULL);
    mimo26_slot_finish(&slot);

    /* Graceful shutdown: refuse new work, finish what is in flight. */
    mimo26_slot_init(&slot);
    ok("draining an idle slot completes at once",
       mimo26_slot_drain(&slot) && slot.phase == MIMO26_SLOT_STOPPED, NULL);
    ok("a stopped slot refuses work as draining",
       mimo26_slot_admit(&slot, 0.0, 0.0, 1u) ==
           MIMO26_SLOT_REJECT_DRAINING,
       NULL);

    mimo26_slot_init(&slot);
    mimo26_slot_admit(&slot, 0.0, 0.0, 10u);
    ok("draining a busy slot must wait", !mimo26_slot_drain(&slot), NULL);
    ok("a draining slot refuses new work",
       mimo26_slot_admit(&slot, 0.1, 0.0, 10u) ==
           MIMO26_SLOT_REJECT_DRAINING,
       NULL);
    ok("the in-flight request may still generate",
       mimo26_slot_step_check(&slot, 0.2) == MIMO26_SLOT_CONTINUE, NULL);
    mimo26_slot_finish(&slot);
    ok("finishing during a drain reaches stopped, not idle",
       slot.phase == MIMO26_SLOT_STOPPED, NULL);

    /* A fault during a drain keeps the quarantine rather than letting the
     * shutdown paper over it. */
    mimo26_slot_init(&slot);
    mimo26_slot_admit(&slot, 0.0, 0.0, 10u);
    mimo26_slot_drain(&slot);
    mimo26_slot_fault(&slot);
    ok("a fault while draining still quarantines",
       slot.phase == MIMO26_SLOT_QUARANTINED, NULL);

    /* Counters back the health endpoint, so they have to be right. */
    mimo26_slot_init(&slot);
    mimo26_slot_admit(&slot, 0.0, 0.0, 10u);
    mimo26_slot_admit(&slot, 0.0, 0.0, 10u);
    mimo26_slot_admit(&slot, 0.0, 0.0, 10u);
    mimo26_slot_cancel(&slot);
    mimo26_slot_finish(&slot);
    snprintf(detail, sizeof detail,
             "admitted %llu, busy %llu, cancelled %llu",
             (unsigned long long)slot.admitted,
             (unsigned long long)slot.rejected_busy,
             (unsigned long long)slot.cancelled);
    ok("counters track admissions, rejections and cancellations",
       slot.admitted == 1u && slot.rejected_busy == 2u &&
           slot.cancelled == 1u,
       detail);

    printf("test_mimo26_server_slot: %s\n", failures == 0 ? "ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
