#ifndef MIMO26_SERVER_SLOT_H
#define MIMO26_SERVER_SLOT_H

/*
 * Admission and lifecycle for the single MiMo execution slot.
 *
 * Separate from k3_server_slot rather than shared: MiMo needs cancellation,
 * deadlines and fault quarantine, which K3's slot does not model, and the GLM
 * lane's serving path must keep working unchanged.
 *
 * This file is deliberately pure logic with no I/O, no GPU and no clock of
 * its own -- time is passed in. That is what makes the behaviours a server
 * is hardest to test by hand testable at all: a request arriving while
 * another runs, a client vanishing mid-generation, a worker faulting and
 * then being restarted, a shutdown racing an in-flight request.
 *
 * The slot holds exactly one request at a time. That is not a placeholder
 * for a queue -- one GPU, one worker, and a KV cache with a single
 * conversation's history in it, so a second concurrent request has nowhere
 * to run. It is rejected rather than queued so a caller learns immediately
 * instead of blocking on a deadline it cannot see.
 */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIMO26_SLOT_IDLE = 0,
    MIMO26_SLOT_RUNNING,
    /* The client is gone or asked to stop; the decode loop will notice
     * between tokens and unwind. Still occupied until it does. */
    MIMO26_SLOT_CANCELLING,
    /* A decode failed. The worker's state is not trusted until a supervised
     * reset succeeds, so requests are refused rather than served from a
     * possibly corrupt KV history. */
    MIMO26_SLOT_QUARANTINED,
    /* Shutting down: no new work, finish what is in flight. */
    MIMO26_SLOT_DRAINING,
    MIMO26_SLOT_STOPPED
} mimo26_slot_phase;

typedef enum {
    MIMO26_SLOT_ADMIT_OK = 0,
    MIMO26_SLOT_REJECT_BUSY,        /* 503, another request holds the slot */
    MIMO26_SLOT_REJECT_QUARANTINED, /* 503, worker not healthy */
    MIMO26_SLOT_REJECT_DRAINING     /* 503, shutting down */
} mimo26_slot_admission;

typedef enum {
    MIMO26_SLOT_CONTINUE = 0,   /* keep generating */
    MIMO26_SLOT_STOP_CANCELLED, /* client went away or asked to stop */
    MIMO26_SLOT_STOP_DEADLINE,  /* wall-clock budget exhausted */
    MIMO26_SLOT_STOP_LENGTH,   /* max_tokens reached */
    MIMO26_SLOT_STOP_SHUTDOWN  /* server stopped at a committed boundary */
} mimo26_slot_step;

typedef struct {
    mimo26_slot_phase phase;
    /* Monotonic across the process; identifies the most recent admission.
     * Not cleared on finish -- it is a correlation id, and reusing it would
     * make two different requests share a log line and a response id. */
    uint64_t request_id;
    double   started_seconds;   /* caller's clock at admission */
    double   deadline_seconds;  /* 0 disables */
    uint32_t max_tokens;
    uint32_t produced;
    bool     cancel_requested;
    /* Counters for the health endpoint. */
    uint64_t admitted;
    uint64_t rejected_busy;
    uint64_t rejected_quarantined;
    uint64_t rejected_draining;
    uint64_t cancelled;
    uint64_t deadline_stops;
    uint64_t faults;
    uint64_t recoveries;
} mimo26_slot;

void mimo26_slot_init(mimo26_slot *slot);

/*
 * Try to take the slot. On success the caller owns it until exactly one of
 * mimo26_slot_finish or mimo26_slot_fault is called, and request_id
 * identifies this occupancy for logging.
 */
mimo26_slot_admission mimo26_slot_admit(mimo26_slot *slot, double now_seconds,
                                        double deadline_seconds,
                                        uint32_t max_tokens);

/*
 * Ask the running request to stop. Safe from another thread and safe when
 * the slot is idle -- a client can disconnect at any moment, including
 * after the last token and before the response is written.
 */
void mimo26_slot_cancel(mimo26_slot *slot);

/* Non-mutating cancellation/deadline check, also safe between prefill chunks.
 * Unlike step_check, this neither counts output tokens nor checks max_tokens. */
mimo26_slot_step mimo26_slot_control_check(const mimo26_slot *slot,
                                           double now_seconds);

/*
 * Called between tokens. Returns what the generation loop should do and
 * counts a token when it says to continue.
 *
 * Checked between tokens rather than mid-token on purpose: a decode step is
 * transactional across all 48 layers, so unwinding inside one would leave
 * the KV cache half-written. Waiting costs at most one token's latency.
 */
mimo26_slot_step mimo26_slot_step_check(mimo26_slot *slot,
                                        double now_seconds);

/* Release the slot after a normal or cancelled completion. */
void mimo26_slot_finish(mimo26_slot *slot);

/*
 * Release the slot after a worker failure and quarantine it. The caller must
 * reset the worker and call mimo26_slot_recover before anything is admitted
 * again.
 */
void mimo26_slot_fault(mimo26_slot *slot);

/* Clear quarantine after a supervised reset. Returns false if the slot was
 * not quarantined, so a spurious recovery cannot mask a live request. */
bool mimo26_slot_recover(mimo26_slot *slot);

/* Stop accepting work. Returns true when the slot is already idle, so the
 * caller knows whether it must wait for an in-flight request. */
bool mimo26_slot_drain(mimo26_slot *slot);

/*
 * Record a rejection the caller issued without going through admit. The
 * server refuses queued connections directly, and without this the health
 * endpoint under-reports exactly the pressure an operator is looking for.
 */
void mimo26_slot_count_rejection(mimo26_slot *slot,
                                 mimo26_slot_admission reason);

bool mimo26_slot_busy(const mimo26_slot *slot);
/* Ready means: healthy, accepting, and able to start work now. */
bool mimo26_slot_ready(const mimo26_slot *slot);
const char *mimo26_slot_phase_name(mimo26_slot_phase phase);
const char *mimo26_slot_finish_reason(mimo26_slot_step step);

#ifdef __cplusplus
}
#endif

#endif
