// Include the actual recovery function; dead-section elimination removes the
// HTTP entry point. Mock the worker only; use the real slot state machine.
#define main mimo26_unused_server_main
#include "../mimo26_server.cu"
#undef main
#include <cassert>
static bool healthy=false;
static unsigned checked=0, cold_resets=0;
extern "C" mimo26_gpu_worker_status mimo26_gpu_worker_reset_context(mimo26_gpu_worker*,char*e,size_t n) {
    ++checked;
    if (!healthy) { snprintf(e,n,"injected unhealthy worker"); return MIMO26_GPU_WORKER_DECODE_FAILED; }
    return MIMO26_GPU_WORKER_OK;
}
extern "C" void mimo26_gpu_worker_reset(mimo26_gpu_worker*) { ++cold_resets; }
int main() {
    server_runtime r{}; mimo26_slot_init(&r.slot);
    assert(attempt_recovery(&r)); assert(!checked && !cold_resets);
    assert(mimo26_slot_admit(&r.slot,0,0,1)==MIMO26_SLOT_ADMIT_OK);
    mimo26_slot_fault(&r.slot);
    for(unsigned i=0;i<MAX_RECOVERY_ATTEMPTS;++i) {
        assert(!attempt_recovery(&r));
        assert(r.slot.phase==MIMO26_SLOT_QUARANTINED && !cold_resets);
        assert(mimo26_slot_admit(&r.slot,0,0,1)==MIMO26_SLOT_REJECT_QUARANTINED);
    }
    assert(checked==MAX_RECOVERY_ATTEMPTS);
    healthy=true; assert(!attempt_recovery(&r)); // retry ceiling stays closed
    assert(checked==MAX_RECOVERY_ATTEMPTS && !cold_resets);
    server_runtime clean{}; mimo26_slot_init(&clean.slot);
    assert(mimo26_slot_admit(&clean.slot,0,0,1)==MIMO26_SLOT_ADMIT_OK);
    mimo26_slot_fault(&clean.slot);
    assert(attempt_recovery(&clean));
    assert(cold_resets==1 && mimo26_slot_ready(&clean.slot));
    puts("PASS actual server recovery: unhealthy stays quarantined, cold reset not called, retry ceiling enforced, healthy path recovers");
}
