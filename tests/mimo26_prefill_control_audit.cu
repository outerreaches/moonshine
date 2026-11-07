// Evidence of remaining controls absent from the current callback, not a safety pass.
#include <initializer_list>
#define main mimo26_unused_server_main
#include "../mimo26_server.cu"
#undef main
#include <cassert>

int main() {
    server_runtime runtime{}; runtime.listener=-1;
    int peers[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,peers));
    prefill_request_context request{&runtime,peers[0]};
    mimo26_slot_init(&runtime.slot);
    assert(mimo26_slot_admit(&runtime.slot,now_seconds(),0,8)==MIMO26_SLOT_ADMIT_OK);
    g_shutdown=1;
    assert(prefill_progress(&request,64,380));
    assert(!runtime.slot.cancel_requested && runtime.slot.produced==0);
    g_shutdown=0;mimo26_slot_finish(&runtime.slot);
    assert(mimo26_slot_admit(&runtime.slot,now_seconds()-10,1,8)==MIMO26_SLOT_ADMIT_OK);
    assert(now_seconds()>runtime.slot.deadline_seconds);
    assert(prefill_progress(&request,128,380));
    assert(!runtime.slot.cancel_requested && runtime.slot.produced==0);
    mimo26_slot_finish(&runtime.slot);close(peers[0]);close(peers[1]);
    puts("REPRODUCED: connected-peer prefill callback ignores shutdown flag and expired generation deadline. NOT a production safety pass.");
}
