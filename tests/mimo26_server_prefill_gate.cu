// CPU-only test of the actual HTTP prefill callback with real sockets.
#include <initializer_list>
#define main mimo26_unused_server_main
#include "../mimo26_server.cu"
#undef main
#include <cassert>
int main() {
    server_runtime runtime{};runtime.listener=-1;
    mimo26_slot_init(&runtime.slot);
    assert(mimo26_slot_admit(&runtime.slot,0,0,8)==MIMO26_SLOT_ADMIT_OK);
    int peer[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,peer)==0);
#ifdef BEFORE_DISCONNECT_FIX
    void*context=&runtime;
#else
    prefill_request_context request{&runtime,peer[0]};
    void*context=&request;
#endif
    assert(prefill_progress(context,64,257));
    assert(runtime.slot.produced==0&&!runtime.slot.cancel_requested);
    close(peer[1]);assert(peer_disconnected(peer[0]));
    assert(!prefill_progress(context,128,257));
    assert(runtime.slot.cancel_requested&&runtime.slot.produced==0);
    mimo26_slot_finish(&runtime.slot);
    assert(mimo26_slot_ready(&runtime.slot)&&runtime.slot.cancelled==1);
    close(peer[0]);
    puts("PASS actual prefill callback: live peer continues; closed peer cancels at a chunk boundary; no output-token count mutation");
}
