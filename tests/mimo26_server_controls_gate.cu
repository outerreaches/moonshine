// Actual server callbacks and sockets, without a GPU. Historical audits stay frozen.
#include <initializer_list>
#define main mimo26_unused_server_main
#include "../mimo26_server.cu"
#undef main
#include <cassert>

int main() {
    server_runtime runtime{}; runtime.listener=-1;
    int peer[2]; assert(!socketpair(AF_UNIX,SOCK_STREAM,0,peer));
    mimo26_slot_init(&runtime.slot);
    auto admit = [&](double duration) {
        assert(mimo26_slot_admit(&runtime.slot,now_seconds(),duration,8)==MIMO26_SLOT_ADMIT_OK);
    };
    admit(0);
    prefill_request_context request{&runtime,peer[0],MIMO26_SLOT_CONTINUE};
    assert(prefill_progress(&request,64,380));
    assert(runtime.slot.produced==0);
    g_shutdown=1;
    assert(!prefill_progress(&request,128,380));
    assert(request.stop==MIMO26_SLOT_STOP_SHUTDOWN);
    assert(runtime.slot.phase==MIMO26_SLOT_DRAINING && !runtime.slot.cancel_requested);
    assert(runtime.slot.produced==0 && runtime.slot.cancelled==0 && runtime.slot.deadline_stops==0);
    mimo26_slot_finish(&runtime.slot);
    assert(runtime.slot.phase==MIMO26_SLOT_STOPPED && runtime.slot.cancelled==0);
    g_shutdown=0;
    mimo26_slot_init(&runtime.slot);admit(1);
    runtime.slot.started_seconds-=10;
    request={&runtime,peer[0],MIMO26_SLOT_CONTINUE};
    // A final-chunk stop also means no final logits may be consumed.
    assert(!prefill_progress(&request,64,64));
    assert(request.stop==MIMO26_SLOT_STOP_DEADLINE && runtime.slot.produced==0);
    assert(!runtime.slot.cancel_requested && runtime.slot.deadline_stops==0);
    runtime.slot.deadline_seconds=0; // Stop reason stays sticky after prefill.
    assert(!prefill_progress(&request,64,64) && request.stop==MIMO26_SLOT_STOP_DEADLINE);
    mimo26_slot_finish(&runtime.slot);
    assert(mimo26_slot_ready(&runtime.slot) && runtime.slot.cancelled==0);
    admit(0);request={&runtime,peer[0],MIMO26_SLOT_CONTINUE};
    assert(prefill_progress(&request,64,380));
    assert(mimo26_slot_step_check(&runtime.slot,now_seconds())==MIMO26_SLOT_CONTINUE);
    assert(runtime.slot.produced==1);
    assert(request_control(&runtime,peer[0])==MIMO26_SLOT_CONTINUE && runtime.slot.produced==1);
    g_shutdown=1;
    assert(request_control(&runtime,peer[0])==MIMO26_SLOT_STOP_SHUTDOWN && runtime.slot.produced==1);
    mimo26_slot_finish(&runtime.slot);g_shutdown=0;
    mimo26_slot_init(&runtime.slot);admit(0);
    close(peer[1]);request={&runtime,peer[0],MIMO26_SLOT_CONTINUE};
    assert(!prefill_progress(&request,64,380) && request.stop==MIMO26_SLOT_STOP_CANCELLED);
    mimo26_slot_finish(&runtime.slot);assert(runtime.slot.cancelled==1);
    close(peer[0]);
    // Deterministic time semantics without mutation at the exact deadline.
    mimo26_slot_init(&runtime.slot);
    assert(mimo26_slot_admit(&runtime.slot,100,5,1)==MIMO26_SLOT_ADMIT_OK);
    assert(mimo26_slot_control_check(&runtime.slot,104.999)==MIMO26_SLOT_CONTINUE);
    assert(mimo26_slot_control_check(&runtime.slot,105)==MIMO26_SLOT_STOP_DEADLINE);
    assert(runtime.slot.produced==0 && runtime.slot.deadline_stops==0);

    runtime.listener=socket(AF_INET,SOCK_STREAM|SOCK_NONBLOCK,0);assert(runtime.listener>=0);
    sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    assert(!bind(runtime.listener,(sockaddr*)&address,sizeof address));assert(!listen(runtime.listener,32));
    socklen_t size=sizeof address;assert(!getsockname(runtime.listener,(sockaddr*)&address,&size));
    int clients[16];
    for(int &fd:clients) {
        fd=socket(AF_INET,SOCK_STREAM,0);assert(fd>=0);
        assert(!connect(fd,(sockaddr*)&address,sizeof address));
    }
    double start=now_seconds();refuse_backlog(&runtime);
    assert(now_seconds()-start<.3 && runtime.slot.rejected_busy==8);
    g_shutdown=1;refuse_backlog(&runtime);assert(runtime.slot.rejected_busy==8);g_shutdown=0;
    refuse_backlog(&runtime);assert(runtime.slot.rejected_busy==16);
    for(int fd:clients) {
        char response[1024];ssize_t n=recv(fd,response,sizeof response-1,0);assert(n>0);response[n]=0;
        assert(strstr(response,"HTTP/1.1 503 ") && strstr(response,"slot_busy"));close(fd);
    }
    close(runtime.listener);
    puts("PASS actual prefill/decode controls, final-chunk sticky reason, no token-count mutation, deadline/cancel separation, bounded backlog");
}
