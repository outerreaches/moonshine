// Actual server request-reset policy with stubbed worker calls; no GPU.
#define main mimo26_unused_server_main
#include "../mimo26_server.cu"
#undef main
#include <cassert>
static unsigned cold_calls, retained_calls;
static bool refuse;
extern "C" void mimo26_gpu_worker_get_stats(const mimo26_gpu_worker *, mimo26_gpu_worker_stats *stats) {
    memset(stats,0,sizeof *stats);
    stats->tokens=stats->expert_accesses=stats->expert_hits=stats->expert_uploads=UINT64_MAX;
}
extern "C" uint64_t mimo26_gpu_worker_resident_bytes(const mimo26_gpu_worker *) { return UINT64_MAX; }
extern "C" void mimo26_gpu_worker_reset(mimo26_gpu_worker *) { ++cold_calls; }
extern "C" mimo26_gpu_worker_status mimo26_gpu_worker_reset_context(mimo26_gpu_worker *, char *error, size_t size) {
    ++retained_calls;
    if (refuse) { snprintf(error,size,"injected sticky fault"); return MIMO26_GPU_WORKER_DECODE_FAILED; }
    return MIMO26_GPU_WORKER_OK;
}
int main() {
    server_runtime runtime{}; mimo26_slot_init(&runtime.slot); char error[128]{};
    assert(mimo26_slot_admit(&runtime.slot,1,600,8)==MIMO26_SLOT_ADMIT_OK);
    assert(reset_request_worker(&runtime,error,sizeof error));
    assert(cold_calls==1 && retained_calls==0);
    runtime.retain_experts=true;
    assert(reset_request_worker(&runtime,error,sizeof error));
    assert(cold_calls==1 && retained_calls==1 && runtime.slot.faults==0);
    refuse=true;
    assert(!reset_request_worker(&runtime,error,sizeof error));
    assert(cold_calls==1 && retained_calls==2 && runtime.slot.faults==1);
    assert(runtime.slot.phase==MIMO26_SLOT_QUARANTINED);
    assert(mimo26_slot_admit(&runtime.slot,2,600,8)==MIMO26_SLOT_REJECT_QUARANTINED);
    assert(!attempt_recovery(&runtime));
    assert(cold_calls==1 && runtime.slot.phase==MIMO26_SLOT_QUARANTINED);
    // Largest counters still fit the actual health response buffer.
    runtime.served=UINT64_MAX;runtime.expert_slots_per_layer=128;runtime.prefill_chunk=128;
    runtime.context_capacity=UINT32_MAX;
    runtime.slot.admitted=runtime.slot.rejected_busy=runtime.slot.rejected_quarantined=UINT64_MAX;
    runtime.slot.cancelled=runtime.slot.deadline_stops=runtime.slot.faults=runtime.slot.recoveries=UINT64_MAX;
    int pair[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,pair));
    send_health(pair[0],&runtime);shutdown(pair[0],SHUT_WR);
    char response[4096];size_t used=0;
    for(;;){ssize_t n=read(pair[1],response+used,sizeof response-1-used);assert(n>=0);if(!n)break;used+=(size_t)n;}
    response[used]=0;
    const char *body=strstr(response,"\r\n\r\n");assert(body);body+=4;
    const char *length=strstr(response,"Content-Length:");assert(length);
    assert(strtoul(length+15,nullptr,10)==strlen(body) && strlen(body)<1024);
    assert(strstr(body,"\"retain_experts\":true") && strstr(body,"\"expert_slots\":128"));
    assert(strstr(body,"\"expert_uploads\":18446744073709551615"));
    assert(body[strlen(body)-1]=='}');close(pair[0]);close(pair[1]);
    puts("PASS cold default, retained opt-in, reset refusal quarantines without cold fallback");
}
