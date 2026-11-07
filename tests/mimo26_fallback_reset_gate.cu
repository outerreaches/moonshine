#include "mimo26_gpu_worker.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
static bool stop(void *context,size_t done,size_t) {
    char error[256];
    assert(done==1);
    assert(mimo26_gpu_worker_reset_context((mimo26_gpu_worker*)context,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    return false;
}
int main(int argc,char**argv) {
    assert(argc==2); setvbuf(stdout,nullptr,_IOLBF,0);
    mimo26_gpu_worker_config c; mimo26_gpu_worker_config_defaults(&c);
    c.global_kv_capacity=16; c.expert_slots_per_layer=16; c.prefill_chunk=0;
    mimo26_gpu_worker*w=nullptr; char error[512]{};
    assert(mimo26_gpu_worker_create(&w,argv[1],&c,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    uint32_t tokens[]={785,3974}; std::vector<float>a(152576),b(152576);
    assert(mimo26_gpu_worker_prefill(w,tokens,2,a.data(),nullptr,nullptr,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    assert(mimo26_gpu_worker_reset_context(w,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    assert(mimo26_gpu_worker_prefill(w,tokens,2,b.data(),stop,w,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    assert(mimo26_gpu_worker_position(w)==1);
    assert(mimo26_gpu_worker_reset_context(w,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    assert(mimo26_gpu_worker_prefill(w,tokens,2,b.data(),nullptr,nullptr,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    assert(!memcmp(a.data(),b.data(),a.size()*sizeof(float)));
    mimo26_gpu_worker_destroy(w);
    puts("PASS fallback callback reset refused; committed cancellation reset succeeds; full logits exact");
}
