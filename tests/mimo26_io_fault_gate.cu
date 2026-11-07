// Real checkpoint reads; failure signals injected only in this linked harness.
#include "mimo26_gpu_worker.h"
#include "k3_io_uring.h"
#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
static const char *mode;
static bool armed=false, fired=false, destroyed=false;
static unsigned waits=0, copies=0, host_frees=0;
static k3_io_uring *seen=nullptr;
extern "C" bool __real_k3_io_uring_submit(k3_io_uring*,const k3_io_request*,uint16_t,char*,size_t);
extern "C" bool __real_k3_io_uring_wait(k3_io_uring*,k3_io_completion*,uint16_t,uint16_t*,char*,size_t);
extern "C" void __real_k3_io_uring_destroy(k3_io_uring*);
extern "C" hipError_t __real_hipMemcpy(void*,const void*,size_t,hipMemcpyKind);
extern "C" hipError_t __real_hipHostFree(void*);
extern "C" bool __wrap_k3_io_uring_submit(k3_io_uring*r,const k3_io_request*q,uint16_t n,char*e,size_t z) {
    seen=r;
    bool ok=__real_k3_io_uring_submit(r,q,n,e,z);
    if(armed && !strcmp(mode,"submit") && ok) {
        armed=false; fired=true;
        snprintf(e,z,"injected failure after real submitted batch");
        return false;
    }
    return ok;
}
extern "C" bool __wrap_k3_io_uring_wait(k3_io_uring*r,k3_io_completion*c,uint16_t n,uint16_t*got,char*e,size_t z) {
    if(armed && !strcmp(mode,"wait")) {
        armed=false; fired=true; *got=0;
        snprintf(e,z,"injected wait failure with real I/O pending"); return false;
    }
    bool partial=armed && !strcmp(mode,"short");
    bool ok=__real_k3_io_uring_wait(r,c,partial?1:n,got,e,z);
    if(partial && ok && *got && ++waits==2) {
        assert(copies>=1); c[0].result=0; armed=false; fired=true;
    }
    return ok;
}
extern "C" hipError_t __wrap_hipMemcpy(void*d,const void*s,size_t n,hipMemcpyKind k) {
    auto result=__real_hipMemcpy(d,s,n,k);
    if(armed && n==13369344 && k==hipMemcpyHostToDevice && result==hipSuccess) ++copies;
    return result;
}
extern "C" void __wrap_k3_io_uring_destroy(k3_io_uring*r) {
    printf("DESTROY pending=%u\n",unsigned(k3_io_uring_outstanding(r)));
    __real_k3_io_uring_destroy(r); destroyed=true; seen=nullptr;
}
extern "C" hipError_t __wrap_hipHostFree(void*p) {
    assert(destroyed); ++host_frees; return __real_hipHostFree(p);
}
int main(int argc,char**argv) {
    assert(argc==4); setvbuf(stdout,nullptr,_IOLBF,0); mode=argv[2];
    assert(!strcmp(mode,"submit")||!strcmp(mode,"wait")||!strcmp(mode,"short"));
    mimo26_gpu_worker_config config; mimo26_gpu_worker_config_defaults(&config);
    config.global_kv_capacity=64; config.expert_slots_per_layer=16;
#ifdef LOOKAHEAD_CANDIDATE
    config.expert_lookahead=true;
#endif
    mimo26_gpu_worker*w=nullptr; char error[512]{};
    assert(mimo26_gpu_worker_create(&w,argv[1],&config,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    assert(mimo26_gpu_worker_reset_context(w,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    std::vector<float>logits(152576); uint32_t token=785;
    assert(mimo26_gpu_worker_decode(w,token,logits.data(),error,sizeof error)==MIMO26_GPU_WORKER_OK);
    FILE*f=fopen(argv[3],"wbx"); assert(f);
    assert(fwrite(logits.data(),sizeof(float),logits.size(),f)==logits.size()); assert(!fclose(f));
    mimo26_gpu_worker_reset(w); armed=true;
    // Short read exercises the layer-major path; submit/wait exercise decode.
    bool use_prefill=!strcmp(mode,"short");
    uint32_t prompt[]={token,3974};size_t count=1;
#ifdef LOOKAHEAD_CANDIDATE
    use_prefill=true;count=2; // include a nonempty future route suffix
#endif
    auto status=use_prefill
        ? mimo26_gpu_worker_prefill(w,prompt,count,logits.data(),nullptr,nullptr,error,sizeof error)
        : mimo26_gpu_worker_decode(w,token,logits.data(),error,sizeof error);
    assert(status==MIMO26_GPU_WORKER_DECODE_FAILED && fired && seen);
    unsigned pending=k3_io_uring_outstanding(seen); assert(pending>0);
    assert(mimo26_gpu_worker_position(w)==0);
    assert(mimo26_gpu_worker_reset_context(w,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    mimo26_gpu_worker_reset(w);
    assert(k3_io_uring_outstanding(seen)==pending);
    assert(mimo26_gpu_worker_reset_context(w,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    printf("PASS %s fault, copies=%u, pending=%u, cold reset does not drain; retention refused\n",mode,copies,pending);
    mimo26_gpu_worker_destroy(w);
    assert(destroyed && host_frees==8);
    puts("PASS destroy returns before all 8 staging frees; process must not reuse faulted worker");
}
