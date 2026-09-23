// Test-only observability and injected I/O failure; never linked by Makefile.
#include "mimo26_gpu_worker.h"
#include "k3_io_uring.h"
#include "mimo26_server_slot.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
static unsigned request_id=0,decode_id=0;
static bool usable_logits=false;
extern "C" mimo26_slot_admission __real_mimo26_slot_admit(mimo26_slot*,double,double,uint32_t);
extern "C" mimo26_slot_admission __wrap_mimo26_slot_admit(mimo26_slot*s,double now,double duration,uint32_t max_tokens) {
    static bool first=true;
    if(first && getenv("MIMO26_TEST_FIRST_DEADLINE_SECONDS")) {
        duration=strtod(getenv("MIMO26_TEST_FIRST_DEADLINE_SECONDS"),nullptr);
        assert(duration>0 && duration<=600);
        fprintf(stderr,"TEST_FIRST_DEADLINE seconds=%.6f\n",duration);
    }
    first=false;
    return __real_mimo26_slot_admit(s,now,duration,max_tokens);
}
static void capture(const char *label,const void*data,size_t bytes) {
    const char*prefix=getenv("MIMO26_TEST_CAPTURE");assert(prefix);
    std::string path=std::string(prefix)+"/"+std::to_string(request_id)+"-"+label+".bin";
    FILE*f=fopen(path.c_str(),"wbx");assert(f);
    assert(fwrite(data,1,bytes,f)==bytes&&!fclose(f));
}
struct Progress {mimo26_gpu_prefill_progress callback;void*context;bool stopped;};
static bool progress(void*opaque,size_t done,size_t total) {
    Progress*p=(Progress*)opaque;
    bool keep=!p->callback||p->callback(p->context,done,total);
    if(!keep)p->stopped=true;
    fprintf(stderr,"TEST_PROGRESS request=%u done=%zu total=%zu keep=%u\n",request_id,done,total,unsigned(keep));
    return keep;
}
extern "C" mimo26_gpu_worker_status __real_mimo26_gpu_worker_prefill(
    mimo26_gpu_worker*,const uint32_t*,size_t,float*,mimo26_gpu_prefill_progress,void*,char*,size_t);
extern "C" mimo26_gpu_worker_status __wrap_mimo26_gpu_worker_prefill(
    mimo26_gpu_worker*w,const uint32_t*t,size_t n,float*l,mimo26_gpu_prefill_progress cb,void*c,char*e,size_t z) {
    ++request_id;decode_id=0;usable_logits=false;capture("tokens",t,n*sizeof *t);
    fprintf(stderr,"TEST_PREFILL_BEGIN request=%u count=%zu\n",request_id,n);
    Progress p{cb,c,false};
    auto status=__real_mimo26_gpu_worker_prefill(w,t,n,l,progress,&p,e,z);
    usable_logits=status==MIMO26_GPU_WORKER_OK&&!p.stopped;
    if(usable_logits)capture("prefill",l,152576*sizeof(float));
    fprintf(stderr,"TEST_PREFILL_END request=%u status=%u position=%llu count=%zu stopped=%u\n",
        request_id,unsigned(status),(unsigned long long)mimo26_gpu_worker_position(w),n,unsigned(p.stopped));
    return status;
}
extern "C" uint32_t __real_mimo26_gpu_worker_argmax(const float*);
extern "C" uint32_t __wrap_mimo26_gpu_worker_argmax(const float*l) {
    assert(usable_logits&&"argmax after stopped/failed prefill reads undefined logits");
    return __real_mimo26_gpu_worker_argmax(l);
}
extern "C" mimo26_gpu_worker_status __real_mimo26_gpu_worker_decode(mimo26_gpu_worker*,uint32_t,float*,char*,size_t);
extern "C" mimo26_gpu_worker_status __wrap_mimo26_gpu_worker_decode(mimo26_gpu_worker*w,uint32_t t,float*l,char*e,size_t z) {
    assert(usable_logits);
    if(getenv("MIMO26_TEST_CONTROL_TRACE"))fprintf(stderr,"TEST_DECODE_BEGIN request=%u step=%u\n",request_id,decode_id);
    auto status=__real_mimo26_gpu_worker_decode(w,t,l,e,z);
    if(status==MIMO26_GPU_WORKER_OK)capture(("decode"+std::to_string(decode_id++)).c_str(),l,152576*sizeof(float));
    else usable_logits=false;
    return status;
}
extern "C" bool __real_k3_io_uring_submit(k3_io_uring*,const k3_io_request*,uint16_t,char*,size_t);
extern "C" bool __wrap_k3_io_uring_submit(k3_io_uring*r,const k3_io_request*q,uint16_t n,char*e,size_t z) {
    static bool fired=false;
    bool ok=__real_k3_io_uring_submit(r,q,n,e,z);
    if(ok&&!fired&&getenv("MIMO26_TEST_HTTP_SUBMIT_FAILURE")) {
        fired=true;fprintf(stderr,"TEST_HTTP_FAULT pending=%u\n",k3_io_uring_outstanding(r));
        snprintf(e,z,"test-only failure after real submitted batch");return false;
    }
    return ok;
}
