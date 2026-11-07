// White-box state-machine checks; no HIP calls or model loading.
#include "../mimo26_gpu_worker.cu"
#include <cassert>
static bool pending_io = false;
extern "C" uint16_t __real_k3_io_uring_outstanding(const k3_io_uring *);
extern "C" uint16_t __wrap_k3_io_uring_outstanding(const k3_io_uring *ring) {
    return pending_io ? 1 : __real_k3_io_uring_outstanding(ring);
}
int main() {
    mimo26_gpu_worker w{};
    char error[256]{};
    assert(mimo26_gpu_worker_reset_context(nullptr,error,sizeof error)==MIMO26_GPU_WORKER_INVALID_ARGUMENT);
    assert(k3_expert_cache_create(&w.cache,1,8,error,sizeof error));
    assert(mimo26_kv_create(&w.kv,8,0)==MIMO26_KV_OK);
    w.position=3; w.resolved_count=2;
    assert(mimo26_gpu_worker_reset_context(&w,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    assert(w.position==0 && w.resolved_count==0);
    pending_io=true; w.position=3; w.resolved_count=2;
    assert(mimo26_gpu_worker_reset_context(&w,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    assert(w.position==3 && w.resolved_count==2);
    pending_io=false;
    assert(mimo26_gpu_worker_reset_context(&w,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    {
        retention_execution_guard g(&w);
        assert(mimo26_gpu_worker_reset_context(&w,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
        g.completed=true;
    }
    assert(!w.execution_active && !w.retention_faulted);
    assert(mimo26_kv_begin(w.kv,0)==MIMO26_KV_OK);
    assert(mimo26_gpu_worker_reset_context(&w,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    assert(mimo26_kv_in_transaction(w.kv));
    assert(mimo26_kv_abort(w.kv)==MIMO26_KV_OK);
    { retention_execution_guard g(&w); }
    assert(!w.execution_active && w.retention_faulted);
    mimo26_gpu_worker_reset(&w);
    { retention_execution_guard g(&w); g.completed=true; }
    assert(w.retention_faulted);
    assert(mimo26_gpu_worker_reset_context(&w,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    mimo26_kv_destroy(w.kv); k3_expert_cache_destroy(w.cache);
    puts("PASS healthy/busy/outstanding-I-O/open-KV/faulted/sticky-after-reset guard transitions");
}
