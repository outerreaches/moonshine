#include "mimo26_gpu_worker.h"
#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
static int inject = 0;
extern "C" hipError_t __real_hipMemcpy(void *, const void *, size_t, hipMemcpyKind);
extern "C" hipError_t __wrap_hipMemcpy(void *dst, const void *src, size_t n, hipMemcpyKind kind) {
    auto result = __real_hipMemcpy(dst, src, n, kind);
    if (inject && n == 13369344 && kind == hipMemcpyHostToDevice && --inject == 0) {
        fprintf(stderr, "INJECT after real expert copy\n");
        return hipErrorUnknown;
    }
    return result;
}
static bool cancel(void *context, size_t done, size_t) {
#ifndef BASELINE
    char error[256];
    assert(mimo26_gpu_worker_reset_context((mimo26_gpu_worker *)context, error, sizeof error)
           == MIMO26_GPU_WORKER_DECODE_FAILED);
#endif
    assert(done == 32);
    return false;
}
int main(int argc, char **argv) {
    assert(argc == 3); setvbuf(stdout, nullptr, _IOLBF, 0);
    char error[512]{};
    mimo26_gpu_worker_config config; mimo26_gpu_worker_config_defaults(&config);
    config.global_kv_capacity = 128; config.expert_slots_per_layer = 16;
    mimo26_gpu_worker *w = nullptr;
    assert(mimo26_gpu_worker_create(&w, argv[1], &config, error, sizeof error) == MIMO26_GPU_WORKER_OK);
    std::vector<uint32_t> a, b;
    const uint32_t aa[] = {785,3974,13876,38835,34208,13};
    const uint32_t bb[] = {40,1097,264,11190,13,576};
    for (unsigned i=0;i<64;++i) { a.push_back(aa[i%6]); b.push_back(bb[i%6]); }
    std::vector<float> logits(152576), refa, refb;
    auto run = [&](const std::vector<uint32_t>& ids, const char *label) {
        assert(mimo26_gpu_worker_prefill(w, ids.data(), ids.size(), logits.data(), nullptr, nullptr,
                                       error, sizeof error) == MIMO26_GPU_WORKER_OK);
        assert(mimo26_gpu_worker_position(w) == 64);
        // Also check decode after retained-reset prefill, not just last prefill output.
        assert(mimo26_gpu_worker_decode(w, 13, logits.data(), error, sizeof error) == MIMO26_GPU_WORKER_OK);
        FILE *f = fopen((std::string(argv[2])+"-"+label+".bin").c_str(), "wbx");
        assert(f && fwrite(logits.data(), sizeof(float), logits.size(), f)==logits.size());
        assert(fclose(f)==0); printf("PASS %s\n", label);
    };
    auto clean = [&]() {
#ifdef BASELINE
        mimo26_gpu_worker_reset(w);
#else
        mimo26_gpu_worker_stats before{}, after{};
        mimo26_gpu_worker_get_stats(w,&before);
        assert(mimo26_gpu_worker_reset_context(w,error,sizeof error)==MIMO26_GPU_WORKER_OK);
        mimo26_gpu_worker_get_stats(w,&after);
        assert(before.expert_accesses==after.expert_accesses && before.expert_hits==after.expert_hits &&
               before.expert_uploads==after.expert_uploads);
#endif
        assert(mimo26_gpu_worker_position(w)==0);
    };
    run(a,"cold-a"); refa=logits; mimo26_gpu_worker_reset(w);
    run(b,"cold-b"); refb=logits;
    clean(); run(a,"warm-a"); assert(!memcmp(refa.data(),logits.data(),logits.size()*4));
    clean(); run(b,"warm-b"); assert(!memcmp(refb.data(),logits.data(),logits.size()*4));
    clean(); run(a,"warm-a2"); assert(!memcmp(refa.data(),logits.data(),logits.size()*4));
    clean();
    assert(mimo26_gpu_worker_prefill(w,b.data(),b.size(),logits.data(),cancel,w,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    assert(mimo26_gpu_worker_position(w)==32);
    clean(); run(a,"cancel-a"); assert(!memcmp(refa.data(),logits.data(),logits.size()*4));
#ifndef BASELINE
    clean();
    assert(mimo26_gpu_worker_prefill(w,a.data(),0,logits.data(),nullptr,nullptr,error,sizeof error)==MIMO26_GPU_WORKER_INVALID_ARGUMENT);
    assert(mimo26_gpu_worker_decode(w,152576,logits.data(),error,sizeof error)==MIMO26_GPU_WORKER_INVALID_ARGUMENT);
    clean();
    mimo26_gpu_worker_reset(w); inject=2;
    assert(mimo26_gpu_worker_decode(w,a[0],logits.data(),error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    assert(inject==0);
    assert(mimo26_gpu_worker_reset_context(w,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    mimo26_gpu_worker_reset(w);
    assert(mimo26_gpu_worker_reset_context(w,error,sizeof error)==MIMO26_GPU_WORKER_DECODE_FAILED);
    puts("PASS partial-copy fault sticky across cold reset; invalid inputs and reentrant reset gated");
#endif
    mimo26_gpu_worker_destroy(w);
    puts("PASS reset gate");
}
