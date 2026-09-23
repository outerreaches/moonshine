// Link-only route capture. No kernel, schedule, cache, or arithmetic changes.
#include "mimo26_rocm_layer.h"
#include <cstdio>
#include <vector>
extern "C" mimo26_rocm_layer_status __real_mimo26_rocm_layer_prefill(
    const mimo26_rocm_layer*,mimo26_rocm_layer_scratch*,void*,void*,void*,
    const void*,const void*,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t*,void*);
extern "C" mimo26_rocm_layer_status __wrap_mimo26_rocm_layer_prefill(
    const mimo26_rocm_layer*l,mimo26_rocm_layer_scratch*s,void*h,void*k,void*v,
    const void*c,const void*sn,uint64_t history,uint64_t first,uint64_t position,
    uint32_t count,uint32_t*routes,void*stream){
    std::vector<uint32_t> captured;
    if(l&&l->weights&&l->weights->is_moe&&!routes){captured.resize(size_t(count)*8);routes=captured.data();}
    auto status=__real_mimo26_rocm_layer_prefill(l,s,h,k,v,c,sn,history,first,position,count,routes,stream);
    if(status==MIMO26_ROCM_LAYER_OK&&l->weights->is_moe){
        for(unsigned t=0;t<count;++t){
            std::fprintf(stderr,"ROUTE %u %llu %u",l->weights->layer,
                (unsigned long long)position,t);
            for(unsigned j=0;j<8;++j)std::fprintf(stderr," %u",routes[size_t(t)*8+j]);
            std::fputc('\n',stderr);
        }
    }
    return status;
}
