#include "../k3_expert_cache.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    k3_expert_cache*c=NULL;char error[256];uint32_t future[32];
    for(unsigned i=0;i<32;++i)future[i]=UINT32_MAX;
    assert(k3_expert_cache_create(&c,2,8,error,sizeof error));
    unsigned payload[16];for(unsigned i=0;i<16;++i)payload[i]=999;
    unsigned seed=1234567;
    for(unsigned round=0;round<3000;++round){
        uint16_t ids[4];unsigned made=0;unsigned layer=round%2;
        while(made<4){
            seed=seed*1664525u+1013904223u;uint16_t id=(seed>>16)%32;bool duplicate=false;
            for(unsigned j=0;j<made;++j)if(ids[j]==id)duplicate=true;
            if(!duplicate)ids[made++]=id;
        }
        for(unsigned i=0;i<32;++i){seed=seed*1664525u+1013904223u;future[i]=(seed>>16)%9;}
        uint16_t before[8],after[8],n=0,m=0;
        assert(k3_expert_cache_snapshot_layer(c,layer,before,8,&n,error,sizeof error));
        k3_expert_cache_access a[4];
        assert(k3_expert_cache_plan_next_use(c,layer,ids,4,future,32,a,error,sizeof error));
        if(round%7==0){
            k3_expert_cache_abort(c,layer);
            assert(k3_expert_cache_snapshot_layer(c,layer,after,8,&m,error,sizeof error));
            assert(n==m&&!memcmp(before,after,n*sizeof(uint16_t)));
            continue;
        }
        for(unsigned i=0;i<4;++i){
            if(a[i].hit)assert(payload[a[i].source_slot]==ids[i]);
            if(a[i].admit){
                assert(a[i].destination_slot/8==layer);
                for(unsigned j=0;j<4;++j){
                    if(a[j].hit)assert(a[i].destination_slot!=a[j].source_slot);
                    if(j!=i&&a[j].admit)assert(a[i].destination_slot!=a[j].destination_slot);
                }
                payload[a[i].destination_slot]=ids[i];
            }
        }
        assert(k3_expert_cache_commit(c,layer,error,sizeof error));
        for(unsigned i=0;i<4;++i)assert(payload[a[i].hit?a[i].source_slot:a[i].destination_slot]==ids[i]);
    }
    uint16_t duplicate[]={1,1},bad[]={32};k3_expert_cache_access a[2];
    assert(!k3_expert_cache_plan_next_use(c,0,duplicate,2,future,32,a,error,sizeof error));
    assert(!k3_expert_cache_plan_next_use(c,0,bad,1,future,32,a,error,sizeof error));
    assert(k3_expert_cache_reset(c,error,sizeof error));
    // Exact eviction decision: preserve near-future oldest entry, evict next one.
    k3_expert_cache_destroy(c);assert(k3_expert_cache_create(&c,1,2,error,sizeof error));
    uint16_t initial[]={1,2},incoming[]={3};
    assert(k3_expert_cache_plan_next_use(c,0,initial,2,future,32,a,error,sizeof error));
    assert(k3_expert_cache_commit(c,0,error,sizeof error));
    future[1]=1;future[2]=9;
    assert(k3_expert_cache_plan_next_use(c,0,incoming,1,future,32,a,error,sizeof error));
    assert(a[0].destination_slot==1);
    assert(!k3_expert_cache_plan_next_use(c,0,incoming,1,future,32,a,error,sizeof error));
    k3_expert_cache_abort(c,0);k3_expert_cache_destroy(c);
    puts("PASS next-use selection, 3000 payload/slot checks, abort, pending plan and invalid IDs");
}
