// Frozen mixed-domain workload; not a language-quality evaluation.
#include "mimo26_gpu_worker.h"
#include "mimo26_tokenizer.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static const char *prompts[]={
    "Review this Python function and explain the off-by-one error. def window_sum(values, width):\n"
    "    total = 0\n    result = []\n    for i, value in enumerate(values):\n"
    "        total += value\n        if i > width: total -= values[i-width]\n"
    "        if i >= width-1: result.append(total)\n    return result\n"
    "Give tests for empty input, a window of one, repeated values, and a window wider than the list. "
    "Preserve the input list. Compare the algorithm with summing each slice independently. ",
    "A warehouse has three shelves holding 17, 24, and 39 boxes. Each box contains six sealed packets. "
    "A shipment removes twelve boxes from the second shelf and nine from the third. Then fifteen boxes arrive. "
    "Explain how to calculate the remaining packet count, showing a conservation equation before arithmetic. "
    "Distinguish boxes from packets and state which details do not affect the total. "
    "Now generalize to n shelves with counts c_i and separate arrival and departure vectors. ",
    "Translate and compare these notices while preserving dates and uncertainty. "
    "English: The museum may close early on Friday if the storm arrives. "
    "Français : La bibliothèque ouvrira à neuf heures samedi, sauf en cas de panne. "
    "Español: El tren podría llegar con veinte minutos de retraso. "
    "日本語：明日の会議は午後三時に始まります。資料を確認してください。 "
    "中文：请核对时间和地点，不要把可能发生的事情写成确定的事实。 "
    "Separate confirmed plans from conditional statements. "
};
static uint64_t read_bytes(){
    FILE*f=fopen("/proc/self/io","r");assert(f);
    char line[256];unsigned long long bytes=0;bool found=false;
    while(fgets(line,sizeof line,f))
        if(sscanf(line,"read_bytes: %llu",&bytes)==1){found=true;break;}
    assert(!fclose(f)&&found);return bytes;
}
static unsigned number(const char*s,unsigned low,unsigned high){
    char*end=nullptr;unsigned long value=strtoul(s,&end,10);
    assert(*s&&!*end&&value>=low&&value<=high);return unsigned(value);
}
int main(int argc,char**argv){
    assert(argc==3||argc==7);setvbuf(stdout,nullptr,_IOLBF,0);
    const unsigned tokens=argc==7?number(argv[3],1,8192):128;
    const unsigned chunk=argc==7?number(argv[4],1,128):32;
    const unsigned first=argc==7?number(argv[5],0,3):0;
    const unsigned cases=argc==7?number(argv[6],1,4-first):3;
    char error[512]{};mimo26_tokenizer*t=nullptr;
    assert(mimo26_tokenizer_create(&t,argv[1],error,sizeof error));
    mimo26_gpu_worker_config c;mimo26_gpu_worker_config_defaults(&c);
    c.global_kv_capacity=tokens+2>256?tokens+2:256;
    c.expert_slots_per_layer=48;c.prefill_chunk=chunk;
#ifdef LOOKAHEAD_CANDIDATE
    c.expert_lookahead=true;
#endif
    mimo26_gpu_worker*w=nullptr;
    assert(mimo26_gpu_worker_create(&w,argv[1],&c,error,sizeof error)==MIMO26_GPU_WORKER_OK);
    std::vector<float>logits(152576);
    for(unsigned p=first;p<first+cases;++p){
        mimo26_gpu_worker_reset(w);
        // Case 3 combines domains. Repetition is intentional systems-test input,
        // not a representative natural-chat workload. Preserve the original
        // two-copy prefix for cases 0..2 and freeze the exact consumed token IDs.
        std::string unit=p==3?std::string(prompts[0])+"\n"+prompts[1]+"\n"+prompts[2]:prompts[p];
        std::string text=unit+unit;mimo26_token_buffer ids{};
        for(;;){
            assert(mimo26_tokenizer_encode(t,text.c_str(),false,&ids,error,sizeof error));
            if(ids.count>=tokens)break;
            mimo26_token_buffer_free(&ids);text+=unit;
        }
        std::string prefix=std::string(argv[2])+"-"+std::to_string(p);
        FILE*f=fopen((prefix+"-tokens.bin").c_str(),"wbx");assert(f);
        assert(fwrite(ids.ids,4,tokens,f)==tokens&&!fclose(f));
        mimo26_gpu_worker_stats before{},after{};mimo26_gpu_worker_get_stats(w,&before);
        fprintf(stderr,"CASE_BEGIN %u\n",p);
        const uint64_t reads_before=read_bytes();
        auto start=std::chrono::steady_clock::now();
        assert(mimo26_gpu_worker_prefill(w,ids.ids,tokens,logits.data(),nullptr,nullptr,error,sizeof error)==MIMO26_GPU_WORKER_OK);
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const uint64_t reads_after=read_bytes();assert(reads_after>=reads_before);
        mimo26_gpu_worker_get_stats(w,&after);
        f=fopen((prefix+"-prefill.bin").c_str(),"wbx");assert(f);
        assert(fwrite(logits.data(),4,logits.size(),f)==logits.size()&&!fclose(f));
        printf("{\"case\":%u,\"tokens\":%u,\"chunk\":%u,\"seconds\":%.9f,\"process_read_bytes\":%llu,\"accesses\":%llu,\"hits\":%llu,\"uploads\":%llu}\n",p,tokens,chunk,seconds,
            (unsigned long long)(reads_after-reads_before),
            (unsigned long long)(after.expert_accesses-before.expert_accesses),
            (unsigned long long)(after.expert_hits-before.expert_hits),
            (unsigned long long)(after.expert_uploads-before.expert_uploads));
        fprintf(stderr,"CASE_END %u\n",p);
        for(unsigned step=0;step<2;++step){
            uint32_t token=mimo26_gpu_worker_argmax(logits.data());
            assert(mimo26_gpu_worker_decode(w,token,logits.data(),error,sizeof error)==MIMO26_GPU_WORKER_OK);
            f=fopen((prefix+"-decode"+std::to_string(step)+".bin").c_str(),"wbx");assert(f);
            assert(fwrite(logits.data(),4,logits.size(),f)==logits.size()&&!fclose(f));
        }
        mimo26_token_buffer_free(&ids);
    }
    mimo26_gpu_worker_destroy(w);mimo26_tokenizer_destroy(t);
}
