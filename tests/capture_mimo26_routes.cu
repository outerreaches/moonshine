// Fixed natural-text route screen. No generated text or quality claim.
#include "mimo26_gpu_worker.h"
#include "mimo26_tokenizer.h"
#include <cstdio>
#include <string>
#include <chrono>
#include <cstdlib>
#include <vector>
int main(int argc,char**argv){
    if(argc!=3&&argc!=4)return 2;setvbuf(stdout,nullptr,_IOLBF,0);
    char error[512]{};mimo26_tokenizer*tokenizer=nullptr;
    if(!mimo26_tokenizer_create(&tokenizer,argv[1],error,sizeof(error)))return 1;
    const char*text=
        "The town library keeps maps, letters, and records of local weather. "
        "Each morning a volunteer checks the catalogue and returns borrowed books to their shelves. "
        "On Tuesday the archivist found a notebook describing how the old bridge was built. "
        "Its drawings showed the river in winter, when ice pressed against the stone supports. "
        "Engineers compared those drawings with recent measurements before planning repairs. "
        "They chose to preserve the original stones while replacing the damaged wooden walkway. "
        "Meanwhile, a school class visited the archive to learn how evidence can change an explanation. "
        "The students recorded their questions, checked dates, and distinguished observations from guesses. "
        "After lunch they wrote a short report explaining which claims the documents actually supported. "
        "The teacher asked them to keep uncertain details visible instead of filling gaps with confident stories.";
    mimo26_token_buffer ids{};
    if(!mimo26_tokenizer_encode(tokenizer,text,false,&ids,error,sizeof(error))||ids.count<128)return 1;
    FILE*f=std::fopen((std::string(argv[2])+"-tokens.bin").c_str(),"wbx");
    if(!f||fwrite(ids.ids,4,128,f)!=128||fclose(f))return 1;
    mimo26_gpu_worker_config config;mimo26_gpu_worker_config_defaults(&config);
    config.global_kv_capacity=256;config.expert_slots_per_layer=48;config.prefill_chunk=32;
    if(argc==4){unsigned slots=unsigned(std::strtoul(argv[3],nullptr,10));if(slots!=48&&slots!=96&&slots!=144)return 2;config.expert_slots_per_layer=slots;}
    mimo26_gpu_worker*w=nullptr;
    if(mimo26_gpu_worker_create(&w,argv[1],&config,error,sizeof(error))!=MIMO26_GPU_WORKER_OK){fprintf(stderr,"%s\n",error);return 1;}
    std::vector<float>logits(152576);
    for(unsigned repeat=0;repeat<2;++repeat){
        if(repeat)mimo26_gpu_worker_reset(w);
        fprintf(stderr,"PASS_BEGIN %u\n",repeat);
        auto started=std::chrono::steady_clock::now();
        if(mimo26_gpu_worker_prefill(w,ids.ids,128,logits.data(),nullptr,nullptr,error,sizeof(error))!=MIMO26_GPU_WORKER_OK){fprintf(stderr,"%s\n",error);return 1;}
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        auto path=std::string(argv[2])+"-"+std::to_string(repeat)+"-logits.bin";
        f=fopen(path.c_str(),"wbx");if(!f||fwrite(logits.data(),4,logits.size(),f)!=logits.size()||fclose(f))return 1;
        mimo26_gpu_worker_stats stats{};mimo26_gpu_worker_get_stats(w,&stats);
        printf("{\"repeat\":%u,\"tokens\":128,\"slots\":%u,\"seconds\":%.9f,\"accesses\":%llu,\"hits\":%llu,\"uploads_cumulative\":%llu,\"argmax\":%u}\n",repeat,unsigned(config.expert_slots_per_layer),elapsed,
            (unsigned long long)stats.expert_accesses,(unsigned long long)stats.expert_hits,
            (unsigned long long)stats.expert_uploads,mimo26_gpu_worker_argmax(logits.data()));
    }
    mimo26_gpu_worker_destroy(w);mimo26_token_buffer_free(&ids);mimo26_tokenizer_destroy(tokenizer);return 0;
}
