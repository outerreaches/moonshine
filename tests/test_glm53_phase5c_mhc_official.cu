#include "../glm53_architecture.h"
#include "../glm53_manifest.h"
#include "../glm53_mhc_ops.h"
#include "../glm53_process_memory.h"
#include "../glm53_vector_ops.h"
#include "../glm53_weights.h"

#include <hip/hip_runtime.h>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); goto done; } } while (0)
#define HIP(x) do { hipError_t e_=(x); if(e_!=hipSuccess){std::fprintf(stderr,"HIP FAIL %s:%d: %s: %s\n",__FILE__,__LINE__,#x,hipGetErrorString(e_));goto done;} } while(0)

static uint16_t rne_bf16(float x) {
    uint32_t u; std::memcpy(&u,&x,4); u += 0x7fffu + ((u >> 16u) & 1u);
    return (uint16_t)(u >> 16u);
}
static float decode_bf16(uint16_t x) {
    uint32_t u=(uint32_t)x << 16u; float f; std::memcpy(&f,&u,4); return f;
}
static float sigmoid_serial(float x) {
    if (x >= 0.0f) return 1.0f/(1.0f+std::exp(-x));
    float e=std::exp(x); return e/(1.0f+e);
}
static uint64_t fnv1a(const void *p,size_t n) {
    const uint8_t *b=(const uint8_t*)p; uint64_t h=UINT64_C(1469598103934665603);
    for(size_t i=0;i<n;i++){h^=b[i];h*=UINT64_C(1099511628211);} return h;
}
static bool exact_pread(int fd,void *dst,size_t n,uint64_t off) {
    uint8_t *p=(uint8_t*)dst; size_t done=0;
    while(done<n){ssize_t r=pread(fd,p+done,n-done,(off_t)(off+done)); if(r<0&&errno==EINTR)continue; if(r<=0)return false; done+=(size_t)r;} return true;
}
static bool shape(const k3_st_tensor *t,k3_st_dtype dt,uint8_t ndim,
                  uint64_t a,uint64_t b=0) {
    if(!t||t->dtype!=dt||t->ndim!=ndim||t->shape[0]!=a)return false;
    if(ndim==2 && t->shape[1]!=b)return false;
    uint64_t elems=a*(ndim==2?b:1), bytes=elems*(dt==K3_ST_DTYPE_F32?4u:2u);
    return t->byte_length==bytes;
}
static bool in_manifest(const glm53_manifest &m,const k3_st_tensor *t) {
    for(size_t i=0;i<m.entry_count;i++) if(!std::strcmp(m.entries[i].name,t->name)) return m.entries[i].shard==t->shard;
    return false;
}
struct read_ledger { size_t calls; uint64_t bytes; size_t mmap_calls; };
static bool read_tensor(const k3_st_model &m,const k3_st_tensor *t,void *out,
                        size_t cap,read_ledger *l) {
    if(!t||t->shard>=m.shard_count||t->byte_length>cap||
       t->physical_offset>m.shards[t->shard].file_bytes||
       t->byte_length>m.shards[t->shard].file_bytes-t->physical_offset)return false;
    if(!exact_pread(m.shards[t->shard].fd,out,(size_t)t->byte_length,t->physical_offset))return false;
    l->calls++; l->bytes+=t->byte_length; return true;
}
static bool clean_memory(const glm53_process_memory &m) {
    return m.vm_swap_bytes==0 && m.smaps_swap_bytes==0 && m.smaps_swap_pss_bytes==0 &&
           m.model_vma_count==0 && m.model_vma_bytes==0 && m.largest_model_vma_bytes==0;
}
static void cleanup_hip(const char *operation, hipError_t status) {
    if (status != hipSuccess)
        std::fprintf(stderr, "cleanup warning: %s: %s\n", operation, hipGetErrorString(status));
}
static void oracle_mhc(const std::vector<uint16_t>& streams,const std::vector<uint16_t>& fn,
 const float *base,const float *scale,std::vector<float>& mix,float pre[4],
 std::vector<uint16_t>& post,std::vector<uint16_t>& comb,std::vector<uint16_t>& collapsed) {
    const size_t D=4096,N=4*D; float ss=0.0f;
    for(size_t i=0;i<N;i++){float x=decode_bf16(streams[i]);ss+=x*x;}
    float rms=1.0f/std::sqrt(ss/(float)N+1.0e-5f);
    for(size_t r=0;r<24;r++){float z=0.0f;for(size_t i=0;i<N;i++)z+=decode_bf16(fn[r*N+i])*(decode_bf16(streams[i])*rms);mix[r]=z;}
    float c[16];
    for(int i=0;i<4;i++){pre[i]=sigmoid_serial(mix[i]*scale[0]+base[i])+1.0e-6f;post[i]=rne_bf16(2.0f*sigmoid_serial(mix[4+i]*scale[1]+base[4+i]));}
    for(int src=0;src<4;src++){float mx=-INFINITY;for(int dst=0;dst<4;dst++){int i=src*4+dst;c[i]=mix[8+i]*scale[2]+base[8+i];mx=std::fmax(mx,c[i]);}float den=0;for(int dst=0;dst<4;dst++){int i=src*4+dst;c[i]=std::exp(c[i]-mx);den+=c[i];}for(int dst=0;dst<4;dst++){int i=src*4+dst;c[i]=c[i]/den+1.0e-6f;}}
    for(int dst=0;dst<4;dst++){float z=1.0e-6f;for(int src=0;src<4;src++)z+=c[src*4+dst];for(int src=0;src<4;src++)c[src*4+dst]/=z;}
    for(int it=1;it<20;it++){for(int src=0;src<4;src++){float z=1.0e-6f;for(int dst=0;dst<4;dst++)z+=c[src*4+dst];for(int dst=0;dst<4;dst++)c[src*4+dst]/=z;}for(int dst=0;dst<4;dst++){float z=1.0e-6f;for(int src=0;src<4;src++)z+=c[src*4+dst];for(int src=0;src<4;src++)c[src*4+dst]/=z;}}
    for(int i=0;i<16;i++)comb[i]=rne_bf16(c[i]);
    for(size_t d=0;d<D;d++){float z=0;for(int src=0;src<4;src++)z+=pre[src]*decode_bf16(streams[src*D+d]);collapsed[d]=rne_bf16(z);}
}
static bool compare_float(const char *label,const float *got,const float *want,size_t n,float rel) {
    double worst=-1;size_t at=0;for(size_t i=0;i<n;i++){double ratio=std::fabs((double)got[i]-want[i])/(rel*(1.0+std::fabs((double)want[i])));if(!std::isfinite(got[i]))return false;if(ratio>worst){worst=ratio;at=i;}}
    std::printf("%s max_ratio=%.7g at=%zu got=%.9g ref=%.9g\n",label,worst,at,got[at],want[at]); return worst<=1.0;
}
static bool compare_bits(const char *label,const std::vector<uint16_t>&a,const std::vector<uint16_t>&b) {
    if(a!=b){size_t i=0;while(i<a.size()&&a[i]==b[i])i++;std::fprintf(stderr,"FAIL %s bf16[%zu] got=%04x ref=%04x\n",label,i,(unsigned)a[i],(unsigned)b[i]);return false;}
    std::printf("%s bf16_hash=%016llx exact\n",label,(unsigned long long)fnv1a(a.data(),a.size()*2));return true;
}
static bool compare_bf16_one_step(const char *label,const std::vector<uint16_t>&a,
                                  const std::vector<uint16_t>&b) {
    size_t mismatches=0,max_distance=0,first=a.size();
    if(a.size()!=b.size())return false;
    for(size_t i=0;i<a.size();i++){
        if(a[i]==b[i])continue;
        size_t distance=a[i]>b[i]?(size_t)(a[i]-b[i]):(size_t)(b[i]-a[i]);
        if(first==a.size())first=i;
        mismatches++;if(distance>max_distance)max_distance=distance;
        if(!std::isfinite(decode_bf16(a[i]))||!std::isfinite(decode_bf16(b[i]))||
           ((a[i]^b[i])&UINT16_C(0x8000))!=0||distance>1){
            std::fprintf(stderr,"FAIL %s bf16[%zu] got=%04x ref=%04x code_distance=%zu\n",
                         label,i,(unsigned)a[i],(unsigned)b[i],distance);return false;
        }
    }
    std::printf("%s bf16_hash=%016llx mismatches=%zu max_code_distance=%zu",
                label,(unsigned long long)fnv1a(a.data(),a.size()*2),mismatches,max_distance);
    if(mismatches)std::printf(" first=%zu got=%04x ref=%04x",first,(unsigned)a[first],(unsigned)b[first]);
    std::putchar('\n');return true;
}

int main(int argc,char **argv) {
    constexpr size_t D=4096,N=4*D,F=24*N;
    glm53_manifest manifest={}; k3_st_model all={},main_model={}; glm53_architecture_report ar={}; glm53_weight_plan wp={};
    glm53_process_memory before={},after={}; read_ledger ledger={}; char error[512]={},prefix[PATH_MAX]={};
    std::vector<uint16_t> embed(D),attn_fn(F),ffn_fn(F),input_w(D),post_w(D),streams(N),ffn_streams(N);
    std::vector<float> attn_base(24),attn_scale(3),ffn_base(24),ffn_scale(3);
    std::vector<float> got_mix(24),ref_mix(24),got_norm(D),ref_norm(D),collapsed_f32(D); float got_pre[4]={},ref_pre[4]={};
    std::vector<uint16_t> got_post(4),got_comb(16),got_coll(D),ref_post(4),ref_comb(16),ref_coll(D),gpu_streams(N);
    std::vector<float> fgot_mix(24),fref_mix(24);float fgot_pre[4]={},fref_pre[4]={};std::vector<uint16_t> fgot_post(4),fgot_comb(16),fgot_coll(D),fref_post(4),fref_comb(16),fref_coll(D);
    const k3_st_tensor *ts[9]={}; size_t main_count=0; int result=1,devices=0; hipStream_t q=nullptr;
    uint16_t *d_hidden=nullptr,*d_streams=nullptr,*d_fn=nullptr,*d_post=nullptr,*d_comb=nullptr,*d_coll=nullptr,*d_weight=nullptr;
    float *d_base=nullptr,*d_scale=nullptr,*d_mix=nullptr,*d_pre=nullptr,*d_collf=nullptr,*d_norm=nullptr;
    if(argc!=2){std::fprintf(stderr,"usage: %s OFFICIAL_ROOT\n",argv[0]);return 2;}
    CHECK(realpath(argv[1],prefix)!=nullptr);
    CHECK(glm53_manifest_load(&manifest,argv[1],error,sizeof(error)));
    CHECK(k3_st_model_open_5digit_total(&all,argv[1],GLM53_SHARD_COUNT,error,sizeof(error)));
    CHECK(glm53_manifest_reconcile(&manifest,&all,error,sizeof(error)));
    CHECK(glm53_architecture_validate(&all,&ar,error,sizeof(error)));
    main_model.tensors=(k3_st_tensor*)std::calloc(GLM53_MAIN_TENSOR_COUNT,sizeof(k3_st_tensor));CHECK(main_model.tensors);
    main_model.shards=all.shards;main_model.shard_count=all.shard_count;main_model.routed_span=all.routed_span;main_model.routed_span_context=all.routed_span_context;
    for(size_t i=0;i<all.tensor_count;i++)if(glm53_architecture_validate_main_tensor(&all.tensors[i],nullptr)){CHECK(main_count<GLM53_MAIN_TENSOR_COUNT);main_model.tensors[main_count++]=all.tensors[i];}
    main_model.tensor_count=main_model.tensor_capacity=main_count;CHECK(main_count==GLM53_MAIN_TENSOR_COUNT);
    CHECK(glm53_architecture_validate_main(&main_model,&ar,error,sizeof(error)));
    CHECK(glm53_weight_plan_build_manifest(&wp,&manifest,&main_model,error,sizeof(error)));
    ts[0]=wp.globals[GLM53_GLOBAL_EMBED_TOKENS];ts[1]=wp.layers[0].roles[GLM53_ROLE_HC_ATTN_FN];ts[2]=wp.layers[0].roles[GLM53_ROLE_HC_ATTN_BASE];ts[3]=wp.layers[0].roles[GLM53_ROLE_HC_ATTN_SCALE];ts[4]=wp.layers[0].roles[GLM53_ROLE_INPUT_NORM];ts[5]=wp.layers[0].roles[GLM53_ROLE_HC_FFN_FN];ts[6]=wp.layers[0].roles[GLM53_ROLE_HC_FFN_BASE];ts[7]=wp.layers[0].roles[GLM53_ROLE_HC_FFN_SCALE];ts[8]=wp.layers[0].roles[GLM53_ROLE_POST_ATTN_NORM];
    CHECK(shape(ts[0],K3_ST_DTYPE_BF16,2,154880,4096));CHECK(shape(ts[1],K3_ST_DTYPE_BF16,2,24,16384));CHECK(shape(ts[2],K3_ST_DTYPE_F32,1,24));CHECK(shape(ts[3],K3_ST_DTYPE_F32,1,3));CHECK(shape(ts[4],K3_ST_DTYPE_BF16,1,D));CHECK(shape(ts[5],K3_ST_DTYPE_BF16,2,24,16384));CHECK(shape(ts[6],K3_ST_DTYPE_F32,1,24));CHECK(shape(ts[7],K3_ST_DTYPE_F32,1,3));CHECK(shape(ts[8],K3_ST_DTYPE_BF16,1,D));
    for(auto t:ts)CHECK(in_manifest(manifest,t));
    CHECK(glm53_process_memory_sample(prefix,&before)==GLM53_PROCESS_MEMORY_OK&&clean_memory(before));
    /* Token ID 1 only: the embedding table itself is never allocated or mapped. */
    CHECK(ts[0]->physical_offset<=UINT64_MAX-2*D);CHECK(exact_pread(all.shards[ts[0]->shard].fd,embed.data(),2*D,ts[0]->physical_offset+2*D));ledger.calls++;ledger.bytes+=2*D;
    CHECK(read_tensor(all,ts[1],attn_fn.data(),attn_fn.size()*2,&ledger));CHECK(read_tensor(all,ts[2],attn_base.data(),attn_base.size()*4,&ledger));CHECK(read_tensor(all,ts[3],attn_scale.data(),attn_scale.size()*4,&ledger));CHECK(read_tensor(all,ts[4],input_w.data(),input_w.size()*2,&ledger));
    CHECK(read_tensor(all,ts[5],ffn_fn.data(),ffn_fn.size()*2,&ledger));CHECK(read_tensor(all,ts[6],ffn_base.data(),ffn_base.size()*4,&ledger));CHECK(read_tensor(all,ts[7],ffn_scale.data(),ffn_scale.size()*4,&ledger));CHECK(read_tensor(all,ts[8],post_w.data(),post_w.size()*2,&ledger));
    CHECK(ledger.calls==9&&ledger.bytes==UINT64_C(1597656)&&ledger.mmap_calls==0);
    HIP(hipGetDeviceCount(&devices));CHECK(devices>0);HIP(hipStreamCreate(&q));
    HIP(hipMalloc((void**)&d_hidden,D*2));HIP(hipMalloc((void**)&d_streams,N*2));HIP(hipMalloc((void**)&d_fn,F*2));HIP(hipMalloc((void**)&d_base,24*4));HIP(hipMalloc((void**)&d_scale,3*4));HIP(hipMalloc((void**)&d_mix,24*4));HIP(hipMalloc((void**)&d_pre,4*4));HIP(hipMalloc((void**)&d_post,4*2));HIP(hipMalloc((void**)&d_comb,16*2));HIP(hipMalloc((void**)&d_coll,D*2));HIP(hipMalloc((void**)&d_collf,D*4));HIP(hipMalloc((void**)&d_weight,D*2));HIP(hipMalloc((void**)&d_norm,D*4));
    HIP(hipMemcpyAsync(d_hidden,embed.data(),D*2,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(d_fn,attn_fn.data(),F*2,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(d_base,attn_base.data(),24*4,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(d_scale,attn_scale.data(),3*4,hipMemcpyHostToDevice,q));
    CHECK(glm53_mhc_replicate_bf16(d_streams,N,d_hidden,D,D,q));CHECK(glm53_mhc_prepare_bf16(d_mix,24,d_pre,4,d_post,4,d_comb,16,d_coll,D,d_streams,N,d_fn,F,d_base,24,d_scale,3,D,q));
    CHECK(glm53_vector_cast_bf16_f32(d_collf,D,d_coll,D,D,q));HIP(hipMemcpyAsync(d_weight,input_w.data(),D*2,hipMemcpyHostToDevice,q));CHECK(glm53_vector_rmsnorm_bf16_weight_f32(d_norm,D,d_collf,D,d_weight,D,D,1.0e-5f,q));
    HIP(hipMemcpyAsync(gpu_streams.data(),d_streams,N*2,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(got_mix.data(),d_mix,24*4,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(got_pre,d_pre,4*4,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(got_post.data(),d_post,4*2,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(got_comb.data(),d_comb,16*2,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(got_coll.data(),d_coll,D*2,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(got_norm.data(),d_norm,D*4,hipMemcpyDeviceToHost,q));HIP(hipStreamSynchronize(q));
    for(size_t s=0;s<4;s++)for(size_t d=0;d<D;d++)CHECK(gpu_streams[s*D+d]==embed[d]);streams=gpu_streams;oracle_mhc(streams,attn_fn,attn_base.data(),attn_scale.data(),ref_mix,ref_pre,ref_post,ref_comb,ref_coll);
    CHECK(compare_float("attn.mix",got_mix.data(),ref_mix.data(),24,3e-5f));CHECK(compare_float("attn.pre",got_pre,ref_pre,4,3e-5f));CHECK(compare_bits("attn.post",got_post,ref_post));CHECK(compare_bits("attn.comb",got_comb,ref_comb));CHECK(compare_bf16_one_step("attn.collapsed",got_coll,ref_coll));
    {float ss=0;for(size_t i=0;i<D;i++){collapsed_f32[i]=decode_bf16(ref_coll[i]);ss+=collapsed_f32[i]*collapsed_f32[i];}float inv=1/std::sqrt(ss/(float)D+1e-5f);for(size_t i=0;i<D;i++)ref_norm[i]=collapsed_f32[i]*inv*decode_bf16(input_w[i]);}
    CHECK(compare_float("input_rmsnorm",got_norm.data(),ref_norm.data(),D,3e-5f));
    /* Four distinct deterministic BF16 streams derived only from the official embedding and post norm. */
    for(size_t s=0;s<4;s++)for(size_t d=0;d<D;d++){float e=decode_bf16(embed[d]),w=decode_bf16(post_w[d]);float delta=((int)((d+17*s)%9)-4)*0.0009765625f*w;ffn_streams[s*D+d]=rne_bf16(e+delta+(float)s*0.00048828125f);}
    HIP(hipMemcpyAsync(d_streams,ffn_streams.data(),N*2,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(d_fn,ffn_fn.data(),F*2,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(d_base,ffn_base.data(),24*4,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(d_scale,ffn_scale.data(),3*4,hipMemcpyHostToDevice,q));CHECK(glm53_mhc_prepare_bf16(d_mix,24,d_pre,4,d_post,4,d_comb,16,d_coll,D,d_streams,N,d_fn,F,d_base,24,d_scale,3,D,q));
    HIP(hipMemcpyAsync(fgot_mix.data(),d_mix,24*4,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(fgot_pre,d_pre,4*4,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(fgot_post.data(),d_post,8,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(fgot_comb.data(),d_comb,32,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(fgot_coll.data(),d_coll,D*2,hipMemcpyDeviceToHost,q));HIP(hipStreamSynchronize(q));
    oracle_mhc(ffn_streams,ffn_fn,ffn_base.data(),ffn_scale.data(),fref_mix,fref_pre,fref_post,fref_comb,fref_coll);CHECK(compare_float("ffn.mix",fgot_mix.data(),fref_mix.data(),24,3e-5f));CHECK(compare_float("ffn.pre",fgot_pre,fref_pre,4,3e-5f));CHECK(compare_bits("ffn.post",fgot_post,fref_post));CHECK(compare_bits("ffn.comb",fgot_comb,fref_comb));CHECK(compare_bf16_one_step("ffn.collapsed",fgot_coll,fref_coll));
    CHECK(glm53_process_memory_sample(prefix,&after)==GLM53_PROCESS_MEMORY_OK&&clean_memory(after));
    std::printf("official hashes embed=%016llx attn_fn=%016llx ffn_fn=%016llx norm=%016llx\n",(unsigned long long)fnv1a(embed.data(),D*2),(unsigned long long)fnv1a(attn_fn.data(),F*2),(unsigned long long)fnv1a(ffn_fn.data(),F*2),(unsigned long long)fnv1a(got_norm.data(),D*4));
    std::printf("read_ledger calls=%zu bytes=%llu mmap=%zu model_vmas=%llu\n",ledger.calls,(unsigned long long)ledger.bytes,ledger.mmap_calls,(unsigned long long)after.model_vma_count);
    std::puts("glm53 Phase5C official mHC/vector qualification: PASS");result=0;
done:
    if(result&&error[0])std::fprintf(stderr,"detail: %s\n",error);
    if(q) cleanup_hip("hipStreamDestroy",hipStreamDestroy(q));
    cleanup_hip("hipFree(d_norm)",hipFree(d_norm));cleanup_hip("hipFree(d_weight)",hipFree(d_weight));
    cleanup_hip("hipFree(d_collf)",hipFree(d_collf));cleanup_hip("hipFree(d_coll)",hipFree(d_coll));
    cleanup_hip("hipFree(d_comb)",hipFree(d_comb));cleanup_hip("hipFree(d_post)",hipFree(d_post));
    cleanup_hip("hipFree(d_pre)",hipFree(d_pre));cleanup_hip("hipFree(d_mix)",hipFree(d_mix));
    cleanup_hip("hipFree(d_scale)",hipFree(d_scale));cleanup_hip("hipFree(d_base)",hipFree(d_base));
    cleanup_hip("hipFree(d_fn)",hipFree(d_fn));cleanup_hip("hipFree(d_streams)",hipFree(d_streams));
    cleanup_hip("hipFree(d_hidden)",hipFree(d_hidden));
    glm53_weight_plan_free(&wp);std::free(main_model.tensors);k3_st_model_close(&all);glm53_manifest_free(&manifest);return result;
}
