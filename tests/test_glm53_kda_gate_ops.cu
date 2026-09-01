#include "glm53_kda_gate_ops.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); return false; } } while (0)
#define HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){std::fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));return false;} } while (0)

static uint16_t f32_to_bf16(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu))
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
static float bf16_to_f32(uint16_t x) {
    uint32_t u=static_cast<uint32_t>(x)<<16; float f;
    std::memcpy(&f,&u,sizeof(f)); return f;
}
static float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f/(1.0f+std::exp(-x));
    float e=std::exp(x); return e/(1.0f+e);
}
static bool close(float a,float b,float scale=2.0e-6f) {
    return std::fabs(static_cast<double>(a)-b) <= scale*(1.0+std::fabs(static_cast<double>(b)));
}
template<class T> static bool alloc(T **p,size_t n) { return hipMalloc(reinterpret_cast<void **>(p),n*sizeof(T))==hipSuccess; }

static bool test_official_nondefault(void) {
    constexpr size_t heads=GLM53_KDA_GATE_OFFICIAL_HEADS;
    constexpr size_t dim=GLM53_KDA_GATE_HEAD_DIM;
    constexpr size_t n=heads*dim;
    std::vector<uint16_t> raw(n), raw_beta(heads);
    std::vector<float> dt(n), alog(heads), want(n), got(n), beta(heads);
    const float beta_inputs[] = {-100.f,-20.f,-8.f,-4.f,-2.f,-1.f,-0.015625f,
        -0.0078125f,-0.0f,0.0f,0.0078125f,0.015625f,1.f,2.f,4.f,8.f,20.f,100.f};
    for(size_t h=0;h<heads;++h) {
        alog[h]=(static_cast<int>(h%9)-4)*0.25f;
        raw_beta[h]=f32_to_bf16(beta_inputs[h%(sizeof(beta_inputs)/sizeof(beta_inputs[0]))]);
        for(size_t d=0;d<dim;++d) {
            size_t i=h*dim+d;
            float source=(static_cast<int>((i*37u)%257u)-128)*0.03125f;
            raw[i]=f32_to_bf16(source);
            dt[i]=(static_cast<int>((i*19u)%113u)-56)*0.015625f;
            float x=bf16_to_f32(raw[i])+dt[i];
            want[i]=-5.0f*stable_sigmoid(std::exp(alog[h])*x);
        }
    }
    hip_bfloat16 *dr=nullptr,*db=nullptr; float *dd=nullptr,*da=nullptr,*dg=nullptr,*dbo=nullptr;
    hipStream_t stream=nullptr;
    CHECK(alloc(&dr,n)&&alloc(&db,heads)&&alloc(&dd,n)&&alloc(&da,heads)&&alloc(&dg,n)&&alloc(&dbo,heads));
    HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    HIP(hipMemcpyAsync(dr,raw.data(),n*sizeof(uint16_t),hipMemcpyHostToDevice,stream));
    HIP(hipMemcpyAsync(db,raw_beta.data(),heads*sizeof(uint16_t),hipMemcpyHostToDevice,stream));
    HIP(hipMemcpyAsync(dd,dt.data(),n*sizeof(float),hipMemcpyHostToDevice,stream));
    HIP(hipMemcpyAsync(da,alog.data(),heads*sizeof(float),hipMemcpyHostToDevice,stream));
    CHECK(glm53_kda_prepare_forget_f32(dg,n,dr,n,dd,n,da,heads,heads,dim,-5.0f,stream));
    CHECK(glm53_kda_prepare_beta_f32(dbo,heads,db,heads,heads,stream));
    HIP(hipMemcpyAsync(got.data(),dg,n*sizeof(float),hipMemcpyDeviceToHost,stream));
    HIP(hipMemcpyAsync(beta.data(),dbo,heads*sizeof(float),hipMemcpyDeviceToHost,stream));
    HIP(hipStreamSynchronize(stream));
    for(size_t i=0;i<n;++i) CHECK(close(got[i],want[i],3.0e-6f));
    for(size_t h=0;h<heads;++h) {
        const float x=bf16_to_f32(raw_beta[h]);
        const uint16_t expected_bits=f32_to_bf16(stable_sigmoid(x));
        uint32_t got_bits; std::memcpy(&got_bits,&beta[h],sizeof(got_bits));
        CHECK((got_bits&0xffffu)==0u);
        CHECK(static_cast<uint16_t>(got_bits>>16)==expected_bits);
    }
    /* Stable sigmoid and BF16 boundary behavior: saturation and the values
       immediately around the 0.5 rounding cell remain exact BF16 payloads. */
    CHECK(beta[0]==0.0f); CHECK(beta[17]==1.0f);
    CHECK(beta[8]==0.5f && beta[9]==0.5f);
    HIP(hipStreamDestroy(stream)); HIP(hipFree(dbo));HIP(hipFree(dg));HIP(hipFree(da));
    HIP(hipFree(dd));HIP(hipFree(db));HIP(hipFree(dr)); return true;
}

static bool test_generic_lower_bound(void) {
    constexpr size_t heads=3,dim=128,n=heads*dim;
    std::vector<uint16_t> raw(n,f32_to_bf16(0.25f));
    std::vector<float> dt(n,-0.5f), a(heads,0.0f), got(n);
    hip_bfloat16 *dr=nullptr; float *dd=nullptr,*da=nullptr,*out=nullptr; hipStream_t s=nullptr;
    CHECK(alloc(&dr,n)&&alloc(&dd,n)&&alloc(&da,heads)&&alloc(&out,n)); HIP(hipStreamCreate(&s));
    HIP(hipMemcpyAsync(dr,raw.data(),2*n,hipMemcpyHostToDevice,s));HIP(hipMemcpyAsync(dd,dt.data(),4*n,hipMemcpyHostToDevice,s));HIP(hipMemcpyAsync(da,a.data(),4*heads,hipMemcpyHostToDevice,s));
    CHECK(glm53_kda_prepare_forget_f32(out,n,dr,n,dd,n,da,heads,heads,dim,-2.25f,s));
    HIP(hipMemcpyAsync(got.data(),out,4*n,hipMemcpyDeviceToHost,s));HIP(hipStreamSynchronize(s));
    float want=-2.25f*stable_sigmoid(-0.25f);for(float x:got)CHECK(close(x,want));
    HIP(hipStreamDestroy(s));HIP(hipFree(out));HIP(hipFree(da));HIP(hipFree(dd));HIP(hipFree(dr));return true;
}

static bool test_rejections(void) {
    void *a=reinterpret_cast<void*>(0x100000u),*b=reinterpret_cast<void*>(0x200000u);
    void *c=reinterpret_cast<void*>(0x300000u),*d=reinterpret_cast<void*>(0x400000u);
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,128,c,128,d,1,0,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,127,b,128,c,128,d,1,1,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,127,c,128,d,1,1,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,128,c,127,d,1,1,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,128,c,128,d,0,1,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,128,c,128,d,1,1,127,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,128,c,128,d,1,1,128,0,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,128,c,128,d,1,1,128,std::numeric_limits<float>::infinity(),nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,a,128,c,128,d,1,1,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,128,reinterpret_cast<char*>(a)+2,128,d,1,1,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(a,128,b,128,c,128,d,1,SIZE_MAX/128u+1u,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_forget_f32(reinterpret_cast<void*>(UINTPTR_MAX-1u),128,b,128,c,128,d,1,1,128,-5,nullptr));
    CHECK(!glm53_kda_prepare_beta_f32(a,0,b,1,1,nullptr));
    CHECK(!glm53_kda_prepare_beta_f32(a,1,a,1,1,nullptr));
    CHECK(!glm53_kda_prepare_beta_f32(reinterpret_cast<void*>(UINTPTR_MAX-1u),1,b,1,1,nullptr));
    CHECK(!glm53_kda_prepare_beta_f32(a,SIZE_MAX,b,1,1,nullptr));
    return true;
}

int main() {
    int dev; hipDeviceProp_t p{};
    if(hipGetDevice(&dev)!=hipSuccess||hipGetDeviceProperties(&p,dev)!=hipSuccess){std::fprintf(stderr,"no HIP device\n");return 1;}
    if(std::strstr(p.gcnArchName,"gfx1151")==nullptr){std::fprintf(stderr,"expected gfx1151, got %s\n",p.gcnArchName);return 1;}
    if(!test_rejections()||!test_official_nondefault()||!test_generic_lower_bound())return 1;
    std::printf("glm53 KDA gate preparation: PASS on %s (official heads=64, nondefault stream)\n",p.gcnArchName);return 0;
}
