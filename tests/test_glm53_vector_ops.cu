#include "glm53_vector_ops.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); return false; } } while(0)
#define HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){std::fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));return false;} }while(0)

static uint16_t f32_to_bf16(float x) {
    uint32_t u; std::memcpy(&u,&x,4);
    if ((u & 0x7f800000u)==0x7f800000u && (u & 0x007fffffu))
        return (uint16_t)((u>>16)|0x0040u);
    u += 0x7fffu + ((u>>16)&1u); return (uint16_t)(u>>16);
}
static float bf16_to_f32(uint16_t x) { uint32_t u=(uint32_t)x<<16; float f;std::memcpy(&f,&u,4);return f; }
static bool near(float a,float b,float atol=2e-6f) { return std::fabs((double)a-b)<=atol*(1.0+std::fabs((double)b)); }

template<class T> static bool alloc(T **p,size_t n){return hipMalloc((void**)p,sizeof(T)*n)==hipSuccess;}
static bool test_all(void) {
    constexpr size_t n=9;
    const float ha[n]={-std::numeric_limits<float>::max(),-8.f,-1.f,-0.f,0.f,0.5f,2.f,8.f,std::numeric_limits<float>::max()};
    const float hb[n]={0.f,-2.f,3.f,-4.f,5.f,-6.f,7.f,-8.f,0.f};
    float *a=nullptr,*b=nullptr,*o=nullptr,*o2=nullptr,*w=nullptr; hip_bfloat16 *q=nullptr,*qw=nullptr; hipStream_t s=nullptr;
    constexpr size_t cap=16;
    CHECK(alloc(&a,cap)&&alloc(&b,cap)&&alloc(&o,cap)&&alloc(&o2,cap)&&alloc(&w,cap)&&alloc(&q,cap)&&alloc(&qw,cap)); HIP(hipStreamCreate(&s));
    HIP(hipMemcpyAsync(a,ha,sizeof(ha),hipMemcpyHostToDevice,s)); HIP(hipMemcpyAsync(b,hb,sizeof(hb),hipMemcpyHostToDevice,s)); HIP(hipMemcpyAsync(w,hb,sizeof(hb),hipMemcpyHostToDevice,s));
    CHECK(glm53_vector_add_f32(o,n,a,n,b,n,n,s));
    std::vector<float> got(n); HIP(hipMemcpyAsync(got.data(),o,sizeof(float)*n,hipMemcpyDeviceToHost,s)); HIP(hipStreamSynchronize(s));
    for(size_t i=0;i<n;i++) CHECK(got[i]==ha[i]+hb[i]);
    CHECK(glm53_vector_multiply_f32(o,n,a,n,b,n,n,s)); CHECK(glm53_vector_sigmoid_f32(o2,n,a,n,n,s));
    HIP(hipMemcpyAsync(got.data(),o2,sizeof(float)*n,hipMemcpyDeviceToHost,s)); HIP(hipStreamSynchronize(s));
    for(size_t i=0;i<n;i++){ double x=ha[i]; float want=(float)(x>=0?1.0/(1.0+std::exp(-x)):std::exp(x)/(1.0+std::exp(x))); CHECK(near(got[i],want)); }
    CHECK(got[0]==0.f && got[n-1]==1.f && got[3]==0.5f && got[4]==0.5f);
    CHECK(glm53_vector_silu_f32(o,n,a,n,n,s)); CHECK(glm53_vector_swiglu_f32(o2,n,a,n,b,n,n,3.0f,s));
    HIP(hipMemcpyAsync(got.data(),o2,sizeof(float)*n,hipMemcpyDeviceToHost,s)); HIP(hipStreamSynchronize(s));
    for(size_t i=0;i<n;i++){double x=std::fmin((double)ha[i],3.0), up=std::fmax(-3.0,std::fmin((double)hb[i],3.0)),sig=x>=0?1/(1+std::exp(-x)):std::exp(x)/(1+std::exp(x)); CHECK(near(got[i],(float)(x*sig*up),3e-5f));}

    CHECK(glm53_vector_cast_f32_bf16(q,n,a,n,n,s)); CHECK(glm53_vector_cast_bf16_f32(o,n,q,n,n,s));
    std::vector<uint16_t> qbits(n); HIP(hipMemcpyAsync(qbits.data(),q,sizeof(uint16_t)*n,hipMemcpyDeviceToHost,s)); HIP(hipMemcpyAsync(got.data(),o,sizeof(float)*n,hipMemcpyDeviceToHost,s)); HIP(hipStreamSynchronize(s));
    for(size_t i=0;i<n;i++){CHECK(qbits[i]==f32_to_bf16(ha[i]));CHECK(got[i]==bf16_to_f32(qbits[i]));}

    const float hx[5]={-3,-1,0.5f,2,4}, hw[5]={1,2,-1,0.5f,3};
    HIP(hipMemcpyAsync(a,hx,sizeof(hx),hipMemcpyHostToDevice,s)); HIP(hipMemcpyAsync(w,hw,sizeof(hw),hipMemcpyHostToDevice,s));
    CHECK(glm53_vector_cast_f32_bf16(qw,n,w,n,5,s));
    CHECK(glm53_vector_rmsnorm_f32(o,n,a,n,w,n,5,0.25f,s)); CHECK(glm53_vector_rmsnorm_bf16_weight_f32(o2,n,a,n,qw,n,5,0.25f,s));
    float gr[5],gb[5];HIP(hipMemcpyAsync(gr,o,sizeof(gr),hipMemcpyDeviceToHost,s));HIP(hipMemcpyAsync(gb,o2,sizeof(gb),hipMemcpyDeviceToHost,s));HIP(hipStreamSynchronize(s));
    float ss=0;for(float x:hx)ss+=x*x;float scale=1/std::sqrt(ss/5+0.25f);for(int i=0;i<5;i++){CHECK(near(gr[i],hx[i]*scale*hw[i],2e-5f));CHECK(near(gb[i],hx[i]*scale*bf16_to_f32(f32_to_bf16(hw[i])),2e-5f));}

    CHECK(glm53_vector_dot_f32(o,n,a,n,w,n,5,s)); float d;HIP(hipMemcpyAsync(&d,o,4,hipMemcpyDeviceToHost,s));HIP(hipStreamSynchronize(s));float wantd=0;for(int i=0;i<5;i++)wantd+=hx[i]*hw[i];CHECK(near(d,wantd));
    const float hm[12]={1,2,3,4,-1,0,1,0,0.5f,-0.5f,2,-2}, hi[4]={2,-1,0.5f,3};
    HIP(hipMemcpyAsync(a,hi,sizeof(hi),hipMemcpyHostToDevice,s));HIP(hipMemcpyAsync(w,hm,sizeof(hm),hipMemcpyHostToDevice,s));CHECK(glm53_vector_matvec_f32(o,n,w,n+3,a,n,3,4,s));float mv[3];HIP(hipMemcpyAsync(mv,o,sizeof(mv),hipMemcpyDeviceToHost,s));HIP(hipStreamSynchronize(s));for(int r=0;r<3;r++){float z=0;for(int c=0;c<4;c++)z+=hm[r*4+c]*hi[c];CHECK(mv[r]==z);}

    /* Gather uses raw, independently rounded BF16 payloads and a queued copy. */
    uint16_t ht[12];for(int i=0;i<12;i++)ht[i]=f32_to_bf16((float)i*0.375f-2.f);HIP(hipMemcpyAsync(q,ht,sizeof(ht),hipMemcpyHostToDevice,s));CHECK(glm53_vector_gather_bf16_f32(o,n,q,12,3,4,2,s));float gg[4];HIP(hipMemcpyAsync(gg,o,sizeof(gg),hipMemcpyDeviceToHost,s));HIP(hipStreamSynchronize(s));for(int i=0;i<4;i++)CHECK(gg[i]==bf16_to_f32(ht[8+i]));

    HIP(hipStreamDestroy(s));HIP(hipFree(qw));HIP(hipFree(q));HIP(hipFree(w));HIP(hipFree(o2));HIP(hipFree(o));HIP(hipFree(b));HIP(hipFree(a));return true;
}

static bool test_wide_bf16_matvec(void) {
    constexpr uint32_t rows = 24u, cols = 16384u;
    const size_t matrix_count = (size_t)rows * cols;
    std::vector<uint16_t> matrix(matrix_count);
    std::vector<float> input(cols), want(rows, 0.0f), got(rows);
    for (uint32_t c = 0; c < cols; ++c)
        input[c] = (float)((int)(c % 31u) - 15) * 0.03125f;
    for (uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            const float source = (float)((int)((r * 17u + c * 13u) % 127u) - 63) * 0.015625f;
            const uint16_t bits = f32_to_bf16(source);
            matrix[(size_t)r * cols + c] = bits;
            sum += bf16_to_f32(bits) * input[c];
        }
        want[r] = sum;
    }
    hip_bfloat16 *dm = nullptr; float *di = nullptr, *dout = nullptr;
    hipStream_t stream = nullptr;
    CHECK(alloc(&dm, matrix_count) && alloc(&di, cols) && alloc(&dout, rows));
    HIP(hipStreamCreate(&stream));
    HIP(hipMemcpyAsync(dm, matrix.data(), matrix_count * sizeof(uint16_t), hipMemcpyHostToDevice, stream));
    HIP(hipMemcpyAsync(di, input.data(), cols * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK(glm53_vector_matvec_bf16_weight_f32(
        dout, rows, dm, matrix_count, di, cols, rows, cols, stream));
    HIP(hipMemcpyAsync(got.data(), dout, rows * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP(hipStreamSynchronize(stream));
    for (uint32_t r = 0; r < rows; ++r) CHECK(near(got[r], want[r], 3.0e-5f));
    HIP(hipStreamDestroy(stream)); HIP(hipFree(dout)); HIP(hipFree(di)); HIP(hipFree(dm));
    return true;
}
static bool test_rejections(void) {
    void *a=(void*)0x100000u,*b=(void*)0x200000u,*c=(void*)0x300000u;
    CHECK(!glm53_vector_add_f32(a,3,b,4,c,4,4,nullptr));
    CHECK(!glm53_vector_add_f32(a,4,a,4,c,4,4,nullptr));
    CHECK(!glm53_vector_silu_f32(nullptr,4,b,4,4,nullptr));
    CHECK(!glm53_vector_swiglu_f32(a,4,b,4,c,4,4,-1.0f,nullptr));
    CHECK(!glm53_vector_rmsnorm_f32(a,4,b,4,c,4,4,-1.f,nullptr));
    CHECK(!glm53_vector_rmsnorm_f32(a,4,b,4,c,4,4,std::numeric_limits<float>::infinity(),nullptr));
    CHECK(!glm53_vector_gather_bf16_f32(a,4,b,11,3,4,2,nullptr));
    CHECK(!glm53_vector_matvec_f32(a,16,b,272,c,16,17,16,nullptr));
    CHECK(!glm53_vector_matvec_bf16_weight_f32(a,24,a,393216,c,16384,24,16384,nullptr));
    CHECK(!glm53_vector_matvec_bf16_weight_f32(a,24,(void*)(UINTPTR_MAX-1u),393216,c,16384,24,16384,nullptr));
    CHECK(!glm53_vector_cast_f32_bf16((void*)(UINTPTR_MAX-1u),4,b,4,4,nullptr));
    return true;
}
int main(){int dev;hipDeviceProp_t p{};if(hipGetDevice(&dev)!=hipSuccess||hipGetDeviceProperties(&p,dev)!=hipSuccess){std::fprintf(stderr,"no HIP device\n");return 1;}if(std::strstr(p.gcnArchName,"gfx1151")==nullptr){std::fprintf(stderr,"expected gfx1151, got %s\n",p.gcnArchName);return 1;}if(!test_rejections()||!test_all()||!test_wide_bf16_matvec())return 1;std::printf("glm53 vector ops: PASS on %s (nondefault stream)\n",p.gcnArchName);return 0;}
