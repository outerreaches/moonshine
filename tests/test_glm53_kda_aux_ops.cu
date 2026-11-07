#include "../glm53_kda_aux_ops.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); return 1; } } while (0)
#define HIP(x) CHECK((x) == hipSuccess)

static uint16_t bf(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    u += UINT32_C(0x7fff) + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
static float fb(uint16_t x) {
    uint32_t u = static_cast<uint32_t>(x) << 16; float f;
    std::memcpy(&f, &u, 4); return f;
}
static float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x); return e / (1.0f + e);
}
template<class T> static T *device(size_t n) {
    T *p = nullptr; return hipMalloc(reinterpret_cast<void **>(&p), n*sizeof(T)) == hipSuccess ? p : nullptr;
}
static uint16_t conv_oracle(const uint16_t *history, const uint16_t *w, uint16_t input,
                            uint16_t *next) {
    next[0]=history[1]; next[1]=history[2]; next[2]=history[3]; next[3]=input;
    float sum = fb(next[0]) * fb(w[0]);
    sum += fb(next[1]) * fb(w[1]);
    sum += fb(next[2]) * fb(w[2]);
    sum += fb(next[3]) * fb(w[3]);
    const float staged = fb(bf(sum));
    return bf(staged * sigmoid(staged));
}
static void rms_oracle(size_t heads, const std::vector<uint16_t>& x,
                       const std::vector<uint16_t>& gate,
                       const std::vector<uint16_t>& weight, float eps,
                       std::vector<uint16_t>& output) {
    for (size_t h=0; h<heads; ++h) {
        float reduction[128];
        for (size_t d=0; d<128; ++d) { float v=fb(x[h*128+d]); reduction[d]=v*v; }
        for (size_t width=64; width; width>>=1)
            for (size_t d=0; d<width; ++d) reduction[d] += reduction[d+width];
        const float scale=1.0f/std::sqrt(reduction[0]/128.0f+eps);
        for (size_t d=0; d<128; ++d) {
            const size_t i=h*128+d;
            output[i]=bf(fb(x[i])*scale*fb(weight[d])*sigmoid(fb(gate[i])));
        }
    }
}
static bool same_or_adjacent(uint16_t a,uint16_t b) {
    return a==b || (a>0 && a-1==b) || (b>0 && b-1==a);
}

int main() {
    hipStream_t stream=nullptr; HIP(hipStreamCreate(&stream));

    /* Small split-cache continuation, reset, source immutability, and staging. */
    constexpr size_t C=9, CN=4*C;
    std::vector<uint16_t> input(C), weight(CN), a(CN), snapshot(CN), got(C), got_cache(CN);
    for(size_t i=0;i<C;i++) input[i]=bf((static_cast<int>(i%7)-3)*0.34375f);
    for(size_t i=0;i<CN;i++) { weight[i]=bf((static_cast<int>(i%13)-6)*0.15625f); a[i]=bf((static_cast<int>(i%11)-5)*0.21875f); }
    snapshot=a;
    uint16_t *di=device<uint16_t>(C), *dw=device<uint16_t>(CN), *da=device<uint16_t>(CN),
             *db=device<uint16_t>(CN), *dout=device<uint16_t>(C);
    CHECK(di&&dw&&da&&db&&dout);
    HIP(hipMemcpyAsync(di,input.data(),2*C,hipMemcpyHostToDevice,stream));
    HIP(hipMemcpyAsync(dw,weight.data(),2*CN,hipMemcpyHostToDevice,stream));
    HIP(hipMemcpyAsync(da,a.data(),2*CN,hipMemcpyHostToDevice,stream));
    CHECK(glm53_kda_conv4_silu_bf16(dout,C,db,CN,di,C,da,CN,dw,CN,C,stream));
    HIP(hipMemcpyAsync(got.data(),dout,2*C,hipMemcpyDeviceToHost,stream));
    HIP(hipMemcpyAsync(got_cache.data(),db,2*CN,hipMemcpyDeviceToHost,stream));
    HIP(hipMemcpyAsync(a.data(),da,2*CN,hipMemcpyDeviceToHost,stream));
    HIP(hipStreamSynchronize(stream));
    for(size_t c=0;c<C;c++){uint16_t next[4];CHECK(got[c]==conv_oracle(&snapshot[4*c],&weight[4*c],input[c],next));for(int k=0;k<4;k++)CHECK(got_cache[4*c+k]==next[k]);}
    CHECK(a==snapshot);
    /* Continue with the destination as source, then reset from a zero cache. */
    for(size_t i=0;i<C;i++) input[i]=bf((static_cast<int>(i%5)-2)*0.53125f);
    HIP(hipMemcpyAsync(di,input.data(),2*C,hipMemcpyHostToDevice,stream));
    CHECK(glm53_kda_conv4_silu_bf16(dout,C,da,CN,di,C,db,CN,dw,CN,C,stream));
    HIP(hipMemcpyAsync(got_cache.data(),da,2*CN,hipMemcpyDeviceToHost,stream));
    HIP(hipStreamSynchronize(stream));
    for(size_t c=0;c<C;c++){CHECK(got_cache[4*c]==snapshot[4*c+2]);CHECK(got_cache[4*c+1]==snapshot[4*c+3]);CHECK(got_cache[4*c+2]==bf((static_cast<int>(c%7)-3)*0.34375f));CHECK(got_cache[4*c+3]==input[c]);}
    std::vector<uint16_t> zeros(CN,0); HIP(hipMemcpyAsync(db,zeros.data(),2*CN,hipMemcpyHostToDevice,stream));
    CHECK(glm53_kda_conv4_silu_bf16(dout,C,da,CN,di,C,db,CN,dw,CN,C,stream));
    HIP(hipMemcpyAsync(got_cache.data(),da,2*CN,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));
    for(size_t c=0;c<C;c++){CHECK(got_cache[4*c]==0);CHECK(got_cache[4*c+1]==0);CHECK(got_cache[4*c+2]==0);CHECK(got_cache[4*c+3]==input[c]);}
    /* A pinned boundary: staging convolution to 0x408b gives output 0x4089;
       evaluating SiLU from the unrounded convolution gives 0x408a. */
    {uint16_t hs[4]={0,UINT16_C(0x3ea5),UINT16_C(0x3f45),UINT16_C(0x4002)};
     uint16_t xi=UINT16_C(0x400a), ww[4]={UINT16_C(0xbe29),UINT16_C(0xbf05),UINT16_C(0x4011),UINT16_C(0x3dc2)}, go=0;
     HIP(hipMemcpyAsync(db,hs,8,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(di,&xi,2,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(dw,ww,8,hipMemcpyHostToDevice,stream));
     CHECK(glm53_kda_conv4_silu_bf16(dout,C,da,CN,di,C,db,CN,dw,CN,1,stream));HIP(hipMemcpyAsync(&go,dout,2,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));CHECK(go==UINT16_C(0x4089));}

    /* Official RMSNormGated shape and an independent host reduction oracle. */
    constexpr size_t H=64,N=H*128;
    std::vector<uint16_t> hx(N),hg(N),hw(128),ho(N),gpu(N);
    for(size_t i=0;i<N;i++){hx[i]=bf((static_cast<int>(i%37)-18)*0.078125f);hg[i]=bf((static_cast<int>(i%41)-20)*0.1875f);}
    for(size_t i=0;i<128;i++)hw[i]=bf(0.625f+(i%17)*0.03125f);
    rms_oracle(H,hx,hg,hw,1e-5f,ho);
    uint16_t *dx=device<uint16_t>(N),*dg=device<uint16_t>(N),*dnw=device<uint16_t>(128),*dro=device<uint16_t>(N);
    CHECK(dx&&dg&&dnw&&dro);HIP(hipMemcpyAsync(dx,hx.data(),2*N,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(dg,hg.data(),2*N,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(dnw,hw.data(),256,hipMemcpyHostToDevice,stream));
    CHECK(glm53_kda_rmsnorm_gated_bf16(dro,N,dx,N,dg,N,dnw,128,H,1e-5f,stream));HIP(hipMemcpyAsync(gpu.data(),dro,2*N,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));
    for(size_t i=0;i<N;i++)CHECK(same_or_adjacent(gpu[i],ho[i]));

    /* Full official convolution shape. */
    constexpr size_t OC=8192,OCN=4*OC;std::vector<uint16_t> oi(OC),ow(OCN),os(OCN),oo(OC),oc(OCN);
    for(size_t i=0;i<OC;i++)oi[i]=bf((static_cast<int>(i%23)-11)*0.0625f);
    for(size_t i=0;i<OCN;i++){ow[i]=bf((static_cast<int>(i%19)-9)*0.046875f);os[i]=bf((static_cast<int>(i%29)-14)*0.03125f);}
    uint16_t *doi=device<uint16_t>(OC),*dow=device<uint16_t>(OCN),*dos=device<uint16_t>(OCN),*doc=device<uint16_t>(OCN),*doo=device<uint16_t>(OC);
    CHECK(doi&&dow&&dos&&doc&&doo);HIP(hipMemcpyAsync(doi,oi.data(),2*OC,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(dow,ow.data(),2*OCN,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(dos,os.data(),2*OCN,hipMemcpyHostToDevice,stream));
    CHECK(glm53_kda_conv4_silu_bf16(doo,OC,doc,OCN,doi,OC,dos,OCN,dow,OCN,OC,stream));HIP(hipMemcpyAsync(oo.data(),doo,2*OC,hipMemcpyDeviceToHost,stream));HIP(hipMemcpyAsync(oc.data(),doc,2*OCN,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));
    for(size_t c=0;c<OC;c++){uint16_t next[4];CHECK(oo[c]==conv_oracle(&os[4*c],&ow[4*c],oi[c],next));for(int k=0;k<4;k++)CHECK(oc[4*c+k]==next[k]);}

    /* Short counts, exact/partial alias, arithmetic/grid overflow, and eps. */
    CHECK(!glm53_kda_conv4_silu_bf16(doo,OC-1,doc,OCN,doi,OC,dos,OCN,dow,OCN,OC,stream));
    CHECK(!glm53_kda_conv4_silu_bf16(doo,OC,doc,OCN,doi,OC,doc,OCN,dow,OCN,OC,stream));
    CHECK(!glm53_kda_conv4_silu_bf16(doo,OC,doc,OCN,doi,OC,reinterpret_cast<char*>(doc)+2,OCN,dow,OCN,OC,stream));
    CHECK(!glm53_kda_conv4_silu_bf16(doo,SIZE_MAX,doc,SIZE_MAX,doi,SIZE_MAX,dos,SIZE_MAX,dow,SIZE_MAX,SIZE_MAX/4+1,stream));
    CHECK(!glm53_kda_rmsnorm_gated_bf16(dro,N,dx,N,dx,N,dnw,128,H,1e-5f,stream));
    CHECK(!glm53_kda_rmsnorm_gated_bf16(dro,N-1,dx,N,dg,N,dnw,128,H,1e-5f,stream));
    CHECK(!glm53_kda_rmsnorm_gated_bf16(dro,N,dx,N,dg,N,dnw,128,H,0.0f,stream));
    CHECK(!glm53_kda_rmsnorm_gated_bf16(dro,N,dx,N,dg,N,dnw,128,H,-1.0f,stream));
    CHECK(!glm53_kda_rmsnorm_gated_bf16(dro,SIZE_MAX,dx,SIZE_MAX,dg,SIZE_MAX,dnw,SIZE_MAX,SIZE_MAX/128+1,1e-5f,stream));

    std::puts("glm53 KDA auxiliary GPU ops: PASS (gfx device, non-default stream, C=8192 H=64)");
    return 0;
}
