#include "../glm53_mhc_ops.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); return 1; } } while (0)
#define HIP(x) CHECK((x)==hipSuccess)
static uint16_t bf(float x) { uint32_t u; std::memcpy(&u,&x,4); u += 0x7fffu+((u>>16)&1u); return uint16_t(u>>16); }
static float fb(uint16_t x) { uint32_t u=uint32_t(x)<<16; float f; std::memcpy(&f,&u,4); return f; }
static float sig(float x) { return x>=0 ? 1.f/(1.f+std::exp(-x)) : std::exp(x)/(1.f+std::exp(x)); }
static bool near(float a,float b,float e=3e-5f){return std::fabs(a-b)<=e*(1.f+std::fabs(b));}
template<class T> static T *dev(size_t n){T*p=nullptr; return hipMalloc((void**)&p,n*sizeof(T))==hipSuccess?p:nullptr;}
static void oracle(size_t D,const std::vector<uint16_t>&s,const std::vector<uint16_t>&fn,
 const float*b,const float*sc,std::vector<float>&mix,float*p,std::vector<uint16_t>&post,
 std::vector<uint16_t>&comb,std::vector<uint16_t>&coll) {
 size_t n=4*D; float ss=0;for(size_t i=0;i<n;i++){float x=fb(s[i]);ss+=x*x;} float r=1/std::sqrt(ss/float(n)+1e-5f);
 for(int row=0;row<24;row++){float z=0;for(size_t i=0;i<n;i++)z+=fb(fn[row*n+i])*(fb(s[i])*r);mix[row]=z;}
 float c[16];for(int i=0;i<4;i++){p[i]=sig(mix[i]*sc[0]+b[i])+1e-6f;post[i]=bf(2*sig(mix[4+i]*sc[1]+b[4+i]));}
 for(int src=0;src<4;src++){float m=-INFINITY;for(int dst=0;dst<4;dst++){int i=8+src*4+dst;c[src*4+dst]=mix[i]*sc[2]+b[i];m=std::fmax(m,c[src*4+dst]);}float den=0;for(int dst=0;dst<4;dst++){int i=src*4+dst;c[i]=std::exp(c[i]-m);den+=c[i];}for(int dst=0;dst<4;dst++)c[src*4+dst]=c[src*4+dst]/den+1e-6f;}
 for(int dst=0;dst<4;dst++){float z=1e-6f;for(int src=0;src<4;src++)z+=c[src*4+dst];for(int src=0;src<4;src++)c[src*4+dst]/=z;}
 for(int it=1;it<20;it++){for(int src=0;src<4;src++){float z=1e-6f;for(int dst=0;dst<4;dst++)z+=c[src*4+dst];for(int dst=0;dst<4;dst++)c[src*4+dst]/=z;}for(int dst=0;dst<4;dst++){float z=1e-6f;for(int src=0;src<4;src++)z+=c[src*4+dst];for(int src=0;src<4;src++)c[src*4+dst]/=z;}}
 for(int i=0;i<16;i++)comb[i]=bf(c[i]);for(size_t d=0;d<D;d++){float z=0;for(int src=0;src<4;src++)z+=p[src]*fb(s[src*D+d]);coll[d]=bf(z);}
}
int main(){
 hipStream_t q=nullptr;HIP(hipStreamCreate(&q)); constexpr size_t D=7,N=4*D,F=24*N;
 std::vector<uint16_t> hs(N),hf(F),hb(D),hr(N);for(size_t i=0;i<N;i++)hs[i]=bf((int(i%13)-6)*.125f);for(size_t i=0;i<F;i++)hf[i]=bf((int(i%17)-8)*.0078125f);for(size_t i=0;i<D;i++)hb[i]=bf((int(i)-3)*.2f);for(size_t i=0;i<N;i++)hr[i]=bf((int(i%11)-5)*.15f);
 float base[24],scale[3]={.7f,-.4f,.9f};for(int i=0;i<24;i++)base[i]=(i-12)*.01f;
 uint16_t *ds=dev<uint16_t>(N),*df=dev<uint16_t>(F),*dp=dev<uint16_t>(4),*dc=dev<uint16_t>(16),*dcol=dev<uint16_t>(D),*db=dev<uint16_t>(D),*dr=dev<uint16_t>(N),*doo=dev<uint16_t>(N),*dm=dev<uint16_t>(D);float *dmi=dev<float>(24),*dpre=dev<float>(4),*dba=dev<float>(24),*dsc=dev<float>(3);CHECK(ds&&df&&dp&&dc&&dcol&&db&&dr&&doo&&dm&&dmi&&dpre&&dba&&dsc);
 HIP(hipMemcpyAsync(ds,hs.data(),2*N,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(df,hf.data(),2*F,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(db,hb.data(),2*D,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(dr,hr.data(),2*N,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(dba,base,sizeof(base),hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(dsc,scale,sizeof(scale),hipMemcpyHostToDevice,q));
 CHECK(glm53_mhc_prepare_bf16(dmi,24,dpre,4,dp,4,dc,16,dcol,D,ds,N,df,F,dba,24,dsc,3,D,q));
 CHECK(glm53_mhc_expand_bf16(doo,N,db,D,dr,N,dp,4,dc,16,D,q));CHECK(glm53_mhc_hyper_mean_bf16(dm,D,ds,N,D,q));
 std::vector<float> gm(24),om(24);float gp[4],op[4];std::vector<uint16_t> gpost(4),gcomb(16),gcol(D),ocol(D),opost(4),ocomb(16),go(N),gmean(D);oracle(D,hs,hf,base,scale,om,op,opost,ocomb,ocol);
 HIP(hipMemcpyAsync(gm.data(),dmi,96,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gp,dpre,16,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gpost.data(),dp,8,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gcomb.data(),dc,32,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gcol.data(),dcol,2*D,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(go.data(),doo,2*N,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gmean.data(),dm,2*D,hipMemcpyDeviceToHost,q));HIP(hipStreamSynchronize(q));
 for(int i=0;i<24;i++)CHECK(near(gm[i],om[i]));for(int i=0;i<4;i++){CHECK(near(gp[i],op[i]));CHECK(gpost[i]==opost[i]);}for(int i=0;i<16;i++)CHECK(gcomb[i]==ocomb[i]);for(size_t i=0;i<D;i++)CHECK(gcol[i]==ocol[i]);
 for(size_t dst=0;dst<4;dst++)for(size_t d=0;d<D;d++){uint16_t a=bf(fb(opost[dst])*fb(hb[d]));float z=0;for(size_t src=0;src<4;src++)z+=fb(ocomb[src*4+dst])*fb(hr[src*D+d]);uint16_t b=bf(z);CHECK(go[dst*D+d]==bf(fb(a)+fb(b)));}for(size_t d=0;d<D;d++){float z=0;for(int s=0;s<4;s++)z+=fb(hs[s*D+d]);CHECK(gmean[d]==bf(z*.25f));}
 /* Rounding boundary: pinned BF16 graph rounds post product and matmul
  * separately before the add. A one-round implementation yields 0x402c. */
 {uint16_t xb[7]={bf(0.79296875f)},xr[28]={0},xp[4]={bf(-1.1484375f),0,0,0},xc[16]={0},xo[4]={0};
  const float cv[4]={-0.9765625f,0.7890625f,-0.1025390625f,-0.890625f};
  const float rv[4]={-0.15234375f,0.97265625f,-0.2734375f,-2.984375f};
  for(int qx=0;qx<4;qx++){xc[qx*4]=bf(cv[qx]);xr[qx]=bf(rv[qx]);}
  HIP(hipMemcpyAsync(db,xb,sizeof(xb),hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(dr,xr,sizeof(xr),hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(dp,xp,sizeof(xp),hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(dc,xc,sizeof(xc),hipMemcpyHostToDevice,q));
  CHECK(glm53_mhc_expand_bf16(doo,N,db,D,dr,N,dp,4,dc,16,1,q));HIP(hipMemcpyAsync(xo,doo,sizeof(xo),hipMemcpyDeviceToHost,q));HIP(hipStreamSynchronize(q));CHECK(xo[0]==UINT16_C(0x402d));}
 /* Exact alias, partial alias, short counts, and overflow are rejected before enqueue. */
 CHECK(!glm53_mhc_expand_bf16(dr,N,db,D,dr,N,dp,4,dc,16,D,q));CHECK(!glm53_mhc_hyper_mean_bf16(ds,D,ds,N,D,q));CHECK(!glm53_mhc_replicate_bf16(ds,N-1,db,D,D,q));CHECK(!glm53_mhc_replicate_bf16(ds,N,db,D,SIZE_MAX/2+1,q));
 /* Official width on the same non-default stream. */
 constexpr size_t OD=4096,ON=4*OD,OF=24*ON;uint16_t *oh=dev<uint16_t>(OD),*os=dev<uint16_t>(ON),*omean=dev<uint16_t>(OD),*ofn=dev<uint16_t>(OF),*ocoll=dev<uint16_t>(OD),*opst=dev<uint16_t>(4),*ocmb=dev<uint16_t>(16);float *omix=dev<float>(24),*opre=dev<float>(4);std::vector<uint16_t> hh(OD),gg(ON),mg(OD),hfn(OF),gc(OD),gpo(4),gco(16),opo(4),oco(16),oco2(OD);for(size_t i=0;i<OD;i++)hh[i]=bf((int(i%29)-14)*.03125f);for(size_t i=0;i<OF;i++)hfn[i]=bf((int(i%23)-11)*.001953125f);HIP(hipMemcpyAsync(oh,hh.data(),2*OD,hipMemcpyHostToDevice,q));HIP(hipMemcpyAsync(ofn,hfn.data(),2*OF,hipMemcpyHostToDevice,q));CHECK(glm53_mhc_replicate_bf16(os,ON,oh,OD,OD,q));CHECK(glm53_mhc_hyper_mean_bf16(omean,OD,os,ON,OD,q));CHECK(glm53_mhc_prepare_bf16(omix,24,opre,4,opst,4,ocmb,16,ocoll,OD,os,ON,ofn,OF,dba,24,dsc,3,OD,q));HIP(hipMemcpyAsync(gg.data(),os,2*ON,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(mg.data(),omean,2*OD,hipMemcpyDeviceToHost,q));std::vector<float> gomi(24),oomi(24);float gopr[4],oopr[4];HIP(hipMemcpyAsync(gomi.data(),omix,96,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gopr,opre,16,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gpo.data(),opst,8,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gco.data(),ocmb,32,hipMemcpyDeviceToHost,q));HIP(hipMemcpyAsync(gc.data(),ocoll,2*OD,hipMemcpyDeviceToHost,q));HIP(hipStreamSynchronize(q));for(size_t s=0;s<4;s++)for(size_t d=0;d<OD;d++)CHECK(gg[s*OD+d]==hh[d]);for(size_t d=0;d<OD;d++)CHECK(mg[d]==hh[d]);oracle(OD,gg,hfn,base,scale,oomi,oopr,opo,oco,oco2);for(int i=0;i<24;i++)CHECK(near(gomi[i],oomi[i]));for(int i=0;i<4;i++){CHECK(near(gopr[i],oopr[i]));CHECK(gpo[i]==opo[i]);}for(int i=0;i<16;i++)CHECK(gco[i]==oco[i]);for(size_t i=0;i<OD;i++)CHECK(gc[i]==oco2[i]);
 std::puts("glm53 mHC GPU ops: PASS (gfx device, non-default stream, D=7 and D=4096)");return 0;
}
