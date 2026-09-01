#include "../glm53_kda_ops.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); return 1; } } while (0)
#define HIP(x) CHECK((x) == hipSuccess)
constexpr size_t D=128, T=5, S=D*D;
template<class X> static X *device(size_t n){X*p=nullptr;return hipMalloc((void**)&p,n*sizeof(X))==hipSuccess?p:nullptr;}
static bool close(float a,float b,float *worst){float e=std::fabs(a-b);if(e>*worst)*worst=e;return std::isfinite(a)&&std::isfinite(b)&&e<=3e-6f+4e-6f*std::fmax(std::fabs(a),std::fabs(b));}
static void oracle(float*out,float*dst,const float*q,const float*k,const float*v,const float*g,const float*beta,const float*src,size_t heads){
 for(size_t h=0;h<heads;h++){size_t b=h*D,s=h*S;float qs=0,ks=0;for(size_t x=0;x<D;x++){qs+=q[b+x]*q[b+x];ks+=k[b+x]*k[b+x];}float qd=std::sqrt(qs+1e-6f),kd=std::sqrt(ks+1e-6f),dd=std::sqrt(float(D));
  for(size_t x=0;x<D;x++)for(size_t y=0;y<D;y++)dst[s+x*D+y]=src[s+x*D+y]*std::exp(g[b+x]);
  for(size_t y=0;y<D;y++){float p=0;for(size_t x=0;x<D;x++)p+=dst[s+x*D+y]*(k[b+x]/kd);float de=(v[b+y]-p)*beta[h];for(size_t x=0;x<D;x++)dst[s+x*D+y]+=(k[b+x]/kd)*de;float z=0;for(size_t x=0;x<D;x++)z+=(q[b+x]/qd/dd)*dst[s+x*D+y];out[b+y]=z;}
 }}
static bool launch(float*o,float*d,const float*q,const float*k,const float*v,const float*g,const float*b,const float*s,size_t heads,hipStream_t stream){size_t n=heads*D,st=n*D;return glm53_kda_recurrent_f32(o,n,d,st,q,n,k,n,v,n,g,n,b,heads,s,st,heads,stream);}
int main(int argc,char**argv){
 const char *path=argc==2?argv[1]:"tests/fixtures/glm53_phase4_kda_v1.bin";CHECK(argc<=2);
 int devices=0;HIP(hipGetDeviceCount(&devices));if(!devices){std::puts("glm53 KDA GPU ops: SKIP (no ROCm device)");return 0;}HIP(hipSetDevice(0));hipDeviceProp_t prop;HIP(hipGetDeviceProperties(&prop,0));
 hipStream_t stream=nullptr;HIP(hipStreamCreate(&stream));
 /* Pinned official head-0 T=5 fixture. */
 FILE*f=std::fopen(path,"rb");CHECK(f);unsigned char hdr[36];CHECK(std::fread(hdr,1,36,f)==36);const unsigned char magic[8]={'G','5','3','K','D','A','4',0};CHECK(!std::memcmp(hdr,magic,8));auto u32=[](const unsigned char*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);};CHECK(u32(hdr+8)==1&&u32(hdr+12)==0x01020304u&&u32(hdr+16)==T&&u32(hdr+20)==D&&u32(hdr+24)==D&&u32(hdr+28)==8);
 size_t nf=u32(hdr+32);CHECK(nf==4*T*D+T+S+T*D+S);std::vector<float>x(nf);CHECK(std::fread(x.data(),4,nf,f)==nf);CHECK(std::fgetc(f)==EOF);std::fclose(f);size_t off=0;float *hq=x.data()+off;off+=T*D;float *hk=x.data()+off;off+=T*D;float *hv=x.data()+off;off+=T*D;float *hg=x.data()+off;off+=T*D;float *hb=x.data()+off;off+=T;float *hi=x.data()+off;off+=S;float *he=x.data()+off;off+=T*D;float *hs=x.data()+off;off+=S;CHECK(off==nf);
 float *dq=device<float>(T*D),*dk=device<float>(T*D),*dv=device<float>(T*D),*dg=device<float>(T*D),*db=device<float>(T),*do1=device<float>(T*D),*do2=device<float>(T*D),*sa=device<float>(S),*sb=device<float>(S),*ta=device<float>(S),*tb=device<float>(S);CHECK(dq&&dk&&dv&&dg&&db&&do1&&do2&&sa&&sb&&ta&&tb);
 HIP(hipMemcpyAsync(dq,hq,4*T*D,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(dk,hk,4*T*D,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(dv,hv,4*T*D,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(dg,hg,4*T*D,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(db,hb,4*T,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(sa,hi,4*S,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(ta,hi,4*S,hipMemcpyHostToDevice,stream));
 float *cur=sa,*next=sb;for(size_t t=0;t<T;t++){CHECK(launch(do1+t*D,next,dq+t*D,dk+t*D,dv+t*D,dg+t*D,db+t,cur,1,stream));float*z=cur;cur=next;next=z;}float *full=cur;
 cur=ta;next=tb;for(size_t t=0;t<2;t++){CHECK(launch(do2+t*D,next,dq+t*D,dk+t*D,dv+t*D,dg+t*D,db+t,cur,1,stream));float*z=cur;cur=next;next=z;}for(size_t t=2;t<T;t++){CHECK(launch(do2+t*D,next,dq+t*D,dk+t*D,dv+t*D,dg+t*D,db+t,cur,1,stream));float*z=cur;cur=next;next=z;}float *split=cur;
 std::vector<float>go1(T*D),go2(T*D),gst(S),gst2(S);HIP(hipMemcpyAsync(go1.data(),do1,4*T*D,hipMemcpyDeviceToHost,stream));HIP(hipMemcpyAsync(go2.data(),do2,4*T*D,hipMemcpyDeviceToHost,stream));HIP(hipMemcpyAsync(gst.data(),full,4*S,hipMemcpyDeviceToHost,stream));HIP(hipMemcpyAsync(gst2.data(),split,4*S,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));float wo=0,ws=0;for(size_t i=0;i<T*D;i++){CHECK(close(go1[i],he[i],&wo));CHECK(go1[i]==go2[i]);}for(size_t i=0;i<S;i++){CHECK(close(gst[i],hs[i],&ws));CHECK(gst[i]==gst2[i]);}
 /* Generic two-head path against an independent serial F32 oracle. */
 constexpr size_t H=2,N=H*D,ST=H*S;std::vector<float>q(N),k(N),v(N),g(N),be(H),src(ST),eo(N),es(ST),got(N),gs(ST);for(size_t i=0;i<N;i++){q[i]=(int(i%23)-11)*.03125f;k[i]=(int(i%19)-9)*.046875f;v[i]=(int(i%17)-8)*.025f;g[i]=-float(i%7)*.0625f;}be[0]=.25f;be[1]=.8f;for(size_t i=0;i<ST;i++)src[i]=(int(i%29)-14)/8192.f;oracle(eo.data(),es.data(),q.data(),k.data(),v.data(),g.data(),be.data(),src.data(),H);
 float *q2=device<float>(N),*k2=device<float>(N),*v2=device<float>(N),*g2=device<float>(N),*b2=device<float>(H),*s2=device<float>(ST),*d2=device<float>(ST),*o2=device<float>(N);CHECK(q2&&k2&&v2&&g2&&b2&&s2&&d2&&o2);HIP(hipMemcpyAsync(q2,q.data(),4*N,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(k2,k.data(),4*N,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(v2,v.data(),4*N,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(g2,g.data(),4*N,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(b2,be.data(),4*H,hipMemcpyHostToDevice,stream));HIP(hipMemcpyAsync(s2,src.data(),4*ST,hipMemcpyHostToDevice,stream));CHECK(launch(o2,d2,q2,k2,v2,g2,b2,s2,H,stream));HIP(hipMemcpyAsync(got.data(),o2,4*N,hipMemcpyDeviceToHost,stream));HIP(hipMemcpyAsync(gs.data(),d2,4*ST,hipMemcpyDeviceToHost,stream));std::vector<float>unchanged(ST);HIP(hipMemcpyAsync(unchanged.data(),s2,4*ST,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));float syn=0;for(size_t i=0;i<N;i++)CHECK(close(got[i],eo[i],&syn));for(size_t i=0;i<ST;i++){CHECK(close(gs[i],es[i],&syn));CHECK(unchanged[i]==src[i]);}
 /* Rejections happen before enqueue: short, exact/partial alias, zero/overflow. */
 CHECK(!glm53_kda_recurrent_f32(o2,N-1,d2,ST,q2,N,k2,N,v2,N,g2,N,b2,H,s2,ST,H,stream));CHECK(!glm53_kda_recurrent_f32(o2,N,s2,ST,q2,N,k2,N,v2,N,g2,N,b2,H,s2,ST,H,stream));CHECK(!glm53_kda_recurrent_f32(o2,N,d2,ST,q2,N,k2,N,v2,N,g2,N,b2,H,(char*)d2+4,ST,H,stream));CHECK(!glm53_kda_recurrent_f32(o2,N,d2,ST,q2,N,k2,N,v2,N,g2,N,b2,H,s2,ST,0,stream));CHECK(!glm53_kda_recurrent_f32(o2,SIZE_MAX,d2,ST,q2,N,k2,N,v2,N,g2,N,b2,H,s2,ST,H,stream));CHECK(!glm53_kda_recurrent_f32(o2,N,d2,ST,q2,N,k2,N,v2,N,g2,N,b2,H,s2,ST,SIZE_MAX,stream));
 std::printf("glm53 KDA GPU ops: PASS (%s, %s, non-default stream, fixture max out=%g state=%g)\n",prop.name,prop.gcnArchName,wo,ws);return 0;
}
