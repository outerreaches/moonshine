#include "glm53_arch_math.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c); goto done; } } while (0)
#define READ(a,n) CHECK(fread((a),sizeof(*(a)),(n),f)==(n))
static bool close_abs(float a,float b,float tol){ return isfinite(a)&&isfinite(b)&&fabsf(a-b)<=tol; }

static float streams[4*4096], mixv[24], basev[24], scalev[3];
static float want_pre[4], want_post[4], want_comb[16], want_collapsed[4096];
static float branchv[4096], want_postout[4*4096], inputv[4096];
static float logits[288], biasv[288], want_router_weights[8];
static uint32_t want_router_indices[8];
static float gatev[2048], upv[2048], want_activation[2048];
static float got_pre[4], got_post[4], got_comb[16], got_collapsed[4096];
static float got_postout[4*4096], got_router_weights[8], got_activation[2048];

int main(int argc,char **argv){
 FILE *f=NULL; int rc=1, extra; size_t i; unsigned char magic[8];
 uint32_t version,endian,width,mix_count,experts,topk,intermediate; float margin;
 char revision[40],transformers[40],llama[40],config_sha[64],index_sha[64];
 size_t got_router_indices[8];
 const uint32_t pinned_top[8]={103,124,96,83,236,25,166,279};
 if(argc!=2){fprintf(stderr,"usage: %s FIXTURE_BIN\n",argv[0]);return 2;}
 f=fopen(argv[1],"rb"); CHECK(f!=NULL);
 READ(magic,8); READ(&version,1); READ(&endian,1); READ(&width,1); READ(&mix_count,1);
 READ(&experts,1); READ(&topk,1); READ(&intermediate,1); READ(&margin,1);
 READ(revision,40); READ(transformers,40); READ(llama,40); READ(config_sha,64); READ(index_sha,64);
 CHECK(memcmp(magic,"G53P4CMP",8)==0 && version==1 && endian==UINT32_C(0x01020304));
 CHECK(width==4096 && mix_count==24 && experts==288 && topk==8 && intermediate==2048);
 CHECK(memcmp(revision,"04c4e9e95c5da8862dced7e5056455116f83a7e0",40)==0);
 CHECK(memcmp(transformers,"eb4d9e2a64a013bec12289288b85d0b1210ba0aa",40)==0);
 CHECK(memcmp(llama,"a771613af20f3dc60247e4b6a3d11516f0664673",40)==0);
 CHECK(memcmp(config_sha,"bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",64)==0);
 CHECK(memcmp(index_sha,"3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05",64)==0);
 CHECK(margin>0.017f);
 READ(streams,4*4096); READ(mixv,24); READ(basev,24); READ(scalev,3);
 READ(want_pre,4); READ(want_post,4); READ(want_comb,16); READ(want_collapsed,4096);
 READ(branchv,4096); READ(want_postout,4*4096); READ(inputv,4096); READ(logits,288); READ(biasv,288);
 READ(want_router_indices,8); READ(want_router_weights,8); READ(gatev,2048); READ(upv,2048); READ(want_activation,2048);
 extra=fgetc(f); CHECK(extra==EOF);
 /* Actual official projections are fixture inputs. mHC fn projection includes
  * pinned unweighted flattened F32 RMSNorm (eps=1e-6). Anchors catch drift. */
 CHECK(close_abs(mixv[0],5.26120520f,1e-6f) && close_abs(mixv[23],3.05740142f,1e-6f));
 CHECK(close_abs(logits[0],0.144367993f,1e-7f) && close_abs(logits[287],-0.423026234f,1e-7f));
 CHECK(close_abs(gatev[0],-0.218653336f,1e-7f) && close_abs(upv[2047],0.359194845f,1e-7f));
 CHECK(glm53_mhc_weights4_f32(got_pre,got_post,got_comb,mixv,basev,scalev,1e-6f)==GLM53_ARCH_MATH_OK);
 for(i=0;i<4;i++){CHECK(close_abs(got_pre[i],want_pre[i],8e-6f));CHECK(close_abs(got_post[i],want_post[i],8e-6f));}
 for(i=0;i<16;i++) CHECK(close_abs(got_comb[i],want_comb[i],8e-6f));
 CHECK(glm53_mhc_collapse4_f32(got_collapsed,streams,4096,got_pre)==GLM53_ARCH_MATH_OK);
 for(i=0;i<4096;i++) CHECK(close_abs(got_collapsed[i],want_collapsed[i],3e-5f));
 CHECK(glm53_mhc_post_apply4_f32(got_postout,streams,branchv,4096,got_post,got_comb)==GLM53_ARCH_MATH_OK);
 for(i=0;i<4*4096;i++) CHECK(close_abs(got_postout[i],want_postout[i],3e-5f));
 CHECK(glm53_router_top8_288_f32(got_router_indices,got_router_weights,logits,biasv,2.5f)==GLM53_ARCH_MATH_OK);
 for(i=0;i<8;i++){CHECK(want_router_indices[i]==pinned_top[i]);CHECK(got_router_indices[i]==want_router_indices[i]);CHECK(close_abs(got_router_weights[i],want_router_weights[i],2e-6f));}
 CHECK(glm53_limited_swiglu_f32(got_activation,gatev,upv,2048,10.0f)==GLM53_ARCH_MATH_OK);
 for(i=0;i<2048;i++) CHECK(close_abs(got_activation[i],want_activation[i],3e-5f));
 puts("glm53 official component fixture tests: ok"); rc=0;
done: if(f) fclose(f); return rc;
}
