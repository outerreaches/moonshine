#include "../glm53_state_oracle.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

static int closef(float a, float b) {
    float d = fabsf(a - b), s = fmaxf(fabsf(a), fabsf(b));
    return d <= 2.0e-6f + 2.0e-6f * s;
}

static int test_kda(void) {
    enum { T = 5, K = 3, V = 2 };
    const float q[T*K] = {
        .2f,-.4f,.7f, -.3f,.1f,.6f, .8f,-.2f,.05f,
        -.7f,.3f,.4f, .11f,.22f,-.33f };
    const float k[T*K] = {
        -.1f,.5f,.2f, .4f,-.6f,.3f, .7f,.2f,-.5f,
        .1f,.9f,-.2f, -.8f,.25f,.45f };
    const float v[T*V] = { .3f,-.2f, .8f,.1f, -.4f,.6f, .2f,.9f, -.7f,.5f };
    const float g[T*K] = {
        -.1f,-.7f,-.3f, -.8f,-.2f,-.5f, -.3f,-.9f,-.4f,
        -.6f,-.15f,-1.1f, -.25f,-.75f,-.45f };
    const float beta[T] = { .2f,.7f,.4f,.9f,.55f };
    const float initial[K*V] = { .1f,-.2f,.3f,.05f,-.15f,.25f };
    float ss[K*V], sc[K*V], split[K*V], os[T*V], oc[T*V], osp[T*V];
    size_t one_n = glm53_kda_scratch_floats(K,V,1);
    size_t all_n = glm53_kda_scratch_floats(K,V,T);
    float *one = (float *)malloc(one_n*sizeof(float));
    float *all = (float *)malloc(all_n*sizeof(float));
    CHECK(one && all && one_n && all_n);
    memcpy(ss, initial, sizeof(ss));
    for (size_t i=0;i<T;i++)
        CHECK(glm53_kda_step_f32(q+i*K,k+i*K,v+i*V,g+i*K,beta[i],
                                 ss,K,V,os+i*V,one,one_n));
    memcpy(sc, initial, sizeof(sc));
    CHECK(glm53_kda_chunk_f32(q,k,v,g,beta,T,sc,K,V,oc,all,all_n));
    CHECK(memcmp(ss,sc,sizeof(ss))==0);
    CHECK(memcmp(os,oc,sizeof(os))==0);

    memcpy(split,initial,sizeof(split));
    CHECK(glm53_kda_chunk_f32(q,k,v,g,beta,2,split,K,V,osp,all,all_n));
    CHECK(glm53_kda_chunk_f32(q+2*K,k+2*K,v+2*V,g+2*K,beta+2,T-2,
                              split,K,V,osp+2*V,all,all_n));
    CHECK(memcmp(ss,split,sizeof(ss))==0 && memcmp(os,osp,sizeof(os))==0);

    /* Per-key-channel g must not be accidentally treated as one scalar. */
    {
        float a[K*V] = {1,1,1,1,1,1}, b[K*V] = {1,1,1,1,1,1};
        float oa[V], ob[V], gz[K] = {g[0],g[0],g[0]};
        CHECK(glm53_kda_step_f32(q,k,v,g,0.0f,a,K,V,oa,one,one_n));
        CHECK(glm53_kda_step_f32(q,k,v,gz,0.0f,b,K,V,ob,one,one_n));
        CHECK(memcmp(a,b,sizeof(a)) != 0);
        CHECK(closef(a[2],expf(g[1])) && closef(a[4],expf(g[2])));
    }
    /* All validation happens before destination writes. */
    {
        float badq[K] = {NAN,0,0}, st[K*V], out[V] = {91,92};
        memcpy(st,initial,sizeof(st));
        CHECK(!glm53_kda_step_f32(badq,k,v,g,.5f,st,K,V,out,one,one_n));
        CHECK(memcmp(st,initial,sizeof(st))==0 && out[0]==91 && out[1]==92);
        CHECK(!glm53_kda_step_f32(q,k,v,g,.5f,st,K,V,out,one,one_n-1));
        CHECK(memcmp(st,initial,sizeof(st))==0 && out[0]==91 && out[1]==92);
    }
    free(all); free(one);
    return 0;
}

static int test_mla_layout(void) {
    size_t x=999;
    CHECK(glm53_mla_kv_b_key_index(2,3,2,4,1,2,3,&x));
    CHECK(x == ((1u*5u+2u)*4u+3u));
    CHECK(glm53_mla_kv_b_value_index(2,3,2,4,1,0,1,&x));
    CHECK(x == ((1u*5u+3u)*4u+1u));
    x=777;
    CHECK(!glm53_mla_kv_b_value_index(2,3,2,4,1,2,0,&x) && x==777);
    CHECK(!glm53_mla_kv_b_key_index(2,3,2,4,2,0,0,&x) && x==777);
    return 0;
}

static int boundary(size_t n, size_t pools_expected, size_t tail_expected) {
    uint8_t *valid=(uint8_t *)malloc(n ? n : 1);
    glm53_index_pool4 *pools=(glm53_index_pool4 *)malloc((pools_expected?pools_expected:1)*sizeof(*pools));
    size_t tail[3]={SIZE_MAX,SIZE_MAX,SIZE_MAX};
    glm53_index_pool4_metadata m;
    CHECK(valid && pools);
    memset(valid,1,n);
    CHECK(glm53_index_pool4_build(valid,n,n-1,pools,pools_expected,
                                   tail,3,&m));
    CHECK(m.first_valid==0 && m.pool_count==pools_expected && m.tail_count==tail_expected);
    if (pools_expected) {
        CHECK(pools[0].start==0 && pools[0].end==3 && pools[0].end_visible);
        CHECK(pools[pools_expected-1].end==pools_expected*4-1);
    }
    for (size_t i=0;i<tail_expected;i++) CHECK(tail[i]==pools_expected*4+i);
    free(pools); free(valid); return 0;
}

static int test_index_pool(void) {
    CHECK(boundary(3,0,3)==0); CHECK(boundary(4,1,0)==0);
    CHECK(boundary(5,1,1)==0); CHECK(boundary(2047,511,3)==0);
    CHECK(boundary(2048,512,0)==0); CHECK(boundary(2049,512,1)==0);
    {
        uint8_t v[8]={0,0,1,1,1,1,1,1};
        glm53_index_pool4 p[1]; size_t tail[3]; glm53_index_pool4_metadata m;
        CHECK(glm53_index_pool4_build(v,8,4,p,1,tail,3,&m));
        CHECK(m.first_valid==2 && m.pool_count==1 && p[0].start==2 && p[0].end==5);
        CHECK(!p[0].end_visible); /* a pool is visible at its final token */
        CHECK(m.tail_count==3 && tail[0]==2 && tail[2]==4);
        CHECK(glm53_index_pool4_build(v,8,5,p,1,tail,3,&m));
        CHECK(p[0].end_visible && m.tail_count==0);
    }
    {
        uint8_t invalid[4]={1,1,2,1}; glm53_index_pool4 p; size_t tail[3];
        glm53_index_pool4_metadata before={9,9,9}, m=before;
        CHECK(!glm53_index_pool4_build(invalid,4,3,&p,1,tail,3,&m));
        CHECK(memcmp(&m,&before,sizeof(m))==0);
    }
    return 0;
}

int main(void) {
    CHECK(test_kda()==0); CHECK(test_mla_layout()==0); CHECK(test_index_pool()==0);
    puts("glm53 state/layout CPU oracles: ok");
    return 0;
}
