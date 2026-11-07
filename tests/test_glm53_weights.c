#include "../glm53_weights.h"
#include "../glm53_architecture.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static k3_st_tensor matrix(char *name,k3_st_dtype dtype,uint64_t a,uint64_t b) {
    k3_st_tensor t; memset(&t,0,sizeof(t)); t.name=name; t.dtype=dtype;
    t.ndim=2u; t.shape[0]=a; t.shape[1]=b;
    t.byte_length=a*b*(dtype==K3_ST_DTYPE_F32?4u:(dtype==K3_ST_DTYPE_BF16?2u:1u));
    return t;
}

static void test_classifier(void) {
    glm53_weight_class c;
    k3_st_tensor t=matrix("model.language_model.layers.3.mlp.experts.0.down_proj.weight",
                          K3_ST_DTYPE_F8_E4M3,4096u,2048u);
    assert(glm53_weight_classify(&t,&c));
    assert(c==GLM53_WEIGHT_STREAMED_ROUTED_EXPERT);
    t=matrix("model.language_model.layers.44.mlp.experts.287.up_proj.weight_scale_inv",
             K3_ST_DTYPE_F32,16u,32u);
    assert(glm53_weight_classify(&t,&c));
    assert(c==GLM53_WEIGHT_RESIDENT_ROUTED_SCALE);
    t=matrix("model.language_model.layers.3.mlp.shared_experts.up_proj.weight",
             K3_ST_DTYPE_F8_E4M3,2048u,4096u);
    assert(glm53_weight_classify(&t,&c)); assert(c==GLM53_WEIGHT_RESIDENT_STATIC);
    t=matrix("lm_head.weight",K3_ST_DTYPE_BF16,154880u,4096u);
    assert(glm53_weight_classify(&t,&c)); assert(c==GLM53_WEIGHT_RESIDENT_STATIC);

    t=matrix("model.language_model.layers.45.mlp.experts.0.down_proj.weight",
             K3_ST_DTYPE_F8_E4M3,4096u,2048u);
    assert(!glm53_weight_classify(&t,&c));
    t=matrix("model.visual.blocks.0.attn.proj.weight",K3_ST_DTYPE_BF16,1024u,1024u);
    assert(!glm53_weight_classify(&t,&c));
    t=matrix("model.language_model.layers.3.no_such.weight",K3_ST_DTYPE_BF16,1u,1u);
    assert(!glm53_weight_classify(&t,&c));
    t=matrix("model.language_model.layers.3.mlp.experts.0.down_proj.weight",
             K3_ST_DTYPE_F8_E4M3,2048u,4096u);
    assert(!glm53_weight_classify(&t,&c));
}

static void test_transactional_failure(void) {
    k3_st_tensor tensors[2]; k3_st_model model; glm53_weight_plan before,after;
    char error[128];
    tensors[0]=matrix("lm_head.weight",K3_ST_DTYPE_BF16,154880u,4096u);
    tensors[1]=tensors[0];
    memset(&model,0,sizeof(model)); model.tensors=tensors; model.tensor_count=2u;
    memset(&before,0xa5,sizeof(before)); after=before;
    assert(!glm53_weight_plan_build(&after,&model,error,sizeof(error)));
    assert(error[0]!='\0'); assert(memcmp(&after,&before,sizeof(after))==0);
}

static void test_engine_resident_helper(void) {
    glm53_weight_plan plan; uint64_t bytes=123u;
    memset(&plan,0,sizeof(plan));
    assert(!glm53_weight_plan_engine_resident_bytes(&plan,&bytes));
    assert(bytes==0u);
    assert(!glm53_weight_plan_engine_resident_bytes(NULL,&bytes));
    assert(bytes==0u);
    assert(!glm53_weight_plan_engine_resident_bytes(&plan,NULL));

    plan.built=true;
    plan.resident_static.tensor_count=GLM53_WEIGHT_RESIDENT_STATIC_COUNT;
    plan.resident_static.bf16_bytes=GLM53_WEIGHT_RESIDENT_STATIC_BYTES;
    plan.resident_static.total_bytes=GLM53_WEIGHT_RESIDENT_STATIC_BYTES;
    plan.resident_routed_scales.tensor_count=GLM53_WEIGHT_ROUTED_SCALE_COUNT;
    plan.resident_routed_scales.f32_bytes=GLM53_WEIGHT_ROUTED_SCALE_BYTES;
    plan.resident_routed_scales.total_bytes=GLM53_WEIGHT_ROUTED_SCALE_BYTES;
    assert(glm53_weight_plan_engine_resident_bytes(&plan,&bytes));
    assert(bytes==GLM53_WEIGHT_ENGINE_RESIDENT_BYTES);
    assert(bytes==UINT64_C(15300311288));
    plan.streamed_routed_experts.total_bytes=UINT64_MAX;
    assert(glm53_weight_plan_engine_resident_bytes(&plan,&bytes));
    assert(bytes==GLM53_WEIGHT_ENGINE_RESIDENT_BYTES);

    plan.resident_static.bf16_bytes=UINT64_MAX;
    plan.resident_static.total_bytes=UINT64_MAX;
    plan.resident_routed_scales.f32_bytes=1u;
    plan.resident_routed_scales.total_bytes=1u;
    bytes=123u;
    assert(!glm53_weight_plan_engine_resident_bytes(&plan,&bytes));
    assert(bytes==0u);
}

static void optional_official(void) {
    const char *root=getenv("GLM53_OFFICIAL_ROOT"); k3_st_model all,main;
    glm53_manifest manifest; glm53_weight_plan plan; char error[512]; size_t i,n=0u;
    if(!root||!root[0]) return;
    memset(&all,0,sizeof(all)); memset(&main,0,sizeof(main));
    memset(&manifest,0,sizeof(manifest)); memset(&plan,0,sizeof(plan));
    assert(glm53_manifest_load(&manifest,root,error,sizeof(error)));
    assert(k3_st_model_open_5digit_total(&all,root,GLM53_SHARD_COUNT,error,sizeof(error)));
    main.tensors=(k3_st_tensor *)calloc(GLM53_MAIN_TENSOR_COUNT,sizeof(*main.tensors));
    assert(main.tensors!=NULL); main.shards=all.shards; main.shard_count=all.shard_count;
    main.routed_span=all.routed_span; main.routed_span_context=all.routed_span_context;
    for(i=0u;i<all.tensor_count;i++) if(glm53_architecture_validate_main_tensor(&all.tensors[i],NULL))
        main.tensors[n++]=all.tensors[i];
    main.tensor_count=n; main.tensor_capacity=n; assert(n==GLM53_MAIN_TENSOR_COUNT);
    assert(glm53_weight_plan_build_manifest(&plan,&manifest,&main,error,sizeof(error)));
    assert(plan.total.total_bytes==GLM53_WEIGHT_MAIN_BYTES);
    { uint64_t resident=0u;
      assert(glm53_weight_plan_engine_resident_bytes(&plan,&resident));
      assert(resident==GLM53_WEIGHT_ENGINE_RESIDENT_BYTES); }
    assert(plan.routed_experts.expert_count==(size_t)GLM53_EXPERT_LAYER_COUNT*GLM53_EXPERTS_PER_LAYER);
    glm53_weight_plan_free(&plan); free(main.tensors);
    /* The shallow main view does not own shard or tensor-name storage. */
    k3_st_model_close(&all); glm53_manifest_free(&manifest);
}

int main(void) {
    test_classifier(); test_transactional_failure();
    test_engine_resident_helper(); optional_official();
    puts("glm53 weights: all tests passed"); return 0;
}
