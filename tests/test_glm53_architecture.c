#include "glm53_architecture.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static k3_st_tensor tensor5(const char *name,k3_st_dtype dtype,uint8_t ndim,
                            uint64_t a,uint64_t b,uint64_t c,uint64_t d,uint64_t e) {
 k3_st_tensor t;memset(&t,0,sizeof t);t.name=(char *)name;t.dtype=dtype;t.ndim=ndim;
 t.shape[0]=a;t.shape[1]=b;t.shape[2]=c;t.shape[3]=d;t.shape[4]=e;return t;
}
static k3_st_tensor tensor(const char *name,k3_st_dtype dtype,uint8_t ndim,
                           uint64_t a,uint64_t b,uint64_t c) {
 return tensor5(name,dtype,ndim,a,b,c,0,0);
}
int main(void) {
 assert(glm53_architecture_layer_kind(0)==GLM53_LAYER_DENSE_KDA);
 assert(glm53_architecture_layer_kind(2)==GLM53_LAYER_DENSE_KDA);
 assert(glm53_architecture_layer_kind(3)==GLM53_LAYER_ROUTED_DSA);
 assert(glm53_architecture_layer_kind(34)==GLM53_LAYER_ROUTED_KDA);
 assert(glm53_architecture_layer_kind(35)==GLM53_LAYER_ROUTED_DSA);
 assert(glm53_architecture_layer_kind(36)==GLM53_LAYER_ROUTED_KDA);
 assert(glm53_architecture_layer_kind(43)==GLM53_LAYER_ROUTED_DSA);
 assert(glm53_architecture_layer_kind(44)==GLM53_LAYER_ROUTED_KDA);
 assert(glm53_architecture_layer_kind(45)==GLM53_LAYER_INVALID);

 k3_st_tensor t=tensor("model.language_model.layers.44.mlp.experts.287.down_proj.weight",
                       K3_ST_DTYPE_F8_E4M3,2,4096,2048,0);
 assert(glm53_architecture_validate_main_tensor(&t,NULL));
 t.shape[0]=2048;t.shape[1]=4096; /* HF shape order is not a GEMM descriptor. */
 assert(!glm53_architecture_validate_main_tensor(&t,NULL));
 t=tensor("model.language_model.layers.44.mlp.experts.287.down_proj.weight_scale_inv",
          K3_ST_DTYPE_F32,2,32,16,0);
 assert(glm53_architecture_validate_main_tensor(&t,NULL));
 t.name="model.language_model.layers.44.mlp.experts.287.down_proj.scale";
 assert(!glm53_architecture_validate_main_tensor(&t,NULL));
 t=tensor("model.language_model.layers.35.self_attn.q_b_proj.weight",
          K3_ST_DTYPE_F8_E4M3,2,16384,1536,0);
 assert(glm53_architecture_validate_main_tensor(&t,NULL));
 t.name="model.language_model.layers.35.self_attn.q_proj.weight";
 assert(!glm53_architecture_validate_main_tensor(&t,NULL));
 t=tensor("model.language_model.layers.045.input_layernorm.weight",
          K3_ST_DTYPE_BF16,1,4096,0,0);
 assert(!glm53_architecture_validate_main_tensor(&t,NULL));
 t.name="model.language_model.layers.45.input_layernorm.weight";
 assert(!glm53_architecture_validate_main_tensor(&t,NULL));
 t.name="model.language_model.layers.44.unknown.weight";
 assert(!glm53_architecture_validate_main_tensor(&t,NULL));

 t=tensor("model.language_model.layers.45.mlp.experts.287.gate_proj.weight_scale_inv",
          K3_ST_DTYPE_F32,2,16,32,0);
 assert(glm53_architecture_validate_mtp_tensor(&t));
 t.shape[1]=31;assert(!glm53_architecture_validate_mtp_tensor(&t));
 t=tensor("model.language_model.layers.045.input_layernorm.weight",
          K3_ST_DTYPE_BF16,1,4096,0,0);
 assert(!glm53_architecture_validate_mtp_tensor(&t));

 t=tensor5("model.visual.patch_embed.proj.weight",K3_ST_DTYPE_BF16,5,
           1024,3,2,14,14);
 assert(glm53_architecture_validate_vision_tensor(&t));
 t.shape[4]=13;assert(!glm53_architecture_validate_vision_tensor(&t));
 t=tensor("model.visual.blocks.23.attn.proj.weight",K3_ST_DTYPE_BF16,2,
          1024,1024,0);
 assert(glm53_architecture_validate_vision_tensor(&t));
 t.name="model.visual.blocks.24.attn.proj.weight";
 assert(!glm53_architecture_validate_vision_tensor(&t));

 k3_st_model empty;memset(&empty,0,sizeof empty);
 glm53_architecture_report report;memset(&report,0x7f,sizeof report);
 char error[128];
 assert(!glm53_architecture_validate_main(&empty,&report,error,sizeof error));
 glm53_architecture_report zero={0};assert(memcmp(&report,&zero,sizeof zero)==0);
 memset(&report,0x7f,sizeof report);
 assert(!glm53_architecture_validate(&empty,&report,error,sizeof error));
 assert(memcmp(&report,&zero,sizeof zero)==0);
 size_t count=123;
 assert(!glm53_architecture_validate_mtp_metadata(&empty,&count,error,sizeof error));
 assert(count==0);
 count=123;
 assert(!glm53_architecture_validate_vision_metadata(&empty,&count,error,sizeof error));
 assert(count==0);
 puts("glm53 architecture tests: ok");return 0;
}
