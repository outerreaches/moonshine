#include "glm53_architecture.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *suffix;
    k3_st_dtype dtype;
    uint8_t ndim;
    uint64_t shape[5];
} contract;

#define C1(n,d,a) {n,d,1,{a,0,0,0,0}}
#define C2(n,d,a,b) {n,d,2,{a,b,0,0,0}}
#define C3(n,d,a,b,c) {n,d,3,{a,b,c,0,0}}
#define C4(n,d,a,b,c,e) {n,d,4,{a,b,c,e,0}}
#define C5(n,d,a,b,c,e,f) {n,d,5,{a,b,c,e,f}}

static const contract globals[] = {
 C2("lm_head.weight",K3_ST_DTYPE_BF16,154880,4096),
 C2("model.language_model.embed_tokens.weight",K3_ST_DTYPE_BF16,154880,4096),
 C1("model.language_model.norm.weight",K3_ST_DTYPE_BF16,4096),
};
static const contract common[] = {
 C1("hc_attn_base",K3_ST_DTYPE_F32,24), C1("hc_attn_scale",K3_ST_DTYPE_F32,3),
 C1("hc_ffn_base",K3_ST_DTYPE_F32,24), C1("hc_ffn_scale",K3_ST_DTYPE_F32,3),
 C2("hc_attn_fn",K3_ST_DTYPE_BF16,24,16384), C2("hc_ffn_fn",K3_ST_DTYPE_BF16,24,16384),
 C1("input_layernorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("post_attention_layernorm.weight",K3_ST_DTYPE_BF16,4096),
};
static const contract dense[] = {
 C2("mlp.down_proj.weight",K3_ST_DTYPE_F8_E4M3,4096,12288),
 C2("mlp.down_proj.weight_scale_inv",K3_ST_DTYPE_F32,32,96),
 C2("mlp.gate_proj.weight",K3_ST_DTYPE_F8_E4M3,12288,4096),
 C2("mlp.gate_proj.weight_scale_inv",K3_ST_DTYPE_F32,96,32),
 C2("mlp.up_proj.weight",K3_ST_DTYPE_F8_E4M3,12288,4096),
 C2("mlp.up_proj.weight_scale_inv",K3_ST_DTYPE_F32,96,32),
};
static const contract routed[] = {
 C1("mlp.gate.e_score_correction_bias",K3_ST_DTYPE_F32,288),
 C2("mlp.gate.weight",K3_ST_DTYPE_BF16,288,4096),
 C2("mlp.shared_experts.down_proj.weight",K3_ST_DTYPE_F8_E4M3,4096,2048),
 C2("mlp.shared_experts.down_proj.weight_scale_inv",K3_ST_DTYPE_F32,32,16),
 C2("mlp.shared_experts.gate_proj.weight",K3_ST_DTYPE_F8_E4M3,2048,4096),
 C2("mlp.shared_experts.gate_proj.weight_scale_inv",K3_ST_DTYPE_F32,16,32),
 C2("mlp.shared_experts.up_proj.weight",K3_ST_DTYPE_F8_E4M3,2048,4096),
 C2("mlp.shared_experts.up_proj.weight_scale_inv",K3_ST_DTYPE_F32,16,32),
};
static const contract expert_proj[] = {
 C2("down_proj.weight",K3_ST_DTYPE_F8_E4M3,4096,2048),
 C2("down_proj.weight_scale_inv",K3_ST_DTYPE_F32,32,16),
 C2("gate_proj.weight",K3_ST_DTYPE_F8_E4M3,2048,4096),
 C2("gate_proj.weight_scale_inv",K3_ST_DTYPE_F32,16,32),
 C2("up_proj.weight",K3_ST_DTYPE_F8_E4M3,2048,4096),
 C2("up_proj.weight_scale_inv",K3_ST_DTYPE_F32,16,32),
};
static const contract kda[] = {
 C1("self_attn.A_log",K3_ST_DTYPE_F32,64), C1("self_attn.dt_bias",K3_ST_DTYPE_F32,8192),
 C2("self_attn.b_proj.weight",K3_ST_DTYPE_BF16,64,4096),
 C2("self_attn.f_a_proj.weight",K3_ST_DTYPE_BF16,128,4096),
 C2("self_attn.f_b_proj.weight",K3_ST_DTYPE_BF16,8192,128),
 C2("self_attn.g_a_proj.weight",K3_ST_DTYPE_BF16,128,4096),
 C2("self_attn.g_b_proj.weight",K3_ST_DTYPE_BF16,8192,128),
 C3("self_attn.k_conv1d.weight",K3_ST_DTYPE_BF16,8192,1,4),
 C2("self_attn.k_proj.weight",K3_ST_DTYPE_BF16,8192,4096),
 C1("self_attn.o_norm.weight",K3_ST_DTYPE_BF16,128),
 C2("self_attn.o_proj.weight",K3_ST_DTYPE_BF16,4096,8192),
 C3("self_attn.q_conv1d.weight",K3_ST_DTYPE_BF16,8192,1,4),
 C2("self_attn.q_proj.weight",K3_ST_DTYPE_BF16,8192,4096),
 C3("self_attn.v_conv1d.weight",K3_ST_DTYPE_BF16,8192,1,4),
 C2("self_attn.v_proj.weight",K3_ST_DTYPE_BF16,8192,4096),
};
static const contract mtp_extra[] = {
 C1("input_layernorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("post_attention_layernorm.weight",K3_ST_DTYPE_BF16,4096),
 C2("eh_proj.weight",K3_ST_DTYPE_BF16,4096,8192),
 C1("enorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("hnorm.weight",K3_ST_DTYPE_BF16,4096),
 C1("shared_head.norm.weight",K3_ST_DTYPE_BF16,4096),
};
static const contract vision_block[] = {
 C1("attn.k_norm.weight",K3_ST_DTYPE_BF16,64),
 C1("attn.proj.bias",K3_ST_DTYPE_BF16,1024),
 C2("attn.proj.weight",K3_ST_DTYPE_BF16,1024,1024),
 C1("attn.q_norm.weight",K3_ST_DTYPE_BF16,64),
 C1("attn.qkv.bias",K3_ST_DTYPE_BF16,3072),
 C2("attn.qkv.weight",K3_ST_DTYPE_BF16,3072,1024),
 C1("mlp.down_proj.bias",K3_ST_DTYPE_BF16,1024),
 C2("mlp.down_proj.weight",K3_ST_DTYPE_BF16,1024,4096),
 C1("mlp.gate_proj.bias",K3_ST_DTYPE_BF16,4096),
 C2("mlp.gate_proj.weight",K3_ST_DTYPE_BF16,4096,1024),
 C1("mlp.up_proj.bias",K3_ST_DTYPE_BF16,4096),
 C2("mlp.up_proj.weight",K3_ST_DTYPE_BF16,4096,1024),
 C1("norm1.weight",K3_ST_DTYPE_BF16,1024),
 C1("norm2.weight",K3_ST_DTYPE_BF16,1024),
};
static const contract vision_extra[] = {
 C1("downsample.bias",K3_ST_DTYPE_BF16,4096),
 C4("downsample.weight",K3_ST_DTYPE_BF16,4096,1024,2,2),
 C2("merger.down_proj.weight",K3_ST_DTYPE_BF16,4096,10240),
 C2("merger.gate_proj.weight",K3_ST_DTYPE_BF16,10240,4096),
 C2("merger.up_proj.weight",K3_ST_DTYPE_BF16,10240,4096),
 C1("merger.post_projection_norm.bias",K3_ST_DTYPE_BF16,4096),
 C1("merger.post_projection_norm.weight",K3_ST_DTYPE_BF16,4096),
 C2("merger.proj.weight",K3_ST_DTYPE_BF16,4096,4096),
 C1("patch_embed.proj.bias",K3_ST_DTYPE_BF16,1024),
 C5("patch_embed.proj.weight",K3_ST_DTYPE_BF16,1024,3,2,14,14),
 C1("post_layernorm.weight",K3_ST_DTYPE_BF16,1024),
};

static const contract dsa[] = {
 C2("self_attn.kv_a_proj_with_mqa.weight",K3_ST_DTYPE_F8_E4M3,512,4096),
 C2("self_attn.kv_a_proj_with_mqa.weight_scale_inv",K3_ST_DTYPE_F32,4,32),
 C1("self_attn.kv_a_layernorm.weight",K3_ST_DTYPE_BF16,512),
 C2("self_attn.kv_b_proj.weight",K3_ST_DTYPE_BF16,32768,512),
 C2("self_attn.o_proj.weight",K3_ST_DTYPE_F8_E4M3,4096,16384),
 C2("self_attn.o_proj.weight_scale_inv",K3_ST_DTYPE_F32,32,128),
 C2("self_attn.q_a_proj.weight",K3_ST_DTYPE_F8_E4M3,1536,4096),
 C2("self_attn.q_a_proj.weight_scale_inv",K3_ST_DTYPE_F32,12,32),
 C1("self_attn.q_a_layernorm.weight",K3_ST_DTYPE_BF16,1536),
 C2("self_attn.q_b_proj.weight",K3_ST_DTYPE_F8_E4M3,16384,1536),
 C2("self_attn.q_b_proj.weight_scale_inv",K3_ST_DTYPE_F32,128,12),
 C2("self_attn.indexer.index_kpool_compress_ape",K3_ST_DTYPE_BF16,4,128),
 C2("self_attn.indexer.index_kpool_compress_gate",K3_ST_DTYPE_BF16,128,4096),
 C1("self_attn.indexer.k_norm.bias",K3_ST_DTYPE_BF16,128),
 C1("self_attn.indexer.k_norm.weight",K3_ST_DTYPE_BF16,128),
 C2("self_attn.indexer.weights_proj.weight",K3_ST_DTYPE_BF16,32,4096),
 C2("self_attn.indexer.wk.weight",K3_ST_DTYPE_BF16,128,4096),
 C2("self_attn.indexer.wq_b.weight",K3_ST_DTYPE_BF16,4096,1536),
};

static void fail(char *e,size_t z,const char *fmt,...) { if(!e||!z)return; va_list ap;va_start(ap,fmt);vsnprintf(e,z,fmt,ap);va_end(ap); }
static bool tensor_is(const k3_st_tensor *t,const contract *c) {
 if (!t || t->dtype!=c->dtype || t->ndim!=c->ndim) return false;
 for(uint8_t i=0;i<c->ndim;i++) if(t->shape[i]!=c->shape[i]) return false;
 return true;
}
static const contract *find_contract(const contract *a,size_t n,const char *s) {
 for(size_t i=0;i<n;i++) { if(strcmp(a[i].suffix,s)==0)return &a[i]; }
 return NULL;
}
static bool decimal_component(const char **p,uint32_t *value) {
 const char *s=*p; if(*s<'0'||*s>'9')return false;
 if(*s=='0' && s[1]>='0'&&s[1]<='9')return false;
 uint32_t v=0; do { unsigned d=(unsigned)(*s-'0'); if(v>1000000u)return false;v=v*10u+d;s++; } while(*s>='0'&&*s<='9');
 if(*s!='.')return false;
 *p=s+1;*value=v;return true;
}

glm53_layer_kind glm53_architecture_layer_kind(uint32_t layer) {
 if(layer>=45u)return GLM53_LAYER_INVALID;
 if(layer<3u)return GLM53_LAYER_DENSE_KDA;
 return ((layer+1u)%4u)==0u ? GLM53_LAYER_ROUTED_DSA : GLM53_LAYER_ROUTED_KDA;
}

static bool main_contract(const k3_st_tensor *t,glm53_layer_kind *out) {
 if(!t||!t->name)return false;
 const contract *c=find_contract(globals,sizeof globals/sizeof globals[0],t->name);
 if(c){if(out)*out=GLM53_LAYER_INVALID;return tensor_is(t,c);}
 static const char prefix[]="model.language_model.layers.";
 if(strncmp(t->name,prefix,sizeof(prefix)-1u)!=0)return false;
 const char *s=t->name+sizeof(prefix)-1u;uint32_t layer;
 if(!decimal_component(&s,&layer)||layer>=45u)return false;
 glm53_layer_kind kind=glm53_architecture_layer_kind(layer);if(out)*out=kind;
 c=find_contract(common,sizeof common/sizeof common[0],s);
 if(!c && layer<3u)c=find_contract(dense,sizeof dense/sizeof dense[0],s);
 if(!c && layer>=3u)c=find_contract(routed,sizeof routed/sizeof routed[0],s);
 if(!c && layer>=3u && strncmp(s,"mlp.experts.",12u)==0) {
   const char *q=s+12u;uint32_t expert;
   if(decimal_component(&q,&expert)&&expert<288u)c=find_contract(expert_proj,sizeof expert_proj/sizeof expert_proj[0],q);
 }
 if(!c && kind!=GLM53_LAYER_ROUTED_DSA)c=find_contract(kda,sizeof kda/sizeof kda[0],s);
 if(!c && kind==GLM53_LAYER_ROUTED_DSA)c=find_contract(dsa,sizeof dsa/sizeof dsa[0],s);
 return c && tensor_is(t,c);
}

bool glm53_architecture_validate_main_tensor(const k3_st_tensor *t,glm53_layer_kind *kind) { return main_contract(t,kind); }

static bool require_one(const k3_st_model *m,const char *name,const contract *c,char *e,size_t z) {
 const k3_st_tensor *t=k3_st_find(m,name);
 if(!t){fail(e,z,"missing required tensor %s",name);return false;}
 if(!tensor_is(t,c)){fail(e,z,"wrong dtype or HF shape for %s",name);return false;}
 return true;
}
static bool require_table(const k3_st_model *m,uint32_t layer,const contract *a,size_t n,char *e,size_t z) {
 char name[192];for(size_t i=0;i<n;i++){snprintf(name,sizeof name,"model.language_model.layers.%u.%s",layer,a[i].suffix);if(!require_one(m,name,&a[i],e,z))return false;}return true;
}

static bool mtp_contract(const k3_st_tensor *t) {
 if(!t||!t->name)return false;
 static const char prefix[]="model.language_model.layers.45.";
 if(strncmp(t->name,prefix,sizeof(prefix)-1u)!=0)return false;
 const char *s=t->name+sizeof(prefix)-1u;
 const contract *c=find_contract(routed,sizeof routed/sizeof routed[0],s);
 if(!c)c=find_contract(dsa,sizeof dsa/sizeof dsa[0],s);
 if(!c)c=find_contract(mtp_extra,sizeof mtp_extra/sizeof mtp_extra[0],s);
 if(!c && strncmp(s,"mlp.experts.",12u)==0) {
  const char *q=s+12u;uint32_t expert;
  if(decimal_component(&q,&expert)&&expert<288u)
   c=find_contract(expert_proj,sizeof expert_proj/sizeof expert_proj[0],q);
 }
 return c&&tensor_is(t,c);
}

static bool vision_contract(const k3_st_tensor *t) {
 if(!t||!t->name)return false;
 static const char prefix[]="model.visual.";
 if(strncmp(t->name,prefix,sizeof(prefix)-1u)!=0)return false;
 const char *s=t->name+sizeof(prefix)-1u;
 const contract *c=find_contract(vision_extra,sizeof vision_extra/sizeof vision_extra[0],s);
 if(!c && strncmp(s,"blocks.",7u)==0) {
  const char *q=s+7u;uint32_t block;
  if(decimal_component(&q,&block)&&block<24u)
   c=find_contract(vision_block,sizeof vision_block/sizeof vision_block[0],q);
 }
 return c&&tensor_is(t,c);
}

bool glm53_architecture_validate_mtp_tensor(const k3_st_tensor *t) { return mtp_contract(t); }
bool glm53_architecture_validate_vision_tensor(const k3_st_tensor *t) { return vision_contract(t); }

static bool valid_model(const k3_st_model *m,char *e,size_t z) {
 if(!m||(!m->tensors&&m->tensor_count)){fail(e,z,"invalid model directory");return false;}
 return true;
}

bool glm53_architecture_validate_main(const k3_st_model *m,
                                      glm53_architecture_report *report,
                                      char *e,size_t z) {
 glm53_architecture_report r={0};if(report)*report=r;if(e&&z)e[0]='\0';
 if(!valid_model(m,e,z))return false;
 static const char layers[]="model.language_model.layers.";
 for(size_t i=0;i<m->tensor_count;i++) {
  const k3_st_tensor *t=&m->tensors[i];
  if(!t->name||!t->name[0]){fail(e,z,"empty tensor name");return false;}
  glm53_layer_kind kind=GLM53_LAYER_INVALID;
  if(main_contract(t,&kind)) {
   r.main_count++;
   if(kind==GLM53_LAYER_DENSE_KDA)r.dense_count++;
   else if(kind==GLM53_LAYER_ROUTED_KDA||kind==GLM53_LAYER_ROUTED_DSA)r.routed_count++;
   if(kind==GLM53_LAYER_DENSE_KDA||kind==GLM53_LAYER_ROUTED_KDA)r.kda_count++;
   else if(kind==GLM53_LAYER_ROUTED_DSA)r.dsa_count++;
   continue;
  }
  if(strncmp(t->name,layers,sizeof(layers)-1u)==0) {
   const char *s=t->name+sizeof(layers)-1u;uint32_t layer;
   if(decimal_component(&s,&layer)&&layer==45u)continue;
   fail(e,z,"unknown or malformed main tensor %s",t->name);return false;
  }
  if(strncmp(t->name,"model.language_model.",21u)==0||
     strncmp(t->name,"lm_head.",8u)==0) {
   fail(e,z,"unknown or malformed main tensor %s",t->name);return false;
  }
 }
 if(r.main_count!=GLM53_MAIN_TENSOR_COUNT){fail(e,z,"wrong main tensor count: %zu",r.main_count);return false;}
 for(size_t i=0;i<sizeof globals/sizeof globals[0];i++)
  if(!require_one(m,globals[i].suffix,&globals[i],e,z))return false;
 for(uint32_t l=0;l<45u;l++) {
  if(!require_table(m,l,common,sizeof common/sizeof common[0],e,z))return false;
  if(l<3u) {
   if(!require_table(m,l,dense,sizeof dense/sizeof dense[0],e,z))return false;
  } else {
   if(!require_table(m,l,routed,sizeof routed/sizeof routed[0],e,z))return false;
   char name[192];
   for(uint32_t x=0;x<288u;x++)for(size_t i=0;i<sizeof expert_proj/sizeof expert_proj[0];i++) {
    snprintf(name,sizeof name,"model.language_model.layers.%u.mlp.experts.%u.%s",l,x,expert_proj[i].suffix);
    if(!require_one(m,name,&expert_proj[i],e,z))return false;
   }
  }
  if(glm53_architecture_layer_kind(l)==GLM53_LAYER_ROUTED_DSA) {
   if(!require_table(m,l,dsa,sizeof dsa/sizeof dsa[0],e,z))return false;
  } else if(!require_table(m,l,kda,sizeof kda/sizeof kda[0],e,z))return false;
 }
 if(r.dense_count!=87u||r.routed_count!=73911u||r.kda_count!=54616u||r.dsa_count!=19382u) {
  fail(e,z,"internal layer partition count mismatch");return false;
 }
 if(report)*report=r;
 return true;
}

bool glm53_architecture_validate_mtp_metadata(const k3_st_model *m,size_t *count,
                                              char *e,size_t z) {
 size_t n=0;if(count)*count=0;if(e&&z)e[0]='\0';if(!valid_model(m,e,z))return false;
 static const char prefix[]="model.language_model.layers.45.";
 for(size_t i=0;i<m->tensor_count;i++) {
  const k3_st_tensor *t=&m->tensors[i];
  if(!t->name||!t->name[0]){fail(e,z,"empty tensor name");return false;}
  if(strncmp(t->name,prefix,sizeof(prefix)-1u)==0) {
   if(!mtp_contract(t)){fail(e,z,"unknown or malformed MTP tensor %s",t->name);return false;}n++;
  }
 }
 if(n!=GLM53_MTP_TENSOR_COUNT){fail(e,z,"wrong MTP tensor count: %zu",n);return false;}
 if(!require_table(m,45u,routed,sizeof routed/sizeof routed[0],e,z)||
    !require_table(m,45u,dsa,sizeof dsa/sizeof dsa[0],e,z)||
    !require_table(m,45u,mtp_extra,sizeof mtp_extra/sizeof mtp_extra[0],e,z))return false;
 char name[192];
 for(uint32_t x=0;x<288u;x++)for(size_t i=0;i<sizeof expert_proj/sizeof expert_proj[0];i++) {
  snprintf(name,sizeof name,"model.language_model.layers.45.mlp.experts.%u.%s",x,expert_proj[i].suffix);
  if(!require_one(m,name,&expert_proj[i],e,z))return false;
 }
 if(count)*count=n;
 return true;
}

bool glm53_architecture_validate_vision_metadata(const k3_st_model *m,size_t *count,
                                                 char *e,size_t z) {
 size_t n=0;if(count)*count=0;if(e&&z)e[0]='\0';if(!valid_model(m,e,z))return false;
 static const char prefix[]="model.visual.";
 for(size_t i=0;i<m->tensor_count;i++) {
  const k3_st_tensor *t=&m->tensors[i];
  if(!t->name||!t->name[0]){fail(e,z,"empty tensor name");return false;}
  if(strncmp(t->name,prefix,sizeof(prefix)-1u)==0) {
   if(!vision_contract(t)){fail(e,z,"unknown or malformed vision tensor %s",t->name);return false;}n++;
  }
 }
 if(n!=GLM53_VISION_TENSOR_COUNT){fail(e,z,"wrong vision tensor count: %zu",n);return false;}
 char name[160];
 for(uint32_t b=0;b<24u;b++)for(size_t i=0;i<sizeof vision_block/sizeof vision_block[0];i++) {
  snprintf(name,sizeof name,"model.visual.blocks.%u.%s",b,vision_block[i].suffix);
  if(!require_one(m,name,&vision_block[i],e,z))return false;
 }
 for(size_t i=0;i<sizeof vision_extra/sizeof vision_extra[0];i++) {
  snprintf(name,sizeof name,"model.visual.%s",vision_extra[i].suffix);
  if(!require_one(m,name,&vision_extra[i],e,z))return false;
 }
 if(count)*count=n;
 return true;
}

bool glm53_architecture_validate(const k3_st_model *m,glm53_architecture_report *report,char *e,size_t z) {
 glm53_architecture_report r={0};size_t mtp=0,vision=0;if(report)*report=r;if(e&&z)e[0]='\0';
 if(!valid_model(m,e,z))return false;
 for(size_t i=0;i<m->tensor_count;i++) {
  const k3_st_tensor *t=&m->tensors[i];
  if(!t->name||!t->name[0]){fail(e,z,"empty tensor name");return false;}
  if(i&&strcmp(m->tensors[i-1].name,t->name)>=0){fail(e,z,"duplicate or unsorted tensor name %s",t->name);return false;}
  if(!main_contract(t,NULL)&&!mtp_contract(t)&&!vision_contract(t)) {
   fail(e,z,"unknown or malformed tensor %s",t->name);return false;
  }
 }
 if(!glm53_architecture_validate_main(m,&r,e,z))return false;
 if(!glm53_architecture_validate_mtp_metadata(m,&mtp,e,z))return false;
 if(!glm53_architecture_validate_vision_metadata(m,&vision,e,z))return false;
 r.mtp_count=mtp;r.vision_count=vision;if(report)*report=r;return true;
}
