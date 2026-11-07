#include "glm53_static_bindings.h"
#include "glm53_architecture.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { glm53_weight_layer_role role; const char *suffix; } role_name;
static const char *const global_names[GLM53_WEIGHT_GLOBAL_COUNT] = {
 "lm_head.weight", "model.language_model.embed_tokens.weight",
 "model.language_model.norm.weight"
};
static const role_name common_roles[] = {
 {GLM53_ROLE_HC_ATTN_BASE,"hc_attn_base"},{GLM53_ROLE_HC_ATTN_SCALE,"hc_attn_scale"},
 {GLM53_ROLE_HC_FFN_BASE,"hc_ffn_base"},{GLM53_ROLE_HC_FFN_SCALE,"hc_ffn_scale"},
 {GLM53_ROLE_HC_ATTN_FN,"hc_attn_fn"},{GLM53_ROLE_HC_FFN_FN,"hc_ffn_fn"},
 {GLM53_ROLE_INPUT_NORM,"input_layernorm.weight"},
 {GLM53_ROLE_POST_ATTN_NORM,"post_attention_layernorm.weight"}
};
static const role_name dense_roles[] = {
 {GLM53_ROLE_MLP_DOWN,"mlp.down_proj.weight"},{GLM53_ROLE_MLP_DOWN_SCALE,"mlp.down_proj.weight_scale_inv"},
 {GLM53_ROLE_MLP_GATE,"mlp.gate_proj.weight"},{GLM53_ROLE_MLP_GATE_SCALE,"mlp.gate_proj.weight_scale_inv"},
 {GLM53_ROLE_MLP_UP,"mlp.up_proj.weight"},{GLM53_ROLE_MLP_UP_SCALE,"mlp.up_proj.weight_scale_inv"}
};
static const role_name routed_roles[] = {
 {GLM53_ROLE_ROUTER_BIAS,"mlp.gate.e_score_correction_bias"},{GLM53_ROLE_ROUTER_WEIGHT,"mlp.gate.weight"},
 {GLM53_ROLE_SHARED_DOWN,"mlp.shared_experts.down_proj.weight"},{GLM53_ROLE_SHARED_DOWN_SCALE,"mlp.shared_experts.down_proj.weight_scale_inv"},
 {GLM53_ROLE_SHARED_GATE,"mlp.shared_experts.gate_proj.weight"},{GLM53_ROLE_SHARED_GATE_SCALE,"mlp.shared_experts.gate_proj.weight_scale_inv"},
 {GLM53_ROLE_SHARED_UP,"mlp.shared_experts.up_proj.weight"},{GLM53_ROLE_SHARED_UP_SCALE,"mlp.shared_experts.up_proj.weight_scale_inv"}
};
static const role_name kda_roles[] = {
 {GLM53_ROLE_ATTN_A_LOG,"self_attn.A_log"},{GLM53_ROLE_ATTN_DT_BIAS,"self_attn.dt_bias"},
 {GLM53_ROLE_ATTN_B_PROJ,"self_attn.b_proj.weight"},{GLM53_ROLE_ATTN_F_A_PROJ,"self_attn.f_a_proj.weight"},
 {GLM53_ROLE_ATTN_F_B_PROJ,"self_attn.f_b_proj.weight"},{GLM53_ROLE_ATTN_G_A_PROJ,"self_attn.g_a_proj.weight"},
 {GLM53_ROLE_ATTN_G_B_PROJ,"self_attn.g_b_proj.weight"},{GLM53_ROLE_ATTN_K_CONV,"self_attn.k_conv1d.weight"},
 {GLM53_ROLE_ATTN_K_PROJ,"self_attn.k_proj.weight"},{GLM53_ROLE_ATTN_O_NORM,"self_attn.o_norm.weight"},
 {GLM53_ROLE_ATTN_O_PROJ,"self_attn.o_proj.weight"},{GLM53_ROLE_ATTN_Q_CONV,"self_attn.q_conv1d.weight"},
 {GLM53_ROLE_ATTN_Q_PROJ,"self_attn.q_proj.weight"},{GLM53_ROLE_ATTN_V_CONV,"self_attn.v_conv1d.weight"},
 {GLM53_ROLE_ATTN_V_PROJ,"self_attn.v_proj.weight"}
};
static const role_name dsa_roles[] = {
 {GLM53_ROLE_DSA_KV_A,"self_attn.kv_a_proj_with_mqa.weight"},{GLM53_ROLE_DSA_KV_A_SCALE,"self_attn.kv_a_proj_with_mqa.weight_scale_inv"},
 {GLM53_ROLE_DSA_KV_A_NORM,"self_attn.kv_a_layernorm.weight"},{GLM53_ROLE_DSA_KV_B,"self_attn.kv_b_proj.weight"},
 {GLM53_ROLE_DSA_O,"self_attn.o_proj.weight"},{GLM53_ROLE_DSA_O_SCALE,"self_attn.o_proj.weight_scale_inv"},
 {GLM53_ROLE_DSA_Q_A,"self_attn.q_a_proj.weight"},{GLM53_ROLE_DSA_Q_A_SCALE,"self_attn.q_a_proj.weight_scale_inv"},
 {GLM53_ROLE_DSA_Q_A_NORM,"self_attn.q_a_layernorm.weight"},{GLM53_ROLE_DSA_Q_B,"self_attn.q_b_proj.weight"},
 {GLM53_ROLE_DSA_Q_B_SCALE,"self_attn.q_b_proj.weight_scale_inv"},
 {GLM53_ROLE_INDEX_APE,"self_attn.indexer.index_kpool_compress_ape"},{GLM53_ROLE_INDEX_GATE,"self_attn.indexer.index_kpool_compress_gate"},
 {GLM53_ROLE_INDEX_K_NORM_BIAS,"self_attn.indexer.k_norm.bias"},{GLM53_ROLE_INDEX_K_NORM,"self_attn.indexer.k_norm.weight"},
 {GLM53_ROLE_INDEX_WEIGHTS,"self_attn.indexer.weights_proj.weight"},{GLM53_ROLE_INDEX_WK,"self_attn.indexer.wk.weight"},
 {GLM53_ROLE_INDEX_WQ_B,"self_attn.indexer.wq_b.weight"}
};

static void fail(char *e,size_t z,const char *fmt,...) { va_list ap; if(!e||!z)return; va_start(ap,fmt); (void)vsnprintf(e,z,fmt,ap); va_end(ap); }
static bool tensor_bytes(const k3_st_tensor *t) {
 uint64_t n=1,w; unsigned i;
 if(!t||!t->name||!t->name[0]||!t->ndim||t->ndim>K3_ST_MAX_DIMS)return false;
 w=t->dtype==K3_ST_DTYPE_F8_E4M3?1u:t->dtype==K3_ST_DTYPE_BF16?2u:t->dtype==K3_ST_DTYPE_F32?4u:0u;
 if(!w)return false;
 for(i=0;i<t->ndim;i++){if(!t->shape[i]||n>UINT64_MAX/t->shape[i])return false;n*=t->shape[i];}
 return n<=UINT64_MAX/w && n*w==t->byte_length;
}

typedef const glm53_static_runtime_entry *(*find_fn)(const void *,const char *);
typedef void *(*pointer_fn)(const void *,const glm53_static_runtime_entry *);
typedef struct { const void *context; find_fn find; pointer_fn pointer; size_t count; uint64_t bytes; } runtime_view;

static bool validate_layout(const glm53_static_layout *l,char *e,size_t z) {
 size_t i; uint64_t logical=0,max=0,end=0;
 if(!l||!l->built){fail(e,z,"unbuilt static layout");return false;}
 if(!l->entries||l->entry_count!=GLM53_STATIC_BINDING_COUNT||l->tensor_count!=l->entry_count){fail(e,z,"static layout entry count mismatch");return false;}
 for(i=0;i<l->entry_count;i++){
  const glm53_static_layout_entry *x=&l->entries[i]; glm53_weight_class c;
  if(!x->tensor||!tensor_bytes(x->tensor)||x->dtype!=x->tensor->dtype||x->logical_bytes!=x->tensor->byte_length||
     x->source_shard!=x->tensor->shard||x->source_physical_offset!=x->tensor->physical_offset||
     x->device_offset%GLM53_STATIC_LAYOUT_ALIGNMENT||
     (i&&strcmp(l->entries[i-1].tensor->name,x->tensor->name)>=0)||
     !glm53_weight_classify(x->tensor,&c)||c==GLM53_WEIGHT_STREAMED_ROUTED_EXPERT){fail(e,z,"invalid static layout entry %zu",i);return false;}
  if(UINT64_MAX-logical<x->logical_bytes){fail(e,z,"static layout overflow");return false;} logical+=x->logical_bytes;
  if(x->logical_bytes>max)max=x->logical_bytes;
  if(x->device_offset>UINT64_MAX-x->logical_bytes){fail(e,z,"static layout extent overflow");return false;}
  end=x->device_offset+x->logical_bytes;
  if(i+1<l->entry_count&&end>l->entries[i+1].device_offset){fail(e,z,"overlapping static layout entries");return false;}
 }
 if(logical!=GLM53_WEIGHT_ENGINE_RESIDENT_BYTES||logical!=l->logical_bytes||max!=l->max_tensor_bytes||
    l->padded_bytes<end||l->padded_bytes-end>=GLM53_STATIC_LAYOUT_ALIGNMENT){fail(e,z,"static layout ledger mismatch");return false;}
 return true;
}

static glm53_static_bindings_status bind_one(glm53_static_binding *b,const k3_st_tensor *t,
 const glm53_static_layout *layout,const runtime_view *v,glm53_static_binding_kind kind,
 uint16_t layer,uint16_t expert,uint16_t role,char *e,size_t z) {
 const glm53_static_layout_entry *le; const glm53_static_runtime_entry *re; void *p; glm53_weight_class c; unsigned i;
 if(!t||!glm53_architecture_validate_main_tensor(t,NULL)||!tensor_bytes(t)||!glm53_weight_classify(t,&c)||c==GLM53_WEIGHT_STREAMED_ROUTED_EXPERT){fail(e,z,"bad expected tensor metadata");return GLM53_STATIC_BINDINGS_BAD_PLAN;}
 le=glm53_static_layout_find_name(layout,t->name);
 if(!le||le->tensor!=t||le->dtype!=t->dtype||le->logical_bytes!=t->byte_length){fail(e,z,"layout mismatch for %s",t->name);return GLM53_STATIC_BINDINGS_BAD_LAYOUT;}
 re=v->find(v->context,t->name);
 if(!re){fail(e,z,"missing runtime tensor %s",t->name);return GLM53_STATIC_BINDINGS_MISSING_TENSOR;}
 if(!re->name||strcmp(re->name,t->name)||re->dtype!=t->dtype||re->logical_bytes!=t->byte_length||re->device_offset!=le->device_offset){fail(e,z,"runtime metadata mismatch for %s",t->name);return GLM53_STATIC_BINDINGS_BAD_STORE;}
 p=v->pointer(v->context,re); if(!p){fail(e,z,"invalid runtime pointer for %s",t->name);return GLM53_STATIC_BINDINGS_BAD_STORE;}
 memset(b,0,sizeof(*b)); b->runtime=re;b->device=p;b->kind=kind;b->dtype=t->dtype;b->ndim=t->ndim;b->logical_bytes=t->byte_length;b->layer=layer;b->expert=expert;b->role=role;
 for(i=0;i<t->ndim;i++) b->shape[i]=t->shape[i];
 return GLM53_STATIC_BINDINGS_OK;
}
static bool exact_name(const k3_st_tensor *t,const char *want,char *e,size_t z){if(!t||!t->name||strcmp(t->name,want)){fail(e,z,"wrong tensor for role %s",want);return false;}return true;}

static glm53_static_bindings_status bind_table(glm53_static_bindings *b,const glm53_weight_plan *p,const glm53_static_layout *l,const runtime_view *v,unsigned layer,const role_name *table,size_t n,char *e,size_t z){
 size_t i; char name[192]; glm53_static_bindings_status s;
 for(i=0;i<n;i++){(void)snprintf(name,sizeof(name),"model.language_model.layers.%u.%s",layer,table[i].suffix); if(!exact_name(p->layers[layer].roles[table[i].role],name,e,z))return GLM53_STATIC_BINDINGS_BAD_PLAN; s=bind_one(&b->layers[layer][table[i].role],p->layers[layer].roles[table[i].role],l,v,GLM53_STATIC_BINDING_LAYER,(uint16_t)layer,UINT16_MAX,(uint16_t)table[i].role,e,z);if(s)return s;b->resident_static_count++;}
 return GLM53_STATIC_BINDINGS_OK;
}

static glm53_static_bindings_status build_common(glm53_static_bindings *out,const glm53_weight_plan *p,const glm53_static_layout *l,const runtime_view *v,char *e,size_t z){
 glm53_static_bindings b; glm53_static_bindings_status s; size_t i,j; char name[256]; static const glm53_expert_tensor_role eroles[3]={GLM53_EXPERT_DOWN_SCALE,GLM53_EXPERT_GATE_SCALE,GLM53_EXPERT_UP_SCALE}; static const char *const enames[3]={"down_proj.weight_scale_inv","gate_proj.weight_scale_inv","up_proj.weight_scale_inv"};
 memset(&b,0,sizeof(b)); if(e&&z)e[0]='\0';
 if(!out||!p||!l||!v){fail(e,z,"invalid argument");return GLM53_STATIC_BINDINGS_INVALID_ARGUMENT;}
 if(out->built){fail(e,z,"output bindings already built");return GLM53_STATIC_BINDINGS_INVALID_ARGUMENT;}
 if(!p->built){fail(e,z,"unbuilt weight plan");return GLM53_STATIC_BINDINGS_UNBUILT_INPUT;}
 if(p->resident_static.tensor_count!=GLM53_WEIGHT_RESIDENT_STATIC_COUNT||p->resident_static.total_bytes!=GLM53_WEIGHT_RESIDENT_STATIC_BYTES||p->resident_routed_scales.tensor_count!=GLM53_WEIGHT_ROUTED_SCALE_COUNT||p->resident_routed_scales.total_bytes!=GLM53_WEIGHT_ROUTED_SCALE_BYTES||p->routed_experts.expert_count!=GLM53_EXPERT_LAYER_COUNT * GLM53_EXPERTS_PER_LAYER||!p->routed_experts.experts){fail(e,z,"weight plan ledger mismatch");return GLM53_STATIC_BINDINGS_BAD_PLAN;}
 if(!validate_layout(l,e,z))return GLM53_STATIC_BINDINGS_BAD_LAYOUT;
 if(v->count!=l->entry_count||v->bytes!=l->padded_bytes){fail(e,z,"runtime store ledger mismatch");return GLM53_STATIC_BINDINGS_BAD_STORE;}
 b.routed_scales=(glm53_static_binding*)calloc(GLM53_WEIGHT_ROUTED_SCALE_COUNT,sizeof(*b.routed_scales));if(!b.routed_scales){fail(e,z,"allocation failed");return GLM53_STATIC_BINDINGS_ALLOCATION_FAILED;}
 for(i=0;i<GLM53_WEIGHT_GLOBAL_COUNT;i++){if(!exact_name(p->globals[i],global_names[i],e,z)){s=GLM53_STATIC_BINDINGS_BAD_PLAN;goto bad;}s=bind_one(&b.globals[i],p->globals[i],l,v,GLM53_STATIC_BINDING_GLOBAL,UINT16_MAX,UINT16_MAX,(uint16_t)i,e,z);if(s)goto bad;b.resident_static_count++;}
 for(i=0;i<GLM53_WEIGHT_LAYER_COUNT;i++){
  glm53_weight_layer_kind want=(glm53_weight_layer_kind)glm53_architecture_layer_kind((uint32_t)i);if(p->layers[i].kind!=want){fail(e,z,"wrong layer kind %zu",i);s=GLM53_STATIC_BINDINGS_BAD_PLAN;goto bad;}
  s=bind_table(&b,p,l,v,(unsigned)i,common_roles,sizeof(common_roles)/sizeof(common_roles[0]),e,z);if(s)goto bad;
  if(i<3)s=bind_table(&b,p,l,v,(unsigned)i,dense_roles,sizeof(dense_roles)/sizeof(dense_roles[0]),e,z);else s=bind_table(&b,p,l,v,(unsigned)i,routed_roles,sizeof(routed_roles)/sizeof(routed_roles[0]),e,z);if(s)goto bad;
  s=bind_table(&b,p,l,v,(unsigned)i,want==GLM53_WEIGHT_LAYER_ROUTED_DSA?dsa_roles:kda_roles,want==GLM53_WEIGHT_LAYER_ROUTED_DSA?sizeof(dsa_roles)/sizeof(dsa_roles[0]):sizeof(kda_roles)/sizeof(kda_roles[0]),e,z);if(s)goto bad;
 }
 if(b.resident_static_count!=GLM53_WEIGHT_RESIDENT_STATIC_COUNT){fail(e,z,"resident role count mismatch");s=GLM53_STATIC_BINDINGS_BAD_PLAN;goto bad;}
 for(i=0;i<GLM53_EXPERT_COUNT;i++){
  const glm53_expert_plan *xp=&p->routed_experts.experts[i]; unsigned layer=GLM53_EXPERT_FIRST_LAYER+(unsigned)(i/GLM53_EXPERTS_PER_LAYER),expert=(unsigned)(i%GLM53_EXPERTS_PER_LAYER);
  if(xp->layer!=layer||xp->expert!=expert){fail(e,z,"expert identity mismatch %zu",i);s=GLM53_STATIC_BINDINGS_BAD_PLAN;goto bad;}
  for(j=0;j<3;j++){size_t ix=i*3u+j;(void)snprintf(name,sizeof(name),"model.language_model.layers.%u.mlp.experts.%u.%s",layer,expert,enames[j]);if(!exact_name(xp->tensors[eroles[j]],name,e,z)){s=GLM53_STATIC_BINDINGS_BAD_PLAN;goto bad;}s=bind_one(&b.routed_scales[ix],xp->tensors[eroles[j]],l,v,GLM53_STATIC_BINDING_ROUTED_SCALE,(uint16_t)layer,(uint16_t)expert,(uint16_t)j,e,z);if(s)goto bad;b.routed_scale_count++;}
 }
 if(b.routed_scale_count!=GLM53_WEIGHT_ROUTED_SCALE_COUNT){s=GLM53_STATIC_BINDINGS_BAD_PLAN;goto bad;} b.binding_count=b.resident_static_count+b.routed_scale_count;b.built=true;*out=b;return GLM53_STATIC_BINDINGS_OK;
bad: free(b.routed_scales);return s;
}

static const glm53_static_runtime_entry *real_find(const void *c,const char *n){return glm53_static_store_find((const glm53_static_store*)c,n);} static void *real_ptr(const void *c,const glm53_static_runtime_entry *e){return glm53_static_store_device_pointer((const glm53_static_store*)c,e);}
glm53_static_bindings_status glm53_static_bindings_build(glm53_static_bindings *o,const glm53_weight_plan *p,const glm53_static_layout *l,const glm53_static_store *st,char *e,size_t z){runtime_view v;if(!st){fail(e,z,"invalid store");return GLM53_STATIC_BINDINGS_INVALID_ARGUMENT;}v.context=st;v.find=real_find;v.pointer=real_ptr;v.count=glm53_static_store_entry_count(st);v.bytes=glm53_static_store_device_bytes(st);return build_common(o,p,l,&v,e,z);}

typedef struct { const glm53_static_runtime_entry **refs;size_t count;void *base;uint64_t bytes; } mock_context;
static int refcmp(const void *a,const void *b){const glm53_static_runtime_entry *const*x=(const glm53_static_runtime_entry*const*)a,*const*y=(const glm53_static_runtime_entry*const*)b;return strcmp((*x)->name,(*y)->name);} static const glm53_static_runtime_entry *mock_find(const void *c,const char*n){const mock_context*m=(const mock_context*)c;size_t lo=0,hi=m->count;while(lo<hi){size_t x=lo+(hi-lo)/2;int q=strcmp(m->refs[x]->name,n);if(q<0)lo=x+1;else hi=x;}return lo<m->count&&!strcmp(m->refs[lo]->name,n)?m->refs[lo]:NULL;} static void *mock_ptr(const void*c,const glm53_static_runtime_entry*e){const mock_context*m=(const mock_context*)c;if(!m->base||e->device_offset>m->bytes||e->logical_bytes>m->bytes-e->device_offset)return NULL;return (void*)((uintptr_t)m->base+(uintptr_t)e->device_offset);}
glm53_static_bindings_status glm53_static_bindings_build_mock(glm53_static_bindings*o,const glm53_weight_plan*p,const glm53_static_layout*l,const glm53_static_runtime_entry*es,size_t n,void*base,uint64_t bytes,char*e,size_t z){mock_context m;runtime_view v;size_t i;glm53_static_bindings_status s;uintptr_t base_address;if(!es||!n||!base){fail(e,z,"invalid mock store");return GLM53_STATIC_BINDINGS_INVALID_ARGUMENT;}if(n!=GLM53_STATIC_BINDING_COUNT||n>SIZE_MAX/sizeof(*m.refs)){fail(e,z,"mock store entry count mismatch");return GLM53_STATIC_BINDINGS_BAD_STORE;}base_address=(uintptr_t)base;if(bytes>(uint64_t)UINTPTR_MAX||base_address>UINTPTR_MAX-(uintptr_t)bytes){fail(e,z,"mock device span address overflow");return GLM53_STATIC_BINDINGS_BAD_STORE;}m.refs=(const glm53_static_runtime_entry**)malloc(n*sizeof(*m.refs));if(!m.refs){fail(e,z,"allocation failed");return GLM53_STATIC_BINDINGS_ALLOCATION_FAILED;}m.count=n;m.base=base;m.bytes=bytes;for(i=0;i<n;i++){if(!es[i].name||!es[i].name[0]){free(m.refs);fail(e,z,"empty runtime name");return GLM53_STATIC_BINDINGS_BAD_STORE;}m.refs[i]=&es[i];}qsort(m.refs,n,sizeof(*m.refs),refcmp);for(i=1;i<n;i++)if(!strcmp(m.refs[i-1]->name,m.refs[i]->name)){free(m.refs);fail(e,z,"duplicate runtime tensor %s",m.refs[i]->name);return GLM53_STATIC_BINDINGS_DUPLICATE_TENSOR;}v.context=&m;v.find=mock_find;v.pointer=mock_ptr;v.count=n;v.bytes=bytes;s=build_common(o,p,l,&v,e,z);free(m.refs);return s;}

const glm53_static_binding *glm53_static_bindings_global(const glm53_static_bindings*b,glm53_global_role r){return b&&b->built&&(unsigned)r<GLM53_WEIGHT_GLOBAL_COUNT?&b->globals[r]:NULL;} const glm53_static_binding *glm53_static_bindings_layer(const glm53_static_bindings*b,uint32_t l,glm53_weight_layer_role r){const glm53_static_binding*x;if(!b||!b->built||l>=GLM53_WEIGHT_LAYER_COUNT||(unsigned)r>=GLM53_WEIGHT_LAYER_ROLE_COUNT)return NULL;x=&b->layers[l][r];return x->runtime?x:NULL;} const glm53_static_binding *glm53_static_bindings_expert_scale(const glm53_static_bindings*b,uint32_t l,uint32_t x,glm53_routed_scale_role r){size_t i;if(!b||!b->built||l<GLM53_EXPERT_FIRST_LAYER||l>GLM53_EXPERT_LAST_LAYER||x>=GLM53_EXPERTS_PER_LAYER||(unsigned)r>=3)return NULL;i=((size_t)(l-GLM53_EXPERT_FIRST_LAYER)*GLM53_EXPERTS_PER_LAYER+x)*3u+(unsigned)r;return &b->routed_scales[i];} const glm53_static_binding *glm53_static_bindings_expert_tensor_scale(const glm53_static_bindings*b,uint32_t l,uint32_t x,glm53_expert_tensor_role r){glm53_routed_scale_role q;if(r==GLM53_EXPERT_DOWN_SCALE)q=GLM53_ROUTED_SCALE_DOWN;else if(r==GLM53_EXPERT_GATE_SCALE)q=GLM53_ROUTED_SCALE_GATE;else if(r==GLM53_EXPERT_UP_SCALE)q=GLM53_ROUTED_SCALE_UP;else return NULL;return glm53_static_bindings_expert_scale(b,l,x,q);} void glm53_static_bindings_free(glm53_static_bindings*b){if(!b)return;free(b->routed_scales);memset(b,0,sizeof(*b));}
const char *glm53_static_bindings_status_string(glm53_static_bindings_status s){switch(s){case GLM53_STATIC_BINDINGS_OK:return "ok";case GLM53_STATIC_BINDINGS_INVALID_ARGUMENT:return "invalid argument";case GLM53_STATIC_BINDINGS_UNBUILT_INPUT:return "unbuilt input";case GLM53_STATIC_BINDINGS_BAD_PLAN:return "bad plan";case GLM53_STATIC_BINDINGS_BAD_LAYOUT:return "bad layout";case GLM53_STATIC_BINDINGS_BAD_STORE:return "bad store";case GLM53_STATIC_BINDINGS_MISSING_TENSOR:return "missing tensor";case GLM53_STATIC_BINDINGS_DUPLICATE_TENSOR:return "duplicate tensor";case GLM53_STATIC_BINDINGS_ALLOCATION_FAILED:return "allocation failed";default:return "unknown status";}}
