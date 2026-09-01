#include "glm53_weights.h"
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

static void fail(char *e, size_t z, const char *fmt, ...) {
    va_list ap; if (!e || z == 0u) return; va_start(ap, fmt);
    (void)vsnprintf(e, z, fmt, ap); va_end(ap);
}
static bool add64(uint64_t *a, uint64_t b) {
    if (UINT64_MAX - *a < b) return false;
    *a += b;
    return true;
}
static int ptr_name_compare(const void *a, const void *b) {
    const k3_st_tensor *const *x = (const k3_st_tensor *const *)a;
    const k3_st_tensor *const *y = (const k3_st_tensor *const *)b;
    return strcmp((*x)->name, (*y)->name);
}
static bool expert_name(const char *name, bool *scale) {
    static const char p[] = "model.language_model.layers.";
    const char *s, *x; unsigned long layer, expert; char *end;
    if (!name || strncmp(name, p, sizeof(p)-1u) != 0) return false;
    s = name + sizeof(p)-1u; layer = strtoul(s, &end, 10);
    if (end == s || layer < 3ul || layer > 44ul ||
        strncmp(end, ".mlp.experts.", 13u) != 0) return false;
    s = end + 13u; expert = strtoul(s, &end, 10);
    if (end == s || expert >= 288ul || *end != '.') return false;
    x = end + 1u;
    if (strcmp(x,"down_proj.weight") && strcmp(x,"gate_proj.weight") &&
        strcmp(x,"up_proj.weight") && strcmp(x,"down_proj.weight_scale_inv") &&
        strcmp(x,"gate_proj.weight_scale_inv") && strcmp(x,"up_proj.weight_scale_inv")) return false;
    *scale = strstr(x, "_scale_inv") != NULL; return true;
}

bool glm53_weight_classify(const k3_st_tensor *t, glm53_weight_class *out) {
    glm53_layer_kind kind; bool scale = false;
    if (!t || !glm53_architecture_validate_main_tensor(t, &kind)) return false;
    (void)kind;
    if (expert_name(t->name, &scale)) {
        if (out) *out = scale ? GLM53_WEIGHT_RESIDENT_ROUTED_SCALE :
                               GLM53_WEIGHT_STREAMED_ROUTED_EXPERT;
    } else if (out) *out = GLM53_WEIGHT_RESIDENT_STATIC;
    return true;
}

static bool tensor_bytes(const k3_st_tensor *t, uint64_t *bytes) {
    uint64_t n = 1u, width; uint8_t i;
    if (!t || t->ndim == 0u || t->ndim > K3_ST_MAX_DIMS) return false;
    width = t->dtype == K3_ST_DTYPE_F8_E4M3 ? 1u :
            (t->dtype == K3_ST_DTYPE_F32 ? 4u :
             (t->dtype == K3_ST_DTYPE_BF16 ? 2u : 0u));
    if (!width) return false;
    for (i=0u;i<t->ndim;i++) {
        if (!t->shape[i] || n > UINT64_MAX/t->shape[i]) return false;
        n *= t->shape[i];
    }
    if (n > UINT64_MAX/width) return false;
    n *= width;
    if (n != t->byte_length) return false;
    *bytes=n;
    return true;
}
static bool ledger_add(glm53_weight_ledger *l, const k3_st_tensor *t,
                       uint64_t n) {
    uint64_t *d = t->dtype == K3_ST_DTYPE_F8_E4M3 ? &l->f8_bytes :
                  (t->dtype == K3_ST_DTYPE_F32 ? &l->f32_bytes :
                                                &l->bf16_bytes);
    if (!add64(d,n) || !add64(&l->total_bytes,n) || l->tensor_count==SIZE_MAX)
        return false;
    l->tensor_count++; return true;
}
static bool bind_table(glm53_weight_layer *layer, const k3_st_model *m,
                       unsigned number, const role_name *table, size_t count,
                       char *e, size_t z) {
    size_t i; char name[192];
    for (i=0u;i<count;i++) {
        int n=snprintf(name,sizeof(name),"model.language_model.layers.%u.%s",
                       number,table[i].suffix);
        const k3_st_tensor *t;
        if (n<0 || (size_t)n>=sizeof(name) || !(t=k3_st_find(m,name))) {
            fail(e,z,"cannot bind layer %u role %s",number,table[i].suffix);
            return false;
        }
        if (layer->roles[table[i].role]) {
            fail(e,z,"duplicate layer role %s",table[i].suffix); return false;
        }
        layer->roles[table[i].role]=t;
    }
    return true;
}
static bool manifest_agrees(const glm53_manifest *mf, const k3_st_model *m,
                            char *e,size_t z) {
    size_t i;
    if (!mf || (!mf->entries && mf->entry_count)) {
        fail(e,z,"invalid manifest"); return false;
    }
    for (i=0u;i<mf->entry_count;i++) {
        if (!mf->entries[i].name || !mf->entries[i].name[0] ||
            (i && strcmp(mf->entries[i-1u].name,mf->entries[i].name)>=0)) {
            fail(e,z,"duplicate or unsorted manifest entry"); return false;
        }
    }
    for(i=0u;i<m->tensor_count;i++) {
        size_t lo=0u,hi=mf->entry_count;
        while(lo<hi) {
            size_t mid=lo+(hi-lo)/2u;
            int order=strcmp(mf->entries[mid].name,m->tensors[i].name);
            if(order<0)lo=mid+1u;else hi=mid;
        }
        if(lo==mf->entry_count || strcmp(mf->entries[lo].name,m->tensors[i].name) ||
           mf->entries[lo].shard!=m->tensors[i].shard) {
            fail(e,z,"manifest mismatch for %s",m->tensors[i].name);return false;
        }
    }
    return true;
}
static bool build(glm53_weight_plan *out,const glm53_manifest *mf,
                  const k3_st_model *m,char *e,size_t z) {
    glm53_weight_plan p; glm53_architecture_report report;
    const k3_st_tensor **sorted=NULL; size_t i; bool ok=false;
    memset(&p,0,sizeof(p)); if(e&&z)e[0]='\0';
    if(!out||!m||(!m->tensors&&m->tensor_count)) {fail(e,z,"invalid argument");return false;}
    if(m->tensor_count) {
        sorted=(const k3_st_tensor **)malloc(m->tensor_count*sizeof(*sorted));
        if(!sorted){fail(e,z,"allocation failed");return false;}
    }
    for(i=0u;i<m->tensor_count;i++) {
        if(!m->tensors[i].name||!m->tensors[i].name[0]){fail(e,z,"empty tensor name");goto done;}
        sorted[i]=&m->tensors[i];
    }
    qsort(sorted,m->tensor_count,sizeof(*sorted),ptr_name_compare);
    for(i=1u;i<m->tensor_count;i++) if(!strcmp(sorted[i-1u]->name,sorted[i]->name)) {
        fail(e,z,"duplicate tensor %s",sorted[i]->name);goto done;
    }
    if(mf&&!manifest_agrees(mf,m,e,z))goto done;
    if(!glm53_architecture_validate_main(m,&report,e,z))goto done;
    if(report.main_count!=m->tensor_count) {
        fail(e,z,"MTP, vision, or unknown tensor present");goto done;
    }
    for(i=0u;i<GLM53_WEIGHT_GLOBAL_COUNT;i++) {
        p.globals[i]=k3_st_find(m,global_names[i]);
        if(!p.globals[i]){fail(e,z,"missing global %s",global_names[i]);goto done;}
    }
    for(i=0u;i<GLM53_WEIGHT_LAYER_COUNT;i++) {
        glm53_weight_layer *l=&p.layers[i]; l->kind=(glm53_weight_layer_kind)glm53_architecture_layer_kind((uint32_t)i);
        if(!bind_table(l,m,(unsigned)i,common_roles,sizeof(common_roles)/sizeof(common_roles[0]),e,z))goto done;
        if(i<3u) { if(!bind_table(l,m,(unsigned)i,dense_roles,sizeof(dense_roles)/sizeof(dense_roles[0]),e,z))goto done; }
        else { if(!bind_table(l,m,(unsigned)i,routed_roles,sizeof(routed_roles)/sizeof(routed_roles[0]),e,z))goto done; }
        if(l->kind==GLM53_WEIGHT_LAYER_ROUTED_DSA) {
            if(!bind_table(l,m,(unsigned)i,dsa_roles,sizeof(dsa_roles)/sizeof(dsa_roles[0]),e,z))goto done;
        } else if(!bind_table(l,m,(unsigned)i,kda_roles,sizeof(kda_roles)/sizeof(kda_roles[0]),e,z))goto done;
    }
    if(glm53_expert_model_plan_build(&p.routed_experts,m,e,z)!=GLM53_EXPERT_PLAN_OK)goto done;
    for(i=0u;i<m->tensor_count;i++) {
        const k3_st_tensor *t=&m->tensors[i]; glm53_weight_class c; uint64_t n;
        glm53_weight_ledger *l;
        if(!glm53_weight_classify(t,&c)||!tensor_bytes(t,&n)) {
            fail(e,z,"bad dtype byte length for %s",t->name);goto done;
        }
        l=c==GLM53_WEIGHT_RESIDENT_STATIC?&p.resident_static:
          (c==GLM53_WEIGHT_STREAMED_ROUTED_EXPERT?&p.streamed_routed_experts:
                                                   &p.resident_routed_scales);
        if(!ledger_add(l,t,n)||!ledger_add(&p.total,t,n)){fail(e,z,"byte ledger overflow");goto done;}
    }
    if(p.total.f8_bytes!=GLM53_WEIGHT_MAIN_F8_BYTES || p.total.f32_bytes!=GLM53_WEIGHT_MAIN_F32_BYTES ||
       p.total.bf16_bytes!=GLM53_WEIGHT_MAIN_BF16_BYTES || p.total.total_bytes!=GLM53_WEIGHT_MAIN_BYTES ||
       p.streamed_routed_experts.total_bytes!=GLM53_WEIGHT_STREAMED_BYTES ||
       p.streamed_routed_experts.tensor_count!=GLM53_WEIGHT_STREAMED_COUNT ||
       p.resident_routed_scales.total_bytes!=GLM53_WEIGHT_ROUTED_SCALE_BYTES ||
       p.resident_routed_scales.tensor_count!=GLM53_WEIGHT_ROUTED_SCALE_COUNT ||
       p.resident_static.total_bytes!=GLM53_WEIGHT_RESIDENT_STATIC_BYTES ||
       p.resident_static.tensor_count!=GLM53_WEIGHT_RESIDENT_STATIC_COUNT) {
        fail(e,z,"official main-text byte ledger mismatch");goto done;
    }
    *out=p; memset(&p,0,sizeof(p)); ok=true;
done:
    free(sorted); glm53_expert_model_plan_free(&p.routed_experts); return ok;
}

bool glm53_weight_plan_build(glm53_weight_plan *p,const k3_st_model *m,
                             char *e,size_t z){return build(p,NULL,m,e,z);}
bool glm53_weight_plan_build_manifest(glm53_weight_plan *p,const glm53_manifest *mf,
                                      const k3_st_model *m,char *e,size_t z){return build(p,mf,m,e,z);}
void glm53_weight_plan_free(glm53_weight_plan *p){if(!p)return;glm53_expert_model_plan_free(&p->routed_experts);memset(p,0,sizeof(*p));}
