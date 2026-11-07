#include "../glm53_static_bindings.h"
#include "../glm53_architecture.h"
#include "../glm53_manifest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); return 1; } } while (0)

static int test_small_contract(void) {
    glm53_static_bindings b;
    CHECK(GLM53_EXPERT_COUNT == 12096u);
    CHECK(GLM53_ROUTED_EXPERTS_PER_LAYER == 288u);
    glm53_static_binding *scales;
    memset(&b, 0, sizeof(b));
    CHECK(strcmp(glm53_static_bindings_status_string(GLM53_STATIC_BINDINGS_OK), "ok") == 0);
    CHECK(glm53_static_bindings_global(&b, GLM53_GLOBAL_LM_HEAD) == NULL);
    b.built = true;
    b.globals[GLM53_GLOBAL_LM_HEAD].runtime = (const glm53_static_runtime_entry *)(uintptr_t)1u;
    CHECK(glm53_static_bindings_global(&b, GLM53_GLOBAL_LM_HEAD) == &b.globals[0]);
    CHECK(glm53_static_bindings_layer(&b, 45u, GLM53_ROLE_INPUT_NORM) == NULL);
    scales = (glm53_static_binding *)calloc(GLM53_WEIGHT_ROUTED_SCALE_COUNT, sizeof(*scales));
    CHECK(scales != NULL);
    b.routed_scales = scales;
    CHECK(glm53_static_bindings_expert_scale(&b, 3u, 0u, GLM53_ROUTED_SCALE_DOWN) == scales);
    CHECK(glm53_static_bindings_expert_tensor_scale(&b, 3u, 0u, GLM53_EXPERT_DOWN_WEIGHT) == NULL);
    glm53_static_bindings_free(&b);
    CHECK(!b.built && b.routed_scales == NULL);
    return 0;
}

/* This is a CPU-only integration test.  Runtime records are synthesized from
 * the layout, so no HIP allocation or payload read occurs. */
static int optional_official(void) {
    const char *root = getenv("GLM53_OFFICIAL_ROOT");
    k3_st_model all, main;
    glm53_manifest manifest;
    glm53_weight_plan plan;
    glm53_static_layout layout;
    glm53_static_bindings bindings, before, candidate, candidate_before;
    glm53_static_runtime_entry *runtime = NULL;
    char error[512];
    size_t i, count = 0;
    if (!root || !root[0]) return 0;
    memset(&all,0,sizeof(all)); memset(&main,0,sizeof(main));
    memset(&manifest,0,sizeof(manifest)); memset(&plan,0,sizeof(plan));
    memset(&layout,0,sizeof(layout)); memset(&bindings,0,sizeof(bindings));
    memset(&candidate,0,sizeof(candidate));
    CHECK(glm53_manifest_load(&manifest,root,error,sizeof(error)));
    CHECK(k3_st_model_open_5digit_total(&all,root,GLM53_SHARD_COUNT,error,sizeof(error)));
    main.tensors=(k3_st_tensor*)calloc(GLM53_MAIN_TENSOR_COUNT,sizeof(*main.tensors)); CHECK(main.tensors);
    main.shards=all.shards;main.shard_count=all.shard_count;main.routed_span=all.routed_span;main.routed_span_context=all.routed_span_context;
    for(i=0;i<all.tensor_count;i++) if(glm53_architecture_validate_main_tensor(&all.tensors[i],NULL)) main.tensors[count++]=all.tensors[i];
    main.tensor_count=count;main.tensor_capacity=count;CHECK(count==GLM53_MAIN_TENSOR_COUNT);
    CHECK(glm53_weight_plan_build_manifest(&plan,&manifest,&main,error,sizeof(error)));
    CHECK(glm53_static_layout_build(&layout,&plan,error,sizeof(error))==GLM53_STATIC_LAYOUT_OK);
    runtime=(glm53_static_runtime_entry*)calloc(layout.entry_count,sizeof(*runtime));CHECK(runtime);
    for(i=0;i<layout.entry_count;i++){runtime[i].name=layout.entries[i].tensor->name;runtime[i].dtype=layout.entries[i].dtype;runtime[i].device_offset=layout.entries[i].device_offset;runtime[i].logical_bytes=layout.entries[i].logical_bytes;}
    {
        glm53_static_bindings_status status = glm53_static_bindings_build_mock(
            &bindings, &plan, &layout, runtime, layout.entry_count,
            (void *)(uintptr_t)0x10000u, layout.padded_bytes,
            error, sizeof(error));
        if (status != GLM53_STATIC_BINDINGS_OK)
            fprintf(stderr, "official mock binding failed: %s: %s\n",
                    glm53_static_bindings_status_string(status), error);
        CHECK(status == GLM53_STATIC_BINDINGS_OK);
    }
    CHECK(bindings.binding_count==GLM53_STATIC_BINDING_COUNT);
    CHECK(glm53_static_bindings_global(&bindings,GLM53_GLOBAL_FINAL_NORM)->ndim>0);
    CHECK(glm53_static_bindings_layer(&bindings,44u,GLM53_ROLE_INPUT_NORM)!=NULL);
    CHECK(glm53_static_bindings_expert_scale(&bindings,44u,287u,GLM53_ROUTED_SCALE_UP)!=NULL);

    before=bindings;
    CHECK(glm53_static_bindings_build_mock(&bindings,&plan,&layout,runtime,
          layout.entry_count,(void*)(uintptr_t)0x10000u,layout.padded_bytes,
          error,sizeof(error))==GLM53_STATIC_BINDINGS_INVALID_ARGUMENT);
    CHECK(!memcmp(&bindings,&before,sizeof(bindings)));

    candidate_before=candidate;
    {
        k3_st_tensor *bad_tensor = (k3_st_tensor *)plan.globals[GLM53_GLOBAL_LM_HEAD];
        char *saved_name = bad_tensor->name;
        bad_tensor->name = NULL;
        CHECK(glm53_static_bindings_build_mock(&candidate,&plan,&layout,runtime,
              layout.entry_count,(void*)(uintptr_t)0x10000u,layout.padded_bytes,
              error,sizeof(error))==GLM53_STATIC_BINDINGS_BAD_LAYOUT);
        CHECK(!memcmp(&candidate,&candidate_before,sizeof(candidate)));
        bad_tensor->name = saved_name;
    }

    runtime[0].logical_bytes++;
    CHECK(glm53_static_bindings_build_mock(&candidate,&plan,&layout,runtime,layout.entry_count,(void*)(uintptr_t)0x10000u,layout.padded_bytes,error,sizeof(error))==GLM53_STATIC_BINDINGS_BAD_STORE);
    CHECK(!memcmp(&candidate,&candidate_before,sizeof(candidate))); runtime[0].logical_bytes--;
    runtime[1].name=runtime[0].name;
    CHECK(glm53_static_bindings_build_mock(&candidate,&plan,&layout,runtime,layout.entry_count,(void*)(uintptr_t)0x10000u,layout.padded_bytes,error,sizeof(error))==GLM53_STATIC_BINDINGS_DUPLICATE_TENSOR);
    CHECK(!memcmp(&candidate,&candidate_before,sizeof(candidate)));
    runtime[1].name=layout.entries[1].tensor->name;
    CHECK(glm53_static_bindings_build_mock(&candidate,&plan,&layout,runtime,
          SIZE_MAX,(void*)(uintptr_t)0x10000u,layout.padded_bytes,
          error,sizeof(error))==GLM53_STATIC_BINDINGS_BAD_STORE);
    CHECK(glm53_static_bindings_build_mock(&candidate,&plan,&layout,runtime,
          layout.entry_count,(void*)(uintptr_t)(UINTPTR_MAX-15u),
          layout.padded_bytes,error,sizeof(error))==GLM53_STATIC_BINDINGS_BAD_STORE);
    CHECK(!memcmp(&candidate,&candidate_before,sizeof(candidate)));

    glm53_static_bindings_free(&bindings);free(runtime);glm53_static_layout_free(&layout);glm53_weight_plan_free(&plan);free(main.tensors);k3_st_model_close(&all);glm53_manifest_free(&manifest);
    return 0;
}
int main(void){CHECK(test_small_contract()==0);CHECK(optional_official()==0);puts("glm53 static bindings: all tests passed");return 0;}
