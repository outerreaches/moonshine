#ifndef GLM53_ENGINE_PLAN_H
#define GLM53_ENGINE_PLAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_ENGINE_GIB UINT64_C(1073741824)
#define GLM53_ENGINE_PLAN_MAX_CONTEXT 512u
#define GLM53_ENGINE_PLAN_DEFAULT_CONTEXT 512u
#define GLM53_ENGINE_PLAN_HARD_CEILING_BYTES (UINT64_C(124) * GLM53_ENGINE_GIB)
#define GLM53_ENGINE_PLAN_MIN_MEM_AVAILABLE_BYTES (UINT64_C(100) * GLM53_ENGINE_GIB)
#define GLM53_ENGINE_PLAN_MIN_ABORT_RESERVE_BYTES (UINT64_C(16) * GLM53_ENGINE_GIB)
#define GLM53_ENGINE_PLAN_MAPPED_STAGING_BYTES UINT64_C(33562624)

typedef enum {
    GLM53_ENGINE_PLAN_OK = 0,
    GLM53_ENGINE_PLAN_INVALID_ARGUMENT,
    GLM53_ENGINE_PLAN_UNSUPPORTED_PROFILE,
    GLM53_ENGINE_PLAN_OVERFLOW,
    GLM53_ENGINE_PLAN_MEM_AVAILABLE_POLICY,
    GLM53_ENGINE_PLAN_ABORT_RESERVE_POLICY,
    GLM53_ENGINE_PLAN_HARD_CEILING
} glm53_engine_plan_status;

/* This is the only input copied from the weights planner. peak_transient_bytes
 * is scratch needed while constructing the resident static tier; it is not
 * assumed to remain live during inference. */
typedef struct {
    uint64_t resident_bytes;
    uint64_t peak_transient_bytes;
} glm53_engine_static_ledger;

typedef struct {
    uint32_t context_tokens;
    uint32_t batch_size;
    uint32_t expert_cache_slots_per_layer;
    uint32_t mapped_staging_slots;
    uint64_t compute_workspace_bytes;
    uint64_t library_workspace_bytes;
    uint64_t allocator_guard_bytes;
    uint64_t runtime_guard_bytes;
    uint64_t host_guard_bytes;
} glm53_engine_plan_config;

typedef struct {
    uint32_t context_tokens;
    uint32_t batch_size;
    uint32_t expert_cache_slots_per_layer;
    uint32_t mapped_staging_slots;
    uint64_t static_resident_bytes;
    uint64_t static_peak_transient_bytes;
    uint64_t expert_resident_bytes;
    uint64_t kda_state_bytes;
    uint64_t kda_conv_bytes;
    uint64_t dsa_compact_bytes;
    uint64_t mapped_staging_bytes;
    uint64_t compute_workspace_bytes;
    uint64_t library_workspace_bytes;
    uint64_t allocator_guard_bytes;
    uint64_t runtime_guard_bytes;
    uint64_t host_guard_bytes;
    uint64_t runtime_bytes_before_guards;
    uint64_t load_bytes_before_guards;
    uint64_t peak_bytes_before_guards;
    uint64_t admitted_bytes;
    uint64_t abort_reserve_bytes;
} glm53_engine_plan_report;

/* Initialize the batch-1, sequential direct-ID proof profile. Expert cache
 * slots remain explicit so Phase 6 can revise the API; Phase 5A admits zero. */
void glm53_engine_plan_config_init(glm53_engine_plan_config *config);

/* Pure checked-arithmetic admission. mem_available_bytes is a policy input,
 * not sampled here. report is modified only after complete success. */
glm53_engine_plan_status glm53_engine_plan_build(
    const glm53_engine_plan_config *config,
    const glm53_engine_static_ledger *static_ledger,
    uint64_t mem_available_bytes,
    glm53_engine_plan_report *report);

const char *glm53_engine_plan_status_string(glm53_engine_plan_status status);

#ifdef __cplusplus
}
#endif
#endif
