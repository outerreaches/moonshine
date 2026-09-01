#include "glm53_engine_plan.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

/* Shape constants from the validated GLM-5.3-Flash architecture. */
#define GLM53_PLAN_KDA_LAYERS UINT64_C(34)
#define GLM53_PLAN_DSA_LAYERS UINT64_C(11)
#define GLM53_PLAN_KDA_HEADS UINT64_C(64)
#define GLM53_PLAN_KDA_KEY_DIM UINT64_C(128)
#define GLM53_PLAN_KDA_VALUE_DIM UINT64_C(128)
#define GLM53_PLAN_KDA_CHANNELS UINT64_C(8192)
#define GLM53_PLAN_KDA_CONV_WIDTH UINT64_C(4)
#define GLM53_PLAN_DSA_LATENT UINT64_C(512)
#define GLM53_PLAN_DSA_INDEX_KEY UINT64_C(128)
#define GLM53_PLAN_DSA_INDEX_GATE UINT64_C(128)
#define GLM53_PLAN_DSA_VALID UINT64_C(1)
#define GLM53_PLAN_BF16_BYTES UINT64_C(2)
#define GLM53_PLAN_F32_BYTES UINT64_C(4)
#define GLM53_PLAN_ALIGNMENT UINT64_C(4096)

static int add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int add_to(uint64_t *total, uint64_t value) {
    return add_u64(*total, value, total);
}

static int align_up_u64(uint64_t value, uint64_t alignment, uint64_t *out) {
    uint64_t remainder;
    uint64_t extra;
    if (alignment == 0u) return 0;
    remainder = value % alignment;
    extra = remainder == 0u ? 0u : alignment - remainder;
    return add_u64(value, extra, out);
}

void glm53_engine_plan_config_init(glm53_engine_plan_config *config) {
    if (config == NULL) return;
    memset(config, 0, sizeof(*config));
    config->context_tokens = GLM53_ENGINE_PLAN_DEFAULT_CONTEXT;
    config->batch_size = 1u;
    config->expert_cache_slots_per_layer = 0u;
    config->mapped_staging_slots = 1u;
    config->compute_workspace_bytes = UINT64_C(256) * UINT64_C(1048576);
    config->library_workspace_bytes = UINT64_C(64) * UINT64_C(1048576);
    config->allocator_guard_bytes = UINT64_C(256) * UINT64_C(1048576);
    config->runtime_guard_bytes = GLM53_ENGINE_GIB;
    config->host_guard_bytes = UINT64_C(4) * GLM53_ENGINE_GIB;
}

glm53_engine_plan_status glm53_engine_plan_build(
    const glm53_engine_plan_config *config,
    const glm53_engine_static_ledger *static_ledger,
    uint64_t mem_available_bytes,
    glm53_engine_plan_report *report) {
    glm53_engine_plan_report next;
    uint64_t value;
    uint64_t per_token;
    uint64_t aligned_expert;
    uint64_t guards = 0u;

    if (config == NULL || static_ledger == NULL || report == NULL ||
        static_ledger->resident_bytes == 0u || config->context_tokens == 0u ||
        config->mapped_staging_slots == 0u) {
        return GLM53_ENGINE_PLAN_INVALID_ARGUMENT;
    }
    /* Phase 5A intentionally admits only the proof profile. */
    if (config->batch_size != 1u ||
        config->context_tokens > GLM53_ENGINE_PLAN_MAX_CONTEXT ||
        config->expert_cache_slots_per_layer != 0u ||
        config->mapped_staging_slots != 1u) {
        return GLM53_ENGINE_PLAN_UNSUPPORTED_PROFILE;
    }
    memset(&next, 0, sizeof(next));
    next.context_tokens = config->context_tokens;
    next.batch_size = config->batch_size;
    next.expert_cache_slots_per_layer = config->expert_cache_slots_per_layer;
    next.mapped_staging_slots = config->mapped_staging_slots;
    next.static_resident_bytes = static_ledger->resident_bytes;
    next.static_peak_transient_bytes = static_ledger->peak_transient_bytes;

    next.expert_resident_bytes = 0u; /* Routed cache begins in Phase 6. */

    /* Preserve recurrent products as binary32. This is deliberately more
     * conservative than a possible future mixed-precision state proof. */
    if (!mul_u64(GLM53_PLAN_KDA_HEADS, GLM53_PLAN_KDA_KEY_DIM, &value) ||
        !mul_u64(value, GLM53_PLAN_KDA_VALUE_DIM, &value) ||
        !mul_u64(value, GLM53_PLAN_F32_BYTES, &value) ||
        !mul_u64(value, GLM53_PLAN_KDA_LAYERS, &next.kda_state_bytes)) {
        return GLM53_ENGINE_PLAN_OVERFLOW;
    }
    /* Three q/k/v depthwise convolution buffers. Charge the full kernel width. */
    if (!mul_u64(UINT64_C(3), GLM53_PLAN_KDA_CHANNELS, &value) ||
        !mul_u64(value, GLM53_PLAN_KDA_CONV_WIDTH, &value) ||
        !mul_u64(value, GLM53_PLAN_F32_BYTES, &value) ||
        !mul_u64(value, GLM53_PLAN_KDA_LAYERS, &next.kda_conv_bytes)) {
        return GLM53_ENGINE_PLAN_OVERFLOW;
    }

    /* Persistent compact DSA row: KV, index key, index gate, valid. */
    if (!add_u64(GLM53_PLAN_DSA_LATENT, GLM53_PLAN_DSA_INDEX_KEY, &per_token) ||
        !add_u64(per_token, GLM53_PLAN_DSA_INDEX_GATE, &per_token) ||
        !add_u64(per_token, GLM53_PLAN_DSA_VALID, &per_token) ||
        !mul_u64(per_token, GLM53_PLAN_BF16_BYTES, &per_token) ||
        !mul_u64(per_token, (uint64_t)config->context_tokens, &value) ||
        !mul_u64(value, GLM53_PLAN_DSA_LAYERS, &next.dsa_compact_bytes)) {
        return GLM53_ENGINE_PLAN_OVERFLOW;
    }

    if (!align_up_u64(GLM53_ENGINE_PLAN_MAPPED_STAGING_BYTES,
                      GLM53_PLAN_ALIGNMENT, &aligned_expert) ||
        !mul_u64(aligned_expert, (uint64_t)config->mapped_staging_slots,
                 &next.mapped_staging_bytes)) {
        return GLM53_ENGINE_PLAN_OVERFLOW;
    }
    next.compute_workspace_bytes = config->compute_workspace_bytes;
    next.library_workspace_bytes = config->library_workspace_bytes;
    next.allocator_guard_bytes = config->allocator_guard_bytes;
    next.runtime_guard_bytes = config->runtime_guard_bytes;
    next.host_guard_bytes = config->host_guard_bytes;

    next.runtime_bytes_before_guards = next.static_resident_bytes;
    if (!add_to(&next.runtime_bytes_before_guards,
                next.expert_resident_bytes) ||
        !add_to(&next.runtime_bytes_before_guards, next.kda_state_bytes) ||
        !add_to(&next.runtime_bytes_before_guards, next.kda_conv_bytes) ||
        !add_to(&next.runtime_bytes_before_guards, next.dsa_compact_bytes) ||
        !add_to(&next.runtime_bytes_before_guards,
                next.mapped_staging_bytes) ||
        !add_to(&next.runtime_bytes_before_guards,
                next.compute_workspace_bytes) ||
        !add_to(&next.runtime_bytes_before_guards,
                next.library_workspace_bytes)) {
        return GLM53_ENGINE_PLAN_OVERFLOW;
    }
    if (!add_u64(next.static_resident_bytes,
                 next.static_peak_transient_bytes,
                 &next.load_bytes_before_guards)) {
        return GLM53_ENGINE_PLAN_OVERFLOW;
    }
    next.peak_bytes_before_guards =
        next.runtime_bytes_before_guards > next.load_bytes_before_guards ?
        next.runtime_bytes_before_guards : next.load_bytes_before_guards;
    if (!add_to(&guards, next.allocator_guard_bytes) ||
        !add_to(&guards, next.runtime_guard_bytes) ||
        !add_to(&guards, next.host_guard_bytes) ||
        !add_u64(next.peak_bytes_before_guards, guards,
                 &next.admitted_bytes)) {
        return GLM53_ENGINE_PLAN_OVERFLOW;
    }
    if (next.admitted_bytes >= GLM53_ENGINE_PLAN_HARD_CEILING_BYTES) {
        return GLM53_ENGINE_PLAN_HARD_CEILING;
    }
    if (mem_available_bytes < GLM53_ENGINE_PLAN_MIN_MEM_AVAILABLE_BYTES ||
        mem_available_bytes < next.admitted_bytes) {
        return GLM53_ENGINE_PLAN_MEM_AVAILABLE_POLICY;
    }
    next.abort_reserve_bytes = mem_available_bytes - next.admitted_bytes;
    if (next.abort_reserve_bytes < GLM53_ENGINE_PLAN_MIN_ABORT_RESERVE_BYTES) {
        return GLM53_ENGINE_PLAN_ABORT_RESERVE_POLICY;
    }
    *report = next;
    return GLM53_ENGINE_PLAN_OK;
}

const char *glm53_engine_plan_status_string(glm53_engine_plan_status status) {
    switch (status) {
        case GLM53_ENGINE_PLAN_OK: return "ok";
        case GLM53_ENGINE_PLAN_INVALID_ARGUMENT: return "invalid argument";
        case GLM53_ENGINE_PLAN_UNSUPPORTED_PROFILE: return "unsupported profile";
        case GLM53_ENGINE_PLAN_OVERFLOW: return "overflow";
        case GLM53_ENGINE_PLAN_MEM_AVAILABLE_POLICY: return "MemAvailable policy";
        case GLM53_ENGINE_PLAN_ABORT_RESERVE_POLICY: return "abort reserve policy";
        case GLM53_ENGINE_PLAN_HARD_CEILING: return "hard ceiling";
        default: return "unknown status";
    }
}
