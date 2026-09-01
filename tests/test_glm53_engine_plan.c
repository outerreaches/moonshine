#include "glm53_engine_plan.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)

static glm53_engine_static_ledger static_ledger(uint64_t resident,
                                                 uint64_t transient) {
    glm53_engine_static_ledger result;
    result.resident_bytes = resident;
    result.peak_transient_bytes = transient;
    return result;
}

static bool unchanged(const glm53_engine_plan_report *a,
                      const glm53_engine_plan_report *b) {
    return memcmp(a, b, sizeof(*a)) == 0;
}

static bool test_default_proof_profile(void) {
    glm53_engine_plan_config config;
    glm53_engine_plan_report report;
    glm53_engine_static_ledger weights =
        static_ledger(UINT64_C(70) * GLM53_ENGINE_GIB,
                      UINT64_C(1) * GLM53_ENGINE_GIB);
    uint64_t want_state = UINT64_C(34) * UINT64_C(64) * UINT64_C(128) *
                          UINT64_C(128) * UINT64_C(4);
    uint64_t want_conv = UINT64_C(34) * UINT64_C(3) * UINT64_C(8192) *
                         UINT64_C(4) * UINT64_C(4);
    uint64_t want_dsa = UINT64_C(11) * UINT64_C(512) *
                        UINT64_C(769) * UINT64_C(2);
    glm53_engine_plan_config_init(&config);
    CHECK(config.batch_size == 1u);
    CHECK(config.context_tokens == 512u);
    CHECK(config.expert_cache_slots_per_layer == 0u);
    CHECK(config.compute_workspace_bytes == UINT64_C(256) * UINT64_C(1048576));
    CHECK(config.library_workspace_bytes == UINT64_C(64) * UINT64_C(1048576));
    CHECK(config.allocator_guard_bytes == UINT64_C(256) * UINT64_C(1048576));
    CHECK(config.runtime_guard_bytes == GLM53_ENGINE_GIB);
    CHECK(config.host_guard_bytes == UINT64_C(4) * GLM53_ENGINE_GIB);
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_MIN_MEM_AVAILABLE_BYTES, &report) ==
          GLM53_ENGINE_PLAN_OK);
    CHECK(report.expert_resident_bytes == 0u);
    CHECK(report.kda_state_bytes == want_state);
    CHECK(report.kda_conv_bytes == want_conv);
    CHECK(report.dsa_compact_bytes == want_dsa);
    CHECK(report.mapped_staging_bytes ==
          GLM53_ENGINE_PLAN_MAPPED_STAGING_BYTES);
    CHECK(report.mapped_staging_bytes % UINT64_C(4096) == 0u);
    CHECK(report.abort_reserve_bytes >=
          GLM53_ENGINE_PLAN_MIN_ABORT_RESERVE_BYTES);
    CHECK(report.admitted_bytes <=
          GLM53_ENGINE_PLAN_MIN_MEM_AVAILABLE_BYTES);
    return true;
}

static bool test_context_and_slot_boundaries(void) {
    glm53_engine_plan_config config;
    glm53_engine_plan_report a, b, sentinel;
    glm53_engine_static_ledger weights =
        static_ledger(GLM53_ENGINE_GIB, 0u);
    glm53_engine_plan_config_init(&config);
    config.context_tokens = 1u;
    memset(&a, 0, sizeof(a));
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES, &a) == GLM53_ENGINE_PLAN_OK);
    config.context_tokens = 512u;
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES, &b) == GLM53_ENGINE_PLAN_OK);
    CHECK(b.dsa_compact_bytes == a.dsa_compact_bytes * UINT64_C(512));
    CHECK(b.expert_resident_bytes == 0u);
    sentinel = b;
    config.context_tokens = 513u;
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES, &b) ==
          GLM53_ENGINE_PLAN_UNSUPPORTED_PROFILE);
    CHECK(unchanged(&b, &sentinel));
    config.context_tokens = 512u;
    config.expert_cache_slots_per_layer = 1u;
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES, &b) ==
          GLM53_ENGINE_PLAN_UNSUPPORTED_PROFILE);
    CHECK(unchanged(&b, &sentinel));
    config.expert_cache_slots_per_layer = 0u;
    config.mapped_staging_slots = 2u;
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES, &b) ==
          GLM53_ENGINE_PLAN_UNSUPPORTED_PROFILE);
    CHECK(unchanged(&b, &sentinel));
    config.mapped_staging_slots = 1u;
    config.batch_size = 2u;
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES, &b) ==
          GLM53_ENGINE_PLAN_UNSUPPORTED_PROFILE);
    CHECK(unchanged(&b, &sentinel));
    return true;
}

static bool test_policy_boundaries_transactional(void) {
    glm53_engine_plan_config config;
    glm53_engine_static_ledger weights =
        static_ledger(UINT64_C(85) * GLM53_ENGINE_GIB, 0u);
    glm53_engine_plan_report report, before;
    uint64_t exact_available;
    glm53_engine_plan_config_init(&config);
    memset(&report, 0xa5, sizeof(report));
    before = report;
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_MIN_MEM_AVAILABLE_BYTES - UINT64_C(1), &report) ==
          GLM53_ENGINE_PLAN_MEM_AVAILABLE_POLICY);
    CHECK(unchanged(&report, &before));
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES - UINT64_C(1), &report) ==
          GLM53_ENGINE_PLAN_OK);
    exact_available = report.admitted_bytes +
                      GLM53_ENGINE_PLAN_MIN_ABORT_RESERVE_BYTES;
    CHECK(exact_available >= GLM53_ENGINE_PLAN_MIN_MEM_AVAILABLE_BYTES);
    CHECK(glm53_engine_plan_build(&config, &weights, exact_available, &report) ==
          GLM53_ENGINE_PLAN_OK);
    CHECK(report.abort_reserve_bytes ==
          GLM53_ENGINE_PLAN_MIN_ABORT_RESERVE_BYTES);
    before = report;
    CHECK(glm53_engine_plan_build(&config, &weights,
          exact_available - UINT64_C(1), &report) ==
          GLM53_ENGINE_PLAN_ABORT_RESERVE_POLICY);
    CHECK(unchanged(&report, &before));
    return true;
}

static bool test_hard_ceiling_exact(void) {
    glm53_engine_plan_config config;
    glm53_engine_static_ledger weights = static_ledger(GLM53_ENGINE_GIB, 0u);
    glm53_engine_plan_report report, exact, before;
    uint64_t increment;
    glm53_engine_plan_config_init(&config);
    CHECK(glm53_engine_plan_build(&config, &weights,
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES, &report) ==
          GLM53_ENGINE_PLAN_OK);
    CHECK(report.admitted_bytes < GLM53_ENGINE_PLAN_HARD_CEILING_BYTES);
    increment = GLM53_ENGINE_PLAN_HARD_CEILING_BYTES - report.admitted_bytes -
                UINT64_C(1);
    weights.resident_bytes += increment;
    CHECK(glm53_engine_plan_build(&config, &weights, UINT64_MAX, &exact) ==
          GLM53_ENGINE_PLAN_OK);
    CHECK(exact.admitted_bytes ==
          GLM53_ENGINE_PLAN_HARD_CEILING_BYTES - UINT64_C(1));
    before = exact;
    weights.resident_bytes += UINT64_C(1);
    CHECK(glm53_engine_plan_build(&config, &weights, UINT64_MAX, &exact) ==
          GLM53_ENGINE_PLAN_HARD_CEILING);
    CHECK(unchanged(&exact, &before));
    return true;
}

static bool test_checked_overflow_and_invalid(void) {
    glm53_engine_plan_config config;
    glm53_engine_static_ledger weights = static_ledger(GLM53_ENGINE_GIB, 0u);
    glm53_engine_plan_report report, before;
    glm53_engine_plan_config_init(&config);
    memset(&report, 0x3c, sizeof(report));
    before = report;
    config.compute_workspace_bytes = UINT64_MAX;
    CHECK(glm53_engine_plan_build(&config, &weights, UINT64_MAX, &report) ==
          GLM53_ENGINE_PLAN_OVERFLOW);
    CHECK(unchanged(&report, &before));
    glm53_engine_plan_config_init(&config);
    weights.peak_transient_bytes = UINT64_MAX;
    CHECK(glm53_engine_plan_build(&config, &weights, UINT64_MAX, &report) ==
          GLM53_ENGINE_PLAN_OVERFLOW);
    CHECK(unchanged(&report, &before));
    weights = static_ledger(GLM53_ENGINE_GIB, 0u);
    CHECK(glm53_engine_plan_build(NULL, &weights, UINT64_MAX, &report) ==
          GLM53_ENGINE_PLAN_INVALID_ARGUMENT);
    CHECK(unchanged(&report, &before));
    return true;
}

int main(void) {
    if (!test_default_proof_profile() ||
        !test_context_and_slot_boundaries() ||
        !test_policy_boundaries_transactional() ||
        !test_hard_ceiling_exact() ||
        !test_checked_overflow_and_invalid()) return 1;
    puts("glm53 engine plan tests: ok");
    return 0;
}
