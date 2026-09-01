#ifndef GLM53_PHASE5C_H
#define GLM53_PHASE5C_H

#include "glm53_static_bindings.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53_PHASE5C_LAYER0_HIDDEN 4096u
#define GLM53_PHASE5C_LAYER0_INTERMEDIATE 12288u
#define GLM53_PHASE5C_LAYER0_VOCAB 154880u
#define GLM53_PHASE5C_LAYER0_EARLY_HEAD_DIAGNOSTIC_BINDING_COUNT 17u
#define GLM53_PHASE5C_WORKSPACE_ALIGNMENT 256u

typedef enum {
    GLM53_PHASE5C_OK = 0,
    GLM53_PHASE5C_INVALID_ARGUMENT,
    GLM53_PHASE5C_BAD_BINDINGS,
    GLM53_PHASE5C_ALLOCATION_FAILED,
    GLM53_PHASE5C_INVALID_STATE,
    GLM53_PHASE5C_INVALID_TOKEN,
    GLM53_PHASE5C_LAUNCH_FAILED,
    GLM53_PHASE5C_PROVIDER_FAILED,
    GLM53_PHASE5C_DEVICE_SYNC_FAILED
} glm53_phase5c_status;

typedef enum {
    GLM53_PHASE5C_SESSION_READY = 1,
    GLM53_PHASE5C_SESSION_BUSY,
    GLM53_PHASE5C_SESSION_FAULT
} glm53_phase5c_session_state;

typedef enum {
    GLM53_PHASE5C_WS_EMBEDDING = 0,
    GLM53_PHASE5C_WS_STREAMS_A,
    GLM53_PHASE5C_WS_STREAMS_B,
    GLM53_PHASE5C_WS_MHC_MIX,
    GLM53_PHASE5C_WS_MHC_PRE,
    GLM53_PHASE5C_WS_MHC_POST,
    GLM53_PHASE5C_WS_MHC_COMB,
    GLM53_PHASE5C_WS_COLLAPSED,
    GLM53_PHASE5C_WS_NORMALIZED,
    GLM53_PHASE5C_WS_BRANCH,
    GLM53_PHASE5C_WS_DENSE_INPUT_F32,
    GLM53_PHASE5C_WS_DENSE_GATE_F32,
    GLM53_PHASE5C_WS_DENSE_UP_F32,
    GLM53_PHASE5C_WS_DENSE_ACTIVATION_F32,
    GLM53_PHASE5C_WS_DENSE_DOWN_F32,
    GLM53_PHASE5C_WS_DENSE_Q8,
    GLM53_PHASE5C_WS_DENSE_DYNAMIC_SCALE,
    GLM53_PHASE5C_WS_LAYER0_EARLY_HEAD_DIAGNOSTIC_LOGITS_0,
    GLM53_PHASE5C_WS_LAYER0_EARLY_HEAD_DIAGNOSTIC_LOGITS_1,
    GLM53_PHASE5C_WS_REGION_COUNT
} glm53_phase5c_workspace_region;

typedef struct {
    uint64_t offset;
    uint64_t bytes;
} glm53_phase5c_workspace_span;

typedef struct {
    glm53_phase5c_workspace_span regions[GLM53_PHASE5C_WS_REGION_COUNT];
    uint64_t workspace_bytes;
    uint64_t accounted_bytes;
    uint32_t alignment;
} glm53_phase5c_workspace_layout;

/* A provider owns attention only. input/output are disjoint BF16 [4096]
 * device vectors. It may only enqueue on stream and must not synchronize. */
typedef bool (*glm53_phase5c_attention_provider_fn)(
    void *output, size_t output_count,
    const void *input, size_t input_count,
    uint64_t position, void *stream, void *context);

typedef struct {
    glm53_phase5c_attention_provider_fn run;
    void *context;
} glm53_phase5c_attention_provider;

typedef struct glm53_phase5c_session glm53_phase5c_session;

typedef struct {
    bool available;
    uint64_t position;
    const void *layer0_early_head_diagnostic_logits; /* BF16 [154880], device */
    size_t layer0_early_head_diagnostic_logits_count;
} glm53_phase5c_layer0_early_head_diagnostic;

/* Deterministic, allocation-free ledger and schema seam. */
bool glm53_phase5c_workspace_layout_build(glm53_phase5c_workspace_layout *out);
glm53_phase5c_status glm53_phase5c_validate_layer0_early_head_diagnostic_bindings(
    const glm53_static_bindings *bindings, char *error, size_t error_size);

/* Cache the exact 17 non-attention layer-0 bindings, create one nonblocking
 * stream, and allocate one contiguous aligned HIP workspace. The bindings
 * object may be freed after creation, but its backing static store and device
 * allocation must outlive the session. */
glm53_phase5c_status glm53_phase5c_session_create(
    glm53_phase5c_session **out, const glm53_static_bindings *bindings,
    char *error, size_t error_size);
void glm53_phase5c_session_destroy(glm53_phase5c_session *session);
/* Drain pending work. On success, clear the published diagnostic and return
 * READY. A synchronization failure leaves the session in FAULT. */
glm53_phase5c_status glm53_phase5c_session_reset(glm53_phase5c_session *session);
glm53_phase5c_session_state glm53_phase5c_session_get_state(
    const glm53_phase5c_session *session);
bool glm53_phase5c_session_get_layer0_early_head_diagnostic(
    const glm53_phase5c_session *session,
    glm53_phase5c_layer0_early_head_diagnostic *out);

/* This is a layer-0 early-head diagnostic, not model logits. Successful work
 * publishes position and the alternate diagnostic-logit slot only after the
 * sole stream synchronization succeeds. All ordinary failures drain the
 * stream and leave the previously published diagnostic unchanged. */
glm53_phase5c_status glm53_phase5c_layer0_early_head_diagnostic_step(
    glm53_phase5c_session *session, uint32_t token, uint64_t position,
    const glm53_phase5c_attention_provider *attention_provider);

/* Pure provider boundary implementation: enqueue BF16 output zero-fill. */
bool glm53_phase5c_zero_attention_provider(
    void *output, size_t output_count,
    const void *input, size_t input_count,
    uint64_t position, void *stream, void *context);

const char *glm53_phase5c_status_string(glm53_phase5c_status status);

#ifdef __cplusplus
}
#endif
#endif
