#ifndef K3_CHAT_H
#define K3_CHAT_H

#include "k3_engine.h"
#include "k3_tokenizer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k3_chat_session k3_chat_session;

enum {
    /* Sequential remains faster through 92 whole tokens; switch at 93. */
    K3_CHAT_MEASURED_SEQUENTIAL_LIMIT = 92,
};

typedef struct {
    const char *model_root;
    const char *system_prompt;
    uint32_t    context;
    uint32_t    sequential_prefill_limit;
    uint16_t    experts_per_layer;
    uint16_t    staging_slots;
    bool        q8_projections;
} k3_chat_session_config;

typedef enum {
    K3_CHAT_PREFILL_SEQUENTIAL = 0,
    K3_CHAT_PREFILL_LAYER_MAJOR = 1,
} k3_chat_prefill_strategy;

typedef enum {
    K3_CHAT_FINISH_END_OF_MESSAGE = 0,
    K3_CHAT_FINISH_LENGTH = 1,
    K3_CHAT_FINISH_TOOL_CALLS = 2,
} k3_chat_finish_reason;

typedef enum {
    K3_CHAT_PREFILL_PROGRESS_TOKENS = 0,
    K3_CHAT_PREFILL_PROGRESS_LAYERS = 1,
} k3_chat_prefill_progress_unit;

typedef struct {
    k3_text_buffer           reasoning_content;
    k3_text_buffer           response;
    k3_tool_call            *tool_calls;
    size_t                   tool_call_count;
    k3_chat_prefill_strategy prefill_strategy;
    k3_chat_finish_reason    finish_reason;
    uint32_t                 prompt_tokens;
    uint32_t                 prompt_evaluated_tokens;
    uint32_t                 prompt_reused_tokens;
    uint32_t                 generated_tokens;
    uint32_t                 forced_trailer_tokens;
    uint32_t                 position;
    bool                     thinking;
    double                   prompt_seconds;
    double                   decode_seconds;
    double                   tokens_per_second;
    k3_engine_prefill_stats  range_stats;
    k3_engine_cache_stats    cache_before;
    k3_engine_cache_stats    cache_after;
} k3_chat_turn_result;

/*
 * Called with raw token-piece bytes as response content becomes available.
 * Individual calls may split one UTF-8 scalar; their concatenation matches
 * result.response exactly.
 */
typedef void (*k3_chat_text_callback)(
    const char *bytes, size_t size, void *user_data);

typedef void (*k3_chat_prefill_progress_callback)(
    k3_chat_prefill_progress_unit unit,
    uint32_t                      completed,
    uint32_t                      total,
    void                         *user_data);

typedef struct {
    /*
     * Reuse the retained causal state only when the newly rendered history is
     * an exact token-prefix extension. A mismatch falls back to a semantic
     * reset and full prefill.
     */
    bool                              reuse_prefix;
    /*
     * Clearing the expert cache is useful for explicit cold-cache benchmarks.
     * Normal serving should leave this false because cached weights are not
     * semantic request state.
     */
    bool                              clear_expert_cache;
    k3_chat_prefill_progress_callback progress_callback;
    void                             *progress_data;
    bool                              thinking;
    const char                       *thinking_effort;
    /* Optional token-live callback for the generated <think> channel. */
    k3_chat_text_callback             reasoning_callback;
    void                             *reasoning_data;
    const char                       *tools_json;
    k3_tool_choice                    tool_choice;
    /* Retain hidden request directives for exact session-history replay. */
    bool                              preserve_tool_choice_history;
} k3_chat_completion_options;

/*
 * Create one stateful non-thinking text chat session. Short prompts use
 * token-major execution to avoid a full routed sweep; prompts above
 * sequential_prefill_limit use layer-major prefill.
 */
bool k3_chat_session_create(
    k3_chat_session              **out,
    const k3_chat_session_config  *config,
    k3_engine_stats              *engine_stats,
    char                         *error,
    size_t                        error_size);

void k3_chat_session_destroy(k3_chat_session *session);
void k3_chat_turn_result_free(k3_chat_turn_result *result);

/*
 * Submit one new user turn. The first turn includes the configured system
 * prompt; later turns encode only the new XTML delta against retained causal
 * state. max_generated_tokens counts model-produced content and structural
 * tokens. A length stop is closed with the shortest missing official trailer.
 */
bool k3_chat_session_turn(
    k3_chat_session       *session,
    const char            *user_text,
    uint32_t               max_generated_tokens,
    k3_chat_text_callback  callback,
    void                  *callback_data,
    k3_chat_turn_result   *result,
    char                  *error,
    size_t                 error_size);

/* Reset to the zero-position semantic state without reloading the model. */
bool k3_chat_session_reset(
    k3_chat_session *session,
    bool             clear_expert_cache,
    char            *error,
    size_t           error_size);

/*
 * OpenAI-style stateless completion: reset semantic state and render the
 * complete supplied history before generation. Resident expert mappings are
 * preserved because they contain immutable model weights.
 */
bool k3_chat_session_complete_messages(
    k3_chat_session        *session,
    const k3_chat_message  *messages,
    size_t                  message_count,
    uint32_t                max_generated_tokens,
    k3_chat_text_callback   callback,
    void                   *callback_data,
    k3_chat_turn_result    *result,
    char                   *error,
    size_t                  error_size);

/*
 * Extended completion interface. Prefix reuse is opt-in and append-only: it
 * activates only when the rendered history exactly extends every token in the
 * retained semantic state. Preserved tool-choice history can reconstruct
 * request-local hidden directives before the same exact comparison. Edited,
 * forked, shorter, or otherwise mismatched histories fall back to the same
 * isolated full-prefill behavior as the stateless interface.
 */
bool k3_chat_session_complete_messages_with_options(
    k3_chat_session                 *session,
    const k3_chat_message           *messages,
    size_t                           message_count,
    uint32_t                         max_generated_tokens,
    const k3_chat_completion_options *options,
    k3_chat_text_callback            callback,
    void                            *callback_data,
    k3_chat_turn_result             *result,
    char                            *error,
    size_t                           error_size);

bool k3_chat_session_export_state(
    const k3_chat_session      *session,
    const char                 *path,
    k3_engine_state_file_info  *info,
    char                       *error,
    size_t                      error_size);

bool k3_chat_session_import_state(
    k3_chat_session            *session,
    const char                 *path,
    k3_engine_state_file_info  *info,
    char                       *error,
    size_t                      error_size);

#ifdef __cplusplus
}
#endif

#endif
