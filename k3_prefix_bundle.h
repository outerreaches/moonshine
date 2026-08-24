#ifndef K3_PREFIX_BUNDLE_H
#define K3_PREFIX_BUNDLE_H

#include "k3_engine.h"
#include "k3_tokenizer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k3_prefix_bundle k3_prefix_bundle;

typedef struct {
    uint32_t format_version;
    uint32_t context;
    uint64_t model_layout_crc64;
    bool q8_projections;
} k3_prefix_bundle_identity;

typedef struct {
    const uint32_t *tokens;
    size_t token_count;
    const k3_tool_choice_marker *tool_choices;
    size_t tool_choice_count;
    const k3_single_tool_call_marker *single_tool_calls;
    size_t single_tool_call_count;
    const k3_response_format_marker *response_formats;
    size_t response_format_count;
} k3_prefix_bundle_snapshot;

typedef struct {
    const char *id;
    const char *state_path;
    const uint32_t *tokens;
    size_t token_count;
    const k3_tool_choice_marker *tool_choices;
    size_t tool_choice_count;
    const k3_single_tool_call_marker *single_tool_calls;
    size_t single_tool_call_count;
    const k3_response_format_marker *response_formats;
    size_t response_format_count;
    const k3_engine_state_file_info *state_info;
    uint64_t publication_sequence;
    uint64_t last_use_sequence;
    uint64_t metadata_bytes;
} k3_prefix_bundle_entry;

bool k3_prefix_bundle_open(
    k3_prefix_bundle **out,
    const char *absolute_root,
    const k3_prefix_bundle_identity *identity,
    uint32_t entry_limit,
    uint64_t byte_limit,
    char *error,
    size_t error_size);
void k3_prefix_bundle_destroy(k3_prefix_bundle *bundle);

size_t k3_prefix_bundle_count(const k3_prefix_bundle *bundle);
bool k3_prefix_bundle_entry_at(
    const k3_prefix_bundle *bundle,
    size_t index,
    k3_prefix_bundle_entry *entry);

bool k3_prefix_bundle_find_exact(
    const k3_prefix_bundle *bundle,
    const uint32_t *tokens,
    size_t token_count,
    size_t *index);

/* Reserve generated state/id paths. No entry is published by this call. */
bool k3_prefix_bundle_allocate_state_path(
    k3_prefix_bundle *bundle,
    char *id,
    size_t id_size,
    char *state_path,
    size_t state_path_size,
    char *error,
    size_t error_size);

/* STATE_PATH must be the generated path returned for ID and already complete. */
bool k3_prefix_bundle_publish(
    k3_prefix_bundle *bundle,
    const char *id,
    const char *state_path,
    const k3_prefix_bundle_snapshot *snapshot,
    const k3_engine_state_file_info *state_info,
    char *error,
    size_t error_size);

bool k3_prefix_bundle_remove(
    k3_prefix_bundle *bundle,
    size_t index,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
