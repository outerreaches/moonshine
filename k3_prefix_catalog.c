#include "k3_prefix_catalog.h"

#include "k3_prefix_reuse.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t *tokens;
    size_t token_count;
    char *state_path;
    k3_engine_state_file_info state_info;
} k3_prefix_catalog_entry;

struct k3_prefix_catalog {
    k3_prefix_catalog_entry *entries;
    size_t count;
    size_t capacity;
};

static bool identity_matches_info(
        const k3_prefix_catalog_identity *identity,
        const k3_engine_state_file_info *info) {
    return identity != NULL && info != NULL &&
        identity->format_version == info->format_version &&
        identity->context == info->context &&
        identity->model_layout_crc64 == info->model_layout_crc64 &&
        identity->q8_projections == info->q8_projections;
}

static bool state_info_valid(
        const k3_engine_state_file_info *info,
        size_t token_count) {
    return info != NULL && token_count != 0u &&
        token_count <= UINT32_MAX &&
        info->format_version != 0u &&
        info->context != 0u &&
        info->token_position == token_count &&
        info->token_position <= info->context &&
        info->model_layout_crc64 != 0u &&
        info->payload_bytes != 0u &&
        info->file_bytes > info->payload_bytes;
}

static bool state_info_equal(
        const k3_engine_state_file_info *left,
        const k3_engine_state_file_info *right) {
    return left->format_version == right->format_version &&
        left->context == right->context &&
        left->token_position == right->token_position &&
        left->model_layout_crc64 == right->model_layout_crc64 &&
        left->payload_bytes == right->payload_bytes &&
        left->file_bytes == right->file_bytes &&
        left->payload_crc64 == right->payload_crc64 &&
        left->q8_projections == right->q8_projections;
}

static bool tokens_equal(
        const k3_prefix_catalog_entry *entry,
        const uint32_t *tokens,
        size_t token_count) {
    return entry->token_count == token_count &&
        token_count <= SIZE_MAX / sizeof(*tokens) &&
        memcmp(entry->tokens, tokens,
               token_count * sizeof(*tokens)) == 0;
}

static bool entry_matches_identity(
        const k3_prefix_catalog_entry *entry,
        const k3_prefix_catalog_identity *identity) {
    return identity_matches_info(identity, &entry->state_info);
}

static void entry_destroy(k3_prefix_catalog_entry *entry) {
    if (entry == NULL) return;
    free(entry->state_path);
    free(entry->tokens);
    memset(entry, 0, sizeof(*entry));
}

bool k3_prefix_catalog_create(k3_prefix_catalog **out) {
    if (out != NULL) *out = NULL;
    if (out == NULL) return false;
    k3_prefix_catalog *catalog =
        (k3_prefix_catalog *)calloc(1u, sizeof(*catalog));
    if (catalog == NULL) return false;
    *out = catalog;
    return true;
}

void k3_prefix_catalog_destroy(k3_prefix_catalog *catalog) {
    if (catalog == NULL) return;
    for (size_t index = 0u; index < catalog->count; index++) {
        entry_destroy(&catalog->entries[index]);
    }
    free(catalog->entries);
    free(catalog);
}

size_t k3_prefix_catalog_count(const k3_prefix_catalog *catalog) {
    return catalog != NULL ? catalog->count : 0u;
}

bool k3_prefix_catalog_add(
        k3_prefix_catalog *catalog,
        const uint32_t *tokens,
        size_t token_count,
        const char *state_path,
        const k3_engine_state_file_info *state_info) {
    if (catalog == NULL || tokens == NULL || state_path == NULL ||
        state_path[0] == '\0' ||
        token_count > SIZE_MAX / sizeof(*tokens) ||
        !state_info_valid(state_info, token_count)) {
        return false;
    }
    const k3_prefix_catalog_identity identity = {
        state_info->format_version,
        state_info->context,
        state_info->model_layout_crc64,
        state_info->q8_projections,
    };
    for (size_t index = 0u; index < catalog->count; index++) {
        k3_prefix_catalog_entry *entry = &catalog->entries[index];
        if (!entry_matches_identity(entry, &identity) ||
            !tokens_equal(entry, tokens, token_count)) {
            continue;
        }
        return strcmp(entry->state_path, state_path) == 0 &&
            state_info_equal(&entry->state_info, state_info);
    }

    if (catalog->count == catalog->capacity) {
        const size_t capacity = catalog->capacity == 0u ?
            4u : catalog->capacity * 2u;
        if (capacity < catalog->capacity ||
            capacity > SIZE_MAX / sizeof(*catalog->entries)) {
            return false;
        }
        k3_prefix_catalog_entry *entries =
            (k3_prefix_catalog_entry *)realloc(
                catalog->entries,
                capacity * sizeof(*catalog->entries));
        if (entries == NULL) return false;
        memset(entries + catalog->capacity, 0,
               (capacity - catalog->capacity) * sizeof(*entries));
        catalog->entries = entries;
        catalog->capacity = capacity;
    }

    uint32_t *token_copy = (uint32_t *)malloc(
        token_count * sizeof(*token_copy));
    const size_t path_bytes = strlen(state_path);
    if (path_bytes == SIZE_MAX) {
        free(token_copy);
        return false;
    }
    char *path_copy = (char *)malloc(path_bytes + 1u);
    if (token_copy == NULL || path_copy == NULL) {
        free(path_copy);
        free(token_copy);
        return false;
    }
    memcpy(token_copy, tokens, token_count * sizeof(*token_copy));
    memcpy(path_copy, state_path, path_bytes + 1u);
    k3_prefix_catalog_entry *entry =
        &catalog->entries[catalog->count++];
    entry->tokens = token_copy;
    entry->token_count = token_count;
    entry->state_path = path_copy;
    entry->state_info = *state_info;
    return true;
}

bool k3_prefix_catalog_remove(
        k3_prefix_catalog *catalog,
        const uint32_t *tokens,
        size_t token_count,
        const k3_prefix_catalog_identity *identity) {
    if (catalog == NULL || tokens == NULL || identity == NULL ||
        token_count == 0u ||
        token_count > SIZE_MAX / sizeof(*tokens)) {
        return false;
    }
    for (size_t index = 0u; index < catalog->count; index++) {
        k3_prefix_catalog_entry *entry = &catalog->entries[index];
        if (!entry_matches_identity(entry, identity) ||
            !tokens_equal(entry, tokens, token_count)) {
            continue;
        }
        entry_destroy(entry);
        if (index + 1u < catalog->count) {
            memmove(entry, entry + 1u,
                    (catalog->count - index - 1u) * sizeof(*entry));
        }
        catalog->count--;
        memset(&catalog->entries[catalog->count], 0,
               sizeof(*catalog->entries));
        return true;
    }
    return false;
}

bool k3_prefix_catalog_find(
        const k3_prefix_catalog *catalog,
        const uint32_t *candidate,
        size_t candidate_count,
        const k3_prefix_catalog_identity *identity,
        k3_prefix_catalog_match *match) {
    if (match != NULL) memset(match, 0, sizeof(*match));
    if (catalog == NULL || candidate == NULL || identity == NULL ||
        match == NULL) {
        return false;
    }
    const k3_prefix_catalog_entry *best = NULL;
    for (size_t index = 0u; index < catalog->count; index++) {
        const k3_prefix_catalog_entry *entry = &catalog->entries[index];
        if ((best != NULL && entry->token_count <= best->token_count) ||
            !entry_matches_identity(entry, identity) ||
            !k3_prefix_reuse_admits(
                entry->tokens, entry->token_count,
                candidate, candidate_count)) {
            continue;
        }
        best = entry;
    }
    if (best == NULL) return false;
    match->tokens = best->tokens;
    match->token_count = best->token_count;
    match->state_path = best->state_path;
    match->state_info = &best->state_info;
    return true;
}
