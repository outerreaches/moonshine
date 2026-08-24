#include "k3_chat.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                     \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

static bool digest_equal(const k3_engine_state_digest *left,
                         const k3_engine_state_digest *right) {
    return left->kda_state_hash == right->kda_state_hash &&
        left->kda_conv_hash == right->kda_conv_hash &&
        left->mla_cache_hash == right->mla_cache_hash &&
        left->attn_res_hash == right->attn_res_hash &&
        left->token_position == right->token_position;
}

static void cleanup_root(const char *root) {
    char entries[4096];
    snprintf(entries, sizeof(entries), "%s/entries", root);
    DIR *directory = opendir(entries);
    if (directory != NULL) {
        struct dirent *item;
        while ((item = readdir(directory)) != NULL) {
            if (strcmp(item->d_name, ".") == 0 ||
                strcmp(item->d_name, "..") == 0) continue;
            char path[4352];
            snprintf(path, sizeof(path), "%s/%s", entries, item->d_name);
            (void)unlink(path);
        }
        (void)closedir(directory);
    }
    char manifest[4096];
    snprintf(manifest, sizeof(manifest), "%s/manifest.bin", root);
    (void)unlink(manifest);
    (void)rmdir(entries);
    (void)rmdir(root);
}

static bool complete(k3_chat_session *session,
                     const k3_chat_message *messages,
                     size_t message_count,
                     k3_chat_turn_result *result,
                     char *error, size_t error_size) {
    const k3_chat_completion_options options = {
        .reuse_prefix = true,
        .preserve_request_directive_history = true,
    };
    return k3_chat_session_complete_messages_with_options(
        session, messages, message_count, 32u, &options,
        NULL, NULL, result, error, error_size);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_ROOT\n", argv[0]);
        return 2;
    }
    int status = 1;
    char root[] = "/tmp/k3-prefix-checkpoint-XXXXXX";
    char error[1024];
    k3_chat_session *session = NULL;
    k3_chat_turn_result first = {0};
    k3_chat_turn_result expected = {0};
    k3_chat_turn_result unrelated = {0};
    k3_chat_turn_result restored = {0};
    k3_chat_turn_result restarted = {0};
    char *first_response = NULL;
    char *expected_response = NULL;
    k3_engine_state_digest expected_digest;
    memset(&expected_digest, 0, sizeof(expected_digest));
    const k3_chat_session_config config = {
        .model_root = argv[1],
        .context = 8192u,
        .sequential_prefill_limit = 7u,
        .experts_per_layer = 32u,
        .staging_slots = 16u,
        .q8_projections = true,
        .capture_state_digest = true,
        .prefix_checkpoint_root = root,
        .prefix_checkpoint_entries = 2u,
        .prefix_checkpoint_bytes = UINT64_C(2) * 1024u * 1024u * 1024u,
    };
    CHECK(mkdtemp(root) != NULL, "checkpoint temporary root");
    k3_engine_stats startup;
    CHECK(k3_chat_session_create(
              &session, &config, &startup,
              error, sizeof(error)), error);
    const k3_chat_message first_messages[] = {
        {.role = K3_CHAT_ROLE_USER, .content = "Say hello."},
    };
    CHECK(complete(session, first_messages, 1u, &first,
                   error, sizeof(error)), error);
    first_response = strdup(first.response.data);
    CHECK(first_response != NULL, "copy first response");
    k3_chat_checkpoint_result published;
    CHECK(k3_chat_session_publish_checkpoint(
              session, &published, error, sizeof(error)) &&
              published.enabled && published.published &&
              published.token_count == first.position &&
              published.entry_count == 1u,
          error);

    const k3_chat_message continuation[] = {
        {.role = K3_CHAT_ROLE_USER, .content = "Say hello."},
        {.role = K3_CHAT_ROLE_ASSISTANT, .content = first_response},
        {.role = K3_CHAT_ROLE_USER, .content = "Now say goodbye."},
    };
    CHECK(complete(session, continuation, 3u, &expected,
                   error, sizeof(error)), error);
    CHECK(!expected.prompt_reused_checkpoint &&
              expected.prompt_reused_tokens == first.position &&
              expected.state_digest_valid,
          "live exact reuse did not remain first priority");
    expected_response = strdup(expected.response.data);
    CHECK(expected_response != NULL, "copy expected response");
    expected_digest = expected.state_digest;

    const k3_chat_message unrelated_messages[] = {
        {.role = K3_CHAT_ROLE_USER, .content = "What is two plus two?"},
    };
    CHECK(complete(session, unrelated_messages, 1u, &unrelated,
                   error, sizeof(error)), error);
    CHECK(complete(session, continuation, 3u, &restored,
                   error, sizeof(error)), error);
    CHECK(restored.prompt_reused_checkpoint &&
              restored.prompt_reused_tokens == first.position &&
              restored.checkpoint_import_seconds > 0.0 &&
              strcmp(restored.response.data, expected_response) == 0 &&
              restored.state_digest_valid &&
              digest_equal(&restored.state_digest, &expected_digest),
          "displaced checkpoint continuation changed");

    k3_chat_session_destroy(session);
    session = NULL;
    CHECK(k3_chat_session_create(
              &session, &config, &startup,
              error, sizeof(error)) &&
              k3_chat_session_checkpoint_count(session) == 1u,
          error);
    CHECK(complete(session, continuation, 3u, &restarted,
                   error, sizeof(error)), error);
    CHECK(restarted.prompt_reused_checkpoint &&
              restarted.prompt_reused_tokens == first.position &&
              strcmp(restarted.response.data, expected_response) == 0 &&
              restarted.state_digest_valid &&
              digest_equal(&restarted.state_digest, &expected_digest),
          "restart checkpoint continuation changed");

    printf("K3 durable exact-prefix checkpoint: PASS\n");
    printf("  checkpoint=%u tokens export=%.3f s import=%.3f/%.3f s\n",
           published.token_count, published.export_seconds,
           restored.checkpoint_import_seconds,
           restarted.checkpoint_import_seconds);
    status = 0;

cleanup:
    free(expected_response);
    free(first_response);
    k3_chat_turn_result_free(&restarted);
    k3_chat_turn_result_free(&restored);
    k3_chat_turn_result_free(&unrelated);
    k3_chat_turn_result_free(&expected);
    k3_chat_turn_result_free(&first);
    k3_chat_session_destroy(session);
    cleanup_root(root);
    return status;
}
