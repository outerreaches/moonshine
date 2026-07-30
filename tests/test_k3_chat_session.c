#include "k3_chat.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", (message)); \
            goto cleanup; \
        } \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s /path/to/moonshotai__Kimi-K3\n",
                argv[0]);
        return 2;
    }

    int result = 1;
    char error[512];
    k3_chat_session *session = NULL;
    k3_chat_turn_result turn = { 0 };
    k3_engine_stats engine_stats;
    const k3_chat_session_config config = {
        .model_root = argv[1],
        .context = 8192u,
        .sequential_prefill_limit = 128u,
        .experts_per_layer = 32u,
        .staging_slots = 16u,
        .q8_projections = true,
    };
    CHECK(k3_chat_session_create(
              &session, &config, &engine_stats,
              error, sizeof(error)),
          error);
    CHECK(k3_chat_session_turn(
              session, "Say hello.", 32u,
              NULL, NULL, &turn, error, sizeof(error)),
          error);
    CHECK(turn.response.data != NULL &&
          strcmp(
              turn.response.data,
              "Hello! 👋 How can I help you today?") == 0,
          "decoded chat response changed");
    CHECK(turn.prompt_tokens == 24u,
          "chat prompt no longer has 24 official XTML tokens");
    CHECK(turn.generated_tokens == 18u,
          "chat completion no longer has 18 tokens including EOM");
    CHECK(turn.forced_trailer_tokens == 0u &&
          turn.finish_reason == K3_CHAT_FINISH_END_OF_MESSAGE,
          "chat completion did not stop naturally");
    CHECK(turn.prefill_strategy == K3_CHAT_PREFILL_SEQUENTIAL,
          "short chat prompt did not use token-major execution");
    CHECK(turn.position == 42u,
          "chat session did not commit the complete turn");

    printf("K3 native chat session: PASS\n");
    printf("  response: %s\n", turn.response.data);
    printf("  startup=%.3f s prompt=%u/%.3f s "
           "generated=%u/%.3f s position=%u\n",
           engine_stats.startup_seconds,
           turn.prompt_tokens, turn.prompt_seconds,
           turn.generated_tokens, turn.decode_seconds,
           turn.position);
    result = 0;

cleanup:
    k3_chat_turn_result_free(&turn);
    k3_chat_session_destroy(session);
    return result;
}
