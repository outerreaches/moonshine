#ifndef K3_OPENAI_H
#define K3_OPENAI_H

#include "k3_chat.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MOONSHINE_MODEL_ID "moonshine"
#define MOONSHINE_SYSTEM_FINGERPRINT "moonshine-q8-greedy"

typedef struct {
    char            *model;
    k3_chat_message *messages;
    size_t           message_count;
    uint32_t         max_tokens;
    bool             stream;
} k3_openai_chat_request;

bool k3_openai_parse_chat_request(
    const char              *json,
    size_t                   json_size,
    k3_openai_chat_request  *request,
    char                    *error,
    size_t                   error_size);

void k3_openai_chat_request_free(k3_openai_chat_request *request);

bool k3_openai_build_chat_response(
    const char                 *completion_id,
    time_t                      created,
    const char                 *model,
    const k3_chat_turn_result  *result,
    char                      **json,
    size_t                     *json_size,
    char                       *error,
    size_t                      error_size);

#ifdef __cplusplus
}
#endif

#endif
