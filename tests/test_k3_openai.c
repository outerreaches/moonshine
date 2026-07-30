#include "k3_openai.h"

#include "k3_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", (message)); \
            goto cleanup; \
        } \
    } while (0)

int main(void) {
    int exit_code = 1;
    char error[512];
    static const char request_json[] =
        "{"
        "\"model\":\"moonshine\","
        "\"messages\":["
        "{\"role\":\"developer\",\"content\":\"Be concise.\"},"
        "{\"role\":\"user\",\"content\":\"First\"},"
        "{\"role\":\"assistant\",\"content\":\"Done\"},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"text\",\"text\":\"Say \"},"
        "{\"type\":\"input_text\",\"text\":\"hello 👋\"}]}],"
        "\"max_completion_tokens\":64,"
        "\"stream\":true,"
        "\"n\":1,"
        "\"tools\":[]"
        "}";
    k3_openai_chat_request request;
    memset(&request, 0, sizeof(request));
    k3_chat_turn_result result;
    memset(&result, 0, sizeof(result));
    char *response_json = NULL;
    size_t response_size = 0u;
    k3_json_document response_document;
    memset(&response_document, 0, sizeof(response_document));

    CHECK(k3_openai_parse_chat_request(
              request_json, sizeof(request_json) - 1u,
              &request, error, sizeof(error)),
          error);
    CHECK(strcmp(request.model, MOONSHINE_MODEL_ID) == 0,
          "model");
    CHECK(request.message_count == 4u &&
          request.messages[0].role == K3_CHAT_ROLE_SYSTEM &&
          request.messages[3].role == K3_CHAT_ROLE_USER,
          "message roles");
    CHECK(strcmp(
              request.messages[3].content,
              "Say hello 👋") == 0,
          "content-part concatenation");
    CHECK(request.max_tokens == 64u && request.stream,
          "generation parameters");

    static const char response_text[] =
        "Hello! 👋 \"ready\"";
    result.response.data = strdup(response_text);
    CHECK(result.response.data != NULL,
          "response allocation");
    result.response.size = strlen(result.response.data);
    result.response.capacity = result.response.size + 1u;
    result.prompt_tokens = 24u;
    result.generated_tokens = 18u;
    result.finish_reason = K3_CHAT_FINISH_END_OF_MESSAGE;
    CHECK(k3_openai_build_chat_response(
              "chatcmpl-moonshine-1", 1785326400,
              MOONSHINE_MODEL_ID, &result,
              &response_json, &response_size,
              error, sizeof(error)),
          error);
    CHECK(response_size == strlen(response_json),
          "response size");
    CHECK(k3_json_parse(
              &response_document,
              response_json, response_size,
              error, sizeof(error)),
          error);
    const int32_t choices = k3_json_object_get(
        &response_document, response_document.root, "choices");
    const int32_t choice = k3_json_array_get(
        &response_document, choices, 0u);
    const int32_t message = k3_json_object_get(
        &response_document, choice, "message");
    char *content = NULL;
    CHECK(k3_json_string_dup(
              &response_document,
              k3_json_object_get(
                  &response_document, message, "content"),
              &content, error, sizeof(error)),
          error);
    CHECK(strcmp(content, response_text) == 0,
          "escaped response content");
    free(content);
    content = NULL;
    const int32_t usage = k3_json_object_get(
        &response_document, response_document.root, "usage");
    uint32_t total = 0u;
    CHECK(k3_json_u32(
              &response_document,
              k3_json_object_get(
                  &response_document, usage, "total_tokens"),
              &total) &&
          total == 42u,
          "usage token count");

    static const char *invalid[] = {
        "{\"model\":\"moonshine\",\"messages\":[]}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"assistant\",\"content\":\"x\"}]}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],\"n\":2}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tools\":[{\"type\":\"function\"}]}",
    };
    for (size_t i = 0u;
         i < sizeof(invalid) / sizeof(invalid[0]);
         i++) {
        k3_openai_chat_request bad;
        memset(&bad, 0, sizeof(bad));
        CHECK(!k3_openai_parse_chat_request(
                  invalid[i], strlen(invalid[i]),
                  &bad, error, sizeof(error)),
              "unsupported OpenAI request was accepted");
        k3_openai_chat_request_free(&bad);
    }

    printf("K3 OpenAI request/response codec: PASS\n");
    exit_code = 0;

cleanup:
    k3_json_document_free(&response_document);
    free(response_json);
    free(result.response.data);
    k3_openai_chat_request_free(&request);
    return exit_code;
}
