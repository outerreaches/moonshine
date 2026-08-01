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
        "{\"role\":\"assistant\",\"reasoning_content\":\"Checked.\","
        "\"content\":\"Done\"},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"text\",\"text\":\"Say \"},"
        "{\"type\":\"input_text\",\"text\":\"hello 👋\"}]}],"
        "\"max_completion_tokens\":64,"
        "\"reasoning_effort\":\"low\","
        "\"response_format\":{\"type\":\"json_object\"},"
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
    CHECK(request.max_tokens == 64u && request.stream &&
          request.thinking &&
          request.response_format == K3_RESPONSE_FORMAT_JSON_OBJECT &&
          strcmp(request.reasoning_effort, "low") == 0 &&
          strcmp(request.messages[2].reasoning_content, "Checked.") == 0,
          "generation parameters");

    static const char agent_request_json[] =
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"Check two cities\"},"
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":["
        "{\"id\":\"call_a\",\"type\":\"function\",\"function\":{"
        "\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Toronto\\\"}\"}},"
        "{\"id\":\"call_b\",\"type\":\"function\",\"function\":{"
        "\"name\":\"get_time\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_b\",\"content\":\"14:00\"},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_a\",\"content\":\"22 C\"}],"
        "\"tools\":["
        "{\"type\":\"function\",\"function\":{\"name\":\"get_time\","
        "\"description\":\"Get time\",\"parameters\":{\"type\":\"object\"}}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\"}}}],"
        "\"tool_choice\":\"required\"}";
    k3_openai_chat_request agent_request;
    memset(&agent_request, 0, sizeof(agent_request));
    k3_openai_chat_request specific_request;
    memset(&specific_request, 0, sizeof(specific_request));
    CHECK(k3_openai_parse_chat_request(
              agent_request_json, strlen(agent_request_json),
              &agent_request, error, sizeof(error)),
          error);
    CHECK(agent_request.tool_count == 2u &&
          agent_request.tool_choice == K3_TOOL_CHOICE_REQUIRED &&
          agent_request.messages[2].role == K3_CHAT_ROLE_TOOL &&
          strcmp(agent_request.messages[2].tool_call_id, "call_a") == 0 &&
          strcmp(agent_request.messages[2].name, "get_weather") == 0 &&
          strcmp(agent_request.messages[3].tool_call_id, "call_b") == 0 &&
          strcmp(agent_request.messages[3].name, "get_time") == 0,
          "agent request tool history normalization");
    CHECK(strcmp(
              agent_request.tools_json,
              "[{\"function\":{\"description\":\"Get time\",\"name\":\"get_time\","
              "\"parameters\":{\"type\":\"object\"}},\"type\":\"function\"},"
              "{\"function\":{\"description\":\"Get weather\",\"name\":\"get_weather\","
              "\"parameters\":{\"type\":\"object\"}},\"type\":\"function\"}]") == 0,
          "tool schema canonicalization");
    static const char specific_request_json[] =
        "{\"model\":\"moonshine\",\"messages\":[{\"role\":\"user\","
        "\"content\":\"Use weather\"}],\"tools\":["
        "{\"type\":\"function\",\"function\":{\"name\":\"get_time\","
        "\"parameters\":{\"type\":\"object\"}}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"parameters\":{\"type\":\"object\"}}}],"
        "\"tool_choice\":{\"type\":\"function\",\"function\":{"
        "\"name\":\"get_weather\"}}}";
    CHECK(k3_openai_parse_chat_request(
              specific_request_json, strlen(specific_request_json),
              &specific_request, error, sizeof(error)),
          error);
    CHECK(specific_request.tool_count == 1u &&
          specific_request.tool_choice == K3_TOOL_CHOICE_REQUIRED &&
          strstr(specific_request.tools_json, "get_weather") != NULL &&
          strstr(specific_request.tools_json, "get_time") == NULL,
          "specific tool_choice filtering");

    static const char response_text[] =
        "Hello! 👋 \"ready\"";
    result.response.data = strdup(response_text);
    CHECK(result.response.data != NULL,
          "response allocation");
    result.response.size = strlen(result.response.data);
    result.response.capacity = result.response.size + 1u;
    result.reasoning_content.data = strdup("I checked readiness.");
    CHECK(result.reasoning_content.data != NULL,
          "reasoning allocation");
    result.reasoning_content.size =
        strlen(result.reasoning_content.data);
    result.reasoning_content.capacity =
        result.reasoning_content.size + 1u;
    result.thinking = true;
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
    char *reasoning = NULL;
    CHECK(k3_json_string_dup(
              &response_document,
              k3_json_object_get(
                  &response_document, message, "reasoning_content"),
              &reasoning, error, sizeof(error)) &&
          strcmp(reasoning, "I checked readiness.") == 0,
          "reasoning response content");
    free(reasoning);
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
    CHECK(!k3_openai_validate_response_format(
              &request, &result, error, sizeof(error)),
          "non-JSON structured response was accepted");
    free(result.response.data);
    result.response.data = strdup("{\"ready\":true}");
    CHECK(result.response.data != NULL,
          "structured response allocation");
    result.response.size = strlen(result.response.data);
    result.response.capacity = result.response.size + 1u;
    CHECK(k3_openai_validate_response_format(
              &request, &result, error, sizeof(error)),
          error);

    k3_json_document_free(&response_document);
    memset(&response_document, 0, sizeof(response_document));
    free(response_json);
    response_json = NULL;
    static k3_tool_call result_calls[] = {
        {
            .name = "get_weather",
            .arguments = "{\"city\":\"Toronto\"}",
        },
    };
    free(result.response.data);
    result.response.data = strdup("");
    result.response.size = 0u;
    result.tool_calls = result_calls;
    result.tool_call_count = 1u;
    result.finish_reason = K3_CHAT_FINISH_TOOL_CALLS;
    CHECK(k3_openai_build_chat_response(
              "chatcmpl-moonshine-tools", 1785326401,
              MOONSHINE_MODEL_ID, &result,
              &response_json, &response_size,
              error, sizeof(error)) &&
          k3_json_parse(
              &response_document,
              response_json, response_size,
              error, sizeof(error)),
          error);
    const int32_t tool_choices = k3_json_object_get(
        &response_document, response_document.root, "choices");
    const int32_t tool_choice = k3_json_array_get(
        &response_document, tool_choices, 0u);
    CHECK(k3_json_string_equal(
              &response_document,
              k3_json_object_get(
                  &response_document, tool_choice, "finish_reason"),
              "tool_calls"),
          "tool response finish reason");
    const int32_t tool_message = k3_json_object_get(
        &response_document, tool_choice, "message");
    const int32_t calls = k3_json_object_get(
        &response_document, tool_message, "tool_calls");
    const int32_t call = k3_json_array_get(
        &response_document, calls, 0u);
    const int32_t function = k3_json_object_get(
        &response_document, call, "function");
    CHECK(k3_json_string_equal(
              &response_document,
              k3_json_object_get(
                  &response_document, function, "name"),
              "get_weather") &&
          k3_json_string_equal(
              &response_document,
              k3_json_object_get(
                  &response_document, function, "arguments"),
              "{\"city\":\"Toronto\"}"),
          "tool response function payload");
    result.tool_calls = NULL;
    result.tool_call_count = 0u;

    static const char *invalid[] = {
        "{\"model\":\"moonshine\",\"messages\":[]}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"assistant\",\"content\":\"x\"}]}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],\"n\":2}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tools\":[{\"type\":\"function\"}]}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tool_choice\":\"required\"}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],"
        "\"parallel_tool_calls\":false}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],"
        "\"reasoning_effort\":\"medium\"}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"x\"}],"
        "\"response_format\":{\"type\":\"json_schema\","
        "\"json_schema\":{\"name\":\"x\",\"schema\":{}}}}",
        "{\"model\":\"moonshine\",\"messages\":["
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":["
        "{\"id\":\"a\",\"type\":\"function\",\"function\":{"
        "\"name\":\"f\",\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"wrong\",\"content\":\"x\"}]}",
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
    free(result.reasoning_content.data);
    free(result.response.data);
    k3_openai_chat_request_free(&request);
    k3_openai_chat_request_free(&agent_request);
    k3_openai_chat_request_free(&specific_request);
    return exit_code;
}
