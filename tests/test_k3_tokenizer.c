#include "k3_tokenizer.h"

#include <inttypes.h>
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

static uint64_t token_hash(const k3_token_buffer *tokens) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0u; i < tokens->count; i++) {
        const uint32_t token = tokens->data[i];
        for (unsigned shift = 0u; shift < 32u; shift += 8u) {
            hash ^= (uint8_t)(token >> shift);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static bool ids_equal(const k3_token_buffer *actual,
                      const uint32_t *expected,
                      size_t expected_count) {
    return actual->count == expected_count &&
           memcmp(actual->data, expected,
                  expected_count * sizeof(*expected)) == 0;
}

static bool check_text_case(k3_tokenizer *tokenizer,
                            const char *text,
                            const uint32_t *expected,
                            size_t expected_count,
                            char *error,
                            size_t error_size) {
    k3_token_buffer encoded = { 0 };
    k3_text_buffer decoded = { 0 };
    bool ok =
        k3_tokenizer_encode(
            tokenizer, text, false, &encoded, error, error_size) &&
        ids_equal(&encoded, expected, expected_count) &&
        k3_tokenizer_decode(
            tokenizer, encoded.data, encoded.count, true,
            &decoded, error, error_size) &&
        strcmp(decoded.data, text) == 0;
    k3_token_buffer_free(&encoded);
    k3_text_buffer_free(&decoded);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s /path/to/moonshotai__Kimi-K3\n",
                argv[0]);
        return 2;
    }

    int result = 1;
    char error[512] = { 0 };
    k3_tokenizer *tokenizer = NULL;
    k3_token_buffer tokens = { 0 };
    k3_token_buffer completed_history = { 0 };
    k3_token_buffer extended_history = { 0 };
    k3_token_buffer required_prompt = { 0 };
    k3_token_buffer generated_tool_tail = { 0 };
    k3_token_buffer retained_tool_turn = { 0 };
    k3_token_buffer augmented_tool_history = { 0 };
    k3_token_buffer canonical_tool_history = { 0 };
    k3_token_buffer single_tool_prompt = { 0 };
    k3_token_buffer retained_single_tool_turn = { 0 };
    k3_token_buffer augmented_single_tool_history = { 0 };
    k3_token_buffer generated_structured_tail = { 0 };
    k3_token_buffer retained_structured_turn = { 0 };
    k3_token_buffer augmented_structured_history = { 0 };
    k3_token_buffer canonical_structured_history = { 0 };
    k3_token_buffer thinking_prompt = { 0 };
    k3_token_buffer thinking_tail = { 0 };
    k3_token_buffer retained_thinking_turn = { 0 };
    k3_token_buffer continued_thinking_history = { 0 };
    k3_token_buffer structured_prompt = { 0 };
    k3_text_buffer text = { 0 };
    k3_assistant_output assistant_output;
    memset(&assistant_output, 0, sizeof(assistant_output));

    CHECK(k3_tokenizer_create(
              &tokenizer, argv[1], error, sizeof(error)),
          error);

    static const uint32_t say_hello[] = {
        71079u, 40493u, 13u,
    };
    CHECK(check_text_case(
              tokenizer, "Say hello.", say_hello,
              sizeof(say_hello) / sizeof(say_hello[0]),
              error, sizeof(error)),
          "ASCII tokenizer oracle changed");

    static const uint32_t english[] = {
        19180u, 11u, 101782u, 0u, 6806u, 10298u, 220u,
        6694u, 12972u, 22u, 341u, 7820u, 2470u, 13u,
    };
    CHECK(check_text_case(
              tokenizer,
              "Hello, WORLD! I'm testing 1234567.\nNext line.",
              english, sizeof(english) / sizeof(english[0]),
              error, sizeof(error)),
          "English case/contraction/number oracle changed");

    static const uint32_t han[] = {
        33845u, 378u, 2243u,
    };
    CHECK(check_text_case(
              tokenizer, "你好，世界", han,
              sizeof(han) / sizeof(han[0]),
              error, sizeof(error)),
          "Han tokenizer oracle changed");

    static const uint32_t multilingual[] = {
        122583u, 1941u, 51158u, 37484u, 4275u, 27842u,
        14915u, 72819u, 7614u, 4275u, 61561u, 7570u,
        4265u, 1063u, 4275u, 130732u, 233u, 92949u, 121u,
    };
    CHECK(check_text_case(
              tokenizer,
              "café café — नमस्ते — مرحبا — 👋🏽",
              multilingual,
              sizeof(multilingual) / sizeof(multilingual[0]),
              error, sizeof(error)),
          "multilingual tokenizer oracle changed");

    static const uint32_t eom_ordinary[] = {
        27u, 91u, 517u, 5118u, 14222u, 91u, 29u,
    };
    CHECK(k3_tokenizer_encode(
              tokenizer, "<|end_of_msg|>", false,
              &tokens, error, sizeof(error)) &&
          ids_equal(
              &tokens, eom_ordinary,
              sizeof(eom_ordinary) / sizeof(eom_ordinary[0])),
          "ordinary control-like text became a special token");
    CHECK(k3_tokenizer_encode(
              tokenizer, "<|end_of_msg|>", true,
              &tokens, error, sizeof(error)) &&
          tokens.count == 1u &&
          tokens.data[0] == K3_TOKEN_END_OF_MSG,
          "allowed special token did not map to the control ID");

    static const k3_chat_message hello_messages[] = {
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Say hello.",
        },
    };
    const k3_chat_options nonthinking = {
        .add_generation_prompt = true,
        .thinking = false,
        .thinking_effort = NULL,
    };
    static const uint32_t hello_prompt[] = {
        163587u, 2778u, 6244u, 878u, 2482u, 1u,
        163589u, 71079u, 40493u, 13u,
        163588u, 2778u, 163589u, 163586u,
        163587u, 2778u, 6244u, 878u, 69702u, 1u,
        163589u, 163587u, 12092u, 163589u,
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, hello_messages,
              sizeof(hello_messages) / sizeof(hello_messages[0]),
              &nonthinking, &tokens, error, sizeof(error)) &&
          ids_equal(
              &tokens, hello_prompt,
              sizeof(hello_prompt) / sizeof(hello_prompt[0])),
          "official non-thinking Say hello XTML oracle changed");

    static const char hello_rendered[] =
        "<|open|>message role=\"user\"<|sep|>Say hello."
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"assistant\"<|sep|>"
        "<|open|>response<|sep|>";
    CHECK(k3_tokenizer_decode(
              tokenizer, tokens.data, tokens.count, true,
              &text, error, sizeof(error)) &&
          strcmp(text.data, hello_rendered) == 0,
          "non-thinking XTML rendering changed");

    const k3_chat_options json_object_options = {
        .add_generation_prompt = true,
        .thinking = false,
        .response_format = K3_RESPONSE_FORMAT_JSON_OBJECT,
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, hello_messages,
              sizeof(hello_messages) / sizeof(hello_messages[0]),
              &json_object_options, &structured_prompt,
              error, sizeof(error)) &&
          k3_tokenizer_decode(
              tokenizer, structured_prompt.data,
              structured_prompt.count, true,
              &text, error, sizeof(error)),
          error);
    CHECK(strstr(
              text.data,
              "<|open|>message role=\"system\" "
              "type=\"response-format\"<|sep|>"
              "The system is invoked with "
              "`response_format=json_object`.\n"
              "Your response must be raw JSON data without markdown code "
              "blocks (```json) or any additional formatting."
              "<|close|>message<|sep|><|end_of_msg|>"
              "<|open|>message role=\"assistant\"") != NULL,
          "official K3 json_object directive rendering changed");

    static const char response_schema[] =
        "{\"additionalProperties\":false,\"properties\":{"
        "\"count\":{\"type\":\"integer\"},"
        "\"greeting\":{\"type\":\"string\"}},"
        "\"required\":[\"greeting\",\"count\"],\"type\":\"object\"}";
    const k3_chat_options json_schema_options = {
        .add_generation_prompt = true,
        .thinking = false,
        .response_format = K3_RESPONSE_FORMAT_JSON_SCHEMA,
        .response_schema_json = response_schema,
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, hello_messages,
              sizeof(hello_messages) / sizeof(hello_messages[0]),
              &json_schema_options, &structured_prompt,
              error, sizeof(error)) &&
          k3_tokenizer_decode(
              tokenizer, structured_prompt.data,
              structured_prompt.count, true,
              &text, error, sizeof(error)),
          error);
    CHECK(strstr(
              text.data,
              "<|open|>message role=\"system\" "
              "type=\"response-format\"<|sep|>"
              "The system is invoked with "
              "`response_format=json_schema`.\n"
              "Your response must be raw JSON data without markdown code "
              "blocks (```json) or any additional formatting.\n"
              "The JSON data must match the following schema:\n"
              "```json\n"
              "{\"additionalProperties\":false,\"properties\":{"
              "\"count\":{\"type\":\"integer\"},"
              "\"greeting\":{\"type\":\"string\"}},"
              "\"required\":[\"greeting\",\"count\"],"
              "\"type\":\"object\"}\n```"
              "<|close|>message<|sep|><|end_of_msg|>"
              "<|open|>message role=\"assistant\"") != NULL,
          "official K3 json_schema directive rendering changed");

    static const char generated_structured_output[] =
        "{\"greeting\":\"hello\",\"count\":1}"
        "<|close|>response<|sep|><|close|>message<|sep|>"
        "<|end_of_msg|>";
    CHECK(k3_tokenizer_encode(
              tokenizer, generated_structured_output, true,
              &generated_structured_tail, error, sizeof(error)),
          error);
    retained_structured_turn.count =
        structured_prompt.count + generated_structured_tail.count;
    retained_structured_turn.capacity = retained_structured_turn.count;
    retained_structured_turn.data = (uint32_t *)malloc(
        retained_structured_turn.count *
            sizeof(*retained_structured_turn.data));
    CHECK(retained_structured_turn.data != NULL,
          "allocating retained structured-turn oracle failed");
    memcpy(
        retained_structured_turn.data,
        structured_prompt.data,
        structured_prompt.count * sizeof(*structured_prompt.data));
    memcpy(
        retained_structured_turn.data + structured_prompt.count,
        generated_structured_tail.data,
        generated_structured_tail.count *
            sizeof(*generated_structured_tail.data));
    static const k3_chat_message structured_history[] = {
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Say hello.",
        },
        {
            .role = K3_CHAT_ROLE_ASSISTANT,
            .content = "{\"greeting\":\"hello\",\"count\":1}",
        },
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Return another JSON object.",
        },
    };
    static const k3_response_format_marker schema_marker[] = {
        {
            .after_message_count = 1u,
            .format = K3_RESPONSE_FORMAT_JSON_SCHEMA,
            .response_schema_json = response_schema,
        },
    };
    const k3_chat_options augmented_structured_options = {
        .add_generation_prompt = true,
        .thinking = false,
        .response_format = K3_RESPONSE_FORMAT_JSON_OBJECT,
        .historical_response_formats = schema_marker,
        .historical_response_format_count =
            sizeof(schema_marker) / sizeof(schema_marker[0]),
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, structured_history,
              sizeof(structured_history) / sizeof(structured_history[0]),
              &augmented_structured_options,
              &augmented_structured_history,
              error, sizeof(error)) &&
          augmented_structured_history.count >
              retained_structured_turn.count &&
          memcmp(
              augmented_structured_history.data,
              retained_structured_turn.data,
              retained_structured_turn.count *
                  sizeof(*retained_structured_turn.data)) == 0,
          "historical response format did not reconstruct causal prefix");
    k3_chat_options canonical_structured_options =
        augmented_structured_options;
    canonical_structured_options.historical_response_formats = NULL;
    canonical_structured_options.historical_response_format_count = 0u;
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, structured_history,
              sizeof(structured_history) / sizeof(structured_history[0]),
              &canonical_structured_options,
              &canonical_structured_history,
              error, sizeof(error)) &&
          (canonical_structured_history.count <
               retained_structured_turn.count ||
           memcmp(
               canonical_structured_history.data,
               retained_structured_turn.data,
               retained_structured_turn.count *
                   sizeof(*retained_structured_turn.data)) != 0),
          "canonical structured history unexpectedly retained a directive");
    static const k3_response_format_marker invalid_format_marker[] = {
        {
            .after_message_count = 1u,
            .format = K3_RESPONSE_FORMAT_TEXT,
        },
    };
    k3_chat_options invalid_response_marker_options =
        augmented_structured_options;
    invalid_response_marker_options.historical_response_formats =
        invalid_format_marker;
    CHECK(!k3_tokenizer_encode_chat(
              tokenizer, structured_history,
              sizeof(structured_history) / sizeof(structured_history[0]),
              &invalid_response_marker_options, &tokens,
              error, sizeof(error)),
          "a text historical response-format marker was accepted");
    static const k3_response_format_marker missing_schema_marker[] = {
        {
            .after_message_count = 1u,
            .format = K3_RESPONSE_FORMAT_JSON_SCHEMA,
            .response_schema_json = NULL,
        },
    };
    invalid_response_marker_options.historical_response_formats =
        missing_schema_marker;
    CHECK(!k3_tokenizer_encode_chat(
              tokenizer, structured_history,
              sizeof(structured_history) / sizeof(structured_history[0]),
              &invalid_response_marker_options, &tokens,
              error, sizeof(error)),
          "an empty historical response schema was accepted");

    const k3_chat_options thinking_max = {
        .add_generation_prompt = true,
        .thinking = true,
        .thinking_effort = "max",
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, hello_messages,
              sizeof(hello_messages) / sizeof(hello_messages[0]),
              &thinking_max, &tokens, error, sizeof(error)) &&
          tokens.count == 91u &&
          token_hash(&tokens) == UINT64_C(0xe215c5053c6bbc41),
          "official max-thinking XTML oracle changed");

    static const k3_chat_message multi_messages[] = {
        {
            .role = K3_CHAT_ROLE_SYSTEM,
            .content = "Be concise.",
        },
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Literal <|end_of_msg|> stays text.",
        },
        {
            .role = K3_CHAT_ROLE_ASSISTANT,
            .content = "Understood.",
            .reasoning_content = "brief",
        },
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Count 1, 2, 3.",
        },
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, multi_messages,
              sizeof(multi_messages) / sizeof(multi_messages[0]),
              &nonthinking, &tokens, error, sizeof(error)) &&
          tokens.count == 86u &&
          token_hash(&tokens) == UINT64_C(0xfd64b7407dbb9baf),
          "multi-turn non-thinking XTML oracle changed");

    const k3_chat_options thinking_without_effort = {
        .add_generation_prompt = true,
        .thinking = true,
        .thinking_effort = NULL,
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, multi_messages,
              sizeof(multi_messages) / sizeof(multi_messages[0]),
              &thinking_without_effort, &tokens,
              error, sizeof(error)) &&
          tokens.count == 93u &&
          token_hash(&tokens) == UINT64_C(0x761236cf0f932769),
          "multi-turn thinking XTML oracle changed");

    CHECK(k3_tokenizer_decode(
              tokenizer, tokens.data, tokens.count, true,
              &text, error, sizeof(error)),
          error);
    CHECK(strstr(
              text.data,
              "Literal <|end_of_msg|> stays text.") != NULL,
          "user text containing a marker was not preserved literally");

    static const k3_chat_message completed_messages[] = {
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Say hello.",
        },
        {
            .role = K3_CHAT_ROLE_ASSISTANT,
            .content = "Hello! 👋 How can I help you today?",
        },
    };
    static const k3_chat_message extended_messages[] = {
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Say hello.",
        },
        {
            .role = K3_CHAT_ROLE_ASSISTANT,
            .content = "Hello! 👋 How can I help you today?",
        },
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Say hello again.",
        },
    };
    const k3_chat_options completed_options = {
        .add_generation_prompt = false,
        .thinking = false,
        .thinking_effort = NULL,
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, completed_messages,
              sizeof(completed_messages) /
                  sizeof(completed_messages[0]),
              &completed_options, &completed_history,
              error, sizeof(error)),
          error);
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, extended_messages,
              sizeof(extended_messages) /
                  sizeof(extended_messages[0]),
              &nonthinking, &extended_history,
              error, sizeof(error)),
          error);
    CHECK(extended_history.count >
              completed_history.count &&
          memcmp(
              extended_history.data,
              completed_history.data,
              completed_history.count *
                  sizeof(*completed_history.data)) == 0,
          "append-only history is not an exact XTML token-prefix extension");

    static const char tools_json[] =
        "[{\"function\":{\"description\":\"Get weather\","
        "\"name\":\"get_weather\",\"parameters\":{"
        "\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"],\"type\":\"object\"}},"
        "\"type\":\"function\"}]";
    static k3_tool_call weather_calls[] = {
        {
            .id = "call_abc",
            .name = "get_weather",
            .arguments = "{\"city\":\"Toronto\"}",
        },
    };
    static const k3_chat_message tool_history[] = {
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Weather in Toronto?",
        },
        {
            .role = K3_CHAT_ROLE_ASSISTANT,
            .content = "",
            .tool_calls = weather_calls,
            .tool_call_count = 1u,
        },
        {
            .role = K3_CHAT_ROLE_TOOL,
            .content = "{\"temperature_c\":22}",
            .name = "get_weather",
            .tool_call_id = "call_abc",
        },
    };
    const k3_chat_options tool_options = {
        .add_generation_prompt = true,
        .thinking = false,
        .thinking_effort = NULL,
        .tools_json = tools_json,
        .tool_choice = K3_TOOL_CHOICE_AUTO,
    };
    static const char expected_tool_history[] =
        "<|open|>message role=\"system\" type=\"tool-declare\"<|sep|>"
        "# Tools\nHere are the available tools, described in JSONSchema.\n\n"
        "```json\n"
        "[{\"function\":{\"description\":\"Get weather\","
        "\"name\":\"get_weather\",\"parameters\":{"
        "\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"],\"type\":\"object\"}},"
        "\"type\":\"function\"}]\n```"
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"user\"<|sep|>Weather in Toronto?"
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"assistant\"<|sep|>"
        "<|open|>response<|sep|><|close|>response<|sep|>"
        "<|open|>tools<|sep|>"
        "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
        "<|open|>argument key=\"city\" type=\"string\"<|sep|>Toronto"
        "<|close|>argument<|sep|><|close|>call<|sep|>"
        "<|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"tool\" tool=\"get_weather\" index=\"1\""
        "<|sep|>{\"temperature_c\":22}"
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"assistant\"<|sep|>"
        "<|open|>response<|sep|>";
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, tool_history,
              sizeof(tool_history) / sizeof(tool_history[0]),
              &tool_options, &tokens, error, sizeof(error)),
          error);
    CHECK(k3_tokenizer_decode(
              tokenizer, tokens.data, tokens.count, true,
              &text, error, sizeof(error)) &&
          strcmp(text.data, expected_tool_history) == 0,
          "official K3 tool history XTML rendering changed");

    static const k3_chat_message required_tool_request[] = {
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Weather in Toronto?",
        },
    };
    const k3_chat_options required_tool_options = {
        .add_generation_prompt = true,
        .thinking = false,
        .thinking_effort = NULL,
        .tools_json = tools_json,
        .tool_choice = K3_TOOL_CHOICE_REQUIRED,
    };
    static const char generated_weather_call[] =
        "<|close|>response<|sep|><|open|>tools<|sep|>"
        "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
        "<|open|>argument key=\"city\" type=\"string\"<|sep|>Toronto"
        "<|close|>argument<|sep|><|close|>call<|sep|>"
        "<|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>";
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, required_tool_request,
              sizeof(required_tool_request) /
                  sizeof(required_tool_request[0]),
              &required_tool_options, &required_prompt,
              error, sizeof(error)) &&
          k3_tokenizer_encode(
              tokenizer, generated_weather_call, true,
              &generated_tool_tail, error, sizeof(error)),
          error);
    retained_tool_turn.count =
        required_prompt.count + generated_tool_tail.count;
    retained_tool_turn.capacity = retained_tool_turn.count;
    retained_tool_turn.data = (uint32_t *)malloc(
        retained_tool_turn.count *
            sizeof(*retained_tool_turn.data));
    CHECK(retained_tool_turn.data != NULL,
          "allocating retained tool-turn oracle failed");
    memcpy(
        retained_tool_turn.data, required_prompt.data,
        required_prompt.count * sizeof(*required_prompt.data));
    memcpy(
        retained_tool_turn.data + required_prompt.count,
        generated_tool_tail.data,
        generated_tool_tail.count *
            sizeof(*generated_tool_tail.data));

    static const k3_tool_choice_marker required_marker[] = {
        {
            .after_message_count = 1u,
            .choice = K3_TOOL_CHOICE_REQUIRED,
        },
    };
    const k3_chat_options augmented_tool_options = {
        .add_generation_prompt = true,
        .thinking = false,
        .thinking_effort = NULL,
        .tools_json = tools_json,
        .tool_choice = K3_TOOL_CHOICE_AUTO,
        .historical_tool_choices = required_marker,
        .historical_tool_choice_count =
            sizeof(required_marker) / sizeof(required_marker[0]),
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, tool_history,
              sizeof(tool_history) / sizeof(tool_history[0]),
              &augmented_tool_options, &augmented_tool_history,
              error, sizeof(error)) &&
          augmented_tool_history.count > retained_tool_turn.count &&
          memcmp(
              augmented_tool_history.data,
              retained_tool_turn.data,
              retained_tool_turn.count *
                  sizeof(*retained_tool_turn.data)) == 0,
          "historical tool choice did not reconstruct the causal prefix");
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, tool_history,
              sizeof(tool_history) / sizeof(tool_history[0]),
              &tool_options, &canonical_tool_history,
              error, sizeof(error)) &&
          (canonical_tool_history.count < retained_tool_turn.count ||
           memcmp(
               canonical_tool_history.data,
               retained_tool_turn.data,
               retained_tool_turn.count *
                   sizeof(*retained_tool_turn.data)) != 0),
          "canonical history unexpectedly retained a hidden directive");

    k3_chat_options single_tool_options = required_tool_options;
    single_tool_options.enforce_single_tool_call = true;
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, required_tool_request,
              sizeof(required_tool_request) /
                  sizeof(required_tool_request[0]),
              &single_tool_options, &single_tool_prompt,
              error, sizeof(error)) &&
          k3_tokenizer_decode(
              tokenizer, single_tool_prompt.data,
              single_tool_prompt.count, true,
              &text, error, sizeof(error)) &&
          strstr(
              text.data,
              "<|open|>message role=\"system\" "
              "type=\"parallel-tool-calls\"<|sep|>"
              "The system is invoked with "
              "`parallel_tool_calls=false`.\n"
              "You MUST make at most one tool call in the next message. "
              "If more than one tool could help, choose only the single "
              "best tool.<|close|>message<|sep|><|end_of_msg|>") != NULL,
          "single-tool-call directive rendering changed");
    retained_single_tool_turn.count =
        single_tool_prompt.count + generated_tool_tail.count;
    retained_single_tool_turn.capacity =
        retained_single_tool_turn.count;
    retained_single_tool_turn.data = (uint32_t *)malloc(
        retained_single_tool_turn.count *
            sizeof(*retained_single_tool_turn.data));
    CHECK(retained_single_tool_turn.data != NULL,
          "allocating retained single-tool oracle failed");
    memcpy(
        retained_single_tool_turn.data,
        single_tool_prompt.data,
        single_tool_prompt.count * sizeof(*single_tool_prompt.data));
    memcpy(
        retained_single_tool_turn.data + single_tool_prompt.count,
        generated_tool_tail.data,
        generated_tool_tail.count * sizeof(*generated_tool_tail.data));
    static const k3_single_tool_call_marker single_marker[] = {
        { .after_message_count = 1u },
    };
    k3_chat_options augmented_single_options = augmented_tool_options;
    augmented_single_options.historical_single_tool_calls = single_marker;
    augmented_single_options.historical_single_tool_call_count =
        sizeof(single_marker) / sizeof(single_marker[0]);
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, tool_history,
              sizeof(tool_history) / sizeof(tool_history[0]),
              &augmented_single_options, &augmented_single_tool_history,
              error, sizeof(error)) &&
          augmented_single_tool_history.count >
              retained_single_tool_turn.count &&
          memcmp(
              augmented_single_tool_history.data,
              retained_single_tool_turn.data,
              retained_single_tool_turn.count *
                  sizeof(*retained_single_tool_turn.data)) == 0,
          "single-tool directive did not reconstruct the causal prefix");

    static const k3_tool_choice_marker invalid_choice_marker[] = {
        {
            .after_message_count = 1u,
            .choice = K3_TOOL_CHOICE_AUTO,
        },
    };
    k3_chat_options invalid_marker_options = augmented_tool_options;
    invalid_marker_options.historical_tool_choices =
        invalid_choice_marker;
    CHECK(!k3_tokenizer_encode_chat(
              tokenizer, tool_history,
              sizeof(tool_history) / sizeof(tool_history[0]),
              &invalid_marker_options, &tokens,
              error, sizeof(error)),
          "an auto historical tool-choice marker was accepted");
    static const k3_tool_choice_marker invalid_boundary_marker[] = {
        {
            .after_message_count = 4u,
            .choice = K3_TOOL_CHOICE_REQUIRED,
        },
    };
    invalid_marker_options.historical_tool_choices =
        invalid_boundary_marker;
    CHECK(!k3_tokenizer_encode_chat(
              tokenizer, tool_history,
              sizeof(tool_history) / sizeof(tool_history[0]),
              &invalid_marker_options, &tokens,
              error, sizeof(error)),
          "an out-of-range historical tool-choice marker was accepted");
    static const k3_tool_choice_marker unordered_markers[] = {
        {
            .after_message_count = 2u,
            .choice = K3_TOOL_CHOICE_NONE,
        },
        {
            .after_message_count = 1u,
            .choice = K3_TOOL_CHOICE_REQUIRED,
        },
    };
    invalid_marker_options.historical_tool_choices = unordered_markers;
    invalid_marker_options.historical_tool_choice_count =
        sizeof(unordered_markers) / sizeof(unordered_markers[0]);
    CHECK(!k3_tokenizer_encode_chat(
              tokenizer, tool_history,
              sizeof(tool_history) / sizeof(tool_history[0]),
              &invalid_marker_options, &tokens,
              error, sizeof(error)),
          "unordered historical tool-choice markers were accepted");
    static const k3_single_tool_call_marker invalid_single_marker[] = {
        { .after_message_count = 4u },
    };
    invalid_marker_options = augmented_tool_options;
    invalid_marker_options.historical_single_tool_calls =
        invalid_single_marker;
    invalid_marker_options.historical_single_tool_call_count =
        sizeof(invalid_single_marker) / sizeof(invalid_single_marker[0]);
    CHECK(!k3_tokenizer_encode_chat(
              tokenizer, tool_history,
              sizeof(tool_history) / sizeof(tool_history[0]),
              &invalid_marker_options, &tokens,
              error, sizeof(error)),
          "an out-of-range single-tool marker was accepted");

    static const char generated_tool_output[] =
        "<|close|>response<|sep|><|open|>tools<|sep|>"
        "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
        "<|open|>argument key=\"city\" type=\"string\"<|sep|>Toronto"
        "<|close|>argument<|sep|>"
        "<|open|>argument key=\"days\" type=\"number\"<|sep|>2"
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|><|close|>tools<|sep|>"
        "<|close|>message<|sep|><|end_of_msg|>";
    CHECK(k3_tokenizer_encode(
              tokenizer, generated_tool_output, true,
              &tokens, error, sizeof(error)) &&
          k3_tokenizer_parse_assistant_output(
              tokenizer, tokens.data, tokens.count,
              false,
              &assistant_output, error, sizeof(error)),
          error);
    CHECK(assistant_output.response.size == 0u &&
          assistant_output.tool_call_count == 1u &&
          strcmp(assistant_output.tool_calls[0].name,
                 "get_weather") == 0 &&
          strcmp(assistant_output.tool_calls[0].arguments,
                 "{\"city\":\"Toronto\",\"days\":2}") == 0,
          "generated XTML tool call parse changed");
    k3_assistant_output_free(&assistant_output);
    static const char generated_raw_tool_output[] =
        "<|close|>response<|sep|><|open|>tools<|sep|>"
        "<|open|>call tool=\"raw_tool\" index=\"1\"<|sep|>"
        "<|open|>json type=\"object\"<|sep|>{not valid json"
        "<|close|>json<|sep|><|close|>call<|sep|>"
        "<|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>";
    CHECK(k3_tokenizer_encode(
              tokenizer, generated_raw_tool_output, true,
              &tokens, error, sizeof(error)) &&
          k3_tokenizer_parse_assistant_output(
              tokenizer, tokens.data, tokens.count,
              false,
              &assistant_output, error, sizeof(error)),
          error);
    CHECK(assistant_output.tool_call_count == 1u &&
          strcmp(assistant_output.tool_calls[0].arguments,
                 "{not valid json") == 0,
          "raw XTML json argument block was not preserved");
    k3_assistant_output_free(&assistant_output);

    static const char generated_thinking_output[] =
        "I should answer briefly."
        "<|close|>think<|sep|><|open|>response<|sep|>Hello."
        "<|close|>response<|sep|>"
        "<|close|>message<|sep|><|end_of_msg|>";
    CHECK(k3_tokenizer_encode(
              tokenizer, generated_thinking_output, true,
              &tokens, error, sizeof(error)) &&
          k3_tokenizer_parse_assistant_output(
              tokenizer, tokens.data, tokens.count, true,
              &assistant_output, error, sizeof(error)),
          error);
    CHECK(strcmp(
              assistant_output.reasoning.data,
              "I should answer briefly.") == 0 &&
          strcmp(assistant_output.response.data, "Hello.") == 0 &&
          assistant_output.tool_call_count == 0u,
          "thinking XTML reasoning/response parse changed");

    static const k3_chat_message first_thinking_request[] = {
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Why?",
        },
    };
    static const k3_chat_message continued_thinking_messages[] = {
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Why?",
        },
        {
            .role = K3_CHAT_ROLE_ASSISTANT,
            .reasoning_content = "I should answer briefly.",
            .content = "Hello.",
        },
        {
            .role = K3_CHAT_ROLE_USER,
            .content = "Continue.",
        },
    };
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, first_thinking_request,
              sizeof(first_thinking_request) /
                  sizeof(first_thinking_request[0]),
              &thinking_max, &thinking_prompt,
              error, sizeof(error)) &&
          k3_tokenizer_encode(
              tokenizer, generated_thinking_output, true,
              &thinking_tail, error, sizeof(error)),
          error);
    retained_thinking_turn.count =
        thinking_prompt.count + thinking_tail.count;
    retained_thinking_turn.capacity = retained_thinking_turn.count;
    retained_thinking_turn.data = (uint32_t *)malloc(
        retained_thinking_turn.count *
            sizeof(*retained_thinking_turn.data));
    CHECK(retained_thinking_turn.data != NULL,
          "allocating retained thinking-turn oracle failed");
    memcpy(
        retained_thinking_turn.data, thinking_prompt.data,
        thinking_prompt.count * sizeof(*thinking_prompt.data));
    memcpy(
        retained_thinking_turn.data + thinking_prompt.count,
        thinking_tail.data,
        thinking_tail.count * sizeof(*thinking_tail.data));
    CHECK(k3_tokenizer_encode_chat(
              tokenizer, continued_thinking_messages,
              sizeof(continued_thinking_messages) /
                  sizeof(continued_thinking_messages[0]),
              &thinking_max, &continued_thinking_history,
              error, sizeof(error)) &&
          continued_thinking_history.count >
              retained_thinking_turn.count &&
          memcmp(
              continued_thinking_history.data,
              retained_thinking_turn.data,
              retained_thinking_turn.count *
                  sizeof(*retained_thinking_turn.data)) == 0,
          "preserved reasoning history is not an exact causal prefix");

    printf("K3 native tokenizer/XTML: PASS\n");
    printf("  vocab=%u base=%u; ICU Unicode pre-tokenizer\n",
           K3_TOKEN_VOCAB_SIZE, 163584u);
    printf("  official oracles: ASCII, contractions/numbers, Han, "
           "multilingual, special-token safety\n");
    printf("  XTML: non-thinking hello exact; thinking/max and "
           "multi-turn hashes exact; tools/results, append-prefix, and "
           "hidden request-directive prefixes exact\n");
    result = 0;

cleanup:
    k3_text_buffer_free(&text);
    k3_assistant_output_free(&assistant_output);
    k3_token_buffer_free(&tokens);
    k3_token_buffer_free(&completed_history);
    k3_token_buffer_free(&extended_history);
    k3_token_buffer_free(&required_prompt);
    k3_token_buffer_free(&generated_tool_tail);
    k3_token_buffer_free(&retained_tool_turn);
    k3_token_buffer_free(&augmented_tool_history);
    k3_token_buffer_free(&canonical_tool_history);
    k3_token_buffer_free(&single_tool_prompt);
    k3_token_buffer_free(&retained_single_tool_turn);
    k3_token_buffer_free(&augmented_single_tool_history);
    k3_token_buffer_free(&generated_structured_tail);
    k3_token_buffer_free(&retained_structured_turn);
    k3_token_buffer_free(&augmented_structured_history);
    k3_token_buffer_free(&canonical_structured_history);
    k3_token_buffer_free(&thinking_prompt);
    k3_token_buffer_free(&thinking_tail);
    k3_token_buffer_free(&retained_thinking_turn);
    k3_token_buffer_free(&continued_thinking_history);
    k3_token_buffer_free(&structured_prompt);
    k3_tokenizer_destroy(tokenizer);
    return result;
}
