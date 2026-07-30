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
    int result = 1;
    char error[256];
    static const char request[] =
        "{"
        "\"model\":\"moonshine\","
        "\"messages\":["
        "{\"role\":\"system\",\"content\":\"Be concise.\"},"
        "{\"role\":\"user\",\"content\":\"Hi \\u4f60\\u597d "
        "\\ud83d\\udc4b\"}],"
        "\"max_tokens\":32,"
        "\"stream\":false"
        "}";
    k3_json_document document;
    memset(&document, 0, sizeof(document));
    char *decoded = NULL;
    char *escaped = NULL;
    size_t escaped_size = 0u;

    CHECK(k3_json_parse(
              &document, request, sizeof(request) - 1u,
              error, sizeof(error)),
          error);
    CHECK(document.root >= 0 &&
          document.tokens[document.root].type == K3_JSON_OBJECT,
          "root object");
    const int32_t model = k3_json_object_get(
        &document, document.root, "model");
    CHECK(k3_json_string_equal(
              &document, model, "moonshine"),
          "model string");
    const int32_t messages = k3_json_object_get(
        &document, document.root, "messages");
    CHECK(messages >= 0 &&
          document.tokens[messages].type == K3_JSON_ARRAY &&
          document.tokens[messages].size == 2u,
          "messages array");
    const int32_t user = k3_json_array_get(
        &document, messages, 1u);
    const int32_t content = k3_json_object_get(
        &document, user, "content");
    CHECK(k3_json_string_dup(
              &document, content, &decoded,
              error, sizeof(error)),
          error);
    CHECK(strcmp(decoded, "Hi 你好 👋") == 0,
          "Unicode and surrogate decoding");
    uint32_t max_tokens = 0u;
    CHECK(k3_json_u32(
              &document,
              k3_json_object_get(
                  &document, document.root, "max_tokens"),
              &max_tokens) &&
          max_tokens == 32u,
          "unsigned integer");
    bool stream = true;
    CHECK(k3_json_bool(
              &document,
              k3_json_object_get(
                  &document, document.root, "stream"),
              &stream) &&
          !stream,
          "boolean");
    static const char response_text[] =
        "quote=\" line\n emoji=👋";
    CHECK(k3_json_escape(
              response_text, strlen(response_text),
              &escaped, &escaped_size,
              error, sizeof(error)),
          error);
    CHECK(strcmp(
              escaped,
              "\"quote=\\\" line\\n emoji=👋\"") == 0 &&
          escaped_size == strlen(escaped),
          "JSON response escaping");

    static const char *invalid[] = {
        "{\"a\":1,}",
        "[1 2]",
        "{\"a\":\"\\x\"}",
        "{\"a\":01}",
        "true false",
        "{\"ignored\":\"\xed\xa0\x80\"}",
        "{\"ignored\":\"\\ud800\"}",
        "{\"ignored\":\"\\udc00\"}",
        "{\"ignored\":\"\\ud800\\u0041\"}",
    };
    for (size_t i = 0u;
         i < sizeof(invalid) / sizeof(invalid[0]);
         i++) {
        k3_json_document bad;
        memset(&bad, 0, sizeof(bad));
        CHECK(!k3_json_parse(
                  &bad, invalid[i], strlen(invalid[i]),
                  error, sizeof(error)),
              "malformed JSON was accepted");
        k3_json_document_free(&bad);
    }

    printf("K3 JSON parser/escaper: PASS\n");
    result = 0;

cleanup:
    free(escaped);
    free(decoded);
    k3_json_document_free(&document);
    return result;
}
