#ifndef K3_JSON_H
#define K3_JSON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    K3_JSON_OBJECT = 1,
    K3_JSON_ARRAY = 2,
    K3_JSON_STRING = 3,
    K3_JSON_NUMBER = 4,
    K3_JSON_TRUE = 5,
    K3_JSON_FALSE = 6,
    K3_JSON_NULL = 7,
} k3_json_type;

typedef struct {
    k3_json_type type;
    size_t       start;
    size_t       end;
    size_t       size;
    int32_t      parent;
    int32_t      first_child;
    int32_t      next_sibling;
    int32_t      last_child;
} k3_json_token;

typedef struct {
    const char    *source;
    size_t         source_size;
    k3_json_token *tokens;
    size_t         token_count;
    size_t         token_capacity;
    int32_t        root;
} k3_json_document;

bool k3_json_parse(k3_json_document *document,
                   const char       *source,
                   size_t            source_size,
                   char             *error,
                   size_t            error_size);

void k3_json_document_free(k3_json_document *document);

int32_t k3_json_object_get(const k3_json_document *document,
                           int32_t                 object,
                           const char             *key);

int32_t k3_json_array_get(const k3_json_document *document,
                          int32_t                 array,
                          size_t                  index);

bool k3_json_string_equal(const k3_json_document *document,
                          int32_t                 token,
                          const char             *value);

bool k3_json_string_dup(const k3_json_document *document,
                        int32_t                 token,
                        char                  **value,
                        char                   *error,
                        size_t                  error_size);

bool k3_json_u32(const k3_json_document *document,
                 int32_t                 token,
                 uint32_t               *value);

bool k3_json_bool(const k3_json_document *document,
                  int32_t                 token,
                  bool                   *value);

/*
 * Allocate a JSON string literal, including surrounding double quotes.
 * Invalid UTF-8 bytes are replaced by U+FFFD.
 */
bool k3_json_escape(const char *text,
                    size_t      text_size,
                    char      **escaped,
                    size_t     *escaped_size,
                    char       *error,
                    size_t      error_size);

#ifdef __cplusplus
}
#endif

#endif
