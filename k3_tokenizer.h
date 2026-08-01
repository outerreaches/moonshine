#ifndef K3_TOKENIZER_H
#define K3_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    K3_TOKEN_BOS = 163584,
    K3_TOKEN_EOS = 163585,
    K3_TOKEN_END_OF_MSG = 163586,
    K3_TOKEN_OPEN = 163587,
    K3_TOKEN_CLOSE = 163588,
    K3_TOKEN_SEP = 163589,
    K3_TOKEN_UNK = 163838,
    K3_TOKEN_PAD = 163839,
    K3_TOKEN_VOCAB_SIZE = 163840,
};

typedef struct k3_tokenizer k3_tokenizer;

typedef struct {
    uint32_t *data;
    size_t    count;
    size_t    capacity;
} k3_token_buffer;

typedef struct {
    char   *data;
    size_t  size;
    size_t  capacity;
} k3_text_buffer;

typedef enum {
    K3_CHAT_ROLE_SYSTEM = 0,
    K3_CHAT_ROLE_USER = 1,
    K3_CHAT_ROLE_ASSISTANT = 2,
    K3_CHAT_ROLE_TOOL = 3,
} k3_chat_role;

typedef struct {
    /* OpenAI call IDs are transport metadata and are not rendered into XTML. */
    const char *id;
    const char *name;
    /* A compact JSON object string. */
    const char *arguments;
} k3_tool_call;

typedef struct {
    k3_chat_role role;
    const char  *content;
    const char  *name;
    const char  *reasoning_content;
    const char  *tool_call_id;
    k3_tool_call *tool_calls;
    size_t        tool_call_count;
    /* Kimi extension: dynamic tool declaration carried by a system message. */
    const char  *tools_json;
} k3_chat_message;

typedef enum {
    K3_TOOL_CHOICE_AUTO = 0,
    K3_TOOL_CHOICE_REQUIRED = 1,
    K3_TOOL_CHOICE_NONE = 2,
} k3_tool_choice;

typedef struct {
    bool        add_generation_prompt;
    bool        thinking;
    /*
     * NULL omits the internal thinking-effort message. "low", "high", and
     * "max" are accepted. The official tokenizer wrapper defaults to "max".
     */
    const char *thinking_effort;
    /* Compact, recursively key-sorted JSON array matching OpenAI tools. */
    const char *tools_json;
    k3_tool_choice tool_choice;
} k3_chat_options;

typedef struct {
    k3_text_buffer response;
    k3_tool_call  *tool_calls;
    size_t         tool_call_count;
} k3_assistant_output;

/*
 * Load the official tiktoken.model from model_root and compile K3's Unicode
 * pre-tokenizer. This is a bounded model-asset operation and does not open
 * weight shards or initialize ROCm.
 */
bool k3_tokenizer_create(k3_tokenizer **out,
                         const char    *model_root,
                         char          *error,
                         size_t         error_size);

void k3_tokenizer_destroy(k3_tokenizer *tokenizer);

void k3_token_buffer_free(k3_token_buffer *buffer);
void k3_text_buffer_free(k3_text_buffer *buffer);

/*
 * Encode one UTF-8 string. When allow_special is false, strings resembling
 * K3 control markers are ordinary user text. The output buffer is replaced.
 */
bool k3_tokenizer_encode(k3_tokenizer *tokenizer,
                         const char   *text,
                         bool          allow_special,
                         k3_token_buffer *output,
                         char          *error,
                         size_t         error_size);

/*
 * Decode complete tokens to UTF-8. include_special controls whether special
 * token spellings such as <|open|> are included. The output is NUL-terminated;
 * size excludes that terminator.
 */
bool k3_tokenizer_decode(const k3_tokenizer *tokenizer,
                         const uint32_t     *tokens,
                         size_t              token_count,
                         bool                include_special,
                         k3_text_buffer      *output,
                         char                *error,
                         size_t               error_size);

/*
 * Render and tokenize the official text-only K3 XTML message form. Structural
 * markers are emitted as control tokens, while message content and attribute
 * values are always encoded as ordinary text to prevent marker injection.
 */
bool k3_tokenizer_encode_chat(k3_tokenizer        *tokenizer,
                              const k3_chat_message *messages,
                              size_t                message_count,
                              const k3_chat_options *options,
                              k3_token_buffer      *output,
                              char                 *error,
                              size_t                error_size);

/*
 * Parse the model-produced suffix after the generation prompt's open
 * <response> tag. Tool arguments are returned as OpenAI JSON object strings.
 */
bool k3_tokenizer_parse_assistant_output(
    const k3_tokenizer       *tokenizer,
    const uint32_t           *tokens,
    size_t                    token_count,
    k3_assistant_output      *output,
    char                     *error,
    size_t                    error_size);

void k3_assistant_output_free(k3_assistant_output *output);

#ifdef __cplusplus
}
#endif

#endif
