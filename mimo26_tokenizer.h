#ifndef MIMO26_TOKENIZER_H
#define MIMO26_TOKENIZER_H

/*
 * MiMo's tokenizer: byte-level BPE from the checkpoint's tokenizer.json.
 *
 * Separate from k3_tokenizer rather than shared, because the asset formats
 * genuinely differ -- K3 loads a tiktoken.model of base64 pieces and ranks,
 * MiMo ships a HuggingFace tokenizer.json with a vocabulary map and an
 * ordered merge list. What IS shared is the mechanism, and it is taken from
 * K3 rather than reinvented: ICU's uregex for the pre-tokenizer split, and
 * the same byte-level alphabet the GPT-2 family uses.
 *
 * Correctness is held against tests/fixtures/mimo26_tokenizer_v1.json, which
 * pins 14 text encodings and 10 chat renderings produced by the reference
 * `tokenizers` library at a known checkpoint revision. Every id in this
 * implementation has to match those exactly -- a tokenizer that is close is
 * a tokenizer that silently shifts every downstream token.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mimo26_tokenizer mimo26_tokenizer;

/* Ids the chat template and the sampler both need. */
#define MIMO26_TOK_IM_START 151644u
#define MIMO26_TOK_IM_END   151645u
#define MIMO26_TOK_ENDOFTEXT 151643u
#define MIMO26_TOK_THINK_OPEN  151667u
#define MIMO26_TOK_THINK_CLOSE 151668u
/*
 * The tool-call wrapper is a pair of special tokens, not text. A streaming
 * decoder that drops specials -- which it must, so <|im_end|> never reaches
 * a client -- therefore drops these too, leaving a bare <function=...> that
 * no parser recognizes. A caller has to re-insert them; see the server's
 * generation loop.
 */
#define MIMO26_TOK_TOOL_CALL_OPEN  151657u
#define MIMO26_TOK_TOOL_CALL_CLOSE 151658u

typedef struct {
    uint32_t *ids;
    size_t    count;
    size_t    capacity;
} mimo26_token_buffer;

void mimo26_token_buffer_free(mimo26_token_buffer *buffer);

/*
 * Load vocabulary, merges and added tokens from `root`/tokenizer.json and
 * compile the pre-tokenizer. Opens no weight shards and initializes no GPU.
 */
bool mimo26_tokenizer_create(mimo26_tokenizer **out, const char *root,
                             char *error, size_t error_size);
void mimo26_tokenizer_destroy(mimo26_tokenizer *tokenizer);

/* Tokens the tokenizer actually defines; ids beyond this are embedding
 * padding and decode to nothing. */
uint32_t mimo26_tokenizer_vocab_size(const mimo26_tokenizer *tokenizer);

/*
 * Encode one UTF-8 string. When allow_special is false, text spelling a
 * control marker such as <|im_start|> is ordinary user text -- which is what
 * untrusted request content must be encoded with, so a caller cannot inject
 * turn boundaries by typing them.
 */
bool mimo26_tokenizer_encode(mimo26_tokenizer *tokenizer, const char *text,
                             bool allow_special, mimo26_token_buffer *out,
                             char *error, size_t error_size);

/*
 * Decode ids to UTF-8. Returns a NUL-terminated buffer the caller frees.
 *
 * Byte-level BPE can split a multi-byte character across two tokens, so a
 * streaming caller must not decode token by token and emit the result. See
 * mimo26_tokenizer_decode_stream.
 */
char *mimo26_tokenizer_decode(const mimo26_tokenizer *tokenizer,
                              const uint32_t *ids, size_t count,
                              bool include_special, char *error,
                              size_t error_size);

/*
 * Streaming decode that only emits complete UTF-8 sequences.
 *
 * Holds back a trailing partial sequence until the bytes that finish it
 * arrive. Without this a server streaming token-by-token emits invalid UTF-8
 * mid-character, which some clients render as a replacement character and
 * others reject outright.
 */
typedef struct {
    unsigned char pending[8];
    size_t        pending_bytes;
} mimo26_decode_stream;

void mimo26_decode_stream_init(mimo26_decode_stream *stream);
/*
 * Append one token's text, returning how many bytes of `out` are complete
 * and safe to send. `out` must hold at least 64 bytes.
 */
bool mimo26_tokenizer_decode_stream(const mimo26_tokenizer *tokenizer,
                                    mimo26_decode_stream *stream, uint32_t id,
                                    char *out, size_t out_size,
                                    size_t *produced);

/*
 * One tool call an assistant turn made.
 *
 * `arguments_json` is a JSON object as text. The template renders each
 * member as <parameter=key>value</parameter>, writing strings raw and
 * anything else as JSON -- so the wire form is lossy about which is which,
 * and the parser recovers the distinction by trying JSON first. A string
 * that happens to spell a JSON scalar, such as a city called "3", comes
 * back as a number. That is inherent to the format, not to this
 * implementation, and is why the schema is the authority.
 */
typedef struct {
    const char *name;
    const char *arguments_json;
} mimo26_tool_call;

/* One chat turn. Content is UTF-8 and is never trusted as markup. */
typedef struct {
    const char *role;      /* "system", "user", "assistant" or "tool" */
    const char *content;
    /*
     * Assistant turns only: the reasoning the model produced for that turn,
     * rendered inside the think block ahead of the content. NULL renders an
     * empty block, which is what the reference template does for an
     * assistant turn with no recorded reasoning. Keeping these separate is
     * what lets a caller replay a conversation without the reasoning
     * leaking into the visible content -- or losing it silently.
     */
    const char *reasoning;
    /* Assistant turns only. */
    const mimo26_tool_call *tool_calls;
    size_t                  tool_call_count;
} mimo26_chat_message;

/*
 * Render and encode a conversation in MiMo's ChatML form.
 *
 * add_generation_prompt appends the assistant turn opener. enable_thinking
 * false additionally appends an empty think block, which is how the template
 * suppresses reasoning. Assistant turns already in the history are rendered
 * with that same empty block, matching the reference template.
 */
bool mimo26_tokenizer_encode_chat(mimo26_tokenizer *tokenizer,
                                  const mimo26_chat_message *messages,
                                  size_t message_count,
                                  const char *tools_json,
                                  bool add_generation_prompt,
                                  bool enable_thinking,
                                  mimo26_token_buffer *out, char *error,
                                  size_t error_size);

/*
 * Render a conversation to text without encoding it.
 *
 * Exposed because the tool round-trip tests compare rendered strings against
 * the pinned fixtures, and a difference in the template shows up there as a
 * readable diff rather than as two id sequences that disagree somewhere.
 * Caller frees.
 */
char *mimo26_tokenizer_render_chat(const mimo26_chat_message *messages,
                                   size_t message_count,
                                   const char *tools_json,
                                   bool add_generation_prompt,
                                   bool enable_thinking, char *error,
                                   size_t error_size);

/*
 * Split assistant output into visible content and the tool calls it made.
 *
 * The model emits <tool_call><function=name><parameter=k>v</parameter>...
 * </function></tool_call>. Anything outside those blocks is content.
 * Malformed blocks are an error rather than being passed through as text: a
 * half-parsed tool call silently becoming prose is how a caller ends up
 * showing a user the model's internal syntax.
 *
 * Caller frees the calls with mimo26_tool_calls_free and the content with
 * free().
 */
typedef struct {
    char *name;
    char *arguments_json;
} mimo26_parsed_tool_call;

bool mimo26_tokenizer_parse_tool_calls(const char *text,
                                       char **content_out,
                                       mimo26_parsed_tool_call **calls_out,
                                       size_t *count_out, char *error,
                                       size_t error_size);
void mimo26_tool_calls_free(mimo26_parsed_tool_call *calls, size_t count);

#ifdef __cplusplus
}
#endif

#endif
