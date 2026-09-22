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

/* One chat turn. Content is UTF-8 and is never trusted as markup. */
typedef struct {
    const char *role;      /* "system", "user" or "assistant" */
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
                                  bool add_generation_prompt,
                                  bool enable_thinking,
                                  mimo26_token_buffer *out, char *error,
                                  size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
