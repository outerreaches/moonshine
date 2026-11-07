# Agentic API and tool use

Moonshine implements a bounded OpenAI-shaped Chat Completions surface at
`POST /v1/chat/completions`, with native Kimi XTML prompt and response
structure. This page describes the K3.1 Pro 0.3.0 research-preview interface.

> Synthetic request/response tests, live tool calling/continuation, structured
> JSON, and the official Python SDK fixture pass. Historical K3 agentic results
> are not K3.1 quality or production qualification.

## Identity and input boundary

The only accepted request/discovery model ID is:

```text
moonshotai/Kimi-K3.1-Pro
```

`moonshine` is the project name, not a model alias. The server version is
`0.3.0-k3.1-research-preview`.

Messages are text-only. `image_url`, audio, and other non-text content parts are
rejected explicitly; they are not ignored or converted to placeholders. The
public endpoint supports one completion choice and one inference slot.

## Reasoning contract

Thinking is enabled on the API. `reasoning_effort` accepts:

- `low`;
- `high`;
- `max` (the default when the field is omitted or null).

Public `medium` requests are rejected with
`reasoning_effort must be low, high, or max`. The internal official-XTML
renderer still supports `medium` for format compatibility, but that internal
value is not part of the public K3.1 contract.

Reasoning is returned separately as JSON `reasoning_content` and streamed as
`delta.reasoning_content` before response content. The completion-token budget
covers reasoning, response/tool content, and structural tokens together.

## Minimal request

```sh
curl --max-time 0 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "moonshotai/Kimi-K3.1-Pro",
    "messages": [{"role": "user", "content": "Say hello."}],
    "reasoning_effort": "low",
    "max_completion_tokens": 64,
    "stream": false
  }'
```

`max_completion_tokens` is preferred; legacy `max_tokens` is also accepted.
Explicit null is treated as unspecified. The request default is 256 tokens or
the smaller effective server limit. This is a cap, not a requested exact output
length. Real K3.1 evidence includes the 8K canary, a 128-token sustained decode,
and 17-token decodes after natural-context prompts through 63,971 tokens. It
does not establish continuous generation at the configured output ceiling.

## Function tools

The implemented request surface supports:

- `tools` arrays containing function definitions with `name`, optional
  `description`, and a JSON Schema `parameters` object;
- recursively key-sorted, compact tool declarations;
- `tool_choice: "auto"`, `"required"`, `"none"`, or a forced named function;
- one or multiple assistant calls with typed arguments;
- complete assistant history and consecutive `role: "tool"` results matched by
  opaque `tool_call_id`;
- JSON responses and complete indexed SSE `delta.tool_calls` items;
- `finish_reason: "tool_calls"` when calls are returned.

Tool arguments are model output. Validate them before executing a function even
when a declaration uses `strict: true`.

### First request

```json
{
  "model": "moonshotai/Kimi-K3.1-Pro",
  "messages": [
    {"role": "user", "content": "What is the weather in Toronto?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ],
  "tool_choice": "required",
  "reasoning_effort": "high"
}
```

An assistant tool-call result has the ordinary OpenAI shape, with Moonshine's
separate reasoning field:

```json
{
  "role": "assistant",
  "reasoning_content": "the exact returned reasoning",
  "content": null,
  "tool_calls": [
    {
      "id": "call_chatcmpl-moonshine-..._1",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\":\"Toronto\"}"
      }
    }
  ]
}
```

After executing the function, submit the complete original history, the exact
assistant object, and the matching tool result:

```json
{
  "model": "moonshotai/Kimi-K3.1-Pro",
  "messages": [
    {"role": "user", "content": "What is the weather in Toronto?"},
    {
      "role": "assistant",
      "reasoning_content": "the exact returned reasoning",
      "content": null,
      "tool_calls": [
        {
          "id": "call_chatcmpl-moonshine-..._1",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"Toronto\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_chatcmpl-moonshine-..._1",
      "content": "{\"weather\":\"sunny\",\"temperature_c\":22}"
    }
  ],
  "tools": ["the same complete function definition"],
  "tool_choice": "auto"
}
```

Do not drop `reasoning_content`, replace tool-call IDs, or abbreviate arguments
on continuation.

### Parallel-call policy

`parallel_tool_calls` defaults to true. Explicit false installs a hidden
at-most-one-call XTML directive and applies a post-parse count check. This is a
policy boundary, not a generation grammar: a multi-call result is rejected
before tool-call SSE items are emitted.

Hidden tool-choice and serial-call directives are causal input but are not part
of returned OpenAI history. The session records their values and message
boundaries so it can reconstruct the actual historical token stream for exact
prefix comparison.

## Structured output

`response_format` supports `json_object` and a bounded `json_schema` subset.
Structured output is post-generation validated rather than grammar-constrained.
Reasoning remains token-live in SSE, while response content is buffered until
the complete value validates.

For `json_object`, the complete response must have a top-level object root. A
tool-call intermediate turn has no final response body and is exempt.

For `json_schema`, use the standard wrapper:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "greeting_record",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "greeting": {"type": "string"},
        "count": {"type": "integer"}
      },
      "required": ["greeting", "count"],
      "additionalProperties": false
    }
  }
}
```

The supported schema subset is deliberately bounded:

- types: object, array, string, number, integer, boolean, and null;
- object keywords: `properties`, `required`, and boolean
  `additionalProperties`;
- arrays require `items`;
- string `title` and `description` metadata are accepted;
- other schema keywords and structures are rejected before inference;
- schema and instance recursion are capped at 64 levels.

The wrapper name must contain 1–64 alphanumeric, underscore, or hyphen
characters. Optional string `description` and boolean `strict` fields are
accepted. Moonshine validates the declared subset whether or not `strict` is
present.

## Streaming behavior

Ordinary response text and reasoning are token-live with UTF-8 boundary
buffering. Tool calls are emitted only after complete XTML parsing and policy
validation, as complete indexed `delta.tool_calls` items. Structured response
content is withheld until validation succeeds.

SSE comments provide prefill and quiet-decode keepalives. They are transport
keepalives, not generated content. JSON responses cannot provide equivalent
progress. A disconnect or termination signal cancels only at a complete
token/layer boundary, resets semantic state, and retains immutable expert-cache
entries.

## Exact causal-prefix reuse

The server retains one causal prefix. A new request reuses it only when its
complete rendered history is an exact append-only extension. Hidden tool,
serial-call, and structured-output directives are restored at their original
message boundaries before comparison.

Edited, forked, shorter, or otherwise mismatched histories take canonical full
prefill. Removing prior `reasoning_content` also breaks exact reuse. Standard
`usage.prompt_tokens_details.cached_tokens` reports the reused prefix; no custom
session header is required.

Unrelated auxiliary completions can occupy the only inference slot and replace
the retained prefix. Route title generation or approval-review requests to
another provider when exact continuation matters.

Optional durable checkpoints use state format v2 and retain the same exact-token
admission rule. They are not semantic or approximate caches. Real-weight K3.1
checkpoint displacement/restart, exact continuation, and corruption/truncation
rejection gates passed; a restart continuation reused 161 tokens.

## Errors and operational limits

- An unavailable model alias receives HTTP 400 and names the required K3.1 ID.
- Public `medium` or unknown reasoning values receive HTTP 400.
- Non-text content is rejected rather than partially accepted.
- Unsupported schema vocabulary is rejected before inference.
- A competing completion receives HTTP 503/`Retry-After` while the one slot is
  occupied; health and model discovery remain independently available.
- The API does not report a separate reasoning-token usage breakdown.

## Verification boundary

The synthetic OpenAI parser/codec tests cover the K3.1 model ID, version,
reasoning default and values, public `medium` rejection, text-only rejection,
tools, structured requests, and response shapes.

The optional official SDK fixture can be run after installing its isolated
dependency:

```sh
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r tests/requirements-sdk.txt
make test-openai-sdk PYTHON=.venv-sdk/bin/python
```

The fixture passed with `openai 2.52.0`. This is an isolated protocol replay,
not a live SDK agent session. In the separate real 8K Beelink canary, generation,
exact prefix reuse, tool calling/tool-result continuation, structured JSON, and
public medium/image/old-model HTTP 400 rejection passed. Synthetic, live-canary,
and SDK-fixture results are distinct evidence classes.

No Pi/Hermes or other third-party agent-client qualification and no production
soak is claimed. Continuous long-output behavior and contexts above 64K remain
open. The dated K3 qualification documents are historical records only.
