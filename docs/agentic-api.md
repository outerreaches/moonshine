# Agentic API and tool use

Moonshine implements the function-tool loop on
`POST /v1/chat/completions`. The transport follows the OpenAI Chat
Completions shape, while the prompt and generated structure follow Kimi K3's
native XTML format. The authoritative external contracts are the
[OpenAI Chat API reference](https://developers.openai.com/api/reference/resources/chat)
and the [Kimi K3 quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart).

## Supported contract

- `tools` arrays containing function definitions with `name`, optional
  `description`, and a JSON Schema `parameters` object;
- recursively key-sorted, compact tool declarations matching K3's official
  prompt normalization;
- `tool_choice: "auto"`, `"required"`, or `"none"`;
- a forced named function object, implemented by exposing only that function
  and applying K3's native required directive;
- one or multiple assistant function calls;
- typed string, number, boolean, null, object, and array arguments;
- preservation of raw malformed JSON argument strings through K3's
  `<json type="object">` compatibility form;
- complete assistant history plus consecutive `role: "tool"` results matched
  by opaque `tool_call_id` and normalized into call order;
- ordinary JSON responses and indexed SSE `delta.tool_calls` chunks;
- `finish_reason: "tool_calls"` whenever calls are returned;
- K3 preserved thinking on every API request, with `reasoning_effort` values
  `low`, `high`, and `max` (default `max`);
- separate JSON `reasoning_content` and live SSE
  `delta.reasoning_content` before response content;
- complete assistant reasoning history rendered back into native `<think>`
  channels for multi-turn and tool continuation.

`parallel_tool_calls` defaults to true. Explicit false is rejected until the
engine can enforce a one-call generation constraint rather than silently
accepting a request it may violate. `response_format: {"type":"json_object"}`
is supported; `json_schema`, image input, sampling, and concurrent request
slots remain outside this checkpoint.

## Minimal two-request loop

First request:

```json
{
  "model": "moonshine",
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
  "tool_choice": "required"
}
```

Moonshine returns an assistant message resembling (reasoning abbreviated):

```json
{
  "role": "assistant",
  "reasoning_content": "I should call the weather tool for Toronto.",
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

Execute the function, then send the complete original history, the complete
assistant message, and the matching result:

```json
{
  "model": "moonshine",
  "messages": [
    {"role": "user", "content": "What is the weather in Toronto?"},
    {
      "role": "assistant",
      "reasoning_content": "the exact reasoning returned above",
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

Tool arguments are model output. Parse and validate them before executing a
function even when its declaration uses `strict: true`. Preserve the complete
assistant object—including `reasoning_content`—when continuing the loop.

## 2026-08-01 non-thinking baseline qualification

The accepted 8K Q8/32 engine completed the loop above without a fallback
parser or prompt convention:

| phase | prompt | completion | time | rate |
|---|---:|---:|---:|---:|
| required tool call | 161 | 45 | 213.120 s / 120.258 s | 0.755 / 0.374 tok/s |
| tool-result answer | 217 | 25 | 224.927 s / 56.759 s | 0.965 / 0.440 tok/s |

The first response called `get_weather` with `{"city":"Toronto"}` and used
`finish_reason: "tool_calls"`. After receiving a synthetic sunny/22 °C result,
K3 answered that Toronto was sunny at 22 °C and stopped naturally.

The Samsung 990 PRO moved from 36 °C composite / 40 °C sensor 2 to 55/58 °C.
It recorded no thermal-management transition, warning-temperature time, media
error, or swap growth.

## Causal-prefix recovery

K3's hidden `tool_choice=required` or `none` message is causal input to an
assistant turn, but is not part of the assistant message returned to the
client. A canonical second request therefore cannot directly extend the
retained token stream from the first request.

For a continuing `X-Moonshine-Session`, Moonshine now records only the prior
directive's message boundary and value. On the next request it renders a
candidate history with that hidden directive restored immediately before the
assistant message it caused. This recreates the actual causal history while
the client continues to send an ordinary, complete OpenAI message list.

The candidate is accepted only when every retained token is byte-for-byte
identical and the new prompt has a non-empty suffix. A changed tool schema,
edited/forked/shorter history, changed session identifier, missing header,
cold-cache benchmark request, invalid marker, or allocation failure takes the
existing semantic reset and canonical full-prefill path. No state file or
approximate matching is involved. Forced named-tool requests also remain
safe: widening the exposed declaration on the next turn changes the prefix
and triggers full replay.

### Live SSE qualification

The same two-request weather loop was repeated through real SSE with
`X-Moonshine-Session: agentic-prefix-qualification`:

| phase | actual prompt | evaluated | reused | prompt time | generation | decode |
|---|---:|---:|---:|---:|---:|---:|
| required tool call | 151 | 151 | 0 | 212.359 s | 45 | 119.409 s |
| tool-result answer | 238 | 42 | 196 | 104.913 s | 25 | 55.863 s |

The 238-token usage count is the actual causal prompt, including the restored
historical directive. The second request reused the complete 151-token prompt
and 45 generated tokens from the first turn. It emitted the indexed
`delta.tool_calls` item, accepted the matching tool result, answered that
Toronto was sunny at 22 °C, and stopped naturally.

Compared with the earlier qualified full-replay result turn, prefill time fell
from 224.927 to 104.913 seconds (53.4%), while complete result-turn latency
fell from 281.686 to 160.776 seconds (42.9%, about 1.75x faster). The 42-token
suffix remains below the measured 93-token layer-major crossover and therefore
uses exact token-major execution.

## Preserved-thinking qualification

K3's current model contract always enables thinking. Moonshine now defaults
the API to `reasoning_effort: "max"`; callers can select `low`, `high`, or
`max`. The generated-token limit covers reasoning, response, tools, and
structural tokens together.

An 8K SSE smoke test used `reasoning_effort: "low"` and `Say hello.`. The
92-token prompt completed in 230.491 seconds. K3 streamed “The user just wants
a greeting. Keep it brief and friendly.” only through
`delta.reasoning_content`, then streamed “Hello! How can I help you today?”
only through `delta.content`. It stopped naturally after 35 generated tokens
in 75.353 seconds.

The exact assistant reasoning and response were then returned with `Now say
goodbye.` under the same session:

| prompt total | evaluated | reused | prefill | generated | decode | finish |
|---:|---:|---:|---:|---:|---:|---|
| 152 | 25 | 127 | 59.473 s | 50 | 116.434 s | `stop` |

The second turn streamed reasoning first, then “Goodbye! Have a great day!
👋”. Reusing all 127 prior prompt/generated tokens proves preserved reasoning
history reconnects to the exact retained causal state, rather than merely
round-tripping as unused JSON metadata.

## JSON-object response format

Moonshine implements K3's native hidden `response_format=json_object`
directive for `response_format: {"type":"json_object"}`. After generation it
parses the complete response and requires a top-level JSON object. A tool-call
intermediate turn is exempt because it has no final response body.

Structured SSE keeps reasoning token-live but buffers response content until
validation succeeds. Invalid or non-object output produces an inference error
without first streaming unvalidated content. A valid object is released as a
single content chunk before the terminal event.

The 8K low-effort live check requested a `greeting` key with value `hello`:

| prompt | prompt time/rate | generation | decode time/rate | finish |
|---:|---:|---:|---:|---|
| 149 | 212.342 s / 0.702 t/s | 52 | 123.780 s / 0.420 t/s | `stop` |

K3 produced `{"greeting":"hello"}`. Moonshine exposed reasoning during
decode, withheld the response, validated its JSON/object root, then emitted
the exact object in one SSE `delta.content` chunk.

`response_format=json_schema` remains rejected until Moonshine has a bounded,
declared JSON Schema validation surface. The next gates are that schema subset,
an enforceable `parallel_tool_calls: false` constraint, and an SDK-level SSE
fixture.
