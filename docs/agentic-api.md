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
- `finish_reason: "tool_calls"` whenever calls are returned.

`parallel_tool_calls` defaults to true. Explicit false is rejected until the
engine can enforce a one-call generation constraint rather than silently
accepting a request it may violate. Structured `response_format`, thinking
and `reasoning_content` output, image input, sampling, and concurrent request
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

Moonshine returns an assistant message resembling:

```json
{
  "role": "assistant",
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
function even when its declaration uses `strict: true`.

## 2026-08-01 live qualification

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

## Agentic prefix limitation and next optimization

Both qualified prompts evaluated in full (`reused=0`). The first request's
hidden `tool_choice=required` message is causal input to the tool call but is
not part of the assistant message returned to the client. The canonical second
request therefore cannot exactly extend Moonshine's retained token history.

The current behavior is safe: exact-prefix comparison fails and the server
resets semantic state before replay. The next optimization should retain a
rewind/checkpoint boundary before the transient tool-choice directive, match
the next canonical history up to that boundary, and replay only the divergent
directive/assistant/tool-result suffix. It must preserve the existing exact
state-equivalence gate and fall back to full prefill on any mismatch.
