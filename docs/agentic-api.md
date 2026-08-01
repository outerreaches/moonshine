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
  `low`, `medium`, `high`, and `max` (default `max`);
- separate JSON `reasoning_content` and live SSE
  `delta.reasoning_content` before response content;
- complete assistant reasoning history rendered back into native `<think>`
  channels for multi-turn and tool continuation.

`parallel_tool_calls` defaults to true. Explicit false is supported through a
Moonshine hidden at-most-one-call directive plus a hard post-parse policy
check. The prompt steers K3 to serialize work across turns; if it nevertheless
emits more than one call, Moonshine returns an inference error before any
tool-call SSE item. This is an enforceable API boundary, not a generation
grammar. `response_format` supports `json_object` and the bounded
`json_schema` subset documented below. Image input, sampling, and concurrent
request slots remain outside this checkpoint.

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

Moonshine records only the prior directive's message boundary and value. On
the next request it renders a candidate history with that hidden directive
restored immediately before the assistant message it caused. This recreates
the actual causal history while the client sends an ordinary, complete OpenAI
message list with no custom session header.

The candidate is accepted only when every retained token is byte-for-byte
identical and the new prompt has a non-empty suffix. A changed tool schema,
edited/forked/shorter history, cold-cache benchmark request, invalid marker,
or allocation failure takes the existing semantic reset and canonical
full-prefill path. No state file or approximate matching is involved. Forced
named-tool requests also remain safe: widening the exposed declaration on the
next turn changes the prefix and triggers full replay.

### Live SSE qualification

The same two-request weather loop was repeated through real SSE:

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
fell from 281.686 to 160.776 seconds (42.9%, about 1.75x faster). This run
predated selected-prefill crossover promotion and used token-major execution.
The exact matched fixture now measures a 42-token selected range at 43.742
seconds versus 92.208 seconds sequentially, so the default schedules this
suffix layer-major. A fresh live API requalification is recorded separately.

## Preserved-thinking qualification

K3's current model contract always enables thinking. Moonshine now defaults
the API to `reasoning_effort: "max"`; callers can select `low`, `medium`,
`high`, or `max`. The generated-token limit covers reasoning, response, tools, and
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

## Structured response formats

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

Moonshine also implements K3's native `response_format=json_schema` directive
for the standard Chat Completions wrapper:

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

This is a declared subset, not a complete JSON Schema implementation:

- every node is an object with one string `type`;
- supported types are object, array, string, number, integer, boolean, and
  null;
- object nodes may use `properties`, `required`, and boolean
  `additionalProperties`;
- array nodes require `items`;
- `title` and `description` string metadata are accepted;
- all other schema keywords and structures are rejected before inference;
- schema and instance recursion is capped at 64 levels.

The wrapper requires a 1-64 character alphanumeric/underscore/hyphen `name`;
optional `description` and boolean `strict` fields are accepted. Moonshine
validates the declared subset after generation whether or not `strict` is
present. Tool-call intermediate turns remain exempt because they have no final
response value.

The accepted 8K low-effort SSE qualification required `greeting` as a string,
`count` as an integer, both fields present, and no additional properties:

| prompt | prompt time/rate | generation | decode time/rate | finish |
|---:|---:|---:|---:|---:|
| 205 | 215.865 s / 0.950 t/s | 57 | 137.230 s / 0.415 t/s | `stop` |

K3 produced `{"greeting":"hello","count":1}`. Reasoning streamed live;
the response body appeared only once, after the recursive validator accepted
it.

### Structured-session causal reuse

Response-format directives are causal input immediately before generation but
are absent from the returned assistant message, just like tool-choice
directives. Moonshine now stores their historical message boundaries and
formats, plus an owned copy of canonical JSON Schema text. It restores each
directive before the assistant turn it caused and still accepts reuse only
after exact retained-token comparison. Malformed, stale, allocation-failed,
or mismatched marker state takes canonical full prefill.

A two-turn 8K low-effort JSON Schema check returned `greeting=hello`, then
continued the exact returned reasoning/content and requested
`greeting=goodbye` under the same schema and session:

| phase | prompt total | evaluated | reused | prefill | generated | decode |
|---|---:|---:|---:|---:|---:|---:|
| first schema turn | 184 | 184 | 0 | 215.694 s | 53 | 128.217 s |
| schema continuation | 354 | 117 | 237 | 210.464 s | 35 | 89.079 s |

The continuation returned validated `{"greeting":"goodbye"}` and retained
the complete 184-token prompt plus 53-token completion. This proved exact
causal reconstruction and localized the remaining latency to the layer-major
routed sweep.

Moonshine then measured the actual router union before changing that schedule.
The 117-token suffix selected only 230–640 of 896 experts per layer, averaging
422.9, while the old path still read every expert: 1,347.431 GiB and 172.447
seconds of read wait. The accepted selected-only path preserves the same
physical ordering and expert arithmetic but omits layouts with zero routes.

The paired live result is:

| phase | old prefill | selected prefill | old/new reads | result |
|---|---:|---:|---:|---|
| first schema turn, 184 tokens | 214.702 s | 129.058 s | 1,347.431 / 777.676 GiB | `greeting=hello` |
| continued suffix, 117 tokens | 211.473 s | 104.469 s | 1,347.431 / 636.038 GiB | `greeting=goodbye` |

The target continuation is 50.6% faster in prefill and 35.6% faster end to
end (300.566 to 193.553 seconds, 1.55x), while still reusing all 237 retained
tokens. The exact two-token range also falls from 203.638 to 7.961 seconds
with every causal hash unchanged, and the locked 512-token fixture falls from
239.325 to 123.519 seconds with the same next token and selected value.

## Non-parallel tool calls

For `parallel_tool_calls: false`, Moonshine places a hidden
`parallel-tool-calls` system message after K3's native tool-choice directive.
The request policy then rejects a parsed result containing more than one call.
Tool calls are emitted only after complete XTML parsing and policy validation,
so a violating call set cannot leak through an earlier indexed SSE chunk.

The hidden directive is absent from returned OpenAI history. Session-prefix
reuse therefore stores its message boundary alongside the historical
tool-choice directive, reconstructs both in their original order, and still
requires an exact token-prefix match. A mismatch takes the ordinary isolated
full-prefill fallback.

The accepted 8K low-effort SSE loop deliberately requested two independent
operations while exposing `get_weather` and `get_time`:

| phase | prompt total | evaluated | reused | prefill | generated | decode | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| first serial call | 326 | 326 | 0 | 225.175 s | 170 | 439.579 s | `tool_calls` |
| second serial call | 630 | 134 | 496 | 211.549 s | 60 | 165.431 s | `tool_calls` |
| final answer | 826 | 136 | 690 | 211.591 s | 40 | 92.510 s | `stop` |

K3 first emitted only `get_weather({"city":"Toronto"})`, then only
`get_time({"city":"Paris"})`, and finally answered with sunny/22 °C and
14:00 after `tool_choice: "none"`. Its first reasoning explicitly planned
sequential calls across assistant turns. All 496 and then 690 prior causal
tokens were reused, proving exact reconstruction of the added directive.

An adversarial first attempt capped at 128 completion tokens exhausted its
budget during this serial-planning reasoning. The strengthened required-tool
check returned an inference error instead of accepting a length-stopped turn
without a call. The normal 256-token default completed the call. This is also
a latency finding: conflicting multi-operation prompts can spend substantial
reasoning tokens deciding call order even at low effort.

## Official Python SDK qualification

The optional `test-openai-sdk` target pins the official Python client at
2.52.0 in an isolated environment. Its local HTTP fixture replays Moonshine's
exact SSE shape and proves that the client:

- ignores Moonshine comment keepalives;
- preserves the extension `delta.reasoning_content` field;
- parses the complete indexed function call, opaque ID, name, and arguments;
- preserves `finish_reason=tool_calls` and terminal usage;
- transmits `parallel_tool_calls=false` and the session header.

The original real-model gate connected the same SDK directly to an 8K
Moonshine server. It exposed non-empty reasoning, parsed
`get_weather({"city":"Toronto"})`, preserved its generated call ID and
terminal usage, and completed with `tool_calls`:

| prompt | prompt time/rate | generation | decode time/rate | usage |
|---:|---:|---:|---:|---:|
| 216 | 216.773 s / 0.996 t/s | 72 | 185.008 s / 0.389 t/s | 216/72/288 |

After selected-expert crossover promotion, the live fixture was extended to
return the complete assistant reasoning/tool-call object, submit a synthetic
sunny/22 °C tool result under the same session, and validate the final answer:

| phase | prompt total | evaluated | reused | prefill | generated | decode | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| required weather call | 216 | 216 | 0 | 139.855 s | 72 | 185.240 s | `tool_calls` |
| weather result answer | 330 | 42 | 288 | 54.107 s | 37 | 85.632 s | `stop` |

The first prefill is 35.5% faster than its exact earlier full-store SDK run.
The second request reused the complete first prompt and completion, scheduled
its 42-token suffix through selected layer-major prefill, exposed reasoning,
made no extra call, and answered with Toronto, sunny, and 22 °C. Its route
union averaged 220.9 experts per layer (118--355), reading 332.167 GiB. The
matched synthetic crossover remains the schedule-comparison authority because
the earlier live 42-token qualification used a different conversation.

The Samsung 990 PRO peaked at 60 °C composite / 69 °C sensor 2 during the
loop, then cooled to 52/57 °C. SMART retained zero warning/critical thermal
time, media/data-integrity errors, and error-log entries. No active swap I/O
was observed, although allocated swap increased by 112,496,640 bytes from the
preceding selected-prefill checkpoint; this run therefore does not claim zero
swap growth.

The SDK is a qualification-only dependency; Moonshine's engine, server, and
native client still require no Python runtime.

### Diagnostic KDA backend replay

The same live fixture was repeated with the opt-in range projection backend
selected by `moonshine-server --range-backend kda-blas`. The default server
path is unchanged. The diagnostic path uses the same per-layer progress
callback, so long prefill retains SSE keepalives before the first model token.

| phase | prompt total | evaluated | reused | prefill | generated | decode | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| required weather call | 216 | 216 | 0 | 137.073 s | 72 | 184.806 s | `tool_calls` |
| weather result answer | 330 | 42 | 288 | 55.925 s | 37 | 85.663 s | `stop` |

OpenAI Python SDK 2.52.0 accepted both SSE responses. The first turn exposed
non-empty reasoning and exactly one `get_weather({"city":"Toronto"})` call;
the continuation reused all 288 causal tokens, accepted the synthetic
sunny/22 C result, exposed reasoning, made no extra call, and answered with
Toronto, sunny, and 22 C. Terminal usage was `216/72/288` then `330/37/367`.

This is a structure and task-quality gate, not a cross-run performance claim.
The backend changes the floating-point reduction schedule and remains an
explicit diagnostic pending broader quality evaluation and review.
