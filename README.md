# Moonshine

Moonshine is an experimental inference engine for running the official Kimi
K3 SafeTensors checkpoint on a single 128 GB AMD Strix Halo system. It keeps
the non-routed model tier resident in ROCm memory, streams only selected
MXFP4 routed experts from NVMe, and exposes stateful one-token decode plus
layer-major prefill.

The name is a nod to Kimi's creator, Moonshot AI. Moonshine is an independent
community project and is not affiliated with or endorsed by Moonshot AI.

This is a research preview, not a general model runner. The current checkpoint
has a C API, exact causal-state persistence, executable
correctness/performance fixtures, a native chat client, and a one-slot
OpenAI-compatible HTTP service with native function-tool calls. It is
deliberately narrow:

- Linux and ROCm only;
- tested on `gfx1151` with ROCm 7.2;
- the pinned official 96-shard Kimi K3 SafeTensors layout only;
- dynamically allocated context, capacity-qualified through 128K;
- greedy next-token inference;
- one large model process at a time.

The accepted Q8/32 configuration uses about 55.27 GiB for resident static
weights and 48.11 GiB for the online routed-expert cache. Runtime state is
0.920 GiB at 8K, 1.566 GiB at 32K, or 4.150 GiB at 128K, plus 0.262 GiB of
mapped staging. The 128K configuration accounts for about 107.793 GiB before
allocator/driver overhead. Fully resident BF16 is intentionally rejected on
the tested 128 GB machine.

## How the engine works

K3 has 93 transformer layers: one dense layer followed by 92 routed-MoE
layers, with 69 KDA and 24 gated MLA attention layers. Each routed layer
selects 16 of 896 experts.

The engine divides the checkpoint into two physical tiers:

```text
official 96-shard SafeTensors checkpoint
             |
             +-- static tier -> Q8/BF16 ROCm residency (~55.27 GiB)
             |     attention, routers, shared experts, norms, output head
             |
             `-- routed MXFP4 experts -> NVMe
                    |
                    +-- O_DIRECT + raw io_uring, QD2
                    +-- HIP-mapped fixed staging buffers
                    `-- 32-slot/layer online LRU cache (~48.11 GiB)
```

Decode is token-major. The engine walks all 93 layers, retains KDA recurrent
state, convolution state, compressed MLA cache, AttnRes state, and routed
expert-cache contents between tokens.

Prefill is layer-major. A token range visits the routed store once per layer
and groups all rows selecting each expert. After routing, it reads only the
unique selected experts in physical shard/offset order. The payload-free plan
retains a roughly 1.447 TB full-store ceiling; actual traffic follows the
per-layer route union. Cold prefill can borrow an explicitly accounted portion
of the empty decode cache for batch workspace.

See [Architecture](docs/architecture.md) for the full graph, memory, I/O, and
correctness model.

## Current measured checkpoint

These are single-run results on an AMD Ryzen AI Max+ 395 / Radeon 8060S
(`gfx1151`), Ubuntu 24.04, Linux 7.0, ROCm 7.2, and a Samsung 990 PRO. The
startup, short-prompt, and decode rows were rerun from a detached clean
checkout of `c041205`. Prefill and agentic rows are accepted fixtures from the
later commits linked in the qualification documents. They are engineering
fixtures, not cross-project benchmark claims.

**Prefill rows are not all measured against the same routed-I/O regime**, so
read the row labels before comparing them. See
[Reading the prefill rows](#reading-the-prefill-rows) below.

| workload | result |
|---|---:|
| Startup, Q8 static + 32 experts/layer | 38.1 s |
| Short 24-token non-thinking sequential prompt | 0.438 tok/s |
| Post-TTFT greedy decode | 0.509 tok/s |
| Selected layer-major prefill, 512 positions | 4.145 tok/s |
| Selected layer-major prefill, 8,192 positions | 8.126 tok/s |
| Historical full-store diagnostic KDA hipBLAS, 8,192 positions | 10.448 tok/s |
| Diagnostic KDA natural retrieval, 472 positions | 2.739 tok/s |
| Causal-state export after 2 positions | 1.043 s |
| Causal-state import after 2 positions | 0.841 s |
| Native chat `Say hello.` prompt | 0.439 tok/s |
| Native chat `Say hello.` completion | 0.500 tok/s |
| OpenAI HTTP/SSE chat at 32K, prompt | 0.437–0.438 tok/s |
| OpenAI HTTP/SSE chat at 32K, completion | 0.499 tok/s |
| Locked chat at 128K capacity, prompt | 0.430 tok/s |
| Locked chat at 128K capacity, completion | 0.508 tok/s |
| OpenAI JSON chat at 128K capacity, prompt | 0.438 tok/s |
| OpenAI JSON chat at 128K capacity, completion | 0.498 tok/s |
| Agentic first turn, 161-token tool prompt | 213.120 s / 0.755 tok/s |
| Agentic first turn, 45-token tool call | 120.258 s / 0.374 tok/s |
| Agentic result turn, 217-token prompt | 224.927 s / 0.965 tok/s |
| Agentic result turn, 25-token answer | 56.759 s / 0.440 tok/s |
| Agentic reused result turn, 42 of 238 tokens evaluated | 104.913 s / 0.400 tok/s |
| Agentic reused result turn, 25-token answer | 55.863 s / 0.448 tok/s |
| Low-effort thinking hello, 92-token prompt | 230.491 s / 0.399 tok/s |
| Low-effort thinking hello, 35-token completion | 75.353 s / 0.464 tok/s |
| Preserved-thinking continuation, 25 of 152 evaluated | 59.473 s / 0.420 tok/s |
| Structured JSON object, 149-token prompt | 212.342 s / 0.702 tok/s |
| Structured JSON object, 52-token completion | 123.780 s / 0.420 tok/s |
| Structured JSON Schema, 205-token prompt | 215.865 s / 0.950 tok/s |
| Structured JSON Schema, 57-token completion | 137.230 s / 0.415 tok/s |
| Full-store structured continuation, 117 of 354 evaluated | 210.464 s / 0.556 tok/s |
| Selected structured first turn, 184 tokens | 129.058 s / 1.426 tok/s |
| Selected structured continuation, 117 of 354 evaluated | 104.469 s / 1.120 tok/s |
| Full-store Python SDK tool prompt, 216 tokens | 216.773 s / 0.996 tok/s |
| Selected Python SDK tool prompt, 216 tokens | 139.855 s / 1.544 tok/s |
| Official Python SDK tool call, 72 tokens | 185.008 s / 0.389 tok/s |
| SDK result suffix, 42 of 330 tokens evaluated | 54.107 s / 0.776 tok/s |
| SDK result answer, 37 tokens | 85.632 s / 0.432 tok/s |

### Reading the prefill rows

Layer-major prefill visits the routed store once per layer. Two different
amounts of that store can be read, and the table contains rows from both:

- **Selected** rows read only each layer's actual route union. This is the
  production path.
- **Full-store** and **Historical full-store** rows read an aggregate
  1,446,793,422,960 bytes across one complete physical sweep of each of the 92
  routed layers. This was the production path before selected reads landed,
  and it is retained for rows that have not been re-measured.

A `Full-store` row and a `Selected` row for the same workload therefore
describe different amounts of I/O and **must not be compared as a backend
result**. The gap between them is workload-dependent rather than fixed: at
filled 8K the union is 69.8% of the store, but natural-text prompts route far
more densely — 96.9% at 16K and 98.0% at 32K — so long natural prompts see
little benefit while short agentic suffixes see a large one.

One consequence is easy to misread. `Historical full-store diagnostic KDA
hipBLAS, 8,192 positions` at 10.448 tok/s appears faster than
`Selected layer-major prefill, 8,192 positions` at 8.126 tok/s, but the two
measure different things: the first read 1.447 TB with a different projection
backend, the second read 1.010 TB with the default one. The hipBLAS backend's
effect on the **filled-8K selected path** has not been measured. The bounded
selected path has been exercised on the 472-token natural-text and live SDK
gates described below, but that does not supply a filled-8K performance row.
The optimizations are independent — one reduces routed I/O, the other reduces
KDA projection cost — so their full-length combination remains unqualified.

The default range path matches the locked sequential state oracle exactly.
Selected-only routed reads reduce the exact two-token range from 203.638 to
7.961 seconds and the locked 512-token range from 239.325 to 123.519 seconds.
The representative 117-token structured continuation drops from 211.473 to
104.469 seconds while preserving its complete 237-token causal prefix and
validated output. Filled 8K remains exact and improves from 1,054.547 to
1,008.104 seconds while avoiding 30.2% of full-store reads.
The faster KDA dequantize-plus-hipBLAS path changes selected values and causal-
state hashes because its BF16 reduction order differs. It now passes a paired,
repeatable 472-token natural-text retrieval gate and a complete official-SDK
tool loop, but remains diagnostic and opt-in pending review and broader task
quality tests.
The complete 128K capacity evidence and its limits are recorded in
[128K configured-context qualification](docs/qualification-128k.md).

## Requirements

- x86-64 Linux with a recent kernel supporting `io_uring` and `O_DIRECT`;
- an AMD ROCm device with enough shared/device-addressable memory;
- ROCm 7.2 with HIP, hipBLAS, and hipBLASLt development files;
- about 128 GB of system/unified memory for the accepted Q8/32 engine;
- about 1.6 TB of local SSD capacity for the official checkpoint;
- a local filesystem that supports aligned direct I/O (the tested setup uses
  ext4 with `noatime`);
- ICU 74 development files for the native Unicode pre-tokenizer;
- the official Kimi K3 checkpoint at revision
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.

Other AMD architectures, ROCm releases, filesystems, model revisions, memory
sizes, and contexts above 128K are not qualified yet. The 128K result covers
allocation, startup, a locked 24-token prompt/18-token completion, and the
OpenAI-compatible request path; filled-128K prefill latency and long-context
quality are not qualified. The official text config advertises 1,048,576
positions, but that is an architectural limit rather than a practical claim
for this 128 GB implementation.

## Build

Install a working ROCm toolchain, a C compiler, GNU Make, binutils, and ICU
development files. On the tested host:

```sh
/opt/rocm/bin/hipcc --version
rocminfo | grep gfx1151
make
```

`make` builds `libmoonshine.a`, `moonshine-chat`, and `moonshine-server`.
The architecture defaults to `gfx1151`; override it only for porting work:

```sh
make ROCM_ARCH=gfx1151
make tests
```

Run the model-free CPU and ROCm tests:

```sh
make test
```

Portable publication CI can run without ROCm or model weights:

```sh
make test-cpu
```

The optional SDK wire fixture uses an isolated Python environment and does not
add Python to Moonshine's build or runtime dependencies:

```sh
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r tests/requirements-sdk.txt
make test-openai-sdk PYTHON=.venv-sdk/bin/python
```

It runs the official OpenAI Python client against a local replay of
Moonshine's exact SSE wire, including comment keepalives, reasoning deltas,
indexed function calls, terminal usage, and `[DONE]`.

## Verify the native tokenizer and XTML renderer

The engine reads the official `tiktoken.model` directly and executes K3's
Unicode pre-tokenizer through ICU. No Python or Transformers runtime is used.

```sh
make test-tokenizer MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

The fixture compares exact official token IDs for English
case/contractions/numbers, Han text, combining marks, Hindi, Arabic, emoji,
and special-token handling. It also locks non-thinking, thinking, multi-turn,
and function-tool XTML. The existing `Say hello.` prompt is reproduced exactly
at all 24 token IDs. Tool fixtures cover recursively sorted declarations,
typed arguments, calls, call-ID-resolved results, and raw JSON arguments.

## Run the interactive client

Stop every other large model process first, then run:

```sh
./moonshine-chat /path/to/moonshotai__Kimi-K3
```

Enter one user message per line. The engine stays resident and retains causal
and expert-cache state across turns. Short prompts use token-major execution
through 7 tokens; prompts of 8 tokens or more use selected layer-major
prefill. A warm-cache, same-engine crossover fixture requires bit-identical
output and complete causal state before reporting either timing.

One-shot mode writes response text to stdout and telemetry to stderr:

```sh
./moonshine-chat /path/to/moonshotai__Kimi-K3 \
  --prompt "Say hello." \
  --max-tokens 32
```

The locked native run produced:

```text
Hello! 👋 How can I help you today?
[prompt=24 token-major 54.682s 0.439 tok/s;
 generated=18 35.973s 0.500 tok/s;
 finish=end_of_message; forced=0; position=42]
```

Interactive `/save PATH` and `/load PATH` commands persist or restore exact
semantic state. `--system`, `--load`, and `--save` are also available; run
`./moonshine-chat --help` for the complete interface.

Use a larger qualified context without changing the weight residency:

```sh
./moonshine-chat /path/to/moonshotai__Kimi-K3 --context 131072
```

## Run the OpenAI-compatible API

The server keeps one Q8 engine resident, defaults to 32 expert-cache slots per
layer, and processes one request at a time.
Every request still supplies and renders the complete OpenAI message history.
The server retains causal state from the most recently completed request and
automatically reuses it when the next rendered history is an exact token-prefix
extension. No custom session header is required. Edited, forked, shorter, or
otherwise mismatched histories fall back to an isolated semantic reset and
full prefill. Moonshine restores prior hidden tool-choice, serial-call, and
structured-response directives at their original message boundaries before
applying the same exact token-prefix gate. The routed expert cache remains
persistent across both paths because it contains immutable model weights, not
conversation state.

Start a loopback-only 128K-capacity service:

```sh
./moonshine-server /path/to/moonshotai__Kimi-K3 \
  --host 127.0.0.1 \
  --port 8080 \
  --context 131072 \
  --experts 30 \
  --max-output-tokens 65536
```

The initial surface provides `GET /health`, `GET /v1/models`, and
`POST /v1/chat/completions`. Both ordinary JSON and incremental SSE responses
are supported. Health and model discovery report `context_length` and the
effective `max_output_tokens` ceiling.

The server's default per-request output ceiling is 8,192 tokens. Set
`--max-output-tokens` to raise it as high as 65,536; the effective ceiling is
never larger than the configured context. Requests may use either
`max_completion_tokens` or legacy `max_tokens`, in that order. Explicit
`null` is treated as unspecified for client compatibility. If neither field
supplies a value, Moonshine uses 256 or the server ceiling when it is smaller.
Generation can stop naturally before the cap and is always clamped to the
context remaining after the prompt and required XTML trailers. The 64K source
setting is qualified for API admission, remaining-context clamping, and two
independent short requests in one persistent 128K/30-expert process. It does
not qualify a 64K uninterrupted decode workload; at the measured decode rate
such a workload would take many hours. See
[64K output and medium-reasoning qualification](docs/qualification-output-64k.md).

On the qualified 128 GB host, use `--experts 30` for a long-lived 128K server.
The original configured-capacity qualification used 32 slots and proved one
cold request, but a later request must retain the warmed cache while allocating
a separate prefill workspace. At 128K, 32 slots can leave less than the
CMA-plus-4-GiB safety reserve and the second prefill is rejected. Two
independent 128K requests passed with 30 slots, whose 45.104 GiB cache leaves
about 3.0 GiB more headroom. Smaller contexts retain the qualified 32-slot
default.

If a genuine warm prefix miss still cannot fit a separate guarded workspace,
Moonshine now lends a slot-aligned tail of the warm expert cache instead of
lowering the memory guard. A live 3,990-token divergent request at the 128K/30
profile borrowed 4.151 GiB (254 slots), retained 40.953 GiB of cache storage,
prefilled successfully at 7.500 tokens/s, and returned HTTP 200. The server log
reported `workspace_borrow=4.151GiB` and the exact prefix mismatch. This is a
resilience path, not a substitute for stable client history: full replay is
still much slower than a small exact continuation.

Before a divergent full range prefill resets an existing conversation,
Moonshine plans that position-zero request against a cache-backed workspace.
If the complete prompt cannot fit one cache loan, the server selects the
largest exact-planner-approved chunk, resets only after that chunk is admitted,
and advances the prompt through bounded range calls. Each call releases its
transient workspace before the next one. Exact-prefix and monolithic prefill
remain the fast paths; chunking preserves the memory guard but repeats layer
sweeps and can make time to first token much longer.

Example requests:

```sh
curl --max-time 0 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "moonshine",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_completion_tokens": 32,
    "stream": false
  }'

curl --max-time 0 -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "moonshine",
    "messages": [{"role": "user", "content": "Say hello."}],
    "stream": true
  }'
```

Function tools use the standard Chat Completions envelope. Moonshine accepts
`tools`, `tool_choice` values `auto`, `required`, and `none`, or a specifically
named function choice. It emits assistant `tool_calls` with
`finish_reason: "tool_calls"`; the client executes each function, appends the
complete assistant message, then appends one `role: "tool"` message with the
matching `tool_call_id` for each result. Multiple calls are supported and
tool-result messages are normalized into call order.

Kimi K3 thinking is enabled on every API request. `reasoning_effort` accepts
`low`, `medium`, `high`, or `max` and defaults to `max`. Ordinary responses
include `reasoning_content`; SSE emits live `delta.reasoning_content` chunks before
`delta.content`. Send the complete returned assistant message—including
`reasoning_content`, `content`, and any `tool_calls`—back in the next request.
Preserved reasoning then participates in the same exact causal-prefix gate.

The Hermes checkout tested on 2026-08-01 stored Moonshine's returned
reasoning but stripped `reasoning_content` when replaying a generic custom
endpoint. That behavior prevents exact K3 prefix reuse. Hermes must classify
the exact `moonshine` model ID as requiring exact reasoning replay (or expose
an equivalent custom-provider option) before a multi-turn result is considered
qualified; storing the reasoning in its session database is not sufficient.
This is not hosted Kimi's padding policy: Moonshine preserves the returned
string byte-for-byte, including an empty string, and does not synthesize a
placeholder for missing reasoning. A patched Hermes checkout using that exact
rule completed a live terminal-tool/result/final-answer loop: the result turn
reused all 4,300 retained tokens and evaluated only its 51-token suffix.

Hermes Agent's generic custom-provider defaults (`max_tokens: 65536` and
`reasoning_effort: medium`) match the 128K/64K Moonshine 0.2.0 source
profile. A running server built before this change still advertises and
enforces 32,768 and rejects `medium`; lower-context profiles also require a
smaller Hermes output cap. Always use the discovered server limits and extend
API/read timeouts to at least 30 minutes. Disable Hermes's automatic title
generation or route it to a separate fast model: its independent 30-second
timeout and retries are unsuitable for Moonshine's one slow request slot and
the divergent prompt replaces the one retained causal prefix. Copy-pasteable
8K, 16K, 32K, and persistent-128K profiles are in [Deployment profiles and
Hermes Agent](docs/deployment-profiles.md).

Hermes `approvals.mode: smart` has the same issue for terminal commands that
its static detector flags: it sends a separate non-streaming security-review
prompt through `auxiliary.approval`, which defaults to the main model. Route
that task to a fast independent provider, or use `approvals.mode: manual` so
an interactive user makes the decision without another model request. Use
`off` only in an intentionally unrestricted trusted environment. One observed
review displaced an 11,673-token chat prefix with a divergent 412-token prompt
and occupied the only slot for 262 seconds. Until Moonshine has a bounded
multi-entry causal-state cache, all auxiliary calls sharing its endpoint have
this correctness and latency cost. The qualified client profile now sets
`approvals.mode: manual` explicitly.

`response_format` supports both `json_object` and a bounded `json_schema`
mode using K3's native response-format directives. `json_object` requires a
top-level object. `json_schema` validates `type`, object `properties`,
`required`, boolean `additionalProperties`, array `items`, and string
`title`/`description`; supported types are object, array, string, number,
integer, boolean, and null. Every schema node needs one string `type`, and
unsupported JSON Schema keywords are rejected before inference. For SSE,
reasoning remains live but response content is buffered until the complete
value passes validation.

```sh
curl --max-time 0 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "moonshine",
    "messages": [{"role": "user", "content": "Weather in Toronto?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "required"
  }'
```

SSE streams response text as it is decoded and emits each completed function
call as an indexed `delta.tool_calls` chunk before the terminal
`finish_reason: "tool_calls"` chunk. See [Agentic API and tool
use](docs/agentic-api.md) for a complete two-request loop and exact support
boundaries.

Layer-major TTFT can be many minutes. Configure OpenAI SDK read/request
timeouts accordingly; the examples use curl's unlimited timeout. Streaming
responses send spec-legal SSE comment keepalives during prefill, with token or
layer progress throttled to ten seconds and emitted at completed-unit
boundaries. Ordinary JSON responses cannot send keepalives.

The server also emits timestamped lifecycle logs to standard error. They make
the exact prefix-reuse decision visible before prefill, distinguish hidden
reasoning from the buffered response/tool region, report coarse progress for
long work, and finish with prompt/decode/cache/I/O accounting. Interactive
terminals receive restrained level color; redirected logs remain plain text,
and `NO_COLOR=1` disables color explicitly. Prompts, reasoning, response text,
tool arguments, request bodies, bearer tokens, and API keys are never logged.
See [Operational logging](docs/observability.md) for the event contract and
examples, and [its qualification](docs/qualification-observability.md) for the
gate evidence.

For a bounded decode investigation, `--decode-diagnostics PREFIX` writes
private (`0600`) cache, per-layer ledger, and content-derived expert-route
CSVs, while `--decode-state-digest` logs non-cryptographic causal-state
comparison fingerprints for paired runs. Both are sensitive, opt-in
qualification tools, not production logging. The schemas, privacy boundary,
required baseline comparison, and
[qualification gate](docs/qualification-decode-diagnostics.md) are documented
in [Operational logging](docs/observability.md). Accepted captures can be
replayed with the deterministic
[offline cache analyzer](docs/offline-decode-cache-analysis.md); its optimized
allocations and full-trace pinning results are explicitly nonpromotable oracles.

Append-only reuse is automatic; send each turn with the complete history:

```sh
curl --max-time 0 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @request-with-complete-history.json
```

All responses report reused prompt tokens through the standard
`usage.prompt_tokens_details.cached_tokens` field. Non-streaming responses
also report evaluated and reused prompt tokens in
`X-Moonshine-Prompt-Evaluated-Tokens` and
`X-Moonshine-Prompt-Reused-Tokens`. SSE streams report the same values in a
final comment. Use `--clear-expert-cache-per-request` only for explicit cold
cache benchmarks; it also disables prefix reuse for that request.

In the live 8K integration check, an appended 67-token second turn reused 42
tokens and evaluated only the 25-token suffix in 54.742 seconds. Replaying all
67 tokens sequentially would have taken about 152 seconds at the measured
short-prompt rate. Preserving only the expert cache improved repeated short
stateless prompts by about 0.8%; exact causal-prefix reuse is the material
agentic latency optimization.

In the qualified SSE function loop, Moonshine reconstructed the prior hidden
`tool_choice: "required"` directive, reused all 196 retained prompt/generated
tokens, and evaluated only the 42-token result-turn suffix. Before selected-
prefill crossover promotion, that token-major suffix took 104.913 seconds;
the matched range fixture now takes 43.742 seconds for 42 tokens. Any
reconstructed prefix mismatch still takes the isolated full-prefill fallback.

In a two-turn JSON Schema continuation, Moonshine reconstructed the prior
schema directive and reused all 237 retained prompt/generated tokens. It
evaluated only the 117-token suffix of the 354-token second prompt and returned
validated `{"greeting":"goodbye"}`. That suffix exceeds the measured
8-token layer-major threshold. Selected-only routed I/O reduced its prefill
from the paired 211.473-second full-store baseline to 104.469 seconds while
preserving exact causal reuse and output. Exact reuse removes history-length
growth; route-union service removes the unused-expert tax from the remaining
suffix.

Set `MOONSHINE_API_KEY` or pass `--api-key` to require bearer authentication.
The server refuses a non-loopback bind without a key. It accepts text-only
system/developer, user, assistant, and tool history, plus Kimi dynamic tool
declarations on system messages. The engine is deterministic and greedy.
`parallel_tool_calls: false` adds a Moonshine hidden one-call directive and a
hard post-parse check; multiple calls can never be returned for that request,
although the constraint is not grammar-enforced during generation. Sampling,
parallel request slots, and HTTP request chunking are not implemented yet.
JSON Schema support is intentionally limited to the declared subset above,
not the complete vocabulary.

## Obtain the model

Weights are not included. They are governed by the
[Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/LICENSE)
published with the
[official Kimi K3 model release](https://huggingface.co/moonshotai/Kimi-K3),
not by this code repository.

With the Hugging Face CLI installed, download the pinned revision to a local
SSD:

```sh
hf download moonshotai/Kimi-K3 \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --local-dir /path/to/moonshotai__Kimi-K3
```

The engine expects all 96 `model-*.safetensors` shards and the original tensor
names/layout. Validate the directory without allocating the full engine:

```sh
make test-model-layout MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

See [Getting started](docs/getting-started.md) for host checks, component
tests, full-residency safeguards, and benchmark commands.

## Run the locked end-to-end fixture

Stop every other large inference process first. Confirm at least 120 GiB is
available and that swap is idle:

```sh
free -h
pgrep -a -f 'llama|ds4|k3'
make test-engine-hello MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-engine-hello \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_CONTEXT=131072
```

This loads the accepted Q8/32 residency and runs a hard-coded, tokenizer-
verified `Say hello.` chat fixture. A passing run should end with:

```text
K3 hello: PASS
```

The fixture reports token IDs, startup time, prompt/decode rate, and cumulative
expert-cache hits. It does not decode arbitrary user text.

## Run layer-major prefill

For any reduction-order or kernel-schedule change, first run the complete
self-hosted qualification bundle:

```sh
make test-reduction-qualification \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

It combines a model-shape scalar/vector MXFP4 numerical envelope with real
expert/MoE, complete routed-layer hash, tokenizer/XTML, and locked chat gates.

The two-token oracle compares range execution with sequential execution,
including KDA, convolution, MLA, and AttnRes state hashes:

```sh
make test-prefill-2 MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

Run the scale fixture:

```sh
make test-prefill-scale \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_PREFILL_TOKENS=512
```

| Qualified selected-prefill fixture | Throughput |
| --- | ---: |
| Filled 8,192 positions | 8.126 tok/s |
| Filled 16,384 positions | 7.929 tok/s |
| Filled 32,768 positions | 7.462 tok/s |

The 32K result retains 94.1% of the 16K throughput while doubling the filled
prefix. Its routed-weight traffic rises only 3.4%, but attention time rises
2.291x; KDA stays near-linear while MLA rises 2.804x. See the
[filled-context qualification](docs/qualification-filled-context.md) for the
complete phase ledger and host evidence.

Run the separate deterministic natural-text retrieval gate:

```sh
make test-long-context-retrieval \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

It fills 15,993 rendered tokens with 389 operational records, places three
retrieval keys near 12.5%, 50%, and 87.5%, and requires the exact decoded
answer `saffron|7319|Nivens`. Set `MOONSHINE_RETRIEVAL_TARGET=512` for a
short staged harness check. Set it to `32000` for the qualified 31,999-token
arm, which uses 781 records and requires the same exact answer. The 32K arm
completed prefill at 7.488 tok/s and retrieved all three values exactly.

For the graduated filled-context program, set both values explicitly:

```sh
make test-prefill-scale \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_CONTEXT=16384 \
  MOONSHINE_PREFILL_TOKENS=16384
```

The accepted selected-expert 16K run took about 34.4 minutes. The accepted
32K run took about 73.2 minutes and can be reproduced by changing both values
to `32768`.

Profile the model-shape MoE tail without weights or SSD reads:

```sh
make test-moe-tail-profile
```

The profiler isolates weighted reduction, RMSNorm, routed-up Q8, residual
adds, and a bounded dequantized-hipBLAS comparison at 8K/16K/32K. The latter
is diagnostic and is not the production path.

The historical 8K full-store backend below is diagnostic:

```sh
make test-prefill-kda-blas \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_PREFILL_TOKENS=8192
```

Run the bounded natural-text retrieval gate through that projection backend:

```sh
make test-long-context-retrieval \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_RETRIEVAL_TARGET=512 \
  MOONSHINE_RETRIEVAL_BACKEND=kda-blas
```

The same diagnostic can be selected for chat/API qualification with
`moonshine-server --range-backend kda-blas`. The default remains `default`;
the option affects multi-token range execution only and retains per-layer SSE
prefill progress keepalives.

## Save and restore causal state

The version-1 state file contains KDA recurrent and convolution state,
occupied MLA rows, AttnRes state, and the token position. It binds that
payload to the exact model tensor layout and static precision mode. Derived
MLA packed keys and the expert cache are deliberately excluded.

Run the exact continuation and corruption fixture:

```sh
make test-state-checkpoint \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_STATE_DIR=/tmp
```

The two-position fixture writes a 433.569 MiB file. Import validates the
complete header, layout identity, file size, and CRC64 payload before it
modifies device state. It then proves bit-exact continuation through three
additional generated tokens on a fresh engine with an empty expert cache.
The same test rejects header corruption, a checksum-valid stale model
identity, payload corruption, and truncation without mutating the destination
engine.

## Project status

Versioned causal-state export/import, the official native tokenizer, the
native XTML chat/tool renderer, deterministic stateful CLI, dynamic context
allocation capacity-qualified through 128K, and one-slot OpenAI-compatible
HTTP/SSE service with a qualified two-turn function-tool loop are now locked.
Agentic hidden-directive prefix recovery and the real SSE tool wire are also
qualified locally, along with preserved thinking history and live reasoning
deltas. Bounded JSON Schema responses and serial non-parallel tool loops are
also locally qualified. The official Python SDK now consumes reasoning and
indexed tool-call SSE across a complete two-request loop. The diagnostic KDA
backend passes the bounded natural-retrieval and SDK tool-loop gates, but
promotion and broader task quality remain open. Filled-128K workload behavior
also remains open.

See [Architecture](docs/architecture.md) for the correctness model and
[Provenance](docs/provenance.md) for code lineage, references, and
acknowledgements.

## Contributing and security

Before contributing, read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of conduct](CODE_OF_CONDUCT.md). Report vulnerabilities through the
private process in [SECURITY.md](SECURITY.md), not a public issue.

Release history and maintainer qualification steps are in
[CHANGELOG.md](CHANGELOG.md) and [RELEASING.md](RELEASING.md).

## License

The code is MIT licensed. Kimi K3 model weights are separate, are not
redistributed here, and remain subject to the
[Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/LICENSE).
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Development disclosure

This work was developed with strong AI coding and review assistance, with the
human maintainer directing architecture, experiments, validation, and release
decisions.
