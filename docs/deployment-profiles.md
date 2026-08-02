# Deployment profiles and Hermes Agent

These profiles describe configurations exercised on the qualified 128 GB
Ryzen AI Max+ 395 / Radeon 8060S host. They are operating points, not claims
that every possible prompt or full-length completion has been run.

## Qualified profiles

| use | context | experts/layer | output ceiling | qualification boundary |
|---|---:|---:|---:|---|
| agentic/API baseline | 8,192 | 32 | 8,192 | official OpenAI SDK two-turn tool loop, reasoning, SSE, and exact prefix reuse |
| bounded long context | 16,384 | 32 | 16,384 | filled-context execution and 15,993-token natural-text retrieval |
| long context | 32,768 | 32 | 32,768 | filled-context execution and 31,999-token natural-text retrieval |
| maximum configured capacity | 131,072 | 30 | 65,536 | 64K admission, remaining-context clamp, and two independent short requests in one persistent server |

The 128K profile uses 30 expert slots deliberately. With 32 slots, the first
cold request can borrow empty expert-cache storage as prefill workspace, but a
later request must retain the warmed cache and allocate separate workspace.
That second allocation can violate the CMA-plus-4-GiB safety reserve. The
30-slot cache is 45.104 GiB and passed two independent 128K requests.

Filled 64K/128K execution, filled-128K quality, and continuous 32K or 64K
completions remain outside the qualification boundary.

## 0.2.0 output expansion

Moonshine 0.2.0 raises the absolute output ceiling to 65,536 and accepts K3's
native `reasoning_effort: medium`. Portable parser, tokenizer, and official-SDK
replay gates passed. The 128K/30-expert configuration also passed live
discovery, 65,536 admission, 65,537 rejection, remaining-context clamping, a
naturally stopped `medium` response, and two independent requests in one
process. The new ceiling is a maximum request budget, not evidence of a
continuous 64K decode or filled-128K prompt. Exact evidence is in
[64K output and medium-reasoning qualification](qualification-output-64k.md).

## Server commands

Use an API key whenever the service binds beyond loopback.

Agentic 8K baseline:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 8192 --experts 32 \
  --max-output-tokens 8192
```

Qualified 32K profile:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 32768 --experts 32 \
  --max-output-tokens 32768
```

Persistent 128K-capacity 0.2.0 profile:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 131072 --experts 30 \
  --max-output-tokens 65536
```

## Hermes Agent compatibility

Hermes Agent's generic custom-provider profile defaults to 65,536 output
tokens and its standard reasoning setting is `medium`. Both match the
128K/64K Moonshine 0.2.0 profile. A server built from the earlier
32K checkpoint still rejects both values, so confirm the running contract
through `/health` or `/v1/models` rather than assuming the checkout and
service binary match.

Set both limits explicitly in `~/.hermes/config.yaml`. This example matches
the persistent 128K Moonshine profile:

```yaml
model:
  default: moonshine
  provider: custom
  base_url: http://MOONSHINE_HOST:8080/v1
  api_key: YOUR_MOONSHINE_API_KEY
  max_tokens: 65536
  context_length: 131072

agent:
  reasoning_effort: medium
  local_stream_stale_timeout: 1800

auxiliary:
  title_generation:
    enabled: false

# Avoid a divergent auxiliary LLM request on Moonshine for flagged commands.
# Keep human approval; alternatively route auxiliary.approval to a separate
# fast provider.
approvals:
  mode: manual
```

`low`, `high`, and `max` are also valid. Restart Hermes after changing the
configuration. For a 32K Moonshine server, set `context_length: 32768` and
`max_tokens: 32768`; a request cap cannot exceed the server's effective
context/output ceiling.

### Reasoning replay is required

Moonshine returns K3 thinking separately as `reasoning_content`. Hermes must
send that exact field back on every historical assistant message. A client
checkout at Hermes commit `87bc710` stored all 181 characters of a first-turn
reasoning trace, but its generic-custom-provider policy removed the field from
the next wire request. The resulting prompt rendered to 3,905 tokens while
Moonshine retained 3,913 causal tokens, so exact reuse was impossible and the
server attempted a full prefill. Restoring the stored reasoning renders 3,946
tokens: the 3,913-token retained prefix plus a 33-token suffix.

Qualified local Hermes commit `a3e7c54` uses a narrow exact-replay rule for the
exact `moonshine` model ID. It copies a stored `reasoning_content` string verbatim,
including `""`, or promotes Hermes's stored `reasoning` string when the wire
field is absent. It does not use hosted Kimi/DeepSeek's single-space padding
behavior. A future Hermes custom-provider option with the same behavior is
equivalent. Merely setting `reasoning_effort: medium` controls generation; it
does not ensure historical reasoning is replayed. Restart Hermes after making
the client change.

The live terminal qualification supplied 4,230 tokens and generated a 70-token
`Terminal("pwd")` call. After the 0.1-second tool execution, Hermes replayed
the complete assistant object and result. Moonshine reported 4,351 prompt
tokens, 4,300 reused/cached tokens, and only 51 evaluated tokens, then generated
a 47-token final answer containing the working directory. Exact reasoning
replay is therefore qualified across the actual Hermes tool boundary.

Moonshine emits SSE progress comments during prefill, but this engine is much
slower than a cloud endpoint. For ordinary short-prompt agent use, make the
Hermes timeouts explicit:

```sh
HERMES_API_TIMEOUT=1800
HERMES_STREAM_READ_TIMEOUT=1800
```

These HTTP/read settings do not override Hermes's independent local-stream
stale detector. Its default is 900 seconds and it does not count Moonshine's
SSE comment keepalives as model-output chunks. Set
`agent.local_stream_stale_timeout: 1800` (or
`HERMES_LOCAL_STREAM_STALE_TIMEOUT=1800`) as well; otherwise a long prefill can
trigger a reconnect at 15 minutes while the original request is still running.

These are seconds. Hermes normally raises the stream read timeout for LAN and
loopback endpoints, but explicit values remove endpoint-detection ambiguity.
An early test was manually interrupted after 14 minutes 49 seconds while it
was still inside the API call; Hermes did not log a timeout exception. The
successful retry finished in 663.7 seconds. Use at least 1,800 seconds for both
request and stream-read timeouts to leave margin for prompt and decode
variation even when the user's visible prompt is short.

### Auxiliary requests and the one causal slot

Disabling title generation does not disable every Hermes auxiliary LLM call.
With the default `approvals.mode: smart`, a shell command that matches Hermes's
static risk detector is assessed by `auxiliary.approval`. That task defaults to
the main provider and model, so it reaches Moonshine as a separate
non-streaming Chat Completions request unless explicitly rerouted. The request
is unrelated to the conversation, replaces Moonshine's one retained causal
prefix when it completes, and blocks the only inference slot while it runs.

During the 2026-08-01 Hermes tool qualification, this path displaced an
11,673-token chat prefix with a 412-token security-review prompt. Moonshine
prefilled it in 185.884 seconds and decoded a 29-token verdict in 76.319
seconds. The safe deployment choices are:

1. set `approvals.mode: manual` for an interactive Hermes client;
2. configure `auxiliary.approval` with a separate fast provider/model; or
3. use `approvals.mode: off` only when unrestricted execution is explicitly
   intended and independently contained.

Increasing the auxiliary timeout does not preserve the chat prefix. This is a
current one-state engine limitation, not an OpenAI wire incompatibility.
Pin `approvals.mode: manual`; the effective mode was confirmed
after the qualification process exited.

The current Hermes checkout can render substantially more context than the
earlier 3,849-token fixture: one fresh session supplied 11,124 tokens and took
1,391.629 seconds to prefill at 7.994 tok/s, then 178.227 seconds to decode 75
tokens. Its 1,569.856-second engine time fits the 1,800-second tier, but only
after raising the independent stale-stream limit as described above.
For deliberately long 16K/32K prompts, start with 7,200 seconds for both
values. The qualified filled-32K prefill took about 73 minutes before decode,
so the ordinary 30-minute tier is not sufficient for that workload.

The server currently has one blocking accept/inference path, so **do not poll
it on a timer**. A probe opened while a request is in flight completes its TCP
handshake in the kernel and then waits in the accept backlog; the client timing
out does not remove it. A five-second monitor was observed filling the
16-entry backlog during a single long request (`Recv-Q=17`), after which the
kernel refuses further connections — including agent continuations.

Until Moonshine offers a busy-safe status path, monitor it passively: the
listening socket proves residency, `/proc/<pid>/cmdline` carries the configured
context, and a non-empty accept queue indicates the slot is occupied. Check
`ss -ltn 'sport = :8080'` rather than issuing a request.

The timeout only prevents premature client cancellation. It does not change
Moonshine's single-request slot, model speed, context accounting, or output
ceiling. Prefer streaming for Hermes so progress keepalives can traverse the
connection; ordinary JSON responses cannot emit keepalives.

Hermes title generation is a separate non-streaming request with its own
30-second default timeout. The observed helper made repeated 30-second
attempts after the successful agent turn and then failed. Do not merely raise
that timeout while using Moonshine's single request slot: the title prompt is
not an append-only continuation and replaces the one retained conversation
prefix. Keep it disabled as above, or configure a separate fast auxiliary
provider for title generation.

The first successful Hermes/Moonshine integration turn supplied 3,849 input
tokens—not the estimated 10K--11K—and generated 64 tokens. Moonshine spent
513.140 seconds on prefill at 7.501 tokens/s and 150.376 seconds on decode at
0.426 tokens/s; Hermes measured 663.7 seconds end to end. This confirms that
the 30-minute main-request tier is adequate for this initial prompt while also
showing why the auxiliary helper's 30-second tier cannot work.

Moonshine selects append-only reuse automatically by exact token content, so
Hermes does not need a nonstandard request header. Confirm a continuation hit
through the standard `usage.prompt_tokens_details.cached_tokens` field. A
zero value means the request did not exactly extend the retained prefix.

At 128K/30 experts, a missed replay of the observed 3,905-token prompt needed a
4.060 GiB warm prefill workspace. The guarded check rejected it with only
12.881 GiB available because workspace + 5.316 GiB CMA reserve + the 4 GiB
host guard required 13.376 GiB. Do not lower the guard to mask a client-history
mismatch. Correct reasoning replay reduces this continuation to 33 evaluated
tokens. Moonshine's warm-cache workspace lease can make a guarded full replay
possible without lowering the memory guard, but it remains a slower fallback
and invalidates only the cache slots whose storage the prefill borrows.

The live fallback gate used a divergent 3,990-token request after the Hermes
loop had warmed the 128K/30-expert service. Moonshine logged
`retained=4398 matched=8 candidate=3990`, borrowed 4.151 GiB / 254 physical
slots, retained 40.953 GiB of non-overlapping cache, and completed the full
prefill in 531.965 seconds at 7.500 tokens/s. The client received HTTP 200 with
standard usage `3990/1/3991`; no memory guard was reduced.

## Client-side failure map

| HTTP 400 message | likely correction |
|---|---|
| `max tokens must be an integer in [1,N]` | set Hermes `model.max_tokens` to the advertised server ceiling `N` or lower |
| `reasoning_effort must be low, high, or max` | the running binary predates `medium`; use `low`, `high`, or `max`, or rebuild after stopping it |
| `K3 prefill workspace rejected` on a continuation | first verify that every prior assistant `reasoning_content` and tool call was replayed; a prefix miss can turn a small suffix into a full warm prefill |
| prompt/context capacity error | reduce history/output cap or select a larger qualified context profile |

Always confirm the running server's effective contract before client tests:

```sh
curl http://MOONSHINE_HOST:8080/health
curl http://MOONSHINE_HOST:8080/v1/models
```

Both responses advertise `context_length` and effective
`max_output_tokens`.
