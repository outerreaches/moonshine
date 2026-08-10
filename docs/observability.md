# Operational logging

`moonshine-server` writes request lifecycle logs to standard error. The text
format is intended to be readable in an interactive terminal and stable enough
for ordinary `journalctl`, `grep`, and log-forwarder ingestion. The values in
this abbreviated field-shape example are illustrative:

```text
2026-08-02T03:15:04.506Z INFO  request.start             id=chatcmpl-moonshine-... peer=192.0.2.10 stream=yes messages=3 max_output=65536 reasoning=medium tools=47 tool_choice=auto parallel_tools=yes format=text
2026-08-02T03:15:04.532Z INFO  request.prefill.start     id=chatcmpl-moonshine-... prompt=3946 evaluated=33 reused=3913 reuse=hit strategy=layer-major
2026-08-02T03:15:48.274Z INFO  request.prefill.complete  id=chatcmpl-moonshine-... evaluated=33 reused=3913 seconds=43.742 rate=0.754_tok/s read=0.000_GiB workspace_borrow=0.000_GiB
2026-08-02T03:15:48.274Z INFO  request.decode.start       id=chatcmpl-moonshine-... phase=reasoning max_output=65536
2026-08-02T03:18:30.118Z INFO  request.decode.phase       id=chatcmpl-moonshine-... phase=response_or_tool generated=68 elapsed=161.844s note=tool_payloads_buffered
2026-08-02T03:21:01.220Z INFO  request.complete           id=chatcmpl-moonshine-... prompt=3946 evaluated=33 reused=3913 prefill=43.742s generated=121 decode=312.946s rate=0.387_tok/s total=356.714s finish=tool_calls tool_calls=1 reasoning_bytes=192 content_bytes=0 stream=yes client=connected cache=52176/113344 cache_total=217327/476928
```

ANSI color is applied only when standard error is an interactive terminal.
Redirected files and systemd journals remain plain text. Set `NO_COLOR=1` to
disable terminal color explicitly.

## Event contract

| Event | Meaning |
|---|---|
| `server.load.start` | Effective model path and memory/scheduling configuration before engine creation. |
| `server.ready` | Listener, version, context/output ceilings, load time, and memory tiers after successful startup. |
| `request.start` | Validated request envelope. It records counts and policy, never message content. |
| `request.prefill.start` | Exact prompt accounting and reuse decision before reset, workspace admission, or inference. |
| `request.prefix.miss` | A retained state existed but the candidate was not an exact extension; includes retained/matched/candidate token counts. |
| `request.prefill.progress` | At most one terminal update per minute during long prefill. SSE clients independently retain their ten-second comment keepalives. |
| `request.prefill.complete` | Prefill duration, evaluated/reused tokens, physical reads, and any guarded cache-tail workspace loan. |
| `request.decode.start` | Decode began in the reasoning or response/tool region. |
| `request.decode.phase` | Thinking closed and generation entered the response-or-tool region. Tool payloads are structurally buffered until they parse and validate. |
| `request.decode.progress` | A heartbeat every 64 model-produced tokens, including the current region and decode elapsed time. |
| `request.complete` | Final prompt/decode/total timing, finish reason, output byte counts, tool-call count, client state, and cache counters. |
| `request.reject` / `request.failed` | Admission or inference failure with an HTTP/inference stage and bounded diagnostic. |
| `request.client_disconnect` | The client disappeared before the initial SSE event could be established. Later disconnects appear as `client=disconnected` on completion. |

The `request.prefill.start` event is intentionally emitted before a divergent
request can reset retained causal state. Operators can therefore distinguish a
small cache-hit suffix from a full replay while the work is in flight. A reuse
miss logs `replacement=guarded`; the existing memory preflight and exact-prefix
predicate are unchanged.

The decode phase events address a different ambiguity. Reasoning is streamed
live, but a client may hide it. Parsed tool calls are emitted only when the
complete XTML call is valid. A quiet user interface can therefore represent
active reasoning or active buffered tool generation rather than a stalled
server. The phase and 64-token heartbeat events make that state visible without
logging model output.

## Privacy and safety

The lifecycle log does not contain:

- system, user, assistant, or tool-result message text;
- generated reasoning or response text;
- tool names or arguments;
- request JSON bodies;
- bearer tokens or configured API keys.

It does contain the numeric client address, model path at startup, request
policy/counts, timing, memory/I/O accounting, and response byte counts. Control
characters in formatted diagnostic values are replaced with spaces, and an
oversized diagnostic is bounded with `log_truncated=yes`.

## Performance boundary

Logging does not alter kernels, model arithmetic, routing, SSD queueing, expert
cache policy, or scheduling. Host callbacks occur only at lifecycle boundaries,
once per prefill minute, and once per 64 generated tokens. There is no per-layer
terminal output other than the existing throttled prefill mechanism and no
per-token logging. Each record is assembled in a bounded stack buffer and
submitted with one best-effort host `write(2)` call, preventing line
interleaving and avoiding repeated stdio operations.

For systemd, keep standard error attached to the journal. For a manual run:

```sh
./moonshine-server /path/to/moonshotai__Kimi-K3 [options] \
  2>>moonshine.log
```

SSE progress comments remain part of the client-facing compatibility surface.
Some clients, including the qualified Hermes configuration, do not count SSE
comments as parsed model-output activity; their read and stale-stream timeouts
must still follow the deployment guidance in
[Deployment profiles](deployment-profiles.md).
