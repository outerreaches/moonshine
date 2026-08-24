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
| `request.state.digest` | Non-cryptographic causal-state comparison fingerprints emitted only with `--decode-state-digest`; expensive and intended for paired qualification. |
| `request.decode.io` | Per-request decode I/O/timing aggregate emitted only when decode diagnostics are enabled. |
| `request.complete` | Final prompt/decode/total timing, finish reason, forced-trailer count, output byte counts, tool-call count, client state, and cache counters. |
| `request.reject` / `request.failed` | Admission or inference failure with an HTTP/inference stage and bounded diagnostic; post-decode failures include the diagnostic capture ID. A busy completion is rejected with status 503 and `reason=busy`. |
| `request.client_disconnect` | The client disappeared before the initial SSE event could be established. |
| `request.cancelled` | A started request stopped at a safe token/layer boundary because the peer disconnected or the server began shutdown. Includes reason, phase, generated-token count, and wall time; semantic state is reset before the slot releases. |

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

## Opt-in decode diagnostics

`--decode-diagnostics PREFIX` creates three new regular files with mode `0600`
and refuses to follow symlinks or overwrite an existing path. The additional
`--router-logits-tap` flag requires that diagnostics prefix and creates a fourth,
more sensitive file:

```text
PREFIX.cache.csv
capture,layer,lru_rank,expert_id

PREFIX.ledger.csv
capture,scope,layer,steps,accesses,hits,misses,read_requests,logical_expert_bytes,physical_read_bytes,wait_calls,completions,max_inflight,pre_moe_seconds,io_wait_seconds,expert_pipeline_seconds,expert_sync_seconds,shared_sync_seconds,host_interval_seconds

PREFIX.routes.csv
capture,step,position,layer,observed_hit_mask,expert_0,...,expert_15

PREFIX.router_logits.f32
# moonshine-router-logits-v1 layers=92 experts=896 dtype=float32 order=step-major-layer-minor
<896 native float32 values for each decode step and routed layer>
```

Capture IDs start at 1 for each server process; aborted attempts can leave gaps
between committed IDs. Route steps start at 0, routed layers are 1 through 92, and hit-mask bit *r* describes `expert_r`. Cache rank 0
is the oldest LRU resident; increasing ranks run to the newest resident. A
header-only cache file is a valid empty pre-decode snapshot. A capture commits
when the native chat decode and optional state fingerprint complete. Engine or
chat-decode failures before that boundary roll every enabled stream back to its
pre-request offset. Later server tool-policy, response-format, allocation, or
transport failures do not invalidate the completed engine trace and can leave a
committed capture beside `request.failed`; those post-decode failures carry the
capture ID so consumers can correlate them when HTTP success is part of the
analysis. A process or storage crash can still leave a
partial final capture, so consumers must validate capture IDs, step/layer
completeness, and ledger reconciliation rather than accepting a CSV merely
because it parses.

The ledger contains one `scope=summary,layer=0` row and 92 `scope=layer` rows
per committed capture. Summary `steps` is the number of decode engine token
evaluations; each layer row records that layer's invocation count. Access,
read, byte, completion, and timing totals aggregate the layer rows. Integer byte
fields are bytes; timing fields are seconds. `host_interval_seconds` is a host
clock interval from layer entry until its final work is enqueued, not an
independent GPU completion time. A following layer's synchronization can charge
prior asynchronous tail work to its own pre-MoE interval, so use the timing
columns for bounded attribution rather than summing them as disjoint phases.

Decode steps include structural tokens evaluated to leave retained chat state
well formed. Consequently `request.decode.io steps` and route step count equal
`generated_tokens + forced_trailer_tokens`; an API `completion_tokens=128`
length stop can, for example, produce 141 routed steps after 13 forced closure
tokens. The route trace contains no token IDs, prompt text, generated text, or
gate weights, but its selected experts and positions are content-derived and
can fingerprint a workload. Router logits are raw model activations and carry a
higher disclosure risk. They are written only while a decode-diagnostics
transaction is active; prompt and prefill router values remain outside the file
and rollback boundary.

`--prefill-diagnostics PREFIX` creates
`PREFIX.prefill.csv` as a private `0600`, exclusive, no-follow file. Each
committed range capture contains exactly 92 routed-layer rows. It records
content-free host phase timings, HIP-event spans for the default/expert/shared
streams and MZG2 decoder, time-weighted QD2 occupancy, physical reads, route
union size, and deterministic command counts. It never records token IDs,
expert IDs, routes, weights, logits, text, or credentials. A failed range
rolls the file back to its pre-capture offset.

Run `tools/analyze_prefill_screen.py --noise-only` first with exactly five
files: discarded warm-up followed by four identical noise-floor baselines. If
the gate is smaller than three times the target-metric spread, the tool returns
`UNRESOLVABLE` and the protocol stops before ABBA. Only a
`READY_FOR_ABBA` result permits the complete nine-file analysis, adding
baseline A, candidate A, candidate B, and baseline B. Both modes reject
route-union or I/O drift; the complete mode gates on the selected subphase and
treats aggregate phase wall as a no-regression guard. JSON reports are
atomically published mode `0600`.

`--decode-state-digest` adds `request.state.digest` before the final
`request.complete` event. It computes deterministic 64-bit FNV-1a comparison
fingerprints over the complete causal-state regions after decode. Matching all
four values and position is a useful paired qualification gate, but the hashes
are non-cryptographic, collision-prone in principle, and not an independent
proof of state equality. Digest computation is deliberately outside the decode
timer and is too expensive for normal service. `request.decode.io` appears in
the same pre-completion diagnostic block when CSV capture is active.

All diagnostics are experimental and off by default. Use a fresh
single-request server for paired measurements: baseline with only
`--decode-state-digest`, then an identical process with decode diagnostics and
only the additional taps required by the experiment. Keep the prefix outside the source
tree and retain the `0600` modes. Raw routes and state fingerprints require an
explicit disclosure review.

The built-in replay gate accepts one complete capture and requires its source
capacity explicitly; snapshots do not encode evicted prefill history. Replay at
or below a warm source capacity is valid. A larger target requires both a
completely empty snapshot and an explicit operator assertion that it came from
a fresh process before any cache history; the tool otherwise rejects expansion
because prior evictions or explicit slot invalidations cannot be reconstructed.
The [offline cache analyzer](offline-decode-cache-analysis.md) applies the same
source-replay trust gate before producing uniform, marginal, fixed-memory, and
explicitly labeled policy counterfactuals. See
[Decode-diagnostics qualification](qualification-decode-diagnostics.md) for
the paired exactness, accounting, privacy, replay, and overhead gates.

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

Default lifecycle logging does not alter kernels, model arithmetic, routing,
SSD queueing, expert-cache policy, or scheduling. Host callbacks occur only at
lifecycle boundaries, once per prefill minute, and once per 64 generated
tokens. Each record is assembled in a bounded stack buffer and submitted with
one best-effort host `write(2)` call.

`--decode-diagnostics` is the explicit exception: it adds per-layer clocks and
buffered CSV writes. The router-logits tap additionally copies and writes 896
float32 values for every routed layer and decode step. Qualification must
compare each enabled configuration with an identical baseline, require exact
output/cache/state results, and bound measured decode overhead before using its
artifacts.

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
