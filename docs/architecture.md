# Architecture

## Scope

Moonshine is a purpose-built execution core for the official Kimi K3 model
layout and one initial hardware class. Its design goal is to make a roughly
1.45 TiB routed MoE checkpoint usable on a 128 GB Strix Halo machine by
keeping model-control weights and mutable state resident while streaming
selected expert weights from NVMe.

It is not a GGUF runtime, framework integration, or portable tensor library.
The public API deliberately exposes engine creation, token/range forward
passes, planning, statistics, and diagnostic state digests rather than tensor
internals.

## Text boundary

The native tokenizer reads the checkpoint's 163,584-entry `tiktoken.model`
without Python or Transformers. ICU executes the official K3/K2 Unicode
pre-tokenizer pattern, including Han separation, Unicode letter/mark classes,
case-aware contractions, one-to-three digit groups, punctuation, and newline
rules. A byte-pair rank table then produces the same 163,840-token ID space,
including 256 control/reserved tokens.

The text-only XTML renderer mirrors the official segment boundary, not merely
its concatenated string. Only structural `<|open|>`, `<|close|>`, `<|sep|>`,
and `<|end_of_msg|>` segments may resolve as control tokens. User/system text,
assistant content/reasoning, names, and escaped attribute values are encoded
as ordinary text, preventing a marker-looking user string from changing chat
structure.

The renderer covers named or unnamed system, user, assistant, and tool
messages; thinking/non-thinking response channels; optional low/medium/high/max
thinking effort; global and dynamically loaded tool declarations; typed or
raw-JSON tool calls; call-ID-resolved tool results; request-local tool choice;
and the open assistant generation prompt. Images and structured response
formats remain open.

## Stateful chat session

`k3_chat_session` composes the tokenizer and engine behind one transport-free
turn API. The first call renders the optional system message plus user turn;
later calls submit only the new user-message and assistant-opener XTML delta
against retained causal state.

Prompt scheduling is length-aware. After selected-only expert reads, the
warm-cache crossover is effectively tied at 3 tokens and has only a 0.6%
range-path lead at 6. The default conservatively uses token-major execution
through 7 tokens and selected layer-major prefill from 8 onward, where two
matched runs showed 1.121x and 1.119x speedups with exact output and causal
state.
Every model-produced token, including the final end-of-message marker, is fed
back into the engine so the position is a complete-turn boundary. A length
stop forces only the missing suffix of the official response/message trailer.

The session emits response token-piece bytes through an optional callback,
also returns the complete response, and reports prompt strategy, TTFT-equivalent
prompt time, decode rate, cache deltas, physical prefill I/O, finish reason,
forced trailer count, and committed position. Export/import delegates to the
exact semantic checkpoint API.

`moonshine-chat` is a thin one-shot or interactive shell over this shared
layer. `moonshine-server` uses the same layer for Chat Completions. It
automatically compares each newly rendered full history with the one retained
in-process causal state. An exact token-prefix extension reuses that state;
any mismatch zero-resets causal storage and prefills the canonical full
history. Historical hidden tool-choice, serial-call, and response-format
directives are restored at their original message boundaries before the same
exact comparison so an ordinary OpenAI client can continue the actual causal
stream without a custom header. Immutable expert-cache mappings survive both
paths.

## OpenAI-compatible transport

The initial server is a deliberately bounded HTTP/1.1 implementation:

- one persistent engine and one inference slot, owned only by the main thread;
- an independent HTTP control thread that continuously accepts requests,
  serves health/model discovery, and rejects competing completions with 503;
- loopback binding by default, with a bearer key required for non-loopback;
- `GET /health`, `GET /v1/models`, and
  `POST /v1/chat/completions`;
- strict native JSON parsing with an 8 MiB default body limit and bounded
  request/send deadlines;
- ordinary JSON completion responses and incremental SSE chunks;
- SSE comment keepalives during both prefill and decode;
- bounded server-side lifecycle logs covering reuse admission, prefill,
  reasoning/response-or-tool decode phases, completion, and failures without
  recording prompt or generated content;
- automatic one-slot append-prefix reuse selected by exact token content;
- UTF-8 boundary buffering so tokenizer byte fragments never corrupt a stream;
- standard OpenAI error envelopes and usage accounting, including cached
  prompt tokens in `usage.prompt_tokens_details`;
- OpenAI function tools, parallel call output, matching tool-result history,
  `auto`/`required`/`none`, and forced named-function selection;
- K3 preserved-thinking history, `low`/`medium`/`high`/`max` effort, and separate
  reasoning/content fields in JSON and SSE;
- native `json_object` and bounded `json_schema` response directives with
  validated, deferred SSE content;
- an optional official Python SDK fixture for keepalives, reasoning
  extensions, indexed function calls, usage, and stream termination;
- fixed greedy execution, with unsupported schema vocabulary rejected before
  inference and non-parallel tool calls hard-checked after parsing.

Tool-call SSE is structurally streamed at message completion: ordinary
response bytes remain token-live, while each parsed call is emitted as one
complete indexed `delta.tool_calls` item before the terminal chunk. This keeps
the OpenAI wire contract without exposing half-parsed XTML attributes.
For `parallel_tool_calls: false`, a Moonshine-specific hidden directive steers
K3 toward one call and the parsed set is rejected if its count exceeds one.
Policy validation precedes emission of those complete call chunks.

The transport observes coarse lifecycle events plus a control checkpoint after
each complete sequential-prefill token, layer-major prefill layer, generated
token, and forced-trailer token. Prefix accounting is reported before a guarded
replacement can reset state; prefill progress is limited to one terminal record
per minute; operator decode progress is reported every 64 generated tokens.
The control checkpoint probes peer state, emits time-bounded SSE comments, and
can stop before the next model unit without changing kernel arithmetic or
scheduling.

Thinking SSE is token-live. The session tracks the native `<think>` close and
`<response>` open sequence exactly, sends pre-transition text only as
`reasoning_content`, and sends post-transition text only as `content`. Natural
completion is parsed again as a complete XTML structure before the session is
accepted. Preserved assistant reasoning is ordinary protected text when it is
rendered into later history and can therefore participate in exact prefix
reuse.

Structured output is post-generation validated rather than grammar-
constrained. Moonshine renders K3's official hidden directive, then requires
either an object root or conformance to its declared recursive JSON Schema
subset. Schema requests are canonicalized before rendering and unsupported
keywords fail before inference. Streaming response bytes are withheld until
the complete check passes, avoiding delivery of an invalid partial contract;
reasoning remains token-live.

K3 renders `required` and `none` as hidden system messages immediately before
the generation prompt. Serial-call and response-format controls have the same
request-local causal property, and JSON Schema controls additionally carry the
canonical schema text. Those messages are absent from the assistant history
returned to the client. For the retained session only, Moonshine records each
directive's value and message boundary and owns a copy of any schema text,
then restores them in their original order before the historical assistant
turn. Reuse proceeds only when this reconstructed prompt exactly extends every
retained token. The canonical full replay remains the fallback; serialized
checkpoints, approximate prefix matching, and state rewinds are not part of
this path.

There is no inference scheduler or hidden model concurrency. The control thread
continues serving discovery while the one inference slot is busy. A peer
disconnect or termination signal cancels after the current complete token/layer,
drains its work, and resets causal state to position zero while retaining
immutable expert-cache mappings. Cancellation never interrupts a HIP kernel,
reuses partial semantic state, or emits a forced trailer for an abandoned peer.

## Model graph

The pinned checkpoint has:

- 93 layers: one dense layer and 92 routed-MoE layers;
- 69 KDA attention layers and 24 gated MLA attention layers;
- hidden width 7,168;
- 896 routed experts per MoE layer, top-16 per token;
- two shared experts;
- 3,584 latent MoE width and 3,072 routed-expert hidden width;
- BF16 static/non-routed tensors;
- group-32 MXFP4 routed weights;
- 12-layer AttnRes blocks.

Layer 0 follows a dense KDA/SiTU path. Layers 1–92 apply AttnRes, KDA or MLA,
router top-16 selection, shared experts, selected routed experts, and the
residual tail. Final AttnRes, RMS normalization, and the resident BF16 language
head produce a greedy token.

## Storage and residency

The loader parses SafeTensors headers from all 96 shards into a sorted tensor
directory without loading payloads. Static tensors always come from this
unchanged source tree. The qualified K3 deployment selects an absolute MZG2
sidecar or complete directory with `MOONSHINE_MZG2_STORE`;
`MOONSHINE_EXPERT_STORE` retains the legacy MZG1 selector. Both paths fail
closed on invalid selection and cannot be enabled together. Unset/`off` uses
raw SafeTensor experts as the portable fallback.

The accepted engine divides memory as follows:

| tier | accepted Q8/32 allocation |
|---|---:|
| Q8/BF16 resident static weights | 55.270 GiB |
| routed-expert cache, 32 slots/layer | 48.111 GiB |
| recurrent/cache/runtime state, 8K | 0.920 GiB |
| recurrent/cache/runtime state, 16K | 1.135 GiB |
| recurrent/cache/runtime state, 32K | 1.566 GiB |
| 16 mapped expert I/O slots | 0.262 GiB |
| 2 MZG2 selected-prefill output slots, when enabled | 0.033 GiB |
| 16 additional legacy MZG1 input slots, when enabled | 0.262 GiB |
| total before allocator/driver overhead | about 104.6 GiB raw / 104.63 GiB MZG2 / 104.9 GiB MZG1 |

Eligible BF16 static projections are quantized one tensor at a time to the
engine's Q8-128 format. Source buffers are released immediately, bounding peak
startup memory. MLA matrices that must remain BF16 are retained as BF16.
Embeddings are read by row; the output head is resident.

Before allocation, the engine calculates load-time and runtime peaks and checks
`MemAvailable` plus the platform's usable CMA reserve. Source-precision BF16
requires an additional guard and is intentionally not accepted on the tested
128 GB host.

## Routed-expert I/O

Each routed expert occupies one contiguous 17,547,264-byte SafeTensors span
containing six MXFP4 data/scale tensors. The baseline engine opens buffered and
`O_DIRECT` shard descriptors, keeps QD2 reads in flight, executes misses from
HIP-mapped staging, and admits them to a device cache asynchronously.

MZG1 is an optional derived expert store beside—not instead of—the official
SafeTensors. Each expert becomes one 4 KiB-aligned block with a 192-byte
descriptor header and twelve Zstandard level-1 frames: two contiguous stripes
for each of the six native tensors. Packed stripes canonicalize E2M1 negative
zero to positive zero; every nonzero code and every scale byte is unchanged.
Zstd content checksums protect every compressed stripe.

The diagnostic MZG path:

1. issues one indexed `O_DIRECT` block read with raw Linux `io_uring`;
2. keeps the next read outstanding at QD2;
3. dispatches twelve stripes over six persistent CPU workers;
4. decodes into HIP-mapped output staging;
5. launches the miss and asynchronously admits it to the ordinary device LRU;
6. publishes cache metadata only after the expert stream completes.

The full store contains 82,432 experts in 1,170.374 GiB, 13.120% below source,
and passes complete block/SHA-256 verification plus deterministic model
qualification. It does **not** pass the engine performance gate. Against the
same engine-hello fixture, MZG prompt wall is 67.210 vs 57.429 seconds and
post-TTFT decode is 42.118 vs 33.766 seconds (0.404 vs 0.503 token/s).
Selected two-token prefill is 8.830 vs 7.970 seconds. MZG1 therefore remains an
opt-in research artifact; raw SafeTensors remain the production default.

MZG2 is the production GPU-decoder store for the qualified K3 deployment. It
divides the canonical expert into independently bounded 16 KiB rANS tiles,
reads each selected block through existing QD2 O_DIRECT staging, and decodes an
admitted miss straight into its device-cache slot on the existing expert
stream. Per-tile checksums, compressed-cursor bounds, fixed output offsets, and
terminal-state checks fail before cache metadata commits. Ninety-two sidecars
contain all 82,432 experts in 1,171.084 GiB, 13.0674105% below source, after
immediate and independent full-source verification. Engine hello improves
about 10.6%, selected prefill 11.6%, and live 128K/30 11.7--12.8%, with exact
outputs. An absolute `MOONSHINE_MZG2_STORE` directory selects it and is mutually
exclusive with MZG1. Unset/`off` retains raw SafeTensors as the rollback path.

## Attention and state

KDA retains recurrent matrix state and causal convolution history per layer.
Its prefill implementation batches projections and convolution, then performs
the recurrence in causal token order.

MLA stores compressed 512-wide latent rows plus the checkpoint's 64-wide
pass-through query/key subspace, 27 KiB per token across all MLA layers. The
official field retains the historical `qk_rope_head_dim` name, but K3 sets
`mla_use_nope=true`, leaves `rotary_emb=None`, and does not rotate this
subspace. Packed key matrices are derived from resident weights at startup;
they are performance data, not causal checkpoint data.

The MLA cache and attention workspace are allocated from the requested
context instead of a compile-time 8K constant. Together they add exactly
28,224 bytes per configured token; the workspace also has a fixed 96 KiB
latent output. Real-residency startup and short-generation checks pass at 8K,
16K, 32K, and 128K. Runtime state is 4,455,923,712 bytes (4.150 GiB) at 128K;
the accepted Q8/32 allocation totals 115,741,620,224 bytes (107.793 GiB)
before allocator and driver overhead. On the qualified 128 GB host, the clean
128K run retained 9.3 GiB `MemAvailable` and did not increase swap use.

This is a configured-capacity qualification: it covers exact state sizing,
full residency, a locked 24-token prompt/18-token completion, and a real JSON
Chat Completions request. It does not establish filled-128K prefill latency or
long-context quality. The model config names a 1,048,576-position
architectural maximum, but memory preflight and latency make that a limit, not
a qualified operating point.

AttnRes retains block inputs and performs the model's learned residual mixing
at exact 12-layer boundaries. Engine state also tracks the current token
position and depth cursor.

At 8K, the exported causal portion is about 649.5 MiB:

- KDA recurrent state: about 414 MiB;
- KDA convolution caches: 19.406 MiB;
- occupied MLA rows: about 216 MiB;
- AttnRes state and token position.

Expert-cache payloads and LRU metadata are performance state and are kept
separate from the semantic checkpoint.

## State persistence

State format v1 uses a fixed 256-byte, explicitly little-endian header and a
compact causal payload. The header records the format and endianness,
configured context, token position, static precision mode, exact segment
sizes, complete payload size, model-layout identity, and CRC64-ECMA checksums
for both header and payload.

The model identity covers shard sizes and data offsets plus the complete sorted
tensor directory. Paths are excluded so a checkpoint remains valid when the
same pinned model tree moves between hosts. The payload contains the complete
KDA recurrent and convolution allocations, only occupied MLA rows, and
AttnRes state. Derived MLA packed keys and routed-expert cache contents are
reconstructed or warmed independently.

Export streams device state through a bounded 16 MiB host buffer, fsyncs a
mode-0600 temporary file, and atomically renames it into place. Import first
validates the complete file, including payload CRC, before changing device
state. A failure after device upload begins invalidates the causal state until
a valid import succeeds.

The locked fresh-engine fixture exports after two tokens, rejects corrupt,
stale-identity, and truncated inputs without mutating the destination, imports
the valid 433.569 MiB file in 0.841 seconds, and matches uninterrupted
execution bit-for-bit across all four state hashes and three more generated
tokens. Export took 1.043 seconds on the qualified host. CRC64 protects
against accidental corruption; the file is not cryptographically
authenticated.

## Decode schedule

`k3_engine_forward_token()` is the regression path:

1. read one embedding row;
2. execute dense layer 0;
3. execute routed layers 1–92 in order;
4. update KDA/MLA/AttnRes state as each layer completes;
5. apply the output head and return greedy ID/value;
6. preserve recurrent and expert-cache state for the next token.

This token-major schedule is intentionally simple and exact. Current
post-TTFT performance is about 0.511 token/s on the tested Q8/32 host.

## Layer-major prefill

Calling the decode API once per prompt token would reread routed weights for
each token. The range path instead holds all hidden rows for a chunk and
executes one layer across the range before advancing:

1. batch the attention and router projections;
2. invert top-16 routes into expert-to-token lists;
3. visit only the unique selected experts in physical `(shard, offset)` order;
4. run one expert for all rows that selected it;
5. complete the shared/routed MoE tail;
6. advance to the next layer.

Every chunk performs 92 routed-layer sweeps. The payload-free plan validates
all 82,432 expert spans and treats 1,446,793,422,960 physical bytes as the
full-store ceiling because routes do not exist until execution. Runtime
compacts each layer's sorted layout to the actual route union, submits one
read per unique expert, and checks that dynamic request/byte ledger exactly.

The portable `k3_prefill_route_index` core was screened as an engine
replacement for the nested scan. It preserved exact token/value and I/O
ledgers, and reduced measured route-index host time from roughly 2.08 s to
0.056 s at 8,192 positions. That removes only 2.0245 s from a 1,365.3795 s
baseline (**0.148% direct ceiling**), far below the 2% gate. The apparent
2.096% wall regression sat inside 2.90--3.88% run spreads, so it is not
attributed to the index. The integration was reverted because its direct
component cannot justify promotion; the core remains non-production
infrastructure.

The routed-prefill harness is opt-in. When enabled, engine creation opens one
private transactional CSV and creates timing-enabled HIP events. Each complete
layer records host phases, default/expert/shared stream spans, accumulated MZG2
decoder time, time-weighted QD2 occupancy, and deterministic command counts.
When disabled, no diagnostic file or HIP event exists and the streaming
schedule executes the same production branches as before.

Its first 512-position qualification preserved token/value, all 35,501 reads,
541,449,752,576 physical bytes, route-union counts, and every command ledger.
The four post-warm-up expert-pipeline arms spread 4.766%, making a 5% gate
unresolvable under the required 14.298% floor; candidate ABBA stopped before
execution. QD2 depth-zero time averaged 0.204 s, only 0.216% of the 94.671 s
routed-stream mean. The event-scheduler lane is parked pending a materially
larger mechanism or lower-variance fixture.

The locked 512-token fixture selected 234–611 experts per layer, averaging
385.9, and reduced traffic to 623,090,706,040 bytes across 35,501 reads.
At filled 8K the union became denser but still averaged only 625.3 experts per
layer (262–876), reducing traffic to 1,009,747,090,312 bytes across 57,531
reads. Its locked token/value stayed exact and wall time improved from
1,054.547 to 1,008.104 seconds.

At filled 16K the selected union averaged 656.2 experts per layer (276–889),
reading 1,059,505,184,536 bytes across 60,366 requests. Wall time was
2,066.332 seconds / 7.929 token/s with locked token `6244` and value `26.875`.
Compared with selected 8K, routed expert-pipeline time scales 1.996x, but the
attention phase scales 2.206x; this is the first measured filled-context
pressure.

At filled 32K, the selected union averages 678.2 experts per layer (286–893),
reading 1,095,169,542,448 bytes across 62,398 requests. Wall time is 4,391.059
seconds / 7.462 token/s with locked token `40493` and value `28.25`. Compared
with 16K, the expert pipeline scales 1.917x and KDA 1.959x, while MLA scales
2.804x and raises total attention to 57.3% of wall time. The complete evidence
and host ledger are in
[filled-context qualification](qualification-filled-context.md).

The separate 15,993-token natural-text retrieval gate routes much more
densely: 721–895 experts per layer (868.0 mean) and 1,401,616,232,728 selected
bytes, or 96.9% of the full-store ceiling. It still sustains 7.924 token/s and
retrieves three exact values from early, middle, and late prompt positions.
At 31,999 tokens, the same semantic gate routes 740–895 experts per layer
(877.9 mean) and 1,417,570,415,176 selected bytes, or 98.0% of the ceiling. It
still retrieves all three exact values at 7.488 token/s. This density makes
further selected-I/O savings a secondary long-context lever relative to MLA,
expert execution, and the MoE tail.

Model-free MoE-tail profiling shows that the Q8 routed-up projection accounts
for 96–97% of its isolated weighted-sum/norm/projection/add sequence. At 32K,
tile 16 takes 919.462 ms/layer; weighted reduction, norm, and two adds total
36.730 ms. A dequantized hipBLAS prototype is much faster in isolation but
changes BF16 outputs and the end-to-end route/value contract, while providing
no selected-512 wall-time gain. It remains rejected; deterministic MLA range
work has the larger measured long-context opportunity.

The matched crossover fixture warms the decode cache, then executes the same
fixed token prefix sequentially and as one selected range on a single
resident engine. Every point must match the greedy token, float bits, token
position, and KDA, convolution, MLA, and AttnRes state hashes. Range prefill
was a tie at 3 tokens, 1.025x faster at 4, 1.006x at 6, 1.119x at 8, 1.222x
at 12, 1.285x at 16, 1.577x at 24, 1.794x at 32, and 2.108x at 42. The
production switch begins at 8 to keep the marginal 3--6-token region on the
simpler schedule.

Replacement prompts that exceed one cache-backed workspace loan use bounded
range calls. A payload-free binary search chooses the largest feasible first
chunk before causal reset; later chunks are replanned at the current engine
position. Successful calls advance the one live causal state and destroy their
transient workspace. They do not create prefix-cache entries. The chat layer
publishes retained prompt tokens only after every chunk succeeds, so a partial
failure leaves the session unhealthy rather than advertising mismatched token
and engine state. Chunk progress is reported against the whole prompt, range
counters and durations are aggregated, and the logged workspace loan is the
largest reported warm-cache loan. A position-zero cold loan uses the empty
cache but is not included in `warm_cache_workspace_bytes`.

Cold prefill may borrow a precisely planned slice of the empty 48.111 GiB
decode cache for workspace. Warm prefill normally retains that useful cache
and allocates only the incremental delta workspace. If the guarded separate
allocation cannot fit, the runtime may instead borrow a slot-aligned cache
tail, invalidate only mappings backed by those physical slots, and preserve
the rest of the warm LRU. Releasing the whole cache would repeat a known
failure from earlier GLM experiments.

A mismatched request that would replace live causal state first derives the
position-zero range plan with a cache-backed lease. Cold-cache benchmark resets
must fit an exact cold loan; normal replacements use a slot-aligned warm tail.
Planning is payload-free and non-mutating. If the monolithic plan fails, the
server admits a bounded first chunk before reset rather than discarding the
conversation speculatively. Only the absence of any two-token feasible chunk
remains a pre-reset workspace rejection.

Range prefill streams routed weights directly and does not admit decode-cache
entries. The workspace loan is therefore exclusive until range execution
destroys it; token decode resumes only afterward and can safely repopulate the
invalidated slots. Live qualification borrowed 254 of 2,760 physical slots
(4.151 GiB), leaving a 40.953 GiB non-overlapping cache region mapped.

## Projection backends

The release/default backend uses custom tile-16 Q8 and MXFP4 kernels. It is the
exact sequential/range oracle path.

The opt-in range-only KDA experiment dequantizes one Q8 matrix at a time into a
reused 168 MiB BF16 buffer and dispatches hipBLAS GEMM. The clean-checkout 8K
pair at `c041205` reduced the complete KDA phase from 334.754 to 62.474 seconds
(19.676 seconds in projection/output), raising full prefill from 7.768 to
10.448 token/s and reducing wall time from 1,054.547 to 784.092 seconds.

The changed reduction order shifts selected values and causal hashes, so it is
not the default. Promotion does not require impossible bit-identical hashes
across different reduction orders. It requires paired same-schedule replay
distributions as a numerical control envelope plus sequence-level natural-text
and task-quality tests showing no schedule-specific decline.

The first bounded quality gate used a deterministic 472-token, nine-record
retrieval prompt. The default and two diagnostic runs returned the identical
`saffron|7319|Nivens` answer with natural stops. The diagnostic runs repeated
their route/read ledger exactly and reduced prompt time from 186.045 seconds
to 173.046 and 172.318 seconds. A live OpenAI Python SDK loop also completed a
required weather call and a causally reused tool-result answer through the
diagnostic backend. This is sufficient to expose the backend for explicit
qualification, not to promote it: broader prompts, task suites, and review are
still required. hipBLASLt exposes block-scale types but returned no usable
native MXFP4/BF16 algorithms on the tested `gfx1151` stack.

## Correctness model

Performance work is gated at several levels:

Exact hashes are algorithm-scoped, not claims that mathematically equivalent
floating-point reduction schedules must be bit-identical. In particular, the
accepted group-vectorized MXFP4 GEMV from native commit `f68aa08` uses a
different FP32 addition order than its scalar parent. The optimized kernel has
its own locked full-layer hashes; promotion of another schedule requires a
same-schedule oracle, a bounded comparison to the reference path, and the
end-to-end token/value gates below.

The model-free MXFP4 envelope retains an exact test-only copy of that
historical 256-thread scalar schedule and compares it with the production
128-thread group schedule at both real expert input widths. Across 256 varied
inputs per shape (262,144 outputs total), seven outputs crossed a BF16 rounding
boundary and every difference was exactly one ULP. Maximum cross-schedule
absolute/relative differences were 1.0 / 0.65%; both schedules remained
within 0.39% of an FP64 accumulation reference. The gate requires at least one
boundary crossing, at most one BF16 ULP between schedules, and a 1% relative
or 0.25 absolute reference envelope.

- CPU or device reference oracles for MXFP4, Q8, SiTU, router, KDA, MLA, and
  AttnRes primitives;
- a multi-input scalar/vector MXFP4 envelope at both model expert widths;
- real-weight layer and component hashes;
- exact two-token comparison of sequential and range execution;
- hashes of KDA recurrent state, KDA convolution state, MLA cache, and
  AttnRes state;
- fresh-engine checkpoint import plus exact multi-token continuation;
- corrupt, stale-model, and truncated checkpoint rejection before mutation;
- exact official tokenizer oracles across ASCII and multilingual text;
- exact non-thinking, thinking, multi-turn, and marker-injection XTML fixtures;
- real JSON and SSE Chat Completions at 32K with exact locked output;
- real 16K and 32K residency allocation and reset checks;
- locked 512- and 8K output token/value fixtures;
- memory/I/O plans that are derived without allocating payloads;
- explicit diagnostic labeling for numerically different backends.

No faster path should become the release default while logits, attention
state, routing, or generated output drift is unexplained.
