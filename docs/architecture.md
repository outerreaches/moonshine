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

The current renderer covers named or unnamed system, user, and assistant
messages; thinking/non-thinking response channels; optional low/high/max
thinking effort; and the open assistant generation prompt. Tool declarations,
tool calls/results, images, and structured response formats remain the next
XTML extension for the OpenAI-compatible service.

## Stateful chat session

`k3_chat_session` composes the tokenizer and engine behind one transport-free
turn API. The first call renders the optional system message plus user turn;
later calls submit only the new user-message and assistant-opener XTML delta
against retained causal state.

Prompt scheduling is length-aware. The measured crossover is 92.4 tokens:
the default uses token-major execution through 92 tokens and layer-major
prefill from 93 tokens onward.
Every model-produced token, including the final end-of-message marker, is fed
back into the engine so the position is a complete-turn boundary. A length
stop forces only the missing suffix of the official response/message trailer.

The session emits response token-piece bytes through an optional callback,
also returns the complete response, and reports prompt strategy, TTFT-equivalent
prompt time, decode rate, cache deltas, physical prefill I/O, finish reason,
forced trailer count, and committed position. Export/import delegates to the
exact semantic checkpoint API.

`moonshine-chat` is a thin one-shot or interactive shell over this shared
layer. `moonshine-server` uses the same layer for Chat Completions. Stateless
requests zero-reset causal storage while retaining immutable expert-cache
mappings. Requests carrying the same `X-Moonshine-Session` identifier may
reuse retained in-process state only when the newly rendered full history is
an exact token-prefix extension. Any mismatch resets causal state and
prefills the full history.

## OpenAI-compatible transport

The initial server is a deliberately bounded HTTP/1.1 implementation:

- one persistent engine and one blocking request slot;
- loopback binding by default, with a bearer key required for non-loopback;
- `GET /health`, `GET /v1/models`, and
  `POST /v1/chat/completions`;
- strict native JSON parsing with an 8 MiB default body limit;
- ordinary JSON completion responses and incremental SSE chunks;
- SSE comment keepalives with token/layer prefill progress;
- opt-in single-session append-prefix reuse through
  `X-Moonshine-Session`;
- UTF-8 boundary buffering so tokenizer byte fragments never corrupt a stream;
- standard OpenAI error envelopes and usage accounting;
- fixed greedy execution, with unsupported tools and structured response
  formats rejected before inference.

There is no scheduler or hidden concurrency. A disconnected streaming client
does not interrupt model execution mid-turn; the engine completes its semantic
trailer so its internal state is never left at a partial token boundary.

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
directory without loading payloads. Every read is then explicit and
role-aware.

The accepted engine divides memory as follows:

| tier | accepted Q8/32 allocation |
|---|---:|
| Q8/BF16 resident static weights | 55.270 GiB |
| routed-expert cache, 32 slots/layer | 48.111 GiB |
| recurrent/cache/runtime state, 8K | 0.920 GiB |
| recurrent/cache/runtime state, 16K | 1.135 GiB |
| recurrent/cache/runtime state, 32K | 1.566 GiB |
| 16 mapped expert staging slots | 0.262 GiB |
| total before allocator/driver overhead | about 104.6 GiB |

Eligible BF16 static projections are quantized one tensor at a time to the
engine's Q8-128 format. Source buffers are released immediately, bounding peak
startup memory. MLA matrices that must remain BF16 are retained as BF16.
Embeddings are read by row; the output head is resident.

Before allocation, the engine calculates load-time and runtime peaks and checks
`MemAvailable` plus the platform's usable CMA reserve. Source-precision BF16
requires an additional guard and is intentionally not accepted on the tested
128 GB host.

## Routed-expert I/O

Each routed expert occupies one contiguous physical SafeTensors span containing
six MXFP4 data/scale tensors. The engine:

1. opens both buffered and `O_DIRECT` shard descriptors;
2. registers caller-owned aligned buffers with raw Linux `io_uring`;
3. keeps two direct reads in flight because the tested 990 PRO saturates at
   QD2;
4. reads into HIP-mapped host memory visible to the GPU;
5. launches the expert without a post-read upload;
6. refills the queue before consuming the completion;
7. accumulates selected experts in completion order.

The cache is a uniform 32-slot-per-layer online LRU. Admission is two-phase:
planning preserves every hit needed by the current batch, and commit may
replace slots only after GPU readers finish. Cache metadata is independent of
HIP allocation ownership.

Two HIP streams separate routed-expert work from shared-expert work. Decode
launches mapped cache hits first while selected misses are outstanding.

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
16K, and 32K. The model config names a 1,048,576-position architectural
maximum, but memory preflight and latency make that a limit, not a qualified
operating point.

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
3. visit all experts in physical `(shard, offset)` order;
4. run one expert for all rows that selected it;
5. complete the shared/routed MoE tail;
6. advance to the next layer.

A 512- or 8,192-token chunk therefore performs the same 92 routed-layer sweeps
and 82,432 expert reads, totaling 1,446,793,422,960 logical bytes. The fixed
sweep is amortized across the range.

Cold prefill may borrow a precisely planned slice of the empty 48.111 GiB
decode cache for workspace. Warm continuation must retain the useful cache and
allocate only the incremental delta workspace; releasing the whole cache would
repeat a known failure from earlier GLM experiments.

## Projection backends

The release/default backend uses custom tile-16 Q8 and MXFP4 kernels. It is the
exact sequential/range oracle path.

The opt-in range-only KDA experiment dequantizes one Q8 matrix at a time into a
reused 168 MiB BF16 buffer and dispatches hipBLAS GEMM. At 8K it reduces
combined KDA input/output projection time from 292.319 to 19.876 seconds and
raises full prefill from 7.746 to 10.447 token/s.

The changed reduction order shifts selected values and causal hashes, so it is
not the default. Promotion does not require impossible bit-identical hashes
across different reduction orders. It requires paired same-schedule replay
distributions as a numerical control envelope plus sequence-level natural-text
and task-quality tests showing no schedule-specific decline. hipBLASLt exposes
block-scale types but returned no usable native MXFP4/BF16 algorithms on the
tested `gfx1151` stack.

## Correctness model

Performance work is gated at several levels:

- CPU or device reference oracles for MXFP4, Q8, SiTU, router, KDA, MLA, and
  AttnRes primitives;
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
