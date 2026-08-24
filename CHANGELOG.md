# Changelog

All notable Moonshine changes will be recorded here. The project follows
Semantic Versioning once its first research-preview tag is published.

## [Unreleased]

### Added

- Timestamped production lifecycle logging for request admission, exact prefix
  reuse or mismatch, throttled prefill progress, reasoning-to-response/tool
  decode transitions, 64-token decode heartbeats, completion, failures, and
  client state. Interactive color is TTY-only and model/request content and
  credentials are excluded.
- A transport-facing coarse chat lifecycle callback used by the server without
  changing inference kernels, arithmetic, routing, I/O, cache policy, or
  scheduling.
- A busy-safe one-slot HTTP control plane. A dedicated acceptor serves health
  and authenticated model discovery during inference, reports slot
  availability, rejects competing completions with HTTP 503/`Retry-After`, and
  bounds request and response socket I/O to 30 seconds.
- Transport-aware cancellation checkpoints after complete prefill
  tokens/layers, generated tokens, and forced-trailer tokens. Peer loss and
  termination signals stop before the next model unit, drain current work,
  reset causal state without clearing immutable expert-cache mappings, and
  emit a typed `request.cancelled` lifecycle record.
- Ten-second decode SSE comments for quiet reasoning, tool, and structured
  output regions, independent of the existing prefill progress comments.
- A production-scale MXFP4/BF16 expert GEMM shape screen spanning selected-row
  counts through 1,170, with numerical envelopes and tile/backend timing.
- Opt-in decode cache snapshots, per-layer I/O/timing ledgers, content-derived
  expert-route traces, and non-cryptographic causal-state comparison
  fingerprints for paired qualification. Private CSV creation, abort rollback,
  and explicit baseline comparison keep the diagnostics outside normal
  production logging.
- Opt-in routed-prefill instrumentation through `--prefill-diagnostics` and
  `MOONSHINE_PREFILL_DIAGNOSTICS`. Private transactional CSVs record one row
  per routed layer with host phases, HIP-event stream/decoder spans,
  time-weighted QD2 occupancy, exact I/O ledgers, and deterministic launch,
  copy, clear, synchronization, and event-wait counts. The disabled path
  creates no events and reads no clocks beyond existing production timers.
- A deterministic prefill-screen analyzer that first enforces one discarded
  warm-up and a four-arm noise floor, stopping before ABBA when the declared
  gate is smaller than three times the spread. A resolvable screen then
  requires ordered interleaved ABBA, route/I/O identity, the mechanism's own
  subphase gate, aggregate phase no-regression, and private atomic JSON output.
  The first 512-position capture was exact but returned `UNRESOLVABLE`: 4.766%
  expert-pipeline spread implies a 14.298% floor for the former 5% P0b gate;
  QD2 depth-zero time was only 0.216% of routed-stream wall.
- An additional explicit router-logits tap for decode qualification. It writes
  private raw float32 logits only inside an active diagnostics transaction,
  participates in rollback, and is labeled more content-sensitive than routes.
- A model-free batch-LRU replay gate that validates complete single-capture
  route structure, strict ledger semantics and totals, source-capacity hit
  masks, provenance-safe expansion, and cache/ledger/route identity before
  reporting fixed-capacity counterfactuals.
- A deterministic standard-library offline cache analyzer with uniform and
  per-layer marginal curves, exact fixed-memory allocation, experimental
  frequency retention, second-touch/2Q/windowed-TinyLFU admission screens,
  explicitly nonpromotable full-trace oracles, and a prefix-trained cold-suffix
  bound. Strict ledgers accept both aligned raw and compressed physical-byte
  accounting. Private reports include stable input hashes and never embed
  absolute host paths or raw routes.
- Planner-driven chunked replacement prefill when a monolithic position-zero
  range cannot fit one cache-backed workspace loan. Each bounded range releases
  transient workspace before the next, reports whole-prompt progress, aggregates
  range telemetry, and publishes retained tokens only after complete success.
- An opt-in MZG1 routed-expert store that preserves the official SafeTensors,
  canonicalizes only E2M1 negative zero, reads 4 KiB-aligned per-expert blocks
  through QD2 io_uring, and decodes twelve two-stripe Zstd frames with six
  persistent workers into mapped staging before ordinary device-cache
  admission. The resumable transcoder verifies blocks immediately and again
  from storage before publication; the native reader fails closed on
  incomplete stores, source-manifest mismatches, or frame corruption. Full
  deterministic qualification passes, but end-to-end decode is slower. Its
  on-host research store was removed after explicit approval; the reader and
  transcoder remain for reproducibility.
- A qualified full MZG2 store format and gfx1151 wave32 decoder. Static rANS
  uses independently bounded 16 KiB tiles, per-tile checksums, mapped
  compressed input, and direct cache-slot output without CPU decode or an
  admission copy. The resumable 92-sidecar workflow immediately and
  independently verifies all 82,432 experts, records sidecar SHA-256 values,
  and atomically publishes 1,171.084 GiB at 13.0674105% reduction. Engine
  hello improves about 10.6%, selected prefill 11.6%, and live 128K/30
  11.7--12.8%, with exact outputs. The qualified K3 deployment now selects it
  through `MOONSHINE_MZG2_STORE`; raw SafeTensors remain the rollback path.
- An opt-in durable exact-prefix checkpoint path for displaced or restarted
  sessions. Model-backed response bytes, cached-token accounting, all four
  causal-state digests, manifest reload, and server publication/restart
  recovery passed. Prefix restoration improved **65.77×** (63.008 s inferred
  evaluation versus 0.958 s import), clearing the 5× gate; total prompt wall
  improved **2.87×** (95.174 → 33.124 s, 65.20% lower). Deployment remains
  explicit through a private checkpoint root.
- The first production checkpoint canary passed byte-exact live, displaced,
  and restart recovery but failed client availability. A valid completion POST
  sent 32 ms after response delivery received HTTP 503 while synchronous
  checkpoint export retained the one request slot; that export finished in
  2.097 seconds. Production was rolled back to the checkpoint-disabled binary.
  Activation now requires publication before the client-visible terminal
  response plus an immediate-next-completion no-503 gate.
- A model-free exact-anchor recovery analyzer. The recorded 60% universal edit
  gate is a **NO-GO even under ideal anchor placement**: historical deep Hermes
  compaction preserved only 8/29,630 tokens (0.027%), and the observability
  edit preserved at most 86/158 (54.43%). Reasoning omission can recover
  3,849/3,905 (98.57%) and exact structured displacement 237/354 (66.95%).
  Activation is therefore narrowed to durable exact-prefix
  displacement/restart recovery, not semantic recovery across deep rewrites.
- A Prime-Agent 0.8.0 source and live-session structure review. Its
  `buildSessionContext()` emits a mutable `compactionSummary` as the first
  conversation message, followed by retained recent messages. Prime
  compaction therefore has no guaranteed exact conversation anchor until a
  production-renderer pre/post-compaction gate proves the stable head or a
  client extension preserves one.
- A model-free one-pass selected-prefill route index. It validates token-major
  top-k routes and per-token uniqueness, produces stable expert-local
  token/output slices in `O(routes + experts)`, reuses allocations, preserves
  the prior index after failed rebuilds, and performs allocation-free slice
  lookup. Its engine integration preserved exact outputs and I/O, but removed
  only 2.0245 seconds of a 1,365.3795-second 8,192-token baseline
  (**0.148% direct ceiling**), far below the 2% promotion gate. The apparent
  2.096% wall regression sat inside 2.90--3.88% run spreads; the integration
  was reverted without attributing that drift to the index.
- A GPU-free static-Q8 compression-screen foundation. The production Q8
  candidate classifier now lives in portable code shared by the engine and
  offline tool; the CPU Q8/128 reference locks BF16 conversion,
  round-to-nearest-even, scale, clamp, and entropy behavior. The streamed
  `screen_static_q8` command validates the pinned 1,135-matrix byte ledger and
  measures independent 16/32/64 KiB value/scale Zstd payloads without retaining
  the 51.370 GiB Q8 tier. The full model scan is intentionally not run beside a
  live service and still requires a later GPU-quantizer identity gate.

### Fixed

- Restrict router-logits capture to an active decode-diagnostics transaction so
  prompt/prefill data remains outside the capture and rollback boundary.
- Fail closed when an explicitly selected MZG1 store is absent, and validate
  every MZG2 probability/LUT table before GPU decode.
- Require explicit arguments for the layer-local MZG2 transcoder so a no-argument
  invocation cannot create a large derived artifact in the source tree.
- Reject the staged selected-prefill route-index engine integration after exact
  512-token and 8,192-token screens. The candidate removed nearly all measured
  route-index host time but that component was only 0.148% of long-prefill wall,
  so it cannot meet the 2% promotion gate; the production nested scan was
  restored. Evidence:
  `moonshine-experiments/prefill-route-index-qualification-20260824/`.
- Prevent timed-out monitoring/retry connections from saturating the listener
  backlog during a long inference request. Health/model discovery now remains
  responsive, abandoned streaming and JSON work releases the inference slot,
  and `SIGINT`/`SIGTERM` drains active inference before engine destruction.

## [0.2.0-research-preview] - 2026-08-01

Adds an agentic serving surface — function tools, preserved thinking, and
validated structured responses — on top of selected-expert prefill, and
qualifies filled context and natural-text retrieval through 32K. Scope is
unchanged from 0.1.0: a research preview for the pinned Kimi K3 checkpoint on
a qualified 128 GB `gfx1151` host, not a general runtime, not portable across
AMD architectures, and not source-precision equivalent.

### Added

- Configurable-context full-residency and locked hello fixtures through the
  `MOONSHINE_CONTEXT` make variable.
- A payload-free 128K prefill-plan regression and real 128K configured-
  capacity qualification on the accepted Q8/32 host.
- Native K3 XTML function-tool declarations, calls, call-ID-resolved results,
  typed arguments, and raw JSON argument blocks.
- OpenAI Chat Completions function tools in JSON and SSE, including parallel
  call output, `auto`/`required`/`none`, forced named-function selection, and
  complete tool-result history validation.
- A real two-turn `get_weather` agent-loop qualification at 8K context.
- Exact agentic causal-prefix recovery by restoring historical hidden
  tool-choice directives at their original message boundaries.
- A real SSE function-loop qualification that reused all 196 retained tokens
  and evaluated only the 42-token tool-result suffix.
- K3 preserved thinking with `low`/`medium`/`high`/`max` effort, native reasoning
  parsing, JSON `reasoning_content`, and live SSE reasoning deltas.
- A two-turn live reasoning qualification whose continuation reused all 127
  prior prompt/generated tokens and evaluated only a 25-token suffix.
- Native `response_format=json_object` XTML, post-generation object
  validation, and deferred-until-valid structured SSE content.
- A live structured-output qualification that returned
  `{"greeting":"hello"}` without exposing unvalidated response bytes.
- Bounded native `response_format=json_schema` rendering and recursive
  post-generation validation for typed objects, arrays, and scalar values.
- An 8K live schema qualification that returned the validated object
  `{"greeting":"hello","count":1}` while withholding response content until
  the complete value passed.
- Enforced `parallel_tool_calls=false` through a hidden single-call directive,
  post-parse call-count validation, and exact historical-directive recovery.
- A three-turn serial tool qualification that called weather and time one at
  a time, reused 496 then 690 causal tokens, and produced a final combined
  answer.
- A pinned official OpenAI Python SDK 2.52.0 SSE replay fixture plus a live
  8K two-request tool loop against Moonshine, including causal reuse, the
  tool-result answer, terminal usage, and indexed streaming calls.
- Exact structured-session prefix recovery by restoring historical JSON-object
  or owned canonical JSON Schema directives at their causal boundaries.
- A two-turn live schema qualification that reused all 237 retained tokens,
  evaluated only the 117-token suffix, and returned validated
  `{"greeting":"goodbye"}`.
- Physical-order selected-expert range prefill with exact dynamic request/byte
  ledgers and per-layer route-union telemetry.
- Exact selected-prefill gates reducing the two-token range from 203.638 to
  7.961 seconds and the locked 512-token range from 239.325 to 123.519 seconds.
- A warm-cache crossover fixture comparing sequential and selected range
  execution on one resident engine with exact output and causal-state gates.
- A model-shape, 262,144-output MXFP4 numerical envelope comparing the exact
  historical scalar reduction with the production group-vectorized schedule
  across inputs that cross BF16 rounding boundaries.
- A single self-hosted reduction-change qualification target covering
  component envelopes, real experts/MoE, complete routed-layer hashes,
  tokenizer/XTML, and the locked chat fixture.
- Explicit configured-context support in the exact scale fixture for graduated
  filled 16K and 32K qualification.
- An exact filled-16K selected-prefill gate at 2,066.332 seconds / 7.929
  token/s, including full phase, selected-I/O, memory, swap, and SSD evidence.
- A deterministic 15,993-token natural-text retrieval gate with three fixed
  needles spanning the prompt and an exact decoded-response oracle.
- A paired live structured continuation whose 117-token prefill fell from
  211.473 to 104.469 seconds while retaining all 237 prior causal tokens.
- A filled-8K exact-output gate at 1,008.104 seconds / 8.126 tok/s, with 30.2%
  fewer reads than the full-store ceiling and no dense-routing regression.
- An exact filled-32K selected-prefill gate at 4,391.059 seconds / 7.462
  token/s, plus a 31,999-token natural-text gate that retrieved fixed early,
  middle, and late values at 7.488 token/s.
- An opt-in range-only KDA dequantize-plus-hipBLAS backend exposed through the
  retrieval and OpenAI qualification paths. It passed repeated bounded
  natural-text and official-SDK tool-loop gates but remains diagnostic.
- A portable model-free `test_k3_prefix_reuse` gate covering session
  prefix-reuse admission, including edited-same-length history and count-
  arithmetic boundaries. It runs in CPU-only CI without ROCm or model weights.
- A non-mutating position-zero workspace preflight before a divergent range
  request resets live causal state. Requests that cannot secure a cache-backed
  cold or warm lease now fail while preserving the prior conversation.
- A configurable server output ceiling through `--max-output-tokens`, with an
  8K default, a bounded 64K maximum, context-aware clamping, and advertised
  context/output limits in health and model discovery.
- A persistent-server 128K qualification with 30 expert slots per layer,
  covering two independent requests while retaining the CMA-aware memory
  guard; the 32-slot 128K configured-capacity result remains a single-request
  fixture.
- Standard cached-prefix accounting through
  `usage.prompt_tokens_details.cached_tokens` in JSON and terminal SSE usage.
- Prefix-miss diagnostics reporting the retained, common, and candidate token
  counts without changing the exact reuse-admission predicate.
- A guarded warm-prefill fallback that lends a slot-aligned decode-cache tail,
  invalidates only overlapping expert mappings, and reports the loan in range
  telemetry.

### Changed

- Extend the documented qualified configured-context capacity from 32K to
  128K. Filled-128K latency and long-context quality remain unqualified.
- Re-lock the synthetic routed-layer hashes and first-token BF16 score for the
  accepted group-vectorized MXFP4 reduction introduced by native commit
  `f68aa08`. Its scalar-parent hashes survived because the optimization
  qualification covered component and real-chat gates but not the synthetic
  full-layer fixture.
- Preserve the exact mismatch/full-prefill gate while allowing session-local
  tool-result and structured-response turns to continue the actual hidden-
  directive causal history.
- Treat payload-free routed I/O totals as validated full-store ceilings;
  runtime now reads and exactly accounts only each layer's routed union.
- Lower the default sequential-prefill limit from 92 to 7 after repeated
  matched runs showed a stable 1.12x range-path lead at 8 tokens and 2.11x at
  42 tokens; retain the marginal 3--6-token region on the sequential path.
- Reject required-tool generations that exhaust their token budget without
  producing a call instead of returning an unsatisfied length-stopped turn.
- Make the OpenAI request parser enforce the active server output ceiling
  instead of a transport-global 8K constant.
- Select the server's one retained causal prefix automatically by exact token
  content. Standard OpenAI clients no longer need `X-Moonshine-Session`, and
  hidden tool/structured-output directives are retained for headerless turns.

### Fixed

- Non-thinking natural-stop completions now retain their ordinary response
  text when the generated XTML suffix is parsed after streaming.
- Treat explicit `null` `max_completion_tokens`/`max_tokens` values as
  unspecified, including fallback from the preferred field to the legacy one.
- Prevent a large warm prefix miss from failing only because its separate
  prefill workspace would violate the CMA-plus-host reserve; the cache loan
  retains the guard and the non-overlapping warm expert mappings.

## [0.1.0-research-preview] - 2026-07-30

First public research preview. Reproduces the native Kimi K3 SafeTensors/ROCm
engine and its correctness and performance fixtures on a qualified 128 GB
`gfx1151` host. It is not a general inference runtime, a finished chat product,
portable across AMD architectures, or source-precision equivalent.

### Added

- Q8/BF16 static residency for the official 96-shard Kimi K3 checkpoint.
- NVMe-streamed checkpoint-native MXFP4 routed experts with an online
  per-layer cache.
- Native KDA, gated MLA, AttnRes, MoE, tokenizer, and text-only XTML paths.
- Token-major decode and layer-major prefill.
- Versioned, checksummed causal-state persistence.
- Interactive `moonshine-chat`.
- One-slot OpenAI-compatible `moonshine-server` with JSON and SSE Chat
  Completions.
- Dynamic context allocation qualified at 8K, 16K, and 32K.
- Opt-in exact append-prefix causal-state reuse through `X-Moonshine-Session`.
- SSE token/layer prefill-progress keepalives and long-timeout guidance.
- Portable CPU-only CI and publication/community documentation.

### Changed

- Preserve immutable routed-expert cache mappings across stateless requests.
- Move the default sequential limit from 128 to 92, selecting layer-major
  prefill from the measured 93-token crossover onward.
- Replace the impossible cross-schedule bit-exact promotion gate with paired
  same-schedule numerical envelopes plus sequence/task quality gates.

### Removed

- `docs/release-plan.md`, whose pre-publication checklist is complete. The
  correctness model lives in `docs/architecture.md`, release qualification in
  `RELEASING.md`, and repository provenance in `docs/provenance.md`.
