# Changelog

All notable Moonshine changes will be recorded here. The project follows
Semantic Versioning once its first research-preview tag is published.

## [Unreleased]

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
- K3 preserved thinking with `low`/`high`/`max` effort, native reasoning
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
- A paired live structured continuation whose 117-token prefill fell from
  211.473 to 104.469 seconds while retaining all 237 prior causal tokens.
- A filled-8K exact-output gate at 1,008.104 seconds / 8.126 tok/s, with 30.2%
  fewer reads than the full-store ceiling and no dense-routing regression.

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
