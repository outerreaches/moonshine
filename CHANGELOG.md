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

### Changed

- Extend the documented qualified configured-context capacity from 32K to
  128K. Filled-128K latency and long-context quality remain unqualified.
- Re-lock the synthetic routed-layer hashes and first-token BF16 score for the
  accepted group-vectorized MXFP4 reduction introduced by native commit
  `f68aa08`. Its scalar-parent hashes survived because the optimization
  qualification covered component and real-chat gates but not the synthetic
  full-layer fixture.
- Preserve the exact mismatch/full-prefill gate while allowing session-local
  tool-result turns to continue the actual hidden-directive causal history.

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
