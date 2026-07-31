# Changelog

All notable Moonshine changes will be recorded here. The project follows
Semantic Versioning once its first research-preview tag is published.

## [Unreleased]

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
