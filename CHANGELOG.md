# Changelog

All notable Moonshine changes will be recorded here. The project follows
Semantic Versioning once its first research-preview tag is published.

## [Unreleased]

- Preserve immutable routed-expert cache mappings across stateless requests.
- Add opt-in exact append-prefix causal-state reuse through
  `X-Moonshine-Session`.
- Add SSE token/layer prefill-progress keepalives and timeout guidance.
- Move the default sequential limit from 128 to 92, selecting layer-major
  prefill from the measured 93-token crossover onward.
- Replace the impossible cross-schedule bit-exact promotion gate with paired
  same-schedule numerical envelopes plus sequence/task quality gates.

## [0.1.0-research-preview] - 2026-07-30

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
- Portable CPU-only CI and publication/community documentation.

The release remains pending until the final clean-checkout qualification and
GitHub publication gates are complete.
