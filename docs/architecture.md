# K3.1 architecture

## Scope and qualification boundary

Moonshine is a purpose-built native execution core for the pinned
`moonshotai/Kimi-K3.1-Pro` SafeTensors layout on ROCm. It is not a GGUF runtime,
a framework integration, or a portable tensor library.

This document describes the implemented 0.3.0 architecture. Synthetic CPU/ROCm
tests pass, including DSA decode/prefill component equivalence. Full MZG2
conversion and verification passed for all 92 routed layers and 82,432 experts.
The verified bundle passed the real 8K Beelink generation/prefix-reuse canary,
live tool/structured-output and state-v2 gates, and natural-text retrieval
through 63,971 tokens. No context above 64K or production soak is qualified.
Historical K3 memory, latency, and quality results are not K3.1 architecture
qualification.

## Pinned model profile

The fail-closed loader requires:

- ID `moonshotai/Kimi-K3.1-Pro` at revision
  `6d0a0202956796543364d666ab19c6ad1a494ee1`;
- architecture `KimiK31ForConditionalGeneration` and model type `kimi_k31`;
- 93 layers: one dense layer and 92 routed-MoE layers;
- 69 KDA layers and 24 Gated DSA layers;
- DSA at every fourth layer and at layer 93, with matching `kda_layers`,
  `full_attn_layers`, and `dsa_layers` arrays;
- 896 routed experts per MoE layer, top-16 routing, and two shared experts;
- hidden width 7,168, latent MoE width 3,584, routed-expert hidden width 3,072;
- BF16/F32 static tensors and group-32 MXFP4 routed weights;
- 12-layer AttnRes blocks.

A missing, conflicting, or changed profile field is an error. Moonshine does not
fall back to the older K3 graph.

## Text-only tokenizer and XTML

The native tokenizer reads the checkpoint's `tiktoken.model` without Python or
Transformers. ICU implements the official Unicode pre-tokenization pattern and
a byte-pair rank table produces the 163,840-token ID space.

The XTML renderer preserves official structural segment boundaries. Only actual
structure resolves to control tokens; marker-looking user/system/assistant/tool
text remains ordinary text. Messages can carry names, reasoning, response
content, tool declarations/calls/results, and request-local response-format or
tool-choice directives.

The internal renderer retains official `low`/`medium`/`high`/`max` XTML effort
values. The public K3.1 Chat Completions parser deliberately narrows this to
`low`, `high`, and `max`, default `max`; public `medium` is rejected. Image and
audio content parts are rejected explicitly. Moonshine does not silently remove
non-text input.

## Execution schedules

### Decode

Decode is token-major. For each token, the engine reads its embedding, executes
the dense layer and all routed layers in order, updates KDA/DSA/AttnRes state,
and applies the resident output head. KDA recurrence and convolution, MLA latent
rows, DSA index keys, AttnRes state, and decoded routed experts persist between
tokens.

### Prefill

Prefill is layer-major over bounded token ranges. It batches attention and
router projections, inverts top-16 routes into expert-to-token lists, visits the
unique selected experts in physical storage order, and completes the MoE tail.
The same persistent DSA index-key allocation is used by decode and prefill;
prefill does not create a second index-cache representation.

Exact append-only reuse compares the complete rendered token history. Edited,
forked, shorter, or mismatched histories reset semantic state and prefill
independently. Immutable expert-cache mappings can survive that reset because
they contain model weights rather than conversation state.

## Gated DSA

The released DSA geometry is fixed and validated:

| field | required value |
|---|---:|
| index heads | 32 |
| index head width | 128 |
| maximum selected keys | 2,048 |
| index RoPE width | 0 |
| key normalization | LayerNorm |
| top-k frequency | every DSA layer |
| skip/top-k offset | 1 |

For each DSA layer, execution:

1. computes bias-free indexer query, key, and per-head weight projections;
2. applies key **LayerNorm**, with BF16 gamma/beta and F32 mean/variance, rather
   than RMSNorm;
3. stores the normalized 128-element key row in the persistent BF16 cache;
4. skips indexer scoring and top-k when every visible row is required; otherwise
   computes F32 indexer scores from BF16 queries/keys, including the released
   head weighting and scaling rules;
5. for sparse execution, selects at most `min(visible_keys, 2048)` positions,
   ordered by descending score with lower token IDs breaking ties;
6. uses dense MLA from the measured 768-row threshold; above the 2,048-row
   sparse window, gathers selected rows once and pads unused selection entries
   with `-1`;
7. projects MLA values and applies the released output gate before `o_proj`.

These execution choices preserve the DSA selection semantics while avoiding
unnecessary indexer/top-k and gather work. Named DSA failure steps identify the
failing stage. Large-window indexer scoring uses BF16 BLAS GEMM followed by
weighted-ReLU F32 reduction; the custom scorer remains below 768 rows. This path
measures 1.094 → 0.075 ms at 8K and passes the real 8K retrieval gate.

No index RoPE is applied because the pinned `index_rope_dim` is zero. Selection
handles causal boundaries, ties, negative-infinity padding, and short visible
ranges deterministically. Derived selected indices are transient execution data.

### Cache precision

The MLA row contains 576 BF16 elements: a 512-wide latent portion and the
64-wide pass-through key subspace. Each DSA index-key row adds 128 BF16
elements. Thus the two persistent semantic cache rows total 704 BF16 elements,
or 1,408 bytes, per token per DSA layer before other state/workspace.

The index-key cache is **BF16 only**. Official FP8 and MXFP4 index-cache formats
are neither implemented nor qualified. Capacity planning must include this BF16
cache and DSA workspace rather than reusing a historical K3 MLA-only ledger.

## Routed-expert storage

The official source has 96 SafeTensors shards. The loader parses their headers
and index into a sorted tensor directory; static/model-control tensors and routed
experts have separate residency paths. Eligible static projections use the
engine's Q8 representation, while tensors requiring source BF16 remain BF16.
Routed experts retain their checkpoint-native MXFP4 storage representation and
are decoded into persistent device-cache slots on demand.

Raw SafeTensors are the defined fallback. K3.1 MZG2 tooling is manifest-driven
and preserves DSA tensors at source precision. Full conversion and verification
passed for all 92 layers and 82,432 experts; the resulting bundle also completed
the real 8K Beelink startup/generation canary. Do not use old K3 bundle manifests,
tensor counts, source revisions, or memory ledgers as K3.1 evidence.

## Offline checkpoint and placement tools

`tools/k31_checkpoint.py` validates the pinned config, index, 96-shard coverage,
SafeTensors headers, complete tensor map, and routed-expert geometry. Metadata
mode reads JSON and headers; `--verify-sha256` additionally streams every input
file. The full pinned-source MZG2 workflow completed conversion and independent
verification for all 92 routed layers and 82,432 experts. Synthetic inspector
fixtures and that real conversion are separate evidence classes; neither alone
is a production qualification.

`tools/moonshine_placement.py` accepts an explicit sidecar profile and assigns
whole layer files with deterministic weighted largest-first placement. It
requires matching model/store source identities, respects per-drive reserves and
capacity, and emits canonical hashed manifests. It does not inspect the host or
move model data. Synthetic planner fixtures pass; no K3.1 multi-drive store is
qualified.

## Attention and causal state

KDA retains recurrent matrix state and causal convolution history per layer.
AttnRes retains block inputs and performs learned residual mixing at exact
12-layer boundaries. DSA layers retain both MLA latent/pass-through rows and
normalized index-key rows.

The engine tracks separate causal-state comparison digests for:

1. KDA recurrent state;
2. KDA convolution state;
3. MLA cache rows;
4. DSA index-key cache rows;
5. AttnRes state.

These are non-cryptographic comparison fingerprints, not authentication or a
substitute for exact state comparison.

Startup statistics report DSA index-cache and workspace bytes. Decode and
prefill telemetry include `dsa_indexer_calls`, `dsa_topk_calls`, and
`dsa_selected_rows`; server decode logging emits a `request.dsa` event when DSA
top-k work occurred.

## State format v2

State v2 retains the fixed 256-byte explicitly little-endian header and adds an
index-state segment and length field. It records format/endian markers,
configured context, token position, static precision, model-layout identity,
segment lengths, complete payload length, and CRC64-ECMA header/payload
checksums.

The payload contains:

- complete KDA recurrent and convolution allocations;
- only occupied MLA rows;
- only occupied DSA index-key rows;
- AttnRes state and token position.

The occupied MLA plus index-key state is therefore proportional to the committed
token position rather than serializing unused configured-capacity rows. Derived
MLA packed keys, transient top-k indices, and routed-expert cache/LRU contents
are excluded and reconstructed or warmed independently.

**V1 state files are rejected.** Renaming or editing the version field is not a
migration: v2 requires the additional semantic DSA state. Export streams through
a bounded host buffer and atomically renames a completed temporary file. Import
validates format, identity, exact file length, and payload CRC before device
mutation; a failure after mutation begins invalidates the destination causal
state until a valid import/reset.

Synthetic state-memory and format tests pass. Real-weight K3.1 state-v2 export,
fresh-process import, and exact continuation also passed, including 161 reused
tokens after restart. Displacement/restart and header, stale-model, payload,
corruption, and truncation rejection gates passed without weakening the exact
prefix admission rule.

## OpenAI-compatible transport

The server implements:

- `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions`;
- only model ID `moonshotai/Kimi-K3.1-Pro`;
- version identity `0.3.0-k3.1-research-preview`;
- one persistent engine and one inference slot, with independent health/model
  discovery while busy;
- JSON and SSE responses, keepalive comments, bounded request/response I/O, and
  safe-boundary cancellation;
- text-only messages and explicit non-text rejection;
- public `low`/`high`/`max` reasoning effort, default `max`;
- function tools, preserved reasoning history, and bounded post-validated
  `json_object`/`json_schema` output;
- exact append-only prefix admission and standard cached-token accounting;
- optional bounded durable exact-prefix checkpoints using state v2.

These interfaces have synthetic parser/codec coverage. Real K3.1 generation,
exact prefix reuse, tool calling and tool-result continuation, structured JSON,
checkpoint continuation, and medium/image/old-model rejection passed in the
documented Beelink canary. The official OpenAI Python SDK 2.52.0 fixture passed
separately. Neither result is third-party agent-client or production-soak
qualification.

## Correctness evidence and open gates

Verified for this cutover:

```sh
make -j4 test-cpu
make -j4 test
```

The CPU suite passed, the ROCm suite passed on Radeon 8060S `gfx1151`, and the
CPU DSA oracle passed ASan/UBSan. DSA GPU tests cover LayerNorm/scoring, stable
top-k boundaries/ties/negative-infinity padding, sparse attention, and
decode/prefill equivalence. These remain model-free synthetic results.

Real-weight evidence is separate: the full MZG2 conversion verified every routed
layer/expert; the 8K canary passed startup, generation, prefix reuse, live
agentic/API, and state-v2 recovery; and natural-text retrieval returned
`saffron|7319|Nivens` exactly at 15,993, 31,999, and 63,971 tokens. The measured
8K DSA sparse-to-dense comparison improved 4.930 → 0.126 ms per layer. Reproduce
that kernel screen with:

```sh
make bench-k31-dsa
```

The 64K natural-context run started in 93.976 seconds, prefilled in 14,547.439
seconds, and decoded 17 tokens in 37.485 seconds. Its KDA, DSA, expert-stream,
and MoE-tail phases took 2,585.803, 8,101.056, 3,463.059, and 248.327 seconds,
respectively. Exact reproduction commands and per-context rates are in
[Getting started](getting-started.md#12-reproduce-natural-context-and-dsa-evidence).

Still open:

- all context and memory claims above 64K;
- continuous long-output and broader performance qualification;
- third-party agent-client and production-soak qualification;

Historical `qualification-*.md` files document earlier K3 releases only. They
are preserved for provenance, not reused as evidence for any item above.
