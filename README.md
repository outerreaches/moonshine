# Moonshine

Moonshine is an independent, experimental native inference engine for AMD
ROCm. This checkout is the **0.3.0 Kimi K3.1 Pro research-preview cutover**.
It is not affiliated with or endorsed by Moonshot AI.

> **Support boundary:** synthetic CPU/ROCm gates pass. Full conversion and
> verification of the pinned K3.1 MZG2 bundle passed for all 92 routed layers
> and 82,432 experts, and the bundle passed real Beelink load, generation,
> prefix-reuse, API, and state-v2 canaries. Natural-text retrieval passes at
> 15,993, 31,999, and 63,971 tokens. No context above 64K and no production
> service has been qualified. Historical Kimi K3 results are not K3.1 evidence.

## Current model contract

Moonshine pins exactly one upstream model:

- model/API ID: `moonshotai/Kimi-K3.1-Pro`;
- revision: `6d0a0202956796543364d666ab19c6ad1a494ee1`;
- architecture: `KimiK31ForConditionalGeneration`;
- model type: `kimi_k31`;
- server version: `0.3.0-k3.1-research-preview`.

The configuration loader rejects other model identities, layer layouts, and
DSA geometries instead of attempting a best-effort load. The public Chat
Completions API is text-only: image and audio content are rejected explicitly.
It exposes reasoning efforts `low`, `high`, and `max`, defaulting to `max`.
`medium` remains accepted by the internal official-XTML renderer for format
compatibility, but is not a public K3.1 API value.

## Implemented subsystems

The current source includes:

- a fail-closed K3.1 configuration/profile loader for the pinned 96-shard
  checkpoint layout;
- token-major decode and layer-major prefill with 69 KDA layers and 24 Gated
  DSA layers;
- BF16 DSA indexer keys and sparse MLA state, F32 indexer scores, stable top-k
  selection, and a persistent index-key cache shared by decode and prefill;
- identity-window indexer/top-k skipping, dense MLA from the measured 768-row
  threshold, and one selected-row gather above the 2,048-key sparse window;
- state format v2, including occupied MLA and DSA index-key rows, with v1
  rejection;
- DSA startup/request counters and causal-state digests;
- native tokenization and text-only XTML chat/tool rendering;
- a one-slot OpenAI-compatible Chat Completions server with JSON and SSE;
- exact append-only causal-prefix reuse and opt-in durable checkpoints;
- an offline checkpoint inspector and deterministic multi-drive sidecar
  placement planner.

DSA decode/prefill integration and synthetic component correctness are
implemented. Official FP8 and MXFP4 index-key cache formats are **not**
implemented or qualified; the index-key cache is BF16. Full K3.1 MZG2 bundle
conversion and verification passed for every routed layer and expert, and the
verified bundle is a real-weight deployment source for the documented canaries.
BLAS-backed indexer scoring is implemented for large windows: BF16 GEMM plus
weighted-ReLU F32 reduction. At 8K the measured indexer path is 1.094 → 0.075 ms
(14.6×), with 2.235e-08 maximum score error. A real 8K retrieval gate passes.

## How K3.1 execution is structured

The pinned text model has 93 layers: one dense layer followed by 92 routed-MoE
layers. Every routed layer selects 16 of 896 experts. Attention comprises 69
KDA layers and 24 Gated DSA layers; DSA occurs at every fourth layer and at
layer 93.

```text
pinned moonshotai/Kimi-K3.1-Pro SafeTensors checkpoint
             |
             +-- static/model-control tier -> Q8/BF16 ROCm residency
             |     KDA, DSA/indexer, routers, shared experts, norms, head
             |
             `-- routed MXFP4 experts -> local NVMe
                    +-- raw SafeTensors path
                    +-- O_DIRECT + io_uring
                    `-- persistent per-layer decoded-expert cache
```

Each DSA layer applies key LayerNorm, computes indexer scores, selects at most
2,048 causal keys with deterministic tie handling, and performs sparse MLA over
the selected latent rows. The engine stores 128 BF16 index-key elements per
token per DSA layer alongside the 576-element BF16 MLA row. Derived top-k
indices are execution data, not checkpoint state.

See [Architecture](docs/architecture.md) for the graph and state contracts.

## Installation

### Requirements

The development host used for the synthetic ROCm gate has:

- x86-64 Linux with `io_uring` and `O_DIRECT` support;
- an AMD Radeon 8060S (`gfx1151`) with shared/device-addressable memory;
- ROCm with HIP, hipBLAS, and hipBLASLt development files;
- a C compiler, GNU Make, binutils, ICU, and Zstandard development files;
- Python 3 for the offline inspector and planner.

Other GPUs, ROCm versions, filesystems, and deployment sizes are unqualified.
Historical K3 memory ledgers must not be reused as K3.1 capacity evidence: the
BF16 DSA index-key cache adds state and workspace.

On Ubuntu, install the ordinary build dependencies after installing ROCm from
AMD's supported packages:

```sh
sudo apt install build-essential binutils libicu-dev libzstd-dev python3
```

### Clone and build

```sh
git clone https://github.com/outerreaches/moonshine.git
cd moonshine

/opt/rocm/bin/hipcc --version
rocminfo | grep gfx1151
make
```

`make` builds `libmoonshine.a`, `moonshine-chat`, and `moonshine-server`.
Building the binaries is not a full-model qualification.

## Download and inspect the pinned model

Weights are not included. They remain governed by the license published with
the [official K3.1 Pro release](https://huggingface.co/moonshotai/Kimi-K3.1-Pro).
Download the exact revision. If `hf` is not installed, use an isolated
environment first:

```sh
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install huggingface_hub
. .venv-hf/bin/activate
```

```sh
hf download moonshotai/Kimi-K3.1-Pro \
  --revision 6d0a0202956796543364d666ab19c6ad1a494ee1 \
  --local-dir /path/to/moonshotai__Kimi-K3.1-Pro
```

The inspector expects all 96 official SafeTensors shards, their index, and the
pinned configuration. Metadata inspection reads JSON and shard headers without
loading tensor payloads:

```sh
python3 tools/k31_checkpoint.py \
  /path/to/moonshotai__Kimi-K3.1-Pro \
  > /path/to/private/k31-source-manifest.json
```

To hash every input file as well, add `--verify-sha256`. That option streams the
whole checkpoint and is an offline integrity operation, not an inference test.
The full pinned-source MZG2 conversion and verification have run successfully.
To reproduce the standalone bundle build and its independent verification:

```sh
export MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3.1-Pro
export MOONSHINE_BUNDLE=/path/to/private/Kimi-K3.1-Pro-Moonshine-MZG2

python3 tools/build_mzg2_bundle.py build \
  --model "$MOONSHINE_MODEL" --out "$MOONSHINE_BUNDLE"
python3 tools/build_mzg2_bundle.py verify \
  --bundle "$MOONSHINE_BUNDLE" --model "$MOONSHINE_MODEL"
```

The accepted conversion verified all 92 routed layers and 82,432 experts. Keep
the source revision and generated manifests bound together; historical K3
bundles and manifests are not substitutes.

## Verified synthetic gates

The exact gates run for this cutover are:

```sh
make -j4 test-cpu
make -j4 test
```

Results:

- `make -j4 test-cpu`: **PASS**, including synthetic K3.1 profile/checkpoint,
  placement, DSA memory, and CPU DSA oracle coverage;
- `make -j4 test`: **PASS** on Radeon 8060S `gfx1151`, including DSA GPU tests;
- the CPU DSA oracle also passed ASan/UBSan.

The DSA GPU coverage includes LayerNorm/scoring, top-k boundaries, ties,
negative-infinity padding, sparse attention, and decode/prefill equivalence.
These are model-free synthetic gates. They do not establish that a complete
checkpoint loads or generates correct text.

The official Python SDK fixture **passed** with `openai 2.52.0`. To reproduce it
in an isolated environment:

```sh
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r tests/requirements-sdk.txt
make test-openai-sdk PYTHON=.venv-sdk/bin/python
```

This fixture establishes SDK protocol compatibility; live quality and production
qualification remain separate.

## Server and API shape

The verified MZG2 bundle can be launched directly. The real 8K canary used a
loopback, one-slot server; that scope does not make this a production profile.

```sh
./moonshine-server "$MOONSHINE_BUNDLE" \
  --host 127.0.0.1 \
  --port 8080 \
  --api-key "$MOONSHINE_API_KEY"
```

The server exposes `GET /health`, `GET /v1/models`, and
`POST /v1/chat/completions`. It has one inference slot. Set an API key whenever
binding beyond loopback.

```sh
curl --max-time 0 http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $MOONSHINE_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "moonshotai/Kimi-K3.1-Pro",
    "messages": [{"role": "user", "content": "Say hello."}],
    "reasoning_effort": "high",
    "max_completion_tokens": 32,
    "stream": false
  }'
```

Only `moonshotai/Kimi-K3.1-Pro` is accepted as the request model. Function
tools, preserved `reasoning_content`, bounded structured output, and exact
prefix reuse are described in [Agentic API and tool use](docs/agentic-api.md).
These API parser/codec paths have synthetic coverage. Real K3.1 tool calling,
tool-result continuation, structured JSON, and medium/image/old-model HTTP 400
rejection also passed in the Beelink canary. They are canary results, not
production-quality or multi-client qualification.

## State and cache behavior

State format v2 serializes KDA recurrence/convolution state, occupied MLA rows,
occupied DSA index-key rows, AttnRes state, and token position. Imports validate
format, model/static identity, lengths, and checksums before mutation; v1 files
are rejected. Derived packed keys, top-k indices, and routed-expert cache/LRU
contents are excluded.

The DSA index-key cache is persistent for the engine lifetime and is shared by
decode and prefill. It uses BF16 elements, not the official FP8/MXFP4 cache
formats. State and server diagnostics expose separate DSA index-cache digests
and indexer/top-k/selected-row counters.

## Offline placement planning

`tools/moonshine_placement.py` assigns whole sidecar layer files to explicit
drive profiles using deterministic, capacity-safe weighted placement. It reads
only the supplied JSON profile and writes only the requested manifest:

```sh
python3 tools/moonshine_placement.py /path/to/profile.json \
  --output /path/to/private/placement.json
```

Synthetic planner fixtures pass. This does not qualify a K3.1 MZG2 store or a
multi-drive deployment.

## Real 8K Beelink canary

The full K3.1 MZG2 bundle passed all 92-layer/82,432-expert verification and
loaded on Beelink in 102.243 seconds at 8K context. Health/model discovery and
real generation passed: 20 tokens decoded at 0.489 tok/s, response `hello`, and
normal stop. A preserved-thinking follow-up reused 114 cached tokens and also
completed normally. A longer 128-token sustained decode completed at
0.501 tok/s with coherent output and the expected `length` stop at the cap.

Live tool calling, tool-result continuation, and structured JSON also passed.
Public medium/image/old-model rejection returned HTTP 400. Real state-v2
checkpoints survived a process restart and reused 161 tokens on the continuation.
Checkpoint displacement/restart and header/stale-model/payload/truncation rejection
passed. These are real-weight canary results, not synthetic fixtures, but they
are not production performance, quality, or soak evidence.

## Natural-context evidence through 64K

The natural-text retrieval gates are distinct from both synthetic component
tests and the short real canary. Every listed prompt returned the exact answer
`saffron|7319|Nivens`:

| prompt tokens | prefill | 17-token decode |
|---:|---:|---:|
| 15,993 | 6.545 tok/s | 0.478 tok/s |
| 31,999 | 5.438 tok/s | 0.462 tok/s |
| 63,971 | 4.397 tok/s | 0.454 tok/s |

Reproduce the three gates independently with the pinned model or matching
verified bundle:

```sh
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=16000
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=32000
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=64000
```

The 63,971-token run loaded in 93.976 seconds, prefilled in 14,547.439 seconds,
and decoded 17 tokens in 37.485 seconds. Its reported prefill phases were:

| phase | seconds |
|---|---:|
| KDA | 2,585.803 |
| DSA | 8,101.056 |
| expert stream | 3,463.059 |
| MoE tail | 248.327 |

These timings describe the measured 64K natural-context run only. They do not
qualify contexts above 64K, continuous 64K-token output, or production service.

## DSA performance and reproduction

The implemented DSA path uses a BF16 index cache, skips indexer/top-k work in
the identity window, switches to dense MLA at the measured 768-row threshold,
and gathers selected rows once above 2,048 keys. At real 8K shapes, the measured
sparse-to-dense attention comparison improved from 4.930 to 0.126 ms per layer.
The short real 128-token canary ran at 0.501 tok/s. Named DSA steps retain
specific failure diagnostics.

```sh
make bench-k31-dsa
```

This benchmark measures DSA kernels, not end-to-end throughput. BLAS-backed
indexer scoring improves 1.094 → 0.075 ms at 8K; the real 8K retrieval gate
passed, but no matched end-to-end speedup is claimed.

The experimental KDA BLAS backend passed the same 8K retrieval and reduced total
prefill 1,068.850 → 809.570 seconds. It remains opt-in pending numerical/state
qualification; use `MOONSHINE_RETRIEVAL_BACKEND=kda-blas` to reproduce it.

## Remaining qualification gates

1. Qualify and measure any context above 64K independently, including memory,
   correctness, and natural-context behavior; 63,971 tokens is the current
   natural-text boundary.
2. Run a production soak with the exact candidate, workload, context/output
   profile, and checkpoint configuration before any production claim.
3. Qualify sustained long-output generation and broader performance separately;
   neither the 128-token canary nor the 17-token long-context decodes establish
   those limits.
4. Require fresh exact-output/state and comparative timing evidence before
   promoting future performance work, including BLAS-backed indexer scoring.

## Historical K3 records

The `qualification-*.md` documents and 0.1.0/0.2.0 changelog entries record the
earlier pinned Kimi K3 implementation. They are preserved as history, including
their former context, SDK, MZG2, and performance claims. They are not evidence
for K3.1 Pro.

## Documentation

- [Getting started](docs/getting-started.md) — download, bundle conversion,
  synthetic gates, real-context reproduction, and remaining qualification gates.
- [Architecture](docs/architecture.md) — K3.1 graph, DSA, residency, and state
  v2 contracts.
- [Agentic API](docs/agentic-api.md) — current text-only API, reasoning values,
  tools, and structured output.
- [Deployment profiles](docs/deployment-profiles.md) — unqualified configuration
  examples and client reasoning maps.
- [Operational logging](docs/observability.md) — lifecycle, DSA counters, and
  diagnostics.
- [Provenance](docs/provenance.md) — source lineage and validation references.
- [Changelog](CHANGELOG.md) and [release qualification](RELEASING.md).

## License and security

Moonshine source is MIT licensed. Model weights are separate, are not
redistributed here, and remain subject to their upstream license. See
[LICENSE](LICENSE), [NOTICE](NOTICE), and [SECURITY.md](SECURITY.md).
