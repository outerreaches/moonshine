# Moonshine

Moonshine is an independent, experimental native inference engine for AMD
ROCm. This checkout is the **0.3.0 Kimi K3.1 Pro research-preview cutover**.
It is not affiliated with or endorsed by Moonshot AI.

## Acknowledgments

Built and maintained by [Alex Mazurovsky](https://alexmaz.com/) — [Outer Reaches](https://outerreaches.space/).

> **Support boundary:** synthetic CPU/ROCm gates pass. Full conversion and
> verification of the pinned K3.1 MZG2 bundle passed for all 92 routed layers
> and 82,432 experts, and the bundle passed real Beelink load, generation,
> prefix-reuse, API, and state-v2 canaries. Natural-text retrieval passes at
> 15,993, 31,999, and 63,971 tokens. No context above 64K and no production
> service has been qualified. Historical Kimi K3 results are not K3.1 evidence.

## Current model contract

The default K3.1 lane pins exactly one upstream model:

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

## Which model are you running?

This tree contains **three model lanes at different maturities**, and they do not
share an entry point. Read this before following any command below, because `make`
alone builds only the K3.1 binaries and the K3.1 instructions will not run the
others.

| Lane | Entry point | Built by | State |
|---|---|---|---|
| **Kimi K3.1 Pro** | `moonshine-chat`, `moonshine-server` | `make` (default) | The current research-preview engine documented throughout this README; natural-context evidence ends at 63,971 tokens |
| **MiMo V2.6 Flash** | `tools/mimo26_server`, `tools/mimo26_supervise.py` | `make tools/mimo26_server` — **not** in the default build | Served, immutable releases, qualified. See [Running MiMo V2.6 Flash](#running-mimo-v26-flash) |
| **GLM 5.3 Flash** | none in this tree | component objects only | **Not servable from here.** See [GLM 5.3 Flash status](#glm-53-flash-status) |

If a command says `moonshine-server` it is the K3.1 engine. If it says
`mimo26_server` it is MiMo. There is no GLM server. Historical Kimi K3 commands
and results are retained only in the explicitly marked historical sections.

Everything else is indexed in [docs/](docs/README.md), which also explains two
conventions these documents rely on: dated note titles and `Evidence/…` paths
refer to engineering records kept outside this repository.

## What is Moonshine?

Moonshine is a purpose-built inference engine for running large official
SafeTensors checkpoints on one 128 GB AMD Strix Halo system. It keeps the
non-routed model tier resident in ROCm memory and streams only selected MXFP4
routed experts from local NVMe. Kimi K3.1 Pro is the lane this README documents
throughout; MiMo V2.6 Flash is served by a second, separately built binary
([Running MiMo V2.6 Flash](#running-mimo-v26-flash)).

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

## Runtime scope

- x86-64 Linux and ROCm only;
- tested on `gfx1151` with ROCm 7.2;
- the pinned official 96-shard Kimi K3.1 Pro checkpoint on the default lane, or
  the pinned `XiaomiMiMo/MiMo-V2.6-Flash-RL` checkpoint on the MiMo lane;
- Q8 static residency with dynamically allocated context; current K3.1
  natural-context evidence ends at 63,971 tokens and does not extend above 64K;
- greedy next-token inference;
- one large model process and one active request at a time.

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
The merged tree also contains the GLM component lane. Its phase-5d KDA test
needs `tests/fixtures/glm53_phase4_kda_v1.bin`, which is derived from official
GLM weights and intentionally not committed. Set `GLM53_OFFICIAL_ROOT` and run
`make test-glm53-phase4-official` to regenerate it before running the combined
`make test`; without that fixture, the GLM gate stops at fixture open and is not
a K3.1 result.

These are model-free synthetic gates. They do not establish that a complete
checkpoint loads or generates correct text.

## Running MiMo V2.6 Flash

A second engine in this tree: MiMo-V2.6-Flash-RL, 310B total / 15B active, 48
layers, 256 experts top-8. Shares K3's expert cache, MXFP4 kernels, SafeTensors
reader and prefix-bundle machinery; everything model-shaped is its own.

**It is not in the default build.**

```sh
make tools/mimo26_server
```

### Serve it

Production runs from an immutable release under a supervisor, not from the build
tree. The supervisor re-resolves `current/` on every start, verifies the release,
and checks the effective profile against `/health` after load:

```sh
python3 tools/mimo26_supervise.py \
    /srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL \
    --release-root /srv/modelstore/private/moonshine-releases/mimo26 \
    --allowed-signers ~/.ssh/allowed_signers --require-signature \
    --port 8080 \
    --request-deadline-seconds 1800 \
    --kv-prefix-reuse on \
    --prefix-cache-dir /srv/modelstore/private/moonshine-prefix-checkpoints/mimo26-v2.6-flash \
    --prefix-cache-gib 24 --prefix-cache-entries 8 \
    --status-file /run/user/1000/mimo26-status.json
```

`--require-signature` makes an unsigned or tampered release a refusal to launch
(`release_unusable`) rather than a service started on something unattested. It needs
`--allowed-signers`, and both `current` and `previous` must be signed or a rollback
cannot start. See [RELEASING.md](RELEASING.md) for signing and for the proof that
this gate can actually refuse.

To run a build directly — for qualification or a bisect, not for production:

```sh
./tools/mimo26_server /srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL \
    --host 127.0.0.1 --port 8080 \
    --slots 160 --context 131072 --prefill-chunk 128 \
    --expert-lookahead on --expert-major on --retain-experts on \
    --kv-prefix-reuse on --request-deadline-seconds 1800
```

The API is OpenAI chat-completions on `/v1/chat/completions`, plus `/health`.
Decoding is **greedy only**: `temperature` and `top_p` are *refused*, not ignored,
because a silently-ignored sampler would invalidate the kernel's bit-exactness
guarantees.

### Operational facts worth knowing before you run it

- **Defaults are a qualified pairing.** 160 expert slots at a 131,072 context needs
  ~108 GiB and clears an 8 GiB floor on an otherwise idle host. Stop other large
  model processes first; the startup guard refuses rather than half-loading.
- **Prefill slows with depth** — roughly 20 t/s shallow, ~9 t/s at 26K — so the
  request deadline, not memory, is what bounds a long prompt. At 1800 s the
  single-request ceiling is around 17.5K tokens.
- **A prompt too long for one deadline converges across retries.** The evaluated
  prefix is checkpointed, so retrying the same request resumes rather than
  restarting. The response says which of those applies.
- **A deadline reached before any token is generated returns HTTP 504**
  (`deadline_exceeded`), not a truncated 200 — the message names elapsed time,
  tokens evaluated and reused, and whether a retry can resume.
- **Context is qualified end to end to 26,026 tokens.** The 131,072 ceiling is
  covered only by synthetic attention bit-exactness and footprint arithmetic; a
  full-depth prefill is tens of hours.

## GLM 5.3 Flash status

**GLM 5.3 Flash cannot be served from this tree.** What is here is component-level:
24 sources (architecture, FP8 codec and oracle, expert plan and stream, manifest,
dense/KDA/MHC ops) with object-file build rules and phase-gated tests
(`make test-glm53-phase2` … `test-glm53-phase5d-kda`). There is no runner, server or
chat binary, and nothing GLM is in the default build.

The FP8 work that *is* qualified — strict manifest, exhaustive OCP E4M3FN oracle,
two-extent expert plan, fused GEMV, bounded BF16-dequant prefill floor — is
synthetic-qualified on gfx1151 and reusable; the checkpoint lives at
`/srv/modelstore/models/glm53-flash` (306 GB).

Active development is in a **separate worktree**,
`moonshine-glm53-20260831`, branch `glm53/native-fp8-foundation-20260831`, tag
`glm53-wip-20260921` (unmerged, never pushed). That lane carries 261 GLM files
against the 90 here, so the sources in this tree are an early subset and not a
current picture of it.

What that lane can already do, so this section is not read as less than it is: all
45 text layers execute natively in official FP8 on gfx1151, behind a single-slot
localhost HTTP adapter (`glm53_server.py` plus a `moonshine-glm53-worker` binary),
qualified at 128 prompt / 128 output tokens on the `context256` profile. It decodes
at roughly **0.44 evaluations/s** — about 2.3 s per token — and a 2,331-token
prefill measured 5,588 s, so context beyond a few hundred tokens is not currently
practical.

What has **not** started, as of 2026-09-22, is the final qualification campaign:
lifecycle (G1), deployment and canary (G4), the 24 h soak (G5) and the release
decision (G6) are written up and ready to execute but awaiting a GPU window, and
two of their harnesses are not yet written. Streaming, concurrency, contexts past
256, and independent quality are all open. GLM 5.3 Flash is therefore **out of
scope for this release** and is the subject of the next one.

## Optional features

### Function tools and reasoning

Chat Completions supports standard function declarations and `tool_choice`
values `auto`, `required`, `none`, or a specifically named function. Multiple
calls and matching tool results are supported. K3.1 thinking is preserved through
`reasoning_content`; public `reasoning_effort` accepts `low`, `high`, or `max`.
The internal renderer also accepts `medium` for format compatibility, but the
public K3.1 API rejects it.

Clients must replay the complete returned assistant message—including
`reasoning_content`, `content`, and `tool_calls`—to remain eligible for exact
causal-prefix reuse. See [Agentic API and tool use](docs/agentic-api.md).

### Structured output

`response_format` supports `json_object` and a bounded `json_schema` subset.
The schema path supports objects, arrays, strings, numbers, integers, booleans,
nulls, `properties`, `required`, `additionalProperties`, and `items`.
Unsupported vocabulary is rejected before inference. SSE reasoning remains live
while structured response content is buffered until validation succeeds.

### Causal-state checkpoints

Moonshine can export and import exact KDA, convolution, MLA, AttnRes, and token
position state. Files bind to the state format, configured context, model
layout, and static precision mode; imports validate size and CRC64 before
mutating device state.

```sh
make test-state-checkpoint \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" \
  MOONSHINE_STATE_DIR=/tmp
```

The experimental opt-in server path can preserve bounded exact session
checkpoints:

```sh
./moonshine-server "$MOONSHINE_BUNDLE" \
  --prefix-checkpoint-root /private/moonshine-checkpoints \
  --prefix-checkpoint-entries 4 \
  --prefix-checkpoint-bytes 21474836480
```

The root is `0700`; state, metadata, and manifest files are `0600`. Admission
remains full-token and identity exact. Current K3.1 model-backed
displacement/restart response bytes, cached-token accounting, and causal-state
digests passed; the real state-v2 results are detailed below.

Historical K3 measurements recorded restored-prefix cost falling from an
inferred 63.008 s of evaluation to 0.958 s import (**65.77×**); total prompt wall
fell 95.174 → 33.124 s (**2.87×**, 65.20% lower). The mechanism passed its ≥5×
prefix-restoration gate. Those numbers are not K3.1 performance evidence.

Production activation of that historical K3 path remained blocked: a live canary
proved exact displacement and restart recovery, but an immediate completion POST
32 ms after response delivery received HTTP 503 while synchronous checkpoint
export held the one request slot for 2.097 seconds. Do not enable the experimental
checkpoint path for a no-retry agent client until the server can accept one
bounded next-turn request during terminal checkpoint export. Neither the
historical K3 result nor the current K3.1 state-v2 canary is production
qualification.

### Diagnostics and offline cache analysis

`--decode-diagnostics PREFIX` writes private cache snapshots, route traces, and
per-layer timing/I/O ledgers. `--router-logits-tap` adds raw router logits;
`--decode-state-digest` adds non-cryptographic state fingerprints for paired
runs. These are opt-in qualification tools and can contain workload-derived
information.

Captured routes can be replayed with `tools/analyze_decode_cache.py`, including
uniform LRU, capacity curves, fixed-total allocation, and scan-resistant policy
screens. `tools/analyze_anchor_recovery.py` bounds exact checkpoint recovery
from recorded common-prefix/anchor positions; it cannot infer semantic reuse.
See [Operational logging](docs/observability.md) and
[Offline decode-cache analysis](docs/offline-decode-cache-analysis.md).

### Diagnostic KDA prefill backend

`--range-backend kda-blas` selects the bounded KDA dequantize-plus-hipBLAS range
path. It improves the measured KDA component but changes BF16 reduction order,
selected values, and causal-state hashes. It remains diagnostic rather than the
production default.

### Official OpenAI SDK fixture

The SDK fixture is isolated from Moonshine's native runtime dependencies. The
official Python SDK fixture **passed** with `openai 2.52.0`. To reproduce it in
an isolated environment:

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

Bundle startup reports `model_source=bundle` and selects its embedded MZG2
store. Do not set an external expert-store selector. The official SafeTensors
root remains the build/archive/rollback source, but is not read by the default
running service. Passing the official model root with an explicit external MZG2
selector remains available as a rollback and comparison profile.

The server exposes `GET /health`, `GET /v1/models`, and
`POST /v1/chat/completions`. It has one inference slot. Set `MOONSHINE_API_KEY`
or pass `--api-key` to require bearer authentication. Moonshine refuses a
non-loopback bind without a key; for example, use `--host 0.0.0.0` together with
`--api-key "$MOONSHINE_API_KEY"` only when remote access is intended.

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

## Additional qualification fixtures

### Two environment variables the deeper gates need

- **`MOONSHINE_EVIDENCE_ROOT`** — the gates in `tests/run_mimo26_*.py` compare
  against artifacts from earlier qualification runs (prior reports, pinned worker
  builds, recorded routes), which are kept outside this repository. Point this at
  that tree; unset, each one refuses with an instruction rather than a stray
  `FileNotFoundError`.
- **`GLM53_OFFICIAL_ROOT`** — required by every `test-glm53-*-official` target.
  It also regenerates the Phase-4 goldens, which are derived from official
  weights and so are not committed; see
  [docs/glm53-phase4-goldens.md](docs/glm53-phase4-goldens.md).

## Historical K3 records

The `qualification-*.md` documents and 0.1.0/0.2.0 changelog entries record the
earlier pinned Kimi K3 implementation. They are preserved as history, including
their former context, SDK, MZG2, and performance claims. They are not evidence
for K3.1 Pro. Everything in the subsections below is likewise historical K3
reproduction material, not a current K3.1 profile or acceptance target.

### Historical K3 bundle construction

```sh
./tools/build_mzg2_bundle.py build \
  --model /path/to/moonshotai__Kimi-K3 \
  --reuse-mzg2 /path/to/expert-store-mzg2 \
  --out /path/to/Kimi-K3-Moonshine-MZG2
```

The reuse path creates hard links and therefore requires one filesystem. The
builder validates the pinned source manifest, writes through resumable partial
state, verifies static tensors and MZG2 sidecars, copies tokenizer/config/license
files, and atomically publishes only a complete bundle. The final public bundle
will be Moonshine-only; it will not be directly loadable by Transformers,
vLLM, or SGLang.

### Historical K3 measured checkpoint

Results below are engineering fixtures on an AMD Ryzen AI Max+ 395 / Radeon
8060S (`gfx1151`), Ubuntu 24.04, Linux 7.0, ROCm 7.2, and a Samsung 990 PRO.
They are not cross-project benchmark claims or K3.1 evidence.

| workload | result |
|---|---:|
| K3 MZG2 startup, 128K/30 | ~43 s |
| Raw engine-hello prompt mean | 65.5965 s |
| MZG2 engine-hello prompt mean | 57.7645 s (**11.94% lower**) |
| Raw engine-hello decode mean | 40.6795 s |
| MZG2 engine-hello decode mean | 34.4120 s (**15.41% lower**) |
| Selected prefill, 512 positions | 4.145 tok/s |
| Selected prefill, 8,192 positions | 8.126 tok/s |
| Selected prefill, 32,768 positions | 7.462 tok/s |
| OpenAI JSON chat at 128K, prompt/decode | 0.438 / 0.498 tok/s |
| 128K runtime state | 4.150 GiB |
| 128K decoded expert cache, 30 slots/layer | 45.104 GiB |
| Long MZG2 agentic session cache hit rate | 35.31% |
| Causal-state export/import after two positions | 1.043 / 0.841 s |

Prefill rows are workload-dependent because Moonshine reads each layer's actual
expert union. Short suffixes can avoid most of the store; long natural-text
prompts often route densely. Full evidence and provenance are in the historical
qualification documents linked by the changelog.

### Historical K3 verification commands

The model-free checks used by that lane were:

```sh
make test-cpu
make test
```

For the locked full-engine fixture, stop other large inference processes,
confirm memory and swap state, then run:

```sh
free -h
pgrep -a -f 'llama|ds4|k3'

make test-engine-hello \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_CONTEXT=131072
```

A passing historical run ends with `K3 hello: PASS` and reports startup, token
IDs, prompt/decode timing, selected values, and cache counters. Its 131,072-token
configuration is not a K3.1 context claim.

The historical prefill and long-context gates were:

```sh
make test-reduction-qualification \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3

make test-prefill-2 \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3

make test-prefill-scale \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_PREFILL_TOKENS=512
```

Use the current K3.1 gates and explicit 64K evidence boundary above for the
0.3.0 candidate; do not substitute these historical commands or results.

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
