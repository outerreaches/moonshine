# Moonshine

## Acknowledgments

Moonshine exists because Moonshot AI published the Kimi K3 architecture,
reference implementation, tokenizer assets, and official model checkpoint. The
project name is a nod to Moonshot AI; Moonshine is an independent community
project and is not affiliated with or endorsed by Moonshot AI.

The standalone source tree was extracted from K3 development in the
MIT-licensed [DwarfStar / ds4](https://github.com/antirez/ds4) codebase. That
lineage also carries acknowledgments to
[llama.cpp](https://github.com/ggml-org/llama.cpp) and
[GGML](https://github.com/ggml-org/ggml) for inference-engineering knowledge,
quantization layouts, kernels, and tests. llama.cpp, vLLM, SGLang, and the
official Kimi code supplied validation references; none is a Moonshine runtime
dependency.

The durable exact-prefix checkpoint work was inspired by
[FreeToken](https://github.com/FlashML-org/FreeToken)'s hybrid radix cache and
semantic anchor placement, alongside
[SGLang](https://github.com/sgl-project/sglang)'s Mamba radix cache and
[vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/)'s exact prefix
caching design. These systems clarified the key invariant: reuse remains
token-exact, while semantic boundaries determine useful snapshot placement.
Moonshine's implementation is independent, uses bounded durable disk bundles
rather than their hot cache structures, and copies no source from them.

This project was developed with strong AI coding and review assistance, with
the human maintainer directing architecture, experiments, validation, and
release decisions. Exact source lineage, pinned revisions, design influences,
and validation oracles are recorded in
[Provenance and acknowledgements](docs/provenance.md) and [NOTICE](NOTICE).

## Which model are you running?

This tree contains **three model lanes at different maturities**, and they do not
share an entry point. Read this before following any command below, because `make`
alone builds only the K3 binaries and the K3 instructions will not run the others.

| Lane | Entry point | Built by | State |
|---|---|---|---|
| **Kimi K3** | `moonshine-chat`, `moonshine-server` | `make` (default) | The engine this README documents throughout |
| **MiMo V2.6 Flash** | `tools/mimo26_server`, `tools/mimo26_supervise.py` | `make tools/mimo26_server` — **not** in the default build | Served, immutable releases, qualified. See [Running MiMo V2.6 Flash](#running-mimo-v26-flash) |
| **GLM 5.3 Flash** | none in this tree | component objects only | **Not servable from here.** See [GLM 5.3 Flash status](#glm-53-flash-status) |

If a command says `moonshine-server` it is the K3 engine. If it says
`mimo26_server` it is MiMo. There is no GLM server.

## What is Moonshine?

Moonshine is a purpose-built inference engine for running the official Kimi K3
SafeTensors checkpoint on one 128 GB AMD Strix Halo system. It keeps the
non-routed model tier resident in ROCm memory and streams only selected MXFP4
routed experts from local NVMe.

Moonshine is a research preview, not a general model runner. It currently
provides:

- a native C API and deterministic greedy K3 execution;
- token-major decode and selected-expert layer-major prefill;
- a persistent online routed-expert cache;
- exact append-only causal-prefix reuse;
- versioned, checksummed causal-state export and import;
- a native tokenizer and K3 XTML chat/tool renderer;
- a stateful command-line client;
- a one-slot OpenAI-compatible Chat Completions server;
- function tools, preserved reasoning, and bounded structured output;
- the qualified MZG2 compressed expert store for the K3 production profile.

The supported profile is deliberately narrow:

- x86-64 Linux and ROCm only;
- tested on `gfx1151` with ROCm 7.2;
- the pinned official 96-shard Kimi K3 checkpoint;
- Q8 static residency with dynamically allocated context, capacity-qualified
  through 128K;
- greedy next-token inference;
- one large model process and one active request at a time.

Other GPU architectures, ROCm releases, filesystems, model revisions, and
contexts above 128K are not qualified yet. Fully resident BF16 is rejected on
the tested 128 GB machine.

## How Moonshine works

Kimi K3 has 93 transformer layers: one dense layer followed by 92 routed-MoE
layers, with 69 KDA and 24 gated-MLA attention layers. Every routed layer
selects 16 of 896 experts.

```text
official 96-shard Kimi K3 SafeTensors checkpoint
             |
             +-- static/model-control tier -> Q8/BF16 ROCm residency
             |     attention, routers, shared experts, norms, output head
             |     accepted Q8 residency: ~55.27 GiB
             |
             `-- routed MXFP4 experts -> local NVMe
                    |
                    +-- MZG2 rANS sidecars in the qualified K3 deployment
                    +-- raw SafeTensors fallback
                    +-- O_DIRECT + io_uring, QD2
                    +-- GPU decode direct to decoded cache slots
                    `-- per-layer online LRU cache
```

### Decode

Decode is token-major. Moonshine walks all 93 layers for each token and retains
KDA recurrent state, convolution state, compressed MLA cache, AttnRes state,
and decoded routed experts between tokens. The accepted 128K profile uses 30
expert slots per routed layer, or 45.104 GiB of cache storage.

### Prefill

Prefill is layer-major. A token range visits each routed layer once, groups
rows by selected expert, and reads only the unique route union in physical
storage order. Cold prefill can borrow an explicitly accounted part of the
empty decode cache for workspace. A guarded warm-miss path can temporarily
lend a slot-aligned cache tail without lowering the host-memory reserve.

### Reuse and isolation

The server retains the most recent successful causal state. A request reuses
that state only when its fully rendered token history is an exact append-only
extension. Edited, forked, shorter, or otherwise mismatched histories reset
semantic state and prefill independently. The expert cache remains reusable
because it contains immutable model weights, not conversation state.

See [Architecture](docs/architecture.md) for the complete graph, memory ledger,
I/O contracts, and failure invariants.

## Installation

### Requirements

The qualified host has:

- x86-64 Linux with recent `io_uring` and `O_DIRECT` support;
- an AMD ROCm device with shared/device-addressable memory;
- ROCm 7.2 with HIP, hipBLAS, and hipBLASLt development files;
- a C compiler, GNU Make, binutils, ICU 74, and Zstandard 1.5 development files;
- approximately 128 GB of system/unified memory;
- an ext4 NVMe filesystem with enough capacity for the model and any derived
  store;
- the official Kimi K3 checkpoint at revision
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.

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

`make` builds `libmoonshine.a`, `moonshine-chat`, and `moonshine-server`. The
default ROCm target is `gfx1151`; override it only for porting work:

```sh
make ROCM_ARCH=gfx1151
make tests
```

### Obtain the model

Weights are not included. They remain governed by the
[Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/LICENSE)
published with the
[official model release](https://huggingface.co/moonshotai/Kimi-K3).

With the Hugging Face CLI installed:

```sh
hf download moonshotai/Kimi-K3 \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --local-dir /path/to/moonshotai__Kimi-K3
```

Moonshine expects all 96 `model-*.safetensors` shards and the original tensor
layout. Validate the download before allocating the full engine:

```sh
make test-model-layout MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-tokenizer MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

Detailed host checks and residency safeguards are in
[Getting started](docs/getting-started.md).

## Quick start

Stop other large model processes before launching Moonshine.

### Interactive client

```sh
./moonshine-chat /path/to/moonshotai__Kimi-K3
```

Enter one user message per line. The process remains resident and retains causal
and expert-cache state across turns. One-shot mode writes response text to
stdout and telemetry to stderr:

```sh
./moonshine-chat /path/to/moonshotai__Kimi-K3 \
  --prompt "Say hello." \
  --max-tokens 32
```

Use `/save PATH` and `/load PATH` interactively, or the corresponding `--save`
and `--load` options, for exact causal-state checkpoints.

### OpenAI-compatible server

The qualified 128K Beelink service runs from the standalone bundle by default:

```sh
./moonshine-server /path/to/Kimi-K3-Moonshine-MZG2 \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 131072 \
  --experts 30 \
  --max-output-tokens 65536
```

Bundle startup reports `model_source=bundle` and selects its embedded MZG2
store. Do not set an external expert-store selector. The official SafeTensor
root remains the build/archive/rollback source, but is not read by the default
running service. Passing the official model root with an explicit external
MZG2 selector remains available as a rollback and comparison profile.

The server exposes:

- `GET /health`;
- `GET /v1/models`;
- `POST /v1/chat/completions` with JSON or SSE responses.

```sh
curl --max-time 0 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "moonshine",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_completion_tokens": 32,
    "stream": false
  }'
```

Set `MOONSHINE_API_KEY` or pass `--api-key` to require bearer authentication.
Moonshine refuses a non-loopback bind without a key.

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
python3 tools/mimo26_supervise.py /srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL     --release-root /srv/modelstore/private/moonshine-releases/mimo26     --port 8080     --request-deadline-seconds 1800     --kv-prefix-reuse on     --prefix-cache-dir /srv/modelstore/private/moonshine-prefix-checkpoints/mimo26-v2.6-flash     --prefix-cache-gib 24 --prefix-cache-entries 8     --status-file /run/user/1000/mimo26-status.json
```

To run a build directly — for qualification or a bisect, not for production:

```sh
./tools/mimo26_server /srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL     --host 127.0.0.1 --port 8080     --slots 160 --context 131072 --prefill-chunk 128     --expert-lookahead on --expert-major on --retain-experts on     --kv-prefix-reuse on --request-deadline-seconds 1800
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
`glm53-wip-20260921`. As of 2026-09-22 that lane is *prep-complete, awaiting a GPU
window* — GPU bring-up has not started. Do not expect the sources here to be a
complete or current picture of it.

## Optional features

### Function tools and reasoning

Chat Completions supports standard function declarations and `tool_choice`
values `auto`, `required`, `none`, or a specifically named function. Multiple
calls and matching tool results are supported. K3 thinking is preserved through
`reasoning_content`; `reasoning_effort` accepts `low`, `medium`, `high`, or
`max`.

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
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_STATE_DIR=/tmp
```

The experimental opt-in server path can preserve bounded exact session
checkpoints:

```sh
./moonshine-server /path/to/moonshotai__Kimi-K3 \
  --prefix-checkpoint-root /private/moonshine-checkpoints \
  --prefix-checkpoint-entries 4 \
  --prefix-checkpoint-bytes 21474836480
```

The root is `0700`; state, metadata, and manifest files are `0600`. Admission
remains full-token and identity exact. Model-backed displacement/restart
response bytes, cached-token accounting, and all four state digests passed.
The measured restored-prefix cost fell from an inferred 63.008 s of evaluation
to 0.958 s import (**65.77×**); total prompt wall fell 95.174 → 33.124 s
(**2.87×**, 65.20% lower). The mechanism passes its ≥5×
prefix-restoration gate; production response ordering is a separate gate.

Production activation is currently blocked. A live canary proved exact
displacement and restart recovery, but an immediate completion POST 32 ms after
response delivery received HTTP 503 while synchronous checkpoint export held
the one request slot for 2.097 seconds. Do not enable this path for a no-retry
agent client until the server can accept one bounded next-turn request during
terminal checkpoint export.

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

The SDK fixture is isolated from Moonshine's native runtime dependencies:

```sh
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r tests/requirements-sdk.txt
make test-openai-sdk PYTHON=.venv-sdk/bin/python
```

It validates SSE comments, reasoning deltas, indexed tool calls, usage, and
`[DONE]` with the official Python client.

## Expert-stream compression

### MZG2 production store

MZG2 is Moonshine's qualified GPU-rANS expert store for Kimi K3. It
canonicalizes only redundant MXFP4 negative zero, encodes independently bounded
16 KiB tiles, and retains per-tile checksums and terminal-state validation.
Selected blocks are read through the existing QD2 direct-I/O path and decoded
on the GPU directly into ordinary decoded cache slots.

The complete verified store contains all 82,432 routed experts:

| item | result |
|---|---:|
| Canonical routed-expert bytes | 1,446,456,066,048 B |
| MZG2 stored block bytes | 1,257,439,830,016 B |
| Final verified store size | 1,171.084 GiB |
| Reduction | 13.0674105% |
| Tile size | 16 KiB |

The store is derived locally and is not distributed with Moonshine. Budget the
official roughly 1.45 TiB checkpoint, about 1.18 TiB for the final MZG2 store,
and temporary headroom while sidecars are verified and published.

Build a derived store without modifying the official checkpoint:

```sh
make tools/transcode_mzg2_layer
./tools/transcode_mzg2_full.py \
  --model /path/to/moonshotai__Kimi-K3 \
  --out /path/to/moonshotai__Kimi-K3/expert-store-mzg2 \
  --jobs 24
```

The orchestrator writes to a partial root, verifies every block immediately and
again against canonicalized SafeTensors, records sidecar SHA-256 values, and
publishes atomically only after all experts pass. The official 96 shards remain
the model identity and rollback source.

Select the store with an absolute path:

```sh
MOONSHINE_MZG2_STORE=/path/to/moonshotai__Kimi-K3/expert-store-mzg2
```

Unset or `off` uses raw SafeTensor experts. `MOONSHINE_EXPERT_STORE` remains the
legacy MZG1 selector and is mutually exclusive with MZG2.

A fresh 128K/30 raw/MZG2/MZG2/raw comparison preserved generated IDs, selected
values, and cache counters while reducing mean prompt wall by 11.94% and mean
decode wall by 15.41%. The qualified K3 production profile therefore uses MZG2.

MZG2 compresses the on-disk/read path, not the resident cache. Cache slots hold
decoded 16.734 MiB experts. A measured compressed-resident C34 prototype gained
2.326 cache-hit points but slowed prompt wall 2.10% and decode wall 4.75%, so it
was reverted.

See [MZG2 architecture and deployment](docs/architecture.md) and the
[MZG2 transcoding instructions](docs/getting-started.md#mzg2-production-gpu-decoder).

### Standalone MZG2 bundle — default Beelink deployment

The standalone bundle removes the runtime dependency on the 96 official
SafeTensor shards. It keeps the 2,460 non-routed language tensors at source
precision in one deterministic `model-static.safetensors` file and stores
routed experts only as MZG2:

```text
model-static.safetensors   ~105.7 GiB
expert-store-mzg2          1,171.084 GiB
total                      ~1.247 TiB
```

A local production-size bundle is now the default weight source on the
qualified Beelink host. At commit `f38a1ce`, the model-free suite,
dense-versus-bundle engine hello and chat comparison, tokenizer, state
checkpoint, and reset/displaced/restart durable-prefix gates pass. Generated
IDs, selected values, the static ledger, cache counters, and recorded response
bytes are exact across the paired core fixtures.

The independent full-bundle verifier passed before the 2026-08-30 cutover. The
128K/30 production service then started with `model_source=bundle`; an
authenticated request returned HTTP 200, natural stop, and exact content
`standalone bundle ready`. After the smoke, its process held normal and direct
descriptors for 93 unique bundle paths—the static pack and 92 MZG2 layers—with
zero descriptors below the official source root.

Clean-machine, archive-restore, and publication qualification remain open.
Retain the official SafeTensors as source and rollback material until those
gates pass; the default running service no longer depends on them.

Users who have the pinned official SafeTensors can build the same standalone
bundle locally:

```sh
make tools/transcode_mzg2_layer
./tools/build_mzg2_bundle.py build \
  --model /path/to/moonshotai__Kimi-K3 \
  --out /path/to/Kimi-K3-Moonshine-MZG2 \
  --jobs 24
```

If a complete verified MZG2 store already exists on the same filesystem, reuse
it without another 1.17 TiB copy:

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

## Measured checkpoint

Results below are engineering fixtures on an AMD Ryzen AI Max+ 395 / Radeon
8060S (`gfx1151`), Ubuntu 24.04, Linux 7.0, ROCm 7.2, and a Samsung 990 PRO.

They are not cross-project benchmark claims.

| workload | result |
|---|---:|
| Current MZG2 startup, 128K/30 | ~43 s |
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
prompts often route densely. Full evidence and provenance are in the
qualification documents linked below.

## Verification and qualification

### Model-free checks

```sh
make test-cpu
make test
```

### Locked full-engine fixture

Stop other large inference processes, confirm memory and swap state, then run:

```sh
free -h
pgrep -a -f 'llama|ds4|k3'

make test-engine-hello \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_CONTEXT=131072
```

A passing run ends with `K3 hello: PASS` and reports startup, token IDs,
prompt/decode timing, selected values, and cache counters.

### Prefill and long-context gates

```sh
make test-reduction-qualification \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3

make test-prefill-2 \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3

make test-prefill-scale \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_PREFILL_TOKENS=512

make test-long-context-retrieval \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

Some real-weight gates run for tens of minutes or hours. Their exact workload,
memory, and pass criteria are documented in `docs/qualification-*.md` and
[RELEASING.md](RELEASING.md).

## Operational constraints

- At 128K on the qualified 128 GB host, use `--experts 30`. Thirty-two slots can
  violate the CMA-plus-4-GiB safety reserve on a later request.
- Runtime state is 0.920 GiB at 8K, 1.566 GiB at 32K, and 4.150 GiB at 128K.
- The server has one inference slot. Health/model discovery remain responsive
  while it is occupied. Inference rejects competing completions with HTTP 503;
  the checkpoint-handoff candidate admits exactly one next completion during
  terminal checkpoint export and rejects further contenders.
- The source ceiling is 65,536 output tokens, but the effective limit is
  clamped to configured and remaining context. A continuous 64K decode has not
  been qualified and would take many hours.
- Layer-major TTFT can be many minutes. Use streaming and long client timeouts.
  SSE emits comment keepalives during both prefill and decode. A disconnected
  client cancels at the next complete token/layer boundary and resets semantic
  state while retaining immutable expert-cache entries.
- Auxiliary client requests can replace the single retained causal prefix and
  occupy the only request slot. Route title generation, security review, and
  similar work to another model when possible.
- Prompts, reasoning, response text, tool arguments, request bodies, bearer
  tokens, and API keys are excluded from production lifecycle logs.

Deployment-specific timeout and client guidance is in
[Deployment profiles](docs/deployment-profiles.md).

## Project status and roadmap

MZG2 is promoted for the qualified K3 deployment. Decisive screens closed
scan-resistant admission and compressed-resident cache slots. The route-index
engine integration is also closed by its 0.148% direct wall-time ceiling.

The durable-checkpoint mechanism and its bounded one-request export handoff
pass model-backed, live/displaced/restart, streaming, disconnect,
publication-failure, and shutdown gates. An opt-in production observation is
active; checkpoints are not yet the canonical default. Exact admission remains
strict: a Prime local refinement changed injected system-prompt material, so
Moonshine correctly declined a post-restart checkpoint and performed a cold
prefill.

The opt-in routed-prefill harness is functionally qualified, but its measured
noise floor leaves the former event-scheduler gate unresolved. Event scheduling
is parked.

Current release work:

1. Finish and review the active checkpoint observation.
2. Run the exact-head clean-checkout release suite and final secret/path audit.
3. Publish the linear local history only after maintainer approval; do not tag
   or distribute model weights or derived stores.
4. After stable checkpoint activation, measure Prime pre/post-compaction token
   boundaries before considering hot in-memory anchors.

Other open work:

- derived static-Q8 startup acceleration;
- dual-device expert streaming after sufficient second-device capacity exists;
- broader quality qualification for the diagnostic KDA backend;
- filled-128K workload characterization.

See [CHANGELOG.md](CHANGELOG.md) for implemented changes and
[RELEASING.md](RELEASING.md) for release gates. Publication remains local until
the active observation closes and a maintainer approves pushing the candidate
history.

## Documentation

- [Getting started](docs/getting-started.md) — host preparation, build, model,
  MZG2, and launch procedures.
- [Architecture](docs/architecture.md) — graph, storage, residency, I/O, state,
  and correctness contracts.
- [Agentic API](docs/agentic-api.md) — tools, thinking, structured output, and
  complete multi-turn examples.
- [Deployment profiles](docs/deployment-profiles.md) — 8K through persistent
  128K profiles and client timeout guidance.
- [Operational logging](docs/observability.md) — lifecycle event and privacy
  contracts.
- [Decode diagnostics qualification](docs/qualification-decode-diagnostics.md)
  and [offline cache analysis](docs/offline-decode-cache-analysis.md).
- [128K qualification](docs/qualification-128k.md),
  [64K output qualification](docs/qualification-output-64k.md), and
  [filled-context qualification](docs/qualification-filled-context.md).
- [Provenance and acknowledgements](docs/provenance.md).

## Contributing and security

Before contributing, read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of conduct](CODE_OF_CONDUCT.md). Report vulnerabilities through the
private process in [SECURITY.md](SECURITY.md), not a public issue.

## License

Moonshine source is MIT licensed. Kimi K3 model weights are separate, are not
redistributed here, and remain subject to the
[Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/LICENSE).
See [LICENSE](LICENSE) and [NOTICE](NOTICE).
