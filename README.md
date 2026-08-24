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

This project was developed with strong AI coding and review assistance, with
the human maintainer directing architecture, experiments, validation, and
release decisions. Exact source lineage, pinned revisions, design influences,
and validation oracles are recorded in
[Provenance and acknowledgements](docs/provenance.md) and [NOTICE](NOTICE).

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

A loopback 128K service using raw SafeTensor experts can be started directly:

```sh
./moonshine-server /path/to/moonshotai__Kimi-K3 \
  --host 127.0.0.1 \
  --port 8080 \
  --context 131072 \
  --experts 30 \
  --max-output-tokens 65536
```

The qualified K3 deployment selects MZG2 explicitly:

```sh
MOONSHINE_EXPERT_STORE=off \
MOONSHINE_MZG2_STORE=/path/to/moonshotai__Kimi-K3/expert-store-mzg2 \
./moonshine-server /path/to/moonshotai__Kimi-K3 \
  --host 127.0.0.1 \
  --port 8080 \
  --context 131072 \
  --experts 30 \
  --max-output-tokens 65536
```

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
| Published store size | 1,171.084 GiB |
| Reduction | 13.0674105% |
| Tile size | 16 KiB |

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
  while it is occupied; a competing completion receives HTTP 503 instead of
  waiting in the listener backlog.
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

The opt-in routed-prefill harness is functionally qualified: exact output and
I/O, private 92-layer captures, deterministic commands, and QD2 occupancy all
passed. Four consecutive 512-position arms had 4.766% expert-pipeline spread,
so the former 5% event-scheduler gate is unresolvable under the required
three-times-noise rule and no ABBA was run. QD2 was fully empty for only 0.216%
of routed-stream wall. Event scheduling is parked rather than promoted.

Current open work:

- engine/server activation of durable exact-prefix checkpoints for displaced
  or restarted sessions; semantic-anchor recovery across deep history rewrites
  is closed by the 8/29,630 exact-common-prefix upper bound;
- derived static-Q8 startup acceleration;
- dual-device expert streaming after sufficient second-device capacity exists;
- broader quality qualification for the diagnostic KDA backend;
- filled-128K workload characterization.

See [CHANGELOG.md](CHANGELOG.md) for implemented changes and
[RELEASING.md](RELEASING.md) for release gates.

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
