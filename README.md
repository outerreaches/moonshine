# Moonshine

Moonshine is an experimental inference engine for running the official Kimi
K3 SafeTensors checkpoint on a single 128 GB AMD Strix Halo system. It keeps
the non-routed model tier resident in ROCm memory, streams only selected
MXFP4 routed experts from NVMe, and exposes stateful one-token decode plus
layer-major prefill.

The name is a nod to Kimi's creator, Moonshot AI. Moonshine is an independent
community project and is not affiliated with or endorsed by Moonshot AI.

This is a research preview, not a general model runner. The current checkpoint
has a C API, exact causal-state persistence, executable
correctness/performance fixtures, a native chat client, and a one-slot
OpenAI-compatible HTTP service. It is deliberately narrow:

- Linux and ROCm only;
- tested on `gfx1151` with ROCm 7.2;
- the pinned official 96-shard Kimi K3 SafeTensors layout only;
- dynamically allocated context, capacity-qualified through 128K;
- greedy next-token inference;
- one large model process at a time.

The accepted Q8/32 configuration uses about 55.27 GiB for resident static
weights and 48.11 GiB for the online routed-expert cache. Runtime state is
0.920 GiB at 8K, 1.566 GiB at 32K, or 4.150 GiB at 128K, plus 0.262 GiB of
mapped staging. The 128K configuration accounts for about 107.793 GiB before
allocator/driver overhead. Fully resident BF16 is intentionally rejected on
the tested 128 GB machine.

## How the engine works

K3 has 93 transformer layers: one dense layer followed by 92 routed-MoE
layers, with 69 KDA and 24 gated MLA attention layers. Each routed layer
selects 16 of 896 experts.

The engine divides the checkpoint into two physical tiers:

```text
official 96-shard SafeTensors checkpoint
             |
             +-- static tier -> Q8/BF16 ROCm residency (~55.27 GiB)
             |     attention, routers, shared experts, norms, output head
             |
             `-- routed MXFP4 experts -> NVMe
                    |
                    +-- O_DIRECT + raw io_uring, QD2
                    +-- HIP-mapped fixed staging buffers
                    `-- 32-slot/layer online LRU cache (~48.11 GiB)
```

Decode is token-major. The engine walks all 93 layers, retains KDA recurrent
state, convolution state, compressed MLA cache, AttnRes state, and routed
expert-cache contents between tokens.

Prefill is layer-major. A token range visits the routed store once per layer
in physical shard/offset order and groups all rows selecting each expert. This
amortizes a roughly 1.447 TB routed sweep over the whole chunk. Cold prefill
can borrow an explicitly accounted portion of the empty decode cache for
batch workspace.

See [Architecture](docs/architecture.md) for the full graph, memory, I/O, and
correctness model.

## Current measured checkpoint

These are single-run results on an AMD Ryzen AI Max+ 395 / Radeon 8060S
(`gfx1151`), Ubuntu 24.04, Linux 7.0, ROCm 7.2, and a Samsung 990 PRO. The
startup, short-prompt, decode, and prefill rows were rerun from a detached clean
checkout of `c041205`; the remaining rows are accepted checkpoint fixtures.
They are engineering fixtures, not cross-project benchmark claims.

| workload | result |
|---|---:|
| Startup, Q8 static + 32 experts/layer | 38.1 s |
| Short 24-token sequential prompt | 0.438 tok/s |
| Post-TTFT greedy decode | 0.509 tok/s |
| Layer-major prefill, 512 positions | 2.145 tok/s |
| Layer-major prefill, 8,192 positions | 7.768 tok/s |
| Diagnostic KDA hipBLAS prefill, 8,192 positions | 10.448 tok/s |
| Causal-state export after 2 positions | 1.043 s |
| Causal-state import after 2 positions | 0.841 s |
| Native chat `Say hello.` prompt | 0.439 tok/s |
| Native chat `Say hello.` completion | 0.500 tok/s |
| OpenAI HTTP/SSE chat at 32K, prompt | 0.437–0.438 tok/s |
| OpenAI HTTP/SSE chat at 32K, completion | 0.499 tok/s |
| Locked chat at 128K capacity, prompt | 0.430 tok/s |
| Locked chat at 128K capacity, completion | 0.508 tok/s |
| OpenAI JSON chat at 128K capacity, prompt | 0.438 tok/s |
| OpenAI JSON chat at 128K capacity, completion | 0.498 tok/s |

The default range path matches the locked sequential state oracle exactly.
The faster KDA dequantize-plus-hipBLAS path preserves the tested greedy token
but changes selected values and causal-state hashes because its BF16 reduction
order differs. It remains diagnostic and opt-in pending broader quality tests.
The complete 128K capacity evidence and its limits are recorded in
[128K configured-context qualification](docs/qualification-128k.md).

## Requirements

- x86-64 Linux with a recent kernel supporting `io_uring` and `O_DIRECT`;
- an AMD ROCm device with enough shared/device-addressable memory;
- ROCm 7.2 with HIP, hipBLAS, and hipBLASLt development files;
- about 128 GB of system/unified memory for the accepted Q8/32 engine;
- about 1.6 TB of local SSD capacity for the official checkpoint;
- a local filesystem that supports aligned direct I/O (the tested setup uses
  ext4 with `noatime`);
- ICU 74 development files for the native Unicode pre-tokenizer;
- the official Kimi K3 checkpoint at revision
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.

Other AMD architectures, ROCm releases, filesystems, model revisions, memory
sizes, and contexts above 128K are not qualified yet. The 128K result covers
allocation, startup, a locked 24-token prompt/18-token completion, and the
OpenAI-compatible request path; filled-128K prefill latency and long-context
quality are not qualified. The official text config advertises 1,048,576
positions, but that is an architectural limit rather than a practical claim
for this 128 GB implementation.

## Build

Install a working ROCm toolchain, a C compiler, GNU Make, binutils, and ICU
development files. On the tested host:

```sh
/opt/rocm/bin/hipcc --version
rocminfo | grep gfx1151
make
```

`make` builds `libmoonshine.a`, `moonshine-chat`, and `moonshine-server`.
The architecture defaults to `gfx1151`; override it only for porting work:

```sh
make ROCM_ARCH=gfx1151
make tests
```

Run the model-free CPU and ROCm tests:

```sh
make test
```

Portable publication CI can run without ROCm or model weights:

```sh
make test-cpu
```

## Verify the native tokenizer and XTML renderer

The engine reads the official `tiktoken.model` directly and executes K3's
Unicode pre-tokenizer through ICU. No Python or Transformers runtime is used.

```sh
make test-tokenizer MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

The fixture compares exact official token IDs for English
case/contractions/numbers, Han text, combining marks, Hindi, Arabic, emoji,
and special-token handling. It also locks non-thinking, thinking, and
multi-turn XTML prompt hashes. The existing `Say hello.` prompt is reproduced
exactly at all 24 token IDs.

## Run the interactive client

Stop every other large model process first, then run:

```sh
./moonshine-chat /path/to/moonshotai__Kimi-K3
```

Enter one user message per line. The engine stays resident and retains causal
and expert-cache state across turns. Short prompts use token-major execution
through 92 tokens; prompts of 93 tokens or more use layer-major prefill. The
92.4-token crossover is derived from the measured
`n / 0.440 tok/s` sequential cost and
`203.638 s + 0.0682 s/token` layer-major cost rather than an arbitrary round
number.

One-shot mode writes response text to stdout and telemetry to stderr:

```sh
./moonshine-chat /path/to/moonshotai__Kimi-K3 \
  --prompt "Say hello." \
  --max-tokens 32
```

The locked native run produced:

```text
Hello! 👋 How can I help you today?
[prompt=24 token-major 54.682s 0.439 tok/s;
 generated=18 35.973s 0.500 tok/s;
 finish=end_of_message; forced=0; position=42]
```

Interactive `/save PATH` and `/load PATH` commands persist or restore exact
semantic state. `--system`, `--load`, and `--save` are also available; run
`./moonshine-chat --help` for the complete interface.

Use a larger qualified context without changing the weight residency:

```sh
./moonshine-chat /path/to/moonshotai__Kimi-K3 --context 131072
```

## Run the OpenAI-compatible API

The server keeps one Q8/32 engine resident and processes one request at a time.
Requests are semantically stateless by default: the engine zero-resets causal
state and renders the complete supplied text history. It preserves the routed
expert cache across requests because the cache contains immutable model
weights, not conversation state.

Append-only prefix reuse is opt-in through `X-Moonshine-Session`. The server
retains one completed session—the most recently successful identifier—and
reuses its in-process causal state only when the next rendered history is an
exact token-prefix extension. Edited, forked, shorter, mismatched, missing, or
different-session histories fall back to an isolated semantic reset and full
prefill. Clients must still send the complete OpenAI message history.

Start a loopback-only 128K-capacity service:

```sh
./moonshine-server /path/to/moonshotai__Kimi-K3 \
  --host 127.0.0.1 \
  --port 8080 \
  --context 131072
```

The initial surface provides `GET /health`, `GET /v1/models`, and
`POST /v1/chat/completions`. Both ordinary JSON and incremental SSE responses
are supported:

```sh
curl --max-time 0 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "moonshine",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_completion_tokens": 32,
    "stream": false
  }'

curl --max-time 0 -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "moonshine",
    "messages": [{"role": "user", "content": "Say hello."}],
    "stream": true
  }'
```

Layer-major TTFT can be many minutes. Configure OpenAI SDK read/request
timeouts accordingly; the examples use curl's unlimited timeout. Streaming
responses send spec-legal SSE comment keepalives during prefill, with token or
layer progress throttled to ten seconds and emitted at completed-unit
boundaries. Ordinary JSON responses cannot send keepalives.

To reuse an append-only conversation, send the same identifier on consecutive
requests:

```sh
curl --max-time 0 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Moonshine-Session: agent-1' \
  --data-binary @request-with-complete-history.json
```

Non-streaming responses report evaluated and reused prompt tokens in
`X-Moonshine-Prompt-Evaluated-Tokens` and
`X-Moonshine-Prompt-Reused-Tokens`. SSE streams report the same values in a
final comment. Use `--clear-expert-cache-per-request` only for explicit cold
cache benchmarks; it also disables prefix reuse for that request.

In the live 8K integration check, an appended 67-token second turn reused 42
tokens and evaluated only the 25-token suffix in 54.742 seconds. Replaying all
67 tokens sequentially would have taken about 152 seconds at the measured
short-prompt rate. Preserving only the expert cache improved repeated short
stateless prompts by about 0.8%; exact causal-prefix reuse is the material
agentic latency optimization.

Set `MOONSHINE_API_KEY` or pass `--api-key` to require bearer authentication.
The server refuses a non-loopback bind without a key. It accepts text-only
system/developer, user, and assistant history. The engine is deterministic
and greedy; tools, tool calls/results, structured response formats, sampling,
parallel slots, and HTTP request chunking are not implemented yet.

## Obtain the model

Weights are not included. Their license and use terms are controlled by the
[official Kimi K3 model repository](https://huggingface.co/moonshotai/Kimi-K3),
not by this code repository.

With the Hugging Face CLI installed, download the pinned revision to a local
SSD:

```sh
hf download moonshotai/Kimi-K3 \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --local-dir /path/to/moonshotai__Kimi-K3
```

The engine expects all 96 `model-*.safetensors` shards and the original tensor
names/layout. Validate the directory without allocating the full engine:

```sh
make test-model-layout MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

See [Getting started](docs/getting-started.md) for host checks, component
tests, full-residency safeguards, and benchmark commands.

## Run the locked end-to-end fixture

Stop every other large inference process first. Confirm at least 120 GiB is
available and that swap is idle:

```sh
free -h
pgrep -a -f 'llama|ds4|k3'
make test-engine-hello MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-engine-hello \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_CONTEXT=131072
```

This loads the accepted Q8/32 residency and runs a hard-coded, tokenizer-
verified `Say hello.` chat fixture. A passing run should end with:

```text
K3 hello: PASS
```

The fixture reports token IDs, startup time, prompt/decode rate, and cumulative
expert-cache hits. It does not decode arbitrary user text.

## Run layer-major prefill

The two-token oracle compares range execution with sequential execution,
including KDA, convolution, MLA, and AttnRes state hashes:

```sh
make test-prefill-2 MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

Run the scale fixture:

```sh
make test-prefill-scale \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_PREFILL_TOKENS=512
```

The maximum tested 8K run takes roughly 13–18 minutes depending on backend.
The faster backend is diagnostic:

```sh
make test-prefill-kda-blas \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_PREFILL_TOKENS=8192
```

## Save and restore causal state

The version-1 state file contains KDA recurrent and convolution state,
occupied MLA rows, AttnRes state, and the token position. It binds that
payload to the exact model tensor layout and static precision mode. Derived
MLA packed keys and the expert cache are deliberately excluded.

Run the exact continuation and corruption fixture:

```sh
make test-state-checkpoint \
  MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3 \
  MOONSHINE_STATE_DIR=/tmp
```

The two-position fixture writes a 433.569 MiB file. Import validates the
complete header, layout identity, file size, and CRC64 payload before it
modifies device state. It then proves bit-exact continuation through three
additional generated tokens on a fresh engine with an empty expert cache.
The same test rejects header corruption, a checksum-valid stale model
identity, payload corruption, and truncation without mutating the destination
engine.

## Project status

Versioned causal-state export/import, the official native tokenizer, the
text-only XTML chat renderer, deterministic stateful CLI, dynamic context
allocation capacity-qualified through 128K, and one-slot OpenAI-compatible
HTTP/SSE service are now locked. Filled-128K workload behavior is still an
open qualification item. Tool declarations/calls/results and structured
responses remain required for the broader agentic surface.

See [Architecture](docs/architecture.md) for the correctness model and
[Provenance](docs/provenance.md) for code lineage, references, and
acknowledgements.

## Contributing and security

Before contributing, read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of conduct](CODE_OF_CONDUCT.md). Report vulnerabilities through the
private process in [SECURITY.md](SECURITY.md), not a public issue.

Release history and maintainer qualification steps are in
[CHANGELOG.md](CHANGELOG.md) and [RELEASING.md](RELEASING.md).

## License

The code is MIT licensed. Kimi K3 model weights are separate and are not
redistributed here. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Development disclosure

This work was developed with strong AI coding and review assistance, with the
human maintainer directing architecture, experiments, validation, and release
decisions.
