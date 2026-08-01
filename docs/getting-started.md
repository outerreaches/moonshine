# Getting started

## Qualified system

The initial reproducible target is:

- AMD Ryzen AI Max+ 395 / Radeon 8060S;
- ROCm target `gfx1151`;
- 128 GB unified memory;
- Ubuntu 24.04;
- Linux 7.0;
- ROCm 7.2;
- Samsung 990 PRO on ext4 with `noatime`;
- official Kimi K3 revision
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.

The code may compile elsewhere, but no other platform is currently qualified.

## 1. Check the host

Confirm that HIP sees the expected device and that the compiler supports the
target:

```sh
/opt/rocm/bin/hipcc --version
rocminfo | grep -E 'gfx1151|Radeon 8060S'
uname -r
```

The accepted full engine needs an otherwise idle 128 GB host. Before any
full-residency test:

```sh
free -h
swapon --show
pgrep -a -f 'llama|ds4|k3'
```

Do not run a second large model process concurrently. The engine performs a
CMA-aware memory preflight, but that is a last guard, not a substitute for
stopping other inference services.

The model filesystem must support aligned direct I/O:

```sh
findmnt -no SOURCE,FSTYPE,OPTIONS /path/to/model
```

ext4 with `noatime` is the tested configuration. Network filesystems and
copy-on-write layers are unqualified.

## 2. Install build dependencies

Install ROCm 7.2, including HIP, hipBLAS, and hipBLASLt development files.
Also install:

- a C99 compiler;
- GNU Make;
- binutils (`ar`);
- ICU development files (`libicu-dev` on Ubuntu);
- Git;
- the Hugging Face CLI if the model is not already present.

No Python, Transformers, or external tensor framework is required to build or
execute the engine and tokenizer.

## 3. Download the pinned checkpoint

Reserve at least 1.6 TB of local SSD space:

```sh
hf download moonshotai/Kimi-K3 \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --local-dir /path/to/moonshotai__Kimi-K3
```

The tested tree contains 96 SafeTensors shards and totals about 1.454 TiB.
Interrupted downloads should be resumed into the same directory rather than
restarted into a second model-sized tree.

## 4. Build

From the repository root:

```sh
make
make tests
```

Override the defaults when ROCm is installed elsewhere:

```sh
make \
  ROCM_HOME=/opt/rocm \
  HIPCC=/opt/rocm/bin/hipcc \
  ROCM_ARCH=gfx1151
```

## 5. Run model-free tests

```sh
make test
```

This runs the cache-policy test and device-side primitive/oracle tests. It does
not open model weights or allocate the full residency.

## 6. Validate the model and bounded components

Set the model location once in the shell:

```sh
export MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

Validate the exact shard/tensor layout and the payload-free prefill ledger:

```sh
make test-model-layout MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

Then run real-weight component oracles:

```sh
make test-model-components MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

These tests read selected tensors and expert spans but do not retain the full
engine.

## 7. Verify tokenizer and text-only XTML parity

```sh
make test-tokenizer MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

This reads only `tiktoken.model`. It checks exact official English,
multilingual, special-token, thinking/non-thinking, and multi-turn prompt
oracles. It does not read model weights or initialize ROCm.

## 8. Allocate the accepted engine

The Q8/32 engine retains about 104.6 GiB before driver/allocator overhead.
Ensure approximately 120 GiB is available and swap has not begun growing:

```sh
free -h
make test-engine-init MOONSHINE_MODEL="$MOONSHINE_MODEL"
make test-engine-init \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" \
  MOONSHINE_CONTEXT=131072
```

Expected default 8K ledger:

```text
resident static: 59,345,729,536 bytes
expert cache:    51,659,145,216 bytes
runtime state:      987,758,592 bytes
staging:            280,821,760 bytes
```

At 128K, runtime state is 4,455,923,712 bytes (4.150 GiB), and the
static/cache/state/staging ledger totals 115,741,620,224 bytes (107.793 GiB)
before allocator and driver overhead. Start with at least 120 GiB
`MemAvailable`, stop other large model processes and storage transfers, and
confirm that swap is not actively growing.

The test also runs and hashes every layer. A preflight refusal is a safe
failure. Do not bypass it or retry BF16 residency on a 128 GB host.

## 9. Run end-to-end greedy decode

```sh
make test-engine-hello MOONSHINE_MODEL="$MOONSHINE_MODEL"
make test-engine-hello \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" \
  MOONSHINE_CONTEXT=131072
```

The fixture embeds a tokenizer-verified rendering of `Say hello.`, walks the
complete graph, verifies the expected token sequence, and reports startup,
prompt, decode, and cache statistics.

The 128K fixture qualifies configured capacity with a short locked request.
It does not fill the context, measure filled-128K prefill, or establish
long-context quality.

## 10. Run native chat

Interactive:

```sh
./moonshine-chat "$MOONSHINE_MODEL"
```

One-shot:

```sh
./moonshine-chat "$MOONSHINE_MODEL" \
  --prompt "Say hello." \
  --max-tokens 32
```

The accepted run returned `Hello! 👋 How can I help you today?`, used the
exact 24-token non-thinking prompt, generated 18 tokens through the committed
end-of-message marker, and ended at position 42. Prompt and completion rates
were 0.439 and 0.500 tok/s.

Use `/save PATH` and `/load PATH` interactively, or `--save` and `--load`, for
exact state persistence. Checkpoint files encode conversation history and
should be kept private.

The equivalent full-residency regression target is:

```sh
make test-chat-hello MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

## 11. Run the OpenAI-compatible server

Start a loopback-only 128K-capacity service:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  --context 131072
```

Verify discovery:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

The OpenAI model ID is `moonshine`. Set `MOONSHINE_API_KEY` or use
`--api-key` before binding beyond loopback. This research server has one
blocking request slot.

Long prefill can take minutes. Configure client request/read timeouts
accordingly; use `curl --max-time 0` for manual checks. Streaming requests emit
SSE comment keepalives with token/layer progress during prefill.

The server preserves immutable expert-cache mappings between requests.
Append-only causal-state reuse is opt-in: send the same
`X-Moonshine-Session` value with consecutive requests and keep supplying the
complete OpenAI message history. Only the most recently completed session is
resident. An exact token-prefix extension reuses prior state; any mismatch or
different identifier performs a semantic reset and full prefill.

Use `--clear-expert-cache-per-request` only to reproduce cold-cache
benchmarks.

Function tools are supported through Chat Completions. Run the complete
declaration, call, result, and final-answer example in [Agentic API and tool
use](agentic-api.md). Keep the complete returned assistant message, including
`reasoning_content` and all `tool_calls`, and return one matching `role:
"tool"` message per call. K3 thinking is always enabled on the API;
`reasoning_effort` accepts `low`, `high`, or `max` and defaults to `max`.
Set `parallel_tool_calls: false` to force serial call turns; Moonshine prompts
for at most one call and rejects a multi-call result before emitting tool-call
SSE. Allow the default 256 completion tokens for prompts that require K3 to
plan an order across several operations.
Use `response_format: {"type":"json_object"}` for a validated top-level JSON
object, or `type: "json_schema"` for Moonshine's bounded typed
object/array/scalar subset. The schema vocabulary and wrapper are documented
in [Agentic API and tool use](agentic-api.md). Unsupported keywords fail
before inference. Structured SSE response content arrives only after complete
validation; reasoning still streams live.

To qualify client compatibility without loading K3, install the pinned
official Python SDK into an isolated environment and run the replay fixture:

```sh
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r tests/requirements-sdk.txt
make test-openai-sdk PYTHON=.venv-sdk/bin/python
```

With an 8K server already listening on port 18084, the optional real-model
gate is:

```sh
.venv-sdk/bin/python tests/qualify_openai_sdk_live.py
```

Neither command is part of the native runtime dependency set.

## 12. Qualify a reduction-schedule change

Before promoting any MXFP4, Q8, attention, or reduction-order change, run the
single self-hosted bundle:

```sh
make test-reduction-qualification \
  MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

This combines the model-free scalar/vector MXFP4 envelope with component
oracles, real expert and MoE gates, complete routed-layer hashes, the exact
tokenizer/XTML fixture, and the locked end-to-end hello. Cross-schedule hashes
are not required to match; each schedule must satisfy its declared numerical
envelope and the promoted schedule must pass its own exact full-engine gates.

## 13. Run prefill fixtures

Exact sequential/range comparison:

```sh
make test-prefill-2 MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

Warm-cache, bit-exact sequential/selected crossover sweep:

```sh
make test-prefill-crossover \
  MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

Override `MOONSHINE_CROSSOVER_TOKENS` with a quoted space-separated list for
a shorter sweep. The qualified default keeps sequential prefill through 7
tokens and selects range prefill from 8 onward.

Default 512-token scale test:

```sh
make test-prefill-scale \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" \
  MOONSHINE_PREFILL_TOKENS=512
```

Default filled-8K test:

```sh
make test-prefill-scale \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" \
  MOONSHINE_PREFILL_TOKENS=8192
```

Graduated filled-context tests require matching token and configured-context
values. Qualify 16K before attempting 32K:

```sh
make test-prefill-scale \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" \
  MOONSHINE_CONTEXT=16384 \
  MOONSHINE_PREFILL_TOKENS=16384
```

The scale fixture currently accepts at most 32K tokens and 128K configured
context. A finite output alone is not a qualification: record the exact next
token/value, complete phase ledger, selected read union, memory/swap counters,
SSD thermals, and a separate long-context quality probe.

The accepted 16K default gate ends with token `6244`, value `26.875`, and
7.929 tok/s. See [filled-context qualification](qualification-filled-context.md)
before attempting the substantially longer 32K arm.

Diagnostic KDA hipBLAS filled-8K test:

```sh
make test-prefill-kda-blas \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" \
  MOONSHINE_PREFILL_TOKENS=8192
```

The 8K tests are long, high-memory runs. The selected-expert default path took
about 16.8 minutes; the historical full-store diagnostic path took about 13.1
minutes on the qualified machine and needs requalification with selected I/O.

## 14. Verify exact state persistence

Choose a local temporary directory with at least 1 GiB free:

```sh
make test-state-checkpoint \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" \
  MOONSHINE_STATE_DIR=/tmp
```

This is a full-residency test. It exports state after two tokens, destroys the
source engine, creates a fresh engine, exercises four invalid-file cases, and
then proves exact imported continuation for three tokens. A passing run ends
with:

```text
K3 state checkpoint: PASS
  format=1 position=2 file=433.569 MiB
  export=1.043 s import=0.841 s
  exact continuation IDs: 414 19180 6949
```

The fixture removes its checkpoint on success. Application checkpoint files
should be treated as private conversation state: they contain no model
weights, but they encode the processed token history. CRC64 detects accidental
corruption and is not an authentication mechanism.

## Troubleshooting

### Residency preflight rejects the run

Stop other model processes and services, then recheck `MemAvailable`, swap,
and `/proc/meminfo` CMA fields. Do not reduce the guard or select BF16 merely
to force startup.

### `io_uring` or fixed-buffer registration fails

Confirm a recent Linux kernel, direct-I/O-capable local filesystem, sufficient
locked-memory allowance, and that a security policy has not disabled
`io_uring`.

### Direct reads fail near the end of a shard

Use the unmodified official 96-shard layout at the pinned revision. The loader
uses buffered fallback only where an aligned direct read would extend beyond
EOF.

### The build cannot find HIP or BLAS

Set `ROCM_HOME` and `HIPCC` explicitly. Verify that hipBLAS and hipBLASLt
development libraries match the selected ROCm installation.

### A different GPU compiles but produces wrong results

Treat it as a port. Run every model-free and component oracle before a full
engine test, and do not publish performance until the exact state and output
fixtures pass on that architecture.
