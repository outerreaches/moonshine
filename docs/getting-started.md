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
- Zstandard development files (`libzstd-dev` on Ubuntu);
- Git;
- the Hugging Face CLI if the model is not already present.

No Python, Transformers, or external tensor framework is required to build or
execute the engine and tokenizer. The optional MZG transcoder requires Python
and the `zstandard` package.

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

### Optional: transcode routed experts to MZG1

MZG1 keeps every official SafeTensor intact and adds an opt-in
`expert-store-mzg1` containing only routed experts. Budget about 1.2 TiB beside
the 1.454 TiB source tree. The verified full store is a research artifact:
despite 13.120% fewer bytes and exact outputs, CPU decode makes end-to-end
decode slower, so production remains on raw SafeTensors.

```sh
python3 -m pip install zstandard
make test-mzg-transcoder
./tools/transcode_mzg.py \
  --model /path/to/moonshotai__Kimi-K3 \
  --out /path/to/moonshotai__Kimi-K3/expert-store-mzg1 \
  --jobs 6 --shard-jobs 2
```

The transcoder writes only to `expert-store-mzg1.partial`, verifies each block
immediately, performs a second full storage pass, writes SHA-256 sidecar
identities, and renames the root only after all 82,432 experts pass. An
interrupted run is resumable: rerun the same command and finalized sidecars in
the partial root are revalidated before remaining shards continue.

After completion, validate the native reader:

```sh
make tests/test_k3_mzg_store
MOONSHINE_EXPERT_STORE=auto \
  ./tests/test_k3_mzg_store /path/to/moonshotai__Kimi-K3
```

Unset or `MOONSHINE_EXPERT_STORE=off` disables legacy MZG1.
`MOONSHINE_EXPERT_STORE=auto` selects its standard subdirectory; an absolute
value selects a nondefault MZG1 store. Startup reports `experts=mzg1` when
selected. A selected incomplete/corrupt store fails closed. When both MZG1 and
MZG2 selectors are unset/`off`, the engine uses raw SafeTensor experts.

### MZG2 production GPU decoder

MZG2 keeps the official SafeTensors authoritative and adds a fully verified
derived expert store. The qualified store reduces routed bytes by 13.0674105%
and improves engine hello, selected prefill, and live 128K/30 by approximately
10.6%, 11.6%, and 11.7--12.8% with exact outputs. Budget about 1.18 TiB.

```sh
make tools/transcode_mzg2_layer
./tools/transcode_mzg2_full.py \
  --model /path/to/moonshotai__Kimi-K3 \
  --out /path/to/moonshotai__Kimi-K3/expert-store-mzg2 \
  --jobs 24

MOONSHINE_EXPERT_STORE=off \
MOONSHINE_MZG2_STORE=/path/to/moonshotai__Kimi-K3/expert-store-mzg2 \
  make test-engine-hello MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

The orchestrator writes only to `expert-store-mzg2.partial`, atomically
publishes verified per-layer sidecars, records SHA-256 identities for
resumption, independently decodes every stored expert against canonicalized
SafeTensors, and renames the root only after all 82,432 experts pass.
Interrupted runs rehash completed sidecars before continuing. Startup reports
`experts=mzg2`. The selector must be `off`, unset, or an absolute
sidecar/directory path; MZG1 and MZG2 selection together fails closed.
Unset/`off` retains raw SafeTensors as the rollback path.

### Offline static-Q8 compression screen

The model-free codec tests and CLI self-test do not open weights or use ROCm:

```sh
make test-static-q8-screen
```

The complete screen quantizes the pinned 1,135 eligible matrices with the CPU
Q8/128 reference, validates the exact 51.370 GiB ledger, and measures
independent 16/32/64 KiB value/scale Zstd payloads:

```sh
make tools/screen_static_q8
./tools/screen_static_q8 \
  /path/to/moonshotai__Kimi-K3 \
  /path/to/static-q8-screen.json
```

This reads approximately 100 GiB from the model. Run it only in a maintenance
window with no latency-sensitive model or competing modelstore traffic. It
does not create a resident store, change the source checkpoint, or use the
GPU. CPU-generated Q8 bytes remain a screening input until a later
full-corpus GPU-quantizer identity gate passes.

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
  --context 131072 \
  --experts 30 \
  --max-output-tokens 65536
```

Verify discovery:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

The OpenAI model ID is `moonshine`. Health and discovery include the configured
`context_length`, effective `max_output_tokens`, and one-slot availability.
Set `MOONSHINE_API_KEY` or use `--api-key` before binding beyond loopback. The
research server has one inference slot, but its independent control thread
keeps health/model discovery responsive while busy. A competing completion is
rejected with HTTP 503 and `Retry-After: 1`; it is never queued behind the
active request.

The output ceiling defaults to 8,192 and can be raised to 65,536 with
`--max-output-tokens`. It cannot exceed the configured context. The Chat
Completions request may use `max_completion_tokens` or legacy `max_tokens`;
explicit `null` is treated as unspecified. Without a value, the request
defaults to 256 tokens or the smaller server ceiling. This is a maximum, not a
forced response length: natural stop and remaining context can end generation
earlier. The 64K setting is qualified for admission, remaining-context
clamping, and repeat short requests in a persistent 128K/30-expert process; it
does not claim qualification of a continuous 64K decode.

The 30-slot setting is deliberate for a persistent 128K service on the
qualified 128 GB host. A cold first request can borrow the empty expert cache
as its prefill workspace, but later requests retain warmed cache entries and
need a separate workspace. The original 32-slot configured-capacity test does
not leave reliable CMA-plus-guard headroom for that second allocation. Two
independent live requests passed at 128K with 30 slots and a 45.104 GiB cache.
Use the faster qualified 32-slot default at smaller contexts.

Standard operating profiles are:

| use | context | experts/layer | output ceiling |
|---|---:|---:|---:|
| agentic/API baseline | 8,192 | 32 | 8,192 |
| filled/natural-text qualified | 16,384 | 32 | 16,384 |
| filled/natural-text qualified | 32,768 | 32 | 32,768 |
| persistent maximum capacity | 131,072 | 30 | 65,536 |

The 128K row is qualified for configured capacity and repeat short requests,
not a filled 128K prompt. See
[Deployment profiles and Hermes Agent](deployment-profiles.md) for exact
commands and qualification boundaries.

Long prefill can take minutes. Configure client request/read timeouts
accordingly; use `curl --max-time 0` for manual checks. Streaming requests emit
ten-second SSE comments during prefill and quiet decode regions. Disconnects
cancel at the next complete token/layer boundary; causal state resets to zero
while immutable expert-cache mappings remain available.

The server preserves immutable expert-cache mappings between requests.
Append-only causal-state reuse is automatic while the next request extends the
one retained exact token prefix. Keep supplying the complete OpenAI message
history; no custom session header is needed. A mismatch performs a semantic
reset and full prefill. Read the standard
`usage.prompt_tokens_details.cached_tokens` value to confirm reuse.

Use `--clear-expert-cache-per-request` only to reproduce cold-cache
benchmarks.

Function tools are supported through Chat Completions. Run the complete
declaration, call, result, and final-answer example in [Agentic API and tool
use](agentic-api.md). Keep the complete returned assistant message, including
`reasoning_content` and all `tool_calls`, and return one matching `role:
"tool"` message per call. K3 thinking is always enabled on the API;
`reasoning_effort` accepts `low`, `medium`, `high`, or `max` and defaults to
`max`.
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

For Hermes Agent against the 128K/64K 0.2.0 profile, set
`model.max_tokens: 65536`, a matching `model.context_length: 131072`, and
`agent.reasoning_effort: medium` in `~/.hermes/config.yaml`. Older binaries
and smaller server profiles require a lower output cap; confirm the effective
limits through discovery before testing. Use `HERMES_API_TIMEOUT` and
`HERMES_STREAM_READ_TIMEOUT` of at least 1,800 seconds for ordinary agentic
use. The qualified first Hermes turn took 663.7 seconds end to end, and an
earlier attempt was still inside the API call when manually interrupted after
14 minutes 49 seconds. Begin at 7,200 seconds for deliberately long 16K/32K
prompts. Disable Hermes automatic title
generation or send it to a separate fast auxiliary model so its independent
30-second request path does not occupy Moonshine's only slot or replace the
retained conversation prefix. The
complete example and failure map are in [Deployment profiles and Hermes
Agent](deployment-profiles.md).

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

Use `--base-url` and `--api-key` when qualifying an already running protected
deployment.

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

For a content-free routed-prefill instrumentation arm, use a new private prefix
outside the checkout:

```sh
MOONSHINE_PREFILL_DIAGNOSTICS=/private/run/baseline-a \
make test-prefill-scale \
  MOONSHINE_MODEL=\"$MOONSHINE_MODEL\" \
  MOONSHINE_PREFILL_TOKENS=8192
```

One arm produces `/private/run/baseline-a.prefill.csv` with 92 layer rows.
First collect warm-up plus four fresh-process noise baselines and run:

```sh
python3 tools/analyze_prefill_screen.py \
  --noise-only \
  --metric expert_pipeline_seconds \
  --gate-pct 5 \
  --output /private/run/noise.json \
  warmup.prefill.csv noise-1.prefill.csv noise-2.prefill.csv \
  noise-3.prefill.csv noise-4.prefill.csv
```

Stop if the result is `UNRESOLVABLE`. A `READY_FOR_ABBA` result permits four
more fresh-process arms—baseline A, candidate A, candidate B, baseline B—and
the same command without `--noise-only`, passing all nine files in that order.
The analyzer refuses route/I/O drift and gates tighter than three times the
four-run noise spread. Collect NVMe temperature and throttle state beside every
arm; those host measurements are intentionally not read by the engine.

Graduated filled-context tests require matching token and configured-context
values:

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
7.929 tok/s. The accepted 32K gate ends with token `40493`, value `28.25`, and
7.462 tok/s; reproduce it by changing both values above to `32768`. See
[filled-context qualification](qualification-filled-context.md) before
attempting the substantially longer arm.

Run semantic retrieval separately from the repeated-token performance gate:

```sh
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

The qualified default renders 15,993 tokens, retrieves three fixed values
from early, middle, and late prompt positions, and requires the exact response
`saffron|7319|Nivens`. A 512-token staged arm is available through
`MOONSHINE_RETRIEVAL_TARGET=512`. The qualified 31,999-token arm uses
`MOONSHINE_RETRIEVAL_TARGET=32000`; it expands to 781 records and requires the
same exact response.

Profile model-shape MoE-tail components without loading weights:

```sh
make test-moe-tail-profile
```

This is a diagnostic timing/numerical envelope, not a production-backend
selection test.

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

The durable exact-prefix correctness fixture is:

```sh
make test-prefix-checkpoint \
  MOONSHINE_MODEL=\"$MOONSHINE_MODEL\"
```

It creates a private temporary checkpoint root, publishes a completed turn,
proves ordinary live reuse remains first priority, displaces the session with
an unrelated request, restores the exact checkpoint, restarts the session, and
requires identical response bytes and all four state digests. The fixture
passed. Prefix restoration improved 65.77× (63.008 s inferred evaluation
versus 0.958 s import), clearing the proposed 5× gate; total prompt wall
improved 2.87× (95.174 → 33.124 s). The mechanism passes those gates.

For manual server qualification only, use an absolute private root plus explicit
limits:

```sh
./moonshine-server \"$MOONSHINE_MODEL\" \
  --prefix-checkpoint-root /private/moonshine-checkpoints \
  --prefix-checkpoint-entries 4 \
  --prefix-checkpoint-bytes 21474836480
```

The first production canary passed exact displacement/restart recovery but an
immediate follow-up received HTTP 503 while post-response checkpoint export
retained the request slot for 2.097 seconds. Candidate `999d6e8` now admits
one bounded next-turn request during terminal checkpoint export. Its
model-backed fixture, non-streaming handoff gate, live/displaced/restart
recovery, and streaming handoff gate passed. Production remains disabled until
the full fresh-root release gate is repeated.

The non-streaming handoff gate uses:

```sh
MOONSHINE_API_KEY='<private>' \
  tools/qualify_checkpoint_handoff.py \
  --request /private/handoff-request.json \
  --output /private/handoff-result
```

The streaming handoff gate uses the same private request/output contract with
`stream:true`:

```sh
MOONSHINE_API_KEY='<private>' \
  tools/qualify_checkpoint_stream_handoff.py \
  --request /private/stream-handoff-request.json \
  --output /private/stream-handoff-result
```

Both gates require the first immediate completion to wait rather than receive
HTTP 503, a second contender to receive `server_busy`, responsive health, and
an idle final state. They store responses privately and emit only
content-free measurements in `result.json`.

The full negative-path release harness is:

```sh
MOONSHINE_API_KEY='<private>' \
  tools/qualify_checkpoint_negative.py \
  --mode disconnect|publication-failure|shutdown \
  --request /private/request.json \
  --output /private/result \
  --checkpoint-root /private/checkpoints
```

The qualified candidate passed all three modes. A fresh-root activation review
and the post-activation observation window remain separate from this release
qualification.

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
