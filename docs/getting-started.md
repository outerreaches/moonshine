# Getting started with the K3.1 research preview

This guide covers the implemented `0.3.0-k3.1-research-preview`. Synthetic
CPU/ROCm gates, full MZG2 conversion/verification, the real 8K Beelink canary,
live API/tool/structured-output checks, state-v2 recovery gates, the official
SDK 2.52.0 fixture, and natural-text retrieval through 63,971 tokens pass.
No context above 64K or production serving has been qualified.

## 1. Confirm the development host

The ROCm synthetic gate ran on:

- AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`);
- 128 GB unified memory;
- x86-64 Linux with `io_uring` and `O_DIRECT` support;
- ROCm HIP, hipBLAS, and hipBLASLt development files.

The documented real K3.1 canaries ran on Beelink; this is not a general host
support claim. Other devices and ROCm releases are unverified. Check the visible
device and compiler:

```sh
/opt/rocm/bin/hipcc --version
rocminfo | grep -E 'gfx1151|Radeon 8060S'
uname -r
```

If later attempting a full-residency gate, stop competing model processes and
record memory/swap state first. Do not reuse historical K3 memory ledgers as
K3.1 capacity evidence.

## 2. Install build and offline-tool dependencies

Install ROCm and the native build dependencies:

```sh
sudo apt install build-essential binutils libicu-dev libzstd-dev python3
```

The engine and tokenizer do not require Python, Transformers, or an external
tensor framework at runtime. Python 3 is used by the offline checkpoint
inspector and placement planner. The Hugging Face CLI is needed only to
download the model. Install it in an isolated environment before downloading:

```sh
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install huggingface_hub
. .venv-hf/bin/activate
```

## 3. Download the exact checkpoint

Moonshine accepts only the pinned K3.1 Pro configuration and layout:

```text
id:          moonshotai/Kimi-K3.1-Pro
revision:    6d0a0202956796543364d666ab19c6ad1a494ee1
architecture: KimiK31ForConditionalGeneration
model type:  kimi_k31
shards:      96
```

Download it without substituting a branch name or another revision:

```sh
hf download moonshotai/Kimi-K3.1-Pro \
  --revision 6d0a0202956796543364d666ab19c6ad1a494ee1 \
  --local-dir /path/to/moonshotai__Kimi-K3.1-Pro
```

The weights are governed by the upstream license and are not included in this
repository. An interrupted download should resume into the same private model
directory.

## 4. Inspect the checkpoint offline

Set the model path:

```sh
export MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3.1-Pro
```

Metadata-only inspection validates the configuration, index, shard coverage,
SafeTensors headers, tensor coverage, and routed-expert geometry without reading
all tensor payloads:

```sh
python3 tools/k31_checkpoint.py "$MOONSHINE_MODEL" \
  > /path/to/private/k31-source-manifest.json
```

For full-file SHA-256 identities, run:

```sh
python3 tools/k31_checkpoint.py "$MOONSHINE_MODEL" \
  --verify-sha256 \
  > /path/to/private/k31-source-manifest-full.json
```

The second command streams the complete checkpoint. Keep manifests private when
host paths or deployment metadata are added around them.

The full pinned-source MZG2 conversion and verification passed for all 92 routed
layers and 82,432 experts. To reproduce the resumable standalone bundle build
and independent verification:

```sh
export MOONSHINE_BUNDLE=/path/to/private/Kimi-K3.1-Pro-Moonshine-MZG2
python3 tools/build_mzg2_bundle.py build \
  --model "$MOONSHINE_MODEL" --out "$MOONSHINE_BUNDLE"
python3 tools/build_mzg2_bundle.py verify \
  --bundle "$MOONSHINE_BUNDLE" --model "$MOONSHINE_MODEL"
```

Keep the source revision and bundle manifest bound together. A historical K3
bundle or manifest is not valid K3.1 input.

## 5. Build

From the repository root:

```sh
make
```

Override the ROCm locations only when needed:

```sh
make \
  ROCM_HOME=/opt/rocm \
  HIPCC=/opt/rocm/bin/hipcc \
  ROCM_ARCH=gfx1151
```

A successful build establishes compilation only.

## 6. Run the verified synthetic gates

These are the exact commands run for the K3.1 cutover:

```sh
make -j4 test-cpu
make -j4 test
```

Observed results:

| gate | observed result | boundary |
|---|---|---|
| `make -j4 test-cpu` | PASS | portable/model-free suites, including K3.1 config, checkpoint inspector, placement planner, DSA memory, and CPU DSA oracle |
| `make -j4 test` | PASS on Radeon 8060S `gfx1151` | model-free ROCm and API/component suites, including DSA GPU tests |

The CPU DSA oracle also passed ASan/UBSan. GPU DSA coverage includes key
LayerNorm and scoring, stable top-k at boundaries and ties, negative-infinity
padding, sparse attention, and decode/prefill equivalence.

The tests use synthetic tensors/fixtures. They do not open the full official
checkpoint or establish correct K3.1 text generation.

The individual offline-tool fixture targets are also available:

```sh
make test-k31-checkpoint
make test-placement
```

Both are already included in `test-cpu`.

## 7. Understand the DSA and state formats

The implemented K3.1 attention layout has 69 KDA layers and 24 Gated DSA
layers. DSA uses 32 indexer heads of width 128, key LayerNorm, F32 scores, and
at most 2,048 selected causal keys. Top-k ties select the lower token ID first.

The persistent decode/prefill index-key cache stores **BF16** elements. Official
FP8 and MXFP4 index-cache representations are not implemented or qualified.
Do not use memory estimates based on those compressed formats.

State format v2 adds occupied DSA index-key rows to the occupied MLA rows and
other causal state. V1 files are rejected. See
[Architecture](architecture.md#state-format-v2) before attempting to migrate or
reuse an older checkpoint.

## 8. Plan sidecar placement offline

The placement planner consumes an explicit
`moonshine-sidecar-profile-v1` JSON document containing model/store identities,
whole-layer files, and drive capacities/reserves/bandwidths. It performs no
device or filesystem discovery and writes only the requested output manifest:

```sh
python3 tools/moonshine_placement.py /path/to/profile.json \
  --output /path/to/private/placement.json
```

Use `-` as the profile path to read JSON from stdin; omit `--output` to write the
manifest to stdout. Synthetic capacity, identity, determinism, and placement
fixtures pass. This is not evidence that a K3.1 sidecar store exists or that a
multi-drive deployment works.

## 9. API and client contract

The public model ID is `moonshotai/Kimi-K3.1-Pro`; `moonshine` is not a public
model alias. The server reports version `0.3.0-k3.1-research-preview`.

Requests are text-only. Image and audio content parts are rejected rather than
ignored. Reasoning is always enabled on the Chat Completions path;
`reasoning_effort` accepts `low`, `high`, or `max` and defaults to `max`.
Public `medium` requests are rejected even though the internal XTML renderer
retains that official format value.

The server implements health/model discovery, JSON and SSE completions,
function tools, preserved `reasoning_content`, bounded structured output, and
one-slot exact-prefix reuse. Synthetic parser/codec tests pass. Real generation,
tool calling/continuation, structured JSON, and medium/image/old-model HTTP 400
rejection passed in the 8K Beelink canary. The official OpenAI SDK 2.52.0
fixture also passed. These results do not qualify third-party agent clients or
production service.

See [Agentic API](agentic-api.md) and
[Deployment profiles](deployment-profiles.md) for request and client shapes.
Context/expert/output settings shown there are configuration examples, not
production profiles; only the explicitly documented canary and natural-context
runs are current K3.1 evidence.

## 10. Launch the verified bundle within the canary boundary

The real 8K loopback canary used this server command shape with the verified
standalone bundle:

```sh
./moonshine-server "$MOONSHINE_BUNDLE" \
  --host 127.0.0.1 --port 8080 \
  --api-key "$MOONSHINE_API_KEY"
```

This command completed real bundle startup, generation, exact prefix reuse,
live tool/structured-output, and state-v2 recovery checks. Do not broaden that
result into production qualification, disable identity checks, reduce memory
guards, substitute K3 weights, or select a different checkpoint revision.

Full K3.1 MZG2 conversion and verification passed for all 92 layers and 82,432
experts. Historical K3 MZG2 instructions and results do not apply to this
preview.

## 11. Optional official SDK fixture

The official Python SDK fixture passed with `openai 2.52.0`. To reproduce it in
an isolated environment:

```sh
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r tests/requirements-sdk.txt
make test-openai-sdk PYTHON=.venv-sdk/bin/python
```

This establishes protocol compatibility, not live quality or production
qualification. The SDK is a qualification-only dependency, not a native runtime
dependency.

## 12. Reproduce natural-context and DSA evidence

The natural-text gates use real weights and are separate from the synthetic
suites and short 8K canary. Each returned `saffron|7319|Nivens` exactly:

| prompt tokens | prefill | 17-token decode |
|---:|---:|---:|
| 15,993 | 6.545 tok/s | 0.478 tok/s |
| 31,999 | 5.438 tok/s | 0.462 tok/s |
| 63,971 | 4.397 tok/s | 0.454 tok/s |

Run each target independently against the pinned source or matching verified
bundle:

```sh
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=16000
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=32000
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=64000
```

The 63,971-token run took 93.976 seconds to start, 14,547.439 seconds to
prefill, and 37.485 seconds to decode 17 tokens. Reported prefill phases were
KDA 2,585.803 seconds, DSA 8,101.056 seconds, expert stream 3,463.059 seconds,
and MoE tail 248.327 seconds.

The optimized DSA implementation uses a BF16 index cache, identity-window
indexer/top-k skipping, dense MLA from the measured 768-row threshold, and one
selected-row gather above 2,048 keys. Reproduce its real-shape kernel screen:

```sh
make bench-k31-dsa
```

The accepted 8K sparse-to-dense comparison was 4.930 → 0.126 ms per layer;
the real 128-token canary ran at 0.501 tok/s. Named DSA failure steps identify
the failing operation. Large-window BLAS indexer scoring is implemented and
verified; at 8K it measures 1.094 → 0.075 ms and passes real retrieval.

## 13. Remaining qualification gates

- No context above 64K is qualified. Re-run correctness, memory, and natural-
  context gates independently before extending the current 63,971-token
  natural-text boundary.
- No production soak has run. Qualify the exact candidate and workload before
  making a production-readiness claim.
- The 128-token canary and 17-token long-context decodes do not qualify
  continuous long output or general performance.
- Further performance changes need fresh exact-output/state and comparative
  timing gates. The implemented BLAS indexer already passed the real 8K gate.
- Historical K3 output, state-size, context, SDK, MZG2, and timing fixtures are
  not transferable. See [RELEASING.md](../RELEASING.md).

## Troubleshooting

### Configuration is rejected

Confirm the full model ID/revision and use the unmodified official `config.json`.
The loader intentionally rejects K3 configurations, changed DSA geometry, and
conflicting attention-layer arrays.

### A v1 state file is rejected

This is required behavior. State v2 includes DSA index-key state and is not
backward-compatible with v1. Start a new exact causal state; do not strip or
relabel the version field.

### Image or audio input is rejected

The public server is text-only. Remove non-text content parts or use a different
multimodal serving path; Moonshine does not silently discard them.

### `reasoning_effort: medium` is rejected

Use public `low`, `high`, or `max`, or omit the field for default `max`. Internal
XTML support for `medium` does not make it a valid K3.1 API request.

### ROCm tests fail on another GPU

Treat that device as a port. Do not infer support from compilation or from the
Radeon 8060S synthetic result.
