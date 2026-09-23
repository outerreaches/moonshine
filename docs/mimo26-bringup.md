# MiMo V2.6 Flash bring-up

Text-only native backend for `XiaomiMiMo/MiMo-V2.6-Flash-RL`, revision
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`. Separate lane: the GLM backend,
its worker, library and defaults are unchanged.

Arithmetic decisions live in `mimo26-precision-contract.md`. This file covers
the artifact, its schema and the layout traps found while auditing it.

## Checkpoint

Local copy: `/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL`, 65
shards, 172,932,505,264 bytes (161.056 GiB) of indexed tensors. A second copy
exists on Sparky; every small file inspected on both hosts hash-matches, and
the indexed byte totals agree, but Sparky's shards are unhashed so whole-file
copy equality is unproven.

Shard names are `model_pp0_ep<N>_shard0.safetensors` plus
`model_mtp.safetensors`. This is **not** GLM's `model-NNNNN-of-NNNNN`
convention, so neither `k3_st_model_open` nor `k3_st_model_open_5digit_total`
can open this checkpoint. The shard set must be driven from
`model.safetensors.index.json`.

`dflash/` and `audio_tokenizer/` are sidecars outside the 65-shard index.
`dflash/config.json` has a trailing comma and fails strict JSON parsing;
the original stays unmodified, and any normalized copy must be explicit
and hashed.

## Graph

48 layers, hidden size 4096, vocab 152576 with untied `lm_head`. Layer 0 is
dense (intermediate 16384). Layers 1..47 are MoE: 256 routed experts, top-8,
no shared expert, `moe_intermediate_size` 2048.

Full-attention layers are 0, 5, 11, 17, 23, 29, 35, 41, 47 — nine of them,
with the other 39 using sliding-window attention. **This pattern is not
periodic**: the first gap is five, every later gap is six. Deriving it from a
period misclassifies layers 12, 24 and 36, whose tensors then fail geometry
validation. Read the explicit 48-entry `hybrid_layer_pattern`; the audit
asserts the pattern is not period-12 so the shortcut cannot creep back in.

Three MTP layers live in `model.mtp.layers.0..2` — a separate namespace that
does **not** continue the text layer numbering the way GLM's does. Their
attention is SWA-shaped and carries a sink; their MLP is dense.

## Tensor inventory

| Group | Count |
| --- | ---: |
| Main text | 382 |
| Routed experts | 72,192 |
| MTP | 48 |
| Vision | 364 |
| Audio encoder | 75 |
| Speech embeddings | 20 |
| Total | 73,081 |

72,192 is exactly 47 x 256 x 6, so the expert count is an exact assertion
rather than a lower bound. Vision and audio schemas are deliberately
unvalidated: the first release is text-only, so those groups are counted for
coverage and recognized as out of scope. Rejecting them outright would reject
the real checkpoint; ignoring them silently would hide a changed artifact.

## Layout traps

**The fused QKV is sharded by TP rank.** This is the trap that cost this lane
the most, and it was misdiagnosed twice before the arithmetic settled it.

The checkpoint's `metadata.tp_size` is 4. It was written by a tensor-parallel
deployment, so `qkv_proj.weight` is **4 rank slices concatenated, each
already de-interleaved to `[Q | K | V]`** — not `[all Q | all K | all V]`,
and not `num_kv_heads` groups of `[Q_g | K_g | V_g]` either. Rank `s` owns
query heads `[16s, 16s+16)` and KV heads `[s·kv/4, …)`, which preserves the
standard `repeat_kv` mapping.

The FP8 block scales tile each rank slice independently, padded up to a whole
block. That one rule reproduces both grids exactly, and nothing else does:

| Layer | Rank slice rows | Blocks | × 4 | Observed |
| --- | --- | --- | ---: | ---: |
| global | 16·192 + 1·192 + 1·128 = 3392 | 26.5 → 27 | 108 | `[108, 32]` |
| SWA | 16·192 + 2·192 + 2·128 = 3712 | 24 + 3 + 2 = **29** | 116 | `[116, 32]` |

Three readings have to be rejected to get here:

1. `[all Q | all K | all V]`, which the shipped `modeling_mimo_v2.py`
   assumes. Scrambles every head.
2. `num_kv_heads` groups of `[Q_g | K_g | V_g]`, which vLLM's
   `_shard_fp8_qkv_proj` **docstring** states. Correct only when a rank owns
   exactly one KV head — true for the 9 global layers, false for the 39 SWA
   layers. Its own code cannot run on the SWA shape: `116 // 8 = 14` scale
   rows expand to 1792 and multiply against an 1856-row group.
3. The earlier guess recorded here, that `[108,32]` was an over-provisioned
   grid left from a 13,824-row layout with V sized as if `head_dim` were 192,
   and that flat `row / 128` indexing stayed correct. Flat indexing is **not**
   correct for global layers: it misassociates the scales of every rank after
   the first, and rows 106-107 are per-rank padding, not stale live values.

A validator asserting `scale_rows == ceil(rows / 128)` rejects the real
checkpoint, and reading 108 as the QKV width mis-splits Q, K and V. Both
failure modes are covered by negative tests, and loading now fails closed
unless the row total is `tp` slices *and* the grid is `tp` times the padded
per-slice block count.

The general lesson, which applies to the next architecture too: **derive the
block-scale geometry from first principles and require it to come out
exactly.** Four numbers discriminated three readings that all produced
plausible-looking weights, and two of those readings came from reading source
code rather than bytes.

### Confirmed by upstream llama.cpp (2026-09-23)

An independent implementation now agrees. `conversion/mimo.py::_tp_aware_qkv_dequant`
in llama.cpp derives the same layout in its own words — ranks stacked as
`[Q_per | K_per | V_per]`, `ceil(rows_per_rank/128)` scale block-rows per rank,
"phantom rows not in the weight", and the same per-row scale index
`rank * blocks_per_rank + (rr // bs)`. It detects `tp` from the scale row count
where this lane pins 4 and fails closed; the two agree at tp=4.

The global layers make this a real check rather than two implementations
sharing a convention: a non-sharded reading needs 106 block-rows and the
checkpoint ships 108, which only per-rank `ceil` at tp=4 produces. The SWA
layers do **not** discriminate — 3712 rows per rank is an exact multiple of
128, so tp=1, 2 and 4 all yield 116 and the scale data cannot distinguish the
row permutation. Agreement on SWA row order rests on the convention plus the
behavioural evidence from bring-up, not on arithmetic.

That closes the deferred upstream-oracle item for the QKV layout. It does not
close output-quality comparison, which is a separate question; see the vault
note `MiMo V2.6 Flash RL Two-Node GGUF Bring-up 2026-09-23` for the two-node
GGUF recipe that makes a greedy-agreement test possible.

Every hyper-parameter the converter writes into the GGUF also matches what this
lane derived: per-layer KV head counts `[4,8,8,8,8,4,…]`, the non-periodic
sliding-window pattern, K 192 / V 128, `rope.dimension_count` 64, dual RoPE
theta 1e7 global / 1e4 SWA, `attention.value_scale` 0.707, 256 experts top-8,
expert FFN 2048, RMS eps 1e-6. Nothing had to be corrected.

**`quantization_config.ignored_layers` names 49 modules.** All 48 text
`o_proj` plus `model.decoder.self_attn.o_proj`, which has no counterpart in
`configuration_mimo_v2.py`. Surface it rather than skipping it.

**Expert shapes are packed byte extents, not logical shapes.** `gate_proj` and
`up_proj` are `[2048,2048]` bytes for a `[2048,4096]` logical matrix;
`down_proj` is `[4096,1024]` bytes for `[4096,2048]` logical. A complete expert
is 12.75 MiB including scales.

**The reference attention comment is stale.** It describes split Flash Q/K/V,
while the config and the actual tensors are fused QKV. Trust the tensors.

## Index

`model.safetensors.index.json` carries `metadata` with three keys:
`save_format: mxfp4`, `tp_size: 4`, and `total_size: 172923364096`. That
total is the **payload** sum — about 8.7 MiB below the sum of file sizes,
the difference being per-shard SafeTensors headers. `mimo26_manifest_reconcile`
checks the payload sum against it rather than against file sizes.

`save_format` is treated as part of the artifact identity: a checkpoint
claiming anything but `mxfp4` is rejected rather than probed.

## What is here

### Schema and artifact

- `mimo26_architecture.{c,h}` — payload-free schema: layer kinds from the
  explicit pattern, per-kind tensor contracts, exact group coverage.
- `mimo26_manifest.{c,h}` — index parse and validation, the index-driven
  shard opener, directory reconciliation, and the routed-expert span planner.
- `k3_st_model_open_paths` — additive opener taking an explicit shard list,
  since MiMo's filenames defeat the numbered family. Existing openers are
  unchanged.
- `tests/audit_mimo26_checkpoint.py` — read-only metadata audit;
  `tests/test_mimo26_audit.py` injects one synthesized fault per check.

### Numerics

- `mimo26_ops`, `mimo26_router`, `mimo26_attention`, `mimo26_kv`,
  `mimo26_weights` — the CPU contract, bit-exact against an independent
  NumPy path over 121M dequantized values and against the reference's own
  modules on real weights.
- `mimo26_rocm_ops.{cu,h}` — MiMo's GPU primitives. Forked from
  `k3_rocm_ops` only where the contracts genuinely differ, which is
  measured rather than assumed: K3's RMSNorm rounds once where MiMo rounds
  twice and differs on 3,197 of 12,288 elements. Where they agree the K3
  kernel is reused, as with the MXFP4 GEMV, which matched on all 8,192
  values checked.

### Execution

- `mimo26_layer`, `mimo26_worker` — the CPU reference worker.
- `mimo26_rocm_layer.{cu,h}` — the GPU layer, decode and layer-major
  prefill. Prefilling a chunk is gated as equal to decoding the same tokens
  one at a time.
- `mimo26_gpu_worker.{cu,h}` — the 48-layer GPU worker. Experts are cached
  **packed**, admitted through `k3_expert_cache` and read through K3's
  `io_uring` into mapped staging.

### Serving

- `mimo26_tokenizer.{c,h}` — byte-level BPE, chat template, tool rendering
  and parsing, streaming decode that never emits partial UTF-8. Exact
  against every in-scope pinned fixture.
- `mimo26_server_slot.{c,h}` — admission, cancellation, deadlines, fault
  quarantine, supervised restart, graceful drain. Pure logic with the clock
  passed in, so the timing-dependent behaviours are actually testable.
- `mimo26_server.cu` — `/health`, `/v1/models`, `/v1/chat/completions` with
  SSE, tools, and reasoning separated from content.

### Tools

`tools/mimo26_run`, `mimo26_gpu_run`, `mimo26_eval`, `mimo26_gpu_bench`,
`mimo26_context_gate`, `mimo26_server`, `mimo26_budget.py`, and
`tests/soak_mimo26_server.py`.

## Memory budget

From `tools/mimo26_budget.py` on this host: 124.94 GiB of RAM, 120.85 GiB
available. Static text is 8.2837 GiB, the SWA rings are a fixed 24.375 MiB
across 39 layers, and global KV costs 23,040 B/token across nine layers.

| Context | Global KV | Cache room | Experts cacheable | Share |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.176 GiB | 100.36 GiB | 8,060 | 67.0% |
| 65,536 | 1.406 GiB | 99.13 GiB | 7,961 | 66.2% |
| 1,048,576 | 22.500 GiB | 78.04 GiB | 6,267 | 52.1% |

At a 12 GiB reserve. **Cache room is not a cache size**: it excludes HIP
overhead, aligned staging, pending leases, verification buffers and SWA
rollback storage. Measure, then choose below it.

**Resident share is not hit rate.** 67% of experts fitting says nothing about
how often a routed expert is already present; that depends on routing locality
and has not been measured. Keep the two numbers separate in every report.

Experts total 149.8 GiB against 124.9 GiB of RAM, so streaming is mandatory
and no configuration can hold them all. Swap must not be relied on.

At an assumed ~5 GiB/s from the local SSD — unmeasured — a 0%-hit token costs
4.68 GiB and ~936 ms, so ~1.07 tok/s; an 80% hit rate gives ~5.3 tok/s and 90%
gives ~10.7. Arithmetic bounds recorded before results exist, not predictions.

### What those bounds turned out to be worth

Scored against measurement, because a bound recorded in advance is only
useful if someone goes back and checks it.

**Resident share against hit rate.** The warning above was the right
warning and the numbers are now in: 144 slots per layer holds 6,768 of
12,032 identities, **56% resident**, and yields a **92% hit rate**. Routing
locality is high, so the two numbers diverge in the favourable direction —
but they did have to be measured, and 56% would have been a poor guess for
92%.

**Throughput.** The projection was optimistic and the reason is worth
keeping. It counted only the 4.68 GiB of expert traffic, and a whole token
also reads **10.72 GB of BF16 projections** — the fused QKV and o_proj,
which are more than double the expert term. Measured decode at a 92% hit
rate is **5.7 tok/s**, against the ~10.7 the 90% line predicted. The
arithmetic was not wrong; its inventory was incomplete.

**Prefill.** Not modelled here at all, because at the time prefill was
assumed to be decode. Going layer-major makes the projections a per-chunk
cost rather than a per-token one, which is the single largest structural
win found in this lane: 0.422 to 0.098 s/token.

## Status

Every milestone through M5 is qualified. MiMo V2.6 Flash serves an
OpenAI-shaped text API on this host, on GPU, with tools, and the context
ladder is walked to 8K.

### Correctness

The model is right. Given "The quick brown fox jumps." twice plus "The quick
brown fox" it continues with all 12 tokens correct, and induction
strengthens with repetition. That took finding the fused-QKV layout, which
is the trap documented above and worth reading before touching this lane.

The held-out quality gate passes 7/7 against thresholds frozen before the
run: mean log-prob **-1.73 nats**, top-1 **51.6%**, top-10 **92.8%**, median
rank 1, no non-finite logits, no aborted steps.

### Backends

Two, and they agree. The CPU worker is the reference; the GPU worker is what
serves. Cross-backend agreement is bounded rather than exact, and the
boundary is measured rather than assumed: device and host `expf` differ by 1
ulp on **6.26%** of inputs, so anything downstream of a transcendental gets
a stated bound and everything else gets none. Reduction order is the other
such boundary; three reductions are pinned to the CPU's order because they
amplify, and `lm_head` deliberately is not, because its input already
differs and the ordered sum cost 9% of a token for a property it could not
deliver.

    CPU worker    ~55 s/token      reference, qualified 15/15
    GPU worker    0.176 s/token    ~200x, qualified 18/18

### Performance

Decode 0.176 s/token at 144 expert slots. Prefill is layer-major and
0.098 s/token, down from 0.422 when it was still the decode loop.

Expert admission is a major bottleneck, but its share depends on the workload.
The 94% prefill figure belongs to the earlier 48-slot short run. At 144 slots,
admission is 38.90 of 52.5 s for 538 prompt tokens and 544.47 of 887.9 s for
8158 tokens. At the longer depth, other layer work accounts for about 38% of
total prefill. Admission includes I/O and copy, not just physical SSD time.
Residency remains a lever: the 48-slot run spends about 152 s in admission
for the short prompt, versus 39 s with 144 slots.

A correction worth carrying: the expert path is **not** the dominant traffic
term. Counting a whole token, the BF16 projections are 10.72 GB against
5.03 GB of packed experts, so the fused QKV and o_proj dominate. An earlier
benchmark here measured the expert path alone at 26.7 tok/s and that figure
was read as a ceiling; the whole-token ceiling is nearer 8.

### Context

| depth | prefill | retrieval | replay | residency |
| ---: | --- | --- | --- | --- |
| 512 | 0.098 s/tok | ok | 0 differ | 95.41 GiB |
| 2048 | 0.088 s/tok | ok | 0 differ | 95.41 GiB |
| 8192 | 0.109 s/tok | ok | 0 differ | 95.41 GiB |

Retrieval places the needle at the **start** of context and asks for it from
the end, so it exercises the nine global layers rather than the 128-token
sliding window. A fact near the end is recoverable from the window alone and
would pass at any depth while proving nothing.

8K prefill is 14.8 minutes. Qualified, not comfortable.

The table's residency is the worker's logical resident-byte ledger, not total
physical host/device use or evidence of no swapping. The 2026-09-22 follow-up
probes at 144 and 96 slots were stopped by a zero-worker-swap guard during
loading, before inference; those isolated runs do not invalidate the earlier
context outputs, but a swap-free serving configuration needs separate
qualification. See the vault's `MiMo Performance Takeover and Experiments
2026-09-22` note and `Evidence/mimo26-takeover-20260922/` for pinned reports.

## Running it

    make test-mimo26-schema                        # no checkpoint needed
    make test-mimo26-checkpoint MIMO26_ROOT=...    # read-only, plus tokenizer
    make mimo26-qualify MIMO26_ROOT=...            # CPU worker, 15 checks
    make mimo26-gpu MIMO26_ROOT=...                # five GPU gates
    make mimo26-eval MIMO26_ROOT=...               # frozen quality gate
    make mimo26-context-gate MIMO26_ROOT=... MIMO26_DEPTHS=512,2048
    tools/mimo26_server ROOT --port 8640 --slots 16 --context 1024
    make mimo26-soak MIMO26_SERVER_URL=http://127.0.0.1:8640

The server is local-only by default and is not installed as a service.
The conservative command above is not a claim of production qualification;
the previous 144-slot example is withdrawn following the swap guard findings.

### Opt-in context-reset candidate (2026-09-22)

`mimo26_gpu_worker_reset_context` clears logical conversation state while
retaining expert weights and cumulative cache telemetry. It refuses during
execution, with outstanding I/O or a KV transaction, or after any execution
fault. The fault marker survives cold reset; recreate a faulted worker before
using retention. This is a single-owner API, not a thread-safety primitive.
The server still calls the original cold reset; no retention deployment or
live fault-recovery qualification is implied. The isolated GPU gate is
`tests/run_mimo26_reset_gate.py`; see the vault note `MiMo Guarded Context Reset
2026-09-22` for exact artifacts and acceptance scope.

Follow-up I/O gates (`tests/run_mimo26_io_fault_gate.py`) pass injected submit,
wait and short-read errors on real checkpoint reads. They confirm cold reset
does not drain pending I/O. Candidate server recovery now checks the guarded
reset before clearing quarantine: execution faults stay quarantined pending
qualified worker/process recreation. The actual recovery function passes a
CPU test with mocked worker calls and the real slot state machine. The new
server binary is isolated, not deployed or live-fault qualified; normal request
weight retention remains disabled.

`tools/mimo26_supervise.py` is a separate opt-in, loopback-only process
supervisor candidate. It has a lifetime replacement budget, waits for owned
child exit, checks port/GPU availability, and never replays requests. Busy or
timed-out health after initial readiness is not treated as a fault. Ten CPU
tests and a persistent injected GPU-fault test pass (one replacement, then
budget exhaustion); automatic transient GPU recovery through a successful
request remains unqualified. No service installation or default change.

Performance replay: chunk-local next-use eviction in
`tests/analyze_mimo26_lookahead.py` predicts 30.38% fewer misses at 48 slots
on the frozen 128-token trace. It uses only already-known chunk routes and
protects the current selection. This is not implemented in the runtime or
a latency result; multi-prompt replay and exact-output gates are next.

Follow-up: `config.expert_lookahead` (default false) uses an optional synchronous
chunk-route callback and additive `k3_expert_cache_plan_next_use` API. On three
additional 128-token prompts at 48 slots/chunk32, loads fell 25.8–28.5%; all nine
full prefill/decode logit vectors and all routes matched the frozen control.
Rebuilt default-off also matched. Total prefill was 77.26 s on versus 100.97 s
rebuilt-off (23.5% lower) in an ordered short-corpus comparison, not a randomized
throughput benchmark. Active-policy I/O fault gates pass; no sampled swap.
Not enabled in the server. See the vault's `MiMo Chunk Lookahead Runtime
Experiment 2026-09-22` for evidence and longer-context promotion gates.

The 2026-09-23 longer-prompt gate extends this to one synthetic 513-token
mixed-domain input with a partial final chunk. Chunk32 off/on/on/off means
138.52/98.16 s (29.1% less prefill time; 1.411x throughput). One chunk64
off/on pair measures 137.84/91.13 s (33.9% less; 1.513x); it is not a repeated
chunk64 benchmark. All six runs agree on input tokens, every selected expert,
and complete prefill plus two decode vectors, including across chunk widths.
Counter replays match; 971 one-second samples show zero worker swap. Kernel
and runtime sources are unchanged from the prior candidate; only harnesses
were extended. Startup/decode excluded, tracing included. Not a serving or
general-quality qualification; no default/server binary change. See the vault's
`MiMo Lookahead Longer-Prompt Qualification 2026-09-23` and its permanent
evidence bundle. Chunk64 still needs broader/repeated tests and the opt-in
server needs mixed-request/cancellation/recovery gates before promotion.

### Opt-in server gate and memory caveat (2026-09-23)

The source candidate now accepts explicit `--expert-lookahead on|off` and
`--prefill-chunk 0..128`. Defaults remain off/chunk32/cache16/context2048.
Lookahead plus chunk0 is rejected. CLI parsing now rejects unknown/duplicate
options, missing values, invalid numeric ranges and junk/overflow; numeric
context acceptance is not a qualification of that context. Health/startup
output reports the selected lookahead/chunk configuration. The original
`tools/mimo26_server` binary is unchanged; isolated candidates are not installed.

Fixed a pre-existing prefill disconnect bug: the callback was not checking the
requesting socket. It now cancels at the next committed chunk boundary, and
the server does not read unproduced logits from a stopped prefill. The actual
callback's CPU socket regression fails before and passes after the fix.
`make test-mimo26-server-options` runs the strict parser and callback tests.

At **cache16/chunk64/context1024**, the isolated loopback off/on mixed-request
test passes JSON/SSE consistency, busy refusal, disconnect at token128/380,
post-cancellation recovery and **56 full-vector validation checks**. An injected
failure after a real I/O submission stays quarantined through five HTTP503
refusals; explicit fresh-process recovery matches prior output. A clean binary
also passes an HTTP smoke request. All five processes exit zero; 501 one-second
worker-swap samples are zero. No automatic-restart, tool-loop, sustained-soak,
hardware-fault, remote-security or upstream-quality qualification is implied.

The broader **cache48** chunk64/129-token three-domain sweep completed
off/on/on with exact outputs and ~30–33% fewer expert loads. Its last off run
hit **197112 KiB worker swap at ~43 s** and was guard-terminated without a
prompt result. This is an incomplete ABBA comparison, not a passing benchmark.
Earlier 48-slot no-swap runs remain valid but do not establish a reliably
swap-free budget. Keep the guard and investigate host/GTT/reclaim conditions;
do not deploy cache48 from the logical ledger alone. See the vault's
`MiMo Opt-In Server and Cancellation Gate 2026-09-23` for preserved evidence.

A separate frozen-binary diagnostic with bounded kernel tracing identifies
14 order-10 Normal-zone kswapd wakeups from the worker through
`amdgpu_gem_object_create` → TTM → `ttm_pool_alloc_page`. At 4-KiB base pages
these requests are 4 MiB. The run still passes exact outputs with 551 zero-swap
worker samples; global swap-out/reclaim increases. Populated zones remain well
above sampled high watermarks, no watermark boost is sampled, and current plus
ancestor cgroups show unlimited high/max and no new high/max/OOM events.
This identifies a reclaim caller, **not** the cause of the historical untraced
worker swap, a safe capacity ceiling, or a fix. The earlier ABBA failure stands;
no global memory/driver policy or allocator change. See
`MiMo Reclaim Trace and MZG2 GPU Gate 2026-09-23` in the vault.

## Not done

**Beyond 8K.** Untested. The checkpoint advertises 1M and nothing here
supports reading that as available.

**Speculative decoding and MTP.** The MTP tensors are loaded and validated
but unused.

**Expert-major grouping in prefill.** Deliberately deferred: it would cut
GTT traffic, and GTT traffic is no longer the constraint. Admission is.

**MZG2-family research.** The earlier 88.7%-of-packed estimate is not a
qualified codec or end-to-end speed result. A reproducible 2026-09-22 CPU
screen trains on three experts and holds out six: byte-exact recovery saves
9.54% at 16 KiB tiles and 10.15% at 64 KiB, including illustrative aligned
expert headers/descriptors but excluding shared tables and the file index.
Packed symbols and scales each supply roughly half of the pre-metadata saving.
This candidate retains all 16 codes; the K3 MZG2 ABI's 15-code alphabet and
negative-zero canonicalization cannot be reused unchanged. No full transcode,
persistent MiMo format or admission-speed win is qualified yet.
The 2026-09-23 standalone GPU prototype now exactly recovers all six held-outs
at 16/32/64-KiB tiles and 1/4/8 waves per block (54 configurations), plus 12
synthetic configurations and 1,722 selected header/descriptor/model/stream
fault checks with intact output canaries. Publication refusal is a test-only
seam, not worker-cache integration. The fused 32-bit checksum is not
cryptographic authentication. One warm-memory screen favors 16-KiB tiles with
4/8 waves (~0.38 ms decode+integrity versus ~0.158 ms raw copy per expert),
despite slightly smaller storage savings; 64-KiB tiles take ~1.03–1.11 ms.
All original packed bytes, including code 8, and all scale bytes are retained.
Next gate: actual raw read+copy versus compressed read+decode+integrity with
expert-compute contention, then real transactional cache-admission fault tests.
Avoid recommending for or against deployment from size or warm timings alone.

Direct-I/O follow-up (2026-09-23): two bounded six-expert runs, 540 trials
(480 measured), exact recovered bytes and concurrent real-MLP outputs, zero
sampled worker swap. QD6 rANS16 admission alone is ~10.65 ms versus raw original
~11.55–11.76 ms, but combined MLP+admission is ~15.1–15.3 ms versus raw
~13.8–14.0 ms. Compute contention reverses the isolated benefit; do not promote
compression into serving. QD6 is limited by six held-outs, not worker QD8
qualification. Raw/cache16 remains the production candidate. Next gates are
automatic transient-fault replacement and sustained mixed-request soak on a
pinned clean profile. Supervisor CLI supports explicit chunk/lookahead with
12 CPU tests passing; no deployment or new live-recovery qualification.
See vault `MiMo Direct-I-O Compression and Production Gates 2026-09-23`.

**Authentication and remote exposure.** Local-only. The plan requires an
explicit authentication and network policy first.

**Supervised recovery now has bounded injected-I/O evidence.** Persistent
fault budget exhaustion and automatic one-time submitted-read failure →
quarantine → owned exit → replacement → exact new request pass. This is not
hardware/driver-fault qualification. The clean cache16/chunk64/context1024/
lookahead-on candidate also passes a478-second mixed soak (955 zero-swap
samples, stable idle FD/RSS). The initial transport audit reproduced permissive
HTTP framing and blocking reads without deadlines, addressed in the follow-up
below. See `docs/mimo26-production-candidate.md` and vault
`MiMo Automatic Recovery and Clean Soak 2026-09-23`.

**Transport follow-up,2026-09-23:** the separately frozen hardened candidate
now has strict restricted HTTP framing, absolute five-second header/body
reads and per-write budgets. CPU ASan/UBSan,1,000 deterministic mutations
and five live malformed/slow-client refusals pass. All56 captured complete
logit vectors remain byte-identical to the prior candidate. Automatic one-time
submitted-read fault replacement is requalified (196 zero-swap samples), as
is a478.17-second clean mixed soak (955 zero-swap samples, idle FD range0,
warmed endpoint RSS+8KiB); functional gates add508 zero-swap samples. No
deployment or default change. New CPU audit reproduces that connected-peer
prefill ignores shutdown and an expired generation deadline; this is the
next runtime fix, not a passing control test. Slow cancellation (~73 seconds),
longer outputs/tools/reasoning, production memory guard and release packaging
remain open. See vault `MiMo HTTP Transport Hardening 2026-09-23` for exact
candidate/report hashes and boundaries. Upstream quality stays deferred.

**Sparky's shards are unhashed**, so copy equality with the local checkpoint
is unproven at whole-file level.

**`moe_router_dtype`** is `bfloat16` in the config while this lane computes
the router in F32, following the executed reference. It was measured and is
not the bug -- it moves ranks marginally -- but deciding it properly needs
an oracle this host cannot run.

**An oracle this host *can* run now exists.** Upstream llama.cpp converts and
serves this checkpoint, and it fits across Beelink + Sparky (163 GiB resident,
~11.7 tok/s, correct output). vLLM 0.30.0 still cannot load it at any legal TP
size, but that no longer matters. The QKV layout question is settled above;
what remains unmeasured is token-level agreement between the two
implementations, which the two-node recipe makes cheap. See the vault note
`MiMo V2.6 Flash RL Two-Node GGUF Bring-up 2026-09-23`.
