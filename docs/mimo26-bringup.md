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

The bottleneck is disk, not arithmetic. Profiling puts expert admission at
94% of prefill and roughly three quarters of decode, while compute sits near
the memory-bandwidth bound. The lever is residency: 48 slots leaves
admission at 152 s for a 538-token prompt, 144 slots at a 92% hit rate
brings it to 39 s.

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

## Running it

    make test-mimo26-schema                        # no checkpoint needed
    make test-mimo26-checkpoint MIMO26_ROOT=...    # read-only, plus tokenizer
    make mimo26-qualify MIMO26_ROOT=...            # CPU worker, 15 checks
    make mimo26-gpu MIMO26_ROOT=...                # five GPU gates
    make mimo26-eval MIMO26_ROOT=...               # frozen quality gate
    make mimo26-context-gate MIMO26_ROOT=... MIMO26_DEPTHS=512,2048
    tools/mimo26_server ROOT --port 8640 --slots 144 --context 4096
    make mimo26-soak MIMO26_SERVER_URL=http://127.0.0.1:8640

The server is local-only by default and is not installed as a service.

## Not done

**Beyond 8K.** Untested. The checkpoint advertises 1M and nothing here
supports reading that as available.

**Speculative decoding and MTP.** The MTP tensors are loaded and validated
but unused.

**Expert-major grouping in prefill.** Deliberately deferred: it would cut
GTT traffic, and GTT traffic is no longer the constraint. Admission is.

**MZG2.** Measured rather than assumed: MiMo's experts compress to **88.7%**
of packed, because MXFP4 is already near its entropy at 3.55-3.78 bits of 4
and essentially all the gain comes from the scales, which are a sixteenth of
the payload. That is ~11% fewer read bytes for a 133 GiB derived artifact
and a decompression path in the hot loop, and decompressing into the cache
costs more GPU traffic than the copy it replaces. Recommended against on
those numbers.

**Authentication and remote exposure.** Local-only. The plan requires an
explicit authentication and network policy first.

**Supervised restart has never run against a real fault.** The path is
exercised by the slot's unit tests; no decode has actually faulted on this
hardware.

**Sparky's shards are unhashed**, so copy equality with the local checkpoint
is unproven at whole-file level.

**`moe_router_dtype`** is `bfloat16` in the config while this lane computes
the router in F32, following the executed reference. It was measured and is
not the bug -- it moves ranks marginally -- but deciding it properly needs
an oracle this host cannot run.
