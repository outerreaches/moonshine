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

## Implemented so far

- `mimo26_architecture.{c,h}` — payload-free schema: layer kinds from the
  explicit pattern, per-kind tensor contracts, exact group coverage.
- `mimo26_manifest.{c,h}` — index parse and validation, the index-driven
  shard opener, directory reconciliation, and the routed-expert span planner.
- `k3_st_model_open_paths` — additive opener taking an explicit shard list,
  since MiMo's filenames defeat the numbered family. Existing openers are
  unchanged.
- `tests/test_mimo26_architecture.c`, `tests/test_mimo26_manifest.c` —
  synthetic positive and negative cases.
- `tests/test_mimo26_official.c` — end-to-end against a real checkpoint.
- `tests/audit_mimo26_checkpoint.py` — full read-only metadata audit with
  machine-readable status; `tests/test_mimo26_audit.py` injects one
  synthesized fault per check.
- `tests/fixtures/mimo26_tokenizer_v1.json` with
  `tests/generate_mimo26_tokenizer_fixture.py` and
  `tests/test_mimo26_tokenizer.py` — pinned token ids and rendered turns,
  produced without `trust_remote_code`.
- `tools/mimo26_budget.py` — allocation, cache and KV ledger from measured
  host memory.

```
make test-mimo26-schema                      # no checkpoint needed
make test-mimo26-checkpoint MIMO26_ROOT=...  # header-only, read-only
make mimo26-budget MIMO26_ROOT=...
```

The audit and the C module encode the same contract independently, so they
disagree loudly if either drifts. Both pass, and the C path independently
reproduces the Python-derived figures: 382 main / 72,192 expert / 48 MTP /
364 vision / 95 audio tensors, 149.8125 GiB of experts across 12,032 resolved
expert identities, 8.2837 GiB of static text.

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

## Status

M0 through M3 are complete and M4 is qualified at the **worker** level. The
48-layer CPU worker generates correct text: given "The quick brown fox
jumps." twice plus "The quick brown fox" it continues with all 12 tokens
right, and induction strengthens with repetition.

`make mimo26-qualify MIMO26_ROOT=...` passes 15/15 against the real
checkpoint — determinism, rollback replay, reset, fault containment and
cache-size independence, all bit-exact on the full 152,576-entry logit
vector. `make mimo26-eval MIMO26_ROOT=...` runs the teacher-forced quality
gate whose corpus and thresholds are frozen in `tests/mimo26_eval_spec.json`.

## Not done

**M4's slot-level gates.** Busy-slot rejection, cancellation and worker
quarantine are named in the plan's M4 gate but are properties of a serving
slot, not of this worker, which has no concurrency. They belong with the
server work.

**M5 entirely.** It needs a GPU path, and the MiMo lane is CPU-only: the
worker decodes at roughly a minute per token, so the context ladder, prefill
latency budgets and the sustained soak are not reachable on this host. A GPU
window is a prerequisite, not a scheduling preference.

**Two loose ends from M0/M1.** `tests/test_mimo26_manifest` still covers
parse, reconcile and span-planner rejection but not a fault-injected shard
file. Sparky's copy of the shards remains unhashed, so equality with the
local copy is unproven at whole-file level.

**One fidelity question left open.** `moe_router_dtype` is `bfloat16` in the
config; vLLM honours it and the shipped reference hardcodes F32, which is
what this lane implements. It was measured and is not the bug — it moves
ranks marginally — but the checkpoint says what it says. Changing it needs an
oracle this host cannot run, so it is recorded rather than guessed at.
