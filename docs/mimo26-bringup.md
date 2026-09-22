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

**Global QKV carries an over-provisioned FP8 scale grid.** SWA layers hold
`qkv_proj.weight [14848,4096]` with `weight_scale_inv [116,32]`, exactly
`ceil(14848/128)`. Global layers hold `[13568,4096]` with `[108,32]`, but
`ceil(13568/128)` is 106. Block rows 106 and 107 contain live plausible scale
values, not padding: the grid was built for a 13,824-row layout, with V sized
as if `head_dim` were 192, and the weights were later sliced to 13,568.

Flat `row / 128` scale indexing stays correct, because every Q/K/V segment
boundary lands on a block boundary. But a validator asserting
`scale_rows == ceil(rows / 128)` rejects the real checkpoint, and reading 108
as the QKV width mis-splits Q, K and V. Both failure modes are covered by
negative tests.

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

## Not done

M0 does not exit until these land: `tests/test_mimo26_manifest` covers parse,
reconcile and span-planner rejection but not a fault-injected shard file; and
the next-stage input list for M1 is still implicit rather than written down.
Beyond M0, nothing in M1's numerical contract past the MXFP4 expert path has
an oracle yet.
