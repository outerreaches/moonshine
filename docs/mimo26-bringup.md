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

## Implemented so far

- `mimo26_architecture.{c,h}` — payload-free schema: layer kinds from the
  explicit pattern, per-kind tensor contracts, exact group coverage.
- `tests/test_mimo26_architecture.c` — synthetic positive and negative cases.
- `tests/audit_mimo26_checkpoint.py` — full read-only metadata audit with
  machine-readable status.
- `tests/test_mimo26_audit.py` — a synthesized fault per audit check.

```
make test-mimo26-schema                      # no checkpoint needed
make test-mimo26-checkpoint MIMO26_ROOT=...  # header-only, read-only
```

The audit and the C module encode the same contract independently, so they
disagree loudly if either drifts. Both currently pass, and the audit reports
`status: pass` on the full 73,081-tensor checkpoint.

## Not done

`mimo26_manifest.{c,h}` with the index-driven shard and span planner;
tokenizer, chat-template and generation fixtures; the static
allocation/cache/KV budget; and the C official-metadata test that needs the
manifest's opener. M0 does not exit until those exist.

Memory planning starts from 8.284 GiB of static text tensors and 12.75 MiB per
expert, against 149.8 GiB of experts total — so experts stream and the resident
cache size is an audited choice, not a maximum. At an assumed ~5 GiB/s from the
local SSD, a 0%-hit token costs ~0.94 s of expert traffic; roughly an 80% hit
rate is needed for ~5 tok/s. Those are arithmetic bounds from unmeasured
bandwidth, recorded before results exist, not predictions.
