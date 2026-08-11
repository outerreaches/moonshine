# Offline decode-cache analysis

`tools/analyze_decode_cache.py` validates and analyzes one committed decode-
diagnostics capture without starting Moonshine or using a GPU. It is a
standard-library-only tool for deterministic qualification reports, not a
workload forecast or an automatic cache-policy selector.

## Inputs and trust gate

The run directory normally contains:

- `decode.cache.csv`: the pre-decode LRU snapshot;
- `decode.ledger.csv`: the summary plus 92 layer accounting rows; and
- `decode.routes.csv`: one ordered 16-expert route batch per step and layer.

The capture ID and source capacity are mandatory. Before any counterfactual is
reported, the analyzer requires exact headers and numeric fields, one complete
capture, contiguous step/position/layer ordering, 16 unique in-range experts
per route row, reconciled ledger totals, and an exact source-capacity replay
with zero observed-hit-mask mismatches. Both the current
`host_interval_seconds` ledger and the explicitly versioned historical
`wall_seconds` ledger are recognized; their different summary-step semantics
are not interchangeable.

A header-only cache file is an empty snapshot, but emptiness alone does not
prove that a larger cache is reconstructible. Any target above the source
capacity requires `--fresh-empty-source`, which is an operator attestation that
the snapshot came from a fresh process with no earlier cache history. The flag
is rejected if the snapshot is nonempty. Without it, larger capacities are
rejected because evicted prefill residents and explicit invalidations are not
recoverable.

## Example

```sh
python3 tools/analyze_decode_cache.py report /path/to/run \
  --capture 1 \
  --observed-capacity 30 \
  --fresh-empty-source \
  --capacities 24,26,28,30,31,32,34,36,40 \
  --marginal-range 16:64 \
  --total-slots 2760 \
  --min-layer-slots 16 \
  --max-layer-slots 64 \
  --pin-counts 4,8,12 \
  --training-steps 35 \
  --out /path/to/private-report
```

Omit `--fresh-empty-source` unless the fresh-process provenance is known. For a
warm snapshot, keep every requested and per-layer maximum capacity at or below
the source capacity. The default marginal range is `16:64`, so it must be
narrowed explicitly when expansion is not justified.

The output directory is made mode `0700`; its deterministic JSON, CSV, and text
files are mode `0600`. Reports contain input SHA-256 values and basenames, not
absolute host paths or raw expert routes. They remain derived workload data and
must not be committed without an explicit disclosure review.

## Reported analyses

- **Uniform LRU curves** replay the runtime's exact batch semantics at each
  capacity: hits are resolved against frozen pre-batch state, rank-ordered
  touches update a pending LRU, only final surviving misses are admitted, and
  commit is atomic.
- **Layer marginal curves** report each layer independently; marginal gains are
  not assumed to be concave.
- **Fixed-total per-layer allocation** uses an exact multiple-choice dynamic
  program. It maximizes hits, then minimizes admissions, then evictions, then
  selects the lexicographically smallest layer-1-through-92 capacity vector.
  Because it optimizes and evaluates on the same complete trace, it is labeled
  **oracle/nonpromotable**.
- **Frequency-retained policy** is an online experimental counterfactual with
  exact per-layer frequency counters. Every requested expert must be resident
  after commit. It is TinyLFU-like only, not canonical Window-TinyLFU and not a
  deployment claim.
- **Full-trace pinned LRU** selects pins from the evaluated future and is
  explicitly **oracle/nonpromotable**.
- **Prefix-trained pinned LRU** selects pins only from an excluded prefix, then
  scores a cold suffix against a separately reported cold-suffix LRU baseline.
  It does not leak suffix routes, but its artificial cold boundary must be kept
  in view.

A warm snapshot exposes only current LRU residents, not evicted prefill history
or frequency metadata. Smaller LRU capacities use the oldest-first projection
implied by the snapshot. Alternative-policy warm starts cannot be reconstructed
exactly and are labeled with their actual initial-state assumptions.

Do not promote a policy or allocation from one cold capture. Use separate
training and evaluation captures representing warm prefixes, longer agentic
turns, code/reasoning work, and tool arguments, and retain the production
memory, exact-output, cache-accounting, and causal-state gates.

## Tests

```sh
make test-cache-analyzer
```

This runs the portable deterministic fixtures. An accepted trace can add the
offline golden replay without starting a server or GPU:

```sh
make test-cache-analyzer \
  MOONSHINE_CACHE_ANALYZER_GOLDEN_RUN=/path/to/accepted-run
```

The optional golden requires source-capacity-30 hits of `85,948`, zero source
mask mismatches, and an exact 2,760-slot allocation result of `87,390` hits over
per-layer bounds 16 through 64.
