# Decode-diagnostics qualification

Date: 2026-08-11

Status: **PASS for implementation commit
`d085e0251cc8a4012291a52886f47689e5cbf497`; publication follow-up
`65a2305d0e484916c9d0fe300117c9d1a27b5663` adds the offline analyzer and
portable tests without changing model execution**.

This gate qualifies the opt-in decode cache snapshot, per-layer ledger,
content-derived route trace, and causal-state comparison fingerprints described
in [the observability contract](observability.md). These artifacts are
qualification-only, sensitive derived workload data. They do not change the
production path unless explicitly enabled.

## Scope and fixed configuration

- AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)
- Linux 7.0.0-28-generic and ROCm 7.2.0
  (`HIP 7.2.53150-7b886380f9`)
- pinned official 96-shard Kimi K3 checkpoint
- one loopback server, 131,072-token context, 30 expert slots per routed
  layer, 16 staging slots, and a 512-token server output ceiling
- one fixed, host-local non-streaming planning request with
  `reasoning_effort=medium` and `max_tokens=128`; its exact bytes and digest
  remain in the private qualification artifact
- fresh process for each side, no competing large-model process or transfer;
  one small steady background vision tenant was unchanged across both sides
- predeclared maximum diagnostic decode overhead: 2.0%

## Required gates

1. Run a baseline with `--decode-state-digest`, then an otherwise identical
   process with that option plus a fresh `--decode-diagnostics PREFIX`.
2. Normalize only response `id` and `created`. Require byte-equivalent
   remaining response JSON, finish/usage, request cache deltas/totals, state
   position, and all four state fingerprints.
3. Require diagnostic events before the final `request.complete`. Baseline must
   produce no decode CSVs or `request.decode.io`; the candidate must produce
   exactly one digest and one decode-I/O aggregate.
4. Require three new regular mode-`0600` CSVs, identical capture IDs, one
   committed capture, 92 ordered layers for every zero-based route step, 16
   distinct in-range experts per route, `steps = generated + forced_trailer`,
   and `route_rows = steps * 92`.
5. Require one ledger summary and 92 layer rows. Reconcile accesses, hits,
   misses, read requests, logical/physical bytes, wait/completion counts, and
   the server aggregate. Summary steps count decode token evaluations; layer
   rows count per-layer invocations.
6. Replay the source capacity with zero observed-mask mismatches. Only then
   report other LRU capacities, supplying the source capacity explicitly and
   rejecting larger targets unless the snapshot is empty and the operator
   explicitly attests that it came from a fresh process with no prior cache
   history.
7. Require candidate decode overhead no greater than 2.0%. Report hardware,
   software, context, cache shape, response/state differences, timings, and
   cache/I/O totals.
8. Audit logs and CSVs for prompt, generated text, token IDs, gate weights,
   credentials, private paths, ANSI/control injection, and unsafe permissions.
   Route IDs, positions, and state fingerprints remain sensitive even when the
   direct-content audit passes.
9. Run `make test-cpu`, a warning-free full build, `make tests`, `make test`,
   `make test-prefill-gemm-shapes`, and the smallest real-weight layout,
   tokenizer, reduction, and chat oracles from a detached clean worktree of the
   exact candidate.

## Preliminary instrumentation run

A pre-commit instrumentation run established that the proposed measurements
are viable, but it predates the final permission, rollback, replay-validation,
accounting-label, and event-order review fixes and therefore is not the final
publication gate.

- normalized response, generated-token count, request cache counters, state
  position, and all four fingerprints: exact
- state position and all four region fingerprints matched exactly; raw
  fingerprints remain in the private qualification artifact
- baseline/candidate decode: 350.079 / 350.215 seconds; measured overhead
  0.038848%
- 128 generated plus 13 forced-trailer evaluations produced 141 decode steps,
  12,972 routes, 207,552 accesses, 85,948 hits, 121,604 misses, and 41.4103%
  hit rate
- exact recorded logical reads were 2,133,817,491,456 bytes and physical reads
  were 2,134,315,581,440 bytes
- explicit I/O wait 290.279 seconds; maximum two reads in flight
- empty pre-decode snapshot; capacity-30 replay reproduced all 12,972 observed
  masks exactly

## Exact-candidate result

The focused, DCO-signed implementation commit
`d085e0251cc8a4012291a52886f47689e5cbf497` passed from a clean detached
worktree. The candidate used the Q8 static server path at 131,072 configured
context with 30 expert slots per routed layer (2,760 slots and 45.104 GiB of
expert payload), 16 staging slots, and the fixed request above. A first pair
was intentionally discarded before completion after unrelated CPU differential
testing overlapped its baseline; none of that pair's timing or output was used.
The accepted pair began after a new large-process/SVM preflight and used fresh
servers on both sides.

The accepted paired result was:

- normalized response JSON (excluding only `id` and `created`), finish reason,
  usage, 128 generated tokens, 13 forced-trailer evaluations, request and total
  cache counters, state position, and all four causal-state fingerprints:
  **exact**
- baseline / diagnostic decode: 350.354 / 350.329 seconds, or **-0.00714%**
  measured overhead; this sub-noise negative value is not claimed as a speedup
  and passes the predeclared 2.0% ceiling
- baseline / diagnostic request wall time: 473.509 / 472.510 seconds
- current reviewed ledger schema with `host_interval_seconds`, summary
  `steps=141`, 12,972 ordered route rows, and 207,552 accesses
- 85,948 hits, 121,604 misses/read requests, and 41.4103% hit rate; exact
  logical reads 2,133,817,491,456 bytes and physical reads
  2,134,315,581,440 bytes
- diagnostic I/O wait 289.278 seconds, expert-pipeline interval 310.772
  seconds, summed layer host intervals 348.854 seconds, 1.476 seconds
  unattributed by that bounded split, and maximum two reads in flight
- one empty pre-decode snapshot with explicit fresh-process provenance;
  capacity 30 reproduced every observed hit mask with zero mismatches
- all three raw CSVs regular mode `0600`; digest and decode-I/O events preceded
  the final completion event; the baseline emitted neither CSVs nor a
  decode-I/O event
- direct-content audit found no prompt/generated text, token IDs, gate weights,
  or credentials in the CSVs; raw logs and reports remain private because
  routes, positions, fingerprints, configured paths, and hashes are sensitive

The exact worktree also passed `make test-cpu` with GCC and Clang 17 from clean
builds, a warning-free full build, `make tests`, `make test`,
`make test-prefill-gemm-shapes`, model layout, tokenizer,
`make test-reduction-qualification`, and the native real-weight chat hello.
The diagnostic server was stopped after capture.

## Offline cache-policy result

The deterministic analyzer added in
`65a2305d0e484916c9d0fe300117c9d1a27b5663` replayed the accepted current-schema
capture. Its source-capacity trust gate reproduced 85,948 hits with zero mask
mismatches before any counterfactual was reported. Selected uniform LRU results
were:

| slots/layer | hits | misses |
| ---: | ---: | ---: |
| 24 | 77,075 | 130,477 |
| 26 | 80,275 | 127,277 |
| 28 | 83,277 | 124,275 |
| 30 | 85,948 | 121,604 |
| 31 | 87,140 | 120,412 |
| 32 | 88,170 | 119,382 |
| 34 | 90,304 | 117,248 |
| 36 | 92,484 | 115,068 |
| 40 | 96,456 | 111,096 |

At the same fixed 2,760-slot total, exact multiple-choice optimization over
per-layer bounds 16 through 64 produced 87,390 hits and 120,162 misses, a gain
of 1,442 hits over uniform capacity 30. This allocation was optimized and
measured on the same complete trace and is therefore an
**oracle/nonpromotable bound**; the per-layer vector remains in the private
report.

The online experimental frequency-retained policy produced 87,296 hits and
120,256 misses with mandatory requested-batch residency intact, 1,348 more
hits than uniform LRU. It uses exact counters for this analysis and is only
TinyLFU-like, not canonical Window-TinyLFU or a deployment claim. Full-trace
pinned-LRU oracle bounds at 4, 8, and 12 pins per layer produced 87,517,
88,191, and 88,513 hits respectively and are likewise nonpromotable.

A leakage-free prefix screen trained pins on the first 35 steps, excluded that
prefix from scoring, and began the 106-step suffix cold. Its separate cold-LRU
suffix baseline had 64,319 hits; 4, 8, and 12 trained pins produced 63,623,
61,880, and 59,984 hits. Thus this cold-trace prefix screen provides no evidence
to promote pinning. No allocation, pinning, or admission policy is promoted
from this single cold trace. Representative warm-prefix, longer agentic,
code/reasoning, and tool-argument training/evaluation captures remain required.

Analyzer outputs were deterministic, with a mode-`0700` directory and
mode-`0600` files. Raw routes, input hashes, state fingerprints, responses, and
the per-layer optimized vector remain host-local and outside Git.
