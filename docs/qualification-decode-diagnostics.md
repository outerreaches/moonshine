# Decode-diagnostics qualification

Date: 2026-08-11

Status: **candidate gate defined; exact post-review candidate run pending**.

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
- fresh process for each side, no competing model process or transfer
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

Pending. Replace this section after the focused implementation commit is
DCO-signed, then rerun the complete gate from its detached clean worktree.
Raw request-dependent traces remain host-local and outside Git.
