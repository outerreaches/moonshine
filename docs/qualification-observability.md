# Production lifecycle logging qualification

Date: 2026-08-10

This gate qualifies the request lifecycle logging added at commit `45a7232`
(`feat: add production request lifecycle logging`) on top of the published
`0.2.0-research-preview` head `e9b7c24`. It qualifies the observability
contract in [observability.md](observability.md): event names, ordering,
bounding, redaction, and presentation. The change adds host-side callbacks
only; inference kernels, arithmetic, routing, I/O, cache policy, and
scheduling are unchanged.

## Scope

- AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)
- pinned official 96-shard Kimi K3 checkpoint
- loopback one-slot server with per-run random API key, 8K context and 32
  expert slots for the live gates; 131,072 context, 30 expert slots, and a
  65,536 output ceiling for the deployment-profile smoke
- one model process and no competing model transfer
- staged harness: `/home/alex/Workspace/moonshine-obs-gates-20260809`
  (host-local qualification tooling, not part of the public tree)

## Model-free gates

- `make test-cpu` passes (expert cache, exact prefix admission, JSON,
  OpenAI codec). The warning-free from-scratch build gate is exercised by
  the clean-checkout suite below; the incremental candidate build was
  already current.
- Truncation and flattening: HTTP reject diagnostics are all static strings
  (no `set_error` call echoes client input, and server error buffers are
  256--1,024 bytes against the 3,072-byte log buffer), so a source-level
  harness drove the real static `server_log` with a ~4 KB diagnostic
  containing CR, LF, TAB, 0x01, and ESC. Result: exactly one bounded
  3,174-byte line carrying `log_truncated=yes` with every control byte
  flattened.
- Pinned official OpenAI Python SDK 2.52.0 SSE replay fixture: PASS in an
  isolated environment.
- `git diff --check` clean.

## Live gates (8K/32 loopback)

- Reject battery: unauthenticated `/v1/models` returns HTTP 401 with
  `http.unauthorized`; malformed JSON and a 8,193-token ceiling overflow
  return HTTP 400 with bounded `request.reject` records (the overflow
  reports `max tokens must be an integer in [1,8192]`).
- Non-streaming completion: `stream=no` `request.start` and
  `request.complete` with `client=connected`, single JSON body, HTTP 200.
- Prefix miss ordering and state survival: an edited history logs
  `request.prefill.start reuse=miss replacement=guarded` immediately followed
  by `request.prefix.miss retained=184 matched=86 candidate=158`, both before
  any reset or inference. A continuation of the post-miss history then
  reused 187/212 prompt tokens (`reuse=hit`), proving the replaced state
  survives and serves exact reuse.
- Client disconnects: an abortive close before the first SSE byte logs
  `request.client_disconnect stage=stream_begin`; an abortive close
  mid-decode ends with `request.complete ... client=disconnected` and a
  clean slot release.
- Captured-log audit: no ANSI escapes under redirection, no raw control
  bytes, no line over 4 KiB, no API key, prompt, or reasoning leakage, and
  per-request event ordering as documented.

## Clean-checkout release suite and 128K smoke

From a detached clean worktree of exact commit `45a7232`:

- warning-free full build; `make test-cpu`, `make`, `make tests`,
  `make test` pass;
- model layout, tokenizer, reduction qualification, and locked chat hello
  pass;
- the 15,993-token natural-text retrieval passes, including the new
  lifecycle-order and prompt-accounting assertions;
- the complete 8K/32 live-gate set above passes a second time against the
  clean build, including the log audit;
- persistent 128K/30-expert/65,536-output profile: startup in 38.589 s;
  65,537 rejected with `max tokens must be an integer in [1,65536]`; two
  independent short requests return HTTP 200 in one process; exact-ceiling
  admission accepted; 11.3 GiB `MemAvailable` retained during residency with
  swap flat; clean `server.stop`; captured-log audit passes.
