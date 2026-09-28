# Releasing Moonshine

## Which lane are you releasing?

Three lanes, three different answers. Everything below the "Repository setup"
heading concerns **Kimi K3** unless a section says otherwise.

| Lane | Release mechanism |
|---|---|
| **Kimi K3** | this document: clean-checkout qualification, then the public repo |
| **MiMo V2.6 Flash** | `tools/mimo26_package.py` — immutable on-disk releases with evidence bound to the binary. See [Releasing MiMo V2.6 Flash](#releasing-mimo-v26-flash) |
| **GLM 5.3 Flash** | **not releasable from this tree.** Component objects and phase tests only; no servable binary. Active work is in `moonshine-glm53-20260831` |

## Releasing MiMo V2.6 Flash

MiMo does not ship through the public repo. It ships as an **immutable on-disk
release** whose qualification evidence is bound to the exact binary, because the
serving profile matters as much as the code.

### 1. Qualify the candidate

Build, serve on the intended profile, and run the battery. The report records the
binary's own hash by hashing the running inode, so it cannot later be attached to a
different build:

```sh
make tools/mimo26_server
./tools/mimo26_server /srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL     --host 127.0.0.1 --port 8080 --slots 160 --context 131072     --prefill-chunk 128 --expert-lookahead on --expert-major on     --retain-experts on --kv-prefix-reuse on --request-deadline-seconds 1800
python3 Scripts/mimo26-qualification/qualify.py --port 8080     --label <what-changed> --out results/<what-changed>.json
```

Eleven checks must pass. For a change that alters model output, that battery is
**not sufficient** — see the quality-screen requirement below.

### 2. Package

```sh
python3 tools/mimo26_package.py build     --root /srv/modelstore/private/moonshine-releases/mimo26     --qualification results/<what-changed>.json     --model /srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL     --note "<one line on what changed>"
```

Refused on a dirty worktree, or if the report names a different binary. It prints
any reason the report would fail activation **at this point**, while you are still
looking, rather than after a signature exists.

### 3. Activate, stating the profile

```sh
python3 tools/mimo26_package.py activate <release-dir>     --root /srv/modelstore/private/moonshine-releases/mimo26     --expect-profile "expert_weight_reuse=true,expert_major=true,expert_lookahead=true,retain_experts=true,kv_prefix_reuse=true,context=131072,expert_slots=160,prefill_chunk=128,request_deadline_seconds=1800"
```

Activation refuses unless every check passed, a versioned mandatory set is covered,
and the intended profile is **stated and matches**. `--expect-profile` is required
for any non-stock report, and the expert kernel must always be named — the two
kernels are not bit-identical and a reader cannot tell which ran from the binary.

### 4. Sign, including the rollback target

Load the key into the agent first — the packager captures `ssh-keygen`'s output, so
an interactive passphrase prompt is at best awkward:

```sh
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/moonshine_signing

python3 tools/mimo26_package.py sign <release-dir> --key ~/.ssh/moonshine_signing
python3 tools/mimo26_package.py verify <release-dir> \
    --allowed-signers ~/.ssh/allowed_signers --require-signature
```

A detached `ssh-keygen -Y sign` over `manifest.sha256`, which covers the manifest
and therefore the binary and all pinned sources. It does **not** attest the model
weights; those carry their own manifest hash. The packager unseals and re-seals the
0555 directory itself.

Sign `previous` too, or a rollback lands on an unsigned release. Once both are
signed the supervisor takes `--allowed-signers` and `--require-signature`.

Then prove the gate can refuse, rather than trusting a check that has only ever
passed:

```sh
python3 tests/prove_mimo26_signature_gate.py \
    --release-root /srv/modelstore/private/moonshine-releases/mimo26 \
    --allowed-signers ~/.ssh/allowed_signers
```

Five cases against the supervisor's own `resolve_release()` on scratch copies:
intact resolves; deleted signature, one flipped bit and a signature from an
unlisted key are each refused; and a control confirms an unsigned release still
resolves with the flag off, so the refusals belong to the flag and not to the
copying. Check the exit status without a pipe.

### 5. Serve it

Restart the supervisor; it re-resolves `current/` and re-verifies on every start.
Add `--allowed-signers ~/.ssh/allowed_signers --require-signature` so an unsigned
release cannot be served — `resolve_release` then reports `release_unusable` and the
supervisor refuses to launch rather than starting something unattested.
See [Running MiMo V2.6 Flash](README.md#running-mimo-v26-flash).

**Wait for the slot to be idle.** One request holds the single execution slot for as
long as its prefill takes — deep prompts run tens of minutes — so check
`inflight` in the status file before restarting, or the restart kills a live request.

### Requirements that are easy to miss

- **Keep a rollback target on a different commit.** A bulk retire once left two
  builds of one commit, so `previous` pointed at an identical binary and rolling
  back changed nothing. `list` prints the commit beside each release for this reason.
- **A change that moves model output needs a quality screen**, not just the battery.
  Pre-register the criteria and freeze the task set before either arm runs; see
  `Scripts/mimo26-qualification/quality_screen.py`.
- **A bit-exact change has a stronger check available**: run
  `quality_screen.py --expect <previous results>` and require zero divergence. Both
  attention-kernel changes shipped on that basis instead of a screen.
- **Toggling the expert kernel invalidates the prefix store**, by design — each
  arithmetic mode gets its own directory (`k1-tiled`, `k1-gemv`).

### Two release names predate the pre-publication rewrite

The first two MiMo releases were cut before history was rewritten to drop a
committed build artifact and three weight-derived goldens, so the short commit in
their directory names does not appear in the published history:

| release name | pre-rewrite | published |
|---|---|---|
| `0.2.0-research-preview-98bf466-20260928T014626` | `98bf466` | `4b3e91d` |
| `0.2.0-research-preview-3ea7d65-20260927T205008` | `3ea7d65` | `53f2bfa` |

Nothing about the releases themselves changed: none of the removed paths is among
the 151 pinned sources, so both still `verify` clean with good signatures. Only
the name-to-commit link needs this table. Releases cut from here on name a commit
that exists.

## Repository setup

1. Use the public `outerreaches/moonshine` repository.
2. Use the display name **Moonshine** and this description: "Moonshine is an
   experimental single-node SafeTensors/ROCm inference engine for Kimi K3 on
   128 GB AMD Strix Halo, with NVMe-streamed MXFP4 experts and layer-major
   prefill."
3. Enable private vulnerability reporting.
4. Enable secret scanning and push protection.
5. Protect `main` and require the portable CI workflow.
6. Do not push a candidate until the local commit passes clean-checkout
   qualification and review.
7. Keep private archive refs and pre-public history bundles local. A maintainer
   checkout containing `refs/heads/archive/*` must use a main-only
   `remote.origin.push` refspec and a pre-push guard that rejects those refs;
   do not bypass the guard with `--no-verify`.

## Release qualification

From a clean checkout of the exact candidate commit:

```sh
make test-cpu
make
make tests
make test
make test-prefill-gemm-shapes
make test-model-layout MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-tokenizer MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-reduction-qualification MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-chat-hello MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

Also verify `moonshine-server` health, model discovery, JSON completion, and
SSE completion on the qualified host. While one completion is active, require
responsive `busy:true` health/model discovery and an immediate HTTP 503 for a
competing completion without backlog growth. Confirm prefill and quiet-decode
SSE keepalives, disconnect cancellation with a successful exact recovery
request, graceful `SIGTERM` during active inference, a warmer second stateless
request, automatic exact append-prefix reuse through
`usage.prompt_tokens_details.cached_tokens`, and mismatch fallback. For a
raised-ceiling candidate,
confirm health/model metadata, acceptance at the configured ceiling, rejection
one token above it, explicit-null fallback, and remaining-context clamping.
The admission gate may stop naturally; it does not need to generate the entire
configured maximum. Record the commit, model revision, hardware, kernel, ROCm
version, context, output ceiling, memory ledger, and timings.

For a routed-prefill timing change, use `--prefill-diagnostics` or
`MOONSHINE_PREFILL_DIAGNOSTICS` only in fresh qualification processes. Discard
one warm-up and publish four identical baseline arms as the noise floor. Run
`tools/analyze_prefill_screen.py --noise-only`; `UNRESOLVABLE` stops the
screen before candidate work. Only `READY_FOR_ABBA` permits
baseline/candidate/candidate/baseline in one uninterrupted window and the
complete nine-file analysis. Require exact route-union and physical-I/O
identity, gate on the mechanism's subphase, and use aggregate phase wall only
as a no-regression guard. Record NVMe temperature/throttle state externally
for every arm.

For a standalone MZG2 bundle, first run `make test-mzg2-bundle` and the
single-file SafeTensors/bundle-manifest portable tests. Build from the pinned
96-shard source in a private root, require exactly 2,460 static tensors and
113,509,540,864 payload bytes, validate all source/MZG2/auxiliary hashes, and
publish atomically. Qualify dense-source+MZG2 against bundle-only+MZG2 for
identical engine ledgers, model-layout identity, output values/bytes, cache
counters, all four state digests, live/displaced/restart checkpoints, tool and
structured output, and 128K/30 startup. The final clean-machine gate must have
no official shards available. Archive source shards only after remote hashes
and a sample restore pass. Never publish model/bundle payloads in the Moonshine
Git repository.

For durable exact-prefix checkpoints, first run
`make test-prefix-checkpoint`. Then qualify the server with a fresh private
root: publish one turn, record an uninterrupted exact continuation, displace
the live session, restore the same continuation, restart the server, and
restore it again. Require identical response bytes and all four state digests,
standard cached-token accounting, ≥5× prefix restoration versus full prefix
evaluation, mode `0700`/`0600`, and fail-closed corrupt metadata/state
behavior. Mark terminal checkpoint export as a distinct slot phase and allow
exactly one next completion to wait in the existing pending slot. An immediate
next completion must not receive HTTP 503; a second contender must still be
rejected without blocking health/model discovery. Require no material
regression in the preceding response's terminal latency and require queued
wait to remain bounded by publication time plus scheduling noise. The disabled
profile must perform no checkpoint I/O; a pre-feature rollback binary may omit
checkpoint health metadata rather than reporting `enabled:false`.
Run `tools/qualify_checkpoint_handoff.py` with a mode-`0600` private request
and mode-`0700` output directory as the immediate-next/two-contender gate.
Run `tools/qualify_checkpoint_stream_handoff.py` with `stream:true` as the
matching SSE terminal gate. Candidate `999d6e8` passed both gates, the
model-backed fixture, and live/displaced/restart recovery; production was
rolled back after qualification and remains disabled.

Run `tools/qualify_checkpoint_negative.py` in all three modes from a fresh
private root: `disconnect`, `publication-failure`, and `shutdown`. The
candidate passed all three. This qualification does not activate production;
perform the 24-hour/ten-publication observation only after an explicit opt-in
launch review.

For a persistent 128K service on the qualified 128 GB host, use
`--experts 30` and complete at least two independent prefills in the same
process. The Q8/32 128K configured-capacity fixture covers one cold request;
after its cache is warm, a separate prefill workspace may violate the retained
CMA-plus-4-GiB guard. Do not weaken the guard to make that configuration pass.

For the Hermes ad hoc gate, record its effective `model.max_tokens`,
`model.context_length`, `agent.reasoning_effort`, API timeout, and streaming
read timeout. Qualify the 128K Moonshine 0.2 profile at 65,536 output tokens
and `reasoning_effort: medium`; confirm acceptance at the ceiling, rejection
at 65,537, remaining-context clamping, and an ordinary naturally stopped
response. Lower-context/server-ceiling profiles must override Hermes's generic
65,536-token default. Test with the profile documented in
`docs/deployment-profiles.md`.

For a server logging change, capture one TTY run and one redirected run. Confirm
that redirected output contains no ANSI escapes; `NO_COLOR=1` disables TTY
color; exact reuse is reported before prefill; reasoning, response-or-tool, and
64-token decode-progress events appear in order; and final usage matches the
API response. Exercise one mismatch, malformed request, authentication failure,
busy rejection, mid-prefill disconnect, buffered-decode disconnect, and
`SIGTERM` during active work. Require a single `request.cancelled` record with
the correct reason and no corresponding completion/failure. Audit the captured
log for prompts, generated text, tool arguments, request bodies, credentials,
control-character line injection, and unbounded records. Use
[the operational logging contract](docs/observability.md) as the expected event
surface.

For a decode-diagnostics change, use fresh single-request servers with the
same model, context, expert capacity, and fixed request. Run the baseline with
`--decode-state-digest`; run the candidate with that option plus
`--decode-diagnostics PREFIX`, using a new prefix outside the checkout. Before
measurement, set the acceptance bound to at most 2.0% candidate decode
overhead.
Normalize only response `id` and `created`, then require exact remaining JSON,
finish/usage, cache deltas/totals, state position, and all four comparison
fingerprints. Diagnostic events must precede the final `request.complete`.

Require all three CSVs to be regular mode-`0600` files. Validate one capture,
92 ordered layers per zero-based step, `routes = steps * 92`, 16 unique experts
per route, ledger summary plus 92 layer rows, integer accounting, and consistent
capture IDs. Remember that decode steps include forced structural trailer
evaluations and can exceed API completion tokens. Replay must include the
explicit live source capacity and must reproduce its observed hit masks exactly:

```sh
make test-decode-cache-replay \
  MOONSHINE_DECODE_CACHE_TRACE=/path/to/capture.cache.csv \
  MOONSHINE_DECODE_LEDGER_TRACE=/path/to/capture.ledger.csv \
  MOONSHINE_DECODE_TRACE=/path/to/capture.routes.csv \
  MOONSHINE_DECODE_CACHE_SOURCE_CAPACITY=32 \
  MOONSHINE_DECODE_CACHE_CAPACITIES='24 28 30 32'
```

Do not interpret targets above the source without a proven fresh-process empty
snapshot and `MOONSHINE_DECODE_CACHE_FRESH_EMPTY_SOURCE=1`. Prior evictions and
explicit slot invalidations are otherwise absent, so the replay gate rejects
the ambiguous expansion. Audit route
traces and state fingerprints as sensitive derived workload data. Record the
full gate in
[decode-diagnostics qualification](docs/qualification-decode-diagnostics.md).

Run a final audit:

```sh
git status --short
git grep -nE '(token|password|secret|api[_-]?key)' -- \
  ':!RELEASING.md' ':!SECURITY.md'
git grep -n '/home/' -- README.md docs .github
```

Review `LICENSE`, `NOTICE`, `docs/provenance.md`, `CHANGELOG.md`, and
`CITATION.cff`. Confirm that the README independence disclaimer remains
visible.

## Tagging

Update the version in `moonshine_version.h`, `CHANGELOG.md`, and
`CITATION.cff`, commit the release, then create a signed annotated tag:

```sh
VERSION=x.y.z-research-preview
git tag -s "$VERSION" -m "Moonshine $VERSION"
```

Do not tag or publish model weights.
