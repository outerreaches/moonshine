# Releasing Moonshine

## Which lane are you releasing?

Three lanes, three different answers. Everything below the "Repository setup"
heading concerns **Kimi K3.1 Pro** unless a section says otherwise.

| Lane | Release mechanism |
|---|---|
| **Kimi K3.1 Pro** | this document: clean-checkout qualification, then the public repo |
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
   experimental single-node SafeTensors/ROCm inference-engine research preview
   for the pinned Kimi K3.1 Pro checkpoint and MiMo V2.6 Flash on 128 GB AMD
   Strix Halo, with BF16 Gated DSA, NVMe-streamed MXFP4 experts, and
   layer-major prefill."

   The description named only K3 until MiMo became a served lane with its own
   entry point and release process. Keep it in step with the lane table at the
   top of the README; a description that omits a served lane sends readers to
   the wrong binary. GLM 5.3 Flash remains component-only and is not described
   as a served lane.
3. Enable private vulnerability reporting.
4. Enable secret scanning and push protection.
5. Protect `main` and require the portable CI workflow.
6. Do not push a candidate until its exact commit passes clean-checkout
   qualification and review and every gate claimed in its release notes.
7. Keep private archive refs and pre-public history bundles local. A maintainer
   checkout containing `refs/heads/archive/*` must use a main-only
   `remote.origin.push` refspec and a pre-push guard that rejects those refs;
   do not bypass the guard with `--no-verify`.

## 0.3.0 evidence boundary

The K3.1 cutover is a research preview. Keep four evidence classes separate:

| evidence class | current result | boundary |
|---|---|---|
| synthetic/model-free | portable CPU and ROCm suites pass; CPU DSA oracle passes ASan/UBSan | does not by itself establish real-weight generation or quality |
| real 8K Beelink canary | full bundle startup, generation, exact prefix reuse, 128-token decode, live tools/structured JSON, HTTP rejection, and state-v2 recovery pass | one-slot loopback canary, not production evidence |
| natural context | exact `saffron|7319|Nivens` retrieval at 15,993, 31,999, and 63,971 tokens | no context above 64K or continuous long-output qualification |
| production | no production soak has run | no production-readiness claim |

The pinned model is `moonshotai/Kimi-K3.1-Pro` at revision
`6d0a0202956796543364d666ab19c6ad1a494ee1`. Full MZG2 conversion and
verification passed for all 92 routed layers and 82,432 experts. The isolated
official Python SDK fixture passed with `openai 2.52.0`; it is separate from
live agent-client and production evidence.

Official FP8/MXFP4 index-key caches are not implemented or qualified; the
implemented persistent index-key cache is BF16. Historical K3 release and
qualification results are not K3.1 evidence.

## Clean-checkout synthetic gate

From a clean checkout of the exact candidate commit, run and record:

```sh
make -j4 test-cpu
make -j4 test
```

Observed K3.1 cutover results:

- `make -j4 test-cpu`: PASS;
- `make -j4 test`: PASS on AMD Radeon 8060S `gfx1151`;
- the CPU DSA oracle also passed ASan/UBSan.

The DSA GPU tests cover LayerNorm/scoring, top-k boundaries/ties,
negative-infinity padding, sparse attention, and decode/prefill equivalence.
These are synthetic/model-free results. Do not relabel synthetic DSA equivalence
as real-weight equivalence.

The portable suite includes synthetic checkpoint-inspector and placement-planner
fixtures. Their individual targets are:

```sh
make test-k31-checkpoint
make test-placement
```

Record the commit, pinned model revision, hardware, kernel, ROCm version, and
test logs. A clean build alone is not model qualification.

## Pinned source and MZG2 bundle reproduction

Inspect the complete official 96-shard source and create a private source
manifest:

```sh
export MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3.1-Pro
export MOONSHINE_BUNDLE=/path/to/private/Kimi-K3.1-Pro-Moonshine-MZG2

python3 tools/k31_checkpoint.py "$MOONSHINE_MODEL" \
  --verify-sha256 > /private/k31-source-manifest.json
```

Build and independently verify the standalone bundle:

```sh
python3 tools/build_mzg2_bundle.py build \
  --model "$MOONSHINE_MODEL" --out "$MOONSHINE_BUNDLE"
python3 tools/build_mzg2_bundle.py verify \
  --bundle "$MOONSHINE_BUNDLE" --model "$MOONSHINE_MODEL"
```

The accepted K3.1 conversion verified all 92 routed layers and 82,432 experts,
and the resulting bundle completed real startup and generation. Bundle mode
must retain the pinned source/layout identity and state-v2 compatibility. Old
K3 tensor counts, payload sizes, CRCs, MZG2 manifests, and timings are not
acceptance criteria.

Synthetic placement-planner success does not qualify a physical multi-drive
store. Record actual inventory/content hashes, whole-layer placement, drive
reserves, failure behavior, and restore/clean-machine tests before making a
multi-drive deployment claim.

## Real-weight component gates

Use the existing targets when their specific coverage is required for a release
claim, and record their K3.1 results independently:

```sh
make test-model-layout MOONSHINE_MODEL="$MOONSHINE_MODEL"
make test-tokenizer MOONSHINE_MODEL="$MOONSHINE_MODEL"
make test-model-components MOONSHINE_MODEL="$MOONSHINE_MODEL"
make test-engine-init MOONSHINE_MODEL="$MOONSHINE_MODEL"
make test-engine-hello MOONSHINE_MODEL="$MOONSHINE_MODEL"
make test-chat-hello MOONSHINE_MODEL="$MOONSHINE_MODEL"
make test-state-checkpoint \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_STATE_DIR=/tmp
make test-prefix-checkpoint MOONSHINE_MODEL="$MOONSHINE_MODEL"
```

Do not infer a pass for an unlisted component gate from the real canary. Require
explicit K3.1 expected outputs and a new memory ledger; do not reuse K3 hello
tokens, state sizes, context limits, or timings as acceptance criteria.

## Real 8K server, agentic, and state-v2 gates

The verified bundle launches with:

```sh
./moonshine-server "$MOONSHINE_BUNDLE" \
  --host 127.0.0.1 --port 8080 \
  --api-key "$MOONSHINE_API_KEY"
```

The current 8K Beelink canary loaded in 102.243 seconds. Real generation returned
`hello` normally, decoding 20 tokens at 0.489 tok/s; a preserved-thinking
follow-up reused 114 cached tokens. A separate 128-token sustained decode ran at
0.501 tok/s and stopped at the configured cap with `length`.

Live tool calling, tool-result continuation, and structured JSON passed. Public
`medium`, image, and old-model requests returned HTTP 400. Discovery and every
request/response must use `moonshotai/Kimi-K3.1-Pro`; the version is
`0.3.0-k3.1-research-preview`. Verify default reasoning `max`, accepted
`low`/`high`/`max`, and rejection of public `medium`. Internal XTML support for
`medium` is not a public API qualification.

Real state-v2 displacement/restart and exact continuation passed, including 161
reused tokens after restart. Header, stale-model, payload, corruption, and
truncation rejection preserved the causal-state boundary. Repeat the applicable
gates on the exact release candidate. State imports must validate format,
identity, lengths, and checksums before mutation; v1 files are rejected. Compare
all five causal-state digests where digest comparison is used: KDA recurrence,
KDA convolution, MLA, DSA index-key cache, and AttnRes.

The isolated official SDK fixture passed with `openai 2.52.0`. Reproduce it with:

```sh
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r tests/requirements-sdk.txt
make test-openai-sdk PYTHON=.venv-sdk/bin/python
```

This is a replay fixture, not a live SDK agent session. Pi Agent, Hermes Agent,
and other third-party clients remain unqualified. Verify health/model discovery,
JSON and SSE behavior, cancellation, busy rejection, graceful termination, and
exact prefix mismatch fallback for any broader serving claim.

## Natural-context and performance reproduction

Run each natural-text gate independently against the pinned source or matching
verified bundle:

```sh
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=16000
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=32000
make test-long-context-retrieval \
  MOONSHINE_MODEL="$MOONSHINE_MODEL" MOONSHINE_RETRIEVAL_TARGET=64000
```

Accepted results all returned `saffron|7319|Nivens` exactly:

| prompt tokens | prefill | 17-token decode |
|---:|---:|---:|
| 15,993 | 6.545 tok/s | 0.478 tok/s |
| 31,999 | 5.438 tok/s | 0.462 tok/s |
| 63,971 | 4.397 tok/s | 0.454 tok/s |

The 63,971-token run started in 93.976 seconds, prefilled in 14,547.439 seconds,
and decoded 17 tokens in 37.485 seconds. Reported prefill phases were KDA
2,585.803 seconds, DSA 8,101.056 seconds, expert stream 3,463.059 seconds, and
MoE tail 248.327 seconds. These are measurements from that run, not production
performance promises.

The implemented DSA path uses a BF16 index cache, identity-window indexer/top-k
skipping, dense MLA from the measured 768-row threshold, and one selected-row
gather above 2,048 keys. Named DSA steps preserve exact failure diagnostics.
Reproduce the real-shape kernel screen:

```sh
make bench-k31-dsa
```

The 8K sparse-to-dense comparison measured 4.930 → 0.126 ms per layer. BLAS-backed
indexer scoring measures 1.094 → 0.075 ms at 8K and passes the real 8K retrieval
gate. These are kernel and retrieval results, not matched end-to-end speedups.
Future performance changes require fresh exact-output/state and comparative timing
gates.

## Remaining context, output, and production gates

1. Qualify every advertised context above 64K independently. The current
   natural-text boundary is 63,971 tokens; a configured ceiling is not filled-
   context or quality evidence.
2. Include the persistent BF16 DSA index-key cache, DSA top-k/attention
   workspace, KDA/AttnRes state, routed-expert cache, static residency, and
   staging in each memory ledger. Test cold and repeat independent requests
   where cache-backed workspace borrowing is involved.
3. Qualify continuous long-output generation separately. Neither the 128-token
   canary nor the 17-token long-context decodes establishes a continuous 64K
   output profile.
4. Perform a separately approved production soak with the exact candidate,
   model revision, context/output profile, checkpoint settings, and client
   workload before any production-readiness statement.
5. Keep research-preview status until the applicable failure paths, performance
   claims, and production observation pass. Do not advertise historical K3
   128K/30-expert or 65,536-output results as K3.1 profiles.

## Diagnostics and privacy

Opt-in decode/prefill diagnostics remain qualification-only. Use fresh
single-request processes and private output prefixes outside the checkout.
Treat route traces, router logits, causal-state fingerprints, request fixtures,
and generated responses as sensitive derived workload data.

State-digest comparisons include the DSA index-cache digest in addition to KDA,
convolution, MLA, and AttnRes. DSA counters and cache/workspace accounting must
be retained in the qualification record. Do not log prompts, generated text,
tool arguments, request bodies, credentials, or API keys.

## Final audit

Run the final repository audit on the exact candidate:

```sh
git status --short
git grep -nE '(token|password|secret|api[_-]?key)' -- \
  ':!RELEASING.md' ':!SECURITY.md'
git grep -n '/home/' -- README.md docs .github
```

Review `LICENSE`, `NOTICE`, `docs/provenance.md`, `CHANGELOG.md`, and
`CITATION.cff`. Confirm that the README independence disclaimer and the
synthetic, real-canary, natural-context, and production evidence boundaries
remain visible. Preserve historical K3 changelog/qualification entries as
history rather than rewriting them into K3.1 evidence.

## Tagging

Update the version in `moonshine_version.h`, `CHANGELOG.md`, and
`CITATION.cff`, commit the release, then create a signed annotated tag. For this
cutover the coordinated version string is:

```sh
VERSION=0.3.0-k3.1-research-preview
git tag -s "$VERSION" -m "Moonshine $VERSION"
```

The citation release date and changelog entry must match the actual publication
date. Do not tag or publish model weights, derived expert stores, private
qualification data, or bundle payloads in the Moonshine Git repository.
