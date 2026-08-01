# Releasing Moonshine

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
make test-model-layout MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-tokenizer MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-reduction-qualification MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-chat-hello MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

Also verify `moonshine-server` health, model discovery, JSON completion, and
SSE completion on the qualified host. Confirm pre-TTFT SSE keepalives, a
warmer second stateless request, exact append-prefix reuse under
`X-Moonshine-Session`, and mismatch fallback. For a raised-ceiling candidate,
confirm health/model metadata, acceptance at the configured ceiling, rejection
one token above it, explicit-null fallback, and remaining-context clamping.
The admission gate may stop naturally; it does not need to generate the entire
configured maximum. Record the commit, model revision, hardware, kernel, ROCm
version, context, output ceiling, memory ledger, and timings.

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
