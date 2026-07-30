# Releasing Moonshine

## Repository setup

1. Create a public repository with the slug `moonshine-k3`.
2. Use the display name **Moonshine** and the description in
   `docs/release-plan.md`.
3. Enable private vulnerability reporting.
4. Enable secret scanning and push protection.
5. Protect `main` and require the portable CI workflow.
6. Add the local remote only after reviewing the initial public repository.

## Release qualification

From a clean checkout of the exact candidate commit:

```sh
make test-cpu
make
make tests
make test
make test-model-layout MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-tokenizer MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-engine-hello MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
make test-chat-hello MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3
```

Also verify `moonshine-server` health, model discovery, JSON completion, and
SSE completion on the qualified host. Record the commit, model revision,
hardware, kernel, ROCm version, context, memory ledger, and timings.

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
git tag -s 0.1.0-research-preview \
  -m "Moonshine 0.1.0 research preview"
```

Do not tag or publish model weights.
