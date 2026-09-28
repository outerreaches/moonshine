# Documentation index

Start with [getting-started.md](getting-started.md), then
[architecture.md](architecture.md). The rest is grouped by what it is for.
[The README's lane table](../README.md#which-model-are-you-running) decides which
of these apply to the binary you are running — the three lanes do not share an
entry point.

## Using it

| Doc | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | Build, obtain weights, first run |
| [architecture.md](architecture.md) | Engine structure and the decode/prefill paths |
| [agentic-api.md](agentic-api.md) | Function tools, preserved thinking, structured output |
| [deployment-profiles.md](deployment-profiles.md) | Serving profiles and agent clients |
| [observability.md](observability.md) | Production lifecycle logging, and what it excludes |
| [offline-decode-cache-analysis.md](offline-decode-cache-analysis.md) | Replaying a decode trace offline |

## Kimi K3 qualification records

[qualification-128k.md](qualification-128k.md),
[qualification-filled-context.md](qualification-filled-context.md),
[qualification-output-64k.md](qualification-output-64k.md),
[qualification-decode-diagnostics.md](qualification-decode-diagnostics.md),
[qualification-observability.md](qualification-observability.md).

Each states its criteria, what passed, and — usually the more useful half — what
it does **not** license. They are dated records rather than current
documentation; where one disagrees with the code, the code is newer.

## MiMo V2.6 Flash

| Doc | What it covers |
|---|---|
| [mimo26-bringup.md](mimo26-bringup.md) | The artifact, its schema, and the layout traps found auditing it |
| [mimo26-precision-contract.md](mimo26-precision-contract.md) | Arithmetic decisions and what must stay bit-exact |
| [mimo26-production-candidate.md](mimo26-production-candidate.md) | Candidate qualification — explicitly not a deployment recommendation |
| [mimo26-full-model-divergence.md](mimo26-full-model-divergence.md) | A resolved full-model divergence investigation |

Packaging, activation, signing and rollback are in
[../RELEASING.md](../RELEASING.md).

## GLM 5.3 Flash

[glm53-phase4-goldens.md](glm53-phase4-goldens.md) — why the Phase-4 goldens are
regenerated rather than committed, and how. The lane itself is **not servable
from this tree**; see [GLM 5.3 Flash status](../README.md#glm-53-flash-status).

## Provenance

[provenance.md](provenance.md) — source lineage, pinned model revisions, design
influences, validation oracles, and the AI-assistance disclosure. Read it before
reusing anything here.

## Two conventions worth knowing

**"The vault", and dated note titles.** Several documents cite notes by a dated
title, sometimes as "the vault". That is the maintainer's Obsidian engineering
notebook, kept outside this repository. Those citations record *where a claim
came from*; they are not files you are missing, and nothing here depends on them.

**`Evidence/…` paths.** Qualification runs emit reports, pinned builds and logs
that are likewise kept outside the tree, identified by the SHA-256 digests quoted
alongside them. `MOONSHINE_EVIDENCE_ROOT` is how the gates in
`tests/run_mimo26_*.py` find a local copy; unset, they refuse with an
instruction rather than a confusing failure.

Both conventions exist because publishing multi-gigabyte run artifacts and a
personal notebook is not a reasonable thing to do to a git repository. The cost
is that some references are provenance only, and this page says so once rather
than each document apologising for it.
