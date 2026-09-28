# GLM-5.3 Phase-4 goldens: provenance in the repository, bytes out of it

The Phase-4 official gates compare against golden arrays computed by running
**official `zai-org/GLM-5.3-Flash` weights** at revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. Those arrays are model-derived data
governed by the publisher's terms, not by this repository's MIT licence, so the
bytes are not committed here.

What *is* committed is everything needed to reconstruct and then check them:
the pinned source identity, the shard and tensor names, the config and index
digests, the synthetic input formulas, the tolerances, and the SHA-256 of each
golden. A locally regenerated golden is therefore verified against the same pin
the original was — removing the bytes did not weaken the gate.

## The three goldens

| Golden | Bytes | Regenerate |
|---|---:|---|
| `tests/fixtures/glm53_phase4_components.bin` | 207,756 | `make` rule, via `tests/generate_glm53_phase4_component_fixture.py` |
| `tests/fixtures/glm53_phase4_kda_v1.bin` | 143,928 | `make` rule, via `tests/generate_glm53_phase4_kda_fixture.py` |
| `tests/fixtures/glm53_official_projection_f32.bin` | 2,048 | **no generator in this tree** |

The first two regenerate automatically:

```sh
GLM53_OFFICIAL_ROOT=/path/to/verified/official make test-glm53-phase4-official
```

Both generators refuse a checkpoint whose `.provenance/source_identity.json`,
`config.json` or `model.safetensors.index.json` digests disagree with the pin, so
a wrong or modified checkpoint fails before it can write a plausible-looking
golden. The component generator rewrites
`tests/fixtures/glm53_phase4_components.json` alongside the binary, which is how
the recorded digest stays consistent with the bytes.

## The projection golden is a genuine gap

`glm53_official_projection_f32.bin` is 512 float32 values — one
`kv_a_proj_with_mqa` projection of a synthetic BF16-exact input at layer 3. No
code in this tree produces it. Its digest pins a specific reduction order
(`official_projection.diagnostic_sequential_f32_sha256` equals
`reference_f32_sha256` by construction), and a reimplementation that summed in a
different order would produce a correct-looking array with the wrong hash.

So `test-glm53-phase4-official` **skips that one sub-check, loudly**, and runs the
other four. Writing the generator — dequantising the pinned FP8 weight with its
block scales and reducing in the recorded order — is the work that closes this,
and it wants the same treatment as any other oracle: derive it from the spec,
then require the pinned digest to match rather than adjusting the digest to the
new code.

## Why not just commit them

`tests/fixtures/*.json` here carries metadata, digests, tolerances, synthetic
input formulas, and a handful of expected scalars (eight router expert IDs, a
selection-boundary margin, the first and last eight projection values). That is
provenance and test expectation. The `.bin` payloads are bulk model-derived
output, which is a different kind of artifact, and the distinction is worth
keeping even when the payload is small: it means "may this be published?" has one
answer for the whole repository rather than one per file.
