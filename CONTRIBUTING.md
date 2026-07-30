# Contributing to Moonshine

Moonshine is an early, model-specific research engine. Contributions are
welcome when they preserve its explicit memory, I/O, and correctness model.

## Before opening a change

- Use an issue for design changes, new backends, quantization changes, or
  changes that can alter generated output.
- Keep model weights, private host details, credentials, and large benchmark
  artifacts out of the repository.
- Do not present unqualified hardware or model revisions as supported.
- Keep third-party code provenance and license notices explicit.

## Build and test

Portable changes must pass:

```sh
make test-cpu
```

ROCm changes must also build every test and pass the model-free device suite:

```sh
make
make tests
make test
```

Changes to model execution should run the smallest relevant real-weight
oracle. Performance changes must report the hardware, ROCm version, context,
static precision mode, expert-cache shape, before/after timing, and whether
tokens, values, or state hashes changed.

## Pull requests

Keep each pull request focused. Describe:

- the problem and chosen approach;
- tests and hardware actually used;
- memory or I/O ledger changes;
- numerical or semantic differences;
- source lineage for adapted code.

AI-assisted contributions are allowed, but the contributor remains
responsible for reviewing, testing, licensing, and accurately describing the
change. Note material AI assistance in the pull-request description.

Moonshine uses the Developer Certificate of Origin. Add a `Signed-off-by`
trailer to each commit:

```sh
git commit -s
```

By contributing, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
