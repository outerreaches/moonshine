# Open-source release plan

## Proposed first release

Use a clearly labeled `0.1.0-research-preview`, not a production or general
inference release.

The first public promise should be:

> Reproduce the native Kimi K3 SafeTensors/ROCm engine and its correctness and
> performance fixtures on a qualified 128 GB `gfx1151` host.

It should not promise a chat product, broad ROCm portability, arbitrary Kimi K3
revisions, or compatibility with GGUF quants.

## Ready now

- standalone K3 source and public headers;
- MIT license with inherited DS4/GGML notices;
- explicit source and reference provenance;
- build system independent of the larger DS4 tree;
- model-free primitive/cache tests;
- real-weight component oracles;
- full Q8/32 engine residency and all-layer hashes;
- coherent locked greedy generation fixture;
- exact sequential/range state comparison;
- 512- and 8K layer-major prefill fixtures;
- diagnostic KDA hipBLAS performance path;
- measured memory, I/O, and phase telemetry.
- standalone build of every test executable;
- passing model-free CPU/ROCm suite;
- passing 497,220-tensor layout and payload-free prefill-plan checks;
- passing real-weight Q8 projection oracle.
- versioned, checksummed causal-state export/import;
- exact fresh-engine continuation with an empty expert cache;
- pre-mutation rejection of corrupt, stale-model, and truncated state files.
- native loading of the official TikToken vocabulary;
- exact multilingual pre-tokenizer and text-only XTML parity fixtures;
- a transport-free stateful chat-session API;
- a deterministic interactive/one-shot CLI with exact checkpoint commands;
- a native end-to-end `Say hello.` result at the locked prompt/output rates.
- dynamic MLA cache/workspace allocation qualified at 8K, 16K, and 32K;
- a one-slot OpenAI-compatible HTTP service with JSON and SSE completions;
- bearer authentication guardrails for non-loopback serving.
- portable GCC/Clang CPU-only GitHub Actions workflow;
- contribution, conduct, security, changelog, citation, and release guides;
- issue forms and a pull-request template.

## Required before publishing

- rerun the standalone repository's full-residency fixtures from a clean
  checkout;
- portable CPU-only CI is complete; ROCm remains a documented maintainer
  qualification because hosted runners do not match the target hardware;
- add a release tag and immutable benchmark/source checkpoint;
- confirm every copied/adapted source notice is sufficient;
- confirm the official model license can be linked and that download
  instructions do not imply redistribution;
- issue and pull-request templates are complete;
- contribution and code-of-conduct policies are complete;
- the security policy uses GitHub private vulnerability reporting; enable it
  when creating the public repository;
- rerun secret scanning and ensure no private hostnames, tokens, logs, or model
  paths beyond documented examples are present;
- create the GitHub repository, set DS4 as historical upstream in the
  description, and push only after the local release commit is reviewed.

## Engine milestones after repository extraction

### 1. Exact causal-state export/import — complete

Format v1 is implemented as a versioned, checksummed checkpoint containing:

- KDA recurrent state;
- KDA convolution caches;
- occupied MLA rows;
- AttnRes state;
- token position;
- model revision/layout identity;
- static-mode identity.

Derived MLA packed keys are not serialized. Expert-cache payloads and LRU
metadata remain separate performance state.

The fresh-engine gate matches uninterrupted token values, IDs, all four state
hashes, and the complete three-token continuation. It also verifies that
header corruption, a checksum-valid stale layout identity, payload corruption,
and truncation are rejected before destination state is changed.

The two-position file is 433.569 MiB; the exact 8K projection is 649.516 MiB.
On the qualified host, export took 1.043 seconds and import took 0.841 seconds.

### 2. Natural-text quality gate

Use the native tokenizer and XTML fixtures to compare:

- default Q8 range versus sequential execution;
- diagnostic KDA hipBLAS versus default on multi-token natural text;
- bounded streamed-static BF16 versus Q8 on selected layers or full tokens.

Do not promote hipBLAS or future Q4 static weights on a same-greedy-token
observation alone.

### 3. Usable application surface

With exact state reuse locked:

- integrate the official tokenizer and XTML prompt renderer — text-only
  system/user/assistant messages are complete;
- add a minimal CLI for deterministic text generation — complete;
- add a one-slot OpenAI-compatible HTTP service — complete for text-only
  Chat Completions, JSON, and SSE;
- add sampling only after greedy parity remains locked;
- extend XTML for tool declarations/calls/results and structured responses;
- persist semantic state atomically with checksum verification;
- expose cache, disk, PSI, startup, TTFT, prefill, and decode telemetry.

### 4. Portability

Treat every new ROCm architecture as a correctness port:

- compile;
- run model-free operations;
- run real-weight components;
- verify complete state hashes and generation;
- profile and select architecture-specific kernels;
- publish results only after exactness gates pass.

CUDA, Vulkan, Metal, GGUF, multi-node execution, and concurrency are separate
projects, not implicit parts of the first release.

## Versioning and branch posture

- `main`: always builds; default path remains the accepted exact path;
- short-lived feature branches for checkpointing, CLI, and new backends;
- diagnostic numerical variants remain explicitly named and opt-in;
- release tags use SemVer after `0.1.0-research-preview`;
- benchmark tables name hardware, software pins, model revision, context,
  static mode, cache size, and whether results are default or diagnostic.

## Suggested GitHub identity

Repository slug: `moonshine-k3`

Display name: `Moonshine`

Description:

> Moonshine is an experimental single-node SafeTensors/ROCm inference engine
> for Kimi K3 on 128 GB AMD Strix Halo, with NVMe-streamed MXFP4 experts and
> layer-major prefill.

The README must retain the independence disclaimer: Moonshine is a nod to
Moonshot AI, not an affiliated or endorsed Moonshot project.

## Suggested repository topics

`kimi-k3`, `rocm`, `strix-halo`, `mixture-of-experts`, `safetensors`,
`io-uring`, `local-llm`, `inference-engine`
