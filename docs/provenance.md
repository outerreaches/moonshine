# Provenance and acknowledgements

This document separates source lineage from design influence and validation
references. Those are not interchangeable.

## Source lineage

The initial standalone repository was extracted from the K3 work developed in
the MIT-licensed [DwarfStar / ds4](https://github.com/antirez/ds4) tree:

- DS4 development base: `0a7ad776b9068348e6cb09df8cafa9cadd285298`;
- first K3 component commit:
  `30069e829f665830bc3dc123a100426ae241dc5d`;
- extracted K3 checkpoint:
  `22c99021b54735b589b55485a945e8154a94224f`.

The DS4 development base is an upstream DS4 commit. The two K3 commits are this
maintainer's own work on a DS4 fork and were never part of the public DS4 tree.
The public `antirez/ds4` source inspected at
`54b36ed9ba42da31b24f2d1a5feb075c2475dbb1` contained no K3 backend, so no
upstream K3 implementation existed to copy or adapt.

The extracted repository retains the DS4 and GGML author notices in
[LICENSE](../LICENSE). DS4 established the small native C/CUDA-style codebase,
ROCm conventions, quantized-kernel patterns, and model-specific performance
posture in which this engine was built.

DS4 itself acknowledges
[llama.cpp](https://github.com/ggml-org/llama.cpp) and
[GGML](https://github.com/ggml-org/ggml) for quantization layouts, kernels,
tests, and inference engineering knowledge. The retained GGML notice reflects
that lineage.

No third-party source tree is vendored in this repository.

## Model and architecture references

The engine implements the tensor layout and graph described by:

- the [official Kimi K3 model release](https://huggingface.co/moonshotai/Kimi-K3),
  pinned locally at
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`;
- the [official Moonshot Kimi K3 reference repository](https://github.com/MoonshotAI/Kimi-K3),
  inspected at `7c5be9599120d7993748de66a76128614f15f210`.

The model weights are not included and are not covered by this repository's
MIT license.

## Validation oracles

The following projects informed validation but are not runtime dependencies:

- [llama.cpp](https://github.com/ggml-org/llama.cpp), Kimi K3 PR #26185 at
  `cf67f0d24511864d2d3da0769108fd6fc16d00d1`, supplied the initial
  conversion, tokenizer/chat, full-model generation, and ROCm comparison
  oracle;
- [vLLM](https://github.com/vllm-project/vllm) at
  `bb3b61f2fd2333ab165ebaba13f133db4210b9f2` and
  [SGLang](https://github.com/sgl-project/sglang) at
  `7f438a6031ec5790c2f60584f0c172f68eceecc8` were inspected as officially
  documented K3 runtimes, not selected as the `gfx1151` backend.

The executable tests in this repository contain independent small CPU/device
oracles and locked real-weight hashes so the native path can be checked without
depending on those frameworks at runtime.

The tokenizer and text-only XTML implementation follows the official
`tokenization_kimi.py`, `encoding_k3.py`, `tokenizer_config.json`, and
`tiktoken.model` shipped at the pinned model revision. The official Python
implementation generated the locked multilingual and chat token-ID oracles;
it is not a runtime dependency.

## Design influence from earlier local work

The strongest operational influence is the preceding GLM 5.2 SSD-streaming
program in DS4. The K3 engine carries forward its measured lessons:

- split resident static/model-control weights from routed expert payloads;
- never pass file-backed mmap pages directly to HIP transfers on this host;
- use bounded pinned or mapped staging;
- plan memory and physical I/O before allocation;
- keep a persistent, layer-aware online expert cache;
- make decode and prefill separate schedules;
- use large layer-major prefill chunks to amortize routed sweeps;
- preserve hot cache state across agentic turns;
- prioritize exact completed-turn state reuse over repeatedly prefilling the
  same history.

Those are scheduling and systems-design lessons. The K3 SafeTensors loader,
cache policy, engine composition, and layer-major implementation in this
repository are K3-specific code.

## Platform dependencies

The runtime directly uses:

- AMD HIP;
- hipBLAS and hipBLASLt;
- ICU's Unicode regular-expression API;
- Linux `io_uring` syscalls;
- Linux `O_DIRECT`;
- the C standard library and POSIX APIs.

Their licenses and distribution terms remain separate from this repository.

## AI assistance disclosure

The engine and its documentation were produced with strong assistance from AI
coding and review systems. The human maintainer selected the architecture,
directed experiments, reviewed changes, operated the hardware, and made
correctness and release decisions.
