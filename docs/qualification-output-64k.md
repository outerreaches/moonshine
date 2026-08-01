# 64K output and medium-reasoning qualification

Date: 2026-08-01

This gate qualifies a 65,536-token maximum request budget and K3's native
`reasoning_effort: medium` on the persistent 128K/30-expert service profile.
It qualifies API admission and capacity accounting, not a continuous 64K
decode or a filled 128K prompt.

## Scope

- AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)
- Linux 7.0.0-28-generic and HIP 7.2.53150
- pinned official 96-shard Kimi K3 checkpoint
- Q8 resident static tier, 30 cached experts per layer, 16 staging slots
- 131,072 configured context positions and 65,536 output-token ceiling
- local Samsung 990 PRO, ext4 `noatime`, raw `io_uring` QD2
- one model process and no competing model transfer

The clean build passed the portable C suite, ROCm component suite, native
tokenizer/XTML fixture, ASan/UBSan request parser, and the pinned official
OpenAI Python SDK 2.52.0 SSE replay before live inference.

## Discovery and admission

The server loaded in 39.078 seconds and reported:

```json
{"context_length":131072,"max_output_tokens":65536,"slots":1}
```

Both `/health` and `/v1/models` exposed the same limits. A request at 65,537
failed before inference with HTTP 400 and the exact error:

```text
max tokens must be an integer in [1,65536]
```

The otherwise identical 65,536-token request was accepted with
`reasoning_effort: medium`.

## Persistent 128K/30-expert requests

Two independent requests completed in one process with no session-prefix
reuse and no memory-guard rejection:

| request | prompt | prefill | generated | decode | finish |
|---:|---:|---:|---:|---:|---|
| `Say hello.` | 92 | 91.306 s / 1.008 tok/s | 43 | 99.121 s / 0.434 tok/s | `stop` |
| `What model are you?` | 99 | 96.370 s / 1.027 tok/s | 79 | 182.326 s / 0.433 tok/s | `stop` |

The first response returned `Hello!` and the second identified Kimi and
Moonshot AI. Both kept reasoning in `reasoning_content` and ordinary response
text in `content`.

After the second request, `MemAvailable` was 12 GiB and swap remained at its
pre-existing 1.8 GiB. The Samsung 990 PRO reported no critical warning,
0 percent used, a 57 C composite temperature, and a 61 C hottest sensor.

## Remaining-context clamp

A temporary 128-position service was started with the same configured
`--max-output-tokens 65536`. Discovery correctly advertised an effective
maximum of 128. A 92-token `medium` request asking for 128 completion tokens
returned HTTP 200 with `finish_reason: length` after 23 reported generated
tokens. The remaining 13 positions were reserved for K3's required structural
trailers instead of overrunning the context.

## Accepted boundary

Accepted for Moonshine 0.2.0:

- absolute server ceiling of 65,536 output tokens;
- exact-ceiling OpenAI request admission and one-token-over rejection;
- `low`, `medium`, `high`, and `max` reasoning effort values;
- context-aware discovery and generation clamping;
- repeat short requests at 128K capacity with 30 expert slots.

Still outside this qualification:

- a continuous 65,536-token decode;
- filled 64K or 128K prompt execution and filled-128K quality;
- contexts above 128K on the qualified 128 GB host;
- concurrent inference slots.
