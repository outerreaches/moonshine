# 128K configured-context qualification

Date: 2026-07-31

This qualification extends the accepted Q8/32 single-node configuration from
32K to a configured capacity of 131,072 positions. It does not claim that a
filled 128K prompt is fast or that long-context quality has been established.

## Scope

- AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)
- Linux 7.0.0-28-generic and ROCm 7.2.53150
- pinned official 96-shard Kimi K3 checkpoint
- Q8 resident static tier, 32 cached experts per layer, 16 staging slots
- local Samsung 990 PRO, ext4 `noatime`, raw `io_uring` QD2
- one model process and no competing model transfer

The first timing pass was discarded after a background file transfer was
discovered. Before the accepted pass, two live one-second `iostat` samples
showed zero I/O on the model NVMe and no transfer or blocked-I/O process.

## Exact memory ledger

Context-dependent state is:

```text
756,547,584 fixed bytes + context * 28,224 bytes
```

At 131,072 positions:

| allocation | bytes | GiB |
|---|---:|---:|
| resident static tier | 59,345,729,536 | 55.270 |
| routed-expert cache | 51,659,145,216 | 48.111 |
| runtime state | 4,455,923,712 | 4.150 |
| mapped staging | 280,821,760 | 0.262 |
| total before allocator/driver overhead | 115,741,620,224 | 107.793 |

The clean run began with 121 GiB `MemAvailable` and 1.6 GiB of pre-existing
swap use. At full 128K residency it retained 9.3–9.4 GiB `MemAvailable`, and
swap remained at 1.6 GiB. The SSD's hottest observed sensor was 65 C; SMART
reported no warning, critical warning, or accumulated warning-temperature
time.

## Matched locked fixture

Both arms used the same 24-token `Say hello.` fixture, Q8 static residency,
32 experts per layer, 16 staging slots, greedy decoding, and an idle model
NVMe.

| configured context | startup | prompt | post-TTFT decode | result |
|---:|---:|---:|---:|---|
| 8,192 | 39.119 s | 54.846 s / 0.438 tok/s | 33.467 s / 0.508 tok/s | PASS |
| 131,072 | 39.227 s | 55.871 s / 0.430 tok/s | 33.461 s / 0.508 tok/s | PASS |

Both contexts produced the same 18 locked token IDs and the same selected
BF16 values. The 1.8% prompt-rate difference is a single-run observation;
startup differed by 0.108 seconds and decode was unchanged.

## OpenAI-compatible API gate

A loopback `moonshine-server --context 131072` loaded in 39.118 seconds and
reported 4.150 GiB of state. `GET /health` and `GET /v1/models` passed. A
non-streaming Chat Completions request returned HTTP 200:

```text
Hello! 👋 How can I help you today?
```

Server telemetry reported 24 prompt tokens in 54.844 seconds (0.438 tok/s)
and 18 generated tokens in 36.114 seconds (0.498 tok/s). Client wall time was
90.952 seconds, and the request ended normally with `finish_reason: stop`.

## Regression coverage

- `MOONSHINE_CONTEXT` now selects context for the full-residency and locked
  hello make targets.
- The full-residency fixture derives its state ledger from the configured
  context rather than assuming 8K.
- The payload-free prefill planner checks a 128K plan without changing routed
  sweep or prompt-workspace accounting.
- The locked hello fixture verifies the same token sequence at 8K and 128K.

During qualification, the synthetic full-residency fixture exposed stale
routed-layer hashes imported from the earlier native prototype. The public
release qualification had run the real locked hello fixture, not this
synthetic fixture. The current public engine repeats a stable routed hash path
and preserves the synthetic greedy token (`220`), with a BF16 score of `6.875`
instead of the imported `6.84375`. The synthetic hashes and score were
re-locked to the current engine; the exact real-chat token/value fixture
remains the stronger end-to-end gate.

## Qualification boundary

Accepted:

- exact allocation and preflight at a configured 131,072 positions;
- short sequential prompt and greedy generation correctness;
- short-request performance parity with the 8K control;
- the loopback OpenAI-compatible JSON path;
- live memory, swap, and SSD thermal headroom on the accepted host.

Not yet accepted:

- a filled or near-filled 128K prompt;
- long-context retrieval or quality;
- state export/import near 128K occupancy;
- concurrent service or transfer pressure;
- contexts above 128K on the 128 GB host.
