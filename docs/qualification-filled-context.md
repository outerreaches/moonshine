# Filled-context qualification

Moonshine separates configured context capacity from the number of positions
actually filled. A large allocation proves only that the state fits; a filled
run is required to measure prefill scaling and lock the model result.

## Method

The scale fixture repeats its locked 24-token chat prompt until it reaches the
requested token count, then performs one selected-expert range prefill. It uses
Q8 activations, a 32-token chunk, borrowed expert-cache memory for the range
workspace, and the production selected-I/O path. Qualification requires:

- an exact greedy token and value;
- a complete phase-time ledger;
- exact selected-expert read counts and bytes;
- workspace and attention-state allocation evidence;
- host memory, swap, SSD temperature, and SMART evidence.

This fixture measures execution and numerical stability. Repetition is not a
semantic long-context test, so natural-text retrieval remains a separate gate.

## Accepted 16K result

The default path completed 16,384 filled positions in a 16,384-position
context:

| Measurement | Result |
| --- | ---: |
| Wall time | 2,066.332 s |
| Throughput | 7.929 tok/s |
| Greedy token / value | `6244` / `26.875` |
| Routed layer sweeps | 92 |
| Selected read requests | 60,366 |
| Selected read bytes | 1,059,505,184,536 |
| Unique experts, min / mean / max | 276 / 656.2 / 889 |
| Borrowed range workspace | 17.034 GiB |
| Startup allocation | 39.101 GiB |

The phase ledger reconciled 2,066.330 of 2,066.332 seconds:

| Phase | Seconds |
| --- | ---: |
| Layer 0 | 22.049 |
| Attention | 1,098.760 |
| KDA within attention | 666.893 |
| MLA within attention | 431.868 |
| Router | 15.094 |
| Expert streaming | 872.429 |
| MoE tail | 57.987 |
| Output | 0.010 |

Within KDA, projection was 462.683 seconds, convolution 1.562, recurrence
82.246, gate/norm 0.933, and output 119.119. Within expert streaming, read
wait was 40.451 seconds, submission 4.661, index construction 2.386, and the
expert pipeline 824.728.

## Scaling from selected 8K

| Measurement | Filled 8K | Filled 16K | Ratio |
| --- | ---: | ---: | ---: |
| Wall time | 1,008.104 s | 2,066.332 s | 2.050x |
| Throughput | 8.126 tok/s | 7.929 tok/s | 0.976x |
| Selected bytes | 1,009,747,090,312 | 1,059,505,184,536 | 1.049x |
| Read requests | 57,531 | 60,366 | 1.049x |
| Expert pipeline | 413.136 s | 824.728 s | 1.996x |
| Attention | 497.989 s | 1,098.760 s | 2.206x |
| Layer 0 | 11.798 s | 22.049 s | 1.869x |
| Router | 7.601 s | 15.094 s | 1.986x |
| MoE tail | 23.876 s | 57.987 s | 2.429x |

Selected weight traffic grows only 4.9% because the union is already dense at
8K. The expert pipeline scales essentially linearly with positions. Attention
is the first important superlinear pressure and should be investigated before
treating a 32K result as merely a longer copy of the same run.

## Host evidence

The unloaded host still reported approximately 122 GiB available. Swap use
grew by 11,776,000 bytes during the run; cumulative counters moved by 5.40 MiB
in and 27.44 MiB out. Sampled SSD temperatures peaked at 56 C composite and
63 C on sensor 2. SMART reported no warning, critical-temperature time, media
errors, or error-log entries. The run read approximately 1.171 TB of SSD data
according to the device data-unit counter.

## 32K decision boundary

The payload-free 32K plan passes: it borrows 34.069 GiB of range workspace,
allocates 864 MiB for MLA append state, stays within the 48.111 GiB expert
cache, and requests no new device allocation. Based on the 16K attention
curve, an execution is expected to take roughly 75--85 minutes.

Before spending that run, add and pass a deterministic natural-text retrieval
probe at 16K. Then either investigate the attention scaling or deliberately
collect the 32K execution as a baseline. The 32K point is planned, not yet
qualified.
