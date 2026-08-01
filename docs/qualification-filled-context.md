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
semantic long-context test, so natural-text retrieval is a separate gate.

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
| Startup time | 39.101 s |

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

## Accepted 16K natural-text retrieval

The semantic gate builds 389 deterministic operational records and renders a
15,993-token non-thinking chat prompt. Three explicitly labeled retrieval keys
sit at records 49, 195, and 341, approximately 12.5%, 50%, and 87.5% through
the record stream. Routine record numbers and shipment codes act as
distractors. The exact required answer is `saffron|7319|Nivens`.

The accepted execution returned that exact decoded response with a natural
end-of-message stop, 17 generated tokens, no forced trailer, no reasoning, and
no tool calls:

| Measurement | Result |
| --- | ---: |
| Prompt prefill | 2,018.210 s / 7.924 tok/s |
| Decode | 39.784 s / 0.427 tok/s |
| Selected read requests | 79,858 |
| Selected read bytes | 1,401,616,232,728 |
| Unique experts, min / mean / max | 721 / 868.0 / 895 |
| Attention | 1,063.846 s |
| Expert streaming | 870.347 s |
| Expert pipeline | 815.293 s |
| Read wait | 44.433 s |

Natural text produces a much denser route union than the repeated 16K
fixture: selected traffic reaches 96.9% of the full-store ceiling. Even so,
throughput remains within 0.1% of the repeated-token result, and the complete
phase ledger reconciles 2,017.949 of 2,017.951 range seconds.

This gate also exposed and fixed a result-boundary regression. Streaming had
emitted the correct ordinary response, but the post-generation non-thinking
XTML parser skipped those leading text tokens and replaced the returned result
with an empty buffer. Commit `04df2a3` scopes the initial ordinary-token scan
to thinking mode and adds a direct parser oracle. Inference was correct; the
returned non-streaming/native result was not.

During the definitive run, swap allocation grew by 11,399,168 bytes;
cumulative counters moved by 10.05 MiB in and 20.58 MiB out. Sampled SSD
temperature peaked at 57 C composite / 67 C sensor 2. SMART retained zero
warning/critical-temperature time, media errors, and error-log entries. The
device data-unit delta was approximately 1.751 TB read.

## Accepted 32K result

The unmodified production path completed 32,768 filled positions in a
32,768-position context:

| Measurement | Result |
| --- | ---: |
| Wall time | 4,391.059 s |
| Throughput | 7.462 tok/s |
| Greedy token / value | `40493` / `28.25` |
| Routed layer sweeps | 92 |
| Selected read requests | 62,398 |
| Selected read bytes | 1,095,169,542,448 |
| Unique experts, min / mean / max | 286 / 678.2 / 893 |
| Borrowed range workspace | 34.069 GiB |
| Startup time | 39.213 s |

The phase ledger reconciled 4,391.057 of 4,391.059 seconds:

| Phase | Seconds |
| --- | ---: |
| Layer 0 | 44.803 |
| Attention | 2,517.497 |
| KDA within attention | 1,306.586 |
| MLA within attention | 1,210.912 |
| Router | 29.365 |
| Expert streaming | 1,628.368 |
| MoE tail | 171.013 |
| Output | 0.010 |

Within KDA, projection was 901.256 seconds, convolution 3.128, recurrence
164.600, gate/norm 1.864, and output 234.839. Within expert streaming, read
wait was 36.421 seconds, submission 5.950, index construction 5.118, and the
expert pipeline 1,580.662.

## Scaling from selected 16K

| Measurement | Filled 16K | Filled 32K | Ratio |
| --- | ---: | ---: | ---: |
| Wall time | 2,066.332 s | 4,391.059 s | 2.125x |
| Throughput | 7.929 tok/s | 7.462 tok/s | 0.941x |
| Selected bytes | 1,059,505,184,536 | 1,095,169,542,448 | 1.034x |
| Read requests | 60,366 | 62,398 | 1.034x |
| Expert pipeline | 824.728 s | 1,580.662 s | 1.917x |
| Attention | 1,098.760 s | 2,517.497 s | 2.291x |
| KDA | 666.893 s | 1,306.586 s | 1.959x |
| MLA | 431.868 s | 1,210.912 s | 2.804x |
| Layer 0 | 22.049 s | 44.803 s | 2.032x |
| Router | 15.094 s | 29.365 s | 1.945x |
| MoE tail | 57.987 s | 171.013 s | 2.949x |

Throughput retains 94.1% of the 16K rate. Selected weight traffic grows only
3.4%, and the expert pipeline remains slightly sublinear. Attention now takes
57.3% of total prefill. KDA remains approximately linear, while MLA grows
2.804x and is the primary 32K optimization target. The MoE tail also needs
profiling before attempting a larger filled-context performance arm.

The unloaded host reported approximately 122 GiB available; 12 GiB remained
available while the range workspace was borrowed. Swap allocation fell by
2,506,752 bytes over the run, while cumulative counters moved by 13.58 MiB in
and 0.70 MiB out. Sampled SSD temperatures peaked at 56 C composite and 60 C
on sensor 2. SMART retained zero warning/critical-temperature time, media
errors, and error-log entries. The SSD data-unit delta was approximately
1.207 TB read and zero units written.

The expensive execution completed immediately before its exact 32K assertion
was added to the scale fixture; the production path and fixture inputs are
otherwise source-equivalent. The separate semantic 32K quality gate below
provides the functional qualification for this context tier.

## Accepted 32K natural-text retrieval

The semantic fixture scales to 781 deterministic operational records and a
31,999-token prompt. Its three needles are records 98, 391, and 684, again
approximately 12.5%, 50%, and 87.5% through the record stream. The exact
answer remains `saffron|7319|Nivens`.

The production selected-prefill path returned that exact response with a
natural end-of-message stop, 17 generated tokens, no forced trailer, no
reasoning, and no tool calls:

| Measurement | Result |
| --- | ---: |
| Prompt prefill | 4,273.283 s / 7.488 tok/s |
| Decode | 40.078 s / 0.424 tok/s |
| Selected read requests | 80,767 |
| Selected read bytes | 1,417,570,415,176 |
| Unique experts, min / mean / max | 740 / 877.9 / 895 |
| Attention | 2,429.727 s |
| KDA within attention | 1,278.750 s |
| MLA within attention | 1,150.977 s |
| Expert streaming | 1,619.315 s |
| Expert pipeline | 1,575.206 s |
| MoE tail | 151.089 s |

The phase ledger reconciled 4,273.074 of 4,273.076 seconds. Compared with the
accepted 15,993-token semantic arm, wall time scales 2.118x, attention 2.284x,
KDA 1.960x, MLA 2.798x, expert pipeline 1.932x, and MoE tail 3.263x.
Throughput retains 94.5%. Selected traffic grows only 1.1% because the 32K
natural route union already reaches 98.0% of the physical routed-store
ceiling.

The run left approximately 122 GiB available after teardown and 12 GiB while
resident. Swap allocation grew by 1,536,000 bytes; cumulative counters moved
by 5.99 MiB in and 5.33 MiB out. Sampled SSD temperature peaked at 59 C
composite / 62 C sensor 2. SMART retained zero warning/critical-temperature
time, media errors, and error-log entries. The device data-unit delta was
approximately 1.767 TB read and zero units written.

This closes the semantic prerequisite through 32K. Filled 64K and 128K remain
deferred while deterministic MLA range work and MoE-tail profiling proceed.
