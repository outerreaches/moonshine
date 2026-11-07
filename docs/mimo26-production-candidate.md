# MiMo local production candidate — qualification, not deployment

> **`Evidence/…` paths below are not in this repository.** Qualification runs
> produce reports, pinned builds and logs that are kept outside the tree; the
> digests quoted here identify them, and `MOONSHINE_EVIDENCE_ROOT` is how the
> gates in `tests/run_mimo26_*.py` locate a local copy. Read those references as
> provenance for a claim, not as files to open.

## Current status — September 24 takeover

No final production identity is promoted. Commit5881291 fixed an undersized
prefill hidden buffer; earlier artifacts below are historical evidence, not
safe rollback/deployment recommendations. The corrected pooled worker is now
being requalified from a fresh isolated build, first at cache16/context1024/
chunk64, then cache128/context2048/chunk128, with explicit lookahead and raw
official packed weights. No MZG2 serving integration or remote exposure.

The corrected128-slot/chunk128 build now passes56 full server vectors against
cache16, mixed/cancellation/quarantine/fresh-process/clean probes and405
zero-worker-swap samples. A same-worker decode comparison passes all five
full vectors at chunk32/64/128 (positions128/129/257 plus two decode steps),
with allocation bounds/red zones and653 zero-worker-swap samples. This is
bounded functional evidence, not final production qualification.

Clean retained-cache benchmark (same binary, off/on/on/off process order):
six A/B/C/A/B/C requests take62.576/32.996/32.972/62.515s respectively.
Across12 requests per condition retention is1.896× faster, reducing request
latency47.3% and worker SSD reads55.7%. All24 responses/usage match;750
zero-worker-swap samples. This is short-request end-to-end latency, not pure
decode throughput or long-context performance. Retention remains opt-in.
Pooling supersedes the old allocation layout; earlier cache48 swap failures
must not be transferred to it without remeasurement, nor erased from history.

The supervisor now always transmits chunk and lookahead, including32/off.
Its deliberate conservative defaults remain16/2048/32/off, while direct
worker defaults are128/2048/128/off. Always specify all four values for a
qualified launch. Unsupported flags on old binaries must fail, never silently
select a different profile. Health now also reports `expert_slots`.

The subsequent guarded-retention candidate adds `--retain-experts on|off`
(defaultoff), always emitted/verified by the supervisor. A healthy reset clears
KV/position while retaining experts; refusal quarantines without cold fallback.
CPU policy, strict parsing, health bounds and ASan/UBSan pass; retained
mixed/cancellation/fault gates pass56 exact full vectors and349 zero-worker-
swap samples. All six active-control cases pass,735 zero-worker-swap samples
and ten exact vectors. Automatic transient I/O replacement also passes: owned
child exit before one replacement, no replay, exact new request. The pre-retention pooled
artifacts require their earlier supervisor/options; do not mix versioned profiles.

Frozen retained clean binary:
`Evidence/mimo26-pooled-baseline-20260924/retention128/server-clean`, SHA256
`32b306fe6fccc20e76864a12333854994bcdc2edb17c307ed8e97dc626486ab7`.
Profile128 slots/context2048/chunk128/lookaheadon/retainon; all flags explicit.
New defaults are not promoted and no production service is installed.

Host isolation remains open: although worker VmSwap stayed zero and over32GiB
was available, the later cache128 startup tests coincided with background-process
page-outs (~279MiB during the monitored server interval). The guarded boundary
tests also saw host page-outs. Do not equate zero worker swap with zero host
reclaim, or change global VM settings to hide it.

Example candidate profile (not a deployment instruction):
`--slots 128 --context 2048 --prefill-chunk 128 --expert-lookahead on`.
For the supervisor also select an experimentally qualified shutdown budget;
the qualification runners use `--shutdown-timeout 120`, not its30s default.

See the vault's `MiMo Pooled Baseline and Retention Takeover 2026-09-24.md`.

## Historical transport candidate

The earlier frozen transport candidate is in the vault:
`Evidence/mimo26-http-transport-20260923/server/server-clean`, SHA256
`2ea091dcb112eb00568df9a89fc3300e8ddac83f22a9a7e137858ec707605a91`.
Existing `tools/mimo26_server` is a different binary and is not overwritten.
Build inputs and matched worker/layer/cache objects are in that archive.
Use the source/object hashes, not just the dirty worktree HEAD, for identity.

## Historical evidence and remaining acceptance work

- [x] Exact lookahead off/on logits, token IDs and routes on bounded corpora.
- [x] cache16 mixed JSON/SSE, busy refusal, committed-chunk disconnect,
  post-cancellation output equality and injected quarantine gates.
- [x] Persistent injected-fault supervisor budget exhaustion (earlier gate).
- [x] Automatic transient submitted-I/O failure → quarantine → owned child
  exit → one replacement → new request with complete captured logits/tokens
  matching the frozen baseline (2026-09-23). No failed-request replay.
- [x] Bounded hardened-clean-binary mixed soak:478.17 seconds, three deterministic
  cycles,16 successful responses, nine invalid/context/model refusals, three
  busy refusals and three cancellations;955 zero-swap samples, stable idle
  FD count and warmed RSS. Longer-duration soak remains open.
- [x] Restricted local HTTP framing and absolute read/per-write deadlines;
  CPU ASan/UBSan,1,000 deterministic mutations and five live refusal probes.
  Requalified hardened identity:56 full vectors exact to prior, automatic
  replacement and clean soak;508/196/955 zero-swap samples respectively.
- [ ] Requalify tool/reasoning behavior on the exact final candidate.
- [x] Active-request control fixes passed six live cases on the separate
  September23 control candidate: final/partial-prefill deadline and reuse,
  active prefill/decode SIGTERM, supervised/clean prefill SIGTERM; four
  pre-worker startup refusals passed. Archived in
  `Evidence/mimo26-controls-20260923/`. This does not qualify the new pooled
  identity; pooled controls/recovery/soak remain to be repeated.
- [x] Corrected pooled-worker full-vector and allocation-boundary gates, with
  zero observed worker swap. Host page-out/isolation concern remains open.
- [ ] Production supervisor memory guard and longer-duration guarded soak.
- [ ] Reproducible release build plus service packaging and rollback trial.
- [ ] Broader quality qualification. The new eight-prompt upstream comparison
  and613-token F32/BF16-router experiment are bounded evidence, not general
  equivalence. F32 routing remains; no cloud evaluation is required for the
  current systems gates.
- [ ] Authentication/network policy before any remote exposure.

## Operating boundaries

Before launch, confirm no unrelated `/dev/kfd` owner, sufficient host
MemAvailable (qualification gates use >70 GiB), no competing model service,
the pinned checkpoint revision/config/index, candidate hashes and free port.
Check worker VmSwap throughout startup and serving; stop qualification on any
nonzero sample. This is currently a test harness guard, **not** an implemented
production supervisor memory guard. System-wide swap can predate the test.

The opt-in supervisor accepts chunk/lookahead flags, keeps loopback binding,
has a lifetime replacement budget and never replays a request. It waits for
owned child exit and checks GPU/port availability before replacement. Busy
health is unknown, not a failure. If a child does not exit within its shutdown
timeout, replacement is refused; it does not force-kill and reuse the GPU.
An operator must inspect such a child. Budget exhaustion requires inspection,
not an unbounded systemd Restart=always policy that resets the budget.

The historical clean soak observed ~73-second cancellation latency on a380-token prompt.
Cleanup correctness does not imply acceptable interactive responsiveness.
The final request allowed32 tokens but stopped after3; longer decode coverage
remains unqualified by that test. Later active-shutdown evidence belongs to
the separately pinned control candidate, not that soak identity.

After startup verify `/health`: ready, idle, `expert_slots`, context, chunk and
lookahead exactly matching the selected profile, faults0; verify `/v1/models`
identity. One slot means
busy503 is expected; client retry is a new explicit request, not recovery of
the failed generation. Set client timeouts for this slow prefill path.

On a fault, quarantine is correct behavior. Do not clear the sticky worker
fault or force retention to reuse its slots. Capture logs, worker state,
profile and input hashes. Process recreation is the tested recovery boundary.
Injected I/O recovery does not establish recovery from hardware/driver faults.

No service unit is installed by these instructions. Release packaging should
use a dedicated immutable candidate directory, source/build manifest and
loopback endpoint. Rollback means stop and verify the GPU owner exited, then
explicitly launch the previously pinned profile; never overwrite a running
binary or launch two model owners. Context/caches must be qualified per profile.

## Hardened transport — bounded local subset

The current candidate uses nonblocking recv/send and a five-second absolute
read deadline spanning headers plus body. Per-byte arrivals do not renew it.
Each `send_all` has its own five-second budget; this is not a whole-response
or whole-SSE-stream deadline. Poll intervals are capped at100ms to observe
shutdown. Headers are bounded to16KiB; total request remains bounded to1MiB.

Content-Length requires an exact header name, valid bounded decimal digits
and no duplicates. Invalid CRLF/header names/control bytes/folding are refused.
POST requires Content-Length; Transfer-Encoding and Expect are unsupported.
The connection closes after one message. This is not full HTTP compliance,
authentication, a security audit or concurrent-client fairness qualification.
An individual slow client can still hold this single-thread server for five
seconds. Later control fixes bound backlog draining to eight clients and a
shared100ms write budget at a safe boundary.

`make test-mimo26-transport` tests actual server code, including framing,
absolute slow-header/body deadlines, stalled writes and SIGTERM during a
blocked read. Expanded ASan/UBSan coverage includes1,000 deterministic parser
mutations. Live clean-server probes reject three malformed requests and two
incomplete requests (both ~5.001 seconds), with no worker admission or faults.
The old `tests/mimo26_transport_audit.cu` is a historical pre-fix bug
reproducer, not the current regression target.

Functional, recovery and soak evidence belongs to this exact new identity;
the previous clean binary `9ea9cb96...` remains separately archived, not
silently relabelled. Source/object hashes and output-oracle provenance are
preserved in `Evidence/mimo26-http-transport-20260923/`.

## Historical blocker — active-prefill control checks (fixed, requalification required)

`tests/mimo26_prefill_control_audit.cu` reproduced that the earlier committed-
chunk callback continues with a connected peer even when shutdown is set or
the generation deadline has expired. It asserts the missing behavior and
is **not** a safety pass. Decode also lacks an explicit shutdown check.
Transport signal responsiveness therefore does not establish active-GPU
shutdown responsiveness.

The subsequent control candidate implements a non-token-counting check at safe boundaries, an explicit
prefill stop reason and no argmax/read of unproduced logits, distinct
deadline/cancellation accounting, and bounded backlog processing. Connected-
peer SIGTERM, expired deadline, post-deadline reuse and exact unaffected
outputs passed on that identity. Qualify each new profile's supervisor stop budget against measured chunk latency;
the present30-second budget may expire before one chunk finishes. A changed
candidate must receive its own qualification before release packaging.
