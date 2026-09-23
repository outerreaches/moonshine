# MiMo local production candidate — qualification, not deployment

Current conservative profile: raw official packed weights, cache16,
context1024, chunk64, explicit lookahead on; loopback only. No MZG2 serving
integration. Cache48 and larger are not qualified as reliably swap-free.
Do not substitute historical logical-residency figures for host memory tests.

The frozen clean candidate is in the vault:
`Evidence/mimo26-http-transport-20260923/server/server-clean`, SHA256
`2ea091dcb112eb00568df9a89fc3300e8ddac83f22a9a7e137858ec707605a91`.
Existing `tools/mimo26_server` is a different binary and is not overwritten.
Build inputs and matched worker/layer/cache objects are in that archive.
Use the source/object hashes, not just the dirty worktree HEAD, for identity.

## Acceptance checklist

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
- [ ] Fix reproduced active-prefill shutdown/deadline omissions, then qualify
  active-request shutdown/deadline and startup failure.
- [ ] Production supervisor memory guard and longer-duration guarded soak.
- [ ] Reproducible release build plus service packaging and rollback trial.
- [ ] Upstream oracle/router-precision quality qualification, still deferred.
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

The clean soak observed ~73-second cancellation latency on a380-token prompt.
Cleanup correctness does not imply acceptable interactive responsiveness.
The final request allowed32 tokens but stopped after3; longer decode coverage
remains unqualified by that test. Idle SIGTERM exit passed, not active shutdown.

After startup verify `/health`: ready, idle, context1024, prefill_chunk64,
expert_lookahead=true, faults0; verify `/v1/models` identity. One slot means
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
seconds; unbounded backlog draining remains a scheduling gap.

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

## Next blocker — active-prefill control checks

`tests/mimo26_prefill_control_audit.cu` reproduces that the current committed-
chunk callback continues with a connected peer even when shutdown is set or
the generation deadline has expired. It asserts the missing behavior and
is **not** a safety pass. Decode also lacks an explicit shutdown check.
Transport signal responsiveness therefore does not establish active-GPU
shutdown responsiveness.

Implement a non-token-counting control check at safe boundaries, an explicit
prefill stop reason and no argmax/read of unproduced logits. Preserve distinct
deadline/cancellation accounting. Bound backlog processing. Test connected-
peer SIGTERM, expired deadline, post-deadline reuse and exact unaffected
outputs. Qualify a supervisor stop budget against measured chunk latency;
the present30-second budget may expire before one chunk finishes. A changed
candidate must receive its own qualification before release packaging.
