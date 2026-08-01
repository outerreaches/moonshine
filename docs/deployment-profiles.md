# Deployment profiles and Hermes Agent

These profiles describe configurations exercised on the qualified 128 GB
Ryzen AI Max+ 395 / Radeon 8060S host. They are operating points, not claims
that every possible prompt or full-length completion has been run.

## Qualified profiles

| use | context | experts/layer | output ceiling | qualification boundary |
|---|---:|---:|---:|---|
| agentic/API baseline | 8,192 | 32 | 8,192 | official OpenAI SDK two-turn tool loop, reasoning, SSE, and exact prefix reuse |
| bounded long context | 16,384 | 32 | 16,384 | filled-context execution and 15,993-token natural-text retrieval |
| long context | 32,768 | 32 | 32,768 | filled-context execution and 31,999-token natural-text retrieval |
| maximum configured capacity | 131,072 | 30 | 65,536 | 64K admission, remaining-context clamp, and two independent short requests in one persistent server |

The 128K profile uses 30 expert slots deliberately. With 32 slots, the first
cold request can borrow empty expert-cache storage as prefill workspace, but a
later request must retain the warmed cache and allocate separate workspace.
That second allocation can violate the CMA-plus-4-GiB safety reserve. The
30-slot cache is 45.104 GiB and passed two independent 128K requests.

Filled 64K/128K execution, filled-128K quality, and continuous 32K or 64K
completions remain outside the qualification boundary.

## 0.2.0 output expansion

Moonshine 0.2.0 raises the absolute output ceiling to 65,536 and accepts K3's
native `reasoning_effort: medium`. Portable parser, tokenizer, and official-SDK
replay gates passed. The 128K/30-expert configuration also passed live
discovery, 65,536 admission, 65,537 rejection, remaining-context clamping, a
naturally stopped `medium` response, and two independent requests in one
process. The new ceiling is a maximum request budget, not evidence of a
continuous 64K decode or filled-128K prompt. Exact evidence is in
[64K output and medium-reasoning qualification](qualification-output-64k.md).

## Server commands

Use an API key whenever the service binds beyond loopback.

Agentic 8K baseline:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 8192 --experts 32 \
  --max-output-tokens 8192
```

Qualified 32K profile:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 32768 --experts 32 \
  --max-output-tokens 32768
```

Persistent 128K-capacity 0.2.0 profile:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 131072 --experts 30 \
  --max-output-tokens 65536
```

## Hermes Agent compatibility

Hermes Agent's generic custom-provider profile defaults to 65,536 output
tokens and its standard reasoning setting is `medium`. Both match the
128K/64K Moonshine 0.2.0 profile. A server built from the earlier
32K checkpoint still rejects both values, so confirm the running contract
through `/health` or `/v1/models` rather than assuming the checkout and
service binary match.

Set both limits explicitly in `~/.hermes/config.yaml`. This example matches
the persistent 128K Moonshine profile:

```yaml
model:
  default: moonshine
  provider: custom
  base_url: http://MOONSHINE_HOST:8080/v1
  api_key: YOUR_MOONSHINE_API_KEY
  max_tokens: 65536
  context_length: 131072

agent:
  reasoning_effort: medium
```

`low`, `high`, and `max` are also valid. Restart Hermes after changing the
configuration. For a 32K Moonshine server, set `context_length: 32768` and
`max_tokens: 32768`; a request cap cannot exceed the server's effective
context/output ceiling.

Moonshine emits SSE progress comments during prefill, but this engine is much
slower than a cloud endpoint. For ordinary short-prompt agent use, make the
Hermes timeouts explicit:

```sh
HERMES_API_TIMEOUT=1800
HERMES_STREAM_READ_TIMEOUT=1800
```

These are seconds. Hermes normally raises the stream read timeout for LAN and
loopback endpoints, but explicit values remove endpoint-detection ambiguity.
On Sparky, a Hermes turn with its full system/tool prompt exhausted an
effective 15-minute path before Moonshine produced output. Use at least 1,800
seconds for both request and stream-read timeouts there; the large framework
prompt makes this necessary even when the user's visible prompt is short.
For deliberately long 16K/32K prompts, start with 7,200 seconds for both
values. The qualified filled-32K prefill took about 73 minutes before decode,
so the ordinary 30-minute tier is not sufficient for that workload.

The timeout only prevents premature client cancellation. It does not change
Moonshine's single-request slot, model speed, context accounting, or output
ceiling. Prefer streaming for Hermes so progress keepalives can traverse the
connection; ordinary JSON responses cannot emit keepalives.

## Client-side failure map

| HTTP 400 message | likely correction |
|---|---|
| `max tokens must be an integer in [1,N]` | set Hermes `model.max_tokens` to the advertised server ceiling `N` or lower |
| `reasoning_effort must be low, high, or max` | the running binary predates `medium`; use `low`, `high`, or `max`, or rebuild after stopping it |
| prompt/context capacity error | reduce history/output cap or select a larger qualified context profile |

Always confirm the running server's effective contract before client tests:

```sh
curl http://MOONSHINE_HOST:8080/health
curl http://MOONSHINE_HOST:8080/v1/models
```

Both responses advertise `context_length` and effective
`max_output_tokens`.
