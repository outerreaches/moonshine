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
| maximum configured capacity | 131,072 | 30 | 32,768 | short requests plus two independent prefills in one persistent server |

The 128K profile uses 30 expert slots deliberately. With 32 slots, the first
cold request can borrow empty expert-cache storage as prefill workspace, but a
later request must retain the warmed cache and allocate separate workspace.
That second allocation can violate the CMA-plus-4-GiB safety reserve. The
30-slot cache is 45.104 GiB and passed two independent 128K requests.

Filled 64K/128K execution, filled-128K quality, and a continuous 32K
completion remain outside the qualification boundary.

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

Persistent 128K-capacity profile:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 131072 --experts 30 \
  --max-output-tokens 32768
```

## Hermes Agent compatibility

Hermes Agent's generic custom-provider profile defaults to 65,536 output
tokens. Its standard reasoning setting is `medium`. Both values are outside
Moonshine 0.2's API contract: the server maximum is 32,768, and K3 accepts
`reasoning_effort` values `low`, `high`, or `max`.

Set both limits explicitly in `~/.hermes/config.yaml`. This example matches
the persistent 128K Moonshine profile:

```yaml
model:
  default: moonshine
  provider: custom
  base_url: http://MOONSHINE_HOST:8080/v1
  api_key: YOUR_MOONSHINE_API_KEY
  max_tokens: 32768
  context_length: 131072

agent:
  reasoning_effort: high
```

`low` and `max` are also valid. Do not leave Hermes on `medium`: Moonshine
rejects it with HTTP 400. Restart Hermes after changing the configuration.
For a 32K Moonshine server, set Hermes `context_length: 32768`; keep
`max_tokens: 32768`.

Moonshine emits SSE progress comments during prefill, but this engine is much
slower than a cloud endpoint. For ordinary short-prompt agent use, make the
Hermes timeouts explicit:

```sh
HERMES_API_TIMEOUT=1800
HERMES_STREAM_READ_TIMEOUT=1800
```

These are seconds. Hermes normally raises the stream read timeout for LAN and
loopback endpoints, but explicit values remove endpoint-detection ambiguity.
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
| `max tokens must be an integer in [1,32768]` | set Hermes `model.max_tokens: 32768` or lower |
| `reasoning_effort must be low, high, or max` | set Hermes `agent.reasoning_effort` to `low`, `high`, or `max` |
| prompt/context capacity error | reduce history/output cap or select a larger qualified context profile |

Always confirm the running server's effective contract before client tests:

```sh
curl http://MOONSHINE_HOST:8080/health
curl http://MOONSHINE_HOST:8080/v1/models
```

Both responses advertise `context_length` and effective
`max_output_tokens`.
