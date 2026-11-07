# Deployment profiles and agent clients

Moonshine 0.3.0 is a K3.1 Pro research preview, not a qualified production
deployment. Synthetic CPU/ROCm suites, full MZG2 conversion/verification, the
real 8K Beelink canary, live API/tool/structured-output and state-v2 gates, and
natural-text retrieval through 63,971 tokens pass. No context above 64K,
third-party agent client, or production soak is qualified. The settings below
are configuration examples, not recommended production operating points.

## Current identity and limits

| field | current value |
|---|---|
| model/API ID | `moonshotai/Kimi-K3.1-Pro` |
| server version | `0.3.0-k3.1-research-preview` |
| public reasoning efforts | `low`, `high`, `max` |
| default reasoning effort | `max` |
| input | text only |
| inference slots | 1 |
| default request output cap | 256, clamped to the effective server limit |
| configurable absolute output ceiling | up to 65,536, also bounded by context |
| real natural-context evidence | exact retrieval at 15,993, 31,999, and 63,971 tokens |
| sustained decode evidence | 128 tokens at 8K; 17-token decode at each natural-context gate |
| production soak | not run |

The absolute output ceiling is an implemented parser/configuration bound, not a
qualified K3.1 generation length. Public `medium` is rejected. Historical K3
8K/16K/32K/128K and 30/32-expert qualifications do not carry over.

## Server configuration examples

Set the pinned source directory or matching verified standalone bundle path:

```sh
export MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3.1-Pro
```

Loopback configuration using the source defaults:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 127.0.0.1 --port 8080
```

Explicit bounded configuration matching the 8K canary shape:

```sh
./moonshine-server "$MOONSHINE_MODEL" \
  --host 127.0.0.1 --port 8080 \
  --api-key "$MOONSHINE_API_KEY" \
  --context 8192 --experts 32 \
  --max-output-tokens 8192
```

The verified MZG2 bundle completed the real 8K loopback canary with generation
and exact prefix reuse. The example does not qualify other memory, expert,
context, or output settings. Before choosing a larger context or output cap,
measure a new memory ledger that includes the persistent **BF16** DSA index-key
cache and DSA workspace. Official FP8/MXFP4 index-cache memory estimates do not
apply.

Health and discovery are the implemented way to inspect the running binary:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $MOONSHINE_API_KEY"
```

Discovery must report `moonshotai/Kimi-K3.1-Pro`; health reports the research
preview version and effective context/output settings. A competing completion
receives HTTP 503 while the one inference slot is occupied.

## Weight-source status

Use the official pinned SafeTensors root or the verified K3.1 MZG2 bundle for
qualification. Full conversion and the loopback-only 8K Beelink canary passed;
no production deployment profile is qualified. Do not use the former
`/path/to/Kimi-K3-Moonshine-MZG2` profile or historical K3 bundle as a K3.1
deployment source.

The offline placement planner can produce a deterministic sidecar layout from
an explicit host profile, but synthetic planner success is not deployment
qualification.

## Pi Agent configuration shape

Pi Agent has not been run against K3.1 Pro. If configuring a future
qualification client, use the full public model ID and hide unsupported
reasoning levels. `moonshine` below is only the local provider key.

`~/.pi/agent/models.json` example:

```json
{
  "providers": {
    "moonshine": {
      "baseUrl": "http://MOONSHINE_HOST:8080/v1",
      "api": "openai-completions",
      "apiKey": "$MOONSHINE_API_KEY",
      "compat": {
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_completion_tokens",
        "requiresReasoningContentOnAssistantMessages": true,
        "thinkingFormat": "reasoning_effort",
        "supportsStrictMode": false
      },
      "models": [{
        "id": "moonshotai/Kimi-K3.1-Pro",
        "name": "Kimi K3.1 Pro (Moonshine research preview)",
        "reasoning": true,
        "thinkingLevelMap": {
          "off": null,
          "minimal": null,
          "low": "low",
          "medium": null,
          "high": "high",
          "xhigh": null,
          "max": "max"
        },
        "input": ["text"],
        "contextWindow": 8192,
        "maxTokens": 8192,
        "cost": {
          "input": 0,
          "output": 0,
          "cacheRead": 0,
          "cacheWrite": 0
        }
      }]
    }
  }
}
```

The 8,192 context/output values above match the explicit 8K canary-shaped server
configuration; they are not production recommendations or a maximum context
claim. Natural-text retrieval is separately verified through 63,971 tokens.
Matching settings shape:

```json
{
  "defaultProvider": "moonshine",
  "defaultModel": "moonshotai/Kimi-K3.1-Pro",
  "defaultThinkingLevel": "high",
  "showCacheMissNotices": true,
  "transport": "sse",
  "httpIdleTimeoutMs": 0,
  "retry": {
    "enabled": false,
    "provider": {
      "maxRetries": 0,
      "maxRetryDelayMs": 60000
    }
  }
}
```

Merge these keys with existing preferences rather than replacing unrelated
settings. Preserve and replay every assistant `reasoning_content` field and all
tool calls. No Pi compatibility or timing result is claimed for this cutover.

## Hermes Agent configuration shape

Hermes has not been run against K3.1 Pro. Its common `medium` reasoning default
is incompatible with the public K3.1 API; select `high` (or another supported
`low`/`max` value) explicitly. Use the full model ID rather than `moonshine`.

`~/.hermes/config.yaml` example for the same 8K canary-shaped configuration:

```yaml
model:
  default: moonshotai/Kimi-K3.1-Pro
  provider: custom
  base_url: http://MOONSHINE_HOST:8080/v1
  api_key: YOUR_MOONSHINE_API_KEY
  max_tokens: 8192
  context_length: 8192

agent:
  reasoning_effort: high
  local_stream_stale_timeout: 1800

auxiliary:
  title_generation:
    enabled: false

approvals:
  mode: manual
```

These values are a configuration starting point, not a tested Hermes profile.
Set request, stream-read, and stale-stream timeouts deliberately for an
experimental one-slot engine; no measured K3.1 timeout tier is available.
Disable retries that could open competing requests behind an active completion.

Hermes must send exact `reasoning_content` back on every historical assistant
message. Setting generation effort does not ensure reasoning replay. A client
rule keyed specifically to the old `moonshine` model ID will not match
`moonshotai/Kimi-K3.1-Pro`; update or replace that rule before qualification.
Do not use hosted-provider reasoning padding conventions in place of exact
replay.

Automatic title generation and smart approval review can issue unrelated
requests to the same endpoint. They occupy the one slot and can displace the
retained causal prefix. Keep them disabled, use manual approval, or route them
to a separate provider.

## Prefix and usage behavior

No nonstandard session header is required. Moonshine automatically reuses only
an exact append-only rendered-token prefix. Confirm a continuation through
`usage.prompt_tokens_details.cached_tokens`; zero means no prefix was reused.

Always replay the complete assistant message, including reasoning and tool
calls. If the client edits or omits history, expect a full-prefill fallback.
Real K3.1 prefill is measured through the 63,971-token natural-text gate, but
that run does not validate arbitrary workspace/cache configurations. Do not
reuse historical K3 workspace sizes or warm-cache-loan assumptions.

## Client failure map

| symptom/message | correction |
|---|---|
| model `"moonshine"` is unavailable | use `moonshotai/Kimi-K3.1-Pro` |
| `reasoning_effort must be low, high, or max` | select a supported public value; do not use `medium` |
| image/audio rejected | use text-only input |
| completion rejected with HTTP 503 | wait for the one inference slot; avoid client retry amplification |
| zero cached tokens on continuation | verify exact reasoning/tool/history replay |
| context/output request rejected | inspect discovery and lower the configured client limits |

Live server generation, tool/structured-output, and state recovery canaries pass,
but Pi Agent, Hermes Agent, other third-party clients, contexts above 64K, and
production service remain unqualified.
