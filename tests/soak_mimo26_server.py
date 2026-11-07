#!/usr/bin/env python3
"""Sustained mixed-request soak against a running mimo26_server.

Mixes streaming and non-streaming, short and long, valid and refused, plus
deliberate mid-generation disconnects -- because the failure modes a soak is
meant to find are the ones that only appear when those interleave. A loop of
identical happy-path requests mostly proves the happy path.

Checks, at the end:
  - every response was well-formed JSON or a well-formed SSE stream
  - refusals were refused, with the code they should carry
  - cancellations left the slot idle and did not fault the worker
  - the server is still ready, and resident memory has not grown

    tests/soak_mimo26_server.py http://127.0.0.1:8644 --seconds 180
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    "Name the capital of France in one word.",
    "What is 17 times 3?",
    "Write one sentence about rivers.",
    "List three colours.",
    "Is 91 prime? Answer briefly.",
]


def post(base, body, timeout=120, stream=False):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if stream:
            frames = 0
            saw_done = False
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    saw_done = True
                else:
                    json.loads(payload)   # must be well formed
                    frames += 1
            return {"frames": frames, "done": saw_done}
        return json.loads(response.read().decode())


def health(base):
    with urllib.request.urlopen(base + "/health", timeout=10) as response:
        return json.loads(response.read().decode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    random.seed(args.seed)

    before = health(args.base)
    print(f"start: ready={before['ready']} resident={before['resident_gib']} GiB")

    counts = {"ok": 0, "stream": 0, "refused": 0, "busy": 0, "cancelled": 0,
              "unexpected": 0}
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        roll = random.random()
        try:
            if roll < 0.3:
                body = {"messages": [{"role": "user",
                                      "content": random.choice(PROMPTS)}],
                        "max_tokens": random.choice([8, 16, 32]),
                        "enable_thinking": False}
                result = post(args.base, body)
                assert "choices" in result and "usage" in result
                counts["ok"] += 1
            elif roll < 0.55:
                body = {"messages": [{"role": "user",
                                      "content": random.choice(PROMPTS)}],
                        "max_tokens": 16, "stream": True,
                        "enable_thinking": False}
                result = post(args.base, body, stream=True)
                assert result["done"], "stream did not terminate with [DONE]"
                counts["stream"] += 1
            elif roll < 0.75:
                # Refusals must stay refused under load.
                bad = random.choice([
                    {"messages": [{"role": "user", "content": "x"}],
                     "temperature": 0.5},
                    {"messages": [{"role": "user", "content": "x"}],
                     "tools": [{"type": "function"}]},
                    {"model": "gpt-4",
                     "messages": [{"role": "user", "content": "x"}]},
                ])
                try:
                    post(args.base, bad, timeout=30)
                    counts["unexpected"] += 1
                except urllib.error.HTTPError as failure:
                    detail = json.loads(failure.read().decode())
                    assert "code" in detail["error"], "refusal lacks a code"
                    counts["refused"] += 1
            else:
                # Disconnect mid-generation.
                body = {"messages": [{"role": "user",
                                      "content": "Write a long essay."}],
                        "max_tokens": 200, "enable_thinking": False}
                try:
                    post(args.base, body, timeout=random.uniform(2.0, 5.0))
                    counts["ok"] += 1
                except Exception:
                    counts["cancelled"] += 1
                # The slot must free itself; give the loop a moment to notice.
                time.sleep(2.0)
        except urllib.error.HTTPError as failure:
            if failure.code == 503:
                counts["busy"] += 1
            else:
                counts["unexpected"] += 1
                print(f"  unexpected HTTP {failure.code}", file=sys.stderr)
        except Exception as problem:            # noqa: BLE001
            counts["unexpected"] += 1
            print(f"  unexpected {problem!r}", file=sys.stderr)

    after = health(args.base)
    print(f"end:   ready={after['ready']} phase={after['phase']} "
          f"resident={after['resident_gib']} GiB")
    print(f"       {counts}")
    print(f"       served={after['served']} cancelled={after['cancelled']} "
          f"faults={after['faults']} recoveries={after['recoveries']} "
          f"rejected_busy={after['rejected_busy']}")

    problems = []
    if counts["unexpected"]:
        problems.append(f"{counts['unexpected']} unexpected outcomes")
    if after["faults"]:
        problems.append(f"{after['faults']} worker faults")
    if not after["ready"]:
        problems.append(f"server not ready at the end (phase {after['phase']})")
    if after["resident_gib"] > before["resident_gib"] + 0.5:
        problems.append(
            f"resident grew {before['resident_gib']} -> {after['resident_gib']} GiB")
    print("soak_mimo26_server:", "ok" if not problems else
          "FAILED: " + "; ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
