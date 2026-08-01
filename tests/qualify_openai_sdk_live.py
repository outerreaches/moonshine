#!/usr/bin/env python3
"""Run one real Moonshine tool-call stream through the official Python SDK."""

from __future__ import annotations

import argparse
import json

import openai
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:18084/v1"
    )
    args = parser.parse_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key="moonshine-local-qualification",
        timeout=1200.0,
    )
    stream = client.chat.completions.create(
        model="moonshine",
        messages=[{
            "role": "user",
            "content": "Use the weather tool to check Toronto.",
        }],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }],
        tool_choice="required",
        reasoning_effort="low",
        max_completion_tokens=192,
        stream=True,
        extra_headers={
            "X-Moonshine-Session": "openai-sdk-live-qualification"
        },
    )

    reasoning: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    usage: tuple[int, int, int] | None = None
    for chunk in stream:
        choice = chunk.choices[0]
        piece = getattr(choice.delta, "reasoning_content", None)
        if piece:
            reasoning.append(piece)
        for call in choice.delta.tool_calls or []:
            accumulated = calls.setdefault(
                call.index, {"id": "", "name": "", "arguments": ""}
            )
            if call.id:
                accumulated["id"] = call.id
            if call.function is not None:
                if call.function.name:
                    accumulated["name"] += call.function.name
                if call.function.arguments:
                    accumulated["arguments"] += call.function.arguments
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
        if chunk.usage is not None:
            usage = (
                chunk.usage.prompt_tokens,
                chunk.usage.completion_tokens,
                chunk.usage.total_tokens,
            )

    if finish_reason != "tool_calls":
        raise AssertionError(f"unexpected finish reason {finish_reason!r}")
    if list(calls) != [0]:
        raise AssertionError(f"unexpected tool-call indexes {list(calls)!r}")
    call = calls[0]
    if call["name"] != "get_weather":
        raise AssertionError(f"unexpected tool name {call['name']!r}")
    if json.loads(call["arguments"]) != {"city": "Toronto"}:
        raise AssertionError(f"unexpected arguments {call['arguments']!r}")
    if not call["id"].startswith("call_chatcmpl-moonshine-"):
        raise AssertionError(f"unexpected tool-call ID {call['id']!r}")
    if not "".join(reasoning):
        raise AssertionError("the SDK did not expose reasoning_content")
    if usage is None or usage[2] != usage[0] + usage[1]:
        raise AssertionError(f"invalid terminal usage {usage!r}")
    print(
        "OpenAI Python SDK live Moonshine tool stream: PASS "
        f"(openai {openai.__version__}, usage={usage[0]}/{usage[1]}/{usage[2]})"
    )


if __name__ == "__main__":
    main()
