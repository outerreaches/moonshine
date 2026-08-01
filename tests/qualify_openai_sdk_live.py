#!/usr/bin/env python3
"""Run a real Moonshine tool loop through the official Python SDK."""

from __future__ import annotations

import argparse
import json

import openai
from openai import OpenAI


def consume_stream(stream):
    reasoning: list[str] = []
    content: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    usage: tuple[int, int, int, int] | None = None
    for chunk in stream:
        choice = chunk.choices[0]
        piece = getattr(choice.delta, "reasoning_content", None)
        if piece:
            reasoning.append(piece)
        if choice.delta.content:
            content.append(choice.delta.content)
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
            details = chunk.usage.prompt_tokens_details
            usage = (
                chunk.usage.prompt_tokens,
                chunk.usage.completion_tokens,
                chunk.usage.total_tokens,
                0 if details is None or details.cached_tokens is None
                else details.cached_tokens,
            )
    if usage is None or usage[2] != usage[0] + usage[1]:
        raise AssertionError(f"invalid terminal usage {usage!r}")
    return "".join(reasoning), "".join(content), calls, finish_reason, usage


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
    messages = [{
        "role": "user",
        "content": "Use the weather tool to check Toronto.",
    }]
    tools = [{
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
    }]
    first_stream = client.chat.completions.create(
        model="moonshine",
        messages=messages,
        tools=tools,
        tool_choice="required",
        reasoning_effort="low",
        max_completion_tokens=192,
        stream=True,
    )
    reasoning, content, calls, finish_reason, first_usage = consume_stream(
        first_stream
    )
    if finish_reason != "tool_calls" or content:
        raise AssertionError(
            f"unexpected first finish/content {finish_reason!r}/{content!r}"
        )
    if list(calls) != [0]:
        raise AssertionError(f"unexpected tool-call indexes {list(calls)!r}")
    call = calls[0]
    if call["name"] != "get_weather":
        raise AssertionError(f"unexpected tool name {call['name']!r}")
    if json.loads(call["arguments"]) != {"city": "Toronto"}:
        raise AssertionError(f"unexpected arguments {call['arguments']!r}")
    if not call["id"].startswith("call_chatcmpl-moonshine-"):
        raise AssertionError(f"unexpected tool-call ID {call['id']!r}")
    if not reasoning:
        raise AssertionError("the SDK did not expose reasoning_content")

    messages.extend([
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": None,
            "tool_calls": [{
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": '{"weather":"sunny","temperature_c":22}',
        },
    ])
    second_stream = client.chat.completions.create(
        model="moonshine",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        reasoning_effort="low",
        max_completion_tokens=192,
        stream=True,
    )
    second_reasoning, answer, second_calls, second_finish, second_usage = (
        consume_stream(second_stream)
    )
    if second_finish != "stop" or second_calls:
        raise AssertionError(
            f"unexpected result turn {second_finish!r}/{second_calls!r}"
        )
    if not second_reasoning:
        raise AssertionError("result turn omitted reasoning_content")
    if first_usage[3] != 0 or second_usage[3] == 0:
        raise AssertionError(
            "automatic exact-prefix reuse was not reported through "
            f"usage.prompt_tokens_details.cached_tokens: "
            f"{first_usage[3]} -> {second_usage[3]}"
        )
    answer_lower = answer.lower()
    if not all(piece in answer_lower for piece in ("toronto", "sunny", "22")):
        raise AssertionError(f"unexpected result answer {answer!r}")
    print(
        "OpenAI Python SDK live Moonshine tool loop: PASS "
        f"(openai {openai.__version__}, "
        f"usage={first_usage[0]}/{first_usage[1]}/{first_usage[2]} -> "
        f"{second_usage[0]}/{second_usage[1]}/{second_usage[2]}, "
        f"cached={first_usage[3]}->{second_usage[3]})"
    )


if __name__ == "__main__":
    main()
