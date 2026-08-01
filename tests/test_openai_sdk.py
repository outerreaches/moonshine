#!/usr/bin/env python3
"""Exercise Moonshine's Chat Completions wire with the official Python SDK."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
from openai import OpenAI


class MoonshineFixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    request_json: dict[str, object] | None = None
    request_error: str | None = None

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            if self.path != "/v1/chat/completions":
                raise AssertionError(f"unexpected path {self.path!r}")
            size = int(self.headers.get("Content-Length", "0"))
            type(self).request_json = json.loads(self.rfile.read(size))
            if self.headers.get("Authorization") != "Bearer local-test":
                raise AssertionError("SDK bearer authorization is missing")
            if self.headers.get("X-Moonshine-Session") != "sdk-fixture":
                raise AssertionError("Moonshine session header is missing")
        except Exception as exc:  # report the handler-thread failure in main
            type(self).request_error = str(exc)
            self.send_error(400)
            return

        chunks = [
            {
                "id": "chatcmpl-moonshine-sdk-1",
                "object": "chat.completion.chunk",
                "created": 1785567600,
                "model": "moonshine",
                "system_fingerprint": "moonshine-q8-greedy",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "",
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-moonshine-sdk-1",
                "object": "chat.completion.chunk",
                "created": 1785567600,
                "model": "moonshine",
                "system_fingerprint": "moonshine-q8-greedy",
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning_content": "Need weather."},
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-moonshine-sdk-1",
                "object": "chat.completion.chunk",
                "created": 1785567600,
                "model": "moonshine",
                "system_fingerprint": "moonshine-q8-greedy",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_chatcmpl-moonshine-sdk-1_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Toronto"}',
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-moonshine-sdk-1",
                "object": "chat.completion.chunk",
                "created": 1785567600,
                "model": "moonshine",
                "system_fingerprint": "moonshine-q8-greedy",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": 151,
                    "completion_tokens": 45,
                    "total_tokens": 196,
                },
            },
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b": moonshine prefill layer 93/93\r\n\r\n")
        for chunk in chunks:
            payload = json.dumps(chunk, separators=(",", ":"))
            self.wfile.write(f"data: {payload}\r\n\r\n".encode())
        self.wfile.write(b"data: [DONE]\r\n\r\n")
        self.wfile.flush()
        self.close_connection = True


def main() -> None:
    MoonshineFixtureHandler.request_json = None
    MoonshineFixtureHandler.request_error = None
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), MoonshineFixtureHandler
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAI(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-test",
            timeout=10.0,
        )
        stream = client.chat.completions.create(
            model="moonshine",
            messages=[{"role": "user", "content": "Weather in Toronto?"}],
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
            parallel_tool_calls=False,
            reasoning_effort="low",
            max_completion_tokens=64,
            stream=True,
            extra_headers={"X-Moonshine-Session": "sdk-fixture"},
        )
        reasoning: list[str] = []
        calls: list[object] = []
        finish_reason: str | None = None
        usage_total: int | None = None
        for chunk in stream:
            choice = chunk.choices[0]
            piece = getattr(choice.delta, "reasoning_content", None)
            if piece:
                reasoning.append(piece)
            if choice.delta.tool_calls:
                calls.extend(choice.delta.tool_calls)
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
            if chunk.usage is not None:
                usage_total = chunk.usage.total_tokens

        if MoonshineFixtureHandler.request_error is not None:
            raise AssertionError(MoonshineFixtureHandler.request_error)
        request = MoonshineFixtureHandler.request_json
        if request is None:
            raise AssertionError("the SDK did not submit a request")
        if request.get("stream") is not True:
            raise AssertionError("the SDK request did not enable streaming")
        if request.get("parallel_tool_calls") is not False:
            raise AssertionError("the SDK lost parallel_tool_calls=false")
        if "".join(reasoning) != "Need weather.":
            raise AssertionError("reasoning_content was not preserved")
        if len(calls) != 1:
            raise AssertionError(f"expected one tool call, got {len(calls)}")
        call = calls[0]
        if (
            call.index != 0
            or call.id != "call_chatcmpl-moonshine-sdk-1_1"
            or call.function.name != "get_weather"
            or call.function.arguments != '{"city":"Toronto"}'
        ):
            raise AssertionError(f"unexpected SDK tool call {call!r}")
        if finish_reason != "tool_calls" or usage_total != 196:
            raise AssertionError("terminal chunk or usage was not preserved")
        print(
            "OpenAI Python SDK Moonshine SSE fixture: PASS "
            f"(openai {openai.__version__})"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
