#!/usr/bin/env python3
"""Qualify the one-request checkpoint-export handoff against a live server."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import stat
import time
from typing import Any


def private_regular(path: Path) -> None:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError(f"input must be a private regular file: {path}")


def write_private(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def request(
    host: str,
    port: int,
    api_key: str,
    method: str,
    path: str,
    body: bytes | None,
    timeout: float,
) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {"Authorization": f"Bearer {api_key}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    started = time.monotonic()
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        return {
            "status": response.status,
            "elapsed_seconds": time.monotonic() - started,
            "body": payload,
            "started": started,
        }
    finally:
        connection.close()


def decode_json(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def health(host: str, port: int, api_key: str) -> dict[str, Any]:
    result = request(host, port, api_key, "GET", "/health", None, 10.0)
    if result["status"] != 200:
        raise RuntimeError(f"health returned HTTP {result['status']}")
    value = decode_json(result["body"])
    value["probe_seconds"] = result["elapsed_seconds"]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send one checkpoint-producing request, queue the first immediate "
            "completion during export, and require a second contender to get 503."
        )
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--export-timeout", default=1800.0, type=float)
    parser.add_argument("--queue-timeout", default=30.0, type=float)
    parser.add_argument("--health-bound", default=0.05, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("MOONSHINE_API_KEY")
    if not api_key:
        raise SystemExit("MOONSHINE_API_KEY is required")
    private_regular(args.request)
    request_body = args.request.read_bytes()
    decode_json(request_body)

    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output, 0o700)

    before = health(args.host, args.port, api_key)
    if not before.get("ready") or before.get("busy"):
        raise SystemExit("server must be ready and idle")
    if before.get("checkpoint_queue_capacity") != 1:
        raise SystemExit("server does not advertise checkpoint queue capacity 1")

    export = request(
        args.host,
        args.port,
        api_key,
        "POST",
        "/v1/chat/completions",
        request_body,
        args.export_timeout,
    )
    response_completed = time.monotonic()
    write_private(args.output / "export-response.json", export["body"])

    probe_body = json.dumps(
        {
            "model": "invalid-checkpoint-handoff-probe",
            "messages": [{"role": "user", "content": "probe"}],
            "max_tokens": 1,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()

    health_samples: list[dict[str, Any]] = []
    queued_seen = False
    second: dict[str, Any] | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(
            request,
            args.host,
            args.port,
            api_key,
            "POST",
            "/v1/chat/completions",
            probe_body,
            args.queue_timeout,
        )
        deadline = time.monotonic() + args.queue_timeout
        while time.monotonic() < deadline and not first_future.done():
            sample = health(args.host, args.port, api_key)
            health_samples.append(sample)
            if (
                sample.get("slot_phase") == "checkpoint_export"
                and sample.get("queued_completions") == 1
            ):
                queued_seen = True
                second = request(
                    args.host,
                    args.port,
                    api_key,
                    "POST",
                    "/v1/chat/completions",
                    probe_body,
                    10.0,
                )
                break
            time.sleep(0.005)
        first = first_future.result(timeout=args.queue_timeout)

    final_deadline = time.monotonic() + args.queue_timeout
    after: dict[str, Any] | None = None
    while time.monotonic() < final_deadline:
        after = health(args.host, args.port, api_key)
        health_samples.append(after)
        if after.get("ready") and not after.get("busy"):
            break
        time.sleep(0.01)

    first_json = decode_json(first["body"])
    second_json = decode_json(second["body"]) if second is not None else None
    max_health = max(
        (float(sample["probe_seconds"]) for sample in health_samples),
        default=0.0,
    )
    decision = (
        export["status"] == 200
        and queued_seen
        and first["status"] == 400
        and second is not None
        and second["status"] == 503
        and isinstance(second_json, dict)
        and second_json.get("error", {}).get("code") == "server_busy"
        and after is not None
        and after.get("ready") is True
        and after.get("busy") is False
        and max_health <= args.health_bound
    )
    result = {
        "schema": "moonshine-checkpoint-handoff-qualification-v1",
        "decision": "PASS" if decision else "FAIL",
        "export_request": {
            "http_status": export["status"],
            "client_seconds": export["elapsed_seconds"],
        },
        "first_immediate_completion": {
            "http_status": first["status"],
            "wait_seconds": first["elapsed_seconds"],
            "started_after_response_seconds": max(
                0.0, float(first["started"]) - response_completed
            ),
            "error_code": first_json.get("error", {}).get("code"),
        },
        "second_contender": {
            "sent": second is not None,
            "http_status": second["status"] if second is not None else None,
            "seconds": second["elapsed_seconds"] if second is not None else None,
            "error_code": (
                second_json.get("error", {}).get("code")
                if isinstance(second_json, dict)
                else None
            ),
        },
        "control_plane": {
            "queued_phase_observed": queued_seen,
            "health_samples": len(health_samples),
            "max_health_seconds": max_health,
            "bound_seconds": args.health_bound,
        },
        "final_health": {
            "ready": after.get("ready") if after else None,
            "busy": after.get("busy") if after else None,
            "slot_phase": after.get("slot_phase") if after else None,
            "queued_completions": (
                after.get("queued_completions") if after else None
            ),
        },
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    write_private(args.output / "result.json", encoded)
    print(f"checkpoint handoff qualification: {result['decision']}")
    print(args.output / "result.json")
    return 0 if decision else 1


if __name__ == "__main__":
    raise SystemExit(main())
