#!/usr/bin/env python3
"""Qualify the checkpoint-export handoff for SSE terminal responses."""

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


def request(host: str, port: int, key: str, body: bytes, timeout: float) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    started = time.monotonic()
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        payload = response.read()
        return {
            "status": response.status,
            "body": payload,
            "elapsed_seconds": time.monotonic() - started,
            "started": started,
        }
    finally:
        connection.close()



def health(host: str, port: int, key: str) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=10.0)
    started = time.monotonic()
    try:
        connection.request(
            "GET",
            "/health",
            headers={"Authorization": f"Bearer {key}"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        payload["probe_seconds"] = time.monotonic() - started
        payload["http_status"] = response.status
        return payload
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualify one queued completion during SSE checkpoint export."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--timeout", default=1800.0, type=float)
    parser.add_argument("--health-bound", default=0.05, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key = os.environ.get("MOONSHINE_API_KEY")
    if not key:
        raise SystemExit("MOONSHINE_API_KEY is required")
    private_regular(args.request)
    body = args.request.read_bytes()
    request_object = json.loads(body)
    if request_object.get("stream") is not True:
        raise SystemExit("request must set stream=true")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output, 0o700)

    probe_body = json.dumps(
        {
            "model": "invalid-stream-handoff-probe",
            "messages": [{"role": "user", "content": "probe"}],
            "max_tokens": 1,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()
    health_samples: list[dict[str, Any]] = []
    queued_seen = False
    second: dict[str, Any] | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        stream_future = executor.submit(
            request, args.host, args.port, key, body, args.timeout
        )
        deadline = time.monotonic() + args.timeout
        first_future: concurrent.futures.Future[dict[str, Any]] | None = None
        while time.monotonic() < deadline and not stream_future.done():
            sample = health(args.host, args.port, key)
            health_samples.append(sample)
            if sample.get("slot_phase") == "checkpoint_export":
                first_future = executor.submit(
                    request, args.host, args.port, key, probe_body, 60.0
                )
                queue_deadline = min(deadline, time.monotonic() + 10.0)
                while time.monotonic() < queue_deadline:
                    queue_sample = health(args.host, args.port, key)
                    health_samples.append(queue_sample)
                    if (
                        queue_sample.get("slot_phase") == "checkpoint_export"
                        and queue_sample.get("queued_completions") == 1
                    ):
                        queued_seen = True
                        second = request(
                            args.host, args.port, key, probe_body, 10.0
                        )
                        break
                    time.sleep(0.005)
                break
            time.sleep(0.005)
        if first_future is None:
            raise SystemExit("checkpoint_export phase was not observed")
        first = first_future.result(timeout=120.0)
        stream = stream_future.result(timeout=args.timeout)

    final = health(args.host, args.port, key)
    stream_body = stream["body"]
    first_body = json.loads(first["body"])
    second_body = json.loads(second["body"]) if second is not None else {}
    max_health = max(
        (float(sample.get("probe_seconds", "0")) for sample in health_samples),
        default=0.0,
    )
    stream_terminal = b"[DONE]" in stream_body
    decision = (
        stream["status"] == 200
        and stream_terminal
        and queued_seen
        and first["status"] == 400
        and second is not None
        and second["status"] == 503
        and second_body.get("error", {}).get("code") == "server_busy"
        and final.get("ready") is True
        and final.get("busy") is False
        and final.get("slot_phase") == "idle"
        and max_health <= args.health_bound
    )
    result = {
        "schema": "moonshine-checkpoint-stream-handoff-qualification-v1",
        "decision": "PASS" if decision else "FAIL",
        "stream": {
            "http_status": stream["status"],
            "client_seconds": stream["elapsed_seconds"],
            "terminal_done_seen": stream_terminal,
        },
        "first_queued_completion": {
            "http_status": first["status"],
            "wait_seconds": first["elapsed_seconds"],
            "error_code": first_body.get("error", {}).get("code"),
        },
        "second_contender": {
            "http_status": second["status"] if second is not None else None,
            "error_code": second_body.get("error", {}).get("code"),
        },
        "control_plane": {
            "queued_phase_observed": queued_seen,
            "health_samples": len(health_samples),
            "max_health_seconds": max_health,
            "bound_seconds": args.health_bound,
        },
        "final_health": {
            "ready": final.get("ready"),
            "busy": final.get("busy"),
            "slot_phase": final.get("slot_phase"),
            "queued_completions": final.get("queued_completions"),
        },
    }
    write_private(args.output / "stream-response.bin", stream_body)
    write_private(
        args.output / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(f"checkpoint stream handoff qualification: {result['decision']}")
    print(args.output / "result.json")
    return 0 if decision else 1


if __name__ == "__main__":
    raise SystemExit(main())
