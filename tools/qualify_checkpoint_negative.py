#!/usr/bin/env python3
"""Exercise checkpoint publication failure, queued disconnect, or shutdown."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import stat
import time
from typing import Any


def private_regular(path: Path) -> None:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError(f"input must be private regular file: {path}")


def write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def post(host: str, port: int, key: str, body: bytes, timeout: float) -> dict[str, Any]:
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
        value = json.loads(response.read())
        value["probe_seconds"] = time.monotonic() - started
        value["http_status"] = response.status
        return value
    finally:
        connection.close()


def open_raw(host: str, port: int, key: str, body: bytes) -> socket.socket:
    connection = socket.create_connection((host, port), timeout=10.0)
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        + f"Host: {host}:{port}\r\n".encode()
        + f"Authorization: Bearer {key}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"Connection: close\r\n\r\n"
        + body
    )
    connection.sendall(request)
    return connection


def raw_status(connection: socket.socket, timeout: float = 30.0) -> int | None:
    connection.settimeout(timeout)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = connection.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    except (TimeoutError, socket.timeout):
        return None
    finally:
        connection.close()
    payload = b"".join(chunks)
    if not payload.startswith(b"HTTP/"):
        return None
    try:
        return int(payload.split(b" ", 2)[1])
    except (IndexError, ValueError):
        return None


def wait_phase(
    host: str,
    port: int,
    key: str,
    phase: str,
    queued: int | None,
    timeout: float,
) -> tuple[bool, list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    samples: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        sample = health(host, port, key)
        samples.append(sample)
        if sample.get("slot_phase") == phase and (
            queued is None or sample.get("queued_completions") == queued
        ):
            return True, samples
        time.sleep(0.005)
    return False, samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("disconnect", "publication-failure", "shutdown"), required=True)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--timeout", default=1800.0, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key = os.environ.get("MOONSHINE_API_KEY")
    if not key:
        raise SystemExit("MOONSHINE_API_KEY is required")
    if args.mode == "shutdown" and args.pid is None:
        raise SystemExit("--pid is required for shutdown mode")
    private_regular(args.request)
    body = args.request.read_bytes()
    json.loads(body)

    result: dict[str, Any] = {
        "schema": "moonshine-checkpoint-negative-qualification-v1",
        "mode": args.mode,
    }
    first: dict[str, Any] | None = None
    queued_socket: socket.socket | None = None
    samples: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        first_future = executor.submit(
            post, args.host, args.port, key, body, args.timeout
        )
        observed, phase_samples = wait_phase(
            args.host, args.port, key, "checkpoint_export", None, args.timeout
        )
        samples.extend(phase_samples)
        result["export_phase_observed"] = observed
        if not observed:
            raise SystemExit("checkpoint_export phase was not observed")

        if args.mode == "publication-failure":
            os.chmod(args.checkpoint_root, 0o500)
            result["root_mode_during_export"] = 0o500
        else:
            queued_socket = open_raw(args.host, args.port, key, body)
            queued, queue_samples = wait_phase(
                args.host, args.port, key, "checkpoint_export", 1, 10.0
            )
            samples.extend(queue_samples)
            result["queued_observed"] = queued
            if not queued:
                queued_socket.close()
                raise SystemExit("queued completion was not observed")
            if args.mode == "shutdown":
                os.kill(args.pid, signal.SIGTERM)
                result["sigterm_sent"] = True
            else:
                queued_socket.close()
                queued_socket = None
                result["queued_client_closed"] = True

        first = first_future.result(timeout=args.timeout)
        result["first_status"] = first["status"]
        result["first_seconds"] = first["elapsed_seconds"]
        if queued_socket is not None:
            result["queued_status"] = raw_status(queued_socket, 60.0)
    if args.mode != "shutdown":
        idle, idle_samples = wait_phase(
            args.host, args.port, key, "idle", 0, 30.0
        )
        samples.extend(idle_samples)
        result["final_idle_observed"] = idle
        if args.mode == "publication-failure":
            os.chmod(args.checkpoint_root, 0o700)
            result["root_mode_after_export"] = 0o700
        final = health(args.host, args.port, key)
        result["final_health"] = {
            "ready": final.get("ready"),
            "busy": final.get("busy"),
            "slot_phase": final.get("slot_phase"),
            "queued_completions": final.get("queued_completions"),
        }
    if args.mode == "disconnect":
        decision = (
            result.get("first_status") == 200
            and result.get("queued_observed") is True
            and result.get("queued_client_closed") is True
            and result.get("final_health", {}).get("slot_phase") == "idle"
        )
    elif args.mode == "publication-failure":
        decision = (
            result.get("first_status") == 200
            and result.get("root_mode_after_export") == 0o700
            and result.get("final_health", {}).get("slot_phase") == "idle"
        )
    else:
        decision = (
            result.get("first_status") == 200
            and result.get("queued_observed") is True
            and result.get("queued_status") == 503
        )
    result["decision"] = "PASS" if decision else "FAIL"
    write_private(args.output / f"{args.mode}.json", result)
    print(f"checkpoint negative qualification ({args.mode}): {result['decision']}")
    return 0 if decision else 1


if __name__ == "__main__":
    raise SystemExit(main())
