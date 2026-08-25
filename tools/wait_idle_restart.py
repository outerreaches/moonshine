#!/usr/bin/env python3
"""Gracefully stop Moonshine when its current inference becomes idle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import stat
import time
import urllib.request


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise ValueError(f"private directory required: {path}")


def write_private(path: Path, value: dict[str, object]) -> None:
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


def health(host: str, port: int, key: str) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://{host}:{port}/health",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=10.0) as response:
        return json.loads(response.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--poll", default=5.0, type=float)
    parser.add_argument("--timeout", default=86400.0, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key = os.environ.get("MOONSHINE_API_KEY")
    if not key:
        raise SystemExit("MOONSHINE_API_KEY is required")
    private_dir(args.output)
    deadline = time.monotonic() + args.timeout
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        try:
            last = health(args.host, args.port, key)
            if (
                last.get("ready") is True
                and last.get("busy") is False
                and last.get("slot_phase") == "idle"
                and last.get("queued_completions") == 0
            ):
                os.kill(args.pid, signal.SIGTERM)
                write_private(args.output / "restart-trigger.json", {
                    "schema": "moonshine-idle-restart-trigger-v1",
                    "decision": "TRIGGERED",
                    "signal": "SIGTERM",
                    "pid": args.pid,
                    "idle_health": last,
                })
                print("idle-triggered graceful restart: TRIGGERED")
                return 0
        except Exception as exc:  # bounded content-free watcher error
            last = {"error_type": type(exc).__name__}
        time.sleep(args.poll)
    write_private(args.output / "restart-trigger.json", {
        "schema": "moonshine-idle-restart-trigger-v1",
        "decision": "TIMEOUT",
        "last_health": last,
    })
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
