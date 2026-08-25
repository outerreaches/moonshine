#!/usr/bin/env python3
"""Wait for the next checkpoint publication, then gracefully stop Moonshine."""

from __future__ import annotations

import argparse
import hashlib
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


def manifest_fingerprint(path: Path) -> dict[str, object]:
    info = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "mtime_ns": info.st_mtime_ns,
        "size": info.st_size,
        "sha256": digest,
    }


def idle_health(host: str, port: int, key: str) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://{host}:{port}/health",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=10.0) as response:
        return json.loads(response.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--poll", default=5.0, type=float)
    parser.add_argument("--idle-timeout", default=120.0, type=float)
    parser.add_argument("--timeout", default=86400.0, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key = os.environ.get("MOONSHINE_API_KEY")
    if not key:
        raise SystemExit("MOONSHINE_API_KEY is required")
    private_dir(args.output)
    baseline = manifest_fingerprint(args.manifest)
    deadline = time.monotonic() + args.timeout
    changed: dict[str, object] | None = None
    while time.monotonic() < deadline:
        time.sleep(args.poll)
        current = manifest_fingerprint(args.manifest)
        if current == baseline:
            continue
        changed = {"before": baseline, "after": current}
        break
    if changed is None:
        write_private(args.output / "restart-trigger.json", {
            "schema": "moonshine-publication-restart-trigger-v1",
            "decision": "TIMEOUT_NO_PUBLICATION",
        })
        return 1

    idle_deadline = time.monotonic() + args.idle_timeout
    last_health: dict[str, object] = {}
    while time.monotonic() < idle_deadline:
        last_health = idle_health(args.host, args.port, key)
        if (
            last_health.get("ready") is True
            and last_health.get("busy") is False
            and last_health.get("slot_phase") == "idle"
            and last_health.get("queued_completions") == 0
        ):
            os.kill(args.pid, signal.SIGTERM)
            write_private(args.output / "restart-trigger.json", {
                "schema": "moonshine-publication-restart-trigger-v1",
                "decision": "TRIGGERED",
                "signal": "SIGTERM",
                "pid": args.pid,
                "manifest_change": changed,
                "idle_health": last_health,
            })
            print("publication-triggered graceful restart: TRIGGERED")
            return 0
        time.sleep(1.0)
    write_private(args.output / "restart-trigger.json", {
        "schema": "moonshine-publication-restart-trigger-v1",
        "decision": "TIMEOUT_WAITING_FOR_IDLE",
        "manifest_change": changed,
        "last_health": last_health,
    })
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
