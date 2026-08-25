#!/usr/bin/env python3
"""Monitor a checkpoint observation window without retaining model content."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import time
import urllib.request


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise ValueError(f"observation directory must be private: {path}")


def append_json(path: Path, value: dict[str, object]) -> None:
    with path.open("ab") as handle:
        handle.write((json.dumps(value, sort_keys=True) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def get_health(host: str, port: int, key: str) -> tuple[dict[str, object], float]:
    request = urllib.request.Request(
        f"http://{host}:{port}/health",
        headers={"Authorization": f"Bearer {key}"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=10.0) as response:
        value = json.loads(response.read())
    return value, time.monotonic() - started


def storage(root: Path) -> dict[str, object]:
    statvfs = os.statvfs(root)
    free_bytes = statvfs.f_bavail * statvfs.f_frsize
    entry_count = 0
    total_bytes = 0
    private_files = True
    orphan_temporary = 0
    for path in root.rglob("*"):
        info = path.stat()
        if path.is_dir():
            private_files = private_files and (info.st_mode & 0o077) == 0
            continue
        if path.name.endswith(".tmp"):
            orphan_temporary += 1
        if path.name.endswith(".state") or path.name.endswith(".meta"):
            entry_count += 1 if path.name.endswith(".meta") else 0
            total_bytes += info.st_size
        elif path.name == "manifest.bin":
            total_bytes += info.st_size
        private_files = private_files and stat.S_ISREG(info.st_mode) and (info.st_mode & 0o077) == 0
    return {
        "free_bytes": free_bytes,
        "entry_count": entry_count,
        "total_bytes": total_bytes,
        "private_files": private_files,
        "orphan_temporary": orphan_temporary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--duration", default=86400.0, type=float)
    parser.add_argument("--interval", default=5.0, type=float)
    parser.add_argument("--storage-interval", default=900.0, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key = os.environ.get("MOONSHINE_API_KEY")
    if not key:
        raise SystemExit("MOONSHINE_API_KEY is required")
    private_dir(args.output)
    if not args.checkpoint_root.is_dir():
        raise SystemExit("checkpoint root does not exist")
    events = args.output / "events.jsonl"
    summary = args.output / "summary.json"
    started = time.time()
    deadline = time.monotonic() + args.duration
    next_storage = 0.0
    errors = 0
    health_samples = 0
    storage_samples = 0
    max_health_seconds = 0.0
    first_error: str | None = None
    while time.monotonic() < deadline:
        now = time.time()
        try:
            value, elapsed = get_health(args.host, args.port, key)
            health_samples += 1
            max_health_seconds = max(max_health_seconds, elapsed)
            append_json(events, {
                "event": "health",
                "timestamp": now,
                "ready": value.get("ready"),
                "busy": value.get("busy"),
                "slot_phase": value.get("slot_phase"),
                "queued_completions": value.get("queued_completions"),
                "checkpoint_queue_capacity": value.get("checkpoint_queue_capacity"),
                "prefix_checkpoints": value.get("prefix_checkpoints"),
                "elapsed_seconds": elapsed,
            })
        except Exception as exc:  # bounded monitor error; no content captured
            errors += 1
            if first_error is None:
                first_error = type(exc).__name__
            append_json(events, {
                "event": "health_error",
                "timestamp": now,
                "error_type": type(exc).__name__,
            })
        current = time.monotonic()
        if current >= next_storage:
            try:
                sample = storage(args.checkpoint_root)
                storage_samples += 1
                append_json(events, {
                    "event": "storage",
                    "timestamp": now,
                    **sample,
                })
            except Exception as exc:  # bounded monitor error; no content captured
                errors += 1
                if first_error is None:
                    first_error = type(exc).__name__
                append_json(events, {
                    "event": "storage_error",
                    "timestamp": now,
                    "error_type": type(exc).__name__,
                })
            next_storage = current + args.storage_interval
        time.sleep(args.interval)
    result = {
        "schema": "moonshine-checkpoint-observation-monitor-v1",
        "started_at": started,
        "duration_seconds": time.time() - started,
        "health_samples": health_samples,
        "storage_samples": storage_samples,
        "errors": errors,
        "first_error_type": first_error,
        "max_health_seconds": max_health_seconds,
        "events": str(events),
    }
    append_json(summary, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
