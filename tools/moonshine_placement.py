#!/usr/bin/env python3
"""Plan deterministic, capacity-safe placement of whole sidecar layer files.

The input is an explicit JSON profile with schema
``moonshine-sidecar-profile-v1``::

    {
      "schema": "moonshine-sidecar-profile-v1",
      "model": {
        "id": "organization/model",
        "revision": "full-source-revision",
        "source_manifest_sha256": "64 lowercase hex characters"
      },
      "store": {
        "schema": "store-format-identity",
        "manifest_sha256": "64 lowercase hex characters",
        "source_manifest_sha256": "64 lowercase hex characters"
      },
      "layers": [
        {"layer": 1, "path": "layer-001.mzg2", "bytes": 123,
         "sha256": "64 lowercase hex characters"}
      ],
      "drives": [
        {"id": "nvme0", "mount_path": "/mnt/nvme0",
         "bandwidth_bytes_per_second": 1000000,
         "free_bytes": 1000, "reserve_bytes": 100}
      ]
    }

The store source identity must equal the model source identity. Layer paths
are file names, not paths with directory components. Mount paths must be
canonical absolute POSIX paths; no path is resolved or accessed by the
planner. Only the explicitly requested output manifest is written.

Layers are assigned largest-first to the eligible drive with the lowest
assigned-bytes/bandwidth ratio. Ties use greater bandwidth and then stable
drive ID. This is weighted longest-processing-time placement: every layer is
whole, and exact integer arithmetic avoids floating-point platform variance.
Informational weighted byte targets use largest-remainder apportionment.

Output is canonical JSON (UTF-8, sorted keys, compact separators, one final
newline). ``host_profile_sha256`` hashes the normalized profile,
``inventory_sha256`` hashes the sorted layer records, and
``manifest_sha256`` hashes the canonical manifest with that field omitted.
All three hashes cover bytes without a trailing newline.
"""

from __future__ import annotations

import argparse
import dataclasses
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


PROFILE_SCHEMA = "moonshine-sidecar-profile-v1"
MANIFEST_SCHEMA = "moonshine-sidecar-placement-v1"
ALGORITHM = "weighted-lpt-largest-remainder-v1"
UINT64_MAX = (1 << 64) - 1
DRIVE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PlacementError(ValueError):
    """The profile cannot produce a validated placement."""


@dataclasses.dataclass(frozen=True)
class _Layer:
    index: int
    path: str
    size: int
    sha256: str

    def record(self) -> dict[str, object]:
        return {
            "layer": self.index,
            "path": self.path,
            "bytes": self.size,
            "sha256": self.sha256,
        }


@dataclasses.dataclass(frozen=True)
class _Drive:
    identifier: str
    mount_path: str
    bandwidth: int
    free_bytes: int
    reserve_bytes: int
    usable_bytes: int


def canonical_bytes(value: object) -> bytes:
    """Encode a JSON value in the canonical form used by manifest hashes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json(value: object) -> str:
    return canonical_bytes(value).decode("utf-8") + "\n"


def _json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _fields(
    value: object, required: set[str], context: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PlacementError(f"{context} must be an object")
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise PlacementError(f"{context} has " + "; ".join(details))
    return value


def _text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise PlacementError(f"{context} must be a non-empty clean string")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PlacementError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _integer(
    value: object, context: str, *, positive: bool = False
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
        or value > UINT64_MAX
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise PlacementError(
            f"{context} must be a {qualifier} unsigned 64-bit integer"
        )
    return value


def _layer_path(value: object, context: str) -> str:
    path = _text(value, context)
    if (
        "\\" in path
        or "/" in path
        or path in {".", ".."}
        or PurePosixPath(path).name != path
    ):
        raise PlacementError(f"{context} must be a safe file name")
    return path


def _mount_path(value: object, context: str) -> str:
    path = _text(value, context)
    parsed = PurePosixPath(path)
    if (
        not path.startswith("/")
        or "\\" in path
        or str(parsed) != path
        or ".." in parsed.parts
    ):
        raise PlacementError(f"{context} must be a canonical absolute POSIX path")
    return path


def _parse_layers(value: object) -> list[_Layer]:
    if not isinstance(value, list) or not value:
        raise PlacementError("layers must be a non-empty array")
    layers: list[_Layer] = []
    seen_indices: set[int] = set()
    seen_paths: set[str] = set()
    for position, raw in enumerate(value):
        context = f"layers[{position}]"
        item = _fields(raw, {"layer", "path", "bytes", "sha256"}, context)
        layer = _Layer(
            index=_integer(item["layer"], context + ".layer", positive=True),
            path=_layer_path(item["path"], context + ".path"),
            size=_integer(item["bytes"], context + ".bytes", positive=True),
            sha256=_sha256(item["sha256"], context + ".sha256"),
        )
        if layer.index in seen_indices:
            raise PlacementError(f"duplicate layer index: {layer.index}")
        if layer.path in seen_paths:
            raise PlacementError(f"duplicate layer path: {layer.path}")
        seen_indices.add(layer.index)
        seen_paths.add(layer.path)
        layers.append(layer)
    layers.sort(key=lambda layer: layer.index)
    return layers


def _parse_drives(value: object) -> list[_Drive]:
    if not isinstance(value, list) or not value:
        raise PlacementError("drives must be a non-empty array")
    drives: list[_Drive] = []
    seen_ids: set[str] = set()
    seen_mounts: set[str] = set()
    for position, raw in enumerate(value):
        context = f"drives[{position}]"
        item = _fields(
            raw,
            {
                "id",
                "mount_path",
                "bandwidth_bytes_per_second",
                "free_bytes",
                "reserve_bytes",
            },
            context,
        )
        identifier = _text(item["id"], context + ".id")
        if not DRIVE_ID_RE.fullmatch(identifier):
            raise PlacementError(f"{context}.id is not a stable safe identifier")
        mount_path = _mount_path(item["mount_path"], context + ".mount_path")
        free_bytes = _integer(item["free_bytes"], context + ".free_bytes")
        reserve_bytes = _integer(item["reserve_bytes"], context + ".reserve_bytes")
        drive = _Drive(
            identifier=identifier,
            mount_path=mount_path,
            bandwidth=_integer(
                item["bandwidth_bytes_per_second"],
                context + ".bandwidth_bytes_per_second",
                positive=True,
            ),
            free_bytes=free_bytes,
            reserve_bytes=reserve_bytes,
            usable_bytes=max(0, free_bytes - reserve_bytes),
        )
        if identifier in seen_ids:
            raise PlacementError(f"duplicate drive ID: {identifier}")
        if mount_path in seen_mounts:
            raise PlacementError(f"duplicate drive mount path: {mount_path}")
        seen_ids.add(identifier)
        seen_mounts.add(mount_path)
        drives.append(drive)
    drives.sort(key=lambda drive: drive.identifier)
    return drives


def _weighted_targets(drives: list[_Drive], total_bytes: int) -> dict[str, int]:
    """Apportion every byte by bandwidth, with deterministic remainder ties."""
    total_bandwidth = sum(drive.bandwidth for drive in drives)
    targets: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for drive in drives:
        target, remainder = divmod(
            total_bytes * drive.bandwidth, total_bandwidth
        )
        targets[drive.identifier] = target
        remainders.append((remainder, drive.identifier))
    unassigned = total_bytes - sum(targets.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, identifier in remainders[:unassigned]:
        targets[identifier] += 1
    return targets


def plan_placement(profile: dict[str, object]) -> dict[str, object]:
    """Validate a decoded profile and return its canonical placement value."""
    top = _fields(
        profile, {"schema", "model", "store", "layers", "drives"}, "profile"
    )
    if top["schema"] != PROFILE_SCHEMA:
        raise PlacementError(f"profile.schema must be {PROFILE_SCHEMA}")

    raw_model = _fields(
        top["model"], {"id", "revision", "source_manifest_sha256"}, "model"
    )
    model = {
        "id": _text(raw_model["id"], "model.id"),
        "revision": _text(raw_model["revision"], "model.revision"),
        "source_manifest_sha256": _sha256(
            raw_model["source_manifest_sha256"], "model.source_manifest_sha256"
        ),
    }
    raw_store = _fields(
        top["store"],
        {"schema", "manifest_sha256", "source_manifest_sha256"},
        "store",
    )
    store = {
        "schema": _text(raw_store["schema"], "store.schema"),
        "manifest_sha256": _sha256(
            raw_store["manifest_sha256"], "store.manifest_sha256"
        ),
        "source_manifest_sha256": _sha256(
            raw_store["source_manifest_sha256"], "store.source_manifest_sha256"
        ),
    }
    if store["source_manifest_sha256"] != model["source_manifest_sha256"]:
        raise PlacementError("model/store source manifest identity mismatch")

    layers = _parse_layers(top["layers"])
    drives = _parse_drives(top["drives"])
    total_bytes = sum(layer.size for layer in layers)
    total_capacity = sum(drive.usable_bytes for drive in drives)
    if total_bytes > total_capacity:
        raise PlacementError(
            f"insufficient unreserved capacity: need {total_bytes} bytes, "
            f"have {total_capacity}"
        )

    targets = _weighted_targets(drives, total_bytes)
    by_id = {drive.identifier: drive for drive in drives}
    used = {drive.identifier: 0 for drive in drives}
    assignments: dict[int, str] = {}
    for layer in sorted(layers, key=lambda item: (-item.size, item.index)):
        candidates = [
            drive
            for drive in drives
            if used[drive.identifier] + layer.size <= drive.usable_bytes
        ]
        if not candidates:
            raise PlacementError(
                f"no drive has {layer.size} unreserved bytes for whole "
                f"layer {layer.index}"
            )
        selected = min(
            candidates,
            key=lambda drive: (
                Fraction(used[drive.identifier], drive.bandwidth),
                -drive.bandwidth,
                drive.identifier,
            ),
        )
        assignments[layer.index] = selected.identifier
        used[selected.identifier] += layer.size

    placements: list[dict[str, object]] = []
    for layer in layers:
        drive = by_id[assignments[layer.index]]
        destination = str(PurePosixPath(drive.mount_path) / layer.path)
        placements.append({
            **layer.record(),
            "drive_id": drive.identifier,
            "destination": destination,
        })

    drive_records = []
    for drive in drives:
        assigned = used[drive.identifier]
        drive_records.append({
            "id": drive.identifier,
            "mount_path": drive.mount_path,
            "bandwidth_bytes_per_second": drive.bandwidth,
            "free_bytes": drive.free_bytes,
            "reserve_bytes": drive.reserve_bytes,
            "usable_bytes": drive.usable_bytes,
            "weighted_target_bytes": targets[drive.identifier],
            "assigned_bytes": assigned,
            "remaining_usable_bytes": drive.usable_bytes - assigned,
        })

    layer_records = [layer.record() for layer in layers]
    normalized_profile = {
        "schema": PROFILE_SCHEMA,
        "model": model,
        "store": store,
        "layers": layer_records,
        "drives": [
            {
                "id": drive.identifier,
                "mount_path": drive.mount_path,
                "bandwidth_bytes_per_second": drive.bandwidth,
                "free_bytes": drive.free_bytes,
                "reserve_bytes": drive.reserve_bytes,
            }
            for drive in drives
        ],
    }
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "algorithm": ALGORITHM,
        "model": model,
        "store": {
            **store,
            "inventory_sha256": _json_sha256(layer_records),
            "layer_count": len(layers),
        },
        "host_profile_sha256": _json_sha256(normalized_profile),
        "total_bytes": total_bytes,
        "drives": drive_records,
        "placements": placements,
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    return manifest


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PlacementError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def decode_profile(text: str) -> dict[str, object]:
    """Decode JSON while rejecting ambiguous duplicate object keys."""
    value = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise PlacementError("profile must be a JSON object")
    return value


def load_profile(path: Path) -> dict[str, object]:
    return decode_profile(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Atomically write only the requested manifest and its sibling temporary."""
    temporary = path.with_name(path.name + ".tmp")
    created = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(canonical_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", help="JSON host profile, or - for stdin")
    parser.add_argument(
        "-o", "--output", type=Path, help="manifest path; default is stdout"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profile = (
            decode_profile(sys.stdin.read())
            if args.profile == "-"
            else load_profile(Path(args.profile))
        )
        manifest = plan_placement(profile)
        if args.output is None:
            sys.stdout.write(canonical_json(manifest))
        else:
            write_manifest(args.output, manifest)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"moonshine-placement: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
