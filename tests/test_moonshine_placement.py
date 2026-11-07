#!/usr/bin/env python3
"""Behavioral tests for deterministic whole-layer sidecar placement."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "tools" / "moonshine_placement.py"
SPEC = importlib.util.spec_from_file_location("moonshine_placement", TOOL)
assert SPEC and SPEC.loader
placement = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = placement
SPEC.loader.exec_module(placement)

SOURCE_SHA256 = "0" * 64
STORE_SHA256 = "1" * 64


def layer_digest(index: int) -> str:
    return hashlib.sha256(f"sidecar-layer-{index}".encode()).hexdigest()


def make_profile(
    sizes: tuple[int, ...] = (10, 10),
    drives: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if drives is None:
        drives = [
            {
                "id": "nvme",
                "mount_path": "/mnt/nvme",
                "bandwidth_bytes_per_second": 100,
                "free_bytes": sum(sizes) + 50,
                "reserve_bytes": 50,
            }
        ]
    return {
        "schema": placement.PROFILE_SCHEMA,
        "model": {
            "id": "example/model",
            "revision": "a" * 40,
            "source_manifest_sha256": SOURCE_SHA256,
        },
        "store": {
            "schema": "example-sidecar-store-v1",
            "manifest_sha256": STORE_SHA256,
            "source_manifest_sha256": SOURCE_SHA256,
        },
        "layers": [
            {
                "layer": index,
                "path": f"layer-{index:03d}.mzg2",
                "bytes": size,
                "sha256": layer_digest(index),
            }
            for index, size in enumerate(sizes, 1)
        ],
        "drives": drives,
    }


def drive(
    identifier: str,
    usable: int,
    bandwidth: int = 100,
    reserve: int = 0,
) -> dict[str, object]:
    return {
        "id": identifier,
        "mount_path": f"/mnt/{identifier}",
        "bandwidth_bytes_per_second": bandwidth,
        "free_bytes": usable + reserve,
        "reserve_bytes": reserve,
    }


def assigned_by_drive(manifest: dict[str, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in manifest["drives"]:
        result[record["id"]] = record["assigned_bytes"]
    return result


class PlacementTests(unittest.TestCase):
    def test_bandwidth_weighted_balance_and_whole_layers(self):
        profile = make_profile(
            (10, 10, 10, 10, 10, 10),
            [drive("slow", 20, 100), drive("fast", 60, 300)],
        )
        manifest = placement.plan_placement(profile)

        self.assertEqual(assigned_by_drive(manifest), {"fast": 40, "slow": 20})
        targets = {
            record["id"]: record["weighted_target_bytes"]
            for record in manifest["drives"]
        }
        self.assertEqual(targets, {"fast": 45, "slow": 15})
        self.assertEqual(manifest["total_bytes"], 60)
        self.assertEqual(len(manifest["placements"]), 6)
        for record in manifest["placements"]:
            self.assertIn(record["drive_id"], {"fast", "slow"})
            self.assertEqual(
                record["destination"],
                f"/mnt/{record['drive_id']}/{record['path']}",
            )
            self.assertNotIn("stripes", record)
            self.assertNotIn("parts", record)

    def test_exact_capacity_and_reserve_boundaries(self):
        profile = make_profile(
            (60, 40),
            [
                drive("a", 60, reserve=20),
                drive("b", 40, reserve=10),
            ],
        )
        manifest = placement.plan_placement(profile)
        records = {record["id"]: record for record in manifest["drives"]}

        self.assertEqual(records["a"]["usable_bytes"], 60)
        self.assertEqual(records["b"]["usable_bytes"], 40)
        self.assertEqual(records["a"]["assigned_bytes"], 60)
        self.assertEqual(records["b"]["assigned_bytes"], 40)
        self.assertEqual(records["a"]["remaining_usable_bytes"], 0)
        self.assertEqual(records["b"]["remaining_usable_bytes"], 0)
        self.assertEqual(
            {item["layer"]: item["drive_id"] for item in manifest["placements"]},
            {1: "a", 2: "b"},
        )

    def test_output_is_deterministic_and_hashes_validate(self):
        profile = make_profile(
            (7, 3, 5, 2),
            [drive("z", 20, 250), drive("a", 20, 100)],
        )
        reordered = copy.deepcopy(profile)
        reordered["layers"].reverse()
        reordered["drives"].reverse()

        first = placement.plan_placement(profile)
        second = placement.plan_placement(reordered)
        self.assertEqual(placement.canonical_bytes(first), placement.canonical_bytes(second))
        self.assertTrue(placement.canonical_json(first).endswith("\n"))
        self.assertEqual(
            placement.canonical_json(first),
            json.dumps(
                first,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ) + "\n",
        )

        unsigned = dict(first)
        digest = unsigned.pop("manifest_sha256")
        self.assertEqual(
            digest, hashlib.sha256(placement.canonical_bytes(unsigned)).hexdigest()
        )
        inventory = [
            {
                key: record[key]
                for key in ("layer", "path", "bytes", "sha256")
            }
            for record in first["placements"]
        ]
        self.assertEqual(
            first["store"]["inventory_sha256"],
            hashlib.sha256(placement.canonical_bytes(inventory)).hexdigest(),
        )
        self.assertEqual(first["model"]["revision"], "a" * 40)
        self.assertEqual(first["store"]["manifest_sha256"], STORE_SHA256)

    def test_largest_remainder_and_tie_are_exact_and_deterministic(self):
        profile = make_profile(
            (2, 1, 1, 1),
            [drive("b", 2), drive("a", 3)],
        )
        manifest = placement.plan_placement(profile)
        records = {record["id"]: record for record in manifest["drives"]}

        self.assertEqual(records["a"]["weighted_target_bytes"], 3)
        self.assertEqual(records["b"]["weighted_target_bytes"], 2)
        self.assertEqual(records["a"]["assigned_bytes"], 3)
        self.assertEqual(records["b"]["assigned_bytes"], 2)
        self.assertEqual(
            sum(record["weighted_target_bytes"] for record in manifest["drives"]),
            manifest["total_bytes"],
        )

    def test_insufficient_total_capacity_fails_closed(self):
        profile = make_profile((11,), [drive("a", 10)])
        with self.assertRaisesRegex(placement.PlacementError, "insufficient"):
            placement.plan_placement(profile)

    def test_layer_must_fit_wholesale_on_one_drive(self):
        profile = make_profile(
            (8, 2, 2),
            [drive("a", 6), drive("b", 6)],
        )
        with self.assertRaisesRegex(placement.PlacementError, "whole layer 1"):
            placement.plan_placement(profile)

    def test_malformed_profiles_are_rejected(self):
        cases = []

        def case(name, mutate):
            cases.append((name, mutate))

        def append_duplicate_index(profile):
            duplicate = copy.deepcopy(profile["layers"][0])
            duplicate["path"] = "different.mzg2"
            profile["layers"].append(duplicate)

        def append_duplicate_path(profile):
            duplicate = copy.deepcopy(profile["layers"][0])
            duplicate["layer"] = 2
            profile["layers"].append(duplicate)

        def append_duplicate_drive_id(profile):
            duplicate = copy.deepcopy(profile["drives"][0])
            duplicate["mount_path"] = "/mnt/different"
            profile["drives"].append(duplicate)

        def append_duplicate_mount(profile):
            duplicate = copy.deepcopy(profile["drives"][0])
            duplicate["id"] = "different"
            profile["drives"].append(duplicate)

        case("duplicate layer index", append_duplicate_index)
        case("duplicate layer path", append_duplicate_path)
        case("duplicate drive ID", append_duplicate_drive_id)
        case("duplicate drive mount", append_duplicate_mount)
        case(
            "zero bandwidth",
            lambda profile: profile["drives"][0].update(
                bandwidth_bytes_per_second=0
            ),
        )
        case(
            "boolean bandwidth",
            lambda profile: profile["drives"][0].update(
                bandwidth_bytes_per_second=True
            ),
        )
        case(
            "zero layer bytes",
            lambda profile: profile["layers"][0].update(bytes=0),
        )
        case(
            "negative layer bytes",
            lambda profile: profile["layers"][0].update(bytes=-1),
        )
        case(
            "negative reserve",
            lambda profile: profile["drives"][0].update(reserve_bytes=-1),
        )
        case(
            "identity mismatch",
            lambda profile: profile["store"].update(
                source_manifest_sha256="f" * 64
            ),
        )
        case(
            "uppercase layer digest",
            lambda profile: profile["layers"][0].update(sha256="A" * 64),
        )
        case(
            "unknown field",
            lambda profile: profile.update(unexpected=True),
        )
        case(
            "wrong schema",
            lambda profile: profile.update(schema="something-else"),
        )

        for name, mutate in cases:
            with self.subTest(name=name):
                profile = make_profile()
                mutate(profile)
                with self.assertRaises(placement.PlacementError):
                    placement.plan_placement(profile)

    def test_unsafe_paths_are_rejected(self):
        for path in ("../layer", "/layer", "directory/layer", "directory\\layer", "."):
            with self.subTest(layer_path=path):
                profile = make_profile()
                profile["layers"][0]["path"] = path
                with self.assertRaises(placement.PlacementError):
                    placement.plan_placement(profile)

        for path in ("relative", "/mnt/../escape", "/mnt/drive/", "/mnt//drive"):
            with self.subTest(mount_path=path):
                profile = make_profile()
                profile["drives"][0]["mount_path"] = path
                with self.assertRaises(placement.PlacementError):
                    placement.plan_placement(profile)

    def test_json_duplicate_keys_are_rejected(self):
        encoded = placement.canonical_json(make_profile())
        duplicate = encoded.replace(
            '"schema":"moonshine-sidecar-profile-v1"',
            '"schema":"moonshine-sidecar-profile-v1",'
            '"schema":"moonshine-sidecar-profile-v1"',
            1,
        )
        with self.assertRaisesRegex(placement.PlacementError, "duplicate JSON"):
            placement.decode_profile(duplicate)


class CommandTests(unittest.TestCase):
    def run_cli(self, profile_path: Path, output: Path):
        return subprocess.run(
            [sys.executable, str(TOOL), str(profile_path), "--output", str(output)],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_cli_writes_only_requested_canonical_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_path = root / "host.json"
            output = root / "placement.json"
            profile_path.write_text(
                placement.canonical_json(make_profile()), encoding="utf-8"
            )
            before = set(root.iterdir())

            result = self.run_cli(profile_path, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(set(root.iterdir()), before | {output})
            manifest = placement.plan_placement(placement.load_profile(profile_path))
            self.assertEqual(output.read_bytes(), placement.canonical_bytes(manifest) + b"\n")

    def test_cli_failure_does_not_create_or_replace_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_path = root / "host.json"
            output = root / "placement.json"
            profile_path.write_text(
                placement.canonical_json(make_profile((11,), [drive("a", 10)])),
                encoding="utf-8",
            )
            output.write_bytes(b"existing-manifest")
            before = {path.name: path.read_bytes() for path in root.iterdir()}

            result = self.run_cli(profile_path, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("insufficient", result.stderr)
            after = {path.name: path.read_bytes() for path in root.iterdir()}
            self.assertEqual(after, before)
    def test_cli_preserves_preexisting_sibling_temp_on_open_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_path = root / "host.json"
            output = root / "placement.json"
            temporary_output = root / "placement.json.tmp"
            profile_path.write_text(
                placement.canonical_json(make_profile()), encoding="utf-8"
            )
            temporary_output.write_bytes(b"foreign-temp")

            result = self.run_cli(profile_path, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(temporary_output.read_bytes(), b"foreign-temp")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
