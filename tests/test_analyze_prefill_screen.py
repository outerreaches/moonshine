from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys

TOOL = Path(__file__).parents[1] / "tools" / "analyze_prefill_screen.py"
SPEC = importlib.util.spec_from_file_location("analyze_prefill_screen", TOOL)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def write_arm(path: Path, target: float, phase: float, idle: float,
              requests: int = 100) -> None:
    fields = sorted(analyzer.REQUIRED)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for layer in range(1, 93):
            row = {field: 0 for field in fields}
            row.update({
                "capture": 1,
                "layer": layer,
                "tokens": 512,
                "unique_experts": 300 + layer % 10,
                "read_requests": requests,
                "physical_read_bytes": requests * 1000,
                "expert_pipeline_seconds": target / 92.0,
                "routed_stream_seconds": phase / 92.0,
                "ring_depth0_seconds": idle / 92.0,
                "ring_depth1_seconds": 0.1,
                "ring_depth2_seconds": 0.2,
                "ring_transitions": 2,
                "ring_max_depth": 2,
                "mzg2_launches": requests,
                "integrity_clears": requests,
                "index_h2d_copies": requests * 2,
                "gathers": requests,
                "expert_gemms": requests * 3,
                "scatters": requests,
                "stream_synchronizes": requests + 5,
            })
            writer.writerow(row)


class PrefillScreenAnalysisTest(unittest.TestCase):
    def make_paths(self, root: Path, targets: list[float],
                   phases: list[float] | None = None) -> list[Path]:
        phases = targets if phases is None else phases
        paths = []
        for index, (target, phase) in enumerate(zip(targets, phases)):
            path = root / f"arm-{index}.csv"
            write_arm(path, target, phase, idle=10.0)
            paths.append(path)
        return paths

    def test_passes_resolved_abba(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(
                Path(temporary),
                [120.0, 100.0, 100.2, 99.9, 100.1,
                 100.0, 94.0, 94.2, 100.1],
                [200.0, 200.0, 200.2, 199.9, 200.1,
                 200.0, 200.1, 200.0, 200.1],
            )
            result = analyzer.analyze(
                paths, "expert_pipeline_seconds", 5.0)
            self.assertEqual(result["decision"], "PASS")
            self.assertTrue(result["noise_floor"]["gate_resolvable"])
            self.assertGreater(result["abba"]["target_improvement_pct"], 5.0)
            self.assertAlmostEqual(
                result["baseline_ring_idle_ceiling_pct"],
                100.0 * 10.0 / 100.05,
                places=6,
            )

    def test_noise_only_stops_before_abba(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(
                Path(temporary),
                [120.0, 100.0, 102.0, 98.0, 101.0],
            )
            result = analyzer.analyze_noise(
                paths, "expert_pipeline_seconds", 5.0)
            self.assertEqual(result["decision"], "UNRESOLVABLE")
            self.assertEqual(
                result["order"], list(analyzer.NOISE_ROLES))
            self.assertGreater(
                result["noise_floor"]["three_sigma_style_floor_pct"],
                5.0,
            )
    def test_refuses_gate_inside_noise_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(
                Path(temporary),
                [120.0, 100.0, 102.0, 98.0, 101.0,
                 100.0, 94.0, 94.0, 100.0],
            )
            result = analyzer.analyze(
                paths, "expert_pipeline_seconds", 5.0)
            self.assertEqual(result["decision"], "UNRESOLVABLE")
            self.assertFalse(result["noise_floor"]["gate_resolvable"])

    def test_rejects_route_union_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_paths(root, [100.0] * 9)
            write_arm(paths[7], 100.0, 100.0, idle=10.0, requests=101)
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.analyze(paths, "expert_pipeline_seconds", 5.0)


if __name__ == "__main__":
    unittest.main()
