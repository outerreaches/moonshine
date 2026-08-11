#!/usr/bin/env python3
"""Tests for the stdlib-only decode-cache analyzer."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "tools" / "analyze_decode_cache.py"
SPEC = importlib.util.spec_from_file_location("analyze_decode_cache", TOOL)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def make_synthetic_run(path: Path, capacity: int = 18, steps: int = 3,
                       corrupt_mask: bool = False,
                       warm_first_layer: bool = False) -> None:
    """Create a strict 92-layer, 16-expert current-schema capture."""
    path.mkdir(parents=True, exist_ok=True)
    states = [analyzer.LayerState.from_experts(capacity, ())
              for _ in range(analyzer.LAYERS)]
    cache_rows = []
    if warm_first_layer:
        if capacity < 24:
            raise ValueError("warm first-layer fixture requires capacity >= 24")
        first_base = 23
        first_seed = tuple(range(first_base, first_base + 24))
        states[0] = analyzer.LayerState.from_experts(capacity, first_seed)
        cache_rows = [(1, 1, rank, expert)
                      for rank, expert in enumerate(first_seed)]
    _write_csv(path / "decode.cache.csv", analyzer.CACHE_HEADER, cache_rows)
    route_rows = []
    observed_hits = [[0] * analyzer.LAYERS for _ in range(steps)]
    experts_by_layer = {}
    for step in range(steps):
        for layer in range(1, analyzer.LAYERS + 1):
            base = (layer * 23) % analyzer.EXPERTS
            if step == 0:
                experts = tuple((base + rank) % analyzer.EXPERTS
                                for rank in range(analyzer.TOP_K))
            elif step == 1:
                experts = tuple((base + 8 + rank) % analyzer.EXPERTS
                                for rank in range(analyzer.TOP_K))
            else:
                # Repeat the first set, making cache behavior nontrivial.
                experts = tuple((base + rank) % analyzer.EXPERTS
                                for rank in range(analyzer.TOP_K))
            experts_by_layer[layer] = experts
            plan = states[layer - 1].plan(experts)
            mask = plan.hit_mask
            states[layer - 1].commit()
            if corrupt_mask and step == 0 and layer == 1:
                mask ^= 1
            observed_hits[step][layer - 1] = mask.bit_count()
            route_rows.append((
                1, step, 100 + step, layer, mask, *experts,
            ))
    _write_csv(path / "decode.routes.csv", analyzer.ROUTES_HEADER, route_rows)

    per_layer = []
    for layer in range(1, analyzer.LAYERS + 1):
        accesses = steps * analyzer.TOP_K
        hits = sum(observed_hits[step][layer - 1] for step in range(steps))
        misses = accesses - hits
        per_layer.append({
            "capture": 1,
            "scope": "layer",
            "layer": layer,
            "steps": steps,
            "accesses": accesses,
            "hits": hits,
            "misses": misses,
            "read_requests": misses,
            "logical_expert_bytes": misses * 100,
            "physical_read_bytes": misses * 100,
            "wait_calls": misses,
            "completions": misses,
            "max_inflight": 1 if misses else 0,
            "pre_moe_seconds": "0.000000000",
            "io_wait_seconds": "0.000000000",
            "expert_pipeline_seconds": "0.000000000",
            "expert_sync_seconds": "0.000000000",
            "shared_sync_seconds": "0.000000000",
            "host_interval_seconds": "0.000000000",
        })
    additive = analyzer.ADDITIVE_INTEGER_FIELDS
    summary = {
        "capture": 1,
        "scope": "summary",
        "layer": 0,
        "steps": steps,
        **{name: sum(row[name] for row in per_layer) for name in additive},
        "max_inflight": max(row["max_inflight"] for row in per_layer),
        "pre_moe_seconds": "0.000000000",
        "io_wait_seconds": "0.000000000",
        "expert_pipeline_seconds": "0.000000000",
        "expert_sync_seconds": "0.000000000",
        "shared_sync_seconds": "0.000000000",
        "host_interval_seconds": "0.000000000",
    }
    ledger_rows = [summary, *per_layer]
    _write_csv(
        path / "decode.ledger.csv",
        (*analyzer.LEDGER_PREFIX, "host_interval_seconds"),
        ([row[name] for name in (*analyzer.LEDGER_PREFIX,
                                  "host_interval_seconds")]
         for row in ledger_rows),
    )


class ExactBatchLRUTest(unittest.TestCase):
    def test_frozen_hits_and_rank_ordered_pending_commit(self) -> None:
        state = analyzer.LayerState.from_experts(2, ())
        plan = state.plan((1, 2, 3))
        self.assertEqual(plan.hit_mask, 0)
        self.assertEqual([access.admit for access in plan.accesses],
                         [False, True, True])
        self.assertEqual([entry.expert for entry in plan.entries], [2, 3])
        state.commit()

        # Expert 3 is a frozen hit even though rank-order touches later evict it.
        plan = state.plan((3, 2, 4))
        self.assertEqual(plan.hit_mask, 0b011)
        self.assertEqual([entry.expert for entry in plan.entries], [2, 4])
        self.assertEqual([access.hit for access in plan.accesses],
                         [True, True, False])
        state.commit()
        self.assertEqual(state.metrics.hits, 2)
        self.assertEqual(state.metrics.accesses, 6)

    def test_abort_is_atomic_and_same_batch_reentry_reuses_slot(self) -> None:
        state = analyzer.LayerState.from_experts(2, (1, 3))
        before = list(state.entries)
        plan = state.plan((3, 1))
        self.assertEqual(plan.hit_mask, 0b11)
        self.assertEqual({entry.slot for entry in plan.entries}, {0, 1})
        state.abort()
        self.assertEqual(state.entries, before)
        self.assertEqual(state.metrics.accesses, 0)
        with self.assertRaises(analyzer.AnalysisError):
            state.commit()

    def test_second_plan_and_duplicate_batch_rejected(self) -> None:
        state = analyzer.LayerState.from_experts(2, ())
        state.plan((1, 2))
        with self.assertRaisesRegex(analyzer.AnalysisError, "pending"):
            state.plan((3, 4))
        state.abort()
        with self.assertRaisesRegex(analyzer.AnalysisError, "duplicate"):
            state.plan((1, 1))

    def test_capacity_below_batch_only_final_misses_admitted(self) -> None:
        state = analyzer.LayerState.from_experts(2, ())
        plan = state.plan((10, 11, 12, 13))
        self.assertEqual([access.admit for access in plan.accesses],
                         [False, False, True, True])
        self.assertEqual([entry.expert for entry in plan.entries], [12, 13])
        self.assertEqual(len({entry.slot for entry in plan.entries}), 2)

    def test_pin_selection_reserves_exact_count_with_stable_zero_ties(self) -> None:
        pins = analyzer.select_pins((), 3)
        self.assertEqual(len(pins), analyzer.LAYERS)
        self.assertTrue(all(layer_pins == (0, 1, 2) for layer_pins in pins))



class DPTest(unittest.TestCase):
    def test_exact_nonconcave_and_lexicographic_tie(self) -> None:
        curves = [
            {1: (0, 5, 5), 2: (1, 4, 4), 3: (10, 3, 3)},
            {1: (0, 5, 5), 2: (6, 4, 4), 3: (6, 3, 3)},
            {1: (0, 5, 5), 2: (6, 4, 4), 3: (6, 3, 3)},
        ]
        # Greedy marginal selection can miss the jump at first-layer C=3.
        vector = analyzer.exact_fixed_total_dp_generic(curves, 7, 1, 3)
        self.assertEqual(vector, (3, 2, 2))

        tied = [
            {1: (1, 1, 1), 2: (1, 1, 1)},
            {1: (1, 1, 1), 2: (1, 1, 1)},
        ]
        self.assertEqual(
            analyzer.exact_fixed_total_dp_generic(tied, 3, 1, 2),
            (1, 2),
        )

    def test_generic_dp_matches_exhaustive(self) -> None:
        curves = [
            {1: (1, 7, 3), 2: (4, 6, 2), 3: (4, 5, 1)},
            {1: (2, 9, 8), 2: (2, 8, 7), 3: (8, 7, 6)},
            {1: (0, 4, 5), 2: (5, 3, 4), 3: (6, 2, 3)},
        ]
        total = 6
        candidates = []
        for vector in itertools.product(range(1, 4), repeat=3):
            if sum(vector) != total:
                continue
            h = sum(curves[i][c][0] for i, c in enumerate(vector))
            a = sum(curves[i][c][1] for i, c in enumerate(vector))
            e = sum(curves[i][c][2] for i, c in enumerate(vector))
            candidates.append(((h, -a, -e), vector))
        best_score = max(score for score, _ in candidates)
        expected = min(vector for score, vector in candidates
                       if score == best_score)
        self.assertEqual(
            analyzer.exact_fixed_total_dp_generic(curves, total, 1, 3),
            expected,
        )


class StrictTraceAndReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        make_synthetic_run(self.run)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_strict_load_reconcile_and_source_mask_gate(self) -> None:
        trace = analyzer.load_trace(self.run, capture=1,
                                    observed_capacity=18)
        self.assertEqual(trace.step_count, 3)
        self.assertEqual(trace.accesses, 3 * 92 * 16)
        self.assertEqual(trace.expert_bytes, 100)
        gate = analyzer.source_replay_gate(trace, 18)
        self.assertEqual(gate.mask_mismatches, 0)
        self.assertEqual(gate.total.hits, int(trace.ledger_summary["hits"]))

    def test_capture_is_explicit_and_absent_capture_fails(self) -> None:
        with self.assertRaisesRegex(analyzer.AnalysisError, "capture 2"):
            analyzer.load_trace(self.run, capture=2,
                                observed_capacity=18)

    def test_symlink_input_is_rejected(self) -> None:
        cache = self.run / "decode.cache.csv"
        target = self.run / "cache-target.csv"
        cache.rename(target)
        cache.symlink_to(target.name)
        with self.assertRaisesRegex(analyzer.AnalysisError, "cannot read input"):
            analyzer.load_trace(self.run, 1, 18)

    def test_mixed_capture_input_is_rejected(self) -> None:
        routes = self.run / "decode.routes.csv"
        with routes.open(newline="") as fh:
            rows = list(csv.reader(fh))
        rows[1][0] = "2"
        _write_csv(routes, rows[0], rows[1:])
        with self.assertRaisesRegex(analyzer.AnalysisError,
                                    "exactly requested capture 1"):
            analyzer.load_trace(self.run, 1, 18)

    def test_duplicate_experts_and_route_order_are_rejected(self) -> None:
        routes = self.run / "decode.routes.csv"
        with routes.open(newline="") as fh:
            rows = list(csv.reader(fh))
        rows[1][-1] = rows[1][-2]
        _write_csv(routes, rows[0], rows[1:])
        with self.assertRaisesRegex(analyzer.AnalysisError, "16 unique"):
            analyzer.load_trace(self.run, 1, 18)

        make_synthetic_run(self.run)
        with routes.open(newline="") as fh:
            rows = list(csv.reader(fh))
        rows[2][3] = "3"
        _write_csv(routes, rows[0], rows[1:])
        with self.assertRaisesRegex(analyzer.AnalysisError, "step-major"):
            analyzer.load_trace(self.run, 1, 18)

    def test_zero_byte_expert_accounting_is_rejected(self) -> None:
        ledger = self.run / "decode.ledger.csv"
        with ledger.open(newline="") as fh:
            rows = list(csv.reader(fh))
        logical = rows[0].index("logical_expert_bytes")
        physical = rows[0].index("physical_read_bytes")
        for row in rows[1:]:
            row[logical] = "0"
            row[physical] = "0"
        _write_csv(ledger, rows[0], rows[1:])
        with self.assertRaisesRegex(analyzer.AnalysisError,
                                    "positive integer expert size"):
            analyzer.load_trace(self.run, 1, 18)

    def test_physical_bytes_without_reads_are_rejected(self) -> None:
        make_synthetic_run(
            self.run, capacity=24, warm_first_layer=True)
        ledger = self.run / "decode.ledger.csv"
        with ledger.open(newline="") as fh:
            rows = list(csv.reader(fh))
        scope = rows[0].index("scope")
        layer = rows[0].index("layer")
        physical = rows[0].index("physical_read_bytes")
        summary = next(row for row in rows[1:] if row[scope] == "summary")
        layer_one = next(
            row for row in rows[1:]
            if row[scope] == "layer" and row[layer] == "1"
        )
        self.assertEqual(layer_one[rows[0].index("read_requests")], "0")
        layer_one[physical] = "1"
        summary[physical] = str(int(summary[physical]) + 1)
        _write_csv(ledger, rows[0], rows[1:])
        with self.assertRaisesRegex(analyzer.AnalysisError,
                                    "physical bytes with no reads"):
            analyzer.load_trace(self.run, 1, 24)

    def test_strict_schema_rejected(self) -> None:
        ledger = self.run / "decode.ledger.csv"
        with ledger.open(newline="") as fh:
            rows = list(csv.reader(fh))
        rows[0][-1] = "renamed_time"
        _write_csv(ledger, rows[0], rows[1:])
        with self.assertRaisesRegex(analyzer.AnalysisError, "unexpected schema"):
            analyzer.load_trace(self.run, 1, 18)

    def test_zero_mismatch_gate_is_mandatory(self) -> None:
        make_synthetic_run(self.run, corrupt_mask=True)
        trace = analyzer.load_trace(self.run, 1, 18)
        with self.assertRaisesRegex(analyzer.AnalysisError, "mask mismatch"):
            analyzer.source_replay_gate(trace, 18)

    def test_deterministic_outputs_and_policy_labels(self) -> None:
        out1 = self.root / "out1"
        out2 = self.root / "out2"
        common = [
            "report", str(self.run), "--capture", "1",
            "--observed-capacity", "18", "--fresh-empty-source",
            "--marginal-range", "16:20",
            "--total-slots", str(18 * 92), "--min-layer-slots", "16",
            "--max-layer-slots", "20", "--pin-counts", "0,1,2",
            "--training-steps", "1",
        ]
        self.assertEqual(analyzer.main([*common, "--out", str(out1)]), 0)
        self.assertEqual(analyzer.main([*common, "--out", str(out2)]), 0)
        names = sorted(path.name for path in out1.iterdir())
        self.assertEqual(names, [
            "allocations.csv", "analysis.json", "layer-marginals.csv",
            "policies.csv", "report.txt", "uniform.csv",
        ])
        self.assertEqual(out1.stat().st_mode & 0o777, 0o700)
        for name in names:
            self.assertEqual((out1 / name).read_bytes(),
                             (out2 / name).read_bytes(), name)
            self.assertEqual((out1 / name).stat().st_mode & 0o777, 0o600)
        analysis = json.loads((out1 / "analysis.json").read_text())
        with (out1 / "allocations.csv").open(newline="") as fh:
            allocation_rows = list(csv.DictReader(fh))
        self.assertEqual(len(allocation_rows), analyzer.LAYERS)
        self.assertTrue(all(
            row["oracle_nonpromotable"] == "True" and
            row["future_data_leakage"] == "True" and
            row["comparison_scope"] == "full_capture" and
            "oracle/nonpromotable" in row["deployability_label"]
            for row in allocation_rows
        ))
        self.assertNotIn("timestamp", json.dumps(analysis).lower())
        self.assertTrue(analysis["source_replay_gate"]["passed"])
        self.assertTrue(analysis["allocation"]["oracle_nonpromotable"])
        self.assertTrue(analysis["allocation"]["future_data_leakage"])
        frequency = next(policy for policy in analysis["policies"]
                         if policy["policy"].startswith("experimental_frequency"))
        self.assertEqual(frequency["requested_batch_residency_violations"], 0)
        self.assertFalse(frequency["future_data_leakage"])
        self.assertEqual(
            frequency["frequency_metadata"]["counter_total"],
            trace_accesses := 3 * 92 * 16,
        )
        self.assertEqual(frequency["accesses"], trace_accesses)
        self.assertIn("not canonical", frequency["deployability_label"])
        oracle = next(policy for policy in analysis["policies"]
                      if policy["policy"] == "oracle_pinned_lru")
        self.assertTrue(oracle["oracle_nonpromotable"])
        self.assertTrue(oracle["future_data_leakage"])
        prefix = next(policy for policy in analysis["policies"]
                      if policy["policy"] == "prefix_trained_pinned_lru")
        self.assertEqual(prefix["comparison_scope"], "cold_suffix_only")
        self.assertFalse(prefix["oracle_nonpromotable"])
        self.assertFalse(prefix["future_data_leakage"])
        self.assertEqual(set(analysis["input"]["sha256"]),
                         {"cache", "ledger", "routes"})
        self.assertEqual(analysis["input"]["paths"], {
            "cache": "decode.cache.csv",
            "ledger": "decode.ledger.csv",
            "routes": "decode.routes.csv",
        })
        self.assertTrue(analysis["parameters"]["fresh_empty_source"])

    def test_larger_capacity_requires_explicit_fresh_empty_provenance(self) -> None:
        trace = analyzer.load_trace(self.run, 1, 18)
        with self.assertRaisesRegex(analyzer.AnalysisError,
                                    "fresh-empty-source provenance"):
            analyzer.require_reconstructible_targets(trace, 18, (19,))
        fresh = analyzer.load_trace(
            self.run, 1, 18, fresh_empty_source=True)
        analyzer.require_reconstructible_targets(fresh, 18, (19,))

    def test_fresh_empty_attestation_rejects_nonempty_snapshot(self) -> None:
        _write_csv(
            self.run / "decode.cache.csv", analyzer.CACHE_HEADER,
            [(1, 1, 0, 17)],
        )
        with self.assertRaisesRegex(analyzer.AnalysisError,
                                    "completely empty cache snapshot"):
            analyzer.load_trace(
                self.run, 1, 18, fresh_empty_source=True)

    def test_warm_source_larger_capacity_rejected_when_any_layer_full(self) -> None:
        trace = analyzer.load_trace(self.run, 1, 18)
        warm = analyzer.Trace(
            capture=trace.capture,
            batches=trace.batches,
            initial_lru=(tuple(range(18)), *trace.initial_lru[1:]),
            ledger_summary=trace.ledger_summary,
            ledger_schema=trace.ledger_schema,
            input_sha256=trace.input_sha256,
            input_modes=trace.input_modes,
            paths=trace.paths,
            step_count=trace.step_count,
            first_position=trace.first_position,
            expert_bytes=trace.expert_bytes,
            fresh_empty_source=False,
        )
        with self.assertRaisesRegex(analyzer.AnalysisError,
                                    "fresh-empty-source provenance"):
            analyzer.require_reconstructible_targets(warm, 18, (19,))


HOST_GOLDEN_VALUE = os.environ.get("MOONSHINE_CACHE_ANALYZER_GOLDEN_RUN")
HOST_GOLDEN = Path(HOST_GOLDEN_VALUE) if HOST_GOLDEN_VALUE else None


@unittest.skipUnless(
    HOST_GOLDEN is not None and HOST_GOLDEN.is_dir(),
    "set MOONSHINE_CACHE_ANALYZER_GOLDEN_RUN to an accepted trace directory",
)
class AcceptedHostTraceGoldenTest(unittest.TestCase):
    def test_capacity30_and_exact_fixed_total_goldens(self) -> None:
        # Offline CSV replay only: this test never starts a server/GPU workload.
        assert HOST_GOLDEN is not None
        trace = analyzer.load_trace(HOST_GOLDEN, 1, 30, fresh_empty_source=True)
        gate = analyzer.source_replay_gate(trace, 30)
        self.assertEqual(gate.total.hits, 85948)
        self.assertEqual(gate.mask_mismatches, 0)
        curves = analyzer.compute_lru_curves(trace, 30, 16, 64)
        vector, metrics = analyzer.exact_fixed_total_dp(
            curves, total=2760, minimum=16, maximum=64)
        self.assertEqual(sum(vector), 2760)
        self.assertGreaterEqual(min(vector), 16)
        self.assertLessEqual(max(vector), 64)
        self.assertEqual(metrics.hits, 87390)
        uniform = analyzer.sum_metrics(curves[30])
        self.assertEqual(uniform.hits, 85948)


if __name__ == "__main__":
    unittest.main(verbosity=2)
