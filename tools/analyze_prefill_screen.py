#!/usr/bin/env python3
"""Validate one routed-prefill warm-up/noise-floor/ABBA screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import tempfile

ROLES = (
    "warmup",
    "noise_1", "noise_2", "noise_3", "noise_4",
    "baseline_a", "candidate_a", "candidate_b", "baseline_b",
)
NOISE_ROLES = ROLES[:5]
METRICS = {
    "expert_pipeline_seconds",
    "routed_stream_seconds",
    "read_wait_seconds",
}
REQUIRED = {
    "capture", "layer", "tokens", "unique_experts", "read_requests",
    "physical_read_bytes", "attention_seconds", "router_seconds",
    "routed_stream_seconds", "read_wait_seconds", "submit_seconds",
    "index_seconds", "expert_pipeline_seconds", "moe_tail_seconds",
    "default_span_ms", "expert_span_ms", "shared_span_ms",
    "mzg2_decoder_gpu_ms", "ring_depth0_seconds", "ring_depth1_seconds",
    "ring_depth2_seconds", "ring_transitions", "ring_max_depth",
    "mzg2_launches", "integrity_clears", "index_h2d_copies", "gathers",
    "expert_gemms", "scatters", "stream_synchronizes", "event_waits",
}
INTEGER_FIELDS = {
    "capture", "layer", "tokens", "unique_experts", "read_requests",
    "physical_read_bytes", "ring_transitions", "ring_max_depth",
    "mzg2_launches", "integrity_clears", "index_h2d_copies", "gathers",
    "expert_gemms", "scatters", "stream_synchronizes", "event_waits",
}


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class Arm:
    path: str
    capture: int
    layers: int
    tokens: int
    target_seconds: float
    phase_seconds: float
    ring_idle_seconds: float
    ring_depth_seconds: tuple[float, float, float]
    read_requests: int
    physical_read_bytes: int
    command_counts: dict[str, int]
    identity: tuple[tuple[int, int, int, int, int], ...]


def _percentage_spread(values: list[float]) -> float:
    average = mean(values)
    return 0.0 if average == 0.0 else 100.0 * (max(values) - min(values)) / average


def _read_arm(path: Path, metric: str) -> Arm:
    try:
        stream = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise AnalysisError(f"opening {path} failed: {exc}") from exc
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != REQUIRED:
            raise AnalysisError(f"{path}: unexpected prefill CSV header")
        rows = list(reader)
    if len(rows) != 92:
        raise AnalysisError(f"{path}: expected 92 routed layers, got {len(rows)}")
    parsed: list[dict[str, int | float]] = []
    for line, row in enumerate(rows, 2):
        values: dict[str, int | float] = {}
        try:
            for field in REQUIRED:
                values[field] = int(row[field]) if field in INTEGER_FIELDS else float(row[field])
        except (TypeError, ValueError) as exc:
            raise AnalysisError(f"{path}:{line}: invalid numeric field") from exc
        parsed.append(values)
    captures = {int(row["capture"]) for row in parsed}
    layers = [int(row["layer"]) for row in parsed]
    tokens = {int(row["tokens"]) for row in parsed}
    if len(captures) != 1 or layers != list(range(1, 93)) or len(tokens) != 1:
        raise AnalysisError(f"{path}: capture/layer/token ledger is invalid")
    for row in parsed:
        depths = sum(float(row[f"ring_depth{depth}_seconds"]) for depth in range(3))
        if depths < 0.0 or int(row["ring_max_depth"]) > 2:
            raise AnalysisError(f"{path}: invalid ring occupancy ledger")
    commands = (
        "mzg2_launches", "integrity_clears", "index_h2d_copies", "gathers",
        "expert_gemms", "scatters", "stream_synchronizes", "event_waits",
    )
    depth = tuple(sum(float(row[f"ring_depth{item}_seconds"]) for row in parsed)
                  for item in range(3))
    return Arm(
        str(path), captures.pop(), len(parsed), tokens.pop(),
        sum(float(row[metric]) for row in parsed),
        sum(float(row["attention_seconds"]) + float(row["router_seconds"]) +
            float(row["routed_stream_seconds"]) + float(row["moe_tail_seconds"])
            for row in parsed),
        depth[0], depth,
        sum(int(row["read_requests"]) for row in parsed),
        sum(int(row["physical_read_bytes"]) for row in parsed),
        {name: sum(int(row[name]) for row in parsed) for name in commands},
        tuple((int(row["layer"]), int(row["tokens"]),
               int(row["unique_experts"]), int(row["read_requests"]),
               int(row["physical_read_bytes"]))
              for row in parsed),
    )


def _validate_identity(arms: dict[str, Arm]) -> None:
    reference = arms["noise_1"].identity
    for role, arm in arms.items():
        if (arm.identity != reference or
            arm.read_requests != arms["noise_1"].read_requests or
            arm.physical_read_bytes !=
                arms["noise_1"].physical_read_bytes):
            raise AnalysisError(
                f"{role}: route-union/I/O identity differs")


def analyze_noise(
        paths: list[Path],
        metric: str,
        gate_pct: float) -> dict[str, object]:
    if len(paths) != len(NOISE_ROLES):
        raise AnalysisError(
            f"expected {len(NOISE_ROLES)} ordered noise-floor files")
    if metric not in METRICS or not (gate_pct > 0.0):
        raise AnalysisError("invalid metric or promotion gate")
    arms = {
        role: _read_arm(path, metric)
        for role, path in zip(NOISE_ROLES, paths)
    }
    _validate_identity(arms)
    noise = [arms[f"noise_{index}"].target_seconds
             for index in range(1, 5)]
    phase_noise = [arms[f"noise_{index}"].phase_seconds
                   for index in range(1, 5)]
    noise_pct = _percentage_spread(noise)
    phase_noise_pct = _percentage_spread(phase_noise)
    resolvable = gate_pct >= 3.0 * noise_pct
    average_target = mean(noise)
    average_idle = mean([
        arms[f"noise_{index}"].ring_idle_seconds
        for index in range(1, 5)
    ])
    idle_ceiling = (
        0.0 if average_target == 0.0
        else 100.0 * average_idle / average_target
    )
    return {
        "schema": "moonshine-prefill-screen-noise-v1",
        "order": list(NOISE_ROLES),
        "metric": metric,
        "gate_pct": gate_pct,
        "noise_floor": {
            "target_values_seconds": noise,
            "target_mean_seconds": average_target,
            "target_spread_pct": noise_pct,
            "phase_spread_pct": phase_noise_pct,
            "three_sigma_style_floor_pct": 3.0 * noise_pct,
            "gate_resolvable": resolvable,
        },
        "ring_idle_ceiling_pct": idle_ceiling,
        "command_counts": {
            role: arms[role].command_counts
            for role in NOISE_ROLES
        },
        "decision": (
            "READY_FOR_ABBA" if resolvable else "UNRESOLVABLE"
        ),
    }

def analyze(paths: list[Path], metric: str, gate_pct: float) -> dict[str, object]:
    if len(paths) != len(ROLES):
        raise AnalysisError(f"expected {len(ROLES)} ordered arm files")
    if metric not in METRICS or not (gate_pct > 0.0):
        raise AnalysisError("invalid metric or promotion gate")
    arms = {role: _read_arm(path, metric) for role, path in zip(ROLES, paths)}
    _validate_identity(arms)
    noise = [arms[f"noise_{index}"].target_seconds for index in range(1, 5)]
    phase_noise = [arms[f"noise_{index}"].phase_seconds for index in range(1, 5)]
    noise_pct = _percentage_spread(noise)
    phase_noise_pct = _percentage_spread(phase_noise)
    baseline = mean([arms["baseline_a"].target_seconds, arms["baseline_b"].target_seconds])
    candidate = mean([arms["candidate_a"].target_seconds, arms["candidate_b"].target_seconds])
    baseline_phase = mean([arms["baseline_a"].phase_seconds, arms["baseline_b"].phase_seconds])
    candidate_phase = mean([arms["candidate_a"].phase_seconds, arms["candidate_b"].phase_seconds])
    improvement = 100.0 * (baseline - candidate) / baseline
    phase_change = 100.0 * (candidate_phase - baseline_phase) / baseline_phase
    resolvable = gate_pct >= 3.0 * noise_pct
    no_regression = phase_change <= 3.0 * phase_noise_pct
    baseline_idle = mean([arms["baseline_a"].ring_idle_seconds,
                          arms["baseline_b"].ring_idle_seconds])
    idle_ceiling = 0.0 if baseline == 0.0 else 100.0 * baseline_idle / baseline
    decision = "PASS" if resolvable and improvement >= gate_pct and no_regression else (
        "UNRESOLVABLE" if not resolvable else "NO_GO")
    return {
        "schema": "moonshine-prefill-screen-analysis-v1",
        "order": list(ROLES),
        "metric": metric,
        "gate_pct": gate_pct,
        "noise_floor": {
            "target_spread_pct": noise_pct,
            "phase_spread_pct": phase_noise_pct,
            "three_sigma_style_floor_pct": 3.0 * noise_pct,
            "gate_resolvable": resolvable,
        },
        "abba": {
            "baseline_mean_seconds": baseline,
            "candidate_mean_seconds": candidate,
            "target_improvement_pct": improvement,
            "baseline_phase_mean_seconds": baseline_phase,
            "candidate_phase_mean_seconds": candidate_phase,
            "phase_change_pct": phase_change,
            "phase_no_regression": no_regression,
        },
        "baseline_ring_idle_ceiling_pct": idle_ceiling,
        "command_counts": {
            role: arms[role].command_counts for role in
            ("baseline_a", "candidate_a", "candidate_b", "baseline_b")
        },
        "decision": decision,
    }


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=sorted(METRICS), required=True)
    parser.add_argument("--gate-pct", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument("arms", nargs="+", type=Path, metavar="CSV")
    args = parser.parse_args()
    result = (
        analyze_noise(args.arms, args.metric, args.gate_pct)
        if args.noise_only
        else analyze(args.arms, args.metric, args.gate_pct)
    )
    _atomic_json(args.output, result)
    print(f"Moonshine prefill screen: {result['decision']}")


if __name__ == "__main__":
    main()
