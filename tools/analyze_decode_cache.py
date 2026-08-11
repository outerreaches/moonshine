#!/usr/bin/env python3
"""Deterministic, offline analysis of Moonshine decode-cache diagnostics.

This stdlib-only tool deliberately models the runtime's two-phase
batch cache contract: every hit is looked up in the frozen pre-batch state,
then the selected experts are touched in gate-rank order on a pending LRU, and
only commit makes that pending state live.

The report is a trace replay/counterfactual, not a workload forecast.  In
particular, full-trace pinned results are explicitly non-promotable oracles,
and prefix-trained pinned results are scored only on a cold suffix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

LAYERS = 92
TOP_K = 16
EXPERTS = 896
SCHEMA_VERSION = "moonshine-decode-cache-analysis-v1"

CACHE_HEADER = ("capture", "layer", "lru_rank", "expert_id")
ROUTES_HEADER = (
    "capture", "step", "position", "layer", "observed_hit_mask",
    *(f"expert_{rank}" for rank in range(TOP_K)),
)
LEDGER_PREFIX = (
    "capture", "scope", "layer", "steps", "accesses", "hits", "misses",
    "read_requests", "logical_expert_bytes", "physical_read_bytes",
    "wait_calls", "completions", "max_inflight", "pre_moe_seconds",
    "io_wait_seconds", "expert_pipeline_seconds", "expert_sync_seconds",
    "shared_sync_seconds",
)
# The host-local accepted trace predates the reviewed accounting label.  These
# are the only two accepted schemas; accepting them is versioned, not lenient.
LEDGER_HEADERS = {
    (*LEDGER_PREFIX, "wall_seconds"): "legacy-v1-wall-layer-step-summary",
    (*LEDGER_PREFIX, "host_interval_seconds"): "reviewed-v2-host-interval",
}
INTEGER_LEDGER_FIELDS = (
    "capture", "layer", "steps", "accesses", "hits", "misses",
    "read_requests", "logical_expert_bytes", "physical_read_bytes",
    "wait_calls", "completions", "max_inflight",
)
ADDITIVE_INTEGER_FIELDS = (
    "accesses", "hits", "misses", "read_requests", "logical_expert_bytes",
    "physical_read_bytes", "wait_calls", "completions",
)
ADDITIVE_FLOAT_FIELDS = (
    "pre_moe_seconds", "io_wait_seconds", "expert_pipeline_seconds",
    "expert_sync_seconds", "shared_sync_seconds",
)
_UINT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class AnalysisError(Exception):
    """A deterministic validation or analysis failure."""


@dataclass(frozen=True)
class Batch:
    step: int
    position: int
    layer: int
    observed_hit_mask: int
    experts: tuple[int, ...]


@dataclass(frozen=True)
class Entry:
    expert: int
    slot: int


@dataclass(frozen=True)
class Access:
    expert: int
    hit: bool
    admit: bool
    source_slot: int | None
    destination_slot: int | None


@dataclass(frozen=True)
class Plan:
    accesses: tuple[Access, ...]
    entries: tuple[Entry, ...]
    hits: int
    misses: int
    admissions: int
    evictions: int

    @property
    def hit_mask(self) -> int:
        return sum((1 << rank) for rank, item in enumerate(self.accesses)
                   if item.hit)


@dataclass
class Metrics:
    batches: int = 0
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    admissions: int = 0
    evictions: int = 0

    def add_plan(self, plan: Plan) -> None:
        self.batches += 1
        self.accesses += len(plan.accesses)
        self.hits += plan.hits
        self.misses += plan.misses
        self.admissions += plan.admissions
        self.evictions += plan.evictions

    def add(self, other: "Metrics") -> None:
        self.batches += other.batches
        self.accesses += other.accesses
        self.hits += other.hits
        self.misses += other.misses
        self.admissions += other.admissions
        self.evictions += other.evictions

    def as_dict(self) -> dict[str, int]:
        return {
            "accesses": self.accesses,
            "admissions": self.admissions,
            "batches": self.batches,
            "evictions": self.evictions,
            "hits": self.hits,
            "misses": self.misses,
        }


@dataclass
class LayerState:
    """Physical-slot-aware exact port of k3_expert_cache plan/commit."""

    capacity: int
    entries: list[Entry] = field(default_factory=list)  # oldest -> newest
    pending: Plan | None = None
    metrics: Metrics = field(default_factory=Metrics)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise AnalysisError("cache capacity must be positive")
        if len(self.entries) > self.capacity:
            raise AnalysisError("initial cache occupancy exceeds capacity")
        experts = [entry.expert for entry in self.entries]
        slots = [entry.slot for entry in self.entries]
        if len(set(experts)) != len(experts):
            raise AnalysisError("initial cache has duplicate experts")
        if len(set(slots)) != len(slots):
            raise AnalysisError("initial cache has duplicate physical slots")
        if any(not 0 <= entry.expert < EXPERTS for entry in self.entries):
            raise AnalysisError("initial cache expert is outside 0..895")
        if any(not 0 <= entry.slot < self.capacity for entry in self.entries):
            raise AnalysisError("initial cache slot is outside capacity")

    @classmethod
    def from_experts(cls, capacity: int,
                     experts: Sequence[int]) -> "LayerState":
        if len(experts) > capacity:
            experts = experts[-capacity:]
        return cls(capacity, [Entry(expert, slot)
                              for slot, expert in enumerate(experts)])

    def plan(self, expert_ids: Sequence[int]) -> Plan:
        if self.pending is not None:
            raise AnalysisError("layer already has a pending cache plan")
        if not expert_ids:
            raise AnalysisError("cache batch must not be empty")
        if len(set(expert_ids)) != len(expert_ids):
            raise AnalysisError("cache batch contains duplicate expert IDs")
        if any(not 0 <= expert < EXPERTS for expert in expert_ids):
            raise AnalysisError("cache batch expert is outside 0..895")

        current = tuple(self.entries)
        current_by_expert = {entry.expert: entry for entry in current}
        # Hit lookup is intentionally frozen before any rank mutates `next_`.
        frozen_hits = [expert in current_by_expert for expert in expert_ids]
        next_ = list(current)

        for expert in expert_ids:
            next_index = next(
                (i for i, entry in enumerate(next_) if entry.expert == expert),
                None,
            )
            slot: int | None = None
            if next_index is not None:
                slot = next_[next_index].slot
                del next_[next_index]
            elif len(next_) == self.capacity:
                del next_[0]
            next_.append(Entry(expert, -1 if slot is None else slot))

        used = {entry.slot for entry in next_ if entry.slot >= 0}
        # Match the C same-batch fall-out/re-entry slot preservation pass.
        fixed: list[Entry] = []
        for entry in next_:
            if entry.slot >= 0:
                fixed.append(entry)
                continue
            old = current_by_expert.get(entry.expert)
            if old is not None and old.slot not in used:
                fixed.append(Entry(entry.expert, old.slot))
                used.add(old.slot)
            else:
                fixed.append(entry)
        next_ = fixed

        fixed = []
        for entry in next_:
            if entry.slot >= 0:
                fixed.append(entry)
                continue
            slot = next((candidate for candidate in range(self.capacity)
                         if candidate not in used), None)
            if slot is None:
                raise AnalysisError("cache plan has no free physical slot")
            fixed.append(Entry(entry.expert, slot))
            used.add(slot)
        next_ = fixed
        next_by_expert = {entry.expert: entry for entry in next_}

        accesses: list[Access] = []
        for expert, hit in zip(expert_ids, frozen_hits):
            source = current_by_expert[expert].slot if hit else None
            destination = None
            admit = False
            if not hit and expert in next_by_expert:
                admit = True
                destination = next_by_expert[expert].slot
            accesses.append(Access(expert, hit, admit, source, destination))

        current_experts = set(current_by_expert)
        final_experts = set(next_by_expert)
        hits = sum(frozen_hits)
        plan = Plan(
            tuple(accesses), tuple(next_), hits, len(expert_ids) - hits,
            sum(item.admit for item in accesses),
            len(current_experts - final_experts),
        )
        self.pending = plan
        return plan

    def commit(self) -> Plan:
        if self.pending is None:
            raise AnalysisError("layer has no pending cache plan")
        plan = self.pending
        self.entries = list(plan.entries)
        self.metrics.add_plan(plan)
        self.pending = None
        return plan

    def abort(self) -> None:
        self.pending = None


@dataclass(frozen=True)
class Trace:
    capture: int
    batches: tuple[Batch, ...]
    initial_lru: tuple[tuple[int, ...], ...]
    ledger_summary: Mapping[str, object]
    ledger_schema: str
    input_sha256: Mapping[str, str]
    input_modes: Mapping[str, int]
    paths: Mapping[str, str]
    step_count: int
    first_position: int
    expert_bytes: int | None
    fresh_empty_source: bool

    @property
    def accesses(self) -> int:
        return len(self.batches) * TOP_K

    @property
    def observed_hits(self) -> int:
        return sum(batch.observed_hit_mask.bit_count()
                   for batch in self.batches)

    def by_layer(self) -> tuple[tuple[Batch, ...], ...]:
        result: list[list[Batch]] = [[] for _ in range(LAYERS)]
        for batch in self.batches:
            result[batch.layer - 1].append(batch)
        return tuple(tuple(items) for items in result)


@dataclass(frozen=True)
class ReplayResult:
    total: Metrics
    per_layer: tuple[Metrics, ...]
    mask_mismatches: int
    first_mismatch: Mapping[str, int] | None


def _uint(value: str, field_name: str, context: str) -> int:
    if not _UINT_RE.fullmatch(value):
        raise AnalysisError(f"{context}: invalid unsigned integer in {field_name}")
    return int(value)


def _nonnegative_float(value: str, field_name: str, context: str) -> float:
    if value != value.strip() or not value:
        raise AnalysisError(f"{context}: invalid float in {field_name}")
    try:
        result = float(value)
    except ValueError as exc:
        raise AnalysisError(
            f"{context}: invalid float in {field_name}") from exc
    if not math.isfinite(result) or result < 0.0:
        raise AnalysisError(f"{context}: invalid float in {field_name}")
    return result


def _read_csv(
        path: Path, allowed_headers: Iterable[tuple[str, ...]], label: str
        ) -> tuple[tuple[str, ...], list[dict[str, str]], str, int]:
    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AnalysisError(
                f"{label}: input is not a real regular file: {path}")
        raw = os.fdopen(descriptor, "rb")
        descriptor = -1  # `raw` owns and closes it from this point.
        with raw:
            payload = raw.read()
    except OSError as exc:
        raise AnalysisError(f"{label}: cannot read input: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        text = payload.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise AnalysisError(f"{label}: cannot parse CSV: {exc}") from exc
    if not rows:
        raise AnalysisError(f"{label}: CSV has no header")
    header = tuple(rows[0])
    allowed = set(allowed_headers)
    if header not in allowed:
        raise AnalysisError(f"{label}: unexpected schema: {','.join(header)}")
    result: list[dict[str, str]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise AnalysisError(
                f"{label} row {row_number}: expected {len(header)} fields, "
                f"found {len(row)}")
        if any("\x00" in value for value in row):
            raise AnalysisError(f"{label} row {row_number}: NUL byte")
        result.append(dict(zip(header, row)))
    return (
        header, result, hashlib.sha256(payload).hexdigest(),
        stat.S_IMODE(metadata.st_mode),
    )


def _capture_blocks(captures: Sequence[int], label: str) -> None:
    seen: set[int] = set()
    previous: int | None = None
    for capture in captures:
        if capture != previous:
            if capture in seen:
                raise AnalysisError(
                    f"{label}: capture {capture} appears in multiple blocks")
            seen.add(capture)
            previous = capture


def load_trace(run_dir: Path | str, capture: int, observed_capacity: int,
               cache_path: Path | str | None = None,
               ledger_path: Path | str | None = None,
               routes_path: Path | str | None = None,
               fresh_empty_source: bool = False) -> Trace:
    if capture <= 0:
        raise AnalysisError("--capture must be a positive integer")
    if observed_capacity <= 0:
        raise AnalysisError("--observed-capacity must be positive")
    run_dir = Path(run_dir)
    paths = {
        "cache": Path(cache_path) if cache_path else run_dir / "decode.cache.csv",
        "ledger": Path(ledger_path) if ledger_path else run_dir / "decode.ledger.csv",
        "routes": Path(routes_path) if routes_path else run_dir / "decode.routes.csv",
    }

    cache_header, cache_rows, cache_sha256, cache_mode = _read_csv(
        paths["cache"], (CACHE_HEADER,), "decode cache")
    ledger_header, ledger_rows, ledger_sha256, ledger_mode = _read_csv(
        paths["ledger"], LEDGER_HEADERS, "decode ledger")
    _, route_rows, routes_sha256, routes_mode = _read_csv(
        paths["routes"], (ROUTES_HEADER,), "decode routes")
    del cache_header  # exact header was already checked

    # Parse captures in every row before filtering, so malformed other capture
    # IDs cannot hide behind --capture.
    cache_captures = [
        _uint(row["capture"], "capture", f"decode cache row {i}")
        for i, row in enumerate(cache_rows, start=2)
    ]
    ledger_captures = [
        _uint(row["capture"], "capture", f"decode ledger row {i}")
        for i, row in enumerate(ledger_rows, start=2)
    ]
    route_captures = [
        _uint(row["capture"], "capture", f"decode routes row {i}")
        for i, row in enumerate(route_rows, start=2)
    ]
    _capture_blocks(cache_captures, "decode cache")
    _capture_blocks(ledger_captures, "decode ledger")
    _capture_blocks(route_captures, "decode routes")
    if cache_captures and set(cache_captures) != {capture}:
        raise AnalysisError(
            f"decode cache: input must contain exactly requested capture {capture}")
    if set(ledger_captures) != {capture}:
        raise AnalysisError(
            f"decode ledger: input must contain exactly requested capture {capture}")
    if set(route_captures) != {capture}:
        raise AnalysisError(
            f"decode routes: input must contain exactly requested capture {capture}")

    selected_cache = [row for row, cap in zip(cache_rows, cache_captures)
                      if cap == capture]
    selected_ledger = [row for row, cap in zip(ledger_rows, ledger_captures)
                       if cap == capture]
    selected_routes = [row for row, cap in zip(route_rows, route_captures)
                       if cap == capture]
    if not selected_ledger:
        raise AnalysisError(f"decode ledger: capture {capture} is absent")
    if not selected_routes:
        raise AnalysisError(f"decode routes: capture {capture} is absent")
    # A globally header-only cache is the documented representation of an
    # empty snapshot.  If other captures have rows, absence is ambiguous.
    if cache_rows and not selected_cache:
        raise AnalysisError(f"decode cache: capture {capture} is absent")

    initial: list[list[int]] = [[] for _ in range(LAYERS)]
    last_key: tuple[int, int] | None = None
    for offset, row in enumerate(selected_cache, start=1):
        context = f"decode cache capture {capture} row {offset}"
        layer = _uint(row["layer"], "layer", context)
        rank = _uint(row["lru_rank"], "lru_rank", context)
        expert = _uint(row["expert_id"], "expert_id", context)
        if not 1 <= layer <= LAYERS:
            raise AnalysisError(f"{context}: layer is outside 1..92")
        if not 0 <= expert < EXPERTS:
            raise AnalysisError(f"{context}: expert is outside 0..895")
        key = (layer, rank)
        if last_key is not None and key <= last_key:
            raise AnalysisError(
                f"{context}: rows are not ordered by layer/LRU rank")
        last_key = key
        values = initial[layer - 1]
        if rank != len(values):
            raise AnalysisError(
                f"{context}: LRU ranks are not contiguous from zero")
        if expert in values:
            raise AnalysisError(f"{context}: duplicate expert in layer")
        values.append(expert)
        if len(values) > observed_capacity:
            raise AnalysisError(
                f"{context}: occupancy exceeds --observed-capacity")
    if fresh_empty_source and any(initial):
        raise AnalysisError(
            "--fresh-empty-source requires a completely empty cache snapshot")

    batches: list[Batch] = []
    current_position: int | None = None
    for index, row in enumerate(selected_routes):
        context = f"decode routes capture {capture} row {index + 1}"
        step = _uint(row["step"], "step", context)
        position = _uint(row["position"], "position", context)
        layer = _uint(row["layer"], "layer", context)
        mask = _uint(row["observed_hit_mask"], "observed_hit_mask", context)
        expected_step, expected_layer0 = divmod(index, LAYERS)
        if step != expected_step or layer != expected_layer0 + 1:
            raise AnalysisError(
                f"{context}: rows must be contiguous step-major/layer-major")
        if mask > 0xFFFF:
            raise AnalysisError(f"{context}: hit mask exceeds 16 bits")
        if layer == 1:
            if current_position is not None and position != current_position + 1:
                raise AnalysisError(
                    f"{context}: token positions are not contiguous")
            current_position = position
        elif position != current_position:
            raise AnalysisError(
                f"{context}: layers in a step have different positions")
        experts = tuple(
            _uint(row[f"expert_{rank}"], f"expert_{rank}", context)
            for rank in range(TOP_K)
        )
        if any(expert >= EXPERTS for expert in experts):
            raise AnalysisError(f"{context}: expert is outside 0..895")
        if len(set(experts)) != TOP_K:
            raise AnalysisError(
                f"{context}: selected expert IDs are not 16 unique values")
        batches.append(Batch(step, position, layer, mask, experts))
    if len(batches) % LAYERS:
        raise AnalysisError("decode routes: final step is incomplete")
    step_count = len(batches) // LAYERS
    if step_count == 0:
        raise AnalysisError("decode routes: capture is empty")

    if len(selected_ledger) != LAYERS + 1:
        raise AnalysisError(
            f"decode ledger: capture must have 1 summary and {LAYERS} layers")
    parsed_ledger: list[dict[str, object]] = []
    float_tail = ledger_header[-1]
    for index, row in enumerate(selected_ledger):
        context = f"decode ledger capture {capture} row {index + 1}"
        parsed: dict[str, object] = {"scope": row["scope"]}
        for name in INTEGER_LEDGER_FIELDS:
            parsed[name] = _uint(row[name], name, context)
        for name in (*ADDITIVE_FLOAT_FIELDS, float_tail):
            parsed[name] = _nonnegative_float(row[name], name, context)
        parsed_ledger.append(parsed)
    summary = parsed_ledger[0]
    if summary["scope"] != "summary" or summary["layer"] != 0:
        raise AnalysisError("decode ledger: first selected row must be summary/layer 0")
    layer_rows = parsed_ledger[1:]
    for layer, row in enumerate(layer_rows, start=1):
        if row["scope"] != "layer" or row["layer"] != layer:
            raise AnalysisError(
                "decode ledger: layer rows must be ordered exactly 1..92")
        if row["steps"] != step_count:
            raise AnalysisError(
                f"decode ledger: layer {layer} steps differ from routes")
        if row["accesses"] != step_count * TOP_K:
            raise AnalysisError(
                f"decode ledger: layer {layer} accesses differ from routes")

    schema_name = LEDGER_HEADERS[ledger_header]
    expected_summary_steps = (
        len(batches) if schema_name.startswith("legacy-v1") else step_count)
    if summary["steps"] != expected_summary_steps:
        raise AnalysisError(
            "decode ledger: summary step semantics do not match its schema")
    if sum(int(row["steps"]) for row in layer_rows) != len(batches):
        raise AnalysisError("decode ledger: layer steps do not sum to route rows")

    for name in ADDITIVE_INTEGER_FIELDS:
        if int(summary[name]) != sum(int(row[name]) for row in layer_rows):
            raise AnalysisError(
                f"decode ledger: summary {name} does not equal layer sum")
    if int(summary["max_inflight"]) != max(
            int(row["max_inflight"]) for row in layer_rows):
        raise AnalysisError(
            "decode ledger: summary max_inflight does not equal layer maximum")
    for name in (*ADDITIVE_FLOAT_FIELDS, float_tail):
        actual = float(summary[name])
        expected = math.fsum(float(row[name]) for row in layer_rows)
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=5e-8):
            raise AnalysisError(
                f"decode ledger: summary {name} does not equal layer sum")

    observed_by_layer = [0] * LAYERS
    for batch in batches:
        observed_by_layer[batch.layer - 1] += batch.observed_hit_mask.bit_count()
    for layer, row in enumerate(layer_rows, start=1):
        accesses = int(row["accesses"])
        hits = int(row["hits"])
        misses = int(row["misses"])
        if hits != observed_by_layer[layer - 1]:
            raise AnalysisError(
                f"decode ledger: layer {layer} hits differ from route masks")
        if hits + misses != accesses:
            raise AnalysisError(
                f"decode ledger: layer {layer} hits + misses != accesses")
        reads = int(row["read_requests"])
        completions = int(row["completions"])
        waits = int(row["wait_calls"])
        maximum = int(row["max_inflight"])
        if reads != misses:
            raise AnalysisError(
                f"decode ledger: layer {layer} reads != misses")
        if completions != reads or waits > completions:
            raise AnalysisError(
                f"decode ledger: layer {layer} completion accounting fails")
        if not 0 <= maximum <= 2 or ((reads == 0) != (maximum == 0)):
            raise AnalysisError(
                f"decode ledger: layer {layer} max-inflight accounting fails")
        physical = int(row["physical_read_bytes"])
        logical_layer = int(row["logical_expert_bytes"])
        if physical < logical_layer:
            raise AnalysisError(
                f"decode ledger: layer {layer} physical bytes < logical bytes")
        if reads == 0 and physical != 0:
            raise AnalysisError(
                f"decode ledger: layer {layer} has physical bytes with no reads")
    if int(summary["accesses"]) != len(batches) * TOP_K:
        raise AnalysisError("decode ledger: summary accesses differ from routes")
    if int(summary["hits"]) != sum(observed_by_layer):
        raise AnalysisError("decode ledger: summary hits differ from route masks")
    if int(summary["hits"]) + int(summary["misses"]) != int(summary["accesses"]):
        raise AnalysisError("decode ledger: summary hit/miss accounting fails")
    if int(summary["read_requests"]) != int(summary["misses"]):
        raise AnalysisError("decode ledger: summary reads differ from misses")
    if int(summary["completions"]) != int(summary["read_requests"]) or             int(summary["wait_calls"]) > int(summary["completions"]):
        raise AnalysisError("decode ledger: summary completion accounting fails")
    summary_maximum = int(summary["max_inflight"])
    summary_reads = int(summary["read_requests"])
    if (not 0 <= summary_maximum <= 2 or
            ((summary_reads == 0) != (summary_maximum == 0))):
        raise AnalysisError("decode ledger: summary max-inflight is invalid")
    summary_physical = int(summary["physical_read_bytes"])
    if summary_physical < int(summary["logical_expert_bytes"]):
        raise AnalysisError("decode ledger: physical reads are below logical reads")
    if summary_reads == 0 and summary_physical != 0:
        raise AnalysisError("decode ledger: summary has physical bytes with no reads")

    misses = int(summary["misses"])
    logical = int(summary["logical_expert_bytes"])
    expert_bytes: int | None
    if misses:
        if logical == 0 or logical % misses:
            raise AnalysisError(
                "decode ledger: logical bytes must imply a positive integer "
                "expert size per miss")
        expert_bytes = logical // misses
        for layer, row in enumerate(layer_rows, start=1):
            if int(row["logical_expert_bytes"]) != int(row["misses"]) * expert_bytes:
                raise AnalysisError(
                    f"decode ledger: layer {layer} logical bytes are inconsistent")
    else:
        expert_bytes = None if logical == 0 else 0
        if expert_bytes == 0:
            raise AnalysisError("decode ledger: nonzero logical bytes with no misses")

    return Trace(
        capture=capture,
        batches=tuple(batches),
        initial_lru=tuple(tuple(values) for values in initial),
        ledger_summary=dict(summary),
        ledger_schema=schema_name,
        input_sha256={
            "cache": cache_sha256,
            "ledger": ledger_sha256,
            "routes": routes_sha256,
        },
        input_modes={
            "cache": cache_mode,
            "ledger": ledger_mode,
            "routes": routes_mode,
        },
        paths={name: path.name for name, path in paths.items()},
        step_count=step_count,
        first_position=batches[0].position,
        expert_bytes=expert_bytes,
        fresh_empty_source=fresh_empty_source,
    )


def _project_initial(trace: Trace, layer: int, capacity: int,
                     observed_capacity: int) -> tuple[int, ...]:
    values = trace.initial_lru[layer - 1]
    if capacity <= observed_capacity:
        return values[-capacity:]
    return values


def require_reconstructible_targets(trace: Trace, observed_capacity: int,
                                    capacities: Iterable[int]) -> None:
    capacities = tuple(capacities)
    if any(capacity <= 0 for capacity in capacities):
        raise AnalysisError("all target capacities must be positive")
    if max(capacities, default=observed_capacity) > observed_capacity:
        if not trace.fresh_empty_source:
            raise AnalysisError(
                "target capacity exceeds the source without explicit "
                "--fresh-empty-source provenance; prior evictions or slot "
                "invalidations are unreconstructible")
        if any(trace.initial_lru):
            raise AnalysisError(
                "target capacity exceeds the source but the asserted fresh "
                "snapshot is not empty")


def replay_lru(trace: Trace, capacities: int | Sequence[int],
               observed_capacity: int, compare_masks: bool = False,
               batches: Sequence[Batch] | None = None,
               cold: bool = False) -> ReplayResult:
    if isinstance(capacities, int):
        vector = (capacities,) * LAYERS
    else:
        vector = tuple(capacities)
        if len(vector) != LAYERS:
            raise AnalysisError("capacity vector must contain exactly 92 values")
    require_reconstructible_targets(trace, observed_capacity, vector)
    states: list[LayerState] = []
    for layer, capacity in enumerate(vector, start=1):
        seed = () if cold else _project_initial(
            trace, layer, capacity, observed_capacity)
        states.append(LayerState.from_experts(capacity, seed))
    mismatch_count = 0
    first_mismatch: dict[str, int] | None = None
    selected_batches = trace.batches if batches is None else batches
    for batch in selected_batches:
        state = states[batch.layer - 1]
        plan = state.plan(batch.experts)
        if compare_masks and plan.hit_mask != batch.observed_hit_mask:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "actual_mask": plan.hit_mask,
                    "expected_mask": batch.observed_hit_mask,
                    "layer": batch.layer,
                    "position": batch.position,
                    "step": batch.step,
                }
        state.commit()
    total = Metrics()
    for state in states:
        total.add(state.metrics)
    return ReplayResult(
        total, tuple(state.metrics for state in states), mismatch_count,
        first_mismatch)


def source_replay_gate(trace: Trace, observed_capacity: int) -> ReplayResult:
    result = replay_lru(trace, observed_capacity, observed_capacity,
                        compare_masks=True)
    ledger_hits = int(trace.ledger_summary["hits"])
    ledger_accesses = int(trace.ledger_summary["accesses"])
    # A mask mismatch is the stronger semantic failure; report it before the
    # consequent aggregate-counter difference.
    if result.mask_mismatches:
        detail = result.first_mismatch or {}
        raise AnalysisError(
            "source-capacity replay gate failed: "
            f"{result.mask_mismatches} frozen pre-batch hit-mask mismatch(es); "
            f"first={json.dumps(detail, sort_keys=True)}")
    if result.total.hits != ledger_hits or result.total.accesses != ledger_accesses:
        raise AnalysisError(
            "source-capacity LRU replay counters differ from the ledger")
    return result


def compute_lru_curves(trace: Trace, observed_capacity: int,
                       minimum: int, maximum: int
                       ) -> dict[int, tuple[Metrics, ...]]:
    if minimum <= 0 or maximum < minimum:
        raise AnalysisError("invalid marginal capacity range")
    require_reconstructible_targets(
        trace, observed_capacity, range(minimum, maximum + 1))
    by_layer = trace.by_layer()
    curves: dict[int, tuple[Metrics, ...]] = {}
    for capacity in range(minimum, maximum + 1):
        layer_metrics: list[Metrics] = []
        for layer in range(1, LAYERS + 1):
            seed = _project_initial(trace, layer, capacity, observed_capacity)
            state = LayerState.from_experts(capacity, seed)
            for batch in by_layer[layer - 1]:
                state.plan(batch.experts)
                state.commit()
            layer_metrics.append(state.metrics)
        curves[capacity] = tuple(layer_metrics)
    return curves


def sum_metrics(items: Iterable[Metrics]) -> Metrics:
    result = Metrics()
    for item in items:
        result.add(item)
    return result


def exact_fixed_total_dp(curves: Mapping[int, Sequence[Metrics]], total: int,
                         minimum: int, maximum: int
                         ) -> tuple[tuple[int, ...], Metrics]:
    """Exact multiple-choice DP with stable multiobjective/lexicographic ties.

    Objective: maximize hits, then minimize admissions, then evictions, then
    choose the lexicographically smallest layer-capacity vector.
    """
    if minimum > maximum:
        raise AnalysisError("minimum layer slots exceed maximum")
    if total < LAYERS * minimum or total > LAYERS * maximum:
        raise AnalysisError("fixed total is infeasible for 92 layer bounds")
    for capacity in range(minimum, maximum + 1):
        if capacity not in curves or len(curves[capacity]) != LAYERS:
            raise AnalysisError("DP curves do not cover every layer/capacity")

    # budget -> (hits, admissions, evictions, lex-rank)
    previous: dict[int, tuple[int, int, int, int]] = {0: (0, 0, 0, 0)}
    parents: list[dict[int, tuple[int, int]]] = []
    for layer0 in range(LAYERS):
        chosen: dict[int, tuple[int, int, int, tuple[int, int], int, int]] = {}
        layers_left = LAYERS - layer0 - 1
        min_final = total - layers_left * maximum
        max_final = total - layers_left * minimum
        for old_budget, (old_h, old_a, old_e, old_rank) in previous.items():
            low = max(minimum, min_final - old_budget)
            high = min(maximum, max_final - old_budget)
            for capacity in range(low, high + 1):
                new_budget = old_budget + capacity
                metric = curves[capacity][layer0]
                h = old_h + metric.hits
                a = old_a + metric.admissions
                e = old_e + metric.evictions
                lex_key = (old_rank, capacity)
                candidate = (h, a, e, lex_key, old_budget, capacity)
                incumbent = chosen.get(new_budget)
                if incumbent is None:
                    chosen[new_budget] = candidate
                    continue
                objective = (h, -a, -e)
                incumbent_objective = (
                    incumbent[0], -incumbent[1], -incumbent[2])
                if (objective > incumbent_objective or
                        (objective == incumbent_objective and
                         lex_key < incumbent[3])):
                    chosen[new_budget] = candidate
        if not chosen:
            raise AnalysisError("fixed-total DP found no feasible state")
        ordered_keys = sorted({candidate[3] for candidate in chosen.values()})
        ranks = {key: rank for rank, key in enumerate(ordered_keys)}
        current: dict[int, tuple[int, int, int, int]] = {}
        layer_parents: dict[int, tuple[int, int]] = {}
        for budget, candidate in chosen.items():
            h, a, e, lex_key, old_budget, capacity = candidate
            current[budget] = (h, a, e, ranks[lex_key])
            layer_parents[budget] = (old_budget, capacity)
        previous = current
        parents.append(layer_parents)

    if total not in previous:
        raise AnalysisError("fixed-total DP did not reach requested total")
    vector = [0] * LAYERS
    budget = total
    for layer0 in range(LAYERS - 1, -1, -1):
        old_budget, capacity = parents[layer0][budget]
        vector[layer0] = capacity
        budget = old_budget
    result = Metrics()
    for layer0, capacity in enumerate(vector):
        result.add(curves[capacity][layer0])
    return tuple(vector), result


def exact_fixed_total_dp_generic(
        curves: Sequence[Mapping[int, tuple[int, int, int]]], total: int,
        minimum: int, maximum: int) -> tuple[int, ...]:
    """Small public generic DP helper used by synthetic/exhaustive tests."""
    count = len(curves)
    if total < count * minimum or total > count * maximum:
        raise AnalysisError("generic fixed total is infeasible")
    states: dict[int, tuple[tuple[int, int, int], tuple[int, ...]]] = {
        0: ((0, 0, 0), ())}
    for curve in curves:
        next_states: dict[int, tuple[tuple[int, int, int], tuple[int, ...]]] = {}
        for budget, (score, vector) in states.items():
            for capacity in range(minimum, maximum + 1):
                if capacity not in curve:
                    raise AnalysisError("generic DP curve is incomplete")
                new_budget = budget + capacity
                if new_budget > total:
                    continue
                h, a, e = curve[capacity]
                new_score = (score[0] + h, score[1] + a, score[2] + e)
                new_vector = (*vector, capacity)
                incumbent = next_states.get(new_budget)
                if (incumbent is None or (
                    (new_score[0], -new_score[1], -new_score[2]) >
                    (incumbent[0][0], -incumbent[0][1], -incumbent[0][2])
                ) or (
                    (new_score[0], -new_score[1], -new_score[2]) ==
                    (incumbent[0][0], -incumbent[0][1], -incumbent[0][2])
                    and new_vector < incumbent[1]
                )):
                    next_states[new_budget] = (new_score, new_vector)
        states = next_states
    if total not in states:
        raise AnalysisError("generic DP did not reach total")
    return states[total][1]


def _frequency_digest(frequencies: Sequence[Mapping[int, int]]) -> str:
    digest = hashlib.sha256()
    for layer, values in enumerate(frequencies, start=1):
        for expert, count in sorted(values.items()):
            digest.update(f"{layer},{expert},{count}\n".encode("ascii"))
    return digest.hexdigest()


def simulate_frequency_retained(trace: Trace, capacity: int,
                                observed_capacity: int) -> dict[str, object]:
    """Feasible experimental frequency-retained policy.

    This is TinyLFU-like only in retaining old residents by exact frequency.
    It is not canonical Window-TinyLFU.  All 16 requested experts are mandatory
    residents after every commit, so no request is rejected merely to protect a
    hot victim.  Frequencies are collision-free cumulative integers.
    """
    if capacity < TOP_K:
        raise AnalysisError(
            "frequency-retained policy requires capacity >= top-k residency")
    require_reconstructible_targets(trace, observed_capacity, (capacity,))
    states: list[list[int]] = []
    frequencies: list[dict[int, int]] = [dict() for _ in range(LAYERS)]
    metrics = [Metrics() for _ in range(LAYERS)]
    residency_violations = 0
    for layer in range(1, LAYERS + 1):
        seed = _project_initial(trace, layer, capacity, observed_capacity)
        states.append(list(seed))
    for batch in trace.batches:
        layer0 = batch.layer - 1
        current = states[layer0]
        current_set = set(current)
        hits = [expert in current_set for expert in batch.experts]
        pending_freq = dict(frequencies[layer0])
        for expert in batch.experts:
            pending_freq[expert] = pending_freq.get(expert, 0) + 1
        requested_set = set(batch.experts)
        candidates = [expert for expert in current
                      if expert not in requested_set]
        recency = {expert: rank for rank, expert in enumerate(current)}
        keep_count = capacity - TOP_K
        chosen = set(sorted(
            candidates,
            key=lambda expert: (
                -pending_freq.get(expert, 0),
                -recency[expert],
                expert,
            ),
        )[:keep_count])
        retained = [expert for expert in current if expert in chosen]
        pending = [*retained, *batch.experts]
        if len(pending) > capacity or not requested_set.issubset(pending):
            residency_violations += 1
        final_set = set(pending)
        hit_count = sum(hits)
        plan = Plan(
            tuple(Access(
                expert=expert,
                hit=hit,
                admit=not hit,  # every requested miss is mandatory-resident
                source_slot=None,
                destination_slot=None,
            ) for expert, hit in zip(batch.experts, hits)),
            tuple(Entry(expert, slot) for slot, expert in enumerate(pending)),
            hit_count,
            TOP_K - hit_count,
            TOP_K - hit_count,
            len(current_set - final_set),
        )
        # Atomic commit of both metadata and residency.
        frequencies[layer0] = pending_freq
        states[layer0] = pending
        metrics[layer0].add_plan(plan)
    total = sum_metrics(metrics)
    if residency_violations:
        raise AnalysisError("frequency-retained mandatory residency invariant failed")
    counter_entries = sum(len(values) for values in frequencies)
    counter_total = sum(sum(values.values()) for values in frequencies)
    return {
        "capacity": capacity,
        "comparison_scope": "full_capture",
        "deployability_label": (
            "experimental online TinyLFU-like frequency-retained; not "
            "canonical Window-TinyLFU and not a deployment claim"),
        "future_data_leakage": False,
        "initial_state": (
            "captured LRU residents with zeroed frequency metadata"),
        "frequency_metadata": {
            "aging": "none_within_capture",
            "collision_model": "none",
            "counter_entries": counter_entries,
            "counter_kind": "exact_nonnegative_integer_per_layer_expert",
            "counter_total": counter_total,
            "sha256": _frequency_digest(frequencies),
        },
        "mandatory_requested_batch_residency": True,
        "requested_batch_residency_violations": residency_violations,
        "selection_tie_rule": (
            "higher exact frequency, then newer current LRU rank, then lower "
            "expert ID; retained residents keep LRU order; request rank order "
            "is newest"),
        **total.as_dict(),
    }


def select_pins(batches: Sequence[Batch], pin_count: int
                ) -> tuple[tuple[int, ...], ...]:
    counts = [Counter() for _ in range(LAYERS)]
    for batch in batches:
        counts[batch.layer - 1].update(batch.experts)
    result: list[tuple[int, ...]] = []
    for layer_counts in counts:
        # Include zero-frequency IDs so every layer reserves exactly pin_count
        # slots even when a short training prefix saw fewer distinct experts.
        selected = sorted(range(EXPERTS), key=lambda expert: (
            -layer_counts[expert], expert))[:pin_count]
        result.append(tuple(selected))
    return tuple(result)


def simulate_pinned_lru(
        batches: Sequence[Batch], capacity: int,
        pins: Sequence[Sequence[int]], initial: Sequence[Sequence[int]] | None = None
        ) -> tuple[Metrics, int, int]:
    if len(pins) != LAYERS:
        raise AnalysisError("pinned policy needs 92 pin sets")
    pin_count = max((len(values) for values in pins), default=0)
    if pin_count > capacity - TOP_K:
        raise AnalysisError(
            "pin count must be <= capacity - 16 for mandatory batch residency")
    seeds = initial if initial is not None else ((),) * LAYERS
    loaded_pins: list[set[int]] = []
    dynamic: list[list[int]] = []
    pin_sets = [set(values) for values in pins]
    metrics = [Metrics() for _ in range(LAYERS)]
    warmup_pin_misses = 0
    violations = 0
    for layer0 in range(LAYERS):
        pin_set = pin_sets[layer0]
        seed = list(seeds[layer0])
        loaded_pins.append(set(seed) & pin_set)
        nonpins = [expert for expert in seed if expert not in pin_set]
        dynamic_capacity = capacity - len(pin_set)
        dynamic.append(nonpins[-dynamic_capacity:])
    for batch in batches:
        layer0 = batch.layer - 1
        pin_set = pin_sets[layer0]
        old_loaded = loaded_pins[layer0]
        old_dynamic = dynamic[layer0]
        old_set = old_loaded | set(old_dynamic)
        hits = [expert in old_set for expert in batch.experts]
        new_loaded = set(old_loaded)
        next_dynamic = list(old_dynamic)
        dynamic_capacity = capacity - len(pin_set)
        for expert, hit in zip(batch.experts, hits):
            if expert in pin_set:
                if not hit:
                    warmup_pin_misses += 1
                new_loaded.add(expert)
                continue
            if expert in next_dynamic:
                next_dynamic.remove(expert)
            elif len(next_dynamic) == dynamic_capacity:
                next_dynamic.pop(0)
            next_dynamic.append(expert)
        final_set = new_loaded | set(next_dynamic)
        requested = set(batch.experts)
        if len(final_set) > capacity or not requested.issubset(final_set):
            violations += 1
        hit_count = sum(hits)
        plan = Plan(
            tuple(Access(expert, hit, not hit, None, None)
                  for expert, hit in zip(batch.experts, hits)),
            (), hit_count, TOP_K - hit_count, TOP_K - hit_count,
            len(old_set - final_set),
        )
        loaded_pins[layer0] = new_loaded
        dynamic[layer0] = next_dynamic
        metrics[layer0].add_plan(plan)
    if violations:
        raise AnalysisError("pinned-LRU mandatory residency invariant failed")
    return sum_metrics(metrics), warmup_pin_misses, violations


def pinned_policy_results(trace: Trace, capacity: int,
                          observed_capacity: int,
                          pin_counts: Sequence[int], training_steps: int
                          ) -> list[dict[str, object]]:
    require_reconstructible_targets(trace, observed_capacity, (capacity,))
    if not 1 <= training_steps < trace.step_count:
        raise AnalysisError(
            "--training-steps must be in 1..(capture_steps - 1)")
    prefix = tuple(batch for batch in trace.batches
                   if batch.step < training_steps)
    suffix = tuple(batch for batch in trace.batches
                   if batch.step >= training_steps)
    results: list[dict[str, object]] = []
    for pin_count in pin_counts:
        if pin_count < 0 or pin_count > capacity - TOP_K:
            raise AnalysisError(
                f"pin count {pin_count} is outside 0..capacity-16")
        oracle_pins = select_pins(trace.batches, pin_count)
        oracle_metrics, oracle_warm, oracle_violations = simulate_pinned_lru(
            trace.batches, capacity, oracle_pins)
        results.append({
            "capacity": capacity,
            "comparison_scope": "full_capture",
            "deployability_label": (
                "oracle/nonpromotable: pin IDs use the evaluated full trace; "
                "not a deployable result"),
            "future_data_leakage": True,
            "initial_state": "cold empty cache",
            "mandatory_requested_batch_residency": True,
            "oracle_nonpromotable": True,
            "pin_count_per_layer": pin_count,
            "pin_selection_tie_rule": "frequency descending, expert ID ascending",
            "policy": "oracle_pinned_lru",
            "requested_batch_residency_violations": oracle_violations,
            "warmup_pin_misses": oracle_warm,
            **oracle_metrics.as_dict(),
        })

        trained_pins = select_pins(prefix, pin_count)
        trained_metrics, trained_warm, trained_violations = simulate_pinned_lru(
            suffix, capacity, trained_pins)
        suffix_lru = _replay_batches_cold(suffix, capacity)
        results.append({
            "baseline_cold_suffix_lru_hits": suffix_lru.hits,
            "capacity": capacity,
            "comparison_scope": "cold_suffix_only",
            "deployability_label": (
                "feasible prefix-trained pinned-LRU counterfactual; pin IDs "
                "use only the excluded prefix and suffix starts cold"),
            "future_data_leakage": False,
            "initial_state": "cold empty cache at scored suffix boundary",
            "evaluation_steps": trace.step_count - training_steps,
            "mandatory_requested_batch_residency": True,
            "oracle_nonpromotable": False,
            "pin_count_per_layer": pin_count,
            "pin_selection_tie_rule": "frequency descending, expert ID ascending",
            "policy": "prefix_trained_pinned_lru",
            "requested_batch_residency_violations": trained_violations,
            "training_steps": training_steps,
            "warmup_pin_misses": trained_warm,
            **trained_metrics.as_dict(),
        })
    return results


def _replay_batches_cold(batches: Sequence[Batch], capacity: int) -> Metrics:
    states = [LayerState.from_experts(capacity, ()) for _ in range(LAYERS)]
    for batch in batches:
        state = states[batch.layer - 1]
        state.plan(batch.experts)
        state.commit()
    return sum_metrics(state.metrics for state in states)


def _parse_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+):([0-9]+)", value)
    if not match:
        raise argparse.ArgumentTypeError("expected MIN:MAX")
    minimum, maximum = map(int, match.groups())
    if minimum <= 0 or maximum < minimum:
        raise argparse.ArgumentTypeError("range must satisfy 0 < MIN <= MAX")
    return minimum, maximum


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    parts = value.split(",")
    if any(not _UINT_RE.fullmatch(part) for part in parts):
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    values = tuple(int(part) for part in parts)
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("integer list contains duplicates")
    return values


def _logical_bytes(misses: int, expert_bytes: int | None) -> int | None:
    return None if expert_bytes is None else misses * expert_bytes


def _slot_bytes(slots: int, expert_bytes: int | None) -> int | None:
    return None if expert_bytes is None else slots * expert_bytes


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                      allow_nan=False) + "\n"


def _csv_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return output.getvalue()


def _write_outputs(out: Path, analysis: Mapping[str, object],
                   uniform_rows: Sequence[Mapping[str, object]],
                   marginal_rows: Sequence[Mapping[str, object]],
                   allocation_rows: Sequence[Mapping[str, object]],
                   policy_rows: Sequence[Mapping[str, object]],
                   report: str) -> None:
    try:
        if out.is_symlink() or (out.exists() and not out.is_dir()):
            raise AnalysisError(
                "output path must be a real directory, not a symlink")
        out.mkdir(parents=True, mode=0o700, exist_ok=True)
        out.chmod(0o700)
    except OSError as exc:
        raise AnalysisError(f"cannot prepare private output directory: {exc}") from exc
    payloads = {
        "analysis.json": _json_text(analysis),
        "uniform.csv": _csv_text((
            "capacity_per_layer", "total_slots", "cache_bytes", "batches",
            "accesses", "hits", "misses", "hit_rate", "admissions",
            "evictions", "logical_read_bytes", "selected",
        ), uniform_rows),
        "layer-marginals.csv": _csv_text((
            "layer", "capacity", "hits", "misses", "admissions",
            "evictions", "marginal_hits", "marginal_admissions",
            "marginal_evictions",
        ), marginal_rows),
        "allocations.csv": _csv_text((
            "policy", "comparison_scope", "oracle_nonpromotable",
            "future_data_leakage", "deployability_label", "layer",
            "capacity", "hits", "misses", "admissions", "evictions",
            "cache_bytes",
        ), allocation_rows),
        "policies.csv": _csv_text((
            "policy", "capacity", "pin_count_per_layer", "comparison_scope",
            "oracle_nonpromotable", "future_data_leakage",
            "deployability_label", "initial_state", "training_steps",
            "evaluation_steps", "batches", "accesses", "hits", "misses",
            "hit_rate", "admissions", "evictions", "logical_read_bytes",
            "warmup_pin_misses", "mandatory_requested_batch_residency",
            "requested_batch_residency_violations",
            "frequency_metadata_sha256",
        ), policy_rows),
        "report.txt": report if report.endswith("\n") else report + "\n",
    }
    for name in sorted(payloads):
        path = out / name
        temporary = out / f".{name}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        created = False
        try:
            descriptor = os.open(temporary, flags, 0o600)
            created = True
            os.fchmod(descriptor, 0o600)
            handle = os.fdopen(
                descriptor, "w", encoding="utf-8", newline="")
            descriptor = -1  # `handle` owns and closes it from this point.
            with handle:
                handle.write(payloads[name])
            os.replace(temporary, path)
            created = False
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            if created:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            raise AnalysisError(
                f"cannot write private output {path}: {exc}") from exc


def analyze(trace: Trace, observed_capacity: int,
            capacity_range: tuple[int, int], selected_capacities: Sequence[int],
            total_slots: int, min_layer_slots: int, max_layer_slots: int,
            pin_counts: Sequence[int], training_steps: int) -> tuple[
                dict[str, object], list[dict[str, object]],
                list[dict[str, object]], list[dict[str, object]],
                list[dict[str, object]], str]:
    gate = source_replay_gate(trace, observed_capacity)
    marginal_predecessor = (capacity_range[0] - 1
                            if capacity_range[0] > 1 else capacity_range[0])
    curve_min = min(marginal_predecessor, min_layer_slots,
                    *(selected_capacities or (observed_capacity,)))
    curve_max = max(capacity_range[1], max_layer_slots,
                    *(selected_capacities or (observed_capacity,)))
    curves = compute_lru_curves(
        trace, observed_capacity, curve_min, curve_max)
    allocation, allocation_metrics = exact_fixed_total_dp(
        curves, total_slots, min_layer_slots, max_layer_slots)

    selected = set(selected_capacities)
    uniform_capacities = sorted(
        set(range(capacity_range[0], capacity_range[1] + 1)) | selected)
    uniform_rows: list[dict[str, object]] = []
    uniform_json: list[dict[str, object]] = []
    for capacity in uniform_capacities:
        metrics = sum_metrics(curves[capacity])
        row = {
            "accesses": metrics.accesses,
            "admissions": metrics.admissions,
            "batches": metrics.batches,
            "cache_bytes": _slot_bytes(capacity * LAYERS, trace.expert_bytes),
            "capacity_per_layer": capacity,
            "evictions": metrics.evictions,
            "hit_rate": f"{metrics.hits / metrics.accesses:.12f}",
            "hits": metrics.hits,
            "logical_read_bytes": _logical_bytes(metrics.misses, trace.expert_bytes),
            "misses": metrics.misses,
            "selected": "yes" if capacity in selected else "no",
            "total_slots": capacity * LAYERS,
        }
        uniform_rows.append(row)
        uniform_json.append({
            key: (float(value) if key == "hit_rate" else value)
            for key, value in row.items() if key != "selected"
        })

    marginal_rows: list[dict[str, object]] = []
    for layer0 in range(LAYERS):
        previous: Metrics | None = (
            curves[capacity_range[0] - 1][layer0]
            if capacity_range[0] > 1 else None)
        for capacity in range(capacity_range[0], capacity_range[1] + 1):
            metric = curves[capacity][layer0]
            marginal_rows.append({
                "admissions": metric.admissions,
                "capacity": capacity,
                "evictions": metric.evictions,
                "hits": metric.hits,
                "layer": layer0 + 1,
                "marginal_admissions": "" if previous is None else
                    metric.admissions - previous.admissions,
                "marginal_evictions": "" if previous is None else
                    metric.evictions - previous.evictions,
                "marginal_hits": "" if previous is None else
                    metric.hits - previous.hits,
                "misses": metric.misses,
            })
            previous = metric

    allocation_rows: list[dict[str, object]] = []
    for layer0, capacity in enumerate(allocation):
        metric = curves[capacity][layer0]
        allocation_rows.append({
            "admissions": metric.admissions,
            "cache_bytes": _slot_bytes(capacity, trace.expert_bytes),
            "capacity": capacity,
            "comparison_scope": "full_capture",
            "deployability_label": (
                "oracle/nonpromotable: optimized and evaluated on the same "
                "full trace"),
            "evictions": metric.evictions,
            "future_data_leakage": True,
            "hits": metric.hits,
            "layer": layer0 + 1,
            "misses": metric.misses,
            "oracle_nonpromotable": True,
            "policy": "full_trace_optimized_per_layer_lru",
        })

    policies = pinned_policy_results(
        trace, observed_capacity, observed_capacity, pin_counts, training_steps)
    frequency = simulate_frequency_retained(
        trace, observed_capacity, observed_capacity)
    frequency["policy"] = "experimental_frequency_retained_tinylfu_like"
    policies.append(frequency)
    policies.sort(key=lambda row: (
        str(row["policy"]), int(row.get("pin_count_per_layer", -1))))
    policy_rows: list[dict[str, object]] = []
    for policy in policies:
        accesses = int(policy["accesses"])
        hits = int(policy["hits"])
        metadata = policy.get("frequency_metadata")
        policy_rows.append({
            "accesses": accesses,
            "admissions": policy["admissions"],
            "batches": policy["batches"],
            "capacity": policy["capacity"],
            "comparison_scope": policy["comparison_scope"],
            "deployability_label": policy["deployability_label"],
            "evaluation_steps": policy.get("evaluation_steps", ""),
            "evictions": policy["evictions"],
            "frequency_metadata_sha256": (
                metadata["sha256"] if isinstance(metadata, dict) else ""),
            "future_data_leakage": policy["future_data_leakage"],
            "hit_rate": f"{hits / accesses:.12f}" if accesses else "0.000000000000",
            "hits": hits,
            "initial_state": policy["initial_state"],
            "logical_read_bytes": _logical_bytes(
                int(policy["misses"]), trace.expert_bytes),
            "mandatory_requested_batch_residency": policy[
                "mandatory_requested_batch_residency"],
            "misses": policy["misses"],
            "oracle_nonpromotable": policy.get("oracle_nonpromotable", False),
            "pin_count_per_layer": policy.get("pin_count_per_layer", ""),
            "policy": policy["policy"],
            "requested_batch_residency_violations": policy[
                "requested_batch_residency_violations"],
            "training_steps": policy.get("training_steps", ""),
            "warmup_pin_misses": policy.get("warmup_pin_misses", ""),
        })

    initial_occupancies = [len(values) for values in trace.initial_lru]
    warm = any(initial_occupancies)
    warnings = [
        {
            "code": "cold_single_trace",
            "message": (
                "All counterfactuals use one captured route trace; cold/single-"
                "trace gains do not establish generalization."),
        },
        {
            "code": "allocation_oracle_nonpromotable",
            "message": (
                "The exact fixed-total capacity vector is optimized and "
                "evaluated on the same full trace; it is an upper-bound-style "
                "oracle result, not a promotion candidate."),
        },
        {
            "code": "pinned_oracle_nonpromotable",
            "message": (
                "Full-trace pinned-LRU chooses pins from the evaluated future "
                "and is an oracle/nonpromotable result."),
        },
        {
            "code": "prefix_suffix_boundary",
            "message": (
                "Prefix-trained pinned-LRU excludes its training prefix from "
                "scoring and starts the scored suffix with a cold cache."),
        },
    ]
    nonprivate = [
        name for name, mode in trace.input_modes.items() if mode != 0o600
    ]
    if nonprivate:
        warnings.append({
            "code": "input_privacy_mode",
            "message": (
                "Input diagnostic files are not all mode 0600 (" +
                ",".join(sorted(nonprivate)) +
                "); accepted historical analysis is possible, but final "
                "qualification requires private regular inputs."),
        })
    if warm:
        warnings.append({
            "code": "warm_snapshot_history_limit",
            "message": (
                "A warm snapshot contains current residents only, not evicted "
                "prefill history, invalidations, or frequency metadata. Smaller "
                "LRU capacities use an oldest-first projection; all larger "
                "capacities are rejected without explicit fresh-empty provenance; "
                "alternative policies cold-start missing metadata and are not "
                "exact warm counterfactuals."),
        })
    else:
        warnings.append({
            "code": "empty_snapshot_cold_start",
            "message": (
                "The selected snapshot is empty, so replay/policy simulations "
                "begin cold. Fresh-process provenance is explicitly asserted: "
                f"{str(trace.fresh_empty_source).lower()}. Results still cover "
                "only this trace."),
        })

    allocation_dict = {
        **allocation_metrics.as_dict(),
        "comparison_scope": "full_capture",
        "deployability_label": (
            "oracle/nonpromotable: the capacity vector is optimized and "
            "evaluated on the same full trace"),
        "future_data_leakage": True,
        "oracle_nonpromotable": True,
        "policy": "full_trace_optimized_per_layer_lru",
        "cache_bytes": _slot_bytes(total_slots, trace.expert_bytes),
        "capacity_vector": list(allocation),
        "logical_read_bytes": _logical_bytes(
            allocation_metrics.misses, trace.expert_bytes),
        "max_layer_slots": max_layer_slots,
        "min_layer_slots": min_layer_slots,
        "tie_rule": (
            "maximize hits; minimize admissions; minimize evictions; "
            "lexicographically smallest layer-1..92 capacity vector"),
        "total_slots": total_slots,
    }
    analysis: dict[str, object] = {
        "allocation": allocation_dict,
        "capture": trace.capture,
        "input": {
            "modes": {
                name: f"{mode:04o}"
                for name, mode in trace.input_modes.items()
            },
            "paths": dict(trace.paths),
            "sha256": dict(trace.input_sha256),
        },
        "ledger": {
            "expert_bytes": trace.expert_bytes,
            "schema": trace.ledger_schema,
            "summary": dict(trace.ledger_summary),
        },
        "parameters": {
            "marginal_capacity_max": capacity_range[1],
            "marginal_capacity_min": capacity_range[0],
            "max_layer_slots": max_layer_slots,
            "min_layer_slots": min_layer_slots,
            "observed_capacity": observed_capacity,
            "fresh_empty_source": trace.fresh_empty_source,
            "pin_counts": list(pin_counts),
            "selected_capacities": list(selected_capacities),
            "total_slots": total_slots,
            "training_steps": training_steps,
        },
        "policies": policies,
        "schema_version": SCHEMA_VERSION,
        "source_replay_gate": {
            **gate.total.as_dict(),
            "mask_mismatches": gate.mask_mismatches,
            "observed_capacity": observed_capacity,
            "passed": True,
            "semantics": (
                "frozen pre-batch hit lookup; rank-ordered pending LRU; "
                "atomic commit"),
        },
        "trace": {
            "accesses": trace.accesses,
            "first_position": trace.first_position,
            "initial_cache_entries": sum(initial_occupancies),
            "initial_cache_max_per_layer": max(initial_occupancies),
            "initial_cache_min_per_layer": min(initial_occupancies),
            "route_rows": len(trace.batches),
            "steps": trace.step_count,
        },
        "uniform_lru": uniform_json,
        "warnings": warnings,
    }

    observed_uniform = sum_metrics(curves[observed_capacity])
    lines = [
        "Moonshine decode-cache offline analysis (deterministic)",
        f"schema: {SCHEMA_VERSION}",
        f"capture: {trace.capture}",
        f"trace: {trace.step_count} steps, {len(trace.batches)} batches, "
        f"{trace.accesses} accesses",
        f"source replay C={observed_capacity}: PASS, "
        f"hits={gate.total.hits}, mask_mismatches=0",
        f"fresh empty source provenance: "
        f"{str(trace.fresh_empty_source).lower()}",
        f"uniform LRU C={observed_capacity}: hits={observed_uniform.hits}, "
        f"misses={observed_uniform.misses}",
        f"exact fixed-total LRU [ORACLE/NONPROMOTABLE]: slots={total_slots}, "
        f"bounds={min_layer_slots}..{max_layer_slots}, "
        f"hits={allocation_metrics.hits}, misses={allocation_metrics.misses}",
        "allocation tie rule: maximize hits; minimize admissions; minimize "
        "evictions; lexicographically smallest capacity vector",
        "policy labels: full-trace optimized allocation and full-trace "
        "pinned-LRU are ORACLE/NONPROMOTABLE; frequency-retained is online "
        "experimental TinyLFU-like, not canonical Window-TinyLFU; "
        "prefix-trained pinned-LRU is scored on a cold suffix",
        "warnings:",
        *(f"- {warning['code']}: {warning['message']}" for warning in warnings),
        "input SHA256:",
        *(f"- {name}: {trace.input_sha256[name]}"
          for name in sorted(trace.input_sha256)),
    ]
    return (analysis, uniform_rows, marginal_rows, allocation_rows,
            policy_rows, "\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic offline Moonshine decode-cache analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report", help="validate, replay, and report")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--capture", type=int, required=True,
                        help="explicit capture ID")
    report.add_argument("--observed-capacity", type=int, required=True,
                        help="explicit source capacity per layer")
    report.add_argument(
        "--fresh-empty-source", action="store_true",
        help=("attest that an empty snapshot came from a fresh process with "
              "no prior cache history; required for larger capacities"))
    report.add_argument("--cache", type=Path, help="override decode.cache.csv")
    report.add_argument("--ledger", type=Path, help="override decode.ledger.csv")
    report.add_argument("--routes", type=Path, help="override decode.routes.csv")
    report.add_argument("--capacities", type=_parse_csv_ints,
                        default=(), help="selected capacities to flag in CSV")
    report.add_argument("--marginal-range", type=_parse_range, default=(16, 64),
                        metavar="MIN:MAX")
    report.add_argument("--total-slots", type=int,
                        help="fixed total; default observed capacity x 92")
    report.add_argument("--min-layer-slots", type=int, default=16)
    report.add_argument("--max-layer-slots", type=int, default=64)
    report.add_argument(
        "--pin-counts", type=_parse_csv_ints, default=None,
        help="pin counts; default feasible subset of 4,8,12 (or 0)")
    report.add_argument("--training-steps", type=int,
                        help="prefix length; default floor(steps/4), at least 1")
    report.add_argument("--out", type=Path,
                        help="output directory; default RUN_DIR/cache-policy-analysis")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Accept both the documented `report RUN_DIR` form and a convenient direct
    # `RUN_DIR` form without weakening the requirement for explicit flags.
    if args_list and args_list[0] != "report" and not args_list[0].startswith("-"):
        args_list.insert(0, "report")
    parser = build_parser()
    args = parser.parse_args(args_list)
    try:
        trace = load_trace(
            args.run_dir, args.capture, args.observed_capacity,
            args.cache, args.ledger, args.routes,
            fresh_empty_source=args.fresh_empty_source)
        total_slots = (args.total_slots if args.total_slots is not None else
                       args.observed_capacity * LAYERS)
        training_steps = (args.training_steps if args.training_steps is not None
                          else max(1, trace.step_count // 4))
        selected = args.capacities or (args.observed_capacity,)
        if args.pin_counts is None:
            pin_counts = tuple(
                count for count in (4, 8, 12)
                if count <= args.observed_capacity - TOP_K)
            if not pin_counts:
                pin_counts = (0,)
        else:
            pin_counts = args.pin_counts
        outputs = analyze(
            trace, args.observed_capacity, args.marginal_range, selected,
            total_slots, args.min_layer_slots, args.max_layer_slots,
            pin_counts, training_steps)
        out = args.out or args.run_dir / "cache-policy-analysis"
        _write_outputs(out, *outputs[:-1], outputs[-1])
        print(outputs[-1], end="")
        print(f"outputs: {out}")
        return 0
    except AnalysisError as exc:
        print(f"analysis FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
