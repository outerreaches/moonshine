#!/usr/bin/env python3
"""Screen exact checkpoint-anchor recovery policies from token-count evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

SCHEMA = "moonshine-anchor-recovery-input-v1"
SCOPES = {"edited_continuation", "new_prompt", "exact_displacement"}


class AnalysisError(ValueError):
    pass


def analyze(document: dict[str, object]) -> dict[str, object]:
    if document.get("schema") != SCHEMA:
        raise AnalysisError("unexpected anchor-recovery input schema")
    gate_pct = document.get("gate_pct")
    shapes = document.get("shapes")
    if not isinstance(gate_pct, (int, float)) or not 0.0 < gate_pct <= 100.0:
        raise AnalysisError("invalid recovery gate")
    if not isinstance(shapes, list) or not shapes:
        raise AnalysisError("anchor-recovery input has no shapes")

    results: list[dict[str, object]] = []
    gated: list[dict[str, object]] = []
    for index, raw in enumerate(shapes):
        if not isinstance(raw, dict):
            raise AnalysisError(f"shape {index} is not an object")
        name = raw.get("name")
        scope = raw.get("scope")
        span = raw.get("span_tokens")
        matched = raw.get("matched_tokens")
        anchors = raw.get("eligible_anchor_positions")
        if (not isinstance(name, str) or not name or scope not in SCOPES or
            not isinstance(span, int) or span < 2 or
            not isinstance(matched, int) or matched < 0 or matched > span or
            not isinstance(anchors, list) or
            any(not isinstance(anchor, int) for anchor in anchors)):
            raise AnalysisError(f"shape {index} has invalid fields")
        if anchors != sorted(set(anchors)):
            raise AnalysisError(f"{name}: anchors must be sorted and unique")
        if any(anchor < 0 or anchor > matched or span - anchor < 2
               for anchor in anchors):
            raise AnalysisError(f"{name}: anchor violates exact-prefix/suffix bounds")
        ideal_anchor = min(matched, span - 2)
        selected_anchor = anchors[-1] if anchors else 0
        ideal_pct = 100.0 * ideal_anchor / span
        selected_pct = 100.0 * selected_anchor / span
        result = {
            "name": name,
            "scope": scope,
            "span_tokens": span,
            "matched_tokens": matched,
            "ideal_exact_anchor_tokens": ideal_anchor,
            "ideal_exact_recovery_pct": ideal_pct,
            "selected_anchor_tokens": selected_anchor,
            "selected_recovery_pct": selected_pct,
            "gate_pass": selected_pct >= float(gate_pct),
            "ideal_gate_possible": ideal_pct >= float(gate_pct),
        }
        results.append(result)
        if scope == "edited_continuation":
            gated.append(result)

    if not gated:
        raise AnalysisError("anchor-recovery input has no edited continuation")
    impossible = [item["name"] for item in gated
                  if not item["ideal_gate_possible"]]
    failed = [item["name"] for item in gated if not item["gate_pass"]]
    return {
        "schema": "moonshine-anchor-recovery-analysis-v1",
        "gate_pct": float(gate_pct),
        "gated_scope": "edited_continuation",
        "results": results,
        "ideal_gate_impossible_shapes": impossible,
        "failed_shapes": failed,
        "decision": "PASS" if not failed else "NO_GO",
        "interpretation": (
            "Exact checkpoints can restore only a prefix no deeper than the "
            "candidate's full exact common prefix."
        ),
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
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as stream:
        document = json.load(stream)
    result = analyze(document)
    _atomic_json(args.output, result)
    print(f"Moonshine anchor recovery: {result['decision']}")


if __name__ == "__main__":
    main()
