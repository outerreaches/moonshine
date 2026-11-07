#!/usr/bin/env python3
"""Apply the frozen M4 quality thresholds to a mimo26_eval JSONL run.

    tools/mimo26_eval ROOT < tokens > eval.jsonl
    tests/score_mimo26_eval.py eval.jsonl

The thresholds live in tests/mimo26_eval_spec.json and were frozen before any
evaluation was run. This script only reads them; it does not know how to
choose them, which is the point. If a threshold needs to change, that is a
commit with a reason, not an edit here.
"""
import json
import pathlib
import statistics
import sys

SPEC = pathlib.Path(__file__).with_name("mimo26_eval_spec.json")


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} EVAL.jsonl", file=sys.stderr)
        return 2
    spec = json.loads(SPEC.read_text())
    limits = spec["thresholds"]

    records, summary = [], None
    for line in pathlib.Path(argv[1]).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("summary"):
            summary = entry
        else:
            records.append(entry)

    if not records:
        print("no scored positions", file=sys.stderr)
        return 1

    ranks = [r["rank"] for r in records]
    logprobs = [r["logprob"] for r in records]
    # worst_rank applies only where the model had context to use; see the
    # spec's rationale for worst_rank_min_position.
    floor = limits.get("worst_rank_min_position", 0)
    with_context = [r for r in records if r["position"] >= floor]
    measured = {
        "positions": len(records),
        "mean_logprob": sum(logprobs) / len(logprobs),
        "top1_accuracy": sum(r["rank"] == 1 for r in records) / len(records),
        "top10_accuracy": sum(r["rank"] <= 10 for r in records) / len(records),
        "median_rank": statistics.median(ranks),
        "worst_rank": max(r["rank"] for r in with_context) if with_context
                      else max(ranks),
        "nonfinite_logits": sum(r["nonfinite"] for r in records),
        "aborted_steps": (summary or {}).get("aborted_steps", 0),
    }

    # Each gate as (measured key, comparison, threshold key).
    gates = [
        ("mean_logprob", "min", "mean_logprob_min"),
        ("top1_accuracy", "min", "top1_accuracy_min"),
        ("top10_accuracy", "min", "top10_accuracy_min"),
        ("median_rank", "max", "median_rank_max"),
        ("worst_rank", "max", "worst_rank_max"),
        ("nonfinite_logits", "max", "nonfinite_logits_max"),
        ("aborted_steps", "max", "aborted_steps_max"),
    ]

    print(f"spec frozen {spec['frozen']}, {measured['positions']} scored "
          f"positions")
    failures = 0
    for key, how, limit_key in gates:
        value, limit = measured[key], limits[limit_key]
        passed = value >= limit if how == "min" else value <= limit
        failures += not passed
        arrow = ">=" if how == "min" else "<="
        print(f"  {'ok ' if passed else 'FAIL'}  {key:<18} {value:>12.4f} "
              f"{arrow} {limit}")

    if summary:
        hits, accesses = summary["expert_hits"], summary["expert_accesses"]
        rate = 100.0 * hits / accesses if accesses else 0.0
        print(f"  --    expert hit rate {rate:.1f}% over {accesses} accesses, "
              f"{summary['resident_bytes'] / 2**30:.2f} GiB resident at "
              f"{summary['slots_per_layer']} slots")

    worst = max(records, key=lambda r: r["rank"])
    print(f"  --    worst overall: position {worst['position']}, fed "
          f"{worst['fed']}, truth {worst['truth']}, rank {worst['rank']}"
          f"{' (excluded from worst_rank: no context)' if worst['position'] < floor else ''}")
    if with_context and worst["position"] < floor:
        w2 = max(with_context, key=lambda r: r["rank"])
        print(f"  --    worst with context: position {w2['position']}, fed "
              f"{w2['fed']}, truth {w2['truth']}, rank {w2['rank']}")

    print(f"score_mimo26_eval: {'ok' if failures == 0 else 'FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
