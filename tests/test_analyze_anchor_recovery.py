from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

TOOL = Path(__file__).parents[1] / "tools" / "analyze_anchor_recovery.py"
SPEC = importlib.util.spec_from_file_location("analyze_anchor_recovery", TOOL)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


class AnchorRecoveryAnalysisTest(unittest.TestCase):
    def test_exact_policy_passes(self) -> None:
        result = analyzer.analyze({
            "schema": analyzer.SCHEMA,
            "gate_pct": 60,
            "shapes": [{
                "name": "turn-boundary-edit",
                "scope": "edited_continuation",
                "span_tokens": 100,
                "matched_tokens": 80,
                "eligible_anchor_positions": [20, 80],
            }],
        })
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["results"][0]["selected_anchor_tokens"], 80)

    def test_ideal_exact_upper_bound_can_fail(self) -> None:
        result = analyzer.analyze({
            "schema": analyzer.SCHEMA,
            "gate_pct": 60,
            "shapes": [{
                "name": "deep-compaction",
                "scope": "edited_continuation",
                "span_tokens": 29630,
                "matched_tokens": 8,
                "eligible_anchor_positions": [8],
            }],
        })
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(
            result["ideal_gate_impossible_shapes"], ["deep-compaction"])

    def test_new_prompt_does_not_fail_edit_gate(self) -> None:
        result = analyzer.analyze({
            "schema": analyzer.SCHEMA,
            "gate_pct": 60,
            "shapes": [
                {
                    "name": "edit",
                    "scope": "edited_continuation",
                    "span_tokens": 100,
                    "matched_tokens": 80,
                    "eligible_anchor_positions": [80],
                },
                {
                    "name": "new-task",
                    "scope": "new_prompt",
                    "span_tokens": 26227,
                    "matched_tokens": 79,
                    "eligible_anchor_positions": [79],
                },
            ],
        })
        self.assertEqual(result["decision"], "PASS")
        self.assertFalse(result["results"][1]["gate_pass"])

    def test_rejects_nonexact_or_unsorted_anchors(self) -> None:
        base = {
            "schema": analyzer.SCHEMA,
            "gate_pct": 60,
            "shapes": [{
                "name": "invalid",
                "scope": "edited_continuation",
                "span_tokens": 10,
                "matched_tokens": 5,
                "eligible_anchor_positions": [6],
            }],
        }
        with self.assertRaises(analyzer.AnalysisError):
            analyzer.analyze(base)
        base["shapes"][0]["eligible_anchor_positions"] = [5, 3]
        with self.assertRaises(analyzer.AnalysisError):
            analyzer.analyze(base)


if __name__ == "__main__":
    unittest.main()
