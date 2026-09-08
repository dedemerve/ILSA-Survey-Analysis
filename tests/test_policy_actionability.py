"""Unit tests for the policy-actionability decision rules."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.enrichment.policy_actionability import (
    apply_decision_rule,
    score_article,
    score_json_file,
)

ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "ilsa_survey_articles" / "json"


class DecisionRuleTests(unittest.TestCase):
    def test_rule1_causal_quantified_bounded(self):
        score, _, unmatched = apply_decision_rule("Causal", "Quantified", "Bounded")
        self.assertEqual(score, 5)
        self.assertFalse(unmatched)

    def test_rule2_correlational_quantified_bounded(self):
        score, _, unmatched = apply_decision_rule("Correlational", "Quantified", "Bounded")
        self.assertEqual(score, 4)
        self.assertFalse(unmatched)

    def test_rule2_ignores_d1_synthesis_if_quantified_bounded(self):
        score, _, _ = apply_decision_rule("Synthesis", "Quantified", "Bounded")
        self.assertEqual(score, 4)

    def test_rule3_correlational_directional_bounded(self):
        score, _, _ = apply_decision_rule("Correlational", "Directional", "Bounded")
        self.assertEqual(score, 3)

    def test_rule3_correlational_quantified_global(self):
        score, _, _ = apply_decision_rule("Correlational", "Quantified", "Global/Unspecified")
        self.assertEqual(score, 3)

    def test_rule4_synthesis(self):
        score, _, _ = apply_decision_rule("Synthesis", "None", "Global/Unspecified")
        self.assertEqual(score, 2)

    def test_rule5_none_bounded(self):
        score, _, _ = apply_decision_rule("None", "None", "Bounded")
        self.assertEqual(score, 1)


class CorpusSmokeTests(unittest.TestCase):
    def test_mcjames_2024_is_score_5(self):
        path = next(JSON_DIR.glob("33. McJames*.json"))
        result = score_json_file(path)
        self.assertEqual(result["D1"], "Causal")
        self.assertEqual(result["D2"], "Quantified")
        self.assertEqual(result["D3"], "Bounded")
        self.assertEqual(result["Score"], 5)

    def test_pan_cutumisu_is_score_4(self):
        path = next(JSON_DIR.glob("48. Pan*.json"))
        result = score_json_file(path)
        self.assertEqual(result["D1"], "Correlational")
        self.assertEqual(result["D2"], "Quantified")
        self.assertEqual(result["D3"], "Bounded")
        self.assertEqual(result["Score"], 4)

    def test_stiff_review_is_synthesis(self):
        path = next(JSON_DIR.glob("105. Stiff*.json"))
        result = score_json_file(path)
        self.assertEqual(result["D1"], "Synthesis")
        self.assertEqual(result["Score"], 2)

    def test_rutkowski_is_score_2(self):
        path = next(JSON_DIR.glob("118. Rutkowski*.json"))
        result = score_json_file(path)
        self.assertEqual(result["D1"], "Synthesis")
        self.assertEqual(result["Score"], 2)

    def test_robitzsch_is_score_1(self):
        path = next(JSON_DIR.glob("94. Robitzsch*.json"))
        result = score_json_file(path)
        self.assertEqual(result["D1"], "None")
        self.assertEqual(result["Score"], 1)

    def test_output_keys(self):
        article = json.loads(next(JSON_DIR.glob("5. Song*.json")).read_text(encoding="utf-8"))
        result = score_article(article)
        for key in ("D1", "D2", "D3", "Score", "D1_evidence", "D2_evidence", "D3_evidence", "Justification"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
