"""Regression tests for the D1–D3 policy-actionability scorer.

Locks the 30-study ground-truth agreement (expert manual score vs algorithm)
and the five mismatch fixes applied after the updated extraction prompt.
"""
from __future__ import annotations

import csv
import collections
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_studies_v2 import (  # noqa: E402
    METHODOLOGY_SOURCE_CATS,
    SYNTHESIS_SOURCE_CATS,
    _d1,
    _d2,
    _d3,
    process_file,
    score,
)

JSON_DIR = ROOT / "ilsa_survey_articles" / "json"
CSV_PATH = ROOT / "rescored_studies.csv"

# Expert manual scores for the 30-study validation set.
MANUAL_SCORES = {
    "bezek-gure-etal2020": 4,
    "demir-karaboga2021": 4,
    "acisli-celik-yesilkanat2023": 4,
    "oz-etal2024": 4,
    "tin-etal2024": 4,
    "song-cutumisu2024": 4,
    "nguyen-etal2023": 4,
    "pejic-etal2021": 4,
    "zhang-etal2023": 4,
    "sun-etal2023": 2,
    "fn-aydin-etal2025": 4,
    "tyack-etal2024": 4,
    "robitzsch-ludtke2022": 1,
    "hernandez-torrano-courtney2021": 2,
    "anghel-etal2024": 2,
    "huang-etal2025": 2,
    "ang-etal2020": 2,
    "jiao-etal2021": 2,
    "maia-etal023": 2,
    "rutkowski-etal2024": 1,
    "stiff-etal2023": 2,
    "scherer-etal2024": 2,
    "fink-etal2024": 2,
    "alvarez-etal2024": 4,
    "zhu-etal2025": 4,
    "elouafi-etal2025": 4,
    "gomez-talal-etal2025": 4,
    "zhai-etal2024": 4,
    "rico-juan-etal2024": 4,
    "khine-etal2024": 4,
}

GT_FILES = {
    "bezek-gure-etal2020": "20. Bezek-Güre et al. (2020) Analysis of Factors Effecting PISA 2015 Mathematic.json",
    "demir-karaboga2021": "11. Demir & Karaboğa. (2021). Modeling mathematics achievement with deep learni.json",
    "acisli-celik-yesilkanat2023": "8. Acıslı-Celik & Yesilkanat. (2023). Predicting science achievement scores with.json",
    "oz-etal2024": "45. Öz et al. (2024). Stacking: An ensemble learning approach to predict studen.json",
    "tin-etal2024": "53. Tin et al. (2024). Educational Big Data Mining: Comparison of Multiple Machi.json",
    "song-cutumisu2024": "5. Song & Cutumisu. (2024). Using machine learning to predict student science ac.json",
    "nguyen-etal2023": "35. Nguyen et al. (2023). An evaluation of tiered machine learning framework to .json",
    "pejic-etal2021": "91. Pejic et al. (2021). Math proficiency prediction in computer-based internati.json",
    "zhang-etal2023": "112. Zhang et al. (2023). Accurate Assessment via Process Data.json",
    "sun-etal2023": "127. Sun et al. (2023). A Theoretical Framework for a Mathematical Cognitive Mod.json",
    "fn-aydin-etal2025": "75. Aydin et al. (2025). Investigating the Predictive Performance of Process Dat.json",
    "tyack-etal2024": "113. Tyack et al. (2024). Using convolutional neural networks to automatically s.json",
    "robitzsch-ludtke2022": "94. Robitzsch & Lüdtke. (2022). Some thoughts on analytical choices in the scal.json",
    "hernandez-torrano-courtney2021": "69. Hernández‑Torrano & Courtney. (2021). Modern international large‑scale asse.json",
    "anghel-etal2024": "67. Anghel et al. (2024). The use of process data in large‑scale assessments: a .json",
    "huang-etal2025": "125. Huang & Keller (2025). Working with missing data in large‑scale assessments.json",
    "ang-etal2020": "110. Ang et al. (2020). Big Educational Data & Analytics: Survey, Architecture a.json",
    "jiao-etal2021": "99. Jiao et al. (2021). Editorial: Process Data in Educational and Psychological.json",
    "maia-etal023": "70. Maia et al. (2023). Applications of Artificial Intelligence Models in Educat.json",
    "rutkowski-etal2024": "118. Rutkowski et al. (2024). The limits of inference: reassessing causality in .json",
    "stiff-etal2023": "105. Stiff et al. (2023). Research engagement in the Progress in International R.json",
    "scherer-etal2024": "103. Scherer et al. (2024). The potential of international large‑scale assessmen.json",
    "fink-etal2024": "109. Fink et al. (2024). Methodological aspects of the highly adaptive testing d.json",
    "alvarez-etal2024": "36. Alvarez-Garcia et al. (2024). Uncovering student profiles. An explainable cl.json",
    "zhu-etal2025": "2. Zhu et al. (2025). Predictive insights into U.S. students’ mathematics perfor.json",
    "elouafi-etal2025": "6. Elouafi. (2025). Uncovering Key Factors of Student Performance in Math_ An Ex.json",
    "gomez-talal-etal2025": "12. Gomez-Talal et al. (2025). Interpretable Machine Learning Models for PISA Re.json",
    "zhai-etal2024": "106. Zhai et al. (2023). Machine learning investigation of optimal psychoemotion.json",
    "rico-juan-etal2024": "50. Rico-Juan et al. (2024). Holistic exploration of reading comprehension skill.json",
    "khine-etal2024": "65. Khine et al. (2024). A Machine-Learning Approach to Predicting the Achieveme.json",
}

EXPECTED_CORPUS_COUNTS = {"1": 7, "2": 22, "3": 1, "4": 187, "5": 2}


def _write_article(tmpdir: str, source_category: str, data: dict) -> str:
    import json

    path = os.path.join(tmpdir, "article.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"title": "t", "source_category": source_category}, "data": data}, fh)
    return path


class TestDimensionRules(unittest.TestCase):
    def test_d1_synthesis_from_review_categories(self):
        for cat in SYNTHESIS_SOURCE_CATS:
            self.assertEqual(_d1(cat, {"ml_techniques": {"primary": "Random Forest"}}), "Synthesis")

    def test_d1_technical_report_is_methodology_synthesis(self):
        # jiao-etal2021 fix: technical_report must not become Correlational via has_ml
        self.assertIn("technical_report", METHODOLOGY_SOURCE_CATS)
        self.assertEqual(
            _d1("technical_report", {"ml_techniques": {"primary": "Neural Network"}}),
            "Synthesis",
        )

    def test_d1_causal_from_bart(self):
        data = {"ml_techniques": {"primary": "BART", "all_techniques": ["BART"]}}
        self.assertEqual(_d1("peer_reviewed_research", data), "Causal")

    def test_d1_correlational_from_ml(self):
        data = {"ml_techniques": {"primary": "Random Forest", "all_techniques": ["RF"]}}
        self.assertEqual(_d1("peer_reviewed_research", data), "Correlational")

    def test_d1_correlational_from_predictive_design(self):
        data = {"research_design_type": "predictive", "ml_techniques": {}}
        self.assertEqual(_d1("peer_reviewed_research", data), "Correlational")

    def test_d1_none_without_signal(self):
        self.assertEqual(_d1("peer_reviewed_research", {}), "None")

    def test_d3_bounded_when_countries_listed(self):
        self.assertEqual(_d3({"sample_details": {"countries": ["TUR"]}}), "Bounded")
        self.assertEqual(_d3({"sample_details": {"countries": []}}), "Global/Unspecified")

    def test_score_table(self):
        self.assertEqual(score("Causal", "Quantified", "Bounded"), 5)
        self.assertEqual(score("Correlational", "Quantified", "Global/Unspecified"), 4)
        self.assertEqual(score("Correlational", "Directional", "Bounded"), 3)
        self.assertEqual(score("Synthesis", "Directional", "Global/Unspecified"), 2)
        self.assertEqual(score("Synthesis", "None", "Global/Unspecified"), 1)
        self.assertEqual(score("None", "Quantified", "Bounded"), 1)


class TestNegationFixes(unittest.TestCase):
    """Five mismatches after re-extraction; strong vs weak empirical disclaimers."""

    def test_strong_negation_does_not_estimate(self):
        # rutkowski-etal2024
        d2 = _d2(
            {
                "main_findings": [],
                "outcome_summary": (
                    "The paper does not estimate predictive or causal effects "
                    "and argues the limits of inference in ILSA."
                ),
            },
            "Synthesis",
        )
        self.assertEqual(d2, "None")

    def test_strong_negation_methodological_contribution(self):
        # robitzsch-ludtke2022
        d2 = _d2(
            {
                "main_findings": [],
                "outcome_summary": (
                    "This article is a methodological contribution on scaling models "
                    "and discusses analytical choices."
                ),
            },
            "Synthesis",
        )
        self.assertEqual(d2, "None")

    def test_weak_negation_does_not_override_directional_summary(self):
        # stiff-etal2023 / sun-etal2023: "not an empirical" / "no predictive"
        # must be checked AFTER directional signals
        d2 = _d2(
            {
                "main_findings": [
                    {
                        "standardized_conclusion": (
                            "The review identifies factors associated with research engagement."
                        )
                    }
                ],
                "outcome_summary": (
                    "Findings identify patterns of research engagement. "
                    "This is not an empirical ML training study and has no predictive model."
                ),
            },
            "Synthesis",
        )
        self.assertEqual(d2, "Directional")

    def test_weak_negation_alone_is_none(self):
        d2 = _d2(
            {
                "main_findings": [],
                "outcome_summary": "The paper is not an empirical analysis of student data.",
            },
            "Synthesis",
        )
        self.assertEqual(d2, "None")

    def test_end_to_end_jiao_technical_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_article(
                tmp,
                "technical_report",
                {
                    "research_design_type": "exploratory",
                    "ml_techniques": {"primary": "process mining", "all_techniques": []},
                    "main_findings": [],
                    "outcome_summary": "Editorial introducing process data papers.",
                    "sample_details": {"countries": []},
                },
            )
            row = process_file(path)
        self.assertEqual(row["D1"], "Synthesis")


class TestGroundTruthAgreement(unittest.TestCase):
    def test_thirty_files_exist(self):
        self.assertEqual(len(GT_FILES), 30)
        self.assertEqual(len(MANUAL_SCORES), 30)
        missing = [ck for ck, name in GT_FILES.items() if not (JSON_DIR / name).is_file()]
        self.assertEqual(missing, [])

    def test_exact_agreement_30_of_30(self):
        mismatches = []
        for citekey, filename in GT_FILES.items():
            row = process_file(str(JSON_DIR / filename))
            expected = MANUAL_SCORES[citekey]
            if row["score"] != expected:
                mismatches.append(
                    {
                        "citekey": citekey,
                        "expected": expected,
                        "got": row["score"],
                        "D1": row["D1"],
                        "D2": row["D2"],
                        "D3": row["D3"],
                        "file": filename,
                    }
                )
        self.assertEqual(mismatches, [], msg=mismatches)


class TestCorpusOutput(unittest.TestCase):
    def test_json_count_and_score_distribution(self):
        files = sorted(JSON_DIR.glob("*.json"))
        self.assertEqual(len(files), 219)
        counts = collections.Counter()
        for path in files:
            counts[str(process_file(str(path))["score"])] += 1
        self.assertEqual(dict(counts), EXPECTED_CORPUS_COUNTS)

    def test_csv_matches_live_scorer(self):
        self.assertTrue(CSV_PATH.is_file())
        with CSV_PATH.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 219)
        live = {r["file"]: str(process_file(str(JSON_DIR / r["file"]))["score"]) for r in rows}
        csv_scores = {r["file"]: r["score"] for r in rows}
        self.assertEqual(csv_scores, live)


class TestSchemaAndPrompt(unittest.TestCase):
    def test_source_category_enum_covers_scorer_categories(self):
        models_src = (ROOT / "src" / "schemas" / "models.py").read_text(encoding="utf-8")
        expected = SYNTHESIS_SOURCE_CATS | METHODOLOGY_SOURCE_CATS | {"peer_reviewed_research"}
        missing = [cat for cat in sorted(expected) if f'"{cat}"' not in models_src]
        self.assertEqual(missing, [])

    def test_system_prompt_lists_extended_source_category(self):
        prompt = (ROOT / "src" / "extractors" / "gpt_extractor.py").read_text(encoding="utf-8")
        for label in (
            "systematic_review",
            "framework_paper",
            "theoretical_paper",
            "opinion_piece",
        ):
            self.assertIn(label, prompt)
        self.assertIn("15) REVIEW / META-ANALYSIS", prompt)
        self.assertIn("16) NON-EMPIRICAL / FRAMEWORK", prompt)


if __name__ == "__main__":
    unittest.main()
