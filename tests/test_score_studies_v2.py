"""Ground-truth scoring cases from the 30-study validation (100% exact agreement)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_studies_v2 import _d1, _d2, _d3, score  # noqa: E402
from src.schemas.models import MetadataBlock  # noqa: E402
from src.schemas.source_categories import SOURCE_CATEGORY_VALUES  # noqa: E402


def _article(source_cat: str, data: dict) -> tuple[str, str, str, int]:
    d1 = _d1(source_cat, data)
    d2 = _d2(data, d1)
    d3 = _d3(data)
    return d1, d2, d3, score(d1, d2, d3)


def test_source_category_schema_accepts_twelve_literals():
    for cat in sorted(SOURCE_CATEGORY_VALUES):
        meta = MetadataBlock(file_name="x.pdf", source_category=cat)
        assert meta.source_category == cat
    assert len(SOURCE_CATEGORY_VALUES) == 13  # 12 subtypes + peer_reviewed_research


def test_jiao_technical_report_is_synthesis():
    """jiao-etal2021: technical_report must not become Correlational via has_ml."""
    data = {
        "ml_techniques": {"primary": "Random Forest", "all_techniques": ["Random Forest"]},
        "research_design_type": "exploratory",
        "main_findings": [],
        "outcome_summary": "Editorial introducing a special issue on process data.",
        "sample_details": {"countries": []},
    }
    d1, d2, d3, s = _article("technical_report", data)
    assert d1 == "Synthesis"
    assert s in (1, 2)


def test_rutkowski_does_not_estimate_is_strong_negation():
    """rutkowski-etal2024: 'does not estimate' forces D2=None before directional verbs."""
    data = {
        "ml_techniques": {"primary": None, "all_techniques": []},
        "research_design_type": "exploratory",
        "main_findings": [],
            "outcome_summary": (
                "The study does not estimate predictive or causal effects "
                "suitable for policy targeting, though it argues for design limits."
            ),
        "sample_details": {"countries": []},
    }
    d1, d2, d3, s = _article("theoretical_paper", data)
    assert d1 == "Synthesis"
    assert d2 == "None"
    assert d3 == "Global/Unspecified"
    assert s == 1


def test_robitzsch_methodological_contribution_is_strong_negation():
    """robitzsch-ludtke2022: 'is a methodological contribution' → D2=None."""
    data = {
        "ml_techniques": {"primary": None, "all_techniques": []},
        "research_design_type": "exploratory",
        "main_findings": [],
        "outcome_summary": (
            "This article is a methodological contribution comparing scaling models "
            "and suggests implications for IRT practice."
        ),
        "sample_details": {"countries": []},
    }
    d1, d2, d3, s = _article("methodology_paper", data)
    assert d1 == "Synthesis"
    assert d2 == "None"


def test_stiff_weak_negation_does_not_override_directional():
    """stiff-etal2023: 'not an empirical' is weak; directional synthesis stays Directional."""
    data = {
        "ml_techniques": {"primary": None, "all_techniques": []},
        "research_design_type": "exploratory",
        "main_findings": [],
        "outcome_summary": (
            "The review identifies patterns of research engagement in PIRLS. "
            "This is not an empirical ML study of student micro-data."
        ),
        "sample_details": {"countries": []},
    }
    d1, d2, d3, s = _article("review_article", data)
    assert d1 == "Synthesis"
    assert d2 == "Directional"
    assert s == 2


def test_sun_conclusions_checked_after_weak_negation():
    """sun-etal2023: 'no predictive' is weak; conclusions can still be Directional."""
    data = {
        "ml_techniques": {"primary": None, "all_techniques": []},
        "research_design_type": "exploratory",
        "main_findings": [{
            "performance_metrics": "Not reported",
            "standardized_conclusion": (
                "The framework identifies cognitive attributes that demonstrate "
                "how mathematical literacy can be modelled."
            ),
        }],
        "outcome_summary": "The paper proposes a theoretical model with no predictive ML evaluation.",
        "sample_details": {"countries": []},
    }
    d1, d2, d3, s = _article("framework_paper", data)
    assert d1 == "Synthesis"
    assert d2 == "Directional"


def test_quantified_correlational_is_score_4():
    data = {
        "ml_techniques": {"primary": "XGBoost", "all_techniques": ["XGBoost", "RF"]},
        "research_design_type": "predictive",
        "main_findings": [{
            "performance_metrics": "AUC=0.82, Accuracy: 85%",
            "standardized_conclusion": "Using PISA 2018, ESCS predicts math literacy.",
        }],
        "outcome_summary": "XGBoost achieved AUC 0.82.",
        "sample_details": {"countries": [{"country_code": "TUR", "n_students": 1000}]},
    }
    d1, d2, d3, s = _article("peer_reviewed_research", data)
    assert (d1, d2, d3, s) == ("Correlational", "Quantified", "Bounded", 4)


def test_causal_quantified_bounded_is_score_5():
    data = {
        "ml_techniques": {"primary": "BART", "all_techniques": ["BART", "BCF"]},
        "research_design_type": "causal_observational",
        "main_findings": [{
            "performance_metrics": "ATE=0.12, p=0.01",
            "standardized_conclusion": "BART estimates a positive effect of ICT.",
        }],
        "outcome_summary": "Causal forest ATE 0.12.",
        "sample_details": {"countries": [{"country_code": "USA", "n_students": 5000}]},
    }
    d1, d2, d3, s = _article("peer_reviewed_research", data)
    assert (d1, d2, d3, s) == ("Causal", "Quantified", "Bounded", 5)
