"""Canonical source_category literals shared by schema, extractor, and scorer."""

from __future__ import annotations

from typing import FrozenSet

# Synthesis subtypes → D1=Synthesis in score_studies_v2.py
SYNTHESIS_SOURCE_CATEGORIES: FrozenSet[str] = frozenset({
    "review_article",
    "systematic_review",
    "scoping_review",
    "meta_analysis",
    "literature_review",
})

# Methodology / non-empirical subtypes → D1=Synthesis in score_studies_v2.py
METHODOLOGY_SOURCE_CATEGORIES: FrozenSet[str] = frozenset({
    "methodology_paper",
    "framework_paper",
    "editorial",
    "theoretical_paper",
    "commentary",
    "opinion_piece",
    "technical_report",
})

SOURCE_CATEGORY_VALUES: FrozenSet[str] = frozenset({
    "peer_reviewed_research",
    *SYNTHESIS_SOURCE_CATEGORIES,
    *METHODOLOGY_SOURCE_CATEGORIES,
})

NON_EMPIRICAL_SOURCE_CATEGORIES: FrozenSet[str] = (
    SYNTHESIS_SOURCE_CATEGORIES | METHODOLOGY_SOURCE_CATEGORIES
)
