"""Offline enrichment utilities for on-disk article JSON."""

from __future__ import annotations

from typing import Any

__all__ = [
    "enrich_article_dict",
    "enrich_article_json_file",
    "score_article",
    "score_corpus",
]


def __getattr__(name: str) -> Any:
    if name in ("enrich_article_dict", "enrich_article_json_file"):
        from src.enrichment.json_gap_fill import enrich_article_dict, enrich_article_json_file

        mapping = {
            "enrich_article_dict": enrich_article_dict,
            "enrich_article_json_file": enrich_article_json_file,
        }
        return mapping[name]
    if name in ("score_article", "score_corpus"):
        from src.enrichment.policy_actionability import score_article, score_corpus

        mapping = {"score_article": score_article, "score_corpus": score_corpus}
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
