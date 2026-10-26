"""
Load the study corpus and attach the derived RQ1 attributes.

The corpus folder is left untouched. Duplicate extractions of the same article
(same DOI and matching title) are collapsed with the project's existing loader
(``scripts/build_tabular_dataset.load_and_dedupe_articles``), which yields the
212 unique articles of the review.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.ml_taxonomy import canonical_name, canonicalize_list
from src.analysis.performance_parser import canonical_metric, parse_performance
from src.analysis.study_attributes import derive_article_attributes, derive_finding_attributes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "ilsa_survey_articles" / "json"


def _load_tabular_module():
    path = PROJECT_ROOT / "scripts" / "build_tabular_dataset.py"
    spec = importlib.util.spec_from_file_location("build_tabular_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> tuple[list[dict[str, Any]], list[str]]:
    tab = _load_tabular_module()
    paths = sorted(Path(corpus_dir).glob("*.json"))
    articles, warnings = tab.load_and_dedupe_articles(paths, Path(corpus_dir), dedupe=True, workers=1)
    articles.sort(key=lambda a: _article_number(a["_json_path"]))
    return articles, warnings


def _article_number(path: str) -> tuple[int, str]:
    name = os.path.basename(path)
    m = re.match(r"(\d+)", name)
    return (int(m.group(1)) if m else 10**6, name)


def article_id(article: dict[str, Any]) -> str:
    return os.path.basename(article["_json_path"]).rsplit(".json", 1)[0]


def _finding_performance(finding: dict[str, Any], study_models: list[str]) -> list[dict[str, Any]]:
    """Structured ``model_performance`` (schema v2) when present, else parse the free text."""
    structured = finding.get("model_performance") or []
    rows: list[dict[str, Any]] = []
    for item in structured:
        model = canonicalize_list([item.get("model")])
        metric = canonical_metric(str(item.get("metric") or ""))
        if not model or metric is None or item.get("value") is None:
            continue
        rows.append({
            "model": model[0].name,
            "model_family": model[0].family,
            "metric": metric[0],
            "value": float(item["value"]),
            "higher_is_better": metric[1],
            "attribution": "extracted_model_performance",
            "clause": str(item.get("evaluation_split") or ""),
        })
    if rows:
        return rows
    return parse_performance(finding.get("performance_metrics"), study_models=study_models)


def build_frames(articles: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Return article-, finding- and performance-level DataFrames."""
    art_rows: list[dict[str, Any]] = []
    find_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for art in articles:
        meta, data = art["metadata"], art["data"]
        aid = article_id(art)
        ml = data.get("ml_techniques") or {}
        raw_techniques = list(ml.get("all_techniques") or [])
        primary_raw = ml.get("primary")
        models = canonicalize_list(raw_techniques + ([primary_raw] if primary_raw else []))
        predictive = [m for m in models if m.role == "predictive"]
        primary = canonical_name(primary_raw)
        attrs = derive_article_attributes(meta, data)

        art_rows.append({
            "article_id": aid,
            "title": meta.get("title"),
            "year": meta.get("year"),
            "doi": meta.get("doi"),
            "source_category": meta.get("source_category"),
            "research_design_type": data.get("research_design_type"),
            "uses_ml": bool(predictive),
            "ml_primary": primary,
            "ml_predictive_models": "; ".join(m.name for m in predictive),
            "ml_other_techniques": "; ".join(m.name for m in models if m.role != "predictive"),
            "n_predictive_models": len(predictive),
            "ilsa_programs": "; ".join(attrs["ilsa_programs"]),
            "cycles": "; ".join(attrs["cycles"]),
            "domains": "; ".join(attrs["domains"]),
            "any_achievement_target": attrs["any_achievement_target"],
            "analysis_levels": "; ".join(attrs["analysis_levels"]),
            "task_types": "; ".join(attrs["task_types"]),
            "validation_scheme": attrs["validation_scheme"],
            "n_countries": len((data.get("sample_details") or {}).get("countries") or []),
        })
        for m in predictive:
            model_rows.append({
                "article_id": aid,
                "model": m.name,
                "family": m.family,
                "is_primary": m.name == primary,
            })

        for idx, f in enumerate(data.get("main_findings") or []):
            if not isinstance(f, dict):
                continue
            fa = derive_finding_attributes(f, meta=meta, data=data)
            fid = f"{aid}#{idx + 1}"
            find_rows.append({
                "finding_id": fid,
                "article_id": aid,
                "dataset_used": f.get("dataset_used"),
                "target_variable": f.get("target_variable"),
                "ilsa_programs": "; ".join(fa["ilsa_programs"]),
                "cycles": "; ".join(fa["cycles"]),
                "domain": fa["domain"],
                "target_type": fa["target_type"],
                "is_achievement": fa["is_achievement"],
                "analysis_level": fa["analysis_level"],
                "task_type": fa["task_type"],
                "validation_scheme": fa["validation_scheme"],
                "performance_metrics": f.get("performance_metrics"),
            })
            for r in _finding_performance(f, raw_techniques):
                perf_rows.append({"finding_id": fid, "article_id": aid, **r})

    return {
        "articles": pd.DataFrame(art_rows),
        "findings": pd.DataFrame(find_rows),
        "performance": pd.DataFrame(perf_rows),
        "article_models": pd.DataFrame(model_rows),
    }
