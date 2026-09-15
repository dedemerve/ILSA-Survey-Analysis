#!/usr/bin/env python3
"""Build Hugging Face-ready package for the N=212 ILSA survey corpus."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "ilsa_survey_articles" / "json"
OUT = ROOT / "outputs" / "huggingface_n212"
SCORES_CSV = ROOT / "rescored_final_csv_212.csv"


def _load_taxonomy():
    spec = importlib.util.spec_from_file_location(
        "academic_taxonomy_standalone",
        ROOT / "src" / "enrichment" / "academic_taxonomy.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["academic_taxonomy_standalone"] = mod
    spec.loader.exec_module(mod)
    return mod


taxonomy = _load_taxonomy()


def _join_list(values: Any, sep: str = ", ") -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return values.strip() or None
    if isinstance(values, list):
        parts = [str(v).strip() for v in values if v is not None and str(v).strip()]
        return sep.join(parts) if parts else None
    return str(values)


def _format_countries(countries: Any) -> str | None:
    if not countries:
        return None
    parts: list[str] = []
    for item in countries:
        if not isinstance(item, dict):
            continue
        code = str(item.get("country_code") or "").strip()
        n = item.get("n_students")
        if code and n is not None and str(n).strip() != "":
            parts.append(f"{code} ({n})")
        elif code:
            parts.append(code)
    return "; ".join(parts) if parts else None


def _doi_url(doi: str | None) -> str:
    if not doi or not str(doi).strip():
        return ""
    d = str(doi).strip()
    if d.startswith("http"):
        return d
    return f"https://doi.org/{d}"


def _first_finding_fields(findings: list[dict[str, Any]]) -> tuple[str, str]:
    if not findings:
        return "", ""
    f0 = findings[0] if isinstance(findings[0], dict) else {}
    return (
        str(f0.get("performance_metrics") or ""),
        str(f0.get("standardized_conclusion") or ""),
    )


def _confounder_names(confounders: list[dict[str, Any]]) -> str:
    names = []
    for c in confounders:
        if isinstance(c, dict) and c.get("variable_name"):
            names.append(str(c["variable_name"]).strip())
    return "; ".join(names)


def load_articles() -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    for path in sorted(JSON_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "metadata" not in raw or "data" not in raw:
            continue
        articles.append(raw)
    return articles


def build_raw_frames(articles: list[dict[str, Any]]):
    master_rows: list[dict[str, Any]] = []
    findings_rows: list[dict[str, Any]] = []
    confounders_rows: list[dict[str, Any]] = []

    for art in articles:
        meta = art.get("metadata") or {}
        data = art.get("data") or {}
        survey = data.get("survey_design") or {}
        sample = data.get("sample_details") or {}
        ml = data.get("ml_techniques") or {}
        findings = data.get("main_findings") or []
        confounders = data.get("confounders_identified") or []
        if not isinstance(findings, list):
            findings = []
        if not isinstance(confounders, list):
            confounders = []

        effect, primary = _first_finding_fields(findings)
        master_rows.append(
            {
                "doi": meta.get("doi"),
                "title": meta.get("title"),
                "authors": _join_list(meta.get("authors"), sep="; "),
                "year": meta.get("year"),
                "source_category": meta.get("source_category"),
                "journal": meta.get("venue"),
                "student_weights_used": survey.get("student_weights_used"),
                "replicate_weights_used": survey.get("replicate_weights_used"),
                "weight_variable_name": survey.get("weight_variable_name"),
                "weight_fields_interpretation": survey.get("weight_fields_interpretation"),
                "plausible_values_handling": data.get("plausible_values_handling"),
                "missing_data_handling": data.get("missing_data_handling"),
                "handling_not_reported_explanation": data.get(
                    "handling_not_reported_explanation"
                ),
                "research_design_type": data.get("research_design_type"),
                "outcome_summary": data.get("outcome_summary"),
                "null_fields_interpretation": data.get("null_fields_interpretation"),
                "ml_primary": ml.get("primary"),
                "ml_all_techniques": _join_list(ml.get("all_techniques")),
                "total_students": sample.get("total_students"),
                "sample_filtering_criteria": sample.get("sample_filtering_criteria"),
                "countries_formatted": _format_countries(sample.get("countries")),
                "effect_size": effect,
                "primary_finding": primary,
                "ml_techniques": ml.get("primary") or _join_list(ml.get("all_techniques")),
                "sample_size": sample.get("total_students"),
                "confounders": _confounder_names(confounders),
                "paper_url": _doi_url(meta.get("doi")),
                "publication_type": meta.get("publication_type"),
                "file_name": meta.get("file_name"),
            }
        )

        for finding in findings:
            if not isinstance(finding, dict):
                continue
            findings_rows.append(
                {
                    "doi": meta.get("doi"),
                    "dataset_used": finding.get("dataset_used"),
                    "target_variable": finding.get("target_variable"),
                    "top_predictors": _join_list(finding.get("top_predictors")),
                    "performance_metrics": finding.get("performance_metrics"),
                    "standardized_conclusion": finding.get("standardized_conclusion"),
                    "source_category": meta.get("source_category"),
                    "effect_size": finding.get("performance_metrics"),
                    "primary_finding": finding.get("standardized_conclusion"),
                    "file_name": meta.get("file_name"),
                }
            )

        for conf in confounders:
            if not isinstance(conf, dict):
                continue
            confounders_rows.append(
                {
                    "doi": meta.get("doi"),
                    "variable_code": conf.get("variable_code"),
                    "variable_name": conf.get("variable_name"),
                    "category": conf.get("category"),
                    "source_category": meta.get("source_category"),
                    "file_name": meta.get("file_name"),
                }
            )

    return (
        pd.DataFrame(master_rows),
        pd.DataFrame(findings_rows),
        pd.DataFrame(confounders_rows),
    )


ARTICLES_COLS = [
    "doi",
    "title",
    "authors",
    "year",
    "source_category",
    "journal",
    "student_weights_used",
    "replicate_weights_used",
    "weight_variable_name",
    "weight_fields_interpretation",
    "plausible_values_handling",
    "missing_data_handling",
    "handling_not_reported_explanation",
    "research_design_type",
    "outcome_summary",
    "null_fields_interpretation",
    "ml_primary",
    "ml_all_techniques",
    "total_students",
    "sample_filtering_criteria",
    "countries_formatted",
    "effect_size",
    "primary_finding",
    "ml_techniques",
    "sample_size",
    "confounders",
    "ml_family",
    "pv_filter_label",
    "md_filter_label",
    "weights_filter",
    "paper_url",
]

FINDINGS_COLS = [
    "doi",
    "dataset_used",
    "target_variable",
    "top_predictors",
    "performance_metrics",
    "standardized_conclusion",
    "source_category",
    "effect_size",
    "primary_finding",
    "target_domain",
    "target_dimension",
    "predictor_filter_categories",
]

CONFOUNDERS_COLS = [
    "doi",
    "variable_code",
    "variable_name",
    "category",
    "source_category",
    "predictor_level",
    "predictor_category",
]


from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


def _clean_excel_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v)
    return ILLEGAL_CHARACTERS_RE.sub("", s)


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    for c in cols:
        out[c] = out[c].map(_clean_excel_str)
    return out[cols]


def write_readme(path: Path, n_articles: int, n_findings: int, n_confounders: int) -> None:
    text = f"""---
license: cc-by-4.0
language:
- en
tags:
- education
- machine-learning
- systematic-review
- PISA
- TIMSS
- PIRLS
- ILSA
- survey
pretty_name: AI & ML in International Large-Scale Assessments — Survey Dataset
size_categories:
- 1K<n<10K
configs:
- config_name: articles_master
  data_files:
  - path: data/articles_master.csv
    split: train
- config_name: main_findings
  data_files:
  - path: data/main_findings.csv
    split: train
- config_name: confounders
  data_files:
  - path: data/confounders.csv
    split: train
- config_name: actionability_scores
  data_files:
  - path: data/actionability_scores.csv
    split: train
---

# AI & ML in International Large-Scale Assessments: Survey Dataset

**Paper:** *Artificial Intelligence Applications in International Large-Scale Assessments: A Survey with LLM-Assisted Evidence Synthesis*  
**Authors:** Merve Dede & Ekrem Çetinkaya (2026)  
**Website:** [dedemerve.github.io/ILSA-Survey-Extractor](https://dedemerve.github.io/ILSA-Survey-Extractor/)  
**GitHub:** [github.com/dedemerve/ILSA-Survey-Analysis](https://github.com/dedemerve/ILSA-Survey-Analysis)

## Dataset Description

A structured, open evidence repository covering **{n_articles} studies** (2020–August 2026) examining how artificial intelligence and machine learning methods are applied to international large-scale assessment (ILSA) data (PISA, TIMSS, PIRLS, TALIS, PIAAC, ICCS, ICILS).

Each full-text article was processed with a schema-constrained LLM-assisted extraction pipeline and **fully human-verified**. The release includes three relational tables, policy-actionability scores, and per-article JSON extracts.

| Table | Records | Description |
|---|---:|---|
| `articles_master` | {n_articles} | One row per study (metadata, ML methods, survey design, quality indicators) |
| `main_findings` | {n_findings} | One row per extracted finding |
| `confounders` | {n_confounders} | One row per predictor–study pair (13 taxonomy categories) |
| `actionability_scores` | {n_articles} | D₁–D₃ dimensions and composite score (1–5) |

**Corpus summary:** mean policy-actionability score **3.70** (SD = 0.80); **85.8%** of studies score 4–5. Synthesis uses a five-category framework (predictive modelling, process data / LA, socio-emotional modelling, assessment engineering, computational psychometrics).

## Files

- `data/articles_master.csv`
- `data/main_findings.csv`
- `data/confounders.csv`
- `data/actionability_scores.csv`
- `ILSA_Survey_Dataset_CLEAN.xlsx`
- `json_extracted/` — per-article verified JSON extracts (n = {n_articles})

## Citation

```bibtex
@article{{dede_cetinkaya2026ilsa_survey,
  title   = {{Artificial Intelligence Applications in International Large-Scale Assessments: A Survey with LLM-Assisted Evidence Synthesis}},
  author  = {{Dede, Merve and Çetinkaya, Ekrem}},
  year    = {{2026}},
  note    = {{Open dataset: HuggingFace Datasets}},
  url     = {{https://huggingface.co/datasets/dedemerve/ILSA-Survey-Dataset}}
}}
```

## License

CC BY 4.0. Metadata extraction is provided for research reuse; original article copyrights remain with their publishers.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    data_dir = OUT / "data"
    data_dir.mkdir()

    articles = load_articles()
    print(f"Loaded {len(articles)} articles")
    df_m, df_f, df_c = build_raw_frames(articles)
    df_m, df_f, df_c = taxonomy.apply_academic_taxonomy(df_m, df_f, df_c)

    articles_csv = _ensure_cols(df_m, ARTICLES_COLS)
    findings_csv = _ensure_cols(df_f, FINDINGS_COLS)
    confounders_csv = _ensure_cols(df_c, CONFOUNDERS_COLS)

    articles_csv.to_csv(data_dir / "articles_master.csv", index=False)
    findings_csv.to_csv(data_dir / "main_findings.csv", index=False)
    confounders_csv.to_csv(data_dir / "confounders.csv", index=False)

    if SCORES_CSV.exists():
        shutil.copy2(SCORES_CSV, data_dir / "actionability_scores.csv")

    xlsx_path = OUT / "ILSA_Survey_Dataset_CLEAN.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        articles_csv.to_excel(writer, sheet_name="1_Articles_Master", index=False)
        findings_csv.to_excel(writer, sheet_name="2_Main_Findings", index=False)
        confounders_csv.to_excel(writer, sheet_name="3_Confounders", index=False)
        if SCORES_CSV.exists():
            pd.read_csv(SCORES_CSV).to_excel(
                writer, sheet_name="4_Actionability_Scores", index=False
            )

    json_dir = OUT / "json_extracted"
    json_dir.mkdir()
    for src in sorted(JSON_DIR.glob("*.json")):
        shutil.copy2(src, json_dir / src.name)

    write_readme(
        OUT / "README.md",
        n_articles=len(articles_csv),
        n_findings=len(findings_csv),
        n_confounders=len(confounders_csv),
    )
    print(
        f"Wrote {OUT}\n"
        f"  articles={len(articles_csv)} findings={len(findings_csv)} "
        f"confounders={len(confounders_csv)} json={len(list(json_dir.glob('*.json')))}"
    )


if __name__ == "__main__":
    main()
