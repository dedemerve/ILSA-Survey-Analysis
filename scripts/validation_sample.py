#!/usr/bin/env python3
"""
Human validation of the automated coding (LLM extraction + rule-derived attributes).

Step 1 — draw a coding sheet:
    python scripts/validation_sample.py draw --n 30
  writes outputs/validation/coding_sheet.csv: one row per sampled finding with the
  automatic codes (auto_*) and empty manual_* columns. Two coders fill manual_* from
  the PDFs independently (copy the file per coder: coding_sheet_coderA.csv, …).

Step 2 — score agreement:
    python scripts/validation_sample.py score outputs/validation/coding_sheet_coderA.csv \
        [outputs/validation/coding_sheet_coderB.csv]
  writes outputs/validation/agreement.csv with, per field, percent agreement and
  Cohen's κ between automatic and manual codes (and between the two coders when a
  second sheet is given).

The sample is stratified by uses_ml × achievement so both RQ1-relevant and
excluded studies are checked. Seeded for reproducibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.corpus import build_frames, load_corpus  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "validation"
FIELDS = [
    "uses_ml", "ml_primary", "ilsa_programs", "domain", "target_type", "is_achievement",
    "analysis_level", "task_type", "validation_scheme", "rf_metric_value", "xgb_metric_value",
]


def _metric_value(perf: pd.DataFrame, fid: str, model: str) -> str:
    rows = perf[(perf["finding_id"] == fid) & (perf["model"] == model)]
    return "; ".join(f"{r.metric}={r.value:g}" for r in rows.itertuples()) if not rows.empty else ""


def draw(n: int, seed: int) -> Path:
    articles, _ = load_corpus()
    frames = build_frames(articles)
    arts, finds, perf = frames["articles"], frames["findings"], frames["performance"]
    finds = finds.merge(arts[["article_id", "uses_ml", "ml_primary", "title"]], on="article_id")
    finds["stratum"] = finds["uses_ml"].astype(str) + "|" + finds["is_achievement"].astype(str)
    one_per_article = finds.groupby("article_id", group_keys=False).sample(n=1, random_state=seed)
    per_stratum = max(1, n // one_per_article["stratum"].nunique())
    sample = (
        one_per_article.groupby("stratum", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_stratum), random_state=seed))
    )
    if len(sample) < n:
        rest = one_per_article.drop(sample.index)
        sample = pd.concat([sample, rest.sample(n=min(len(rest), n - len(sample)), random_state=seed)])
    sheet = pd.DataFrame({
        "finding_id": sample["finding_id"],
        "title": sample["title"],
        "dataset_used": sample["dataset_used"],
        "target_variable": sample["target_variable"],
        "performance_metrics": sample["performance_metrics"],
        "auto_uses_ml": sample["uses_ml"],
        "auto_ml_primary": sample["ml_primary"],
        "auto_ilsa_programs": sample["ilsa_programs"],
        "auto_domain": sample["domain"],
        "auto_target_type": sample["target_type"],
        "auto_is_achievement": sample["is_achievement"],
        "auto_analysis_level": sample["analysis_level"],
        "auto_task_type": sample["task_type"],
        "auto_validation_scheme": sample["validation_scheme"],
        "auto_rf_metric_value": [_metric_value(perf, f, "Random Forest") for f in sample["finding_id"]],
        "auto_xgb_metric_value": [_metric_value(perf, f, "XGBoost") for f in sample["finding_id"]],
    })
    for f in FIELDS:
        sheet[f"manual_{f}"] = ""
    sheet["coder_notes"] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "coding_sheet.csv"
    sheet.sort_values("finding_id").to_csv(path, index=False)
    return path


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _norm(v) -> str:
    s = "" if pd.isna(v) else str(v).strip().lower()
    return "; ".join(sorted(p.strip() for p in s.split(";") if p.strip()))


def score(sheet_a: Path, sheet_b: Path | None) -> Path:
    a = pd.read_csv(sheet_a)
    b = pd.read_csv(sheet_b) if sheet_b else None
    rows = []
    for f in FIELDS:
        coded = a[a[f"manual_{f}"].notna() & (a[f"manual_{f}"].astype(str).str.strip() != "")]
        auto = [_norm(v) for v in coded[f"auto_{f}"]]
        man = [_norm(v) for v in coded[f"manual_{f}"]]
        row = {
            "field": f,
            "n_coded": len(coded),
            "agreement_auto_vs_A": round(sum(x == y for x, y in zip(auto, man)) / len(man), 3) if man else None,
            "kappa_auto_vs_A": round(cohen_kappa(auto, man), 3) if man else None,
        }
        if b is not None:
            merged = coded[["finding_id", f"manual_{f}"]].merge(
                b[["finding_id", f"manual_{f}"]], on="finding_id", suffixes=("_a", "_b")
            ).dropna()
            ma = [_norm(v) for v in merged[f"manual_{f}_a"]]
            mb = [_norm(v) for v in merged[f"manual_{f}_b"]]
            row["n_double_coded"] = len(ma)
            row["kappa_A_vs_B"] = round(cohen_kappa(ma, mb), 3) if ma else None
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "agreement.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--n", type=int, default=30)
    d.add_argument("--seed", type=int, default=2026)
    s = sub.add_parser("score")
    s.add_argument("sheet_a", type=Path)
    s.add_argument("sheet_b", type=Path, nargs="?")
    args = parser.parse_args()
    path = draw(args.n, args.seed) if args.cmd == "draw" else score(args.sheet_a, args.sheet_b)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
