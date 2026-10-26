#!/usr/bin/env python3
"""
RQ1 (part A) — Which ML models are most frequently used to predict student
achievement in ILSA research, and how do Random Forest and XGBoost perform
relative to the other learners reported in the same studies?

Reads the 212-article corpus (ilsa_survey_articles/json, unchanged) and writes
outputs/rq1/:

  articles_coded.csv          one row per article with derived attributes
  findings_coded.csv          one row per main finding with derived attributes
  model_performance_long.csv  model × metric × value parsed from performance_metrics
  model_frequency.csv         model counts per analysis subset (mention / primary)
  family_frequency.csv        model-family counts per subset
  model_trend_by_year.csv     articles mentioning each top model per publication year
  rf_vs_xgb_head_to_head.csv  within-study RF vs XGBoost comparisons on the same metric
  pairwise_win_matrix.csv     within-study win rates between the most frequent models
  best_model_per_finding.csv  best model per finding on its priority metric
  country_level_studies.csv   findings whose target is a country-level outcome
  rq1_literature_summary.md   auto-generated summary of the tables above

Subsets
  all_ml            articles using ≥1 predictive ML model
  achievement_ml    empirical (peer_reviewed_research) articles using ML with ≥1
                    achievement target (score or proficiency category) — the RQ1 population
  achievement_ml|PISA, |TIMSS, |PIRLS, |Mathematics, |Reading, |Science — breakdowns
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.corpus import build_frames, load_corpus  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "rq1"

# One metric per finding for win/loss counting (avoids counting a study 5× for 5 metrics).
METRIC_PRIORITY = ["RMSE", "MAE", "R2", "AUC", "F1", "Balanced accuracy", "Accuracy", "Kappa", "MSE", "MAPE"]
FOCUS = ("Random Forest", "XGBoost")


def _subsets(articles: pd.DataFrame) -> dict[str, pd.Series]:
    ml = articles["uses_ml"]
    ach = ml & articles["any_achievement_target"] & (articles["source_category"] == "peer_reviewed_research")
    out = {"all_ml": ml, "achievement_ml": ach}
    for prog in ("PISA", "TIMSS", "PIRLS"):
        out[f"achievement_ml|{prog}"] = ach & articles["ilsa_programs"].str.contains(rf"\b{prog}\b", regex=True)
    for dom in ("Mathematics", "Reading", "Science"):
        out[f"achievement_ml|{dom}"] = ach & articles["domains"].str.contains(dom, regex=False)
    return out


def model_frequency(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    arts, am = frames["articles"], frames["article_models"]
    rows, fam_rows = [], []
    for name, mask in _subsets(arts).items():
        ids = set(arts.loc[mask, "article_id"])
        n = len(ids)
        sub = am[am["article_id"].isin(ids)]
        if sub.empty:
            continue
        grp = sub.groupby(["model", "family"]).agg(
            n_articles=("article_id", "nunique"),
            n_primary=("is_primary", "sum"),
        ).reset_index()
        grp["pct_articles"] = (100 * grp["n_articles"] / n).round(1)
        grp["pct_primary"] = (100 * grp["n_primary"] / n).round(1)
        grp["rank_mention"] = grp["n_articles"].rank(ascending=False, method="min").astype(int)
        grp["rank_primary"] = grp["n_primary"].rank(ascending=False, method="min").astype(int)
        grp.insert(0, "subset", name)
        grp.insert(1, "subset_n_articles", n)
        rows.append(grp.sort_values(["n_articles", "n_primary"], ascending=False))

        fam = sub.groupby("family")["article_id"].nunique().reset_index(name="n_articles")
        fam["pct_articles"] = (100 * fam["n_articles"] / n).round(1)
        fam.insert(0, "subset", name)
        fam.insert(1, "subset_n_articles", n)
        fam_rows.append(fam.sort_values("n_articles", ascending=False))
    return pd.concat(rows, ignore_index=True), pd.concat(fam_rows, ignore_index=True)


def trend_by_year(frames: dict[str, pd.DataFrame], top_n: int = 8) -> pd.DataFrame:
    arts, am = frames["articles"], frames["article_models"]
    top = am["model"].value_counts().head(top_n).index
    merged = am[am["model"].isin(top)].merge(arts[["article_id", "year"]], on="article_id")
    table = merged.pivot_table(index="year", columns="model", values="article_id", aggfunc="nunique", fill_value=0)
    totals = arts[arts["uses_ml"]].groupby("year")["article_id"].nunique().rename("n_ml_articles")
    return table.join(totals, how="outer").fillna(0).astype(int).reset_index()


def _per_model_values(perf: pd.DataFrame) -> pd.DataFrame:
    """Best reported value per (finding, metric, model)."""
    attributed = perf.dropna(subset=["model"])
    if attributed.empty:
        return attributed
    ranked = attributed.assign(
        _score=attributed["value"].where(attributed["higher_is_better"], -attributed["value"])
    ).sort_values("_score", ascending=False)
    return (
        ranked.drop_duplicates(subset=["finding_id", "metric", "model"])
        .drop(columns="_score")
        .reset_index(drop=True)
    )


def _priority_metric(values: pd.DataFrame) -> pd.DataFrame:
    """Keep, per finding, only the highest-priority metric that ≥2 models report."""
    keep = []
    for fid, g in values.groupby("finding_id"):
        counts = g.groupby("metric")["model"].nunique()
        for metric in METRIC_PRIORITY:
            if counts.get(metric, 0) >= 2:
                keep.append(g[g["metric"] == metric])
                break
    return pd.concat(keep, ignore_index=True) if keep else values.iloc[0:0]


def head_to_head(values: pd.DataFrame, findings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fid, metric), g in values.groupby(["finding_id", "metric"]):
        models = set(g["model"])
        if not set(FOCUS) <= models:
            continue
        rf = g[g["model"] == FOCUS[0]].iloc[0]
        xg = g[g["model"] == FOCUS[1]].iloc[0]
        hib = bool(rf["higher_is_better"])
        if rf["value"] == xg["value"]:
            winner = "tie"
        elif (rf["value"] > xg["value"]) == hib:
            winner = FOCUS[0]
        else:
            winner = FOCUS[1]
        diff = (xg["value"] - rf["value"]) * (1 if hib else -1)
        rows.append({
            "finding_id": fid,
            "metric": metric,
            "higher_is_better": hib,
            "rf_value": rf["value"],
            "xgb_value": xg["value"],
            "winner": winner,
            "xgb_advantage": round(diff, 6),
            "xgb_advantage_pct": round(100 * diff / abs(rf["value"]), 2) if rf["value"] else None,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.merge(
        findings[["finding_id", "ilsa_programs", "domain", "target_type", "is_achievement", "analysis_level"]],
        on="finding_id",
        how="left",
    )


def pairwise_matrix(values: pd.DataFrame, top_models: list[str]) -> pd.DataFrame:
    prio = _priority_metric(values)
    rows = []
    for a, b in combinations(top_models, 2):
        wins_a = wins_b = ties = 0
        for _fid, g in prio.groupby("finding_id"):
            ga, gb = g[g["model"] == a], g[g["model"] == b]
            if ga.empty or gb.empty:
                continue
            va, vb, hib = ga["value"].iloc[0], gb["value"].iloc[0], bool(ga["higher_is_better"].iloc[0])
            if va == vb:
                ties += 1
            elif (va > vb) == hib:
                wins_a += 1
            else:
                wins_b += 1
        n = wins_a + wins_b + ties
        if n:
            rows.append({"model_a": a, "model_b": b, "n_comparisons": n, "wins_a": wins_a,
                         "wins_b": wins_b, "ties": ties, "win_rate_a": round(wins_a / n, 3)})
    return pd.DataFrame(rows)


def best_model_per_finding(values: pd.DataFrame, findings: pd.DataFrame) -> pd.DataFrame:
    prio = _priority_metric(values)
    rows = []
    for fid, g in prio.groupby("finding_id"):
        hib = bool(g["higher_is_better"].iloc[0])
        best_val = g["value"].max() if hib else g["value"].min()
        winners = sorted(g.loc[g["value"] == best_val, "model"])
        rows.append({
            "finding_id": fid,
            "metric": g["metric"].iloc[0],
            "n_models_compared": g["model"].nunique(),
            "models_compared": "; ".join(sorted(g["model"].unique())),
            "best_model": "; ".join(winners),
            "best_value": best_val,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.merge(findings[["finding_id", "ilsa_programs", "domain", "is_achievement", "analysis_level"]],
                     on="finding_id", how="left")


def _md_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    sub = df[cols].head(n)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in sub.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_summary(out: Path, frames, freq, h2h, best, warnings) -> None:
    arts = frames["articles"]
    subsets = _subsets(arts)
    lines = [
        "# RQ1 (A) — ML model use and performance in the ILSA literature",
        "",
        "_Auto-generated by `scripts/rq1_literature_analysis.py`; every number below is computed from the CSVs in this folder._",
        "",
        "## Corpus",
        f"- Unique articles: **{len(arts)}** (from {len(list((PROJECT_ROOT / 'ilsa_survey_articles' / 'json').glob('*.json')))} JSON files; duplicate extractions collapsed by DOI + title).",
        f"- Articles using ≥1 predictive ML model: **{int(subsets['all_ml'].sum())}**.",
        f"- RQ1 population (empirical, ML, achievement target): **{int(subsets['achievement_ml'].sum())}**.",
        "",
    ]
    for line in warnings:
        if "CONFLICT" in line:
            lines.append(f"- ⚠ {line}")
    for subset in ("all_ml", "achievement_ml"):
        f = freq[freq["subset"] == subset]
        lines += ["", f"## Most frequent models — `{subset}` (n = {int(f['subset_n_articles'].iloc[0])})", "",
                  _md_table(f, ["model", "family", "n_articles", "pct_articles", "rank_mention", "n_primary", "rank_primary"])]
    if not h2h.empty:
        rf_w = int((h2h["winner"] == "Random Forest").sum())
        xg_w = int((h2h["winner"] == "XGBoost").sum())
        ties = int((h2h["winner"] == "tie").sum())
        n_find = h2h["finding_id"].nunique()
        ach = h2h[h2h["is_achievement"] == True]  # noqa: E712
        lines += [
            "",
            "## Random Forest vs XGBoost (same study, same target, same metric)",
            f"- All findings: **{len(h2h)}** metric pairs across **{n_find}** findings — "
            f"XGBoost better **{xg_w}**, Random Forest better **{rf_w}**, ties **{ties}**; "
            f"median relative XGBoost advantage **{h2h['xgb_advantage_pct'].median():.2f}%**.",
            f"- Achievement targets only: **{len(ach)}** pairs across **{ach['finding_id'].nunique()}** findings — "
            f"XGBoost better **{int((ach['winner'] == 'XGBoost').sum())}**, "
            f"Random Forest better **{int((ach['winner'] == 'Random Forest').sum())}**, "
            f"ties **{int((ach['winner'] == 'tie').sum())}**"
            + (f"; median relative XGBoost advantage **{ach['xgb_advantage_pct'].median():.2f}%**." if len(ach) else "."),
            "- These are student-level (mostly cross-sectional) results; they are not country-level forecasts — "
            "the forecasting comparison is produced by `scripts/rq1_forecast_comparison.py`.",
        ]
    if not best.empty:
        counts = best["best_model"].str.split("; ").explode().value_counts().head(10)
        lines += ["", "## Best model per finding (priority metric, ≥2 models compared)", ""]
        lines += [f"- {m}: {c}" for m, c in counts.items()]
    lines += [
        "",
        "## Caveats",
        "- Attributes (program, domain, target type, level, validation) are rule-derived from the LLM extraction; "
        "check agreement with manual coding via `scripts/validation_sample.py` before reporting.",
        "- Metric values are parsed from free text; rows with `attribution = unattributed` are excluded from comparisons.",
    ]
    (out / "rq1_literature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    articles, warnings = load_corpus()
    frames = build_frames(articles)
    frames["articles"].to_csv(out / "articles_coded.csv", index=False)
    frames["findings"].to_csv(out / "findings_coded.csv", index=False)
    frames["performance"].to_csv(out / "model_performance_long.csv", index=False)

    freq, fam = model_frequency(frames)
    freq.to_csv(out / "model_frequency.csv", index=False)
    fam.to_csv(out / "family_frequency.csv", index=False)
    trend_by_year(frames).to_csv(out / "model_trend_by_year.csv", index=False)

    values = _per_model_values(frames["performance"])
    h2h = head_to_head(values, frames["findings"])
    h2h.to_csv(out / "rf_vs_xgb_head_to_head.csv", index=False)
    top_models = list(freq[freq["subset"] == "all_ml"].head(8)["model"])
    pairwise_matrix(values, top_models).to_csv(out / "pairwise_win_matrix.csv", index=False)
    best = best_model_per_finding(values, frames["findings"])
    best.to_csv(out / "best_model_per_finding.csv", index=False)
    country = frames["findings"][frames["findings"]["analysis_level"] == "country"]
    country.to_csv(out / "country_level_studies.csv", index=False)

    write_summary(out, frames, freq, h2h, best, warnings)
    print(f"Wrote RQ1 literature tables to {out}")
    print((out / "rq1_literature_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
