#!/usr/bin/env python3
"""
RQ2 — Does integrating the literature knowledge base improve country-level
forecasts of mean achievement?

The knowledge base (outputs/knowledge_base/country_covariate_priors.json, built by
scripts/build_predictor_knowledge_base.py) is integrated in three ways and each
is compared with the same model without it (ablation, rolling-origin evaluation):

  1. Selection   — use only covariates the literature supports (≥ --min-kb-studies studies)
  2. Constraints — XGBoost monotone constraints / sign-bounded panel coefficients
                   from the literature's effect direction
  3. Prior strength — weaker ridge shrinkage for better-supported covariates (panel model)

Variants
  *_lags           no covariates (RQ1 baseline)
  *_allcov         every covariate in the file (data-driven, no KB)
  *_kb_select      KB-selected covariates
  *_kb_constrained KB-selected covariates + KB signs (XGB monotone / panel sign bounds + prior strength)
  *_random_k_sN    control: random covariate subsets of the same size as the KB selection

Input  data/country_panel/ilsa_country_means.csv, data/country_panel/ilsa_country_covariates.csv
Output outputs/rq2_kb_forecast/ (predictions, accuracy, DM tests vs *_lags and *_allcov, summary)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.evaluation import accuracy, common_points, dm_table, rolling_origin  # noqa: E402
from src.forecasting.models import LOCAL_MODELS, PanelRegression, XGBoostChange  # noqa: E402
from src.forecasting.panel import build_supervised, load_covariates, load_panel  # noqa: E402

DATA = PROJECT_ROOT / "data" / "country_panel"
DEFAULT_PRIORS = PROJECT_ROOT / "outputs" / "knowledge_base" / "country_covariate_priors.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "rq2_kb_forecast"
LAG_FEATURES = ["d1", "dev_from_mean", "group_mean_d1", "rel_to_group", "gap_years"]


def cov_features(cols: list[str]) -> list[str]:
    return [f for c in cols for f in (f"cov_{c}", f"dcov_{c}")]


def build_variants(
    base_features: list[str],
    covariates: list[str],
    priors: dict[str, dict],
    *,
    min_kb_studies: int,
    n_random: int,
    seed: int,
) -> tuple[dict, dict]:
    supported = [
        c for c in covariates
        if priors.get(c, {}).get("supported") and priors[c].get("n_studies_top", 0) >= min_kb_studies
    ]
    signs = {c: int(priors[c].get("sign", 0)) for c in supported}
    weights = {c: float(priors[c].get("weight", 0.0)) for c in supported}

    def sign_map(cols: list[str]) -> dict[str, int]:
        # Level and change of a covariate share the literature's sign.
        return {f: signs[c] for c in cols for f in (f"cov_{c}", f"dcov_{c}") if signs.get(c)}

    def penalty_map(cols: list[str]) -> dict[str, float]:
        return {f: 1.0 - 0.9 * weights.get(c, 0.0) for c in cols for f in (f"cov_{c}", f"dcov_{c}")}

    v: dict = {}
    for tag, cols in (("lags", []), ("allcov", covariates), ("kb_select", supported)):
        v[f"XGB_{tag}"] = (lambda cols: lambda tr: XGBoostChange(f"XGB_{tag}", base_features + cov_features(cols), random_state=seed))(cols)
        v[f"Panel_{tag}"] = (lambda cols: lambda tr: PanelRegression(f"Panel_{tag}", LAG_FEATURES + cov_features(cols)))(cols)
    v["XGB_kb_constrained"] = lambda tr: XGBoostChange(
        "XGB_kb_constrained", base_features + cov_features(supported), monotone=sign_map(supported), random_state=seed)
    v["Panel_kb_constrained"] = lambda tr: PanelRegression(
        "Panel_kb_constrained", LAG_FEATURES + cov_features(supported),
        signs=sign_map(supported), penalty_scale=penalty_map(supported))

    rng = np.random.default_rng(seed)
    k = len(supported)
    if 0 < k < len(covariates):
        for s in range(n_random):
            pick = sorted(rng.choice(covariates, size=k, replace=False).tolist())
            v[f"XGB_random_k{k}_s{s}"] = (lambda cols: lambda tr: XGBoostChange("XGB_random", base_features + cov_features(cols), random_state=seed))(pick)
            v[f"Panel_random_k{k}_s{s}"] = (lambda cols: lambda tr: PanelRegression("Panel_random", LAG_FEATURES + cov_features(cols)))(pick)

    info = {
        "covariates_in_panel": covariates,
        "kb_selected": supported,
        "kb_signs": signs,
        "kb_weights": weights,
        "not_in_kb_map": [c for c in covariates if c not in priors],
    }
    return v, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=DATA / "ilsa_country_means.csv")
    parser.add_argument("--covariates", type=Path, default=DATA / "ilsa_country_covariates.csv")
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    parser.add_argument("--kb-scope", default="all", help="Prior scope: all | program=PISA | domain=Mathematics …")
    parser.add_argument("--min-kb-studies", type=int, default=3)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--max-covariate-lag", type=int, default=3)
    parser.add_argument("--min-train-cycles", type=int, default=3)
    parser.add_argument("--min-train-rows", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for p, hint in ((args.panel, "data/country_panel/README.md"), (args.covariates, "scripts/fetch_country_covariates.py"),
                    (args.priors, "scripts/build_predictor_knowledge_base.py")):
        if not p.exists():
            sys.exit(f"Missing input {p} — see {hint}.")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.panel)
    cov = load_covariates(args.covariates)
    covariates = [c for c in cov.columns if c not in ("country_code", "year")]
    priors_all = json.loads(args.priors.read_text(encoding="utf-8"))["priors"]
    if args.kb_scope not in priors_all:
        sys.exit(f"Scope {args.kb_scope!r} not in priors; available: {sorted(priors_all)}")
    priors = priors_all[args.kb_scope]

    sup = build_supervised(panel, cov, covariates, max_covariate_lag=args.max_covariate_lag)
    variants, info = build_variants(
        sup.feature_columns, covariates, priors,
        min_kb_studies=args.min_kb_studies, n_random=args.n_random, seed=args.seed,
    )
    (out / "kb_integration_config.json").write_text(json.dumps({"scope": args.kb_scope, **info}, indent=2) + "\n")

    preds = rolling_origin(
        panel, sup.frame,
        local_models={"Naive": LOCAL_MODELS["Naive"]},
        global_models=variants,
        min_train_cycles=args.min_train_cycles,
        min_train_rows=args.min_train_rows,
    )
    preds.to_csv(out / "predictions.csv", index=False)
    fair = common_points(preds)
    overall = accuracy(fair)
    overall.to_csv(out / "accuracy_overall.csv", index=False)
    accuracy(fair, ["program", "domain", "grade"]).to_csv(out / "accuracy_by_program_domain.csv", index=False)

    pairs = []
    for fam in ("XGB", "Panel"):
        for tag in ("kb_select", "kb_constrained"):
            pairs += [(f"{fam}_{tag}", f"{fam}_lags"), (f"{fam}_{tag}", f"{fam}_allcov")]
    dm = dm_table(fair, pairs)
    dm.to_csv(out / "dm_tests.csv", index=False)

    rand = overall[overall["model"].str.contains("_random_")]
    rand_summary = (
        rand.assign(family=rand["model"].str.split("_").str[0])
        .groupby("family")[["RMSE", "MAE"]].agg(["mean", "min", "max"])
        if not rand.empty else pd.DataFrame()
    )
    rand_summary.to_csv(out / "random_selection_control.csv")

    lines = [
        "# RQ2 — Knowledge-base integration ablation",
        "",
        "_Auto-generated by `scripts/rq2_kb_forecast_ablation.py`._",
        "",
        f"- KB scope: `{args.kb_scope}`; covariates in panel: {', '.join(covariates)}",
        f"- KB-selected (≥ {args.min_kb_studies} studies): {', '.join(info['kb_selected']) or 'none'}",
        f"- KB signs: {info['kb_signs']}",
        "",
        "## Accuracy (common forecast points)",
        "",
        "| model | n | RMSE | MAE | RelMAE_vs_naive |",
        "|---|---|---|---|---|",
    ]
    for _, r in overall[~overall["model"].str.contains("_random_")].iterrows():
        lines.append(f"| {r['model']} | {int(r['n_forecasts'])} | {r['RMSE']:.3f} | {r['MAE']:.3f} | {r['RelMAE_vs_naive']:.3f} |")
    if not rand_summary.empty:
        lines += ["", "Random-selection control (same number of covariates as the KB selection): "
                  "see `random_selection_control.csv`."]
        for fam, r in rand_summary.iterrows():
            lines.append(f"- {fam}: RMSE mean {r[('RMSE', 'mean')]:.3f} (min {r[('RMSE', 'min')]:.3f}, max {r[('RMSE', 'max')]:.3f})")
    lines += ["", "## Diebold–Mariano tests (negative mean = KB variant better)", "",
              "| model_a | model_b | n | mean_loss_diff | t | p_value |", "|---|---|---|---|---|---|"]
    for _, r in dm.iterrows():
        lines.append(f"| {r['model_a']} | {r['model_b']} | {r['n']} | {r['mean_loss_diff']:.3f} | {r['t']:.3f} | {r['p_value']:.3f} |")
    (out / "rq2_kb_forecast_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
