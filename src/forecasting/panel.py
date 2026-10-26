"""
Country-level ILSA panel: loading, validation and supervised feature rows.

Panel file (``data/country_panel/ilsa_country_means.csv``), one row per
program × domain × country × cycle:

    program        PISA | TIMSS | PIRLS | …
    domain         Mathematics | Reading | Science | …
    grade          optional (e.g. 4 / 8 for TIMSS); part of the series key when present
    country_code   ISO 3166-1 alpha-3
    year           assessment cycle year
    mean_score     country mean on the reporting scale
    se             optional standard error of the mean

Covariate file (``data/country_panel/ilsa_country_covariates.csv``), optional, wide:

    country_code, year, <covariate_1>, <covariate_2>, …

Covariates are attached *as of* the forecast origin (latest value with
year ≤ origin year, at most ``max_covariate_lag`` years old), so a forecast never
uses information from after its origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["program", "domain", "country_code", "year", "mean_score"]
SERIES_KEY = ["program", "domain", "grade", "country_code"]
GROUP_KEY = ["program", "domain", "grade"]


def load_panel(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Panel {path} lacks required columns: {missing}")
    if "grade" not in df.columns:
        df["grade"] = "all"
    df["grade"] = df["grade"].fillna("all").astype(str)
    df["country_code"] = df["country_code"].astype(str).str.upper().str.strip()
    df["year"] = df["year"].astype(int)
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
    df = df.dropna(subset=["mean_score"])
    dupes = df.duplicated(subset=SERIES_KEY + ["year"], keep=False)
    if dupes.any():
        raise ValueError(
            "Panel has duplicate program/domain/grade/country/year rows:\n"
            + df.loc[dupes, SERIES_KEY + ["year"]].head(10).to_string()
        )
    return df.sort_values(SERIES_KEY + ["year"]).reset_index(drop=True)


def load_covariates(path: Path | str | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    cov = pd.read_csv(path)
    if not {"country_code", "year"} <= set(cov.columns):
        raise ValueError("Covariate file needs country_code and year columns")
    cov["country_code"] = cov["country_code"].astype(str).str.upper().str.strip()
    cov["year"] = cov["year"].astype(int)
    return cov.sort_values(["country_code", "year"]).reset_index(drop=True)


def covariates_asof(
    cov: pd.DataFrame | None,
    keys: pd.DataFrame,
    columns: list[str],
    *,
    max_lag: int = 3,
) -> pd.DataFrame:
    """Latest covariate value with year ≤ keys.year (and ≥ year − max_lag) per row of ``keys``."""
    out = pd.DataFrame(index=keys.index, columns=columns, dtype=float)
    if cov is None or not columns:
        return out
    for col in columns:
        series = cov[["country_code", "year", col]].dropna()
        left = keys[["country_code", "year"]].reset_index().sort_values("year")
        merged = pd.merge_asof(
            left,
            series.sort_values("year").rename(columns={"year": "cov_year"}),
            left_on="year",
            right_on="cov_year",
            by="country_code",
            direction="backward",
            tolerance=max_lag,
        ).set_index("index")
        out[col] = merged[col].reindex(keys.index).astype(float)
    return out


@dataclass
class SupervisedRows:
    """One row per consecutive cycle pair (origin → target) of every series."""

    frame: pd.DataFrame
    feature_columns: list[str]
    covariate_columns: list[str]


def build_supervised(
    panel: pd.DataFrame,
    cov: pd.DataFrame | None = None,
    covariate_columns: list[str] | None = None,
    *,
    max_covariate_lag: int = 3,
) -> SupervisedRows:
    """Features known at the origin cycle; target = change to the next cycle."""
    covariate_columns = list(covariate_columns or [])
    rows = []
    for key, g in panel.groupby(SERIES_KEY, sort=False):
        g = g.sort_values("year")
        years = g["year"].to_numpy()
        ys = g["mean_score"].to_numpy(dtype=float)
        for i in range(len(g)):
            hist = ys[: i + 1]
            row = dict(zip(SERIES_KEY, key))
            row.update({
                "origin_year": int(years[i]),
                "prev_year": int(years[i - 1]) if i >= 1 else np.nan,
                "target_year": int(years[i + 1]) if i + 1 < len(g) else np.nan,
                "y_origin": hist[-1],
                "y_lag1": hist[-2] if len(hist) >= 2 else np.nan,
                "y_lag2": hist[-3] if len(hist) >= 3 else np.nan,
                "d1": hist[-1] - hist[-2] if len(hist) >= 2 else np.nan,
                "d2": hist[-2] - hist[-3] if len(hist) >= 3 else np.nan,
                "hist_mean": float(hist.mean()),
                "dev_from_mean": hist[-1] - float(hist.mean()),
                "n_hist": len(hist),
                "gap_years": (years[i + 1] - years[i]) if i + 1 < len(g) else np.nan,
                "y_target": ys[i + 1] if i + 1 < len(g) else np.nan,
            })
            rows.append(row)
    df = pd.DataFrame(rows)
    df["target_change"] = df["y_target"] - df["y_origin"]

    # Cross-country context known at the origin: mean level and mean last change of the group.
    grp = df.groupby(GROUP_KEY + ["origin_year"])
    df["group_mean_origin"] = grp["y_origin"].transform("mean")
    df["group_mean_d1"] = grp["d1"].transform("mean")
    df["rel_to_group"] = df["y_origin"] - df["group_mean_origin"]

    feature_columns = [
        "y_origin", "y_lag1", "y_lag2", "d1", "d2", "hist_mean", "dev_from_mean",
        "n_hist", "gap_years", "group_mean_d1", "rel_to_group",
    ]
    for col, prefix in (("program", "prog_"), ("domain", "dom_")):
        dummies = pd.get_dummies(df[col], prefix=prefix.rstrip("_"), dtype=float)
        df = pd.concat([df, dummies], axis=1)
        feature_columns += list(dummies.columns)

    if covariate_columns:
        keys = df[["country_code"]].assign(year=df["origin_year"])
        now = covariates_asof(cov, keys, covariate_columns, max_lag=max_covariate_lag)
        # Change since the previous cycle (3 years back when the series has no earlier cycle).
        prev_year = df["prev_year"].fillna(df["origin_year"] - 3).astype(int)
        prev_keys = df[["country_code"]].assign(year=prev_year)
        prev = covariates_asof(cov, prev_keys, covariate_columns, max_lag=max_covariate_lag)
        for col in covariate_columns:
            df[f"cov_{col}"] = now[col].to_numpy()
            df[f"dcov_{col}"] = (now[col] - prev[col]).to_numpy()
    return SupervisedRows(df, feature_columns, covariate_columns)
