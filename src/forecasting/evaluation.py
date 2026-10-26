"""
Rolling-origin evaluation of next-cycle forecasts.

For every series group (program × domain × grade) and every origin cycle ``o``
with at least ``min_train_cycles`` cycles up to ``o``:

* local models get the country's history ``year ≤ o``;
* global models are trained on all supervised rows whose *target* year is ``≤ o``
  (pooled over the whole panel, so ML models see every program/domain), then
  predict the change from ``o`` to the next cycle.

No information after the origin enters a forecast. Accuracy is summarised by
RMSE, MAE, bias and MAE relative to the naive forecast; models are compared with a
Diebold–Mariano-type test on squared-error differentials with standard errors
clustered by country (loss differentials of one country across origins are not
independent).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.forecasting.models import GlobalModel, LocalForecast
from src.forecasting.panel import GROUP_KEY, SERIES_KEY

GlobalFactory = Callable[[pd.DataFrame], GlobalModel]


def rolling_origin(
    panel: pd.DataFrame,
    supervised: pd.DataFrame,
    *,
    local_models: dict[str, LocalForecast],
    global_models: dict[str, GlobalFactory],
    min_train_cycles: int = 3,
    min_train_rows: int = 30,
    origins: list[int] | None = None,
) -> pd.DataFrame:
    """Return one row per (series, origin, model) with forecast and actual."""
    out: list[dict] = []
    labelled = supervised.dropna(subset=["target_change"])
    fitted_cache: dict[tuple[str, int], GlobalModel | None] = {}

    def fitted(name: str, origin: int) -> GlobalModel | None:
        key = (name, origin)
        if key not in fitted_cache:
            train = labelled[labelled["target_year"] <= origin]
            if len(train) < min_train_rows:
                fitted_cache[key] = None
            else:
                fitted_cache[key] = global_models[name](train).fit(train)
        return fitted_cache[key]

    history = {k: g.sort_values("year") for k, g in panel.groupby(SERIES_KEY, sort=False)}

    for gkey, group in labelled.groupby(GROUP_KEY, sort=False):
        cycles = sorted(panel.merge(pd.DataFrame([dict(zip(GROUP_KEY, gkey))]), on=GROUP_KEY)["year"].unique())
        for origin in sorted(group["origin_year"].unique()):
            if origins is not None and origin not in origins:
                continue
            if sum(c <= origin for c in cycles) < min_train_cycles:
                continue
            test = group[group["origin_year"] == origin]
            for _, row in test.iterrows():
                skey = tuple(row[k] for k in SERIES_KEY)
                h = history[skey]
                h = h[h["year"] <= origin]
                years = h["year"].to_numpy()
                ys = h["mean_score"].to_numpy(dtype=float)
                base = {k: row[k] for k in SERIES_KEY}
                base.update({
                    "origin_year": int(origin),
                    "target_year": int(row["target_year"]),
                    "n_history": len(ys),
                    "actual": float(row["y_target"]),
                    "last_value": float(row["y_origin"]),
                })
                for name, fn in local_models.items():
                    value, used = fn(years, ys, int(row["target_year"]))
                    out.append({**base, "model": name, "model_kind": "time_series", "forecast": value, "variant_used": used})
            for name in global_models:
                model = fitted(name, int(origin))
                if model is None:
                    continue
                changes = model.predict_change(test)
                for (_, row), change in zip(test.iterrows(), changes):
                    base = {k: row[k] for k in SERIES_KEY}
                    out.append({
                        **base,
                        "origin_year": int(origin),
                        "target_year": int(row["target_year"]),
                        "n_history": int(row["n_hist"]),
                        "actual": float(row["y_target"]),
                        "last_value": float(row["y_origin"]),
                        "model": name,
                        "model_kind": "global",
                        "forecast": float(row["y_origin"] + change),
                        "variant_used": name,
                    })
    preds = pd.DataFrame(out)
    if not preds.empty:
        preds["error"] = preds["forecast"] - preds["actual"]
    return preds


def common_points(preds: pd.DataFrame) -> pd.DataFrame:
    """Keep only forecast points that every model produced (fair comparison)."""
    key = SERIES_KEY + ["origin_year"]
    n_models = preds["model"].nunique()
    counts = preds.groupby(key)["model"].transform("nunique")
    return preds[counts == n_models].copy()


def accuracy(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = list(by or [])
    naive = preds[preds["model"] == "Naive"].set_index(SERIES_KEY + ["origin_year"])["error"].abs()

    def summarise(g: pd.DataFrame) -> pd.Series:
        e = g["error"].to_numpy()
        idx = pd.MultiIndex.from_frame(g[SERIES_KEY + ["origin_year"]])
        naive_abs = naive.reindex(idx).to_numpy()
        ok = ~np.isnan(naive_abs)
        rel = np.mean(np.abs(e[ok])) / np.mean(naive_abs[ok]) if ok.any() and np.mean(naive_abs[ok]) > 0 else np.nan
        return pd.Series({
            "n_forecasts": len(e),
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE": float(np.mean(np.abs(e))),
            "Bias": float(np.mean(e)),
            "RelMAE_vs_naive": float(rel),
        })

    rows = []
    for keys, g in preds.groupby(by + ["model", "model_kind"], sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(by + ["model", "model_kind"], keys)), **summarise(g).to_dict()})
    out = pd.DataFrame(rows)
    return out.sort_values(by + ["RMSE"]).reset_index(drop=True)


def dm_test(preds: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Diebold–Mariano-type test of equal squared-error loss, country-clustered SE.

    d = e_a² − e_b²; negative mean ⇒ model_a more accurate. t has G − 1 df
    (G = number of countries); p is two-sided.
    """
    from scipy import stats

    key = SERIES_KEY + ["origin_year"]
    a = preds[preds["model"] == model_a].set_index(key)["error"]
    b = preds[preds["model"] == model_b].set_index(key)["error"]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(joined) < 5:
        return {"model_a": model_a, "model_b": model_b, "n": len(joined), "mean_loss_diff": np.nan,
                "t": np.nan, "p_value": np.nan, "n_clusters": 0}
    d = (joined["a"] ** 2 - joined["b"] ** 2).to_numpy()
    clusters = joined.index.get_level_values("country_code").to_numpy()
    mean = d.mean()
    resid = d - mean
    sums = pd.Series(resid).groupby(clusters).sum().to_numpy()
    G = len(sums)
    n = len(d)
    var = (G / (G - 1)) * np.sum(sums**2) / n**2 if G > 1 else np.nan
    t = mean / np.sqrt(var) if var and var > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=G - 1) if np.isfinite(t) else np.nan
    return {"model_a": model_a, "model_b": model_b, "n": n, "n_clusters": G,
            "mean_loss_diff": float(mean), "t": float(t), "p_value": float(p)}


def dm_table(preds: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([dm_test(preds, a, b) for a, b in pairs])
