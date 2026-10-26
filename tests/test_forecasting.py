import numpy as np

from src.forecasting.evaluation import accuracy, common_points, dm_test, rolling_origin
from src.forecasting.models import LOCAL_MODELS, PanelRegression, XGBoostChange
from src.forecasting.panel import build_supervised
from synthetic_panel import make_panel


def test_rolling_origin_has_no_lookahead():
    panel, cov = make_panel(n_countries=8, seed=1)
    sup = build_supervised(panel, cov, ["escs_mean"])
    seen = []

    def spy(train):
        seen.append(int(train["target_year"].max()))
        return PanelRegression("spy", ["d1", "cov_escs_mean"])

    seen_hist = []

    def local_spy(years, ys, target_year):
        seen_hist.append((int(years.max()), target_year))
        return float(ys[-1]), "naive"

    preds = rolling_origin(panel, sup.frame, local_models={"Naive": local_spy}, global_models={"spy": spy},
                           min_train_rows=10)
    assert not preds.empty
    assert all(last < target for last, target in seen_hist)
    origins = sorted(preds["origin_year"].unique())
    assert max(seen) <= max(origins)
    g = preds[preds["model"] == "spy"]
    assert (g["origin_year"] < g["target_year"]).all()


def test_covariates_are_as_of_origin():
    panel, cov = make_panel(n_countries=3, seed=2)
    cov = cov.copy()
    cov.loc[cov["year"] > 2000, "escs_mean"] = np.nan  # only early values exist
    sup = build_supervised(panel, cov, ["escs_mean"], max_covariate_lag=3).frame
    late = sup[sup["origin_year"] >= 2006]
    assert late["cov_escs_mean"].isna().all()  # older than max lag → not carried forward


def test_end_to_end_metrics_and_dm():
    panel, cov = make_panel(n_countries=10, seed=3)
    sup = build_supervised(panel, cov, ["escs_mean", "anxiety_mean"])
    feats = sup.feature_columns + ["cov_escs_mean", "dcov_escs_mean"]
    preds = rolling_origin(
        panel, sup.frame,
        local_models={k: LOCAL_MODELS[k] for k in ("Naive", "Drift")},
        global_models={
            "XGB": lambda tr: XGBoostChange("XGB", feats, n_estimators=50, monotone={"dcov_escs_mean": 1}),
            "Panel": lambda tr: PanelRegression("Panel", ["d1", "dcov_escs_mean"], signs={"dcov_escs_mean": 1}),
        },
    )
    fair = common_points(preds)
    acc = accuracy(fair)
    assert set(acc["model"]) == {"Naive", "Drift", "XGB", "Panel"}
    res = dm_test(fair, "Panel", "Naive")
    assert res["n_clusters"] == 10 and np.isfinite(res["t"])
