"""
Forecasters for country-level mean achievement.

Two kinds:

* **Local time-series models** see one country's own series (``years``, ``ys``)
  and forecast the next cycle: naive (last value), historical mean, drift, damped
  Holt, small-order ARIMA chosen by AICc.
* **Global (pooled) models** learn the change to the next cycle from all
  series' supervised rows (``src.forecasting.panel.build_supervised``): pooled
  panel regression (optionally sign-constrained by knowledge-base priors),
  Random Forest and XGBoost.

ILSA series are short (5–9 cycles), so every local model falls back to a simpler
one when the history is too short and records the fallback.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# ── Local models ──────────────────────────────────────────────────────────────

LocalForecast = Callable[[np.ndarray, np.ndarray, int], tuple[float, str]]


def naive(years: np.ndarray, ys: np.ndarray, target_year: int) -> tuple[float, str]:
    return float(ys[-1]), "naive"


def historical_mean(years: np.ndarray, ys: np.ndarray, target_year: int) -> tuple[float, str]:
    return float(np.mean(ys)), "mean"


def drift(years: np.ndarray, ys: np.ndarray, target_year: int) -> tuple[float, str]:
    """Random walk with drift; the slope is per calendar year, so uneven cycle gaps are respected."""
    if len(ys) < 2 or years[-1] == years[0]:
        return naive(years, ys, target_year)
    slope = (ys[-1] - ys[0]) / (years[-1] - years[0])
    return float(ys[-1] + slope * (target_year - years[-1])), "drift"


def holt_damped(years: np.ndarray, ys: np.ndarray, target_year: int) -> tuple[float, str]:
    if len(ys) < 4:
        value, used = drift(years, ys, target_year)
        return value, f"{used}(fallback)"
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(ys, trend="add", damped_trend=True, initialization_method="estimated").fit()
            return float(fit.forecast(1)[0]), "holt_damped"
        except Exception:  # noqa: BLE001 — estimation failures on tiny series
            value, used = drift(years, ys, target_year)
            return value, f"{used}(fallback)"


_ARIMA_ORDERS = [((1, 0, 0), "c"), ((0, 1, 1), "n"), ((1, 1, 0), "n"), ((0, 1, 0), "t")]


def arima_aicc(years: np.ndarray, ys: np.ndarray, target_year: int) -> tuple[float, str]:
    """Pick the best of a few low-order ARIMA models by AICc (cycles treated as equally spaced)."""
    if len(ys) < 5:
        value, used = naive(years, ys, target_year)
        return value, f"{used}(fallback)"
    from statsmodels.tsa.arima.model import ARIMA

    best: tuple[float, float, str] | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, trend in _ARIMA_ORDERS:
            try:
                res = ARIMA(ys, order=order, trend=trend).fit()
            except Exception:  # noqa: BLE001
                continue
            aicc = res.aicc if np.isfinite(res.aicc) else res.aic
            if best is None or aicc < best[0]:
                best = (aicc, float(res.forecast(1)[0]), f"arima{order}{trend}")
    if best is None:
        value, used = naive(years, ys, target_year)
        return value, f"{used}(fallback)"
    return best[1], best[2]


LOCAL_MODELS: dict[str, LocalForecast] = {
    "Naive": naive,
    "Mean": historical_mean,
    "Drift": drift,
    "Holt_damped": holt_damped,
    "ARIMA": arima_aicc,
}

# ── Global models ─────────────────────────────────────────────────────────────


@dataclass
class GlobalModel:
    """Fits on supervised rows, predicts ``target_change``."""

    name: str
    features: list[str]

    def fit(self, train: pd.DataFrame) -> "GlobalModel":  # pragma: no cover - interface
        raise NotImplementedError

    def predict_change(self, test: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class PanelRegression(GlobalModel):
    """Pooled ridge regression of the cycle-to-cycle change, with optional sign bounds.

    ``signs`` maps a feature to +1/−1 (coefficient bounded to that sign, e.g. from
    the literature knowledge base); ``penalty_scale`` multiplies the ridge penalty of
    a feature (< 1 = weaker shrinkage for well-supported predictors).
    """

    alpha: float = 1.0
    signs: dict[str, int] = field(default_factory=dict)
    penalty_scale: dict[str, float] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "PanelRegression":
        from scipy.optimize import lsq_linear

        X = train[self.features].astype(float)
        self.medians_ = X.median().fillna(0.0)
        X = X.fillna(self.medians_)
        self.mu_ = X.mean()
        self.sd_ = X.std(ddof=0).replace(0, 1.0).fillna(1.0)
        Z = ((X - self.mu_) / self.sd_).to_numpy()
        y = train["target_change"].to_numpy(dtype=float)
        n, p = Z.shape
        Za = np.hstack([np.ones((n, 1)), Z])
        lam = np.array([self.alpha * self.penalty_scale.get(f, 1.0) for f in self.features])
        A = np.vstack([Za, np.hstack([np.zeros((p, 1)), np.diag(np.sqrt(lam))])])
        b = np.concatenate([y, np.zeros(p)])
        lower = np.full(p + 1, -np.inf)
        upper = np.full(p + 1, np.inf)
        for j, f in enumerate(self.features, start=1):
            s = self.signs.get(f, 0)
            if s > 0:
                lower[j] = 0.0
            elif s < 0:
                upper[j] = 0.0
        res = lsq_linear(A, b, bounds=(lower, upper))
        self.coef_ = res.x
        return self

    def predict_change(self, test: pd.DataFrame) -> np.ndarray:
        X = test[self.features].astype(float).fillna(self.medians_)
        Z = ((X - self.mu_) / self.sd_).to_numpy()
        return self.coef_[0] + Z @ self.coef_[1:]

    def coefficients(self) -> dict[str, float]:
        """Coefficients on the original feature scale (per unit of the feature)."""
        return {f: float(self.coef_[j + 1] / self.sd_[f]) for j, f in enumerate(self.features)}


@dataclass
class RandomForestChange(GlobalModel):
    n_estimators: int = 500
    min_samples_leaf: int = 3
    random_state: int = 0

    def fit(self, train: pd.DataFrame) -> "RandomForestChange":
        from sklearn.ensemble import RandomForestRegressor

        X = train[self.features].astype(float)
        self.medians_ = X.median().fillna(0.0)
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=0.5,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X.fillna(self.medians_), train["target_change"].astype(float))
        return self

    def predict_change(self, test: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(test[self.features].astype(float).fillna(self.medians_))


@dataclass
class XGBoostChange(GlobalModel):
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    monotone: dict[str, int] = field(default_factory=dict)
    random_state: int = 0

    def fit(self, train: pd.DataFrame) -> "XGBoostChange":
        from xgboost import XGBRegressor

        constraints = tuple(int(self.monotone.get(f, 0)) for f in self.features)
        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            monotone_constraints=constraints if any(constraints) else None,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(train[self.features].astype(float), train["target_change"].astype(float))
        return self

    def predict_change(self, test: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(test[self.features].astype(float))
