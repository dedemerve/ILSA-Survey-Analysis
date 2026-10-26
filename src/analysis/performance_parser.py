"""
Parse free-text ``performance_metrics`` into a long model × metric × value table.

The extraction stored metrics as one string per finding, e.g.

    "Model comparison: RF rMSE=69.82, XGBoost rMSE=66.39, LightGBM rMSE=66.36"

Each metric token is attributed to the closest *preceding* model mention in the
same clause (clauses split on ';', '|', newlines and sentence ends). When a clause
has no model mention, the token is attributed to the study's primary model only if
the study lists exactly one predictive technique; otherwise it is kept with
``model=None`` so it is never silently assigned to the wrong learner.

Every row keeps the source clause so that coders can verify it
(``scripts/validation_sample.py``).
"""

from __future__ import annotations

import re
from typing import Any

from src.analysis.ml_taxonomy import canonicalize_list, find_models_in_text

# metric label → (canonical name, higher_is_better)
METRICS: dict[str, tuple[str, bool]] = {
    "auc": ("AUC", True),
    "roc-auc": ("AUC", True),
    "roc auc": ("AUC", True),
    "auroc": ("AUC", True),
    "balanced accuracy": ("Balanced accuracy", True),
    "accuracy": ("Accuracy", True),
    "acc": ("Accuracy", True),
    "f1": ("F1", True),
    "f1-score": ("F1", True),
    "f1 score": ("F1", True),
    "macro-f1": ("F1", True),
    "macro f1": ("F1", True),
    "precision": ("Precision", True),
    "recall": ("Recall", True),
    "sensitivity": ("Recall", True),
    "specificity": ("Specificity", True),
    "kappa": ("Kappa", True),
    "qwk": ("Kappa", True),
    "r²": ("R2", True),
    "r2": ("R2", True),
    "r^2": ("R2", True),
    "r-squared": ("R2", True),
    "rmse": ("RMSE", False),
    "mae": ("MAE", False),
    "mse": ("MSE", False),
    "mape": ("MAPE", False),
}

_METRIC_TOKEN = (
    r"balanced accuracy|roc[\s-]auc|auroc|(?<!pr-)(?<!pr_)(?<!pr )\bauc\b|\baccuracy\b|\bacc\b|macro[\s-]f1|f1[\s-]score|\bf1\b|"
    r"\bprecision\b|\brecall\b|\bsensitivity\b|\bspecificity\b|\bkappa\b|\bqwk\b|"
    r"r²|\br\^2|\br2\b|r-squared|\br?rmse\b|\bmae\b|\bmse\b|\bmape\b"
)
# metric [optional words like "of", "="/":", "≈", "was"] number [%]
_METRIC_VALUE_RE = re.compile(
    rf"(?P<metric>{_METRIC_TOKEN})\s*(?:\((?:[^)]{{0,20}})\))?\s*(?:of|was|is|=|:|≈|~|reached|=\s*approximately)?\s*"
    rf"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<pct>%)?",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r";|\||\n|(?<=[a-z0-9%)])\.\s+(?=[A-Z])")


def canonical_metric(token: str) -> tuple[str, bool] | None:
    key = token.lower().strip()
    key = re.sub(r"\s+", " ", key)
    if key == "rrmse":
        key = "rmse"
    return METRICS.get(key) or METRICS.get(key.replace(" ", "-"))


def _normalize_value(metric: str, value: float, pct: bool) -> float:
    """Proportion-type metrics reported in percent → 0–1 scale."""
    proportion = metric in {"AUC", "Accuracy", "Balanced accuracy", "F1", "Precision", "Recall", "Specificity", "Kappa", "R2"}
    if proportion and (pct or value > 1.0) and value <= 100:
        return round(value / 100.0, 6)
    return value


def parse_performance(
    text: str | None,
    *,
    study_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return rows {model, model_family, metric, value, higher_is_better, clause, attribution}."""
    if not text or str(text).strip().lower() in {"not reported", "n/a", "na", "none", ""}:
        return []
    predictive = canonicalize_list(study_models or [], predictive_only=True)
    sole_model = predictive[0] if len(predictive) == 1 else None

    rows: list[dict[str, Any]] = []
    for clause in _CLAUSE_SPLIT.split(str(text)):
        clause = clause.strip()
        if not clause:
            continue
        mentions = [(pos, m) for pos, m in find_models_in_text(clause) if m.role == "predictive"]
        for mv in _METRIC_VALUE_RE.finditer(clause):
            canon = canonical_metric(mv.group("metric"))
            if canon is None:
                continue
            metric, higher = canon
            value = _normalize_value(metric, float(mv.group("value")), bool(mv.group("pct")))
            preceding = [m for pos, m in mentions if pos <= mv.start()]
            if preceding:
                model, attribution = preceding[-1], "clause_preceding_mention"
            elif mentions:
                model, attribution = mentions[0][1], "clause_following_mention"
            elif sole_model is not None:
                model, attribution = sole_model, "sole_study_model"
            else:
                model, attribution = None, "unattributed"
            rows.append({
                "model": model.name if model else None,
                "model_family": model.family if model else None,
                "metric": metric,
                "value": value,
                "higher_is_better": higher,
                "attribution": attribution,
                "clause": clause[:300],
            })
    return rows
