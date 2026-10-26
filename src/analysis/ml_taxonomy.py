"""
Canonical ML-technique taxonomy with word-boundary matching.

Replaces the earlier substring matcher (``fragment in key``), which mapped e.g.
"Gradient Boosted Decision Trees" → Decision Tree, "Semi-supervised learning" →
Structural Equation Modeling ("sem") and "Principal components regression" → PCA.

Matching order:
  1. exact alias lookup on a normalized key;
  2. ordered regex rules (most specific first), all anchored on word boundaries;
  3. unmatched labels are returned unchanged with family ``Other``.

Each canonical model also carries a ``role``:
  * ``predictive``      — supervised learner that produces predictions (counted in RQ1 tables);
  * ``unsupervised``    — clustering / dimensionality reduction;
  * ``feature_or_xai``  — text features or explanation tools (TF-IDF, Word2Vec, SHAP);
  * ``learning_paradigm`` — generic labels such as "Semi-supervised learning".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CanonicalModel:
    name: str
    family: str
    role: str = "predictive"


# Families
TREE_ENSEMBLE = "Tree ensemble"
SINGLE_TREE = "Single tree"
NEURAL = "Neural network / deep learning"
KERNEL = "Kernel / SVM"
REG_LINEAR = "Regularized linear"
LINEAR = "Linear / GLM"
BAYES = "Bayesian / probabilistic"
INSTANCE = "Instance-based"
META_ENSEMBLE = "Meta-ensemble"
UNSUP = "Unsupervised"
FEATURE = "Feature extraction / XAI"
OTHER = "Other"

_M = CanonicalModel

# (pattern, canonical model) — evaluated in order; first match wins.
_RULES: list[tuple[str, CanonicalModel]] = [
    # Explanation / feature tools (never counted as predictive models)
    (r"\bshap\b|shapley", _M("SHAP", FEATURE, "feature_or_xai")),
    (r"\blime\b", _M("LIME", FEATURE, "feature_or_xai")),
    (r"\btf[\s-]?idf\b", _M("TF-IDF", FEATURE, "feature_or_xai")),
    (r"\bword2vec\b", _M("Word2Vec", FEATURE, "feature_or_xai")),
    (r"\bt[\s-]?sne\b", _M("t-SNE", UNSUP, "unsupervised")),
    (r"\battention mechanism\b", _M("Attention mechanism", NEURAL, "feature_or_xai")),
    (r"\bsemi[\s-]?supervised\b", _M("Semi-supervised learning", OTHER, "learning_paradigm")),
    (r"\bactive learning\b", _M("Active learning", OTHER, "learning_paradigm")),
    # Causal / Bayesian tree models (before generic forest / tree rules)
    (r"\bbcf\b|bayesian causal forest", _M("BCF", BAYES)),
    (r"\bbart\b|bayesian additive regression tree", _M("BART", BAYES)),
    (r"conditional inference (?:random )?forests?\b|\bcforest\b", _M("Conditional Inference Forest", TREE_ENSEMBLE)),
    (r"\bcausal forests?\b", _M("Causal Forest", TREE_ENSEMBLE)),
    (r"\bisolation forests?\b", _M("Isolation Forest", UNSUP, "unsupervised")),
    # Boosting family (specific → general)
    (r"\bxgb(?:oost)?\b|extreme gradient boost", _M("XGBoost", TREE_ENSEMBLE)),
    (r"\blight\s?gbm\b|\blgbm\b", _M("LightGBM", TREE_ENSEMBLE)),
    (r"\bcat\s?boost\b", _M("CatBoost", TREE_ENSEMBLE)),
    (r"\bada\s?boost\b", _M("AdaBoost", TREE_ENSEMBLE)),
    (r"\bhist(?:ogram)?[\s-]?(?:based )?(?:gbr|gbm|gradient boost\w*)\b", _M("Gradient Boosting", TREE_ENSEMBLE)),
    (r"gradient[\s-]?boost\w*|\bgbm\b|\bgbdt\b|\bgbr\b|stochastic gradient boost\w*", _M("Gradient Boosting", TREE_ENSEMBLE)),
    # Forests / bagging
    (r"extra(?:ordinarily)?[\s-]?trees\b|extremely randomi[sz]ed trees", _M("Extra Trees", TREE_ENSEMBLE)),
    (r"\brandom[\s-]?forests?\b|\brf\b|distributed random forest", _M("Random Forest", TREE_ENSEMBLE)),
    (r"\bbagg(?:ing|ed)\b", _M("Bagging", TREE_ENSEMBLE)),
    # Meta-ensembles
    (r"\bstack(?:ing|ed)\b", _M("Stacking", META_ENSEMBLE)),
    (r"\bblending\b", _M("Blending", META_ENSEMBLE)),
    (r"\bvoting\b", _M("Voting ensemble", META_ENSEMBLE)),
    (r"^ensemble(?: learning)?$", _M("Ensemble (unspecified)", META_ENSEMBLE)),
    # Single trees
    (r"conditional inference trees?\b|\bctree\b", _M("Conditional Inference Tree", SINGLE_TREE)),
    (r"\bdecision trees?\b|\bcart\b|classification (?:and regression )?trees?\b|regression trees?\b|\bc4\.5\b|\bc5\.0\b|\bchaid\b|\bid3\b", _M("Decision Tree", SINGLE_TREE)),
    (r"\bmars\b|multivariate adaptive regression spline", _M("MARS", OTHER)),
    # Neural / deep learning
    (r"\blstm\b|long short[\s-]term memory", _M("LSTM", NEURAL)),
    (r"\bgru\b|gated recurrent", _M("GRU", NEURAL)),
    (r"\brnn\b|recurrent neural", _M("RNN", NEURAL)),
    (r"\bcnn\b|convolutional", _M("CNN", NEURAL)),
    (r"\bautoencoders?\b", _M("Autoencoder", NEURAL)),
    (r"\b(?:ro)?bert(?:a)?\b|\btransformers?\b|\bgpt\b|large language model|\bllm\b", _M("Transformer / LLM", NEURAL)),
    (r"\banfis\b|neuro[\s-]?fuzzy", _M("ANFIS", NEURAL)),
    (r"\bmlp\b|multi[\s-]?layer perceptron|perceptron|neural net\w*|\bann\b|\bdnn\b|deep learning|elman|jordan", _M("Neural Network", NEURAL)),
    # Kernel
    (r"\bsvm\b|\bsvr\b|\bsvc\b|support vector", _M("SVM", KERNEL)),
    (r"fast large margin", _M("Fast Large Margin", KERNEL)),
    # Regularized linear
    (r"glmm\s?lasso", _M("glmmLasso", REG_LINEAR)),
    (r"group\s?mnet", _M("Group Mnet", REG_LINEAR)),
    (r"elastic[\s-]?net|\benr\b", _M("Elastic Net", REG_LINEAR)),
    (r"\blasso\b", _M("LASSO", REG_LINEAR)),
    (r"least angle regression|\blars\b", _M("Least Angle Regression", REG_LINEAR)),
    (r"bayesian ridge", _M("Bayesian Ridge", BAYES)),
    (r"\bridge\b", _M("Ridge Regression", REG_LINEAR)),
    # Bayesian / probabilistic
    (r"naive[\s-]?bayes|\bnb\b", _M("Naive Bayes", BAYES)),
    (r"bayesian network", _M("Bayesian Network", BAYES)),
    (r"gaussian mixture|\bgmm\b", _M("Gaussian Mixture Model", UNSUP, "unsupervised")),
    # Instance-based
    (r"\bk[\s-]?nn\b|k[\s-]?nearest", _M("k-NN", INSTANCE)),
    # Linear / GLM
    (r"logistic regression|\blogit\b", _M("Logistic Regression", LINEAR)),
    (r"(?:linear |quadratic )?discriminant analysis|\blda\b|\bqda\b", _M("Discriminant Analysis", LINEAR)),
    (r"hierarchical general linear model|generalized linear (?:mixed )?model|\bglmm?\b", _M("Generalized Linear Model", LINEAR)),
    (r"(?:multiple )?linear regression|\bols\b", _M("Linear Regression", LINEAR)),
    # Unsupervised
    (r"\bk[\s-]?means\b", _M("k-means", UNSUP, "unsupervised")),
    (r"\bk[\s-]?medoids\b|\bpam\b", _M("k-medoids", UNSUP, "unsupervised")),
    (r"\bdbscan\b", _M("DBSCAN", UNSUP, "unsupervised")),
    (r"hierarchical clustering|agglomerative", _M("Hierarchical clustering", UNSUP, "unsupervised")),
    (r"latent profile|\blpa\b", _M("Latent Profile Analysis", UNSUP, "unsupervised")),
    (r"latent class|\blca\b", _M("Latent Class Analysis", UNSUP, "unsupervised")),
    (r"principal components? regression|\bpcr\b", _M("Principal Components Regression", REG_LINEAR)),
    (r"principal component|\bpca\b", _M("PCA", UNSUP, "unsupervised")),
    (r"\bcluster", _M("Cluster analysis", UNSUP, "unsupervised")),
]

_COMPILED: list[tuple[re.Pattern[str], CanonicalModel]] = [
    (re.compile(p, re.IGNORECASE), m) for p, m in _RULES
]

_BLANK = {"", "none", "null", "n/a", "na", "not reported", "not_reported"}


def _normalize(label: str) -> str:
    text = str(label).strip().lower()
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text)


def canonicalize(label: str | None) -> CanonicalModel | None:
    """Map one raw technique label to its canonical model (None for blanks)."""
    if label is None:
        return None
    key = _normalize(label)
    if key in _BLANK:
        return None
    for pattern, model in _COMPILED:
        if pattern.search(key):
            return model
    return CanonicalModel(str(label).strip(), OTHER)


def canonical_name(label: str | None) -> str | None:
    model = canonicalize(label)
    return model.name if model else None


def canonicalize_list(labels: Iterable[str] | None, *, predictive_only: bool = False) -> list[CanonicalModel]:
    """Canonicalize a list of labels, dropping duplicates while preserving order."""
    out: list[CanonicalModel] = []
    seen: set[str] = set()
    for raw in labels or []:
        model = canonicalize(raw)
        if model is None or model.name in seen:
            continue
        if predictive_only and model.role != "predictive":
            continue
        seen.add(model.name)
        out.append(model)
    return out


def find_models_in_text(text: str | None) -> list[tuple[int, CanonicalModel]]:
    """Return (position, model) for every model mention in free text, sorted by position.

    Used by the performance-metric parser to attribute metric values to models.
    Overlapping matches keep the earliest-listed (most specific) rule.
    """
    if not text:
        return []
    hits: list[tuple[int, int, CanonicalModel]] = []
    taken: list[tuple[int, int]] = []
    for pattern, model in _COMPILED:
        if pattern.pattern.startswith("^"):
            continue
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if any(not (span[1] <= s or span[0] >= e) for s, e in taken):
                continue
            taken.append(span)
            hits.append((m.start(), m.end(), model))
    hits.sort(key=lambda h: h[0])
    return [(start, model) for start, _end, model in hits]


# Family labels used by the legacy knowledge-synthesis tables.
LEGACY_METHOD_FAMILY: dict[str, str] = {
    TREE_ENSEMBLE: "Ensemble_Learning",
    META_ENSEMBLE: "Ensemble_Learning",
    NEURAL: "Deep_Learning",
    UNSUP: "Unsupervised",
    SINGLE_TREE: "Supervised_General",
    KERNEL: "Supervised_General",
    BAYES: "Supervised_General",
    INSTANCE: "Supervised_General",
    REG_LINEAR: "Supervised_General",
    LINEAR: "Supervised_General",
}
