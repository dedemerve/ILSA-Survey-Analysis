"""Deterministic policy-actionability rubric (D1 / D2 / D3 → Score 1–5).

Applied to extracted ILSA article JSON. Full PDF abstract/methods/results are
not required; scoring uses title, techniques, sample, findings, and summaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

D1_NONE = "None"
D1_SYNTHESIS = "Synthesis"
D1_CORRELATIONAL = "Correlational"
D1_CAUSAL = "Causal"

D2_NONE = "None"
D2_DIRECTIONAL = "Directional"
D2_QUANTIFIED = "Quantified"

D3_GLOBAL = "Global/Unspecified"
D3_BOUNDED = "Bounded"

# Standard supervised/unsupervised ML (rubric D1 = Correlational).
_CORRELATIONAL_ML_RE = re.compile(
    r"\b(?:random\s*forest|xgboost|lightgbm|catboost|svm|support\s+vector|"
    r"neural\s+network|deep\s+learning|lasso|elastic\s*net|ridge|"
    r"gradient\s+boost(?:ing|ed)?|decision\s+tree|logistic\s+regression|"
    r"linear\s+regression|k-?means|k-?nn|knn|clustering|latent\s+(?:class|profile)|"
    r"nlp|bert|transformer|cnn|lstm|autoencoder|stacking|bagging|"
    r"naive\s+bayes|discriminant\s+analysis|boruta|shap\b|lime\b|"
    r"multilevel|hierarchical\s+linear|sem\b|irt\b|cdm\b)\b",
    re.IGNORECASE,
)

# Quasi-causal / causal ML (do not match "cannot establish causality").
_CAUSAL_METHOD_RE = re.compile(
    r"\b(?:bayesian\s+causal\s+forest|\bbcf\b|bayesian\s+additive\s+regression\s+trees|"
    r"\bbart\b|causal\s+forest|generalized\s+random\s+forest|\bgrf\b|"
    r"difference[\s-]*in[\s-]*differences|\bdid\b|"
    r"regression\s+discontinuity|\brdd\b|"
    r"instrumental\s+variables?|\b2sls\b|"
    r"propensity\s+score(?:\s+matching)?|\bpsm\b|"
    r"average\s+treatment\s+effect|\bcate\b|\bate\b|"
    r"counterfactual\s+prediction|treatment\s+effect\s+estimation|"
    r"interventional\s+shap|do-calculus|"
    r"causal\s+inference\s+machine\s+learning|"
    r"what\s+would\s+happen\s+if)\b",
    re.IGNORECASE,
)

_CANNOT_CAUSAL_RE = re.compile(
    r"cannot\s+establish\s+causal|does\s+not\s+claim\s+causal|"
    r"no\s+causal\s+(?:claim|inference|effect)|cross-sectional.{0,40}causal",
    re.IGNORECASE | re.DOTALL,
)

_SYNTHESIS_HINT_RE = re.compile(
    r"\b(?:systematic\s+review|literature\s+review|meta-analys[ie]s|scientometric|"
    r"bibliometric|this\s+is\s+not\s+an\s+empirical|"
    r"does\s+not\s+(?:train|conduct|analyse|analyze)\s+(?:an?\s+)?original|"
    r"no\s+original\s+(?:ilsa\s+)?microdata|"
    r"conceptual/review|methodological\s+(?:review|critique))\b",
    re.IGNORECASE,
)

_NO_DATA_ANALYSIS_RE = re.compile(
    r"\b(?:purely\s+technical|framework\s+paper|no\s+original\s+(?:analytic\s+)?sample|"
    r"does\s+not\s+report\s+an\s+original\s+analytic\s+sample|"
    r"no\s+dataset/cycle\s+is\s+analyzed|"
    r"not\s+an\s+empirical\s+ml\s+study)\b",
    re.IGNORECASE,
)

_METRIC_RE = re.compile(
    r"\b(?:accuracy|acc|f1(?:-score)?|auc|roc(?:-auc)?|rmse|mae|mse|"
    r"r\s*[²2]|r-?squared|precision|recall|specificity|"
    r"odds\s+ratio|\bor\b|beta|shap|feature\s+importance|"
    r"variable\s+importance|ate|cate|treatment\s+effect|"
    r"pearson|spearman|correlation\s+coefficient|"
    r"pccr|accr|credible\s+interval|crI)\b",
    re.IGNORECASE,
)

# Numbers that quantify findings (not years, grades, or sample-size n=).
_FINDING_NUMBER_RE = re.compile(
    r"(?<!n\s)(?<!n=)(?<!N\s)(?<!N=)"
    r"(?:(?:accuracy|auc|f1|rmse|mae|mse|r²|r2|r\^2|precision|recall|"
    r"ate|or|β|beta|shap)\s*[=:]?\s*)?"
    r"[-+]?\d+\.\d+(?:\s*%|\s*points?)?"
    r"|(?:accuracy|auc|f1|rmse|mae|r²|r2)\s*[=:]\s*\d+"
    r"|\b\d+\.\d+\s*%"
    r"|\bR²\s*[=≈≈]?\s*0\.\d+",
    re.IGNORECASE,
)

_DIRECTIONAL_RE = re.compile(
    r"\b(?:positively\s+predict|negatively\s+predict|positive(?:ly)?\s+associat|"
    r"negative(?:ly)?\s+associat|higher\s+(?:than|in)|lower\s+(?:than|in)|"
    r"increased|decreased|more\s+important|strongest\s+predictor|"
    r"outperform|better\s+than)\b",
    re.IGNORECASE,
)

_ILSA_CYCLE_RE = re.compile(
    r"\b(?:PISA|TIMSS|PIRLS|TALIS|ICILS|ICCS|PIAAC|INVALSI)"
    r"(?:\s+\d{4})?\b",
    re.IGNORECASE,
)

_GRADE_RE = re.compile(
    r"\b(?:grade\s*[0-9]{1,2}|[0-9]{1,2}th[\s-]?grade|"
    r"15-year-olds?|eighth[\s-]?grade|fourth[\s-]?grade|"
    r"lower\s+secondary|primary\s+teachers?)\b",
    re.IGNORECASE,
)

_STUDENT_GROUP_RE = re.compile(
    r"\b(?:immigrant\s+students?|low[\s-]ses|high[\s-]ses|"
    r"resilient\s+students?|multilingual\s+learners?|"
    r"girls|boys|disadvantaged)\b",
    re.IGNORECASE,
)

_COUNTRY_NAME_RE = re.compile(
    r"\b(?:Turkey|Türkiye|Singapore|Germany|Ireland|England|United\s+States|"
    r"USA|UK|Japan|China|Finland|Korea|Australia|Canada|France|Spain|"
    r"Italy|Netherlands|Sweden|Norway|Denmark|Poland|Portugal|Greece|"
    r"Mexico|Brazil|Chile|Colombia|Indonesia|Thailand|Vietnam|"
    r"Philippines|Macau|Hong\s+Kong|Chinese\s+Taipei|Taiwan|"
    r"Saudi\s+Arabia|Qatar|UAE|Israel|New\s+Zealand|South\s+Africa|"
    r"OECD\s+countries)\b",
    re.IGNORECASE,
)

_ISO3_RE = re.compile(r"^[A-Z]{3}$")

_NOT_REPORTED_RE = re.compile(
    r"^\s*(?:not\s+reported|n/?a|none|null)(?:\b.*)?$",
    re.IGNORECASE,
)

_YEAR_ONLY_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _blob(article: dict[str, Any]) -> str:
    meta = article.get("metadata") or {}
    data = article.get("data") or {}
    parts: list[str] = [
        str(meta.get("title") or ""),
        str(meta.get("file_name") or ""),
        str(data.get("outcome_summary") or ""),
        str(data.get("null_fields_interpretation") or ""),
        str((data.get("survey_design") or {}).get("weight_fields_interpretation") or ""),
        str((data.get("sample_details") or {}).get("sample_filtering_criteria") or ""),
    ]
    ml = data.get("ml_techniques") or {}
    if ml.get("primary"):
        parts.append(str(ml["primary"]))
    for t in ml.get("all_techniques") or []:
        parts.append(str(t))
    for finding in data.get("main_findings") or []:
        if not isinstance(finding, dict):
            continue
        parts.extend(
            [
                str(finding.get("dataset_used") or ""),
                str(finding.get("target_variable") or ""),
                str(finding.get("performance_metrics") or ""),
                str(finding.get("standardized_conclusion") or ""),
            ]
        )
    return "\n".join(parts)


def _finding_is_placeholder(finding: dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(finding.get("target_variable") or ""),
            str(finding.get("performance_metrics") or ""),
            str(finding.get("standardized_conclusion") or ""),
        ]
    ).lower()
    markers = (
        "literature synthesis",
        "not an empirical",
        "not a re-estimated",
        "does not conduct an original",
        "no ml algorithm is trained",
        "no analytic subsample",
        "review of how",
        "conceptual rather than estimating",
    )
    return any(m in blob for m in markers)


def _has_empirical_analysis(article: dict[str, Any], text: str) -> bool:
    data = article.get("data") or {}
    meta = article.get("metadata") or {}
    ml = data.get("ml_techniques") or {}
    sd = data.get("sample_details") or {}
    source = meta.get("source_category") or ""
    title = str(meta.get("title") or "")
    has_ml = bool(ml.get("primary") or (ml.get("all_techniques") or []))
    has_n = bool(sd.get("total_students"))
    countries = sd.get("countries") or []
    has_country_n = any(
        isinstance(c, dict) and c.get("n_students") for c in countries
    )

    reviewish = source == "review_article" or bool(
        re.search(
            r"\b(?:systematic\s+review|literature\s+review|integrative\s+review|"
            r"scoping\s+review|bibliometric|meta-analys[ie]s)\b",
            title,
            re.IGNORECASE,
        )
    )
    if reviewish and not has_ml and not has_n:
        return False

    if has_n or has_country_n or has_ml:
        return True
    real_findings = [
        f
        for f in (data.get("main_findings") or [])
        if isinstance(f, dict) and not _finding_is_placeholder(f)
    ]
    if real_findings and not reviewish:
        return True
    if _SYNTHESIS_HINT_RE.search(text) or _NO_DATA_ANALYSIS_RE.search(text):
        return False
    return False


def classify_d1(article: dict[str, Any], text: str) -> tuple[str, str]:
    data = article.get("data") or {}
    meta = article.get("metadata") or {}
    design = (data.get("research_design_type") or "") or ""
    ml = data.get("ml_techniques") or {}
    techniques = " ".join(
        [str(ml.get("primary") or "")] + [str(t) for t in (ml.get("all_techniques") or [])]
    )
    source = meta.get("source_category") or ""

    causal_hit = _CAUSAL_METHOD_RE.search(techniques) or _CAUSAL_METHOD_RE.search(
        str(meta.get("title") or "")
    )
    # Ignore boilerplate "cannot establish causality" in standardized_conclusion
    # when looking at the full blob for causal methods — search techniques + title
    # + outcome_summary only if design is already causal.
    if design in ("causal_observational", "causal_experimental"):
        evidence = (
            f"research_design_type={design}"
            + (f"; method={ml.get('primary')}" if ml.get("primary") else "")
        )
        return D1_CAUSAL, evidence
    if causal_hit:
        return D1_CAUSAL, f"Causal/quasi-causal method detected: {causal_hit.group(0)}"

    empirical = _has_empirical_analysis(article, text)
    method_scope = " ".join(
        [techniques, str(meta.get("title") or ""), str(data.get("outcome_summary") or "")[:800]]
    )
    if empirical:
        if _CORRELATIONAL_ML_RE.search(techniques) or _CORRELATIONAL_ML_RE.search(method_scope):
            primary = ml.get("primary") or "standard ML/statistical modelling"
            return D1_CORRELATIONAL, f"Analyses new data with {primary}"
        if findings_have_targets(data):
            return (
                D1_CORRELATIONAL,
                "Analyses new empirical data with inferential/predictive modelling",
            )
        return (
            D1_CORRELATIONAL,
            "Analyses new empirical ILSA (or related) data",
        )

    if source in ("methodology_paper", "technical_report") and not empirical:
        label = (
            "Methodological/technical paper with no original data analysis"
            if source == "methodology_paper"
            else "Technical/framework document with no original inferential analysis"
        )
        return D1_NONE, label
    if source == "review_article" or _SYNTHESIS_HINT_RE.search(text):
        return (
            D1_SYNTHESIS,
            "Summarises or critiques existing literature without new microdata analysis",
        )
    if _NO_DATA_ANALYSIS_RE.search(text):
        return D1_NONE, "Methodological/technical paper with no original data analysis"
    return D1_NONE, "No empirical inference detected"


def findings_have_targets(data: dict[str, Any]) -> bool:
    for finding in data.get("main_findings") or []:
        if (
            isinstance(finding, dict)
            and finding.get("target_variable")
            and not _finding_is_placeholder(finding)
        ):
            return True
    return False


def _metrics_text(article: dict[str, Any]) -> str:
    data = article.get("data") or {}
    chunks: list[str] = [str(data.get("outcome_summary") or "")]
    for finding in data.get("main_findings") or []:
        if isinstance(finding, dict) and not _finding_is_placeholder(finding):
            chunks.append(str(finding.get("performance_metrics") or ""))
            chunks.append(str(finding.get("standardized_conclusion") or ""))
    return "\n".join(chunks)


def _strip_non_finding_numbers(text: str) -> str:
    """Drop years and n=/N= sample sizes so they do not count as D2."""
    text = _YEAR_ONLY_RE.sub(" ", text)
    text = re.sub(r"\b[nN]\s*=\s*[\d,]+\b", " ", text)
    text = re.sub(r"\bgrade\s*\d+\b", " ", text, flags=re.IGNORECASE)
    return text


def classify_d2(article: dict[str, Any], text: str) -> tuple[str, str]:
    metrics_raw = _metrics_text(article)
    metrics = _strip_non_finding_numbers(metrics_raw)
    data = article.get("data") or {}

    quantified_bits: list[str] = []
    for finding in data.get("main_findings") or []:
        if not isinstance(finding, dict) or _finding_is_placeholder(finding):
            continue
        pm = (finding.get("performance_metrics") or "").strip()
        if pm and not _NOT_REPORTED_RE.match(pm):
            stripped = _strip_non_finding_numbers(pm)
            if _FINDING_NUMBER_RE.search(stripped) or _METRIC_RE.search(pm):
                if re.search(r"\d", stripped):
                    quantified_bits.append(pm[:220])

    if not quantified_bits:
        if _FINDING_NUMBER_RE.search(metrics) and _METRIC_RE.search(metrics):
            m = _FINDING_NUMBER_RE.search(metrics)
            quantified_bits.append((m.group(0) if m else metrics[:180]).strip())
        elif re.search(
            r"\b(?:accuracy|auc|f1|rmse|mae|r²|r2|ate)\b.{0,40}\d",
            metrics,
            re.IGNORECASE | re.DOTALL,
        ):
            quantified_bits.append(metrics[:220].replace("\n", " "))

    if quantified_bits:
        return D2_QUANTIFIED, quantified_bits[0].replace("\n", " ")

    if _DIRECTIONAL_RE.search(text) or findings_have_targets(data):
        return (
            D2_DIRECTIONAL,
            "States direction or relative importance of effects without a numerical finding metric",
        )
    return D2_NONE, "No empirical findings with direction or magnitude are reported"


def classify_d3(article: dict[str, Any], text: str) -> tuple[str, str]:
    data = article.get("data") or {}
    sd = data.get("sample_details") or {}
    countries = sd.get("countries") or []
    named = []
    for c in countries:
        if isinstance(c, dict) and c.get("country_code"):
            code = str(c["country_code"]).strip().upper()
            if _ISO3_RE.match(code) and code not in {"XXX", "NA", "N/A"}:
                named.append(code)
    if named:
        n = sd.get("total_students")
        extra = f", n={n}" if n else ""
        return D3_BOUNDED, f"Named countries/economies: {', '.join(named[:12])}{extra}"

    cycle = _ILSA_CYCLE_RE.search(text)
    if cycle:
        return D3_BOUNDED, f"Named ILSA/assessment context: {cycle.group(0)}"
    grade = _GRADE_RE.search(text)
    if grade:
        return D3_BOUNDED, f"Named grade/age group: {grade.group(0)}"
    group = _STUDENT_GROUP_RE.search(text)
    if group:
        return D3_BOUNDED, f"Named student group: {group.group(0)}"
    country = _COUNTRY_NAME_RE.search(text)
    if country:
        return D3_BOUNDED, f"Named population: {country.group(0)}"
    return D3_GLOBAL, "No country, ILSA cycle, grade, or student group is named"


def apply_decision_rule(d1: str, d2: str, d3: str) -> tuple[int, str, bool]:
    """Return (score, justification, unmatched). First matching rule wins."""
    if d1 == D1_CAUSAL and d2 == D2_QUANTIFIED and d3 == D3_BOUNDED:
        return 5, "Rule 1: Causal + Quantified + Bounded.", False
    if d2 == D2_QUANTIFIED and d3 == D3_BOUNDED:
        return 4, "Rule 2: Quantified + Bounded (D1 does not matter).", False
    if d1 == D1_CORRELATIONAL and (d2 == D2_DIRECTIONAL or d3 == D3_GLOBAL):
        return 3, "Rule 3: Correlational with directional-only findings or unbounded population.", False
    if d1 == D1_SYNTHESIS or d2 == D2_DIRECTIONAL or (
        d3 == D3_GLOBAL and d1 in (D1_NONE, D1_SYNTHESIS)
    ):
        return 2, "Rule 4: Synthesis, directional-only, or unbounded with no empirical inference.", False
    if d1 == D1_NONE:
        return 1, "Rule 5: D1 = None (no empirical inference).", False
    # Implied remainder (not in the five-rule chain, e.g. Causal + Directional + Bounded).
    if d2 == D2_QUANTIFIED:
        return 4, "Fallback: quantified findings (unmatched five-rule chain).", True
    if d1 in (D1_CORRELATIONAL, D1_CAUSAL):
        return 3, "Fallback: empirical inference without a full Rule 1–2 match.", True
    return 2, "Fallback: no primary decision-rule match.", True


def score_article(article: dict[str, Any]) -> dict[str, Any]:
    text = _blob(article)
    d1, d1_ev = classify_d1(article, text)
    d2, d2_ev = classify_d2(article, text)
    d3, d3_ev = classify_d3(article, text)
    score, justification, unmatched = apply_decision_rule(d1, d2, d3)
    meta = article.get("metadata") or {}
    result = {
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "Score": score,
        "D1_evidence": d1_ev,
        "D2_evidence": d2_ev,
        "D3_evidence": d3_ev,
        "Justification": justification,
    }
    result["title"] = meta.get("title")
    result["file_name"] = meta.get("file_name")
    result["doi"] = meta.get("doi")
    result["year"] = meta.get("year")
    if unmatched:
        result["unmatched_rule_chain"] = True
    return result


def score_json_file(path: Path) -> dict[str, Any]:
    article = json.loads(path.read_text(encoding="utf-8"))
    result = score_article(article)
    result["source_json"] = path.name
    return result


def score_corpus(json_dir: Path) -> list[dict[str, Any]]:
    files = sorted(json_dir.glob("*.json"))
    return [score_json_file(p) for p in files]
