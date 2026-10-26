"""
Rule-based derivation of the study attributes RQ1 needs but the extraction
schema stored only as free text.

Derived per finding (``main_findings`` row):
  * ilsa_programs      — PISA, TIMSS, PIRLS, TALIS, PIAAC, ICILS, ICCS, PISA-VET, regional ILSAs
  * cycles             — assessment years mentioned next to a program (e.g. "PISA 2018")
  * domain             — Mathematics, Reading, Science, Problem solving, … or Non-cognitive / Teacher / Measurement
  * target_type        — achievement_score | achievement_category | non_cognitive | teacher_school |
                         item_measurement | process_behavior | synthesis | other
  * is_achievement     — target_type in {achievement_score, achievement_category}
  * analysis_level     — student | teacher | school | country | item | literature
  * task_type          — regression | classification | clustering | not_determinable
  * validation_scheme  — k_fold_cv | repeated_split | train_test_split | group_cv |
                         temporal_holdout | external | not_reported

The rules only read what the extraction already contains (dataset_used,
target_variable, performance_metrics, standardized_conclusion, outcome_summary,
title). Nothing is imputed: when no rule fires the value is ``not_determinable``
(or an empty list). These are coding aids — the human-validation sample produced by
``scripts/validation_sample.py`` measures their agreement with manual coding.
"""

from __future__ import annotations

import re
from typing import Any

PROGRAM_PATTERNS: list[tuple[str, str]] = [
    ("PISA-VET", r"\bpisa[\s-]?vet\b"),
    ("PISA", r"\bpisa\b"),
    ("TIMSS", r"\be?timss\b"),
    ("PIRLS", r"\b(?:e|e-)?pirls\b|\bprepirls\b"),
    ("TALIS", r"\btalis\b"),
    ("PIAAC", r"\bpiaac\b"),
    ("ICILS", r"\bicils\b"),
    ("ICCS", r"\biccs\b"),
    ("LLECE/ERCE", r"\b(?:llece|terce|serce|erce)\b"),
    ("SACMEQ", r"\bsacmeq\b"),
    ("PASEC", r"\bpasec\b"),
    ("NAEP", r"\bnaep\b"),
]
_PROGRAM_RE = [(name, re.compile(p, re.IGNORECASE)) for name, p in PROGRAM_PATTERNS]

# Program token followed (within a few chars) by a 4-digit year or a year range.
_CYCLE_RE = re.compile(
    r"\b(pisa|e?timss|pirls|talis|piaac|icils|iccs)\b[^0-9]{0,12}"
    r"((?:19|20)\d{2})(?:\s*(?:[-–—/&,]|and|to)\s*((?:19|20)\d{2}))?",
    re.IGNORECASE,
)

DOMAIN_RULES: list[tuple[str, str]] = [
    ("Teacher outcome", r"\bteacher|job satisfaction|satjob|satteach|\bself-efficacy of teachers|principal|administrator|school leader"),
    ("Collaborative problem solving", r"collaborative problem|\bcps\b"),
    ("Problem solving", r"problem[\s-]solving|\bps-?tre\b|\bpstre\b|complex problem"),
    ("Creative thinking", r"creativ"),
    ("Financial literacy", r"financial literacy"),
    ("Global competence", r"global competen|intercultural"),
    ("Computer & information literacy", r"\bcil\b|computer and information literacy|digital (?:skill|literacy|competenc)|\bct\b computational|computational thinking"),
    ("Civic knowledge", r"civic|citizenship"),
    ("Numeracy", r"numeracy"),
    ("Mathematics", r"\bmath|mathemat|\bpvmath|pv\d*math"),
    ("Reading", r"\bread|literacy in reading|pv\d*read|comprehension"),
    ("Science", r"\bscien|pv\d*scie"),
]
_DOMAIN_RE = [(name, re.compile(p, re.IGNORECASE)) for name, p in DOMAIN_RULES]

_SYNTHESIS_RE = re.compile(r"literature synthesis|systematic review|meta-analys|scoping review|bibliometric", re.I)
_TEACHER_RE = re.compile(r"\bteacher|\bprincipal|administrator|job satisfaction|satjob|satteach|school climate|talis", re.I)
_ITEM_RE = re.compile(
    r"\bitem\b|\bdif\b|differential item|automated scor|human rater|rater agreement|scoring|"
    r"item difficulty|correctness|authorship|response time|irt|scaling model|measurement invariance",
    re.I,
)
_PROCESS_RE = re.compile(r"process data|action sequence|log data|clickstream|response process|engagement|disengage|strateg", re.I)
_NONCOG_RE = re.compile(
    r"well-?being|life satisfaction|self-?efficacy|anxiety|belonging|motivation|attitude|"
    r"environmental action|absente|bullying|aspiration|expectation|interest|enjoyment|"
    r"grade repetition|dropout|truancy|ai usage|ai use|civic engagement|happiness|emotion",
    re.I,
)
_ACH_RE = re.compile(
    r"achiev|perform|literacy|score|proficien|plausible|\bpvs?\b|pv\d|success|competenc|"
    r"resilien|mathemat|\bmath|reading|science|high[\s-]?perform|low[\s-]?perform|benchmark|level [1-6]|ability",
    re.I,
)
_CATEGORY_RE = re.compile(
    r"binary|class(?:es|ification)?\b|categor|quartile|tercile|level|high vs|low vs|vs\.? low|"
    r"successful|unsuccessful|resilien|top \d|benchmark|\(low/medium/high\)|above|below",
    re.I,
)
_COUNTRY_LEVEL_RE = re.compile(
    r"country[\s-]level|country means?|national means?|country averages?|cross-country (?:panel|test scores)|"
    r"harmoni[sz]ed (?:cross-country )?test scores|macro[\s-]?(?:level|economic)|education system level|system-level",
    re.I,
)
_SCHOOL_LEVEL_RE = re.compile(r"school[\s-]level (?:outcome|efficiency|achievement)|school inefficiency|school effectiveness|school means?", re.I)

_CLASSIF_METRIC_RE = re.compile(r"\bauc\b|roc|accuracy|\bacc\b|\bf1\b|f-score|precision|recall|sensitivity|specificity|kappa|balanced accuracy", re.I)
_REG_METRIC_RE = re.compile(r"\brmse\b|\bmae\b|\bmse\b|\bmape\b|r²|r\^2|\br2\b|r-squared|explained variance", re.I)
_CLUSTER_RE = re.compile(r"silhouette|cluster|latent profile|latent class|davies|calinski", re.I)

VALIDATION_RULES: list[tuple[str, str]] = [
    ("temporal_holdout", r"temporal (?:hold-?out|validation)|out-of-time|forecast(?:ing)? (?:origin|window)|later cycle|next cycle"),
    ("group_cv", r"leave[\s-]one[\s-](?:group|country|school)[\s-]out|\blogo\b|grouped? (?:k-?fold )?cv|group k-?fold"),
    ("external", r"external validation|independent (?:test )?sample|out-of-sample country|cross-country validation"),
    ("repeated_split", r"repeated (?:\d+ times|random|80/20|train|split)|\d+ repeated|monte carlo cross"),
    ("k_fold_cv", r"\d+[\s-]?fold|k[\s-]?fold|cross[\s-]?validat|\bcv\b"),
    ("train_test_split", r"\d{2}\s*[/:]\s*\d{2}|train(?:ing)?[\s/-]+(?:and )?test|hold[\s-]?out|test set|validation set|training set"),
]
_VALIDATION_RE = [(name, re.compile(p, re.IGNORECASE)) for name, p in VALIDATION_RULES]


def _text(*parts: Any) -> str:
    return " ".join(str(p) for p in parts if p)


def detect_programs(text: str) -> list[str]:
    found: list[str] = []
    for name, pattern in _PROGRAM_RE:
        if pattern.search(text):
            if name == "PISA" and "PISA-VET" in found and not re.search(r"\bpisa\b(?![\s-]?vet)", text, re.I):
                continue
            found.append(name)
    return found


def detect_cycles(text: str) -> list[str]:
    cycles: set[str] = set()
    for m in _CYCLE_RE.finditer(text):
        prog = m.group(1).upper().lstrip("E") if m.group(1).lower() == "etimss" else m.group(1).upper()
        start = int(m.group(2))
        end = m.group(3)
        cycles.add(f"{prog} {start}")
        if end:
            cycles.add(f"{prog} {int(end)}")
    return sorted(cycles)


def detect_domain(target: str, dataset: str) -> str:
    """Domain from the target first (what is predicted), then from the dataset label."""
    for source in (target, dataset):
        hits = [name for name, pattern in _DOMAIN_RE if pattern.search(source or "")]
        if hits:
            cognitive = {"Mathematics", "Reading", "Science"}
            if len(set(hits) & cognitive) > 1 and source is target:
                return "Multiple domains"
            return hits[0]
    return "not_determinable"


def classify_target(target: str, dataset: str, conclusion: str, source_category: str) -> str:
    joined = _text(target, dataset)
    if _SYNTHESIS_RE.search(joined) or (
        source_category in {"review_article"} and not _ACH_RE.search(target or "")
    ):
        return "synthesis"
    if _TEACHER_RE.search(target or "") or (re.search(r"\btalis\b", dataset or "", re.I) and not _ACH_RE.search(target or "")):
        return "teacher_school"
    if _ITEM_RE.search(target or ""):
        return "item_measurement"
    if _NONCOG_RE.search(target or "") and not re.search(r"achiev|performance|literacy|score|resilien", target or "", re.I):
        return "non_cognitive"
    if _PROCESS_RE.search(target or "") and not re.search(r"achiev|performance|success|ability|score", target or "", re.I):
        return "process_behavior"
    if _ACH_RE.search(target or ""):
        return "achievement_category" if _CATEGORY_RE.search(target or "") else "achievement_score"
    if _NONCOG_RE.search(target or ""):
        return "non_cognitive"
    return "other"


def detect_analysis_level(target: str, dataset: str, target_type: str, title: str) -> str:
    joined = _text(target, dataset, title)
    if target_type == "synthesis":
        return "literature"
    if _COUNTRY_LEVEL_RE.search(_text(target, dataset)):
        return "country"
    if target_type == "item_measurement":
        return "item"
    if target_type == "teacher_school":
        return "teacher" if re.search(r"teacher|satjob|satteach", target or "", re.I) else "school"
    if _SCHOOL_LEVEL_RE.search(joined):
        return "school"
    return "student"


def detect_task_type(metrics: str, target: str, target_type: str) -> str:
    if target_type == "achievement_category":
        return "classification"
    has_cls = bool(_CLASSIF_METRIC_RE.search(metrics or ""))
    has_reg = bool(_REG_METRIC_RE.search(metrics or ""))
    if has_cls and not has_reg:
        return "classification"
    if has_reg and not has_cls:
        return "regression"
    if has_cls and has_reg:
        return "classification" if _CATEGORY_RE.search(target or "") else "regression"
    if _CLUSTER_RE.search(_text(metrics, target)):
        return "clustering"
    if target_type == "achievement_score":
        return "regression"
    return "not_determinable"


def detect_validation(*texts: str) -> str:
    joined = _text(*texts)
    for name, pattern in _VALIDATION_RE:
        if pattern.search(joined):
            return name
    return "not_reported"


def derive_finding_attributes(
    finding: dict[str, Any],
    *,
    meta: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    dataset = str(finding.get("dataset_used") or "")
    target = str(finding.get("target_variable") or "")
    metrics = str(finding.get("performance_metrics") or "")
    conclusion = str(finding.get("standardized_conclusion") or "")
    title = str(meta.get("title") or "")
    summary = str(data.get("outcome_summary") or "")
    source_category = str(meta.get("source_category") or "")

    programs = detect_programs(_text(dataset, target)) or detect_programs(_text(title, summary))
    target_type = classify_target(target, dataset, conclusion, source_category)
    return {
        "ilsa_programs": programs,
        "cycles": detect_cycles(_text(dataset, target, title)),
        "domain": detect_domain(target, dataset),
        "target_type": target_type,
        "is_achievement": target_type in {"achievement_score", "achievement_category"},
        "analysis_level": detect_analysis_level(target, dataset, target_type, title),
        "task_type": detect_task_type(metrics, target, target_type),
        "validation_scheme": detect_validation(metrics, summary),
    }


def derive_article_attributes(meta: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Article-level roll-up of the finding-level attributes."""
    findings = [f for f in (data.get("main_findings") or []) if isinstance(f, dict)]
    rows = [derive_finding_attributes(f, meta=meta, data=data) for f in findings]
    title = str(meta.get("title") or "")
    summary = str(data.get("outcome_summary") or "")
    programs: list[str] = []
    for r in rows:
        for p in r["ilsa_programs"]:
            if p not in programs:
                programs.append(p)
    if not programs:
        programs = detect_programs(_text(title, summary))
    domains = sorted({r["domain"] for r in rows if r["domain"] != "not_determinable"})
    levels = sorted({r["analysis_level"] for r in rows})
    tasks = sorted({r["task_type"] for r in rows if r["task_type"] != "not_determinable"})
    validations = [r["validation_scheme"] for r in rows if r["validation_scheme"] != "not_reported"]
    return {
        "ilsa_programs": programs,
        "cycles": sorted({c for r in rows for c in r["cycles"]}),
        "domains": domains,
        "any_achievement_target": any(r["is_achievement"] for r in rows),
        "analysis_levels": levels,
        "task_types": tasks,
        "validation_scheme": validations[0] if validations else detect_validation(summary),
    }
