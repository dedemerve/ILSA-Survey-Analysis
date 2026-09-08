"""
Policy Actionability Scorer for ILSA Survey articles.
Three-dimension rubric: D1 (Inferential Warrant), D2 (Effect Specification), D3 (Population Boundedness).

Rules: FIX-I, FIX-J, FIX-K plus:
  FIX-DESIGN-CAUSAL: research_design_type=causal_observational → D1=Causal
  FIX-DESIGN-ML: research_design_type in {predictive, exploratory} → D1=Correlational when ml_techniques is empty
  FIX-METH-ALWAYS: methodology_paper (regardless of has_ml) → D1=Synthesis
  FIX-SYNTH-D2: Synthesis papers use conservative D2 detection (no Quantified from outcome_summary)
"""
import csv
import glob
import json
import os
import re
import sys

JSON_DIR = os.path.join(os.path.dirname(__file__), "ilsa_survey_articles", "json")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "rescored_studies.csv")

CAUSAL_KEYWORDS = re.compile(
    r"\b(bart|bcf|causal[\s_-]?forest|difference[\s-]in[\s-]differences?|\bdid\b"
    r"|regression[\s_-]discontinuity|\brdd\b|instrumental[\s_-]variable|\biv\b"
    r"|propensity[\s_-]score[\s_-]match|\bpsm\b|counterfactual)",
    re.IGNORECASE,
)

SYNTHESIS_SOURCE_CATS = {
    "review_article", "systematic_review", "meta_analysis",
    "literature_review", "scoping_review",
}

# Methodology/non-empirical papers → D1=Synthesis regardless of has_ml
METHODOLOGY_SOURCE_CATS = {
    "methodology_paper", "framework_paper", "editorial",
    "theoretical_paper", "commentary", "opinion_piece",
    "technical_report",
}

_DIRECTIONAL_RE = re.compile(
    r"\b(increas|decreas|higher|lower|positiv|negativ|improv|reduc|outperform|better|worse"
    r"|predicts?|associat|influen|effect|contribut|significan|argu|suggest"
    r"|synthes|report|present|identif|reveal|show|demonstrat|find|found|indicat)\w*",
    re.IGNORECASE,
)
# Conservative version for Synthesis papers: excludes "argu" (advocacy papers) and "report/present/show"
# (too common as non-finding nouns). Relies on more unambiguous finding verbs only.
_DIRECTIONAL_SYNTH_RE = re.compile(
    r"\b(increas|decreas|higher|lower|positiv|negativ|improv|reduc|outperform|better|worse"
    r"|predicts?|associat|influen|effect|contribut|significan|suggest"
    r"|synthes|identif|reveal|demonstrat|find|found|indicat)\w*",
    re.IGNORECASE,
)
# Strong markers: paper explicitly disclaims findings — checked BEFORE directional signals
_NO_EMPIRICAL_RE = re.compile(
    r"does not train|not a[n]? ml|no empirical"
    r"|does not estimate"                      # e.g. "does not estimate predictive or causal effects"
    r"|is a methodological contribution",      # e.g. "This article is a methodological contribution"
    re.IGNORECASE,
)
# Weak markers: caveats LLM appends even to papers with synthesis findings — checked AFTER directional
_WEAK_NO_EMPIRICAL_RE = re.compile(
    r"not an empirical|no predictive",
    re.IGNORECASE,
)
_QUANTIFIED_RE = re.compile(
    r"\d+\.?\d*\s*%|r²|r\^2|\bauc\b|\brmse\b|\bf1\b|accuracy|\bcoefficient\b|\bbeta\b"
    r"|\bacc\b|\bmse\b|\bmae\b|odds\s*ratio|effect\s*size|p\s*[=<>]\s*[\d.]|\baic\b|\bbic\b|silhouette"
    r"|precision|recall|f-score|kappa|concordance",
    re.IGNORECASE,
)


def _has_ml(data: dict) -> bool:
    ml = data.get("ml_techniques")
    if not ml:
        return False
    primary = ml.get("primary")
    all_tech = ml.get("all_techniques", [])
    return bool(primary) or bool(all_tech)


def _is_causal(data: dict) -> bool:
    design = (data.get("research_design_type") or "").lower()
    if design == "causal_observational" or "causal" in design:
        return True
    ml = data.get("ml_techniques") or {}
    primary = ml.get("primary") or ""
    all_tech = " ".join(ml.get("all_techniques") or [])
    return bool(CAUSAL_KEYWORDS.search(f"{primary} {all_tech}"))


def _d1(source_cat: str, data: dict) -> str:
    if source_cat in SYNTHESIS_SOURCE_CATS:
        return "Synthesis"
    # FIX-METH-ALWAYS: methodology papers → Synthesis regardless of has_ml
    if source_cat in METHODOLOGY_SOURCE_CATS:
        return "Synthesis"

    if _is_causal(data):
        return "Causal"
    if _has_ml(data):
        return "Correlational"

    # FIX-DESIGN-ML: infer from research_design_type when ml_techniques is empty
    design = (data.get("research_design_type") or "").lower()
    if design in {"predictive", "exploratory", "causal_observational"}:
        return "Correlational"

    return "None"


def _d2_for_synthesis(findings: list, summary: str) -> str:
    """Conservative D2 for Synthesis papers: no Quantified from outcome_summary."""
    # Only Quantified from performance_metrics
    for f in findings:
        perf = (f.get("performance_metrics") or "").strip()
        lower_perf = perf.lower()
        if perf and lower_perf not in ("", "not_reported", "n/a", "na", "null", "none"):
            if _QUANTIFIED_RE.search(perf):
                return "Quantified"
    # Strong negation: paper explicitly disclaims any findings → D2=None immediately
    if _NO_EMPIRICAL_RE.search(summary):
        return "None"
    # Directional: check summary first, then standardized conclusions as fallback
    if _DIRECTIONAL_SYNTH_RE.search(summary):
        return "Directional"
    for f in findings:
        conclusion = (f.get("standardized_conclusion") or "").strip()
        if _DIRECTIONAL_SYNTH_RE.search(conclusion):
            return "Directional"
    # Weak negation: LLM caveat that may appear even in papers with synthesis findings
    if _WEAK_NO_EMPIRICAL_RE.search(summary):
        return "None"
    return "None"


def _d2(data: dict, d1: str) -> str:
    findings = data.get("main_findings") or []
    summary = (data.get("outcome_summary") or "")

    if d1 == "Synthesis":
        return _d2_for_synthesis(findings, summary)

    # Non-synthesis: full detection
    for f in findings:
        perf = (f.get("performance_metrics") or "").strip()
        lower_perf = perf.lower()
        if perf and lower_perf not in ("", "not_reported", "n/a", "na", "null", "none"):
            if _QUANTIFIED_RE.search(perf):
                return "Quantified"
            if re.search(r"\d", perf):
                return "Quantified"

    # Check standardized_conclusion for quantified signals
    for f in findings:
        conclusion = (f.get("standardized_conclusion") or "").strip()
        if _QUANTIFIED_RE.search(conclusion):
            return "Quantified"

    # Fall back to outcome_summary for quantified
    if _QUANTIFIED_RE.search(summary):
        return "Quantified"

    # Directional signals
    for f in findings:
        conclusion = (f.get("standardized_conclusion") or "").strip()
        if _DIRECTIONAL_RE.search(conclusion):
            return "Directional"

    if _DIRECTIONAL_RE.search(summary):
        return "Directional"

    return "None"


def _country_count(data: dict) -> int:
    sample = data.get("sample_details") or {}
    return len(sample.get("countries") or [])


def _d3(data: dict) -> str:
    # Any explicitly listed countries = Bounded (even 10+), 0 = Global/Unspecified
    n = _country_count(data)
    if n == 0:
        return "Global/Unspecified"
    return "Bounded"


def score(d1: str, d2: str, d3: str) -> int:
    # FIX-K: D1=Synthesis/None + D2=None + D3=Global → Score 1
    if d1 in ("Synthesis", "None") and d2 == "None" and d3 == "Global/Unspecified":
        return 1
    # Rule 5
    if d1 == "None":
        return 1
    # Rule 1
    if d1 == "Causal" and d2 == "Quantified" and d3 == "Bounded":
        return 5
    # FIX-I: Synthesis skips Rule 2 → falls to Rule 4
    # Rule 2 (non-Synthesis): Quantified results → Score 4 regardless of D3
    # (Global/multi-country ILSA studies with quantified ML metrics are equally actionable)
    if d1 not in ("Synthesis", "None") and d2 == "Quantified":
        return 4
    # Rule 3: Directional only (D3 no longer demotes Correlational+Quantified)
    if d1 == "Correlational" and d2 == "Directional":
        return 3
    # Rule 4
    if d1 in ("Synthesis", "None") or d2 == "Directional" or d3 == "Global/Unspecified":
        return 2
    return 2


def process_file(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)

    meta = doc.get("metadata", {})
    data = doc.get("data", {})

    source_cat = (meta.get("source_category") or "").lower().strip()
    d1 = _d1(source_cat, data)
    d2 = _d2(data, d1)
    d3 = _d3(data)
    s = score(d1, d2, d3)

    return {
        "file": os.path.basename(path),
        "title": meta.get("title", ""),
        "source_category": source_cat,
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "n_countries": _country_count(data),
        "score": s,
    }


def main():
    files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    if not files:
        print(f"No JSON files found in {JSON_DIR}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in files:
        try:
            rows.append(process_file(f))
        except Exception as e:
            print(f"ERROR {f}: {e}", file=sys.stderr)

    fieldnames = ["file", "title", "source_category", "D1", "D2", "D3", "n_countries", "score"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scored {len(rows)} studies → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
