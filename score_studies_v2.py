#!/usr/bin/env python3
"""
Apply policy actionability rubric to all ILSA survey article JSON files.
v2 — corrected based on systematic stress-test findings:
  FIX-1  Exact-match source_category for Synthesis detection
         (prevents 'peer_reviewed_research' from triggering via 'review' substring)
  FIX-2  PSM / DiD / RDD etc. also detected from performance_metrics field
         (only for unambiguous method names that cannot appear in boilerplate)
  FIX-3  source_category='methodology_paper' with no techniques → D1=None
  FIX-4  D2=Quantified requires real metric patterns, not bare number presence
         (stops year numbers / descriptive counts from triggering Quantified)
"""

import json
import os
import re
import csv

JSON_DIR  = "/home/user/ILSA-Survey-Analysis/ilsa_survey_articles/json"
OUTPUT_CSV = "/home/user/ILSA-Survey-Analysis/rescored_studies.csv"

# ── Causal methods to detect in ml_techniques (structured field) ──────────────
CAUSAL_IN_TECHNIQUES = [
    'bart', 'bayesian additive regression tree',
    'bcf', 'bayesian causal forest',
    'causal forest', 'generalized random forest', 'grf',
    'difference-in-differences', 'diff-in-diff',
    'regression discontinuity',
    'instrumental variable',
    'propensity score matching', 'psm',
    'fixed effects',
    'counterfactual',
    'treatment effect',
    'interventional shap',
    'average treatment effect', 'ate',
    'bartcause',
]

# Causal method names that are unambiguous enough to detect from performance_metrics
# (these CANNOT appear in the boilerplate "cannot establish causality" disclaimer)
# FIX-2: extend causal detection beyond ml_techniques for these specific phrases
# FIX-D: removed 'propensity score matching' and 'propensity score' — PSM is often
# a secondary validation technique, not the primary causal method; only count it
# when explicitly listed in ml_techniques (CAUSAL_IN_TECHNIQUES handles that)
CAUSAL_IN_METRICS = [
    'difference-in-differences',
    'regression discontinuity',
    'instrumental variable',
    'average treatment effect',
    'causal forest',
    'bartcause',
    'bcf',
]

# research_design_type values that signal a review/synthesis
REVIEW_DESIGN_TYPES = {
    'review', 'synthesis', 'meta-analysis', 'bibliometric',
    'systematic_review', 'literature_review', 'scoping_review',
    'systematic review', 'literature review', 'scoping review',
}

# FIX-1: EXACT match on source_category (no substring matching)
REVIEW_SOURCE_CATS = {
    'review_article', 'systematic_review', 'meta_analysis',
    'literature_review', 'scoping_review',
}

# source_category values that indicate no empirical data analysis at all
# FIX-3: methodology / framework papers treated as D1=None when no techniques
NON_EMPIRICAL_CATS = {
    'methodology_paper', 'framework_paper', 'editorial',
    'theoretical_paper', 'commentary', 'opinion_piece',
}

# Reliable metric keyword patterns for D2=Quantified (FIX-4)
# A finding must contain one of these to be considered truly Quantified
METRIC_PATTERNS = [
    r'\baccuracy\s*[=:]\s*[\d.]',
    r'\bauc\s*[=:]?\s*[\d.]',
    r'\bf1[\s\-]?score\s*[=:]?\s*[\d.]',
    r'\bf\s*[\-]?measure\s*[=:]?\s*[\d.]',
    r'\brmse\s*[=:]?\s*[\d.]',
    r'\bmae\s*[=:]?\s*[\d.]',
    r'\bmse\s*[=:]?\s*[\d.]',
    r'\br[²2]\s*[=:]?\s*[\d.]',
    r'\bprecision\s*[=:]?\s*[\d.]',
    r'\brecall\s*[=:]?\s*[\d.]',
    r'\bkappa\s*[=:]?\s*[\d.]',
    r'\bcohen',
    r'\bpearson\s+r\s*[=:]?\s*[\d.]',
    r'\bspearman',
    r'\bcorrelation\s*[=:]?\s*-?[\d.]',
    r'\bβ\s*[=:]?\s*-?[\d.]',
    r'\bbeta\s*[=:]?\s*-?[\d.]',
    r'\bodds\s+ratio',
    r'\bate\s*[=:]?\s*-?[\d.]',
    r'\brmsd\s*[=:]?\s*[\d.]',
    r'\bate\b.*\d+\.\d+',
    r'r\s*=\s*[\-−]?\s*\d+\.\d+',
    r'\d+\.?\d*\s*%\s*(accuracy|f1|auc|recall|precision)',
    r'accuracy\s+[\d.]+',
    r'auc\s+[\d.]+',
    # Metric abbreviations used as key results even when exact value is in a figure/table
    r'\b(?:mse|mae)\b',
    # IRT / psychometric
    r'\brmse[a]?\s*[=:]?\s*[\d.]',
    r'\bsrmr\s*[=:]?\s*[\d.]',
    r'\bicc\s*[=:]?\s*[\d.]',
    r'\bentropy\s*[=:]?\s*[\d.]',
    r'\bbic\s*[=:]?\s*[\d.]',
    r'\baic\s*[=:]?\s*[\d.]',
    # Effect sizes
    r"\bcohen'?s?\s+d\s*[=:]?\s*-?[\d.]",
    r'\beffect\s+size\s*[=:]?\s*[\d.]',
    r'\bvar(?:iance)?\s+explained\s*[=:]?\s*[\d.]',
    r'\bp\s*<\s*[\d.]',
    r'\bp\s*=\s*[\d.]',
    r'\bchi.?squ',
    r'\bf\s*\(\s*\d',
    r'\bt\s*\(\s*\d',
    r'\bse\s*[=:]?\s*[\d.]',
    r'explained\s+\d+',
    r'variable\s+importance',
    r'feature\s+importance',
    r'\bshap\b',
    r'%\s*inc\s*mse',
    r'sensitivity\s+[\d.]',
    r'specificity\s+[\d.]',
    r'\bG-mean\s',
    r'class\s+size',
    r'cluster.*sil',
    r'silhouette',
    # Regression coefficients
    r'est\.\s*=\s*-?[\d.]',
    r'coeff?icient\s*[=:]?\s*-?[\d.]',
    r'residual.*\d+',
    r'median.*\d+',
    # General: "X = number" style — negative lookahead to avoid M=, SD=, SE=, n=, N=
    r'(?<![MmNnSs])(?<![SD])=\s*-?0\.\d+',
    r'(?<![MmNnSs])(?<![SD])=\s*-?\d+\.\d+',
]
_METRIC_RE = re.compile('|'.join(METRIC_PATTERNS), re.IGNORECASE)


def has_real_metric(text: str) -> bool:
    """Return True if text contains a genuine performance/effect metric."""
    if not text:
        return False
    # Exclude clearly boilerplate text
    cleaned = str(text)
    if cleaned.strip().lower() in {
        'not reported', 'not applicable', 'none', '', 'null', 'n/a',
        'no metrics reported',
    }:
        return False
    return bool(_METRIC_RE.search(cleaned))


def text_contains(text: str, keywords) -> bool:
    t = str(text).lower()
    return any(kw in t for kw in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# D1 — Inferential Warrant
# ─────────────────────────────────────────────────────────────────────────────
def extract_d1(data: dict, metadata: dict):
    ml       = data.get('ml_techniques') or {}
    primary  = (ml.get('primary') or '').strip()
    all_tech = [str(t).strip() for t in (ml.get('all_techniques') or [])]

    has_technique = bool(
        (primary and primary.lower() not in {'none', 'null', ''}) or
        [t for t in all_tech if t.lower() not in {'none', 'null', ''}]
    )

    all_tech_lower  = ' '.join(t.lower() for t in all_tech) + ' ' + primary.lower()
    research_design = str(data.get('research_design_type') or '').lower()
    source_cat      = str(metadata.get('source_category') or '').lower()
    findings        = data.get('main_findings') or []

    # ── Is this a review/synthesis? ──────────────────────────────────────────
    # FIX-1: exact set membership, never substring match
    is_review_design = research_design in REVIEW_DESIGN_TYPES or \
                       research_design.replace('-', '_') in REVIEW_DESIGN_TYPES
    is_review_cat    = source_cat in REVIEW_SOURCE_CATS   # exact match only

    is_synthesis = is_review_design or is_review_cat

    # A true review has no ML techniques AND is tagged as a review
    if is_synthesis and not has_technique:
        return ('Synthesis',
                f"Review/synthesis: source_category='{metadata.get('source_category')}', "
                f"design='{data.get('research_design_type')}'.")

    # ── FIX-3: methodology/framework/editorial papers with no techniques ─────
    is_non_empirical_cat = source_cat in NON_EMPIRICAL_CATS
    # Also catch studies whose every finding targets "literature synthesis" or
    # "not student-level prediction" (inserted by the extraction pipeline for
    # papers that analyse no student-level outcome data)
    target_vars = [str(f.get('target_variable') or '').lower() for f in findings]
    all_targets_synthetic = bool(target_vars) and all(
        'literature synthesis' in tv or 'not student-level' in tv
        for tv in target_vars
    )

    if not has_technique and (is_non_empirical_cat or all_targets_synthetic):
        if is_synthesis or all_targets_synthetic:
            # FIX-E: purely theoretical / framework papers whose target_variable
            # was set to "Literature synthesis outcome (not student-level prediction)"
            # should be Synthesis, not Correlational, even when source_category is
            # methodology_paper.  Synthesis correctly routes them to Score ≤ 2 when
            # they have no quantified policy outcomes (e.g. ISM / conceptual models).
            return ('Synthesis',
                    f"Review/synthesis/theoretical: source_category='{metadata.get('source_category')}', "
                    f"all_targets_synthetic={all_targets_synthetic}, is_review_cat={is_review_cat}.")
        # FIX-J: methodology/framework papers with no ML technique cap at Score 2.
        # Even when they report analytical metrics (simulation stats, IRT design
        # comparisons, uncertainty estimates), they produce no student-level
        # predictive finding that can directly inform policy.  Treating them as
        # Synthesis routes them to Rule 4 → Score 2, matching manual scores.
        return ('Synthesis',
                f"Methodology/framework paper (no ML technique): capped at Score 2 "
                f"(source_category='{metadata.get('source_category')}', design='{data.get('research_design_type')}').")

    # ── Causal detection ─────────────────────────────────────────────────────
    # Primary check: structured ml_techniques + research_design_type
    structured_text = all_tech_lower + ' ' + research_design
    if text_contains(structured_text, CAUSAL_IN_TECHNIQUES):
        return ('Causal',
                f"Causal method in ml_techniques: primary='{primary}', "
                f"all={ml.get('all_techniques')}, design='{data.get('research_design_type')}'.")

    # FIX-2: Secondary check — unambiguous causal method names in performance_metrics
    # (these phrases cannot appear in the boilerplate disclaimer "cannot establish causality")
    metrics_text = ' '.join(
        str(f.get('performance_metrics') or '') for f in findings
    ).lower()
    if text_contains(metrics_text, CAUSAL_IN_METRICS):
        matched = next(cm for cm in CAUSAL_IN_METRICS if cm in metrics_text)
        return ('Causal',
                f"Causal method '{matched}' detected in performance_metrics "
                f"(ml_techniques primary='{primary}').")

    # ── Standard ML / statistical inference ──────────────────────────────────
    if has_technique or (findings and not all_targets_synthetic):
        return ('Correlational',
                f"Standard ML/statistical method: primary='{primary}', all={ml.get('all_techniques')}.")

    # ── No empirical inference ───────────────────────────────────────────────
    return ('None', "No technique and no empirical findings detected.")


# ─────────────────────────────────────────────────────────────────────────────
# D2 — Effect Specification
# ─────────────────────────────────────────────────────────────────────────────
def extract_d2(data: dict, d1_hint: str = ''):
    findings       = data.get('main_findings') or []
    outcome_summary = str(data.get('outcome_summary') or '')

    # Check performance_metrics first (most reliable)
    for f in findings:
        pm = str(f.get('performance_metrics') or '')
        if has_real_metric(pm):
            return ('Quantified',
                    f"Performance metric reported: '{pm[:200]}'")

    # FIX-4: For outcome_summary and standardized_conclusion, require
    # genuine metric patterns — not just any digit (avoids year numbers,
    # sample sizes, and boilerplate "Using PISA 2018" triggers)
    for f in findings:
        sc = str(f.get('standardized_conclusion') or '')
        # Skip pure boilerplate conclusions generated by the extraction pipeline
        if sc.startswith('Using ') and 'leveraged the reported predictors' in sc:
            # Only accept if there's a real metric pattern BEYOND the boilerplate header
            if has_real_metric(sc):
                return ('Quantified',
                        f"Metric in conclusion: '{sc[:200]}'")
        elif has_real_metric(sc):
            return ('Quantified',
                    f"Metric in conclusion: '{sc[:200]}'")

    if has_real_metric(outcome_summary):
        return ('Quantified',
                f"Metric in outcome_summary: '{outcome_summary[:200]}'")

    # RULE-1 (FIX-A/B): any decimal or percentage in outcome_summary / pm / sc
    # counts as Quantified. Requires 2+ decimal places to avoid bare years.
    _DECIMAL_OR_PCT = re.compile(
        r'\b\d+\.\d{2,}'           # decimal with 2+ places (e.g. 49.44, 0.02)
        r'|\b\d+\s*%'              # percentage (e.g. 67%)
        r'|(?<!\d)\d{1,3}(?:,\d{3})+\b'  # comma-formatted integers (e.g. 2,233)
    )
    all_fields_for_rule1 = [outcome_summary]
    for f in findings:
        all_fields_for_rule1.append(str(f.get('performance_metrics') or ''))
        all_fields_for_rule1.append(str(f.get('standardized_conclusion') or ''))
    combined_rule1 = ' '.join(all_fields_for_rule1)
    if _DECIMAL_OR_PCT.search(combined_rule1):
        return ('Quantified',
                f"Numerical finding (decimal/%/count) in study fields: '{outcome_summary[:200]}'")

    # RULE-1b: For synthesis/methodology papers only — non-year integers count as Quantified.
    # FIX-F: For Synthesis papers, require 3+ digit numbers. Structural counts like
    # "16 attributes" or "52 experts" (2-digit) must NOT trigger D2=Quantified.
    # Real outcomes (N=221, M=337) use 3-digit numbers or are caught by _DECIMAL_OR_PCT.
    if d1_hint in ('Synthesis', 'Correlational', 'None'):
        if d1_hint == 'Synthesis':
            _ANY_NUM_NONYR = re.compile(r'\b(?!(?:18|19|20)\d{2}\b)\d{3,}\b')
        else:
            _ANY_NUM_NONYR = re.compile(r'\b(?!(?:18|19|20)\d{2}\b)\d{2,}\b')
        if _ANY_NUM_NONYR.search(combined_rule1):
            return ('Quantified',
                    f"Non-year numerical count in study fields: '{outcome_summary[:200]}'")

    # Directional: has some text finding but no numeric metric
    for f in findings:
        sc = str(f.get('standardized_conclusion') or '').strip()
        if sc and sc.lower() not in {'not reported', 'none', 'n/a', 'null'}:
            # Exclude boilerplate-only conclusions with no additional signal
            if not (sc.startswith('Using ') and 'leveraged the reported predictors' in sc
                    and len(sc) < 300):
                return ('Directional',
                        f"Direction stated without quantification: '{sc[:200]}'")

    return ('None', "No empirical findings reported.")


# ─────────────────────────────────────────────────────────────────────────────
# D3 — Population Boundedness
# ─────────────────────────────────────────────────────────────────────────────
def extract_d3(data: dict):
    sample      = data.get('sample_details') or {}
    countries   = sample.get('countries') or []
    codes       = [c.get('country_code','') for c in countries
                   if isinstance(c, dict) and c.get('country_code')]

    findings     = data.get('main_findings') or []
    dataset_text = ' '.join(str(f.get('dataset_used') or '') for f in findings)
    summary      = str(data.get('outcome_summary') or '')
    sfc          = str(sample.get('sample_filtering_criteria') or '')
    combined     = dataset_text + ' ' + summary + ' ' + sfc

    if codes:
        # FIX-G: 10+ countries = international scope → Global (not bounded to specific context)
        # Studies with 2–9 named countries are still context-specific/bounded.
        # 10+ threshold captures "all OECD", "all PISA participants" etc.
        if len(codes) >= 10:
            return ('Global/Unspecified',
                    f"Multi-country study ({len(codes)} countries: {codes[:4]}…) = international scope.")
        return ('Bounded', f"Specific countries identified: {codes[:9]}.")

    ILSA = re.compile(
        r'\b(PISA|TIMSS|PIRLS|TALIS|ICILS|PIAAC|PASEC|SERCE|LLECE|SACMEQ|NAEP|ICCS)\s*\d{4}\b',
        re.IGNORECASE,
    )
    m = ILSA.search(combined)
    if m:
        # FIX-H: ILSA cycle with no specific country codes → international scope → Global
        return ('Global/Unspecified',
                f"ILSA cycle ('{m.group()}') without country restriction = international scope.")

    GRADE = re.compile(
        r'\b(grade\s*\d+|\d+th\s*grade|15.year.old|fourth.grade|eighth.grade|year\s*\d+)\b',
        re.IGNORECASE,
    )
    gm = GRADE.search(combined)
    if gm:
        return ('Bounded', f"Specific grade/age group: '{gm.group()}'.")

    COUNTRY_NAMES = [
        'turkey','china','singapore','germany','finland','japan','korea',
        'australia','brazil','indonesia','england','canada','france',
        'philippines','macau','spain','italy','portugal','vietnam',
        'hong kong','taiwan','iran','saudi','egypt','ghana','south africa',
        'norway','sweden','denmark','netherlands','belgium','austria',
        'switzerland','poland','czech','hungary','romania','greece',
        'israel','thailand','malaysia','india','pakistan','russia',
        'ukraine','mexico','argentina','chile','colombia','peru',
        'morocco','algeria','tunisia','senegal','kenya','tanzania',
        'new zealand','ireland','scotland','wales','united states','u.s.',
        'united kingdom','u.k.','luxembourg','estonia','latvia','lithuania',
        'croatia','serbia','bulgaria','slovakia','slovenia','iceland',
        'hungary','georgia','albania','kosovo','north macedonia',
        'qatar','bahrain','kuwait','oman','jordan','lebanon',
        'cambodia','myanmar','vietnam','thailand','indonesia','malaysia',
    ]
    cl = combined.lower()
    for cn in COUNTRY_NAMES:
        if cn in cl:
            return ('Bounded', f"Country name '{cn}' in study text.")

    return ('Global/Unspecified',
            "No specific country, ILSA cycle, grade, or student group identified.")


# ─────────────────────────────────────────────────────────────────────────────
# Decision rule
# ─────────────────────────────────────────────────────────────────────────────
def apply_rule(d1, d2, d3):
    if d1 == 'Causal' and d2 == 'Quantified' and d3 == 'Bounded':
        return (5, "Rule 1: D1=Causal, D2=Quantified, D3=Bounded → Score 5.")
    # FIX-I: Synthesis papers do not qualify for Rule 2 (Score 4) — they cap at Rule 4 (Score 2)
    if d1 != 'Synthesis' and d2 == 'Quantified' and d3 == 'Bounded':
        return (4, "Rule 2: D2=Quantified, D3=Bounded → Score 4 (D1 irrelevant).")
    if d1 == 'Correlational' and (d2 == 'Directional' or d3 == 'Global/Unspecified'):
        return (3, "Rule 3: D1=Correlational with D2=Directional or D3=Global/Unspecified → Score 3.")
    # FIX-K: Review/synthesis papers with no quantified finding AND no bounded population
    # are purely descriptive/scoping → Score 1.  Requiring D3=Global ensures papers that
    # synthesise findings within a specific context (D3=Bounded) still reach Score 2.
    if d1 in ('Synthesis', 'None') and d2 == 'None' and d3 == 'Global/Unspecified':
        return (1, "Rule 3b: D1=Synthesis/None, D2=None, D3=Global → purely descriptive, Score 1.")
    if d1 in ('Synthesis', 'None') or d2 == 'Directional' or d3 == 'Global/Unspecified':
        return (2, "Rule 4: D1=Synthesis/None or D2=Directional or D3=Global/Unspecified → Score 2.")
    return (1, "Rule 5: D1=None → Score 1.")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def score_file(filepath, filename):
    with open(filepath, encoding='utf-8') as fh:
        doc = json.load(fh)
    meta = doc.get('metadata', {})
    data = doc.get('data', {})
    title = meta.get('title', filename)

    d1, d1_ev = extract_d1(data, meta)
    d2, d2_ev = extract_d2(data, d1_hint=d1)
    d3, d3_ev = extract_d3(data)
    score, justification = apply_rule(d1, d2, d3)

    return {
        'filename':      filename,
        'title':         title,
        'D1':            d1,
        'D2':            d2,
        'D3':            d3,
        'Score':         score,
        'D1_evidence':   d1_ev,
        'D2_evidence':   d2_ev,
        'D3_evidence':   d3_ev,
        'Justification': justification,
    }


def main():
    files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith('.json'))
    print(f"Processing {len(files)} files …")

    rows, errors = [], []
    for fname in files:
        try:
            rows.append(score_file(os.path.join(JSON_DIR, fname), fname))
        except Exception as exc:
            errors.append(fname)
            rows.append({
                'filename': fname, 'title': fname,
                'D1': 'ERROR', 'D2': 'ERROR', 'D3': 'ERROR', 'Score': 0,
                'D1_evidence': str(exc), 'D2_evidence': '',
                'D3_evidence': '', 'Justification': 'Processing error.',
            })

    fieldnames = ['filename','title','D1','D2','D3','Score',
                  'D1_evidence','D2_evidence','D3_evidence','Justification']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    scores = Counter(r['Score'] for r in rows)
    d1s    = Counter(r['D1']    for r in rows)
    print(f"\nWritten {len(rows)} rows → {OUTPUT_CSV}")
    print("\nScore distribution:")
    for s in sorted(scores):
        print(f"  Score {s}: {scores[s]}")
    print("\nD1 distribution:")
    for k, v in sorted(d1s.items()):
        print(f"  D1={k}: {v}")
    if errors:
        print(f"\nErrors ({len(errors)}): {errors}")


if __name__ == '__main__':
    main()
