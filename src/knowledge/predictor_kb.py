"""
Predictor-level knowledge base for RQ2.

The earlier semantic knowledge base (``final_knowledge_synthesis_v4.csv``)
aggregates *outcomes* (Method × Outcome × "trend"). A forecasting model needs
knowledge about *predictors*: which constructs the ILSA literature finds
important for achievement, how consistently, and in which direction. This module
builds that table from the corpus:

  predictor_evidence.csv   one row per (scope, predictor_group) with
      n_studies_top        studies listing the group among top predictors of achievement
      n_studies_considered studies that included the group as a predictor at all
      salience             n_studies_top / n_studies_considered
      mean_rank            mean position in the top-predictor list (1 = most important)
      n_positive / n_negative / n_unspecified   sentence-level direction votes
      direction            positive | negative | mixed | unspecified
      direction_consensus  (pos − neg) / (pos + neg)

``scope`` is ``all`` or a program / domain slice (``program=PISA``,
``domain=Mathematics``) so priors can be specialised per forecasting series.

``country_covariate_priors`` then turns the evidence into priors for the
country-level covariates listed in ``config/kb_covariate_map.json``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.corpus import article_id
from src.analysis.study_attributes import derive_finding_attributes
from src.enrichment.canonical_taxonomy import UNCATEGORIZED, map_variable_to_canonical

# Schema confounder categories → readable predictor domain labels.
SCHEMA_DOMAIN = {
    "socioeconomic": "Socioeconomic",
    "demographic": "Demographic",
    "student_attitude": "Student attitude",
    "student_behavior": "Student behaviour",
    "teacher": "Teacher",
    "school": "School",
    "ict": "ICT",
    "curriculum": "Curriculum / instruction",
    "parent_home": "Parent / home",
    "process_data": "Process data",
    "prior_achievement": "Prior achievement",
    "peer_effects": "Peer effects",
    "system_level": "System level",
}

_OUTCOME = r"(?:\w+\s+){0,2}(?:achievement|performance|scores?|literacy|proficiency|outcomes?|results?|resilience)"
# Association wording only — model-quality words ("outperformed", "strong AUC") are not effect directions.
_POS = re.compile(
    rf"\bpositive(?:ly)?\b|\b(?:higher|better|greater|increased?|improved?)\s+{_OUTCOME}|"
    rf"\bboost(?:s|ed)?\s+{_OUTCOME}|\bbeneficial\b",
    re.I,
)
_NEG = re.compile(
    rf"\bnegative(?:ly)?\b|\b(?:lower|worse|poorer|reduced?|decreased?)\s+{_OUTCOME}|"
    rf"\bdetrimental\b|\bhinder\w*|\bimpair\w*",
    re.I,
)
_TEMPLATE_PREAMBLE = re.compile(r"^.*?\bfinding that\b", re.I | re.S)
_IMPLICATION = re.compile(r"^\s*this (?:indicates|suggests|implies)", re.I)
_SENTENCE = re.compile(r"(?<=[.;])\s+")
_STOP = {"of", "the", "and", "in", "at", "to", "a", "an", "for", "on", "with", "index", "level", "student", "students", "s"}


def _predictor_group(name: str, schema_category: str | None) -> tuple[str, str]:
    """(predictor_group, predictor_domain) for one predictor label."""
    group = map_variable_to_canonical(name, use_smart_resolver=True)
    domain = SCHEMA_DOMAIN.get(schema_category or "", None)
    is_outcome_label = group.endswith("_Achievement") and schema_category != "prior_achievement"
    if group in (UNCATEGORIZED, "[IGNORE]") or is_outcome_label:
        group = domain or "Other"
    if domain is None:
        domain = "Other"
    return group, domain


_GENERIC = {"status", "use", "time", "level", "score", "scores", "class", "type", "country", "school", "home"}
_CLAUSE = re.compile(r",|;|\bwhile\b|\bwhereas\b|\bbut\b|\balthough\b", re.I)
_NEGATION = re.compile(r"\b(?:not|no|nor|without|n't|neither)\b", re.I)


def _matcher(name: str, code: str | None):
    """Return a predicate telling whether a clause mentions the predictor.

    An acronym/variable code (e.g. ESCS, ANXMAT) matches on its own; otherwise the
    first two content words must both occur, so generic words such as "status" or
    "time" do not attach a direction to the wrong construct.
    """
    codes = {c.lower() for c in re.findall(r"\(([A-Za-z0-9_]{2,})\)", name)}
    if code and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{2,}", code) and not code.islower():
        codes.add(code.lower())
    codes |= {w.lower() for w in re.findall(r"\b[A-Z]{3,}[A-Z0-9]*\b", name)}
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", re.sub(r"\(.*?\)", "", name))
             if w.lower() not in _STOP]
    content = [w for w in words if w not in _GENERIC] or words
    head = content[:2]

    def mentions(clause: str) -> bool:
        low = clause.lower()
        if any(re.search(rf"\b{re.escape(c)}\b", low) for c in codes):
            return True
        return bool(head) and all(w in low for w in head)

    return mentions


def predictor_direction(name: str, code: str | None, texts: list[str]) -> str:
    """positive / negative / unspecified from the clauses that mention the predictor."""
    mentions = _matcher(name, code)
    pos = neg = 0
    for text in texts:
        for sentence in _SENTENCE.split(text or ""):
            if _IMPLICATION.match(sentence):
                continue
            # Drop the "Using … the study leveraged <predictor list> to predict …" preamble.
            sentence = _TEMPLATE_PREAMBLE.sub("", sentence)
            for clause in _CLAUSE.split(sentence):
                if not mentions(clause) or _NEGATION.search(clause):
                    continue
                pos += len(_POS.findall(clause))
                neg += len(_NEG.findall(clause))
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "unspecified"


def predictor_mentions(articles: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (finding, top predictor) for achievement findings, plus 'considered' rows."""
    rows: list[dict[str, Any]] = []
    for art in articles:
        meta, data = art["metadata"], art["data"]
        aid = article_id(art)
        confounders = {
            str(c.get("variable_name") or "").strip().lower(): c
            for c in data.get("confounders_identified") or []
            if isinstance(c, dict)
        }
        summary = str(data.get("outcome_summary") or "")
        ach_findings = []
        for idx, f in enumerate(data.get("main_findings") or []):
            if not isinstance(f, dict):
                continue
            attrs = derive_finding_attributes(f, meta=meta, data=data)
            if attrs["is_achievement"]:
                ach_findings.append((idx, f, attrs))
        if not ach_findings:
            continue
        programs = sorted({p for _, _, a in ach_findings for p in a["ilsa_programs"]})
        domains = sorted({a["domain"] for _, _, a in ach_findings if a["domain"] != "not_determinable"})

        # 'considered': every confounder of an achievement study
        for c in confounders.values():
            group, pdomain = _predictor_group(str(c.get("variable_name") or ""), c.get("category"))
            rows.append({
                "article_id": aid, "finding_id": None, "kind": "considered",
                "predictor": c.get("variable_name"), "predictor_group": group, "predictor_domain": pdomain,
                "rank": None, "direction": None,
                "programs": "; ".join(programs), "domains": "; ".join(domains),
            })
        for idx, f, attrs in ach_findings:
            texts = [str(f.get("standardized_conclusion") or ""), summary]
            extracted = {
                str(e.get("predictor") or "").strip().lower(): e.get("direction")
                for e in f.get("predictor_effects") or []
                if isinstance(e, dict)
            }
            for rank, pred in enumerate(f.get("top_predictors") or [], start=1):
                pred = str(pred).strip()
                if not pred:
                    continue
                c = confounders.get(pred.lower(), {})
                group, pdomain = _predictor_group(pred, c.get("category"))
                direction = extracted.get(pred.lower())
                if direction not in ("positive", "negative", "null", "nonlinear"):
                    direction = predictor_direction(pred, c.get("variable_code"), texts)
                rows.append({
                    "article_id": aid, "finding_id": f"{aid}#{idx + 1}", "kind": "top",
                    "predictor": pred, "predictor_group": group, "predictor_domain": pdomain,
                    "rank": rank,
                    "direction": direction,
                    "programs": "; ".join(attrs["ilsa_programs"]),
                    "domains": attrs["domain"],
                })
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, scope: str, level: str) -> list[dict[str, Any]]:
    out = []
    top = df[df["kind"] == "top"]
    considered = df[df["kind"] == "considered"]
    for group, g in top.groupby(level):
        # A study counts once per group; its direction is the majority of its votes.
        per_study = g.groupby("article_id")["direction"].agg(lambda s: s.value_counts().idxmax())
        pos = int((per_study == "positive").sum())
        neg = int((per_study == "negative").sum())
        uns = int((per_study == "unspecified").sum())
        n_top = int(g["article_id"].nunique())
        n_cons = int(max(considered.loc[considered[level] == group, "article_id"].nunique(), n_top))
        consensus = (pos - neg) / (pos + neg) if pos + neg else 0.0
        if pos + neg == 0:
            direction = "unspecified"
        elif consensus >= 0.5:
            direction = "positive"
        elif consensus <= -0.5:
            direction = "negative"
        else:
            direction = "mixed"
        out.append({
            "scope": scope,
            "level": level,
            "predictor_group": group,
            "n_studies_top": n_top,
            "n_findings_top": int(g["finding_id"].nunique()),
            "n_studies_considered": n_cons,
            "salience": round(n_top / n_cons, 3) if n_cons else None,
            "mean_rank": round(float(g["rank"].mean()), 2),
            "n_positive": pos,
            "n_negative": neg,
            "n_unspecified": uns,
            "direction_consensus": round(consensus, 3),
            "direction": direction,
            "example_predictors": " | ".join(g["predictor"].value_counts().head(4).index),
        })
    return out


def build_predictor_evidence(mentions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=mentions.index))]
    for prog in ("PISA", "TIMSS", "PIRLS"):
        scopes.append((f"program={prog}", mentions["programs"].fillna("").str.contains(rf"\b{prog}\b", regex=True)))
    for dom in ("Mathematics", "Reading", "Science"):
        scopes.append((f"domain={dom}", mentions["domains"].fillna("").str.contains(dom, regex=False)))
    for scope, mask in scopes:
        sub = mentions[mask]
        if sub.empty:
            continue
        for level in ("predictor_group", "predictor_domain"):
            rows.extend(_aggregate(sub, scope, level))
    out = pd.DataFrame(rows)
    return out.sort_values(["scope", "level", "n_studies_top"], ascending=[True, True, False]).reset_index(drop=True)


def load_covariate_map(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def country_covariate_priors(
    evidence: pd.DataFrame,
    covariate_map: dict[str, Any],
    *,
    scope: str = "all",
    min_studies: int = 2,
    min_direction_votes: int = 2,
) -> dict[str, dict[str, Any]]:
    """Translate predictor evidence into priors for country-level covariates.

    For each covariate the matching evidence rows (by predictor_group first, then
    predictor_domain) are pooled. ``sign`` is +1 / −1 only when the literature's
    direction is unanimous enough (``direction`` positive/negative) and the
    covariate's ``orientation`` (+1 when a higher value means more of the construct,
    −1 when it is reverse-coded) is applied; otherwise 0 (= no sign constraint).
    ``weight`` ∈ [0, 1] is the evidence share used to scale prior strength.
    """
    def rows_for(scope_name: str) -> pd.DataFrame:
        return evidence[evidence["scope"] == scope_name]

    scoped = rows_for(scope)
    fallback = rows_for("all")
    max_top = max(int(fallback["n_studies_top"].max() or 1), 1) if not fallback.empty else 1
    priors: dict[str, dict[str, Any]] = {}
    for cov, spec in covariate_map.get("covariates", {}).items():
        match = pd.DataFrame()
        source_scope = scope
        for table, sc in ((scoped, scope), (fallback, "all")):
            g = table[(table["level"] == "predictor_group") & table["predictor_group"].isin(spec.get("predictor_groups", []))]
            if g["n_studies_top"].sum() < min_studies:
                g = table[(table["level"] == "predictor_domain") & table["predictor_group"].isin(
                    [SCHEMA_DOMAIN.get(d, d) for d in spec.get("predictor_domains", [])]
                )]
            if g["n_studies_top"].sum() >= min_studies:
                match, source_scope = g, sc
                break
        if match.empty:
            priors[cov] = {"supported": False, "n_studies_top": 0, "sign": 0, "weight": 0.0,
                           "direction": "no_evidence", "scope": None, "groups": []}
            continue
        pos, neg = int(match["n_positive"].sum()), int(match["n_negative"].sum())
        consensus = (pos - neg) / (pos + neg) if pos + neg else 0.0
        if pos + neg < min_direction_votes:
            direction = "unspecified"
        elif consensus >= 0.5:
            direction = "positive"
        elif consensus <= -0.5:
            direction = "negative"
        else:
            direction = "mixed"
        orientation = int(spec.get("orientation", 1))
        sign = {"positive": 1, "negative": -1}.get(direction, 0) * orientation
        n_top = int(match["n_studies_top"].sum())
        priors[cov] = {
            "supported": True,
            "scope": source_scope,
            "groups": sorted(match["predictor_group"].unique().tolist()),
            "n_studies_top": n_top,
            "salience": round(float(match["salience"].mean()), 3),
            "direction": direction,
            "direction_votes": {"positive": pos, "negative": neg},
            "direction_consensus": round(consensus, 3),
            "sign": sign,
            "weight": round(min(1.0, n_top / max_top), 3),
        }
    return priors
