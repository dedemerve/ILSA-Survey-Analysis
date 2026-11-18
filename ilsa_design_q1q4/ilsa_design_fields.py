#!/usr/bin/env python3
"""
ilsa_design_fields.py
Extract Q1-Q4 ILSA design indicators from PDFs via LLM, then review/apply/merge/summarise.

Subcommands
-----------
extract   --pdf-dir DIR --json-dir DIR --out DIR [--only ids.txt] [--model MODEL]
review    --out DIR --csv FILE --json-dir DIR
apply     --out DIR --csv FILE
merge     --out DIR --json-dir DIR --dest DIR
summarise --json-dir DIR [--include ids206.txt]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Keywords for paragraph filtering
# ---------------------------------------------------------------------------
KEYWORDS_Q1 = [
    "plausible value", "plausible_value", " pv1", " pv2", " pv3", " pv4", " pv5",
    "five plausible", "ten plausible", "pv combination", "rubin", "rubin's rules",
    "within-imputation", "between-imputation", "multiple imputation",
    "pvw", "totwgt", "final weight", "irt theta", "wle", "eap",
]
KEYWORDS_Q2 = [
    "weight", "sampling weight", "survey weight", "weighted", "unweighted",
    "pweight", "analytic weight", "student weight", "senate weight",
    "totwgt", "schwgt", "w_fstuwt", "design weight",
    "sampling design", "complex sample", "survey design", "replicate weight",
    "balanced repeated replication", "jackknife", "brr", "fay",
]
KEYWORDS_Q3 = [
    "multilevel", "hierarchical linear", " hlm", "cluster", "nested",
    "intraclass", " icc ", "random effect", "school effect", "school-level",
    "mixed model", "lme4", " mlm", "between-school", "within-school",
    "school.*split", "train.*school", "cluster.*split",
]
KEYWORDS_Q4 = [
    "external validation", "cross-country", "another country", "different country",
    "different cycle", "another cycle", "replication", "out-of-sample",
    "external data", "external sample", "holdout", "hold-out", "held-out",
    "transfer", "generali", "new dataset", "independent dataset",
    "cross-validation", "cross validation", "k-fold", "kfold", "k fold",
    "5-fold", "10-fold", "nested cv", "train test", "train/test",
    "test set", "test data", "validation set", "validation data",
    "train set", "training set", "split", "partition",
]

ALL_KEYWORDS = KEYWORDS_Q1 + KEYWORDS_Q2 + KEYWORDS_Q3 + KEYWORDS_Q4

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a methodologist reviewing machine-learning studies that use ILSA
(international large-scale assessment) microdata. For each study, rate four design quality
indicators based ONLY on what is explicitly stated in the provided text excerpts.

IMPORTANT CONTEXT: You will receive the study's source_category. Use it:
  - methodology_paper / framework_paper / review → these do NOT analyse student-level
    ILSA microdata, so Q1–Q3 are typically not_applicable (Q4 may still apply).
  - automated_scoring / nlp_scoring → these score open-ended items; they do NOT use
    plausible values or sampling weights, so Q1 and Q2 are not_applicable.
  - process_data → these analyse log/clickstream data; Q1 is not_applicable unless
    the study also uses achievement scores.
  - peer_reviewed_research → standard empirical study; judge Q1–Q4 on the evidence.

Definitions
-----------
Q1 plausible_values:
  met           = study used multiple plausible values AND applied Rubin's rules / proper combination
  not_met       = study used a single PV, averaged PVs, used raw scores, WLE/theta, or EAP
  not_reported  = study uses ILSA achievement data as outcome/predictor but does not mention
                  PV handling at all
  not_applicable = achievement (latent proficiency) is NOT an outcome or predictor in this study
                   (e.g., process-data-only, automated scoring, methodology paper, review,
                    or the study builds a dataset without running predictive models)

Q2 sampling_weights:
  met           = study explicitly applied sampling/replicate weights in the analysis
  not_met       = study explicitly states it did NOT apply weights OR clearly treats data
                  as a convenience sample without weighting
  not_reported  = study uses ILSA student-level data but makes no mention of weights
  not_applicable = study does not use student-level survey data (e.g., pure process-data,
                   automated scoring, methodology paper, framework, or review)

Q3 school_clustering:
  met           = study used multilevel modelling, cluster-robust / BRR / JK replicate SE,
                  or performed train/test split at the school level
  not_met       = study used student-level models ignoring school clustering
  not_reported  = no information about clustering
  not_applicable = study does not analyse student-level survey data where schools are the
                   sampling unit (e.g., process-data, automated scoring, methodology paper)

Q4 validation_beyond_estimation_sample:
  met           = model/findings validated on data from another country, another cycle,
                  or an entirely external dataset not used for training
  not_met       = study performed ONLY internal validation (k-fold CV, nested CV, hold-out
                  split, train/test split, bootstrap, or any resampling from the same dataset).
                  If the study mentions ANY of these internal methods, the answer is not_met,
                  NOT not_reported.
  not_reported  = no validation information at all — no mention of CV, train/test, hold-out,
                  or any other validation strategy

For each indicator output EXACTLY one of its allowed labels and a one-sentence rationale.
Be conservative: if evidence is unclear, choose not_reported rather than met."""

EXTRACTION_PROMPT_TEMPLATE = """Below are excerpts from the study. Rate the four indicators.

Study: {title} ({year})
Source category: {source_category}

--- PRIOR EXTRACTION (from structured JSON, may be incomplete) ---
{prior_fields}
--- END PRIOR ---

--- EXCERPTS FROM PDF ---
{text}
--- END EXCERPTS ---

IMPORTANT REMINDERS:
- Q4: If the excerpts mention ANY of these: cross-validation, k-fold, train/test split,
  hold-out, nested CV, bootstrap, random subset — the answer is **not_met** (internal
  validation), NOT not_reported. Only use not_reported if there is truly NO mention of
  any validation strategy at all.
- Q1: If prior extraction says average_pv, single_pv, wle, irt_theta, or eap → not_met.
  If it says rubin_rules or all_pv → met. If it says not_applicable → not_applicable.
- Q3: If the study models individual students from ILSA data (which is cluster-sampled
  by schools) using flat models (no multilevel, no cluster-robust SE, no school-level
  split), the answer is **not_met**, not not_reported.
- Q2: If the study analyses student-level ILSA microdata (PISA, TIMSS, PIRLS) and
  never mentions sampling weights, the answer is **not_met** (they clearly ran
  unweighted analysis), NOT not_reported. Use not_reported only when it is genuinely
  ambiguous whether weights were applied."""


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        import fitz
    except ImportError:
        sys.exit("PyMuPDF not installed. Run: pip install pymupdf")
    # Open via byte stream to avoid filename encoding issues on macOS
    with open(pdf_path, "rb") as fh:
        doc = fitz.open(stream=fh.read(), filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    full = "\n".join(pages)
    full = full.encode("utf-8", errors="ignore").decode("utf-8")
    for marker in ["\nReferences\n", "\nREFERENCES\n", "\nBibliography\n",
                   "\nREFERENCES ", "\nReferences "]:
        idx = full.find(marker)
        if idx != -1:
            full = full[:idx]
            break
    return full


def _keyword_paragraphs(text: str, max_chars: int = 12000) -> str:
    paragraphs = re.split(r"\n{2,}", text)
    matched = []
    total = 0
    for para in paragraphs:
        low = para.lower()
        if any(kw in low for kw in ALL_KEYWORDS):
            matched.append(para.strip())
            total += len(para)
            if total >= max_chars:
                break
    return "\n\n".join(matched)


# ---------------------------------------------------------------------------
# Filename matching
# ---------------------------------------------------------------------------
def _number_from_name(name: str) -> Optional[int]:
    m = re.match(r'^(\d+)[\.\s]', name)
    return int(m.group(1)) if m else None


def _title_words(name: str) -> set:
    name = re.sub(r'^\d+[\.\s]+', '', name)
    name = re.sub(r'\.(json|pdf)$', '', name, flags=re.I)
    name = re.sub(r'^[^.]+\(\d{4}\)\.\s*', '', name)
    return set(re.findall(r'[a-z]{3,}', name.lower()))


def find_pdf_for_json(json_name: str, pdf_dirs: list) -> Optional[Path]:
    """Search multiple PDF directories. Match by leading number AND title-word overlap.

    Strategy: collect all PDFs whose number matches, pick the one with highest
    title-word overlap (must be >= 0.3).  If none match by number, fall back to
    pure title overlap >= 0.6 across ALL PDFs.  This avoids the bug where PDF
    "194. dos Santos" is served for JSON "194. Jung" when the correct PDF is
    numbered differently (e.g. "214. Jung").
    """
    num = _number_from_name(json_name)
    j_words = _title_words(json_name)
    number_candidates = []
    title_candidates = []

    for pdf_dir in pdf_dirs:
        for pdf in pdf_dir.glob("*.pdf"):
            p_num = _number_from_name(pdf.name)
            p_words = _title_words(pdf.name)
            if not j_words or not p_words:
                continue
            overlap = len(j_words & p_words) / max(len(j_words), len(p_words))

            if p_num is not None and p_num == num:
                number_candidates.append((overlap, pdf))

            # Always collect title-based candidates as fallback
            if overlap >= 0.4:
                title_candidates.append((overlap, pdf))

    # Prefer number-matched PDFs with reasonable title overlap
    if number_candidates:
        number_candidates.sort(key=lambda x: -x[0])
        best_overlap, best_pdf = number_candidates[0]
        if best_overlap >= 0.3:
            return best_pdf
        # Number matched but title didn't — flag this, still return
        # but only if no better title candidate exists
        if not title_candidates:
            return best_pdf

    # Fall back to best title match (any number)
    if title_candidates:
        title_candidates.sort(key=lambda x: -x[0])
        best_overlap, best_pdf = title_candidates[0]
        if best_overlap >= 0.5:
            return best_pdf

    return None


# ---------------------------------------------------------------------------
# API call (structured output)
# ---------------------------------------------------------------------------
def _call_api(title: str, year, text: str, model: str, client,
              source_category: str = "peer_reviewed_research",
              prior_fields: str = "None available") -> dict:
    from pydantic import BaseModel

    class IndicatorResult(BaseModel):
        label: str
        rationale: str

    class DesignFields(BaseModel):
        q1_plausible_values: IndicatorResult
        q2_sampling_weights: IndicatorResult
        q3_school_clustering: IndicatorResult
        q4_validation_beyond_estimation: IndicatorResult

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        title=title, year=year, text=text, source_category=source_category,
        prior_fields=prior_fields)
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=DesignFields,
        temperature=0,
    )
    r = response.choices[0].message.parsed
    return {
        "q1_plausible_values": r.q1_plausible_values.label,
        "q1_rationale": r.q1_plausible_values.rationale,
        "q2_sampling_weights": r.q2_sampling_weights.label,
        "q2_rationale": r.q2_sampling_weights.rationale,
        "q3_school_clustering": r.q3_school_clustering.label,
        "q3_rationale": r.q3_school_clustering.rationale,
        "q4_validation_beyond_estimation": r.q4_validation_beyond_estimation.label,
        "q4_rationale": r.q4_validation_beyond_estimation.rationale,
    }


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def cmd_extract(args):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai not installed.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENAI_API_KEY not set.")
    if not api_key.startswith("sk-"):
        sys.exit("OPENAI_API_KEY does not start with 'sk-'. Check for stray characters.")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        sys.exit("OPENAI_API_KEY contains non-ASCII characters (e.g. ç, ü). "
                 "Re-export the key without extra characters.")
    client = OpenAI(api_key=api_key)

    pdf_dirs = [Path(d) for d in args.pdf_dir]
    for d in pdf_dirs:
        if not d.is_dir():
            sys.exit(f"PDF directory not found: {d}")
    n_pdfs = sum(len(list(d.glob("*.pdf"))) for d in pdf_dirs)
    print(f"[INFO] {len(pdf_dirs)} PDF dir(s), {n_pdfs} total PDFs")
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    only_ids = None
    if args.only:
        with open(args.only) as f:
            only_ids = {int(line.strip()) for line in f if line.strip()}

    json_files = sorted(json_dir.glob("*.json"),
                        key=lambda p: (_number_from_name(p.name) or 9999, p.name))
    problems = []
    n_processed = n_no_pdf = n_no_text = 0

    for jf in json_files:
        num = _number_from_name(jf.name)
        if only_ids and num not in only_ids:
            continue
        out_file = out_dir / jf.name
        if out_file.exists():
            print(f"[SKIP existing] {jf.name[:70]}")
            continue

        data = json.loads(jf.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        title = meta.get("title", jf.stem)
        year = meta.get("year", "?")

        pdf_path = find_pdf_for_json(jf.name, pdf_dirs)
        if pdf_path is None:
            print(f"[MATCHING PROBLEM] No PDF: {jf.name[:70]}")
            problems.append({"id": num, "json": jf.name, "issue": "no_pdf_match"})
            fields = _stub("No PDF found.")
            fields["_status"] = "no_pdf"
            _write(out_dir, jf, meta, fields)
            n_no_pdf += 1
            continue

        full_text = _extract_pdf_text(pdf_path)
        excerpts = _keyword_paragraphs(full_text)
        if not excerpts.strip():
            print(f"[NO KEYWORDS] {jf.name[:60]}  pdf={pdf_path.name[:40]}")
            problems.append({"id": num, "json": jf.name, "pdf": pdf_path.name,
                              "issue": "no_keyword_match"})
            fields = _stub("No relevant paragraphs.")
            fields["_status"] = "no_keywords"
            fields["_pdf_matched"] = pdf_path.name
            _write(out_dir, jf, meta, fields)
            n_no_text += 1
            continue

        source_cat = meta.get("source_category", "peer_reviewed_research")
        orig_data = data.get("data", {})
        prior_lines = []
        pv = orig_data.get("plausible_values_handling")
        if pv:
            prior_lines.append(f"plausible_values_handling: {pv}")
        sd = orig_data.get("survey_design", {})
        if isinstance(sd, dict):
            sw = sd.get("student_weights_used")
            if sw is not None:
                prior_lines.append(f"student_weights_used: {sw}")
            interp = sd.get("weight_fields_interpretation", "")
            if interp:
                prior_lines.append(f"weight_interpretation: {interp[:300]}")
        hnr = orig_data.get("handling_not_reported_explanation", "")
        if hnr:
            prior_lines.append(f"handling_note: {hnr[:300]}")
        prior_str = "\n".join(prior_lines) if prior_lines else "None available"

        print(f"[EXTRACT] {num}. {title[:60]} ({year})  [{source_cat}]")
        try:
            fields = _call_api(title, year, excerpts, args.model, client,
                               source_category=source_cat,
                               prior_fields=prior_str)
            fields["_status"] = "extracted"
            fields["_pdf_matched"] = pdf_path.name
        except Exception as e:
            import traceback
            print(f"  API ERROR: {e}")
            traceback.print_exc()
            fields = _stub(f"API error: {e}")
            fields["_status"] = f"api_error"
        _write(out_dir, jf, meta, fields)
        n_processed += 1

    print(f"\n--- SUMMARY ---")
    print(f"API calls:      {n_processed}")
    print(f"No PDF match:   {n_no_pdf}")
    print(f"No keywords:    {n_no_text}")
    if problems:
        print(f"\nMATCHING PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  {p}")


def _stub(reason: str) -> dict:
    d = {}
    for q in ["q1_plausible_values", "q2_sampling_weights",
              "q3_school_clustering", "q4_validation_beyond_estimation"]:
        d[q] = "not_reported"
    for q in ["q1", "q2", "q3", "q4"]:
        d[f"{q}_rationale"] = reason
    return d


def _write(out_dir: Path, jf: Path, meta: dict, fields: dict):
    (out_dir / jf.name).write_text(
        json.dumps({"metadata": meta, "design_fields": fields},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------
def cmd_review(args):
    out_dir = Path(args.out)
    json_dir = Path(args.json_dir)
    csv_path = Path(args.csv)

    rows = []
    for jf in sorted(out_dir.glob("*.json"),
                     key=lambda p: (_number_from_name(p.name) or 9999, p.name)):
        data = json.loads(jf.read_text(encoding="utf-8"))
        df = data.get("design_fields", {})
        meta = data.get("metadata", {})
        num = _number_from_name(jf.name)

        old_q1 = old_q2 = "?"
        orig_jf = json_dir / jf.name
        if orig_jf.exists():
            orig = json.loads(orig_jf.read_text(encoding="utf-8"))
            od = orig.get("data", {})
            old_q1 = str(od.get("plausible_values_handling", "?"))
            sd = od.get("survey_design", {})
            old_q2 = str(sd.get("student_weights_used", "?") if isinstance(sd, dict) else "?")

        new_q1 = df.get("q1_plausible_values", "?")
        new_q2 = df.get("q2_sampling_weights", "?")

        conflict_q1 = "YES" if (
            (old_q1 in {"True", "met"} and new_q1 in {"not_met", "not_reported"}) or
            (old_q1 in {"False", "not_met"} and new_q1 == "met")
        ) else ""
        conflict_q2 = "YES" if (
            (old_q2 in {"True", "met"} and new_q2 in {"not_met", "not_reported"}) or
            (old_q2 in {"False", "not_met"} and new_q2 == "met")
        ) else ""

        rows.append({
            "id": num,
            "json_file": jf.name,
            "title": meta.get("title", "")[:80],
            "year": meta.get("year", ""),
            "status": df.get("_status", ""),
            "old_q1_pv": old_q1,
            "new_q1": new_q1,
            "q1_rationale": df.get("q1_rationale", ""),
            "CONFLICT_q1": conflict_q1,
            "q1_final": "",
            "old_q2_weights": old_q2,
            "new_q2": new_q2,
            "q2_rationale": df.get("q2_rationale", ""),
            "CONFLICT_q2": conflict_q2,
            "q2_final": "",
            "new_q3": df.get("q3_school_clustering", ""),
            "q3_rationale": df.get("q3_rationale", ""),
            "q3_final": "",
            "new_q4": df.get("q4_validation_beyond_estimation", ""),
            "q4_rationale": df.get("q4_rationale", ""),
            "q4_final": "",
            "notes": "",
        })

    if not rows:
        print("No files found in --out directory.")
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_conflict = sum(1 for r in rows if r["CONFLICT_q1"] or r["CONFLICT_q2"])
    print(f"review.csv: {len(rows)} rows, {n_conflict} with CONFLICT")
    print("Fill q*_final where you want to override new_* value; leave blank to accept it.")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def cmd_apply(args):
    out_dir = Path(args.out)
    log = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            jf = out_dir / row["json_file"]
            if not jf.exists():
                print(f"[WARN] Not found: {jf}")
                continue
            data = json.loads(jf.read_text(encoding="utf-8"))
            df = data.setdefault("design_fields", {})
            changed = False
            for q_key, final_key in [
                ("q1_plausible_values", "q1_final"),
                ("q2_sampling_weights", "q2_final"),
                ("q3_school_clustering", "q3_final"),
                ("q4_validation_beyond_estimation", "q4_final"),
            ]:
                final = row.get(final_key, "").strip()
                if final and final != df.get(q_key, ""):
                    log.append({"json": row["json_file"], "field": q_key,
                                 "old": df.get(q_key, ""), "new": final})
                    df[q_key] = final
                    changed = True
            if changed:
                jf.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    log_path = out_dir / "corrections_log.csv"
    if log:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["json", "field", "old", "new"])
            writer.writeheader()
            writer.writerows(log)
    print(f"Applied {len(log)} corrections → {log_path}")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def cmd_merge(args):
    out_dir = Path(args.out)
    json_dir = Path(args.json_dir)
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    merged = 0
    for jf in sorted(json_dir.glob("*.json")):
        orig = json.loads(jf.read_text(encoding="utf-8"))
        design_file = out_dir / jf.name
        if design_file.exists():
            design = json.loads(design_file.read_text(encoding="utf-8"))
            orig["design_fields"] = design.get("design_fields", {})
        else:
            orig["design_fields"] = {"_status": "not_extracted"}
        (dest_dir / jf.name).write_text(
            json.dumps(orig, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        merged += 1
    print(f"Merged {merged} files → {dest_dir}")


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------
def cmd_summarise(args):
    json_dir = Path(args.json_dir)
    include_ids = None
    if args.include:
        with open(args.include) as f:
            include_ids = {int(line.strip()) for line in f if line.strip()}

    counts = {"q1": Counter(), "q2": Counter(), "q3": Counter(), "q4": Counter()}
    n_microdata = n_total = 0

    for jf in sorted(json_dir.glob("*.json")):
        num = _number_from_name(jf.name)
        if include_ids and num not in include_ids:
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        df = data.get("design_fields", {})
        orig_data = data.get("data", {})

        counts["q1"][df.get("q1_plausible_values", "missing")] += 1
        counts["q2"][df.get("q2_sampling_weights", "missing")] += 1
        counts["q3"][df.get("q3_school_clustering", "missing")] += 1
        counts["q4"][df.get("q4_validation_beyond_estimation", "missing")] += 1

        sd = orig_data.get("survey_design", {})
        if isinstance(sd, dict) and sd.get("uses_ilsa_microdata"):
            n_microdata += 1
        n_total += 1

    print(f"\n=== SUMMARY (n={n_total}) ===\n")
    labels = {
        "q1": "Q1 Plausible Values",
        "q2": "Q2 Sampling Weights",
        "q3": "Q3 School Clustering",
        "q4": "Q4 Validation Beyond Sample",
    }
    for q, label in labels.items():
        print(f"\n{label}:")
        for k in ["met", "not_met", "not_reported", "not_applicable", "missing"]:
            v = counts[q].get(k, 0)
            if v:
                pct = 100 * v / n_total if n_total else 0
                print(f"  {k:<22} {v:4d}  ({pct:.1f}%)")
        print()
    print(f"uses_ilsa_microdata=True : {n_microdata}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(prog="ilsa_design_fields.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--pdf-dir", required=True, nargs="+",
                   help="One or more directories containing PDFs")
    p.add_argument("--json-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default=None, help="File with one ID per line")
    p.add_argument("--model", default="gpt-4o-mini")

    p = sub.add_parser("review")
    p.add_argument("--out", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--json-dir", required=True)

    p = sub.add_parser("apply")
    p.add_argument("--out", required=True)
    p.add_argument("--csv", required=True)

    p = sub.add_parser("merge")
    p.add_argument("--out", required=True)
    p.add_argument("--json-dir", required=True)
    p.add_argument("--dest", required=True)

    p = sub.add_parser("summarise")
    p.add_argument("--json-dir", required=True)
    p.add_argument("--include", default=None, help="File with IDs to include")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"extract": cmd_extract, "review": cmd_review, "apply": cmd_apply,
     "merge": cmd_merge, "summarise": cmd_summarise}[args.cmd](args)
