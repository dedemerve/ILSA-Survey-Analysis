#!/usr/bin/env python3
"""
Find and re-extract the 30 manually scored studies from the Web of Science
PDF directory.  Works even when PDFs have cryptic DOI-based filenames by
scanning the first page of every PDF for author / title keywords.

Usage (run from project root, branch claude/new-session-9t4ayc):

    python scripts/reextract_30_studies.py --pdf-dir "~/Desktop/ILSA Papers & Reports LLM /Web of Science"
    python scripts/reextract_30_studies.py --pdf-dir "..." --dry-run   # show matches only
    python scripts/reextract_30_studies.py --pdf-dir "..." --keys bezek-gure-etal2020 tin-etal2024
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed.  Run: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

from src.extractors.gpt_extractor import GPTExtractor
from src.extractors.pdf_processor import process_pdf

OUTPUT_DIR = PROJECT_ROOT / "ilsa_survey_articles" / "json"

# ──────────────────────────────────────────────────────────────────────────────
# Keyword sets for each of the 30 manually scored studies.
# A PDF matches a study if its first-page text contains ALL keywords in at
# least one tuple in the list (OR across tuples, AND within a tuple).
# Keywords are case-insensitive.
# ──────────────────────────────────────────────────────────────────────────────
MATCH_RULES: dict[str, list[tuple[str, ...]]] = {
    "bezek-gure-etal2020":            [("bezek",), ("güre",), ("güre",)],
    "demir-karaboga2021":             [("demir", "karaboğa"), ("demir", "karaboga"),
                                       ("demir", "deep learning", "math")],
    "acisli-celik-yesilkanat2023":    [("acıslı",), ("acisli",), ("yesilkanat",)],
    "oz-etal2024":                    [("öz", "stacking"), ("öz", "stacking"),
                                       ("oz", "stacking", "pisa")],
    "tin-etal2024":                   [("tin", "big data"), ("educational big data mining",),
                                       ("paper_64",)],
    "song-cutumisu2024":              [("song", "cutumisu"), ("cutumisu", "timss"),
                                       ("science curriculum type", "timss")],
    "nguyen-etal2023":                [("nguyen", "singapore"), ("tiered machine learning",),
                                       ("nguyen", "tiered")],
    "pejic-etal2021":                 [("pejic", "math proficiency"), ("pejic", "stanic"),
                                       ("pejic", "molcer", "math")],
    "zhang-etal2023":                 [("zhang", "accurate assessment"),
                                       ("zhang", "process data", "2023")],
    "sun-etal2023":                   [("sun", "mathematical cognitive model"),
                                       ("theoretical framework", "mathematical cognitive")],
    "fn-aydin-etal2025":              [("aydin", "process data", "predictive"),
                                       ("aydin", "pisa", "2025")],
    "tyack-etal2024":                 [("tyack",), ("convolutional neural", "automatically")],
    "robitzsch-ludtke2022":           [("robitzsch", "lüdtke"), ("robitzsch", "ludtke"),
                                       ("robitzsch", "scaling model")],
    "hernandez-torrano-courtney2021": [("hernández", "torrano"), ("hernandez", "torrano"),
                                       ("torrano", "courtney")],
    "anghel-etal2024":                [("anghel",), ("process data", "large-scale assessment",
                                                      "review")],
    "huang-etal2025":                 [("huang", "keller", "missing data"),
                                       ("huang", "machine learning", "process", "2025")],
    "ang-etal2020":                   [("ang", "big educational data"),
                                       ("ang", "analytics", "architecture", "2020")],
    "jiao-etal2021":                  [("jiao", "editorial"), ("jiao", "process data", "2021")],
    "maia-etal023":                   [("maia", "artificial intelligence", "education"),
                                       ("maia", "ai", "ilsa")],
    "rutkowski-etal2024":             [("rutkowski", "limits of inference"),
                                       ("rutkowski", "causality", "international")],
    "stiff-etal2023":                 [("stiff", "pirls"), ("stiff", "research engagement")],
    "scherer-etal2024":               [("scherer", "potential", "large-scale"),
                                       ("scherer", "ilsa", "2024")],
    "fink-etal2024":                  [("fink", "adaptive testing"),
                                       ("fink", "methodological", "cat")],
    "alvarez-etal2024":               [("alvarez", "student profiles"),
                                       ("alvarez", "explainable", "pisa")],
    "zhu-etal2025":                   [("zhu", "mathematics", "u.s."),
                                       ("zhu", "predictive", "pisa", "2025"),
                                       ("zhu", "math", "united states", "2025")],
    "elouafi-etal2025":               [("elouafi",)],
    "gomez-talal-etal2025":           [("gomez", "interpretable machine learning"),
                                       ("interpretable machine learning", "pisa", "mathematics")],
    "zhai-etal2024":                  [("zhai", "psychoemotional"), ("zhai", "optimal",
                                        "machine learning")],
    "rico-juan-etal2024":             [("rico", "juan", "reading"),
                                       ("rico-juan",), ("rico juan",)],
    "khine-etal2024":                 [("khine", "achievement", "2024"),
                                       ("khine", "machine learning", "predict")],
}

MANUAL_SCORES = {
    "bezek-gure-etal2020": 4, "demir-karaboga2021": 4,
    "acisli-celik-yesilkanat2023": 4, "oz-etal2024": 4,
    "tin-etal2024": 4, "song-cutumisu2024": 4,
    "nguyen-etal2023": 4, "pejic-etal2021": 4,
    "zhang-etal2023": 4, "sun-etal2023": 2,
    "fn-aydin-etal2025": 4, "tyack-etal2024": 4,
    "robitzsch-ludtke2022": 1, "hernandez-torrano-courtney2021": 2,
    "anghel-etal2024": 2, "huang-etal2025": 2,
    "ang-etal2020": 2, "jiao-etal2021": 2,
    "maia-etal023": 2, "rutkowski-etal2024": 1,
    "stiff-etal2023": 2, "scherer-etal2024": 2,
    "fink-etal2024": 2, "alvarez-etal2024": 4,
    "zhu-etal2025": 4, "elouafi-etal2025": 4,
    "gomez-talal-etal2025": 4, "zhai-etal2024": 4,
    "rico-juan-etal2024": 4, "khine-etal2024": 4,
}


def _first_page_text(pdf_path: Path, max_chars: int = 3000) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars].lower()
    except Exception:
        return ""


def _matches(text: str, rules: list[tuple[str, ...]]) -> bool:
    for kw_tuple in rules:
        if all(kw.lower() in text for kw in kw_tuple):
            return True
    return False


def find_matches(pdf_dir: Path) -> dict[str, Path]:
    """Return {citekey: pdf_path} for all 30 studies found in pdf_dir."""
    all_pdfs = list(pdf_dir.rglob("*.pdf"))
    print(f"Scanning {len(all_pdfs)} PDFs in {pdf_dir} …\n")

    matched: dict[str, Path] = {}
    claimed: set[Path] = set()

    for citekey, rules in MATCH_RULES.items():
        for pdf_path in all_pdfs:
            if pdf_path in claimed:
                continue
            # fast filename check first
            fname_lower = pdf_path.name.lower()
            fname_match = _matches(fname_lower, rules)
            if fname_match:
                matched[citekey] = pdf_path
                claimed.add(pdf_path)
                break

        if citekey in matched:
            continue

        # full text scan if filename didn't match
        for pdf_path in all_pdfs:
            if pdf_path in claimed:
                continue
            text = _first_page_text(pdf_path)
            if _matches(text, rules):
                matched[citekey] = pdf_path
                claimed.add(pdf_path)
                break

    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", required=True,
                        help="Root directory that contains the 30 PDFs (searched recursively)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show matched PDFs without extracting")
    parser.add_argument("--keys", nargs="+", default=None,
                        help="Restrict to specific citekeys (useful for re-running failures)")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    if not pdf_dir.is_dir():
        print(f"ERROR: {pdf_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    matched = find_matches(pdf_dir)
    not_found = [k for k in MATCH_RULES if k not in matched]

    print(f"{'='*70}")
    print(f"  Matched  : {len(matched)}/30")
    print(f"  Not found: {len(not_found)}")
    print(f"{'='*70}\n")

    for k, p in sorted(matched.items()):
        print(f"  ✓  {k:<40}  {p.name[:50]}")
    for k in not_found:
        print(f"  ✗  {k}")

    if args.dry_run:
        print("\n[dry-run] Extraction skipped.")
        return

    if not_found:
        print(f"\nWARNING: {len(not_found)} studies not found. "
              f"Extraction will continue for the {len(matched)} found ones.")
        input("Press Enter to continue, Ctrl-C to abort …\n")

    keys_to_run = set(args.keys) if args.keys else set(matched.keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor  = GPTExtractor()
    total_cost = 0.0
    ok         = 0
    failures   = []

    for i, citekey in enumerate(keys_to_run, 1):
        if citekey not in matched:
            print(f"\n[{i}/{len(keys_to_run)}] SKIP (not found): {citekey}")
            continue

        pdf_path = matched[citekey]
        safe     = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
        out_path = OUTPUT_DIR / f"{safe}.json"

        print(f"\n{'='*70}")
        print(f"[{i}/{len(keys_to_run)}] {citekey}")
        print(f"  PDF : {pdf_path.name[:65]}")
        print("="*70, flush=True)

        t0        = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="articles")
        if not processed.extraction_text:
            print(f"  SKIP: No text extracted ({processed.parse_errors})")
            failures.append(citekey)
            continue

        result  = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            failures.append(citekey)
            continue

        total_cost += result.cost_usd
        ok += 1
        output = result.extraction.model_dump(mode="json")
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        meta = output["metadata"]
        print(f"  Duration : {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title    : {(meta.get('title') or 'N/A')[:75]}")
        print(f"  src_cat  : {meta.get('source_category')}")
        print(f"  Saved    : {out_path.name}", flush=True)

    print(f"\n{'='*70}")
    print(f"Done: {ok}/{len(keys_to_run)} extracted | Total cost: ${total_cost:.4f}")
    if failures:
        print(f"Failures : {failures}")
        print(f"Re-run with: --keys {' '.join(failures)}")


if __name__ == "__main__":
    main()
