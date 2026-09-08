#!/usr/bin/env python3
"""
Re-extract the 30 manually scored studies using the updated prompt.
PDFs must be in ~/Desktop/articles/ with their numbered prefix.

Usage:
    python scripts/reextract_30_studies.py
    python scripts/reextract_30_studies.py --dry-run   # list matched PDFs only
    python scripts/reextract_30_studies.py --nums 20 45 53  # subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.extractors.gpt_extractor import GPTExtractor
from src.extractors.pdf_processor import process_pdf

ARTICLES_DIR = Path.home() / "Desktop" / "articles"
OUTPUT_DIR   = PROJECT_ROOT / "ilsa_survey_articles" / "json"

# The 30 manually scored studies: numbered JSON prefix → citekey
TARGET_PREFIXES = {
    "20.", "11.", "8.",  "45.", "53.",  "5.",  "35.", "91.",
    "112.","127.","75.", "113.","94.",  "69.", "67.", "71.",
    "110.","99.", "70.", "118.","105.","103.","109.","36.",
    "2.",  "6.",  "12.", "106.","50.",  "65.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print matched PDFs without extracting")
    parser.add_argument("--nums", nargs="+", default=None,
                        help="Override: extract only these number prefixes (e.g. 20 45)")
    args = parser.parse_args()

    targets = {f"{n}." for n in args.nums} if args.nums else TARGET_PREFIXES

    pdfs = sorted(
        p for p in ARTICLES_DIR.glob("*.pdf")
        if any(p.name.startswith(t) for t in targets)
    )

    if not pdfs:
        print(f"No matching PDFs found in {ARTICLES_DIR}")
        print(f"Expected prefixes: {sorted(targets)}")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDFs to re-extract → {OUTPUT_DIR}\n")
    for p in pdfs:
        print(f"  {p.name[:70]}")

    if args.dry_run:
        print("\n[dry-run] No extraction performed.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor  = GPTExtractor()
    total_cost = 0.0
    ok = 0

    for i, pdf_path in enumerate(pdfs, 1):
        safe_name = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
        out_path  = OUTPUT_DIR / f"{safe_name}.json"

        print(f"\n{'='*70}")
        print(f"[{i}/{len(pdfs)}] {pdf_path.name[:65]}")
        print("="*70, flush=True)

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="articles")
        if not processed.extraction_text:
            print(f"  SKIP: No text ({processed.parse_errors})")
            continue

        result  = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            continue

        total_cost += result.cost_usd
        ok += 1
        output = result.extraction.model_dump(mode="json")
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        meta = output["metadata"]
        print(f"  Duration : {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title    : {(meta.get('title') or 'N/A')[:75]}")
        print(f"  src_cat  : {meta.get('source_category')}")
        print(f"  Saved    : {out_path.name}", flush=True)

    print(f"\n{'='*70}")
    print(f"Done: {ok}/{len(pdfs)} extracted | Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
