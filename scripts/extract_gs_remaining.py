#!/usr/bin/env python3
"""
Extract PDFs from the Google Scholar folder that have NOT already been
extracted from the Web of Science folder.

Deduplication strategy (filename-independent):
  1. Skip the 30 manually-scored ground-truth studies (MATCH_RULES).
  2. Load titles from every existing JSON in ilsa_survey_articles/json/.
  3. For each Google Scholar PDF, extract first-page text and check whether
     any existing title appears in it (or vice-versa). If so, skip.

Usage (run from project root):

    python scripts/extract_gs_remaining.py \\
        --pdf-dir "~/Desktop/.../Google Scholar"

    python scripts/extract_gs_remaining.py --pdf-dir "..." --dry-run
    python scripts/extract_gs_remaining.py --pdf-dir "..." --force
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

try:
    import fitz
except ImportError:
    print("PyMuPDF not installed.  Run: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

from src.extractors.gpt_extractor import GPTExtractor
from src.extractors.pdf_processor import process_pdf
from scripts.reextract_30_studies import MATCH_RULES as GROUND_TRUTH_RULES

OUTPUT_DIR = PROJECT_ROOT / "ilsa_survey_articles" / "json"

# ── helpers ───────────────────────────────────────────────────────────────────

def _first_page_text(pdf_path: Path, max_chars: int = 4000) -> str:
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


def _matches_ground_truth(pdf_path: Path) -> bool:
    fname_lower = pdf_path.name.lower()
    for rules in GROUND_TRUTH_RULES.values():
        for kw_tuple in rules:
            if all(kw.lower() in fname_lower for kw in kw_tuple):
                return True
    text = _first_page_text(pdf_path)
    for rules in GROUND_TRUTH_RULES.values():
        for kw_tuple in rules:
            if all(kw.lower() in text for kw in kw_tuple):
                return True
    return False


def _normalise(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy title match."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_existing_titles() -> list[str]:
    titles = []
    for fn in os.listdir(OUTPUT_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            doc = json.loads((OUTPUT_DIR / fn).read_text(encoding="utf-8"))
            t = (doc.get("metadata") or {}).get("title") or ""
            n = _normalise(t)
            if len(n) > 10:          # ignore very short / empty titles
                titles.append(n)
        except Exception:
            pass
    return titles


def _already_extracted(pdf_text_norm: str, existing_titles: list[str],
                        min_title_len: int = 15) -> bool:
    """Return True if a sufficiently long existing title appears in the PDF text."""
    for title in existing_titles:
        if len(title) < min_title_len:
            continue
        # use first 60 chars of title as a robust key
        key = title[:60].strip()
        if key and key in pdf_text_norm:
            return True
    return False


def _out_path(pdf_path: Path) -> Path:
    safe = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
    return OUTPUT_DIR / f"{safe}.json"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", required=True,
                        help="Google Scholar PDF folder (searched recursively)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be extracted without extracting")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if JSON output already exists")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    if not pdf_dir.is_dir():
        print(f"ERROR: {pdf_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    all_pdfs = sorted(pdf_dir.rglob("*.pdf"))
    print(f"Found {len(all_pdfs)} PDFs in {pdf_dir}")
    print("Loading existing JSON titles for deduplication …")
    existing_titles = _load_existing_titles()
    print(f"  {len(existing_titles)} existing titles loaded\n")

    skip_gt = skip_dup = skip_exists = 0
    to_extract: list[Path] = []

    print("Classifying PDFs …")
    for pdf in all_pdfs:
        if _matches_ground_truth(pdf):
            skip_gt += 1
            continue
        text_norm = _normalise(_first_page_text(pdf))
        if _already_extracted(text_norm, existing_titles):
            skip_dup += 1
            continue
        to_extract.append(pdf)

    print(f"\n  Ground-truth skipped : {skip_gt}")
    print(f"  Duplicate skipped    : {skip_dup}")
    print(f"  To extract           : {len(to_extract)}\n")

    if args.dry_run:
        print("[dry-run] Would extract:")
        for p in to_extract:
            status = "EXISTS" if _out_path(p).exists() else "new"
            print(f"  [{status}]  {p.name[:80]}")
        print("\n[dry-run] Extraction skipped.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor  = GPTExtractor()
    total_cost = 0.0
    ok = 0
    failures: list[str] = []

    for i, pdf_path in enumerate(to_extract, 1):
        out = _out_path(pdf_path)

        if out.exists() and not args.force:
            print(f"[{i}/{len(to_extract)}] SKIP (exists): {out.name[:70]}")
            ok += 1
            continue

        print(f"\n{'='*70}")
        print(f"[{i}/{len(to_extract)}] {pdf_path.name[:65]}")
        print("="*70, flush=True)

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="articles")
        if not processed.extraction_text:
            print(f"  SKIP: No text ({processed.parse_errors})")
            failures.append(pdf_path.name)
            continue

        result  = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            failures.append(pdf_path.name)
            continue

        total_cost += result.cost_usd
        ok += 1
        output = result.extraction.model_dump(mode="json")
        out.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                       encoding="utf-8")

        meta = output["metadata"]
        print(f"  Duration : {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title    : {(meta.get('title') or 'N/A')[:75]}")
        print(f"  src_cat  : {meta.get('source_category')}")
        print(f"  Saved    : {out.name}", flush=True)

    print(f"\n{'='*70}")
    print(f"Done: {ok}/{len(to_extract)} extracted | Total cost: ${total_cost:.4f}")
    if failures:
        print(f"Failures ({len(failures)}): {failures}")
    print("="*70)


if __name__ == "__main__":
    main()
