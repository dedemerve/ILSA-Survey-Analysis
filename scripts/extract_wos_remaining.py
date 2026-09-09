#!/usr/bin/env python3
"""
Extract all PDFs from the Web of Science folder that are NOT among the 30
manually scored ground-truth studies.

Saves JSONs to ilsa_survey_articles/json/ and writes a manifest of processed
PDF filenames to ilsa_survey_articles/wos_manifest.json so a later run over a
second folder can skip duplicates.

Usage (run from project root):

    python scripts/extract_wos_remaining.py \\
        --pdf-dir "~/Desktop/.../Web of Science"

    python scripts/extract_wos_remaining.py \\
        --pdf-dir "..." --dry-run        # show what would be extracted

    python scripts/extract_wos_remaining.py \\
        --pdf-dir "..." --force          # re-extract even if JSON exists
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

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed.  Run: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

from src.extractors.gpt_extractor import GPTExtractor
from src.extractors.pdf_processor import process_pdf

OUTPUT_DIR  = PROJECT_ROOT / "ilsa_survey_articles" / "json"
MANIFEST    = PROJECT_ROOT / "ilsa_survey_articles" / "wos_manifest.json"

# ── Import 30-study match rules (reuse exactly, no duplication) ───────────────
from scripts.reextract_30_studies import MATCH_RULES as GROUND_TRUTH_RULES


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _matches_any_rule(pdf_path: Path, rules_map: dict) -> bool:
    """Return True if this PDF matches any citekey in rules_map."""
    fname_lower = pdf_path.name.lower()
    for rules in rules_map.values():
        for kw_tuple in rules:
            if all(kw.lower() in fname_lower for kw in kw_tuple):
                return True
    # Full-text scan only when filename gave nothing
    text = _first_page_text(pdf_path)
    for rules in rules_map.values():
        for kw_tuple in rules:
            if all(kw.lower() in text for kw in kw_tuple):
                return True
    return False


def _out_path(pdf_path: Path) -> Path:
    safe = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
    return OUTPUT_DIR / f"{safe}.json"


def _load_manifest() -> set[str]:
    if MANIFEST.exists():
        return set(json.loads(MANIFEST.read_text(encoding="utf-8")))
    return set()


def _save_manifest(names: set[str]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(sorted(names), indent=2, ensure_ascii=False),
                        encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", required=True,
                        help="Web of Science PDF folder (searched recursively)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be extracted without actually extracting")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if JSON already exists")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    if not pdf_dir.is_dir():
        print(f"ERROR: {pdf_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    all_pdfs = sorted(pdf_dir.rglob("*.pdf"))
    print(f"Found {len(all_pdfs)} PDFs in {pdf_dir}\n")

    # ── Partition: ground-truth (skip) vs remaining (extract) ─────────────────
    to_skip:    list[Path] = []
    to_extract: list[Path] = []

    print("Classifying PDFs …")
    for pdf in all_pdfs:
        if _matches_any_rule(pdf, GROUND_TRUTH_RULES):
            to_skip.append(pdf)
        else:
            to_extract.append(pdf)

    print(f"\n  Ground-truth (30, skipped) : {len(to_skip)}")
    print(f"  Remaining (to extract)     : {len(to_extract)}\n")

    if args.dry_run:
        print("[dry-run] Would extract:")
        for p in to_extract:
            status = "EXISTS" if _out_path(p).exists() else "new"
            print(f"  [{status}]  {p.name[:80]}")
        print("\n[dry-run] Extraction skipped.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    extractor  = GPTExtractor()
    total_cost = 0.0
    ok = 0
    skipped_exists = 0
    failures: list[str] = []

    for i, pdf_path in enumerate(to_extract, 1):
        out = _out_path(pdf_path)

        if out.exists() and not args.force:
            print(f"[{i}/{len(to_extract)}] SKIP (exists): {out.name[:70]}")
            manifest.add(pdf_path.name)
            skipped_exists += 1
            ok += 1
            continue

        print(f"\n{'='*70}")
        print(f"[{i}/{len(to_extract)}] {pdf_path.name[:65]}")
        print("="*70, flush=True)

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="articles")
        if not processed.extraction_text:
            print(f"  SKIP: No text extracted ({processed.parse_errors})")
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
        out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.add(pdf_path.name)
        _save_manifest(manifest)

        meta = output["metadata"]
        print(f"  Duration : {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title    : {(meta.get('title') or 'N/A')[:75]}")
        print(f"  src_cat  : {meta.get('source_category')}")
        print(f"  Saved    : {out.name}", flush=True)

    # ── Final manifest save ───────────────────────────────────────────────────
    _save_manifest(manifest)

    print(f"\n{'='*70}")
    print(f"Done: {ok}/{len(to_extract)} extracted | "
          f"({skipped_exists} skipped-existing) | Total cost: ${total_cost:.4f}")
    print(f"Manifest saved: {MANIFEST}  ({len(manifest)} filenames)")
    if failures:
        print(f"Failures ({len(failures)}): {failures}")
    print("="*70)


if __name__ == "__main__":
    main()
