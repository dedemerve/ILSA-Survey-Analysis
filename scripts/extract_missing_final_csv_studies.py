#!/usr/bin/env python3
"""
Extract + score the Final-CSV studies that still lack an LLM extract.

Targets (unique papers; #187/#206 are the same Delprato paper → extract once as #187):
  #133 Kraus, #134 Chen, #143 Bezek Güre, #173 Mang,
  #187 Delprato, #188 König, #202 Elouafi, #209 Qabliyane

Usage (from project root):

    python scripts/extract_missing_final_csv_studies.py \\
        --pdf-dir /path/to/folder_with_pdfs --dry-run

    python scripts/extract_missing_final_csv_studies.py \\
        --pdf-dir /path/to/Google_Scholar \\
        --pdf-dir /path/to/Web_of_Science

    python scripts/extract_missing_final_csv_studies.py --pdf-dir ... --score

Requires OPENAI_API_KEY and: pip install openai pymupdf python-dotenv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

try:
    import fitz
except ImportError:
    print("PyMuPDF not installed. Run: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

OUTPUT_DIR = PROJECT_ROOT / "ilsa_survey_articles" / "json"

# csv_no -> (stem_slug_for_filename, match keyword tuples)
# A PDF matches if first-page text OR filename contains ALL keywords in at least one tuple.
TARGETS: dict[int, dict] = {
    133: {
        "slug": "Kraus et al",
        "year": "2026",
        "title": "The Essence of Creative Thinking in PISA 2022: An Item-Level Machine Learning Approach to Construct Validity at the Facet Level",
        "rules": [
            ("kraus", "creative thinking", "pisa"),
            ("kraus", "goecke", "jaggy"),
            ("essence of creative thinking",),
        ],
    },
    134: {
        "slug": "Chen et al",
        "year": "2026",
        "title": "Interpretable Machine Learning Reveals Key Determinants of Creative Thinking Among East Asian Students: Evidence From PISA 2022",
        "rules": [
            ("chen", "creative thinking", "east asian"),
            ("chen", "lightgbm", "creative"),
            ("key determinants of creative thinking",),
        ],
    },
    143: {
        "slug": "Bezek Güre et al",
        "year": "2023",
        "title": "Reviewing the Factors Affecting PISA Reading Skills by Using Random Forest and MARS Methods",
        "rules": [
            ("bezek", "gure", "mars"),
            ("bezek", "güre"),
            ("bezek güre", "reading"),
            ("random forest", "mars", "pisa reading"),
        ],
    },
    173: {
        "slug": "Mang et al",
        "year": "2021",
        "title": "Sampling Weights in Multilevel Modelling: An Investigation Using PISA Sampling Structures",
        "rules": [
            ("mang", "sampling weights", "multilevel"),
            ("küchenhoff", "meinck"),
            ("kuchenhoff", "meinck"),
            ("sampling weights in multilevel modelling",),
        ],
    },
    187: {
        "slug": "Delprato",
        "year": "2026",
        "title": "Cognitive and Non-Cognitive Efficiency Gaps Between Private and Public Schools and Their Determinants in the Latin American Context",
        "rules": [
            ("delprato", "efficiency", "private"),
            ("delprato", "dea"),
            ("non-cognitive efficiency gaps",),
            ("cognitive and non-cognitive efficiency",),
        ],
    },
    188: {
        "slug": "König et al",
        "year": "2026",
        "title": "Effects of the 'highly adaptive testing design for PISA' on plausible value-based population estimates",
        "rules": [
            ("könig", "adaptive", "pisa"),
            ("konig", "adaptive", "pisa"),
            ("koenig", "adaptive", "pisa"),
            ("highly adaptive testing design", "plausible value"),
        ],
    },
    202: {
        "slug": "Elouafi et al",
        "year": "2025",
        "title": "Explaining Student Absenteeism in Morocco Using TIMSS 2023 Data: A Deep Learning and Explainable AI Approach",
        "rules": [
            ("elouafi", "absenteeism"),
            ("absenteeism", "morocco", "timss"),
            ("nouna", "absenteeism"),
        ],
    },
    209: {
        "slug": "Qabliyane et al",
        "year": "2026",
        "title": "Supervised Machine Learning-Based Multiclass Classification and Interpretable Feature Importance Analysis of Teacher Job Satisfaction",
        "rules": [
            ("qabliyane",),
            ("qabliyane", "teacher", "job"),
            ("multiclass", "teacher job satisfaction", "talis"),
        ],
    },
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", s)


def _first_page_text(pdf_path: Path, max_chars: int = 5000) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN cannot read {pdf_path.name}: {exc}")
        return ""


def _matches(blob: str, rules: list[tuple[str, ...]]) -> bool:
    b = _norm(blob)
    for tup in rules:
        if all(_norm(kw) in b for kw in tup):
            return True
    return False


def _trunc_title(title: str, max_len: int = 55) -> str:
    title = re.sub(r"\s+", " ", title).strip().replace("/", "-").replace(":", "")
    if len(title) <= max_len:
        return title
    cut = title[:max_len]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip(" .-")


def _out_stem(no: int, meta: dict) -> str:
    slug = meta["slug"]
    author_part = slug if slug.endswith(".") else slug + "."
    return f"{no}. {author_part} ({meta['year']}). {_trunc_title(meta['title'])}"


def _already_present(no: int) -> Path | None:
    hits = list(OUTPUT_DIR.glob(f"{no}. *.json"))
    return hits[0] if hits else None


def _collect_pdfs(dirs: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for d in dirs:
        d = d.expanduser().resolve()
        if not d.exists():
            print(f"WARN missing pdf-dir: {d}")
            continue
        pdfs.extend(sorted(d.rglob("*.pdf")))
        pdfs.extend(sorted(d.rglob("*.PDF")))
    # dedupe by resolve
    seen = set()
    out = []
    for p in pdfs:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pdf-dir",
        action="append",
        required=True,
        help="Folder with PDFs (repeatable). Search is recursive.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Match only; do not call LLM")
    ap.add_argument("--force", action="store_true", help="Re-extract even if JSON exists")
    ap.add_argument(
        "--score",
        action="store_true",
        help="After extraction, run score_studies_v2.py",
    )
    ap.add_argument(
        "--nos",
        nargs="*",
        type=int,
        default=None,
        help="Subset of CSV numbers to process (default: all targets)",
    )
    args = ap.parse_args()

    wanted = set(args.nos) if args.nos else set(TARGETS)
    targets = {k: v for k, v in TARGETS.items() if k in wanted}

    pdfs = _collect_pdfs([Path(p) for p in args.pdf_dir])
    print(f"PDF files found: {len(pdfs)}")
    print(f"Targets: {sorted(targets)}")

    # Match each target to at most one PDF
    assignments: dict[int, Path] = {}
    unmatched_targets = []
    used_pdfs: set[str] = set()

    # Prefer filename matches first (cheaper), then first-page text
    for no, meta in targets.items():
        existing = _already_present(no)
        if existing and not args.force:
            print(f"#{no} already extracted: {existing.name}")
            continue
        hit = None
        for pdf in pdfs:
            if str(pdf.resolve()) in used_pdfs:
                continue
            if _matches(pdf.name, meta["rules"]):
                hit = pdf
                break
        if hit is None:
            for pdf in pdfs:
                if str(pdf.resolve()) in used_pdfs:
                    continue
                text = _first_page_text(pdf)
                if _matches(pdf.name + "\n" + text, meta["rules"]):
                    hit = pdf
                    break
        if hit is None:
            unmatched_targets.append(no)
            print(f"#{no} NO PDF MATCH for '{meta['title'][:60]}'")
        else:
            used_pdfs.add(str(hit.resolve()))
            assignments[no] = hit
            print(f"#{no} -> {hit}")

    if args.dry_run:
        print(f"\nDry-run done. matched={len(assignments)} unmatched={unmatched_targets}")
        return

    if not assignments:
        print("Nothing to extract.")
        if args.score:
            os.system(f"{sys.executable} {PROJECT_ROOT / 'score_studies_v2.py'}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    from src.extractors.gpt_extractor import GPTExtractor
    from src.extractors.pdf_processor import process_pdf

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = GPTExtractor()
    ok = 0
    total_cost = 0.0

    for no, pdf_path in sorted(assignments.items()):
        meta = targets[no]
        stem = _out_stem(no, meta)
        out_path = OUTPUT_DIR / f"{stem}.json"
        print(f"\n{'=' * 70}")
        print(f"Extracting #{no}: {pdf_path.name[:70]}")
        print(f"  -> {out_path.name}")
        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="final_csv_missing")
        if not processed.extraction_text:
            print(f"  SKIP: no text ({processed.parse_errors})")
            continue
        result = extractor.extract(processed)
        elapsed = time.perf_counter() - t0
        if not result.success:
            print(f"  FAILED: {result.error}")
            continue
        total_cost += result.cost_usd
        output = result.extraction.model_dump(mode="json")
        output.setdefault("metadata", {})
        output["metadata"]["file_name"] = stem + ".pdf"
        if not (output["metadata"].get("title") or "").strip():
            output["metadata"]["title"] = meta["title"]
        # remove any prior file for this number
        for old in OUTPUT_DIR.glob(f"{no}. *.json"):
            if old != out_path:
                old.unlink()
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ok += 1
        print(f"  OK {elapsed:.1f}s ${result.cost_usd:.4f} title={(output['metadata'].get('title') or '')[:70]}")

    print(f"\nExtracted {ok}/{len(assignments)} | cost ${total_cost:.4f}")
    print(f"Unmatched targets: {unmatched_targets}")

    if args.score:
        print("\nRunning score_studies_v2.py ...")
        rc = os.system(f"{sys.executable} {PROJECT_ROOT / 'score_studies_v2.py'}")
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
