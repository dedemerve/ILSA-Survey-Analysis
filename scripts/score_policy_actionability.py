#!/usr/bin/env python3
"""Score extracted ILSA articles with the policy-actionability rubric (D1–D3).

Usage:
  python scripts/score_policy_actionability.py
  python scripts/score_policy_actionability.py --json-dir ilsa_survey_articles/json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.enrichment.policy_actionability import score_corpus

DEFAULT_JSON_DIR = PROJECT_ROOT / "ilsa_survey_articles" / "json"
DEFAULT_OUT = PROJECT_ROOT / "ilsa_survey_articles" / "policy_actionability_scores.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = score_corpus(args.json_dir)
    payload = {
        "rubric": "policy_actionability_D1_D2_D3",
        "n_studies": len(records),
        "score_counts": dict(sorted(Counter(r["Score"] for r in records).items())),
        "d1_counts": dict(sorted(Counter(r["D1"] for r in records).items())),
        "studies": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            core = {
                "D1": rec["D1"],
                "D2": rec["D2"],
                "D3": rec["D3"],
                "Score": rec["Score"],
                "D1_evidence": rec["D1_evidence"],
                "D2_evidence": rec["D2_evidence"],
                "D3_evidence": rec["D3_evidence"],
                "Justification": rec["Justification"],
            }
            fh.write(json.dumps(core, ensure_ascii=False) + "\n")

    print(f"Scored {len(records)} studies")
    print("Score counts:", payload["score_counts"])
    print("D1 counts:", payload["d1_counts"])
    print(f"Wrote {args.output}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
