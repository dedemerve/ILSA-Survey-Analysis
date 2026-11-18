#!/usr/bin/env python3 -I
"""Phase 1: Corpus reconciliation — extract cite keys from Table 6, Section 4 text,
and cross-check against references.bib."""

import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("audit/main-7.tex")
BIB = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("audit/references-2.bib")

tex = TEX.read_text(encoding="utf-8")
bib_text = BIB.read_text(encoding="utf-8")

bib_keys = set(re.findall(r'@\w+\{([^,\s]+),', bib_text))
print(f"=== BIB ENTRIES: {len(bib_keys)} ===")

tab6_match = re.search(
    r'\\label\{tab:taxonomy-ilsa\}(.*?)\\end\{table\*\}',
    tex, re.DOTALL
)
tab6_text = tab6_match.group(1) if tab6_match else ""

tab6_cites_raw = re.findall(r'\\textcite\{([^}]+)\}', tab6_text)
tab6_keys_all = []
for c in tab6_cites_raw:
    for k in re.split(r'\s*,\s*', c):
        k = k.strip()
        if k:
            tab6_keys_all.append(k)

tab6_unique = sorted(set(tab6_keys_all))
tab6_counts = Counter(tab6_keys_all)
tab6_dupes = {k: v for k, v in tab6_counts.items() if v > 1}

print(f"Table 6: {len(tab6_keys_all)} total, {len(tab6_unique)} unique, {len(tab6_dupes)} dupes")

cat_keys = defaultdict(set)
tab6_rows = re.findall(
    r'\\ref\{(subsubsec[\d.]+)\}\s+([A-Z].*?)\n\s*&\s*(.*?)\\\\',
    tab6_text, re.DOTALL
)
for ref, heading, studies_text in tab6_rows:
    keys_in_row = []
    for c in re.findall(r'\\textcite\{([^}]+)\}', studies_text):
        for k in re.split(r'\s*,\s*', c):
            k = k.strip()
            if k:
                keys_in_row.append(k)
    m = re.match(r'subsubsec(4\.\d+)', ref)
    if m:
        cat_keys[m.group(1)].update(keys_in_row)

grand_total = set()
for cat in sorted(cat_keys):
    grand_total.update(cat_keys[cat])
    print(f"  Section {cat}: {len(cat_keys[cat])} unique")
print(f"  GRAND TOTAL: {len(grand_total)}")

not_in_bib = sorted(grand_total - bib_keys)
print(f"Table 6 keys NOT in bib: {not_in_bib}")

all_cites = []
for pattern in [r'\\textcite\{([^}]+)\}', r'\\citep?\{([^}]+)\}']:
    for c in re.findall(pattern, tex):
        for k in re.split(r'\s*,\s*', c):
            k = k.strip()
            if k:
                all_cites.append(k)
all_unique = set(all_cites)
print(f"All doc citations: {len(all_unique)} unique")
print(f"In doc but not in bib: {sorted(all_unique - bib_keys)}")
