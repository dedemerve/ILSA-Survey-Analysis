# Actionability scores for final CSV (N coding = 212)

**Coding list:** `ilsa_survey_articles/ILSA_Literature_Review_Template_All_Studies.csv` (212 rows, Nos 1–212)

**Study-level scores (joined to final CSV):** `rescored_final_csv_212.csv`

**Extract JSON corpus:** `ilsa_survey_articles/json/`

**Full pipeline rescore table:** `rescored_studies.csv`

**Scorer:** `score_studies_v2.py`

## Current computable descriptives

| | Value |
|--|--|
| CSV rows | 212 |
| Scored (extract available or near-dup mapped) | **204** |
| Still missing extracts | **8** → Nos [133, 134, 143, 173, 187, 188, 202, 209] |
| Mean (SD) | **3.70 (0.80)** |
| Score 1 / 2 / 3 / 4 / 5 | **7 / 21 / 1 / 173 / 2** |
| Sum / N | 754 / 204 = 3.70 |

Near-dup score reuse (same paper, second CSV row): `#136←#2`, `#200←#137`, `#206←#210`.

## LaTeX-ready paragraph (N = 204 scored)

```latex
Across the final coding corpus ($N = 204$ studies with Layer~2 scores in
\texttt{rescored\_final\_csv\_212.csv}; coding list:
\texttt{ilsa\_survey\_articles/ILSA\_Literature\_Review\_Template\_All\_Studies.csv};
extracts: \texttt{ilsa\_survey\_articles/json/}; scorer:
\texttt{score\_studies\_v2.py}), the mean actionability score was
$3.70$ ($\mathrm{SD} = 0.80$). Score distributions were:
Score~1 ($n = 7$), Score~2 ($n = 21$), Score~3 ($n = 1$), Score~4 ($n = 173$), Score~5 ($n = 2$), yielding
$\bar{x} = (1{\times}7+2{\times}21+3{\times}1+4{\times}173+5{\times}2) \div 204 = 754 \div 204 = 3.70$.
Each study receives exactly one integer score (1--5) from a hierarchical
rule set over the $(D_1, D_2, D_3)$ triple. Score assignment was applied
programmatically in Layer~2. The full breakdown by methodological family
is presented in Section~5 alongside Table~\ref{tab:actionability-summary}
and Figure~\ref{fig:actionability_by_method}.
% NOTE: Final CSV lists 212 studies; 8 still lack extracts
% (Nos 133, 134, 143, 173, 187, 188, 202, 209) and are excluded from N=204 until scored.

```

## Pending for full N = 212

Upload/extract JSON for: #133, #134, #143, #173, #187, #188, #202, #209. Then re-run `python score_studies_v2.py` and rebuild `rescored_final_csv_212.csv`.
