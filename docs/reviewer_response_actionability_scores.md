# Actionability scores for final CSV (N = 212)

**Coding list:** `ilsa_survey_articles/ILSA_Literature_Review_Template_All_Studies.csv` (212 rows)

**Extract JSON corpus:** `ilsa_survey_articles/json/` (212 files; local path on author machine: `~/Desktop/ILSA-Survey-Analysis/ilsa_survey_articles/json/`)

**Study-level scores:** `rescored_studies.csv` (produced by Layer-2 scorer)

**Scorer:** `score_studies_v2.py`

## Descriptives (N = 212)

| | Value |
|--|--|
| N | **212** |
| Mean (SD) | **3.70 (0.80)** |
| Score 1 / 2 / 3 / 4 / 5 | **7 / 22 / 1 / 180 / 2** |
| Sum / N | 784 / 212 = 3.70 |

## LaTeX-ready paragraph

```latex
Across the corpus ($N = 212$ scored extracts in
\texttt{ilsa\_survey\_articles/json/}; study-level scores in
\texttt{rescored\_studies.csv}; scorer: \texttt{score\_studies\_v2.py}),
the mean actionability score was $3.70$ ($\mathrm{SD} = 0.80$).
Score distributions were: Score~1 ($n = 7$), Score~2 ($n = 22$),
Score~3 ($n = 1$), Score~4 ($n = 180$), and Score~5 ($n = 2$), yielding
$\bar{x} = (1{\times}7 + 2{\times}22 + 3{\times}1 + 4{\times}180 + 5{\times}2)
\div 212 = 784 \div 212 = 3.70$.
Each study receives exactly one integer score (1--5) from a hierarchical
rule set over the $(D_1, D_2, D_3)$ triple (Score~5:
Causal\,+\,Quantified\,+\,Bounded; Score~4: Correlational\,+\,Quantified,
$D_3$ unrestricted; Score~3: Correlational\,+\,Directional; lower scores
cover Synthesis/$D_2$=None/Global combinations as specified in the Methods).
Score assignment was applied programmatically in Layer~2; the scoring
function is included in the publicly available pipeline scripts
(see Data Availability Statement). The full breakdown by methodological
family is presented in Section~5 alongside
Table~\ref{tab:actionability-summary} and
Figure~\ref{fig:actionability_by_method}.
```
