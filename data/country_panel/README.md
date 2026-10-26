# Country-level panel for RQ1 (B) and RQ2

The forecasting analyses need **official country mean scores per ILSA cycle**.
They are not in this repository (the article corpus contains no score panel), and they
must come from the official OECD / IEA releases — never typed from memory.

## 1. `ilsa_country_means.csv` (required)

Copy `ilsa_country_means.template.csv` and fill one row per program × domain × grade × country × cycle:

| column | content |
|---|---|
| `program` | `PISA`, `TIMSS`, `PIRLS` |
| `domain` | `Mathematics`, `Reading`, `Science` |
| `grade` | `4` / `8` for TIMSS, `4` for PIRLS, empty for PISA |
| `country_code` | ISO 3166-1 alpha-3 (`TUR`, `FIN`, …); sub-national entities with their ILSA code |
| `year` | cycle year (PISA 2000…2022, TIMSS 1995…2023, PIRLS 2001…2021) |
| `mean_score` | country mean on the international reporting scale |
| `se` | standard error of the mean (optional) |
| `source` | table / file the value came from (optional, recommended) |

Sources
- **PISA**: OECD *PISA Data Explorer* (https://pisadataexplorer.oecd.org) — export "Mean" by country and year for each
  domain; or the trend tables in the annex of *PISA 2022 Results (Volume I)*.
  Use only cycles where the domain was a comparable scale (Reading 2000+, Mathematics 2003+, Science 2006+).
- **TIMSS**: IEA *TIMSS 2023 International Results* trend exhibits / IEA IDB Analyzer, grades 4 and 8.
- **PIRLS**: IEA *PIRLS 2021 International Results* trend exhibits.
- Optional long series: Lee & Lee (2025), *An Extended Dataset of Educational Quality Across Countries
  (1970–2023)*, doi:10.5281/zenodo.16778072 (article no. 1 of the corpus) — harmonised to the TIMSS 1995 scale;
  load it as its own `program` (e.g. `HARMONIZED`) rather than mixing it with official means.

Keep breaks in mind: cycles flagged by OECD/IEA as not comparable should be left out or kept as a separate
series (e.g. a different `grade`/`program` label).

## 2. `ilsa_country_covariates.csv` (required for RQ2)

Wide file `country_code, year, <covariate…>` (see `ilsa_country_covariates.template.csv`). Column names must
match the keys of `config/kb_covariate_map.json`, which links each covariate to the literature knowledge base.

- `gdp_per_capita_log`, `education_expenditure_pct_gdp`: fetched automatically by
  `python scripts/fetch_country_covariates.py` (World Bank WDI).
- `escs_mean`, `ict_home_access`, `anxiety_mean`, `self_efficacy_mean`, `belonging_mean`, `immigrant_share`,
  `school_resources`, `learning_time`, `parental_involvement`: country means from the OECD PISA / IEA
  international databases (same Data Explorer exports as above, index variables by country and cycle).
- `teacher_qualification`: TALIS / TIMSS teacher questionnaire country tables.

Covariates are matched *as of* the forecast origin (latest value ≤ origin year, at most 3 years old), so a
forecast never uses later information. Any subset of the columns may be supplied.

## 3. Run

```bash
python scripts/build_predictor_knowledge_base.py      # literature KB → covariate priors
python scripts/rq1_forecast_comparison.py             # RQ1 (B)
python scripts/rq2_kb_forecast_ablation.py            # RQ2
```
