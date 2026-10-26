# ILSA Survey Analysis

Structured metadata extracted from **212 research articles** on machine learning with International
Large-Scale Assessment (ILSA) data (PISA, TIMSS, PIRLS, TALIS, PIAAC, ICILS, ICCS), plus the analysis
pipeline for two research questions:

- **RQ1.** Which ML models are most frequently used to predict student achievement in ILSA research, and how
  does the forecasting performance of the most prevalent models (Random Forest, XGBoost) compare with
  time-series models in predicting country-level mean achievement across ILSA programs and domains?
- **RQ2.** To what extent can a structured knowledge base derived from the ILSA research corpus be integrated
  with time-series forecasting models to improve predictions of country-level mean achievement?

## Repository map

| Path | Content |
|---|---|
| `ilsa_survey_articles/json/` | The study corpus: 219 JSON files = 212 unique articles (7 articles were extracted twice; loaders collapse them by DOI + title) |
| `ilsa_survey_articles/wos_manifest.json` | Web of Science PDF manifest |
| `src/extractors/`, `src/schemas/` | PDF → LLM → Pydantic extraction pipeline and schema |
| `src/analysis/` | Canonical ML taxonomy, rule-derived study attributes, metric parser, corpus loader |
| `src/knowledge/` | Predictor-level knowledge base (RQ2) |
| `src/forecasting/` | Country panel, time-series / pooled ML forecasters, rolling-origin evaluation |
| `src/enrichment/` | Variable taxonomy and the outcome-level semantic knowledge base |
| `config/kb_covariate_map.json` | Links country-level covariates to knowledge-base predictor groups |
| `data/country_panel/` | Where the country mean-score panel and covariates go (see its README) |
| `outputs/` | Generated tables (see below) |
| `tests/` | Unit and leakage tests (`pytest`) |

## Pipeline

```bash
pip install -r requirements.txt

# Corpus tables and the outcome-level knowledge base (212 articles)
python scripts/build_tabular_dataset.py               # outputs/ILSA_Meta_Analysis_Dataset[_CLEAN].xlsx
python scripts/build_structured_meta_analysis.py      # outputs/ILSA_Structured_Meta_Analysis.xlsx
python scripts/build_semantic_knowledge_base_v2.py    # outputs/final_knowledge_synthesis_v4.csv, semantic_knowledge_base.json
python score_studies_v2.py                            # rescored_studies.csv (policy-actionability rubric)

# RQ1 (A): model frequency and within-study RF vs XGBoost comparisons
python scripts/rq1_literature_analysis.py             # outputs/rq1/

# RQ2: predictor-level knowledge base → priors for country-level covariates
python scripts/build_predictor_knowledge_base.py      # outputs/knowledge_base/

# RQ1 (B) and RQ2 forecasting — need the country panel (data/country_panel/README.md)
python scripts/fetch_country_covariates.py            # World Bank covariates (needs internet)
python scripts/rq1_forecast_comparison.py             # outputs/rq1_forecast/
python scripts/rq2_kb_forecast_ablation.py            # outputs/rq2_kb_forecast/

# Human validation of the automated coding (Cohen's kappa)
python scripts/validation_sample.py draw --n 30       # outputs/validation/coding_sheet.csv
python scripts/validation_sample.py score <coderA.csv> [<coderB.csv>]

pytest tests
```

### RQ1 (A) — literature
`scripts/rq1_literature_analysis.py` canonicalises model names with word-boundary rules
(`src/analysis/ml_taxonomy.py`), derives program / cycle / domain / target type / analysis level / task /
validation scheme for every finding (`src/analysis/study_attributes.py`), parses free-text metrics into a
model × metric × value table (`src/analysis/performance_parser.py`) and reports:
model frequency (mention and primary model) for all ML articles and for the RQ1 population (empirical ML
studies with an achievement target), by program and domain; publication-year trends; within-study
Random Forest vs XGBoost comparisons; a pairwise win matrix; the best model per finding.

### RQ1 (B) — forecasting
`scripts/rq1_forecast_comparison.py` forecasts the next cycle of every country series with time-series models
(naive, mean, drift, damped Holt, AICc-selected ARIMA) and pooled models (panel AR, Random Forest, XGBoost)
in a rolling-origin design, then compares RMSE / MAE / bias / MAE relative to naive overall, per
program × domain and per origin, with country-clustered Diebold–Mariano tests.

### RQ2 — knowledge-base integration
`scripts/build_predictor_knowledge_base.py` aggregates, over achievement findings, how often each predictor
group is a top predictor (salience) and its reported direction; `config/kb_covariate_map.json` links groups
to country-level covariates. `scripts/rq2_kb_forecast_ablation.py` integrates the knowledge base by covariate
selection, sign / monotone constraints and evidence-weighted shrinkage, and compares each variant with the
same model without the knowledge base, with all covariates, and with random covariate subsets of equal size.

## Data structure of an article JSON

`metadata`: `file_name`, `title`, `authors`, `year`, `doi`, `venue`, `publication_type`, `open_access`,
`source_category`.

`data`: `survey_design` (weights), `plausible_values_handling`, `missing_data_handling`, `sample_details`
(students, countries), `ml_techniques` (`primary`, `all_techniques`), `confounders_identified` (13 predictor
categories: socioeconomic, demographic, student attitude, student behaviour, teacher, school, ICT,
curriculum, parent/home, process data, prior achievement, peer effects, system level), `main_findings`
(`dataset_used`, `target_variable`, `top_predictors`, `performance_metrics`, `standardized_conclusion`),
`outcome_summary`, `research_design_type`.

Schema v2 (`src/schemas/models.py`) adds optional per-finding fields for future extractions:
`ilsa_program`, `assessment_cycles`, `domain`, `analysis_level`, `task_type`, `validation_scheme`,
`predictor_effects` (direction / size per top predictor) and `model_performance` (one value per model and
metric). Existing JSON files remain valid; when these fields are present the analysis uses them instead of
the rule-based derivations.

## Known data issues (corpus left unchanged)

- `53. Tin et al. (2024)` carries the DOI of `168. Chen et al. (2021)`; loaders keep both records and log a
  `DOI CONFLICT` line instead of merging them.
- Seven articles have two JSON files (e.g. `81.`/`123.` Schipper et al.); loaders collapse them.

## Citation

If you use this dataset, please cite the original research articles and acknowledge this collection.

## License

The metadata extraction is provided for research purposes. Original article copyrights remain with their
respective publishers.
