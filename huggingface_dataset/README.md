---
license: cc-by-4.0
language:
- en
tags:
- education
- machine-learning
- systematic-review
- PISA
- TIMSS
- PIRLS
- ILSA
- survey
pretty_name: AI & ML in International Large-Scale Assessments — Survey Dataset
size_categories:
- 1K<n<10K
configs:
- config_name: articles_master
  data_files:
  - path: data/articles_master.csv
    split: train
  features:
  - name: doi
    dtype: string
  - name: title
    dtype: string
  - name: authors
    dtype: string
  - name: year
    dtype: string
  - name: source_category
    dtype: string
  - name: journal
    dtype: string
  - name: student_weights_used
    dtype: string
  - name: replicate_weights_used
    dtype: string
  - name: weight_variable_name
    dtype: string
  - name: weight_fields_interpretation
    dtype: string
  - name: plausible_values_handling
    dtype: string
  - name: missing_data_handling
    dtype: string
  - name: handling_not_reported_explanation
    dtype: string
  - name: research_design_type
    dtype: string
  - name: outcome_summary
    dtype: string
  - name: null_fields_interpretation
    dtype: string
  - name: ml_primary
    dtype: string
  - name: ml_all_techniques
    dtype: string
  - name: total_students
    dtype: string
  - name: sample_filtering_criteria
    dtype: string
  - name: countries_formatted
    dtype: string
  - name: effect_size
    dtype: string
  - name: primary_finding
    dtype: string
  - name: ml_techniques
    dtype: string
  - name: sample_size
    dtype: string
  - name: confounders
    dtype: string
  - name: ml_family
    dtype: string
  - name: pv_filter_label
    dtype: string
  - name: md_filter_label
    dtype: string
  - name: weights_filter
    dtype: string
  - name: paper_url
    dtype: string
- config_name: main_findings
  data_files:
  - path: data/main_findings.csv
    split: train
  features:
  - name: doi
    dtype: string
  - name: dataset_used
    dtype: string
  - name: target_variable
    dtype: string
  - name: top_predictors
    dtype: string
  - name: performance_metrics
    dtype: string
  - name: standardized_conclusion
    dtype: string
  - name: source_category
    dtype: string
  - name: effect_size
    dtype: string
  - name: primary_finding
    dtype: string
  - name: target_domain
    dtype: string
  - name: target_dimension
    dtype: string
  - name: predictor_filter_categories
    dtype: string
- config_name: confounders
  data_files:
  - path: data/confounders.csv
    split: train
  features:
  - name: doi
    dtype: string
  - name: variable_code
    dtype: string
  - name: variable_name
    dtype: string
  - name: category
    dtype: string
  - name: source_category
    dtype: string
  - name: predictor_level
    dtype: string
  - name: predictor_category
    dtype: string
- config_name: actionability_scores
  data_files:
  - path: data/actionability_scores.csv
    split: train
  features:
  - name: csv_no
    dtype: string
  - name: authors
    dtype: string
  - name: year
    dtype: string
  - name: title
    dtype: string
  - name: extract_file
    dtype: string
  - name: match
    dtype: string
  - name: source_category
    dtype: string
  - name: D1
    dtype: string
  - name: D2
    dtype: string
  - name: D3
    dtype: string
  - name: n_countries
    dtype: string
  - name: score
    dtype: string
---

# AI & ML in International Large-Scale Assessments: Survey Dataset

**Paper:** *Artificial Intelligence Applications in International Large-Scale Assessments: A Survey with LLM-Assisted Evidence Synthesis*  
**Authors:** Merve Dede & Ekrem Çetinkaya (2026)  
**Website:** [dedemerve.github.io/ILSA-Survey-Extractor](https://dedemerve.github.io/ILSA-Survey-Extractor/)  
**GitHub:** [github.com/dedemerve/ILSA-Survey-Analysis](https://github.com/dedemerve/ILSA-Survey-Analysis)

## Dataset Description

A structured, open evidence repository covering **212 studies** (2020–August 2026) examining how artificial intelligence and machine learning methods are applied to international large-scale assessment (ILSA) data (PISA, TIMSS, PIRLS, TALIS, PIAAC, ICCS, ICILS).

Each full-text article was processed with a schema-constrained LLM-assisted extraction pipeline and **fully human-verified**. The release keeps the original Hugging Face table schemas and adds policy-actionability scores plus per-article JSON extracts.

| Table | Records | Columns | Description |
|---|---:|---:|---|
| `articles_master` | 212 | 31 | Study-level metadata, ML methods, survey design, quality indicators |
| `main_findings` | 382 | 12 | One row per extracted finding / outcome |
| `confounders` | 3,088 | 7 | One row per predictor–study pair |
| `actionability_scores` | 212 | 12 | D₁–D₃ dimensions and composite score (1–5) |

**Corpus summary:** mean policy-actionability score **3.70** (SD = 0.80); **85.8%** of studies score 4–5.

## Files

- `data/articles_master.csv`
- `data/main_findings.csv`
- `data/confounders.csv`
- `data/actionability_scores.csv`
- `ILSA_Survey_Dataset_CLEAN.xlsx` — Excel workbook with the same tables
- `json_extracted/` — verified per-article JSON extracts (**n = 212**), same item schema as the earlier release
- `Q1_ILSA_Policy_Synthesis.csv` — supplementary policy synthesis table (retained from prior release)

## JSON item schema (unchanged)

Each file in `json_extracted/` contains:

- **metadata:** `file_name`, `title`, `authors`, `year`, `doi`, `venue`, `publication_type`, `open_access`, `source_category`
- **data:** `survey_design`, `plausible_values_handling`, `missing_data_handling`, `handling_not_reported_explanation`, `sample_details`, `ml_techniques`, `confounders_identified`, `main_findings`, `outcome_summary`, `research_design_type`, `null_fields_interpretation`
- **main_findings[]:** `dataset_used`, `target_variable`, `top_predictors`, `performance_metrics`, `standardized_conclusion`
- **confounders_identified[]:** `variable_code`, `variable_name`, `category`

## Citation

```bibtex
@article{dede_cetinkaya2026ilsa_survey,
  title   = {Artificial Intelligence Applications in International Large-Scale Assessments: A Survey with LLM-Assisted Evidence Synthesis},
  author  = {Dede, Merve and Çetinkaya, Ekrem},
  year    = {2026},
  note    = {Open dataset: HuggingFace Datasets},
  url     = {https://huggingface.co/datasets/dedemerve/ILSA-Survey-Dataset}
}
```

## License

CC BY 4.0. Metadata extraction is provided for research reuse; original article copyrights remain with their publishers.
