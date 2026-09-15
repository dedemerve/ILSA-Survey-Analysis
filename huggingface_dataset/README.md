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
- config_name: main_findings
  data_files:
  - path: data/main_findings.csv
    split: train
- config_name: confounders
  data_files:
  - path: data/confounders.csv
    split: train
- config_name: actionability_scores
  data_files:
  - path: data/actionability_scores.csv
    split: train
---

# AI & ML in International Large-Scale Assessments: Survey Dataset

**Paper:** *Artificial Intelligence Applications in International Large-Scale Assessments: A Survey with LLM-Assisted Evidence Synthesis*  
**Authors:** Merve Dede & Ekrem Çetinkaya (2026)  
**Website:** [dedemerve.github.io/ILSA-Survey-Extractor](https://dedemerve.github.io/ILSA-Survey-Extractor/)  
**GitHub:** [github.com/dedemerve/ILSA-Survey-Analysis](https://github.com/dedemerve/ILSA-Survey-Analysis)

## Dataset Description

A structured, open evidence repository covering **212 studies** (2020–August 2026) examining how artificial intelligence and machine learning methods are applied to international large-scale assessment (ILSA) data (PISA, TIMSS, PIRLS, TALIS, PIAAC, ICCS, ICILS).

Each full-text article was processed with a schema-constrained LLM-assisted extraction pipeline and **fully human-verified**. The release includes three relational tables, policy-actionability scores, and per-article JSON extracts.

| Table | Records | Description |
|---|---:|---|
| `articles_master` | 212 | One row per study (metadata, ML methods, survey design, quality indicators) |
| `main_findings` | 382 | One row per extracted finding |
| `confounders` | 3088 | One row per predictor–study pair (13 taxonomy categories) |
| `actionability_scores` | 212 | D₁–D₃ dimensions and composite score (1–5) |

**Corpus summary:** mean policy-actionability score **3.70** (SD = 0.80); **85.8%** of studies score 4–5. Synthesis uses a five-category framework (predictive modelling, process data / LA, socio-emotional modelling, assessment engineering, computational psychometrics).

## Files

- `data/articles_master.csv`
- `data/main_findings.csv`
- `data/confounders.csv`
- `data/actionability_scores.csv`
- `ILSA_Survey_Dataset_CLEAN.xlsx`
- `json_extracted/` — per-article verified JSON extracts (n = 212)

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
