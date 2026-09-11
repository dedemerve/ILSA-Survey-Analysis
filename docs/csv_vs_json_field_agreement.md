# Manual CSV vs LLM JSON — field-level agreement

**Gold standard:** `ILSA Literature Review Template(All Studies).csv` (author-coded, N=212)
**Comparison:** LLM extracts in `ilsa_survey_articles/json/` (+ HF `json_extractions/` fill-ins when local JSON missing)

- CSV rows: **212**
- Studies with a matched JSON: **212**
- CSV rows with no JSON match: **0**

## Summary table

| Field | N evaluable | Agree | Partial | Disagree | Missing/unavailable | Agree % | Agree+Partial % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `year` | 212 | 201 | 8 | 3 | 0 | 94.8 | 98.6 |
| `title` | 212 | 210 | 1 | 1 | 0 | 99.1 | 99.5 |
| `authors` | 212 | 113 | 24 | 75 | 0 | 53.3 | 64.6 |
| `ilsa_assessment` | 196 | 188 | 3 | 5 | 16 | 95.9 | 97.4 |
| `ml_methods` | 198 | 127 | 35 | 36 | 14 | 64.1 | 81.8 |
| `software` | 162 | 2 | 37 | 123 | 50 | 1.2 | 24.1 |
| `countries` | 168 | 101 | 25 | 42 | 44 | 60.1 | 75.0 |
| `dependent_outcome` | 212 | 88 | 78 | 46 | 0 | 41.5 | 78.3 |
| `journal_venue` | 198 | 178 | 11 | 9 | 14 | 89.9 | 95.5 |

## Interpretation notes

- This is **author-manual CSV (gold) vs LLM JSON**, not dual human coding.
- Free-text fields (methods, DV, countries) use fuzzy/keyword overlap; they understate true semantic agreement when wording differs.
- `software` is weakly represented in the JSON schema, so many rows are *unavailable* rather than true disagreements.
- Strongest structural matches expected on `year`, `title`, `authors`, `ilsa_assessment`, and coarse `ml_methods`.

## Files

- Detail: `csv_vs_json_field_agreement_detail.csv`
- Summary: `csv_vs_json_field_agreement_summary.csv`
