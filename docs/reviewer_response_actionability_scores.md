# Updated manuscript paragraph + reviewer note (policy actionability scores)

Corpus after re-extraction with the expanded `source_category` prompt and
rescoring with `score_studies_v2.py`: **219 unique studies** in
`ilsa_survey_articles/json/` (not 222 — three files under `json_test/`
duplicate names already present in `json/`).

## Corrected Results paragraph (replace the n = 130 text)

## Final LaTeX-ready paragraph

```latex
Across the corpus, the mean actionability score was $3.71$
($\mathrm{SD} = 0.79$). Score distributions were: Score~1 ($n = 7$),
Score~2 ($n = 22$), Score~3 ($n = 1$), Score~4 ($n = 187$), and
Score~5 ($n = 2$), yielding
$\bar{x} = (1{\times}7 + 2{\times}22 + 3{\times}1 + 4{\times}187 + 5{\times}2)
\div 219 = 812 \div 219 = 3.71$.
Each study receives exactly one integer score (1--5) from a hierarchical
rule set over the $(D_1, D_2, D_3)$ triple (Score~5:
Causal\,+\,Quantified\,+\,Bounded; Score~4: Correlational\,+\,Quantified,
$D_3$ unrestricted; Score~3: Correlational\,+\,Directional; lower scores
cover Synthesis/$D_2$=None/Global combinations as specified in the Methods).
Score assignment was applied programmatically in Layer~2; the scoring
function is included in the publicly available pipeline scripts
(see Data Availability Statement).
```

## Why the distribution changed vs. the old n = 130 / $\bar{x}=3.20$ text

| | Prior draft | Current corpus |
|--|-------------|----------------|
| $N$ | 130 | **219** |
| Mean (SD) | 3.20 (0.87) | **3.71 (0.79)** |
| Score 1 / 2 / 3 / 4 / 5 | 4 / 18 / 63 / 37 / 8 | **7 / 22 / 1 / 187 / 2** |

Two drivers:

1. **Corpus expansion:** WoS remaining + Google Scholar extractions after
   title-based deduplication → 219 unique JSONs.
2. **Rubric clarification (FIX-I):** Correlational\,+\,Quantified maps to
   Score~4 **regardless of $D_3$**. Under the earlier wording that implied
   all three dimensions must jointly meet a level, many multi-country ILSA
   ML papers with quantified metrics were counted as Score~3; they are now
   correctly Score~4. This shifts mass from Score~3 ($63 \to 1$) to
   Score~4 ($37 \to 187$) and raises the mean.

**Also correct in any reply:** the old identity $416 \div 130 = 3.20$ was
arithmetically off ($1{\times}4+2{\times}18+3{\times}63+4{\times}37+5{\times}8 = 417$).

## Suggested reviewer response (EN)

> Thank you for requesting clarification of the actionability-score
> distribution. After expanding the extraction schema (`source_category`
> subtypes) and re-extracting the full corpus, we rescored **219** unique
> studies with the deterministic Layer-2 function in `score_studies_v2.py`
> (public pipeline). The updated descriptive statistics are:
> mean $= 3.71$ (SD $= 0.79$); Score 1 $n=7$, Score 2 $n=22$, Score 3 $n=1$,
> Score 4 $n=187$, Score 5 $n=2$
> ($\bar{x}=(1{\times}7+2{\times}22+3{\times}1+4{\times}187+5{\times}2)/219=812/219=3.71$).
>
> Relative to the earlier $N=130$ draft, two changes explain the shift:
> (i) additional WoS/Google Scholar articles after duplicate removal; and
> (ii) an explicit scoring rule that Correlational + Quantified evidence
> yields Score 4 irrespective of population boundedness ($D_3$), which is
> appropriate for multi-country ILSA designs that report clear performance
> metrics. Ground-truth agreement on the 30 manually coded studies remains
> 30/30 (Cohen’s $\kappa=1.00$). A second independent coder and
> Krippendorff’s $\alpha$ will be reported in the revision as noted.

## Önerilen hakem yanıtı (TR)

> Politika uygulanabilirliği skor dağılımına ilişkin netleştirme talebiniz
> için teşekkür ederiz. `source_category` alt türlerinin genişletilmesi ve
> tüm derlemenin yeniden extract edilmesinin ardından **219** benzersiz
> çalışma, herkese açık pipeline’daki deterministik Layer-2 fonksiyonu
> (`score_studies_v2.py`) ile yeniden skorlandı. Güncel betimleyici
> istatistikler: ortalama $= 3.71$ (SS $= 0.79$); Skor 1 $n=7$, Skor 2
> $n=22$, Skor 3 $n=1$, Skor 4 $n=187$, Skor 5 $n=2$
> ($\bar{x}=(1{\times}7+2{\times}22+3{\times}1+4{\times}187+5{\times}2)/219=812/219=3.71$).
>
> Önceki $N=130$ taslağına göre iki değişiklik dağılımı açıklıyor:
> (i) mükerrer ayıklamasından sonra ek WoS/Google Scholar makaleleri; ve
> (ii) Correlational + Quantified kanıtın $D_3$’ten bağımsız olarak Skor 4
> vermesi kuralının açık hale getirilmesi (çok ülkeli ILSA + metrik
> raporlayan çalışmalar için uygundur). 30 manuel kodlanmış çalışmada
> uzman–algoritma uyumu 30/30’dur (Cohen $\kappa=1.00$). Revizyonda ikinci
> bağımsız kodlayıcı ve Krippendorff $\alpha$ raporlanacaktır.

## Note on “222”

Filesystem inventory can show **222** `.json` paths if `json_test/` (3 files)
is counted with `json/` (219). Those three test copies are duplicates of
studies already in `json/` and must **not** enter $N$. Use **$N = 219$**.
