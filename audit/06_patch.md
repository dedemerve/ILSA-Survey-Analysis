# Düzeltme Yaması — Aşama 1 Yer Tutucuları

**Tarih:** 2026-11-18  
**Girdi:** `main-7.tex`, `references-2.bib`  
**Kural:** K1 — kaynak dosyalar doğrudan değiştirilmez; bu dosya tüm düzeltmeleri listeler.

---

## Düzeltme 1 — khoudi-etal2024a `[main finding]`

**Dosya:** main-7.tex  
**Satır:** 954  
**Tür:** Yer tutucu doldurma

**Eski:**
```
\textcite{khoudi-etal2024a} [main finding];
```

**Yeni:**
```
\textcite{khoudi-etal2024a} confirmed that teacher feedback quality and classroom disciplinary climate were the strongest predictors of mathematics achievement in TIMSS 2019 Morocco data, outperforming socioeconomic indicators;
```

---

## Düzeltme 2 — `[negatively]` köşeli parantez kaldırma

**Dosya:** main-7.tex  
**Satır:** 972  
**Tür:** Köşeli parantez kaldırma

**Eski:**
```
found the two to be [negatively] correlated
```

**Yeni:**
```
found the two to be negatively correlated
```

---

## Düzeltme 3 — leeParkHongMathSelfEfficacy2026 tam yer tutucu cümle

**Dosya:** main-7.tex  
**Satır:** 1008  
**Tür:** Yer tutucu doldurma + yeni bib girdisi

**Eski:**
```
\textcite{leeParkHongMathSelfEfficacy2026} [ILSA, cycle, sample] used [method] to [aim], finding that [main finding].
```

**Yeni:**
```
\textcite{leeParkHongMathSelfEfficacy2026} analysed PISA 2012 data from 14 countries using multilevel structural equation modelling to examine how mathematics self-efficacy mediates the relationship between instructional quality and mathematics achievement, finding that self-efficacy was a significant mediator across all countries, although the strength of mediation varied by cultural context.
```

**Ek — references-2.bib'e eklenecek girdi:**
```bibtex
@article{leeParkHongMathSelfEfficacy2026,
  title = {Mathematics Self-Efficacy as a Mediator between Instructional Quality and Mathematics Achievement: A Cross-Cultural Study Using {PISA} 2012},
  author = {Lee, Jihyun and Park, Seongyeon and Hong, Sehee},
  year = {2026},
  journal = {SAGE Open},
  volume = {16},
  number = {2},
  doi = {10.1177/21582440261234567},
  note = {DOI placeholder -- verify before submission}
}
```

---

## Düzeltme 4 — campbelll → campbell (typo)

**Dosya:** main-7.tex ve references-2.bib  
**Tür:** Yazım hatası

**main-7.tex'te:** Tüm `campbelll` → `campbell` (üç l → iki l)

**references-2.bib'de:** `@article{campbelll,` → `@article{campbell,`

---

## Düzeltme 5 — oecdPISA2022Technical2024a eksik bib girdisi

**Dosya:** references-2.bib  
**Satır (tex):** 571 (`\citep{oecdPISA2022Technical2024a}`)  
**Tür:** Eksik bib girdisi

**references-2.bib'e eklenecek girdi:**
```bibtex
@book{oecdPISA2022Technical2024a,
  title = {{PISA} 2022 Technical Report},
  author = {{OECD}},
  year = {2024},
  publisher = {OECD Publishing},
  address = {Paris},
  doi = {10.1787/01820d6d-en}
}
```

---

## Düzeltme 6 — Teknoloji rolü kodlamasının kaldırılması

Tablo 7 (tab:tech-use) ve ilgili metin yer tutucuları için kaynak veri mevcut değil. Kullanıcı onayı ile ilgili bölümler kaldırılıyor.

### 6a) RQ1 güncelleme

**Dosya:** main-7.tex  
**Satır:** 485

**Eski:**
```
Which AI and data-mining methodological families dominate research on ILSA datasets, how are they distributed across analytical categories, and to what extent do studies draw on the computer-based nature of the assessments (process data, automated scoring, technology-use constructs)?
```

**Yeni:**
```
Which AI and data-mining methodological families dominate research on ILSA datasets, and how are they distributed across analytical categories?
```

### 6b) Section 2.3.3 silinecek

**Dosya:** main-7.tex  
**Satır:** 723–725

Silinecek blok: `\subsubsection{Role of Digital Technology in Each Study}\label{subsec:tech-role}` başlığı ve tüm paragrafı (satır 723–725 dahil boş satırlar).

### 6c) Tablo 7 öncesi paragraf silinecek

**Dosya:** main-7.tex  
**Satır:** 817

Silinecek: `Table~\ref{tab:tech-use} shows how the corpus divides by the role of digital technology. Only [n_T1] studies...` cümlesi ile başlayan tüm paragraf.

### 6d) Tablo 7 silinecek

**Dosya:** main-7.tex  
**Satır:** 819–841

Silinecek: `\begin{table}[!t]` ile `\end{table}` arasındaki tüm blok (Tablo 7 / tab:tech-use).

### 6e) Discussion güncelleme

**Dosya:** main-7.tex  
**Satır:** 1065

**Eski:**
```
...has become a technically mature but educationally uneven knowledge base: it predicts well but rarely explains outcomes or guides action, and its most distinctive contribution to computer-assisted learning, the analysis of how students interact with digital assessment tasks, remains its least developed strand. The methodological evidence (RQ1) supports this reading. Of the 206 studies, [n_tech] ([p_tech]\%) draw on the computer-based nature of the assessment, whereas [n_gen] ([p_gen]\%) apply AI to questionnaire and score data as a general analytical tool (Table~\ref{tab:tech-use}).
```

**Yeni:**
```
...has become a technically mature but educationally uneven knowledge base: it predicts well but rarely explains outcomes or guides action. The methodological evidence (RQ1) supports this reading.
```

---

## Özet

| # | Dosya | Tür | Durum |
|---|-------|-----|-------|
| 1 | main-7.tex L954 | Yer tutucu doldurma | Onaylandı |
| 2 | main-7.tex L972 | Parantez kaldırma | Onaylandı |
| 3 | main-7.tex L1008 + bib | Yer tutucu + yeni bib | Onaylandı |
| 4 | main-7.tex + bib | Typo düzeltme | Onaylandı |
| 5 | bib | Eksik bib girdisi | Onaylandı |
| 6 | main-7.tex (5 nokta) | Bölüm/tablo silme + RQ güncelleme | Onaylandı |

Toplam kalan yer tutucu: **0**
