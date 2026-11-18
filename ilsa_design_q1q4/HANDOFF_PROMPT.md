# ILSA Design Quality Indicators — Devam Prompt'u

Türkçe yanıt ver. Kod yorumları İngilizce olabilir.

## Proje Özeti
Sistematik derleme makalesi: "Artificial Intelligence in Computer-Based International Large-Scale Assessments: A Systematic Review" — JCAL'a gönderilecek, çift-kör. 212 çalışma, 206'sı makalede (6 tanesi actionability score ile elenmiş — henüz hangileri belirlenmedi).

## Şu An Yapılan İş: Madde 13 / 36 revizyon listesinden
4 ILSA tasarım göstergesi (Q1–Q4) çıkarılıyor — LLM ile PDF'lerden, ardından manuel doğrulama.

### Göstergeler
- **Q1 plausible values**: met / not_met / not_reported / not_applicable
- **Q2 sampling weights**: met / not_met / not_reported / not_applicable  
- **Q3 school clustering**: met / not_met / not_reported
- **Q4 validation beyond estimation sample**: met (başka ülke/döngü/dış veri) / not_met (sadece iç CV/hold-out) / not_reported

## Dosya Konumları (Mac)
- **Script:** `~/ILSA-Survey-Analysis/ilsa_design_q1q4/ilsa_design_fields.py` (branch: `claude/blissful-bell-4a15v2`)
- **Kaynak JSON'lar (READ-ONLY):** `~/ILSA-Survey-Analysis/outputs/huggingface_n212/json_extracted/` (212 dosya)
- **Çıktı klasörü:** `~/ILSA-Survey-Analysis/ilsa_design_q1q4/design_out/`
- **PDF'ler:** `~/Desktop/quality_annotation/` (85 PDF) + `~/Downloads/` (62 PDF) + muhtemelen başka yerler
- **venv:** `~/ILSA-Survey-Analysis/ilsa_design_q1q4/.venv/` (openai, pymupdf, pydantic yüklü)
- **Excel (makalede kullanılan):** `~/ILSA-Survey-Analysis/outputs/huggingface_n212/ILSA_Survey_Dataset_CLEAN.xlsx` — henüz repo'da değil, `4_Actionability_Scores` sheet'i burada
- **trial_ids.txt:** `~/ILSA-Survey-Analysis/ilsa_design_q1q4/trial_ids.txt` (1, 12, 58, 104, 194)

## Script Özellikleri (ilsa_design_fields.py)
- Subcommands: `extract`, `review`, `apply`, `merge`, `summarise`
- `--pdf-dir` birden fazla klasör kabul eder: `--pdf-dir ~/Desktop/quality_annotation ~/Downloads ~/other`
- Model: `gpt-4o-mini` (default), `--model` ile değiştirilebilir
- Keyword-filtered paragraph extraction (≤12,000 chars), references truncated
- Pydantic structured output, temperature 0
- API key validation: startup'ta sk- prefix, ASCII-only kontrolü var
- Mevcut dosyaları atlar (`[SKIP existing]`) — yeniden çalıştırmak için `rm -f design_out/*`

## Tamamlanan Adımlar
1. ✅ Script yazıldı ve repo'ya push edildi
2. ✅ 212 JSON doğrulandı (unique id'ler 1–212)
3. ✅ gpt-4.1-nano → gpt-4o-mini değiştirildi  
4. ✅ API key validation eklendi (ç karakteri sorunu çözüldü)
5. ✅ UTF-8 auto-restart bloğu ve httpx2 monkeypatch temizlendi
6. ✅ Q4 prompt düzeltildi (internal CV = not_met, boş = not_reported)
7. ✅ Multi-dir PDF desteği eklendi
8. ✅ Deneme çalışması kısmen yapıldı: id 1, 12, 104 başarılı (3 API çağrısı)

## YAPILACAKLAR (sırayla)

### A. PDF Toplama (ÖNCELİKLİ)
212 çalışmanın PDF'leri dağınık. Yapılması gereken:
1. Tüm disk'te numaralı PDF'leri bul (`find ~ -maxdepth 7 -name "*.pdf" ... | grep "^[0-9]+"`)
2. Hepsini `~/ILSA-Survey-Analysis/ilsa_survey_articles/pdf/` klasörüne kopyala
3. JSON'larla eşleştirip kaç tanesinin eksik olduğunu raporla
4. Zotero'da da arayabilirsin: `~/Zotero/storage/*/` (her alt klasörde 1 PDF)

Bilinen eksikler:
- **id 58** (Tahri et al. 2026): `~/Desktop/quality_annotation/` ve `~/Downloads/`'da YOK
- **id 194** (Jung et al. 2026): Hiçbir yerde YOK — DOI: 10.1016/j.caeai.2026.100558

### B. 5 Çalışmalık Deneme (Adım 2-3)
1. `design_out/*` sil
2. Tüm PDF klasörlerini vererek 5 deneme çalıştır:
   ```
   python3 ilsa_design_q1q4/ilsa_design_fields.py extract \
     --pdf-dir ~/Desktop/quality_annotation ~/Downloads [DİĞER] \
     --json-dir outputs/huggingface_n212/json_extracted \
     --out ilsa_design_q1q4/design_out \
     --only ilsa_design_q1q4/trial_ids.txt
   ```
3. Sonuçları PDF'e karşı kontrol et. Özellikle:
   - **#58 Tahri 2026:** Q2=not_reported, Q4=not_met bekleniyor (sadece iç nested CV)
   - **#12 Gomez-Talal 2025:** PISA ML çalışması, muhtemelen train/test split var → Q4=not_met
   - **#1 Lee & Lee 2025:** methodology paper → Q1=not_applicable
4. Prompt/keyword sorunu varsa düzelt, KULLANICIYA GÖSTER, onay al

### C. Tam Çalıştırma (Adım 4) — KULLANICI ONAYI GEREKLİ
1. Onay al
2. `rm -f design_out/*`
3. Tüm 212 çalışma için extract çalıştır
4. `review` → `review.csv` oluştur
5. CONFLICT ve flagged satırları ayrı Excel sheet'te listele

### D. Doğrulama Sonrası (Adım 5)
1. Kullanıcı review.csv'yi dolduracak
2. `apply` → corrections uygula
3. `merge` → orijinal JSON'larla birleştir
4. `summarise` → özet tablo

### E. 212↔206 Farkı (Adım 6)
- `ILSA_Survey_Dataset_CLEAN.xlsx` → `4_Actionability_Scores` sheet'inden:
  - Score dağılımı: 1=7, 2=21, 3=1, 4=175, 5=2 → toplam 206
  - Kategori: 4.1=80, 4.2=45, 4.3=42, 4.4=10, 4.5=29
- Hangi 6 kaydın elendiğini bul → `ids206.txt`
- Belirlenemezse tahmin etme, raporla

### F. Son Rapor (Adım 7)
- `summarise --include ids206.txt` çıktısı
- Düzeltme sayısı/oranı (corrections_log)
- CONFLICT sayıları
- `uses_ilsa_microdata=true` olan n

## KESİN KURALLAR
1. `json_extracted/`, Excel ve PDF'ler YALNIZCA OKUNUR. Üzerine yazma. Tüm çıktılar `ilsa_design_q1q4/` içine.
2. Mevcut scriptin üzerine yazma. Değişiklik gerekirse önce göster ve onay al.
3. API çağrısı ücretli. Tam çalıştırmadan önce 5 çalışmalık deneme yap ve sonucu göster. Onay olmadan tam çalıştırma yapma.
4. Uydurma sayı yok. Makaleye girecek her sayı bir komut çıktısından gelmeli.
5. İkinci kodlayıcı YOK. ERIC ve Scopus ek araması YAPILMAYACAK. Bunları önerme.

## Git
- Branch: `claude/blissful-bell-4a15v2`
- Push: `git push -u origin claude/blissful-bell-4a15v2`
- Makaleyi (main.tex) şimdilik düzenleme

## Loglama (bu session'da işlem yapma)
- Bazı 4.2 "process data" çalışmaları gerçekte process data kullanmıyor olabilir
- `jhaUsingMachineLearning2026` key'i iki farklı çalışmaya eşlenmiş olabilir
- `InfluentialFactorsMathematical2023` ampirik ama review kategorisinde

## OpenAI API Key
- `read -s OPENAI_API_KEY && export OPENAI_API_KEY` ile gir
- Türkçe klavyeden ç/ü gibi karakterler yapışabilir — script başlangıçta kontrol eder
- ÖNCEKİ ANAHTAR AÇIĞA ÇIKTI — kullanıcı platform.openai.com'dan rotasyona almalı

## Beklenen Çıktılar (makale için)
- Yeni tablo `tab:ilsa-quality`: Met / Not met / NR / N/A × Q1–Q4
- RQ2 tartışma cümlesi: dört yüzde
- Methods cümlesi: "[k] of [4×n] field values (x%) were corrected on manual verification"
- Limitations'taki yanlış cümle düzeltilecek (extraction template PV/weight alanlarını İÇERİYORDU, sadece yanlış kodlanmıştı)
