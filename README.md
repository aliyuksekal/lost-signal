# Lost Signal 📻

İngilizce 3D animasyon dizisi **Lost Signal** (kanal: Nightdial Media) için üretim hattı.
Karakterler, sahne, seslendirme ve kurgu kodla üretilir; ağır render Kaggle'ın ücretsiz GPU'sunda yapılır.

## Nasıl çalışır?
1. **Actions → "1 - Render başlat (Kaggle GPU)" → Run workflow**: ne render edileceğini seç.
2. Kaggle GPU'da render başlar (fragman ~1-2 saat, bölüm daha uzun).
3. **"2 - Sonucu al"** 30 dakikada bir kontrol eder; video bitince **Releases** bölümüne koyar.
4. Telefondan Releases → ilgili sürüm → Assets → `.mp4` dosyasını indir.

## Bir kerelik ayar
- `Settings → Secrets and variables → Actions → Secrets`: **KAGGLE_API_TOKEN** = Kaggle API token
- `Settings → Secrets and variables → Actions → Variables`: **KAGGLE_USERNAME** = Kaggle kullanıcı adı

## Klasörler
- `studio/` – karakterler (`chars.py`, `chars2.py`), sahne ve animasyon (`scene.py`), ses miksajı (`mix.py`), video birleştirme (`finish.py`)
- `kaggle/` – Kaggle'da çalışan render işi
- `.github/workflows/` – otomasyon
