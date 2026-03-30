# Deteksi Anomali Pasar Saham Berbasis Hybrid Time Series Decomposition dan Machine Learning dengan Interpretasi SHAP

**Data Analysis Competition — Matematika Fair 2026**  
**Tema:** Peran Matematika Kritis dalam Membangun Inovasi dan Solusi di Era Digital

## Deskripsi

Model keuangan klasik umumnya mengasumsikan bahwa return saham berdistribusi normal dan bergerak secara independen antar waktu (*random walk*). Dalam praktiknya, asumsi ini jarang terpenuhi. Return saham menunjukkan distribusi berekor tebal (*fat-tails*), volatilitas yang mengelompok (*volatility clustering*), dan autokorelasi yang mengindikasikan adanya memori pasar. Kegagalan asumsi ini menyebabkan metode deteksi anomali berbasis model parametrik tradisional (misalnya Z-score) menjadi tidak reliabel.

Proyek ini membangun **Early Warning System** untuk mendeteksi anomali pada pasar saham Indonesia dengan studi kasus saham **BBCA** (Bank Central Asia). Pendekatan yang digunakan bersifat hybrid menggabungkan dekomposisi time series klasik dengan algoritma machine learning modern:

1. **Validasi asumsi secara empiris.** Uji Kolmogorov-Smirnov dan Durbin-Watson digunakan untuk membuktikan secara statistik bahwa return BBCA melanggar asumsi normalitas dan independensi. Langkah ini menjadi dasar argumentasi mengapa pendekatan nonparametrik diperlukan.

2. **Ekstraksi sinyal laten via STL Decomposition.** Harga saham didekomposisi menjadi komponen tren, musiman, dan residual. Komponen residual merepresentasikan deviasi harga dari pola historis normal dan menjadi sinyal utama deteksi anomali.

3. **Deteksi anomali nonparametrik via Isolation Forest.** Algoritma ini bekerja berdasarkan prinsip bahwa anomali adalah observasi yang mudah diisolasi dalam ruang fitur. Tiga fitur digunakan: residual STL, rolling volatility 30 hari, dan volume ratio. Parameter `contamination=0.02` mengasumsikan sekitar 2% hari perdagangan merupakan hari anomali.

4. **Interpretasi transparan via SHAP.** Setiap prediksi anomali disertai penjelasan kuantitatif tentang kontribusi masing-masing fitur, menjawab pertanyaan "mengapa alarm ini berbunyi?"
   
Seluruh tahapan analisis disertai justifikasi matematis yang merujuk pada filosofi **Matematika Kritis**: tidak sekadar menerima asumsi model klasik, melainkan menguji, mendekomposisi, dan menginterpretasi data secara kritis.

## Metodologi

```
Data Saham Historis (29 emiten BEI, 2004–2024)
       │
       ▼
┌──────────────────┐
│  Preprocessing   │  ← Filter BBCA, hitung log-return: r_t = ln(P_t / P_{t-1})
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Uji Asumsi      │  ← KS Test (normalitas) + Durbin-Watson (autokorelasi)
│  Statistik       │    Konfirmasi fat-tails & memory effect
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  STL             │  ← P_t = T_t + S_t + R_t
│  Decomposition   │    period=252 (1 tahun trading), robust=True
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Feature         │  ← X = [Residual, Rolling_Vol_30, Volume_Ratio]
│  Engineering     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Isolation       │  ← n_estimators=300, contamination=0.02
│  Forest          │    Anomaly score: s(x,n) = 2^{-E[h(x)]/c(n)}
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  SHAP            │  ← Permutation-based explainer
│  Explainability  │    Summary plot + Waterfall per anomali
└──────────────────┘
```

## Struktur Repositori

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── DAC_Anomaly_Detection_BBCA.ipynb    # Notebook analisis utama
└── Dataset_DAC_.csv                     # Dataset (tidak di-track, lihat .gitignore)
```

## Quick Start

```bash
# Clone repo
git clone https://github.com/<username>/DAC-Anomaly-Detection-BBCA.git
cd DAC-Anomaly-Detection-BBCA

# Install dependencies
pip install -r requirements.txt

# Jalankan notebook
jupyter notebook DAC_Anomaly_Detection_BBCA.ipynb
```

Atau buka langsung di Google Colab:

[![Open In Colab]([https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/DAC-Anomaly-Detection-BBCA/blob/main/DAC_Anomaly_Detection_BBCA.ipynb](https://colab.research.google.com/drive/1TEi539vJladgyxegjs1_CmXQ-7tIKo2w?usp=sharing))



## Dataset

Dataset berisi data saham historis **29 emiten** di Bursa Efek Indonesia (BEI) periode 2004–2024, dengan total 136.599 records. Kolom yang tersedia: `Stock_Name`, `Date`, `Open`, `High`, `Low`, `Close`, `Volume`. Delimiter file CSV menggunakan titik koma (`;`).

Proyek ini menggunakan subset **BBCA** (4.803 records) sebagai prototipe model. Dataset disediakan oleh panitia DAC Matematika Fair 2026 dan **tidak disertakan dalam repositori ini**. Letakkan file `Dataset_DAC_.csv` di root directory sebelum menjalankan notebook.

## Tech Stack

| Kategori | Library | Fungsi |
|---|---|---|
| Data Processing | `pandas`, `numpy` | Manipulasi dataframe, komputasi numerik |
| Statistical Testing | `scipy.stats`, `statsmodels` | KS test, Durbin-Watson, deskriptif |
| Time Series | `statsmodels.tsa.seasonal.STL` | Dekomposisi tren-musiman-residual |
| Machine Learning | `scikit-learn` (Isolation Forest) | Deteksi anomali nonparametrik |
| Explainability | `shap` | Interpretasi prediksi berbasis Shapley value |
| Visualization | `matplotlib`, `seaborn` | Plotting dan visual analisis |

## Hasil

- **Uji KS** mengkonfirmasi return BBCA tidak berdistribusi normal (fat-tails dan excess kurtosis), memvalidasi kebutuhan pendekatan nonparametrik
- **Durbin-Watson** menunjukkan struktur autokorelasi pada return, mengindikasikan adanya memory effect di pasar
- **Isolation Forest** mendeteksi ~2% hari perdagangan sebagai anomali, berkorespondensi dengan periode krisis pasar yang diketahui
- **SHAP** mengidentifikasi kontribusi relatif Residual, Volatilitas, dan Volume terhadap setiap alarm anomali secara transparan

## Lisensi

Proyek ini dibuat untuk keperluan kompetisi akademik (DAC Matematika Fair 2026, Universitas Negeri Medan).
