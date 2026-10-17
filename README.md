# Studi Kasus: Penerapan Machine Learning dari Materi "Introduction to Machine Learning"

Empat studi kasus Python yang mengikuti **alur kerja ML** dari seminar *Informatic Practitioner Lecture Series 2026* (Universitas Bengkulu, narasumber Dr. Taniza Tajuddin, UiTM Kedah). Setiap skrip mewakili satu jenis ML yang dibahas di materi.

> **Semua data bersifat sintetis** (dibangkitkan lewat kode) dan hanya untuk mendemonstrasikan alur kerja. Angka hasil **bukan temuan penelitian** dan tidak boleh dikutip sebagai bukti ilmiah.

## Isi Repositori

```
ml-intro-case-study/
├── README.md
├── requirements.txt
├── docs/
│   ├── 01_resume_materi.md        # resume materi seminar
│   ├── 02_insight_skripsi.md      # ide topik, gap, metodologi skripsi
│   └── 03_studi_kasus.md          # penjelasan studi kasus + hasil run
├── src/
│   ├── common.py                  # seed, folder output, helper
│   ├── 01_house_price_regression.py   # Supervised - Regresi
│   ├── 02_loan_classification.py      # Supervised - Klasifikasi
│   ├── 03_customer_segmentation.py    # Unsupervised - Clustering + PCA
│   ├── 04_anomaly_detection.py        # Unsupervised - Anomaly detection
│   └── run_all.py
└── outputs/                       # gambar hasil (dibuat saat skrip dijalankan)
```

## Cara Menjalankan

```bash
git clone <url-repo-anda>
cd ml-intro-case-study

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/run_all.py              # jalankan semua
# atau satu per satu:
python src/01_house_price_regression.py
```

Diuji dengan Python 3 dan scikit-learn 1.9.1. Tidak perlu mengunduh dataset; semua data dibuat di dalam kode dengan seed tetap (`SEED = 42`) sehingga hasil dapat direproduksi.

## Ringkasan Studi Kasus

| # | Jenis ML | Masalah | Model | Metrik |
|---|---|---|---|---|
| 1 | Supervised: regresi | Prediksi harga rumah | Baseline, Linear Regression, Random Forest | MAE, RMSE, R², CV |
| 2 | Supervised: klasifikasi | Persetujuan pinjaman | Baseline, Logistic Regression, Decision Tree, Random Forest | Accuracy, precision, recall, F1, confusion matrix |
| 3 | Unsupervised: clustering | Segmentasi pelanggan | K-Means, Hierarchical, PCA | Elbow, silhouette |
| 4 | Unsupervised: anomaly | Transaksi janggal | Isolation Forest | Precision, recall, PR-AUC |

Detail, hasil, dan interpretasi ada di [`docs/03_studi_kasus.md`](docs/03_studi_kasus.md).

## Praktik yang Diterapkan

- Preprocessing di dalam `Pipeline` sehingga tidak ada *data leakage*
- Selalu ada **baseline** pembanding
- Train/test split plus **cross-validation**
- Pengecekan **overfitting** (skor train vs test)
- Metrik yang sesuai kasus (anomali: bukan accuracy)

## Mengganti dengan Data Asli

Di tiap skrip, ganti fungsi `make_data()` dengan pembacaan CSV, misalnya:

```python
df = pd.read_csv("data/properti_bengkulu.csv")
```

lalu sesuaikan nama kolom di `build_pipeline` / daftar `features`.

## Sumber Materi

Seminar *Introduction to Machine Learning*, Informatic Practitioner Lecture Series 2026, Program Studi Informatika Universitas Bengkulu. Resume lengkap di `docs/01_resume_materi.md`.

## Lisensi

Tambahkan lisensi sesuai kebutuhan (mis. MIT) sebelum dipublikasikan.
