# Studi Kasus dan Implementasi

Setiap studi kasus mengikuti workflow materi: **data → preprocessing → split → pilih model → training → evaluasi → prediksi/interpretasi**. Angka di bawah adalah hasil run nyata (`SEED=42`, scikit-learn 1.9.1) pada **data sintetis**.

---

## Studi Kasus 1: Regresi, Prediksi Harga Rumah
**File:** `src/01_house_price_regression.py` | **Materi:** supervised learning, regresi (harga rumah)

**Data:** 1.000 rumah; fitur luas (m²), kamar tidur, kamar mandi, lokasi (pusat kota/pinggiran/desa); target harga (juta rupiah). 3% nilai luas sengaja dikosongkan (32 baris) untuk melatih penanganan *missing value*.

**Preprocessing (dalam Pipeline):** imputasi median, `StandardScaler` untuk numerik, `OneHotEncoder` untuk lokasi.

**Hasil (data uji 20%):**

| Model | MAE | RMSE | R² test | R² train | R² CV 5-fold |
|---|---|---|---|---|---|
| Baseline (rata-rata) | 230,3 | 277,4 | -0,001 | 0,000 | -0,006 ± 0,005 |
| Linear Regression | 47,1 | 60,0 | 0,953 | 0,952 | 0,950 ± 0,009 |
| Random Forest | 54,5 | 71,4 | 0,934 | 0,989 | 0,931 ± 0,011 |

**Interpretasi:**
- Baseline R² ≈ 0 adalah acuan: model yang berguna harus jauh di atasnya.
- **Linear Regression menang** karena data sintetis ini dibangkitkan dengan rumus linear. Ini mengingatkan pesan materi: *algoritma harus cocok dengan karakter data*; model yang lebih kompleks tidak otomatis lebih baik.
- Random Forest menunjukkan gejala overfitting ringan (R² train 0,989 vs test 0,934).
- Prediksi contoh: rumah 120 m², 3 KT, 2 KM di pusat kota ≈ Rp 1.261 juta (hanya ilustrasi, bukan harga pasar).

**Output gambar:** `outputs/01_regresi_aktual_vs_prediksi.png`

---

## Studi Kasus 2: Klasifikasi, Persetujuan Pinjaman
**File:** `src/02_loan_classification.py` | **Materi:** klasifikasi biner (loan approval), metrik, overfitting

**Data:** 1.500 pemohon; fitur pendapatan, credit score, rasio utang, status pekerjaan; target disetujui (58,2%) / ditolak (41,8%). Split 80:20 berstrata.

**Hasil:**

| Model | Acc train | Acc test | Precision | Recall | F1 | F1 CV |
|---|---|---|---|---|---|---|
| Baseline (mayoritas) | 0,582 | 0,583 | 0,583 | 1,000 | 0,737 | 0,736 ± 0,002 |
| Logistic Regression | 0,863 | 0,887 | 0,917 | 0,886 | 0,901 | 0,878 ± 0,022 |
| Decision Tree (tanpa batas) | **1,000** | 0,793 | 0,834 | 0,806 | 0,820 | 0,813 ± 0,028 |
| Decision Tree (depth=4) | 0,832 | 0,793 | 0,884 | 0,743 | 0,807 | 0,810 ± 0,010 |
| Random Forest | 1,000 | 0,883 | 0,907 | 0,891 | 0,899 | 0,869 ± 0,022 |

**Interpretasi:**
- **Overfitting nyata:** Decision Tree tanpa batas kedalaman mencapai akurasi train 1,0 tetapi hanya 0,793 di data uji. Persis peringatan materi: akurasi terlalu sempurna saat training patut dicurigai.
- Baseline "selalu setuju" sudah memberi akurasi ~58% dan recall 1,0, jadi **accuracy dan recall saja tidak cukup**; lihat precision dan F1 sekaligus.
- Logistic Regression dan Random Forest setara (F1 ≈ 0,90); pilih yang lebih sederhana dan mudah dijelaskan bila performa sama (relevan untuk explainability di sektor keuangan).
- Skrip juga menampilkan *feature importance* untuk interpretabilitas dasar.

**Output gambar:** `outputs/02_klasifikasi_confusion_matrix.png`, `outputs/02_klasifikasi_feature_importance.png`

---

## Studi Kasus 3: Clustering, Segmentasi Pelanggan
**File:** `src/03_customer_segmentation.py` | **Materi:** unsupervised learning, K-Means, hierarchical, PCA, customer segmentation

**Data:** 450 pelanggan dengan 4 fitur (pendapatan, spending score, kunjungan/bulan, rata-rata belanja). Terdapat 3 kelompok tersembunyi; label asli **tidak dipakai** saat clustering.

**Langkah:** standardisasi (wajib untuk metode berbasis jarak) → pilih K via elbow dan silhouette → K-Means → pembanding Hierarchical → PCA 2D → profil cluster.

**Hasil:**
- Silhouette tertinggi pada **K = 3** (0,613); K = 2 hampir sama (0,611), sehingga elbow perlu dilihat bersama silhouette.
- K-Means dan Hierarchical sama-sama silhouette 0,613.
- PCA 2 komponen menjelaskan 94,7% variansi.

| Cluster | Jumlah | Pendapatan (juta) | Spending score | Kunjungan/bln | Rata-rata belanja (ribu) | Tafsiran |
|---|---|---|---|---|---|---|
| 0 | 150 | 88,7 | 79,7 | 9,1 | 221,2 | Premium |
| 1 | 150 | 34,8 | 29,7 | 2,1 | 39,7 | Hemat |
| 2 | 150 | 59,6 | 55,2 | 4,8 | 107,7 | Menengah/loyal |

**Interpretasi:** nilai bisnis clustering ada pada **profil dan tindakan** (mis. promo loyalitas untuk premium, diskon untuk segmen hemat), bukan sekadar skor silhouette. Kecocokan sempurna dengan kelompok asli (ARI = 1,0) terjadi karena kelompok sintetis sengaja dibuat terpisah jelas; data nyata hampir tidak pernah serapi ini.

**Output gambar:** `outputs/03_clustering_pemilihan_k.png`, `outputs/03_clustering_pca.png`

---

## Studi Kasus 4: Anomaly Detection, Transaksi Janggal
**File:** `src/04_anomaly_detection.py` | **Materi:** anomaly detection, deteksi fraud kartu kredit

**Data:** 5.060 transaksi, hanya 60 (1,19%) yang janggal (nominal besar, dini hari, luar negeri, beruntun). **Isolation Forest dilatih tanpa label**; label hanya untuk evaluasi.

**Hasil:**
- Menebak "semua normal" menghasilkan accuracy 98,81%, **menyesatkan**, sehingga yang dilaporkan adalah precision/recall/PR-AUC.
- Confusion matrix: 4.999 normal benar, 1 normal salah ditandai, 0 anomali terlewat, 60 anomali tertangkap.
- Precision anomali 0,984; recall 1,000; PR-AUC 1,000 (baseline acak 0,012).

**Peringatan penting (sejalan dengan materi):** hasil ini **terlalu sempurna** karena pola anomali sintetis dibuat sangat berbeda dari data normal. Di dunia nyata, anomali tumpang tindih dengan data normal, label langka, dan pola fraud berubah. Jika Anda mendapat angka seperti ini pada data asli, **curigai data leakage** dan periksa ulang datanya, seperti pesan narasumber.

**Output gambar:** `outputs/04_anomali_pr_curve_dan_skor.png`

---

## Pelajaran Lintas Studi Kasus

1. **Selalu pakai baseline** (kasus 1 dan 2).
2. **Bandingkan skor train vs test** untuk mendeteksi overfitting (kasus 1 dan 2).
3. **Pilih metrik sesuai masalah**, accuracy menipu pada data tidak seimbang (kasus 4).
4. **Cocokkan algoritma dengan data**, bukan dengan tren (kasus 1).
5. **Interpretasi hasil** lebih penting daripada angka (kasus 3).
6. **Hasil terlalu sempurna = tanda bahaya** (kasus 3 dan 4).

## Ide Pengembangan Lanjutan

- Ganti data sintetis dengan dataset asli (Kaggle, data terbuka pemerintah, data hasil survei).
- Tambahkan hyperparameter tuning (`GridSearchCV`/`RandomizedSearchCV`).
- Tambahkan SHAP atau permutation importance untuk interpretabilitas.
- Studi kasus 2: tangani ketidakseimbangan kelas dan lakukan **fairness check** antar kelompok.
- Studi kasus 4: bandingkan dengan One-Class SVM dan Autoencoder.
- Tambahkan semi-supervised (self-training) dan reinforcement learning (Q-Learning grid world) agar mencakup seluruh jenis ML di materi.
