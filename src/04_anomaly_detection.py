"""Studi Kasus 4: ANOMALY DETECTION (transaksi kartu kredit janggal).

Terinspirasi cerita di materi: transaksi luar negeri yang tidak biasa membuat kartu diblokir bank.
Isolation Forest dilatih TANPA label; label hanya dipakai untuk mengevaluasi.
Karena data sangat tidak seimbang, yang dilaporkan adalah precision/recall/PR-AUC, bukan accuracy.

Data SINTETIS, hanya untuk demonstrasi.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             precision_recall_curve)
from sklearn.preprocessing import StandardScaler

from common import SEED, header, save_fig


def make_data(n_normal: int = 5000, n_fraud: int = 60, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame({
        "amount_ribu": rng.lognormal(4.0, 0.7, n_normal),             # sekitar 55 rb
        "hour": np.clip(rng.normal(14, 4, n_normal), 0, 23),
        "is_overseas": rng.random(n_normal) < 0.02,
        "tx_last_24h": rng.poisson(2, n_normal),
        "is_anomaly": 0,
    })
    fraud = pd.DataFrame({
        "amount_ribu": rng.lognormal(6.2, 0.6, n_fraud),              # jauh lebih besar
        "hour": rng.choice([1, 2, 3, 4, 23], n_fraud),                # dini hari
        "is_overseas": rng.random(n_fraud) < 0.85,
        "tx_last_24h": rng.poisson(8, n_fraud),                       # beruntun
        "is_anomaly": 1,
    })
    df = pd.concat([normal, fraud], ignore_index=True)
    df["is_overseas"] = df["is_overseas"].astype(int)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    header("STUDI KASUS 4 - ANOMALY DETECTION: TRANSAKSI JANGGAL")
    df = make_data()
    features = ["amount_ribu", "hour", "is_overseas", "tx_last_24h"]
    y = df["is_anomaly"].to_numpy()
    print(f"Total {len(df)} transaksi | anomali: {y.sum()} ({y.mean():.2%}) -> sangat tidak seimbang")
    print(f"Jika menebak 'semua normal', accuracy = {(1 - y.mean()):.2%} (menyesatkan!)\n")

    X = StandardScaler().fit_transform(df[features])
    contamination = 0.012  # perkiraan proporsi anomali; di praktik harus diperkirakan/ditune
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=SEED).fit(X)
    pred = (iso.predict(X) == -1).astype(int)
    score = -iso.score_samples(X)  # makin besar = makin janggal

    print("Confusion matrix [baris=aktual, kolom=prediksi] (0=normal, 1=anomali):")
    print(confusion_matrix(y, pred))
    print("\n" + classification_report(y, pred, target_names=["normal", "anomali"], digits=3))
    print(f"PR-AUC (average precision) = {average_precision_score(y, score):.3f} "
          f"(baseline acak = {y.mean():.3f})")

    # Trade-off: ubah ambang -> recall vs precision
    prec, rec, _ = precision_recall_curve(y, score)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(rec, prec); axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall curve")
    axes[1].hist(score[y == 0], bins=50, alpha=0.7, label="normal", density=True)
    axes[1].hist(score[y == 1], bins=30, alpha=0.7, label="anomali", density=True)
    axes[1].set_xlabel("anomaly score"); axes[1].legend(); axes[1].set_title("Distribusi skor")
    save_fig(fig, "04_anomali_pr_curve_dan_skor.png")

    print("\nCatatan: menaikkan recall (menangkap lebih banyak fraud) biasanya menurunkan precision"
          "\n(lebih banyak transaksi normal ikut diblokir). Pilih ambang sesuai biaya kesalahan.")


if __name__ == "__main__":
    main()
