"""Studi Kasus 3: UNSUPERVISED LEARNING - CLUSTERING + PCA (segmentasi pelanggan).

Langkah: data tanpa label -> scaling -> pilih K (elbow + silhouette) -> K-Means ->
bandingkan Hierarchical -> PCA untuk visualisasi 2D -> profil tiap cluster.

Data SINTETIS dengan 3 kelompok pelanggan tersembunyi.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import SEED, header, save_fig


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = [  # (n, income, spend_score, visits_per_month, avg_basket)
        (150, 35, 30, 2, 40),    # hemat
        (150, 90, 80, 9, 220),   # premium
        (150, 60, 55, 5, 110),   # menengah / loyal
    ]
    frames = []
    for gid, (n, inc, sp, vis, bsk) in enumerate(groups):
        frames.append(pd.DataFrame({
            "income_juta": rng.normal(inc, inc * 0.12, n),
            "spending_score": rng.normal(sp, 8, n).clip(1, 100),
            "visits_per_month": rng.normal(vis, 1.2, n).clip(0.5, None),
            "avg_basket_ribu": rng.normal(bsk, bsk * 0.15, n),
            "_true_group": gid,  # hanya untuk validasi demo; TIDAK dipakai saat clustering
        }))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    header("STUDI KASUS 3 - CLUSTERING: SEGMENTASI PELANGGAN")
    df = make_data()
    features = ["income_juta", "spending_score", "visits_per_month", "avg_basket_ribu"]
    X = StandardScaler().fit_transform(df[features])  # K-Means berbasis jarak -> wajib diskalakan

    # 1) Pilih K: elbow (inertia) dan silhouette
    ks = range(2, 9)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
    best_k = list(ks)[int(np.argmax(sil))]
    print("K  | inertia | silhouette")
    for k, i, s in zip(ks, inertia, sil):
        print(f"{k}  | {i:7.1f} | {s:.3f}{'  <- terbaik' if k == best_k else ''}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(list(ks), inertia, "o-"); axes[0].set_title("Elbow (inertia)"); axes[0].set_xlabel("K")
    axes[1].plot(list(ks), sil, "o-", color="green"); axes[1].set_title("Silhouette"); axes[1].set_xlabel("K")
    save_fig(fig, "03_clustering_pemilihan_k.png")

    # 2) K-Means final dan Hierarchical sebagai pembanding
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    hc = AgglomerativeClustering(n_clusters=best_k).fit(X)
    df["cluster_kmeans"] = km.labels_
    print(f"\nK terpilih = {best_k}")
    print(f"Silhouette K-Means      : {silhouette_score(X, km.labels_):.3f}")
    print(f"Silhouette Hierarchical : {silhouette_score(X, hc.labels_):.3f}")
    print(f"Kecocokan dg kelompok asli (ARI, hanya validasi demo): "
          f"{adjusted_rand_score(df['_true_group'], km.labels_):.3f}")

    # 3) Profil cluster = interpretasi bisnis (bagian terpenting)
    profile = df.groupby("cluster_kmeans")[features].mean().round(1)
    profile.insert(0, "jumlah", df["cluster_kmeans"].value_counts().sort_index())
    print("\nProfil rata-rata per cluster:")
    print(profile.to_string())

    # 4) PCA 2D untuk visualisasi
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)
    print(f"\nPCA: 2 komponen menjelaskan {pca.explained_variance_ratio_.sum():.1%} variansi")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="viridis", s=14)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"K-Means (K={best_k}) pada ruang PCA")
    ax.legend(*sc.legend_elements(), title="cluster")
    save_fig(fig, "03_clustering_pca.png")


if __name__ == "__main__":
    main()
