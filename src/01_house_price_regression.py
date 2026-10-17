"""Studi Kasus 1: SUPERVISED LEARNING - REGRESI (prediksi harga rumah).

Mengikuti workflow di materi:
  data -> preprocessing -> split train/test -> pilih model -> train -> evaluasi -> prediksi

Catatan: data SINTETIS (dibangkitkan dengan rumus + noise) hanya untuk demonstrasi.
Ganti `make_data()` dengan pd.read_csv(...) untuk data asli.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import SEED, header, save_fig


def make_data(n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Data rumah sintetis: luas, kamar tidur, kamar mandi, lokasi -> harga (juta rupiah)."""
    rng = np.random.default_rng(seed)
    size = rng.normal(100, 35, n).clip(30, 300)  # m2
    bedrooms = rng.integers(1, 6, n)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, None)
    location = rng.choice(["pusat_kota", "pinggiran", "desa"], n, p=[0.3, 0.45, 0.25])
    loc_premium = {"pusat_kota": 600, "pinggiran": 250, "desa": 0}
    price = (
        4 * size
        + 40 * bedrooms
        + 30 * bathrooms
        + np.array([loc_premium[x] for x in location])
        + rng.normal(0, 60, n)  # noise
    )
    df = pd.DataFrame(
        {"size_m2": size, "bedrooms": bedrooms, "bathrooms": bathrooms,
         "location": location, "price_juta": price}
    )
    # Sisipkan sedikit missing value agar langkah preprocessing bermakna
    mask = rng.random(n) < 0.03
    df.loc[mask, "size_m2"] = np.nan
    return df


def build_pipeline(model) -> Pipeline:
    numeric = ["size_m2", "bedrooms", "bathrooms"]
    categorical = ["location"]
    # Imputasi + scaling dilakukan DI DALAM pipeline -> tidak ada data leakage
    from sklearn.impute import SimpleImputer

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    header("STUDI KASUS 1 - REGRESI: PREDIKSI HARGA RUMAH")
    df = make_data()
    print(f"Jumlah data: {len(df)} | missing size_m2: {df['size_m2'].isna().sum()}")
    print(df.head(), "\n")

    X, y = df.drop(columns="price_juta"), df["price_juta"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Split 80:20 -> train={len(X_tr)}, test={len(X_te)}")

    models = {
        "Baseline (rata-rata)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED),
    }

    rows, fitted = [], {}
    for name, m in models.items():
        pipe = build_pipeline(m).fit(X_tr, y_tr)
        fitted[name] = pipe
        pred_te = pipe.predict(X_te)
        cv_r2 = cross_val_score(build_pipeline(m), X_tr, y_tr, cv=5, scoring="r2")
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_te, pred_te),
            "RMSE": np.sqrt(mean_squared_error(y_te, pred_te)),
            "R2_test": r2_score(y_te, pred_te),
            "R2_train": r2_score(y_tr, pipe.predict(X_tr)),
            "R2_CV(5-fold)": f"{cv_r2.mean():.3f} +/- {cv_r2.std():.3f}",
        })
    res = pd.DataFrame(rows).round(3)
    print("\nHasil evaluasi (satuan harga: juta rupiah):")
    print(res.to_string(index=False))
    print("\nCek overfitting: bandingkan R2_train vs R2_test. Selisih besar = curiga overfitting.")

    best = fitted["Linear Regression"]
    new_house = pd.DataFrame([{"size_m2": 120, "bedrooms": 3, "bathrooms": 2, "location": "pusat_kota"}])
    print(f"\nPrediksi rumah baru (120 m2, 3 KT, 2 KM, pusat kota): "
          f"Rp {best.predict(new_house)[0]:,.0f} juta")

    # Visualisasi aktual vs prediksi
    fig, ax = plt.subplots(figsize=(5.5, 5))
    pred = fitted["Random Forest"].predict(X_te)
    ax.scatter(y_te, pred, s=14, alpha=0.6)
    lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    ax.plot(lims, lims, "r--", label="prediksi sempurna")
    ax.set_xlabel("Harga aktual (juta)")
    ax.set_ylabel("Harga prediksi (juta)")
    ax.set_title("Random Forest: aktual vs prediksi")
    ax.legend()
    save_fig(fig, "01_regresi_aktual_vs_prediksi.png")


if __name__ == "__main__":
    main()
