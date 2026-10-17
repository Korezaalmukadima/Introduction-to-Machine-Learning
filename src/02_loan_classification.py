"""Studi Kasus 2: SUPERVISED LEARNING - KLASIFIKASI (persetujuan pinjaman).

Fitur: pendapatan, riwayat kredit, rasio utang, status pekerjaan -> approved (1) / rejected (0).
Menampilkan metrik yang dibahas di materi: accuracy, precision, recall, F1, confusion matrix,
serta pengecekan overfitting (train vs test).

Data SINTETIS, hanya untuk demonstrasi alur kerja.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import SEED, header, save_fig


def make_data(n: int = 1500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=2.3, sigma=0.5, size=n)         # juta/bulan
    credit_score = rng.normal(650, 80, n).clip(300, 850)
    debt_ratio = rng.beta(2, 5, n)                              # 0..1
    employment = rng.choice(["tetap", "kontrak", "wiraswasta", "menganggur"], n,
                            p=[0.45, 0.25, 0.22, 0.08])
    emp_bonus = {"tetap": 0.8, "kontrak": 0.3, "wiraswasta": 0.2, "menganggur": -1.5}
    z = (0.25 * (income - 10) + 0.012 * (credit_score - 650) - 4.0 * (debt_ratio - 0.28)
         + np.array([emp_bonus[e] for e in employment]) + rng.normal(0, 0.8, n))
    approved = (z > 0).astype(int)
    return pd.DataFrame({"income_juta": income, "credit_score": credit_score,
                         "debt_ratio": debt_ratio, "employment": employment,
                         "approved": approved})


def make_pipe(model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), ["income_juta", "credit_score", "debt_ratio"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["employment"]),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    header("STUDI KASUS 2 - KLASIFIKASI: PERSETUJUAN PINJAMAN")
    df = make_data()
    print(f"Distribusi kelas (1=disetujui):\n{df['approved'].value_counts(normalize=True).round(3)}\n")

    X, y = df.drop(columns="approved"), df["approved"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Baseline (kelas mayoritas)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree (tanpa batas)": DecisionTreeClassifier(random_state=SEED),
        "Decision Tree (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, m in models.items():
        pipe = make_pipe(m).fit(X_tr, y_tr)
        fitted[name] = pipe
        p = pipe.predict(X_te)
        cv_f1 = cross_val_score(make_pipe(m), X_tr, y_tr, cv=cv, scoring="f1")
        rows.append({
            "Model": name,
            "Acc_train": accuracy_score(y_tr, pipe.predict(X_tr)),
            "Acc_test": accuracy_score(y_te, p),
            "Precision": precision_score(y_te, p, zero_division=0),
            "Recall": recall_score(y_te, p, zero_division=0),
            "F1": f1_score(y_te, p, zero_division=0),
            "F1_CV": f"{cv_f1.mean():.3f}+/-{cv_f1.std():.3f}",
        })
    res = pd.DataFrame(rows).round(3)
    print(res.to_string(index=False))
    print("\nPerhatikan 'Decision Tree (tanpa batas)': akurasi train ~1.0 tetapi test lebih rendah"
          "\n-> contoh nyata OVERFITTING seperti yang diingatkan di materi.")

    print("\nLaporan klasifikasi Random Forest (data uji):")
    print(classification_report(y_te, fitted["Random Forest"].predict(X_te),
                                target_names=["ditolak", "disetujui"]))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(fitted["Random Forest"], X_te, y_te,
                                          display_labels=["ditolak", "disetujui"], ax=ax,
                                          colorbar=False)
    ax.set_title("Confusion matrix - Random Forest")
    save_fig(fig, "02_klasifikasi_confusion_matrix.png")

    # Feature importance (interpretabilitas dasar)
    rf = fitted["Random Forest"]
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax)
    ax.set_title("Feature importance - Random Forest")
    save_fig(fig, "02_klasifikasi_feature_importance.png")


if __name__ == "__main__":
    main()
