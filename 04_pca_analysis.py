"""
04_pca_analysis.py  -  Does PCA help?
Creates:
  figures/fig4_pca_variance.png
  results/table5_pca.csv
  results/pca_summary.csv
Run:  python 04_pca_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (CAT_FEATURES, FIG_DIR, NUM_FEATURES, RES_DIR, Winsorizer,
                   build_pipeline, load_data, reg_metrics, split_data)


def main():
    df = load_data()
    train, test, lockdown = split_data(df)
    feats = NUM_FEATURES + CAT_FEATURES

    # ---- PCA explained variance on the numeric features (training data) ---
    prep = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("win", Winsorizer()),
                     ("sc", StandardScaler())])
    Z = prep.fit_transform(train[NUM_FEATURES])
    pca = PCA(random_state=42).fit(Z)
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    n90 = int(np.argmax(cum >= 0.90) + 1)
    n95 = int(np.argmax(cum >= 0.95) + 1)
    summary = pd.DataFrame({
        "n_numeric_features": [len(NUM_FEATURES)],
        "components_for_90%": [n90],
        "components_for_95%": [n95],
        "first_component_%": [round(100 * ratio[0], 1)],
    })
    summary.to_csv(RES_DIR / "pca_summary.csv", index=False)
    print(summary.to_string(index=False))

    x = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, ratio * 100, label="Individual")
    ax.plot(x, cum * 100, "o-", color="darkorange", label="Cumulative")
    ax.axhline(95, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_pca_variance.png", dpi=200)
    plt.close()

    # ---- Table 5 : all features vs PCA 95% --------------------------------
    X_tr, y_tr = train[feats], train["AQI_next"]
    X_te, y_te = test[feats], test["AQI_next"]
    setups = {
        "OLS, all features": build_pipeline(LinearRegression()),
        "OLS, PCA 95%": build_pipeline(LinearRegression(), use_pca=True),
        "k-NN (k=15), all features": build_pipeline(KNeighborsRegressor(15)),
        "k-NN (k=15), PCA 95%": build_pipeline(KNeighborsRegressor(15), use_pca=True),
    }
    rows = []
    for name, pipe in setups.items():
        pipe.fit(X_tr, y_tr)
        rows.append({"Setting": name, **reg_metrics(y_te, pipe.predict(X_te))})
    t5 = pd.DataFrame(rows).round(3)
    t5.to_csv(RES_DIR / "table5_pca.csv", index=False)
    print("\n", t5.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
