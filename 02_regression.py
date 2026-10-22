"""
02_regression.py  -  Next-day AQI regression.
Models: OLS, Ridge, Lasso, Polynomial(2)+Ridge, Decision Tree, kNN, SVR, MLP
Baseline: persistence (tomorrow = today)

Creates:
  results/table2_best_params_regression.csv
  results/table3_regression.csv
  results/lasso_coefficients.csv
  results/table5b_lockdown.csv
  results/tableA1_per_city.csv
  figures/fig6_observed_vs_predicted.png
  models/best_regressor.joblib
Run:  python 02_regression.py     (takes a few minutes: SVR/MLP are slow)
"""
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils import (CAT_FEATURES, FIG_DIR, MODEL_DIR, NUM_FEATURES,
                   RANDOM_STATE, RES_DIR, build_pipeline, load_data,
                   reg_metrics, split_data)


def main():
    df = load_data()
    train, test, lockdown = split_data(df)
    feats = NUM_FEATURES + CAT_FEATURES
    X_tr, y_tr = train[feats], train["AQI_next"]
    X_te, y_te = test[feats], test["AQI_next"]
    print(f"train={len(train)}  test={len(test)}  lockdown={len(lockdown)}")

    # ---- models and (small) grids -----------------------------------------
    # every parameter name starts with 'model__' because the estimator is the
    # last step of the pipeline.
    rs = RANDOM_STATE
    models = {
        "Linear reg. (OLS)": (build_pipeline(LinearRegression()), {}),
        "Ridge reg.": (build_pipeline(Ridge()),
                       {"model__alpha": [0.1, 1, 10, 100]}),
        "Lasso reg.": (build_pipeline(Lasso(max_iter=20000)),
                       {"model__alpha": [0.01, 0.1, 1, 5]}),
        "Polynomial (deg 2) + Ridge": (
            build_pipeline(Ridge(),
                           extra_steps=[("poly", PolynomialFeatures(2, include_bias=False))]),
            {"model__alpha": [10, 100, 1000]}),
        "Decision tree": (build_pipeline(DecisionTreeRegressor(random_state=rs)),
                          {"model__max_depth": [4, 6, 8],
                           "model__min_samples_leaf": [20, 50]}),
        "k-NN regressor": (build_pipeline(KNeighborsRegressor()),
                           {"model__n_neighbors": [5, 15, 31],
                            "model__weights": ["uniform", "distance"]}),
        "SVR (RBF)": (build_pipeline(SVR(kernel="rbf")),
                      {"model__C": [10, 100], "model__epsilon": [5, 15]}),
        "MLP (64-32)": (
            build_pipeline(MLPRegressor(hidden_layer_sizes=(64, 32),
                                        early_stopping=True, max_iter=500,
                                        random_state=rs)),
            {"model__alpha": [1e-4, 1e-2, 1]}),
    }

    cv = TimeSeriesSplit(n_splits=5)       # forward-chaining CV (train is date-sorted)
    rows, params_rows, fitted = [], [], {}

    # ---- persistence baseline ---------------------------------------------
    m = reg_metrics(y_te, test["AQI"])
    rows.append({"Model": "Persistence baseline", **m, "CV_RMSE": np.nan})

    for name, (pipe, grid) in models.items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error",
                          n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        pred = gs.predict(X_te)
        m = reg_metrics(y_te, pred)
        rows.append({"Model": name, **m, "CV_RMSE": -gs.best_score_})
        params_rows.append({"Model": name,
                            "Selected": {k.replace("model__", ""): v
                                         for k, v in gs.best_params_.items()}})
        fitted[name] = gs.best_estimator_
        print(f"{name:30s} RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}  "
              f"CV-RMSE={-gs.best_score_:.2f}  ({time.time() - t0:.0f}s)")

    table3 = pd.DataFrame(rows).round(3)
    table3.to_csv(RES_DIR / "table3_regression.csv", index=False)
    pd.DataFrame(params_rows).to_csv(RES_DIR / "table2_best_params_regression.csv",
                                     index=False)
    print("\n", table3.to_string(index=False))

    # ---- choose best model by CV-RMSE (more honest than picking on test) ---
    learned = table3[table3["Model"] != "Persistence baseline"]
    best_cv_name = learned.loc[learned["CV_RMSE"].idxmin(), "Model"]
    best_test_name = learned.loc[learned["RMSE"].idxmin(), "Model"]
    print(f"\nBest by CV-RMSE : {best_cv_name}")
    print(f"Best by test RMSE: {best_test_name}")
    best_name = best_test_name        # same as the sample paper; change to
    best = fitted[best_name]          # best_cv_name if you prefer
    joblib.dump(best, MODEL_DIR / "best_regressor.joblib")
    persist_rmse = table3.loc[table3["Model"] == "Persistence baseline", "RMSE"].iloc[0]
    best_rmse = table3.loc[table3["Model"] == best_name, "RMSE"].iloc[0]
    print(f"{best_name} is {100 * (1 - best_rmse / persist_rmse):.1f}% better "
          f"than persistence (RMSE)")

    # ---- Fig 6 : observed vs predicted ------------------------------------
    pred = best.predict(X_te)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_te, pred, s=4, alpha=0.3)
    plt.plot([0, 500], [0, 500], color="orange")
    plt.xlabel("Observed next-day AQI")
    plt.ylabel("Predicted next-day AQI")
    mm = reg_metrics(y_te, pred)
    plt.title(f"{best_name}\nRMSE={mm['RMSE']:.1f}, R2={mm['R2']:.3f}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_observed_vs_predicted.png", dpi=200)
    plt.close()

    # ---- Lasso coefficients (feature importance) --------------------------
    lasso = fitted["Lasso reg."]
    names = lasso.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names,
                          "coef": lasso.named_steps["model"].coef_})
    coefs["abs"] = coefs["coef"].abs()
    coefs = coefs.sort_values("abs", ascending=False).drop(columns="abs")
    coefs.to_csv(RES_DIR / "lasso_coefficients.csv", index=False)
    print("\nTop lasso coefficients:\n", coefs.head(6).to_string(index=False))
    print("Coefficients exactly zero:", int((coefs["coef"] == 0).sum()))

    # ---- Lockdown stress test ---------------------------------------------
    y_ld = lockdown["AQI_next"]
    ld = pd.DataFrame([
        {"Setting": "Lockdown: persistence", **reg_metrics(y_ld, lockdown["AQI"])},
        {"Setting": f"Lockdown: {best_name}",
         **reg_metrics(y_ld, best.predict(lockdown[feats]))},
    ]).round(3)
    ld.to_csv(RES_DIR / "table5b_lockdown.csv", index=False)
    print("\n", ld.to_string(index=False))

    # ---- Per-city errors ---------------------------------------------------
    out = []
    for city, g in test.groupby("City"):
        p = best.predict(g[feats])
        out.append({
            "City": city,
            "n_train": int((train["City"] == city).sum()),
            "n_test": len(g),
            "Mean_AQI": round(g["AQI_next"].mean()),
            "RMSE_model": round(reg_metrics(g["AQI_next"], p)["RMSE"], 1),
            "RMSE_persistence": round(reg_metrics(g["AQI_next"], g["AQI"])["RMSE"], 1),
        })
    pc = pd.DataFrame(out).sort_values("n_test", ascending=False)
    pc["worse_than_persistence"] = pc["RMSE_model"] > pc["RMSE_persistence"]
    pc.to_csv(RES_DIR / "tableA1_per_city.csv", index=False)
    print("\nModel beat persistence in",
          int((~pc["worse_than_persistence"]).sum()), "of", len(pc), "cities")
    print("Worse in:", list(pc.loc[pc["worse_than_persistence"], "City"]))
    print("\nDone.")


if __name__ == "__main__":      # this guard is REQUIRED on Windows for n_jobs=-1
    main()
