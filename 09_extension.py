"""
09_extension.py  -  YOUR extension beyond the basic study.

Two questions, answered with a 2 x 2 experiment:
  (a) MODEL:    does gradient boosting (HistGradientBoostingRegressor) beat the
                final model Polynomial(2) + Ridge?
  (b) FEATURES: do extra features help?  Extra = PM2.5 and PM10 of the previous
                day (lagged pollutants), 7-day mean AQI, and a season indicator.

                         base features        extended features
  Polynomial + Ridge        A                      C
  Gradient boosting         B                      D

Rules (same as the rest of the project):
  * models are compared by forward-chaining CV (mean +/- sd), NOT on the test set
  * the test set is looked at once, for reporting
  * for every variant we give a 95% block-bootstrap interval (weekly blocks) for
    the RMSE difference to A. If the interval contains 0, the change is NOT
    a reliable improvement - that is a perfectly good (negative) result.

Creates: results/table11_extension.csv
Run:     python 09_extension.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures

from utils import (CAT_FEATURES, EXTRA_CAT_FEATURES, EXTRA_NUM_FEATURES,
                   NUM_FEATURES, RANDOM_STATE, RES_DIR, build_pipeline,
                   load_data, reg_metrics, split_data)

N_BOOT = 2000


def block_bootstrap_diff(test, pred_a, pred_x):
    """95% CI for RMSE(A) - RMSE(X), resampling calendar weeks. >0: X is better."""
    y = test["AQI_next"].values
    iso = test["Date"].dt.isocalendar()
    block = (iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)).values
    ids, inv = np.unique(block, return_inverse=True)
    n = np.bincount(inv).astype(float)
    sa = np.bincount(inv, weights=(y - pred_a) ** 2)
    sx = np.bincount(inv, weights=(y - pred_x) ** 2)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    d = np.sqrt(sa[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sx[idx].sum(1) / n[idx].sum(1))
    point = np.sqrt(sa.sum() / n.sum()) - np.sqrt(sx.sum() / n.sum())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return point, lo, hi


def main():
    df = load_data()
    train, test, _ = split_data(df)
    y_tr, y_te = train["AQI_next"], test["AQI_next"]

    base_num, base_cat = NUM_FEATURES, CAT_FEATURES
    ext_num = NUM_FEATURES + EXTRA_NUM_FEATURES
    ext_cat = CAT_FEATURES + EXTRA_CAT_FEATURES

    def poly_ridge(num, cat):
        return (build_pipeline(Ridge(), num_cols=num, cat_cols=cat,
                               extra_steps=[("poly", PolynomialFeatures(2, include_bias=False))]),
                {"model__alpha": [10, 100, 1000]})

    def hgb(num, cat):
        return (build_pipeline(HistGradientBoostingRegressor(
                    early_stopping=False, random_state=RANDOM_STATE),
                    num_cols=num, cat_cols=cat),
                {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5],
                 "model__max_iter": [100, 300]})

    experiments = {
        "A: Poly+Ridge, base features (final model)": (poly_ridge(base_num, base_cat), base_num + base_cat),
        "B: Gradient boosting, base features": (hgb(base_num, base_cat), base_num + base_cat),
        "C: Poly+Ridge, extended features": (poly_ridge(ext_num, ext_cat), ext_num + ext_cat),
        "D: Gradient boosting, extended features": (hgb(ext_num, ext_cat), ext_num + ext_cat),
    }

    cv = TimeSeriesSplit(n_splits=5)
    rows, preds = [], {}
    for name, ((pipe, grid), cols) in experiments.items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error",
                          n_jobs=-1)
        gs.fit(train[cols], y_tr)
        p = gs.predict(test[cols])
        preds[name] = p
        m = reg_metrics(y_te, p)
        rows.append({"Variant": name,
                     "CV_RMSE": -gs.best_score_,
                     "CV_RMSE_sd": gs.cv_results_["std_test_score"][gs.best_index_],
                     "Test_MAE": m["MAE"], "Test_RMSE": m["RMSE"], "Test_R2": m["R2"],
                     "Selected": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}})
        print(f"{name:46s} CV-RMSE={-gs.best_score_:.2f} +/- "
              f"{rows[-1]['CV_RMSE_sd']:.2f}   test RMSE={m['RMSE']:.2f}  "
              f"({time.time() - t0:.0f}s)")

    tab = pd.DataFrame(rows)

    # bootstrap difference to A (positive = variant better than A)
    names = list(experiments)
    ref = preds[names[0]]
    diffs = []
    for n in names:
        if n == names[0]:
            diffs.append((np.nan, np.nan, np.nan))
        else:
            diffs.append(block_bootstrap_diff(test, ref, preds[n]))
    tab["RMSE_gain_vs_A"] = [d[0] for d in diffs]
    tab["gain_CI95_low"] = [d[1] for d in diffs]
    tab["gain_CI95_high"] = [d[2] for d in diffs]
    tab = tab.round(3)
    tab.to_csv(RES_DIR / "table11_extension.csv", index=False)

    print("\n", tab.drop(columns="Selected").to_string(index=False))
    best_cv = tab.loc[tab["CV_RMSE"].idxmin(), "Variant"]
    print(f"\nBest variant by CV-RMSE: {best_cv}")
    print("RMSE_gain_vs_A > 0 means the variant has a LOWER test RMSE than A.")
    for _, r in tab.iloc[1:].iterrows():
        if r["gain_CI95_low"] > 0:
            verdict = "reliable improvement (interval above 0)"
        elif r["gain_CI95_high"] < 0:
            verdict = "reliably WORSE (interval below 0)"
        else:
            verdict = "no reliable difference (interval contains 0)"
        print(f"  {r['Variant'][:2]} vs A: gain {r['RMSE_gain_vs_A']:+.2f} "
              f"[{r['gain_CI95_low']:+.2f}, {r['gain_CI95_high']:+.2f}]  -> {verdict}")
    print("\nDone.")


if __name__ == "__main__":
    main()