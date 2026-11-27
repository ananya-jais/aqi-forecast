"""
06_uncertainty.py  -  How sure are we about the numbers?

A single RMSE (e.g. 35.26) is a point estimate. This script adds:
  1. CV-RMSE as  mean +/- sd  across the 5 forward-chaining folds
  2. A 95% block-bootstrap interval for the test RMSE of the final model
  3. A 95% block-bootstrap interval for the DIFFERENCE between persistence
     and the final model (is the improvement real or just luck?)

Why a BLOCK bootstrap? Neighbouring days (and different cities on the same
day) are strongly correlated, so resampling single rows would give intervals
that are too narrow. We resample whole calendar WEEKS (all cities together).

Needs:   python 02_regression.py   (creates test_predictions.csv)
Creates: results/table3_cv_mean_sd.csv
         results/table6_bootstrap_ci.csv
Run:     python 06_uncertainty.py
"""
import numpy as np
import pandas as pd

from utils import RANDOM_STATE, RES_DIR

N_BOOT = 2000


def main():
    # ---- 1. CV mean +/- sd ---------------------------------------------------
    t3 = pd.read_csv(RES_DIR / "table3_regression.csv")
    cv = t3[t3["Model"] != "Persistence baseline"][["Model", "CV_RMSE", "CV_RMSE_sd"]].copy()
    cv["CV_RMSE (mean +/- sd)"] = (cv["CV_RMSE"].round(2).astype(str) + " +/- "
                                   + cv["CV_RMSE_sd"].round(2).astype(str))
    cv.to_csv(RES_DIR / "table3_cv_mean_sd.csv", index=False)
    print("CV-RMSE across the 5 folds:\n", cv[["Model", "CV_RMSE (mean +/- sd)"]]
          .to_string(index=False))

    # ---- 2. block bootstrap on the test set ----------------------------------
    pred = pd.read_csv(RES_DIR / "test_predictions.csv", parse_dates=["Date"])
    final = (RES_DIR / "selected_model.txt").read_text().strip()

    y = pred["AQI_next"]
    se_model = (y - pred["pred_model"]) ** 2          # squared errors, final model
    se_pers = (y - pred["AQI_today"]) ** 2            # squared errors, persistence

    iso = pred["Date"].dt.isocalendar()
    block = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    g = (pd.DataFrame({"block": block, "m": se_model, "p": se_pers})
         .groupby("block").agg(m=("m", "sum"), p=("p", "sum"), n=("m", "size")))
    M, P, N = g["m"].values, g["p"].values, g["n"].values
    n_blocks = len(g)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, n_blocks, size=(N_BOOT, n_blocks))   # resample weeks
    rmse_m = np.sqrt(M[idx].sum(axis=1) / N[idx].sum(axis=1))
    rmse_p = np.sqrt(P[idx].sum(axis=1) / N[idx].sum(axis=1))
    diff = rmse_p - rmse_m                      # > 0 means the model is better
    rel = 100 * (1 - rmse_m / rmse_p)           # % reduction in RMSE

    point_m = np.sqrt(M.sum() / N.sum())
    point_p = np.sqrt(P.sum() / N.sum())

    def ci(a):
        return np.percentile(a, [2.5, 97.5])

    rows = [
        (f"RMSE: {final}", point_m, *ci(rmse_m)),
        ("RMSE: persistence", point_p, *ci(rmse_p)),
        ("RMSE difference (persistence - model)", point_p - point_m, *ci(diff)),
        ("RMSE reduction vs persistence (%)", 100 * (1 - point_m / point_p), *ci(rel)),
    ]
    out = pd.DataFrame(rows, columns=["Quantity", "Estimate", "CI95_low", "CI95_high"]).round(2)
    out.to_csv(RES_DIR / "table6_bootstrap_ci.csv", index=False)

    print(f"\nBlock bootstrap: {N_BOOT} resamples of {n_blocks} calendar weeks "
          f"({len(pred)} test rows)")
    print(out.to_string(index=False))
    print(f"\nShare of resamples in which the model beats persistence: "
          f"{(diff > 0).mean():.1%}")
    print("Note: these intervals reflect test-sample variability only; they do "
          "not include uncertainty from training or model selection.")


if __name__ == "__main__":
    main()
