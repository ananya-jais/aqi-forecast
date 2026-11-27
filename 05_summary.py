"""
05_summary.py  -  Prints all results in one clean, readable summary
and saves it to results/summary.txt.
Run it AFTER scripts 01-04:   python 05_summary.py
"""
import pandas as pd
from utils import RES_DIR

SECTIONS = [
    ("DATA SPLIT (chronological)",              "split_summary.csv"),
    ("MISSING VALUES (%) - Xylene dropped",     "table1_missing_values.csv"),
    ("TABLE 2a: BEST HYPER-PARAMETERS (REGRESSION)",     "table2_best_params_regression.csv"),
    ("TABLE 2b: BEST HYPER-PARAMETERS (CLASSIFICATION)", "table2_best_params_classification.csv"),
    ("TABLE 3: NEXT-DAY AQI REGRESSION (test set)",      "table3_regression.csv"),
    ("TABLE 3b: CV-RMSE MEAN +/- SD ACROSS 5 FOLDS",     "table3_cv_mean_sd.csv"),
    ("TABLE 6: BOOTSTRAP 95% INTERVALS (test set)",      "table6_bootstrap_ci.csv"),
    ("TABLE 4: HEALTH-ALERT CLASSIFICATION (test set)",  "table4_classification.csv"),
    ("CONFUSION MATRIX (final classifier, chosen by CV-F1)", "confusion_matrix_best.csv"),
    ("TABLE 7: ALERT THRESHOLDS (test set)",    "table7_threshold_sweep.csv"),
    ("TABLE 8: CALIBRATION BINS (test set)",    "table8_calibration_bins.csv"),
    ("TABLE 9: ERROR BY SEASON",                "table9_error_by_season.csv"),
    ("TABLE 10: ERROR BY TODAY'S AQI LEVEL",    "table10_error_by_aqi_level.csv"),
    ("TABLE 11: EXTENSION (gradient boosting, extra features)", "table11_extension.csv"),
    ("TABLE 5: PCA vs ALL FEATURES",            "table5_pca.csv"),
    ("PCA SUMMARY",                             "pca_summary.csv"),
    ("TABLE 5b: COVID LOCKDOWN STRESS TEST",    "table5b_lockdown.csv"),
    ("TOP 8 LASSO COEFFICIENTS",                "lasso_coefficients.csv"),
    ("TABLE A1: PER-CITY RMSE (model vs persistence)",   "tableA1_per_city.csv"),
    ("TABLE A2: CITY DIAGNOSTICS (region, training data, change vs persistence)", "tableA2_city_diagnostics.csv"),
]

lines = []
for title, fname in SECTIONS:
    path = RES_DIR / fname
    lines += ["", "=" * 78, title, "=" * 78]
    if not path.exists():
        lines.append(f"(missing: run the script that creates {fname})")
        continue
    df = pd.read_csv(path)
    if fname == "table1_missing_values.csv":
        df.columns = ["column", "missing_percent"]
    if fname == "lasso_coefficients.csv":
        df = df.head(8)
    lines.append(df.to_string(index=False, max_colwidth=60))

# short plain-English takeaways
try:
    t3 = pd.read_csv(RES_DIR / "table3_regression.csv")
    t4 = pd.read_csv(RES_DIR / "table4_classification.csv")
    base = t3.loc[t3.Model == "Persistence baseline", "RMSE"].iloc[0]
    learned = t3[t3.Model != "Persistence baseline"]
    sel = (RES_DIR / "selected_model.txt").read_text().strip()   # chosen by CV
    b = learned[learned.Model == sel].iloc[0]
    c = t4[t4.Model != "Persistence baseline"]
    selc = (RES_DIR / "selected_classifier.txt").read_text().strip()   # chosen by CV-F1
    cb = c[c.Model == selc].iloc[0]
    cbase = t4.loc[t4.Model == "Persistence baseline", "F1"].iloc[0]
    lines += ["", "=" * 78, "KEY TAKEAWAYS", "=" * 78,
              f"Final regressor (chosen by CV-RMSE): {b.Model}  test RMSE={b.RMSE:.2f}, R2={b.R2:.3f}",
              f"  -> {100 * (1 - b.RMSE / base):.1f}% lower RMSE than persistence ({base:.2f})",
              f"Final classifier (chosen by CV-F1): {cb.Model}  F1={cb.F1:.3f}, "
              f"Recall={cb.Recall:.3f}, AUC={cb.AUC:.3f}  (persistence F1={cbase:.3f})"]
except Exception as e:
    lines.append(f"(could not build takeaways: {e})")

text = "\n".join(lines)
print(text)
(RES_DIR / "summary.txt").write_text(text, encoding="utf-8")
print("\nSaved to results/summary.txt")