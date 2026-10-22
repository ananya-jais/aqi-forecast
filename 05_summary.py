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
    ("TABLE 4: HEALTH-ALERT CLASSIFICATION (test set)",  "table4_classification.csv"),
    ("CONFUSION MATRIX (best F1 model)",        "confusion_matrix_best.csv"),
    ("TABLE 5: PCA vs ALL FEATURES",            "table5_pca.csv"),
    ("PCA SUMMARY",                             "pca_summary.csv"),
    ("TABLE 5b: COVID LOCKDOWN STRESS TEST",    "table5b_lockdown.csv"),
    ("TOP 8 LASSO COEFFICIENTS",                "lasso_coefficients.csv"),
    ("TABLE A1: PER-CITY RMSE (model vs persistence)",   "tableA1_per_city.csv"),
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
    b = learned.loc[learned.RMSE.idxmin()]
    c = t4[t4.Model != "Persistence baseline"]
    cb = c.loc[c.F1.idxmax()]
    cbase = t4.loc[t4.Model == "Persistence baseline", "F1"].iloc[0]
    lines += ["", "=" * 78, "KEY TAKEAWAYS", "=" * 78,
              f"Best regressor (lowest test RMSE): {b.Model}  RMSE={b.RMSE:.2f}, R2={b.R2:.3f}",
              f"  -> {100 * (1 - b.RMSE / base):.1f}% lower RMSE than persistence ({base:.2f})",
              f"Best classifier (highest F1): {cb.Model}  F1={cb.F1:.3f}, "
              f"Recall={cb.Recall:.3f}, AUC={cb.AUC:.3f}  (persistence F1={cbase:.3f})"]
except Exception as e:
    lines.append(f"(could not build takeaways: {e})")

text = "\n".join(lines)
print(text)
(RES_DIR / "summary.txt").write_text(text, encoding="utf-8")
print("\nSaved to results/summary.txt")