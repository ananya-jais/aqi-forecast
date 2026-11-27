"""
08_error_analysis.py  -  WHERE and WHY does the final regressor fail?

An overall RMSE hides local problems. This script breaks the test-set error of
the final regressor (chosen by CV in 02_regression.py) down by:
  1. SEASON            (winter / summer / monsoon / post-monsoon)
  2. POLLUTION LEVEL   (today's AQI category - known when the forecast is made)
  3. CITY              (data volume, region, how volatile the city's AQI is)
Every row compares the model with persistence (tomorrow = today).
'Bias' = mean(predicted - observed); negative means under-prediction.

Needs:   python 02_regression.py
Creates: results/table9_error_by_season.csv
         results/table10_error_by_aqi_level.csv
         results/tableA2_city_diagnostics.csv
         figures/fig9_error_analysis.png
Run:     python 08_error_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FIG_DIR, RES_DIR

# Broad geographic / climate region of each city (approximate, for discussion)
REGION = {
    "Ahmedabad": "West (semi-arid)", "Amaravati": "South-East (tropical)",
    "Amritsar": "North (Indo-Gangetic plain)", "Bengaluru": "South (plateau)",
    "Bhopal": "Central", "Brajrajnagar": "East (industrial, Odisha)",
    "Chandigarh": "North", "Chennai": "South-East coast",
    "Coimbatore": "South (tropical)", "Delhi": "North (Indo-Gangetic plain)",
    "Ernakulam": "South (Kerala coast)", "Gurugram": "North (Indo-Gangetic plain)",
    "Guwahati": "North-East", "Hyderabad": "South (plateau)",
    "Jaipur": "North-West (semi-arid)", "Jorapokhar": "East (industrial, Jharkhand)",
    "Kochi": "South (Kerala coast)", "Kolkata": "East (humid)",
    "Lucknow": "North (Indo-Gangetic plain)", "Mumbai": "West coast",
    "Patna": "East (Indo-Gangetic plain)", "Shillong": "North-East (hills)",
    "Talcher": "East (industrial, Odisha)",
    "Thiruvananthapuram": "South (Kerala coast)",
    "Visakhapatnam": "South-East coast",
}


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def summarise(df, by):
    rows = []
    for key, g in df.groupby(by, observed=True):
        rows.append({by: key, "n": len(g),
                     "mean_AQI_tomorrow": round(g["AQI_next"].mean(), 1),
                     "RMSE_model": round(rmse(g["AQI_next"], g["pred_model"]), 2),
                     "RMSE_persistence": round(rmse(g["AQI_next"], g["AQI_today"]), 2),
                     "bias_model": round((g["pred_model"] - g["AQI_next"]).mean(), 2),
                     "bias_persistence": round((g["AQI_today"] - g["AQI_next"]).mean(), 2)})
    out = pd.DataFrame(rows)
    out["model_better"] = out["RMSE_model"] < out["RMSE_persistence"]
    return out


def main():
    final = (RES_DIR / "selected_model.txt").read_text().strip()
    pred = pd.read_csv(RES_DIR / "test_predictions.csv", parse_dates=["Date"])
    print(f"Final regressor: {final}  |  test rows: {len(pred)}")

    # ---- 1. by season -------------------------------------------------------
    m = pred["Date"].dt.month
    pred["season"] = np.select(
        [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8, 9])],
        ["winter (Dec-Feb)", "summer (Mar-May)", "monsoon (Jun-Sep)"],
        default="post-monsoon (Oct-Nov)")
    by_season = summarise(pred, "season")
    by_season.to_csv(RES_DIR / "table9_error_by_season.csv", index=False)
    print("\nERROR BY SEASON\n", by_season.to_string(index=False))

    # ---- 2. by pollution level (today's AQI category) ------------------------
    labels = ["Good (0-50)", "Satisfactory (51-100)", "Moderate (101-200)",
              "Poor (201-300)", "Very Poor (301-400)", "Severe (401-500)"]
    pred["level_today"] = pd.cut(pred["AQI_today"],
                                 [-np.inf, 50, 100, 200, 300, 400, np.inf],
                                 labels=labels)
    by_level = summarise(pred, "level_today")
    by_level.to_csv(RES_DIR / "table10_error_by_aqi_level.csv", index=False)
    print("\nERROR BY TODAY'S AQI LEVEL\n", by_level.to_string(index=False))

    # ---- 3. by city: why does the model lose in some cities? -----------------
    pc = pd.read_csv(RES_DIR / "tableA1_per_city.csv")
    pc["region"] = pc["City"].map(REGION)
    pc["change_vs_persistence_%"] = (100 * (pc["RMSE_model"] - pc["RMSE_persistence"])
                                     / pc["RMSE_persistence"]).round(1)
    pc["training_data"] = pd.cut(pc["n_train"], [-1, 0, 300, np.inf],
                                 labels=["none (unseen city)", "small (1-300 days)",
                                         "larger (>300 days)"])
    pc = pc.sort_values("change_vs_persistence_%", ascending=False)
    pc.to_csv(RES_DIR / "tableA2_city_diagnostics.csv", index=False)

    print("\nCITIES WHERE THE MODEL IS WORSE THAN PERSISTENCE")
    print(pc[pc["worse_than_persistence"]][
        ["City", "region", "n_train", "RMSE_model", "RMSE_persistence",
         "change_vs_persistence_%"]].to_string(index=False))

    grp = (pc.groupby("training_data", observed=True)["worse_than_persistence"]
           .agg(cities="size", worse="sum").reset_index())
    print("\nWORSE-THAN-PERSISTENCE BY AMOUNT OF TRAINING DATA\n",
          grp.to_string(index=False))

    rho = pc["RMSE_persistence"].corr(pc["change_vs_persistence_%"], method="spearman")
    print(f"\nSpearman correlation between persistence RMSE (how volatile the "
          f"city is) and the model's change vs persistence: {rho:.2f}")
    print("  (negative = the MORE volatile the city, the MORE the model gains; "
          "very stable cities are hard to beat)")

    # ---- figure ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, tab, key, title in [(ax[0], by_season, "season", "By season"),
                               (ax[1], by_level, "level_today", "By today's AQI level")]:
        x = np.arange(len(tab))
        a.bar(x - 0.2, tab["RMSE_persistence"], 0.4, label="persistence", color="grey")
        a.bar(x + 0.2, tab["RMSE_model"], 0.4, label=final)
        a.set_xticks(x)
        a.set_xticklabels([str(v).split(" (")[0] for v in tab[key]], rotation=30,
                          ha="right", fontsize=8)
        a.set_ylabel("Test RMSE")
        a.set_title(title)
        a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_error_analysis.png", dpi=200)
    plt.close()
    print("\nDone.")


if __name__ == "__main__":
    main()