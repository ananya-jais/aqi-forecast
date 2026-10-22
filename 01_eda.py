"""
01_eda.py  -  Exploratory data analysis.
Creates:
  figures/fig1_missing_values.png
  figures/fig2_monthly_aqi.png
  figures/fig3_correlation.png
  results/table1_missing_values.csv
  results/split_summary.csv
Run:  python 01_eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import (DATA_FILE, FIG_DIR, RES_DIR, POLLUTANTS,
                   load_data, split_data)

raw = pd.read_csv(DATA_FILE, parse_dates=["Date"])
print("Raw shape:", raw.shape, "| cities:", raw["City"].nunique())
print("Date range:", raw["Date"].min().date(), "to", raw["Date"].max().date())

# ---- Table 1 / Fig 1 : missing values --------------------------------------
missing = (raw.isna().mean() * 100).round(1).sort_values(ascending=False)
missing.rename("missing_percent").to_csv(RES_DIR / "table1_missing_values.csv")
print("\nMissing % per column:\n", missing)

m = missing.drop(["City", "Date"], errors="ignore")
m = m[m.index != "AQI_Bucket"].sort_values()
colors = ["red" if v > 50 else "#1f3b6f" for v in m.values]
plt.figure(figsize=(6, 5))
plt.barh(m.index, m.values, color=colors)
plt.axvline(50, color="grey", linestyle="--")
plt.xlabel("Missing values (%)")
plt.title("Missing values per variable (red: >50%, dropped)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_missing_values.png", dpi=200)
plt.close()

# ---- Fig 2 : mean AQI by month for six cities -------------------------------
raw["month"] = raw["Date"].dt.month
cities = ["Delhi", "Ahmedabad", "Bengaluru", "Lucknow", "Mumbai", "Chennai"]
plt.figure(figsize=(8, 4.5))
for c in cities:
    s = raw[raw["City"] == c].assign(AQI=lambda d: d["AQI"].clip(upper=500)) \
            .groupby("month")["AQI"].mean()   # AQI capped at 500
    plt.plot(s.index, s.values, marker="o", label=c)
plt.axhline(200, color="grey", linestyle=":", label="AQI = 200 (Poor)")
plt.xticks(range(1, 13), list("JFMAMJJASOND"))
plt.xlabel("Month")
plt.ylabel("Mean daily AQI")
plt.legend(ncol=3, fontsize=8)
plt.title("Mean daily AQI by month (2015-2020)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_monthly_aqi.png", dpi=200)
plt.close()

# ---- Fig 3 : correlation heatmap (training years only) ----------------------
df = load_data()
train, test, lockdown = split_data(df)
cols = POLLUTANTS + ["AQI", "AQI_next"]
corr = train[cols].corr()          # pairwise-complete by default
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            annot_kws={"size": 6})
plt.title("Pearson correlation (training years)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_correlation.png", dpi=200)
plt.close()
print("\nTop correlations with next-day AQI:")
print(corr["AQI_next"].drop("AQI_next").sort_values(ascending=False).round(2).head(6))

# ---- Split summary ----------------------------------------------------------
summary = pd.DataFrame({
    "set": ["train (2015-2018)", "test (2019-Feb 2020)", "lockdown (25 Mar-30 Jun 2020)"],
    "samples": [len(train), len(test), len(lockdown)],
    "alert_rate_%": [round(100 * d["alert"].mean(), 1) for d in (train, test, lockdown)],
})
summary.to_csv(RES_DIR / "split_summary.csv", index=False)
print("\n", summary)
print("\nAQI > 500 records capped (before capping, in raw data):",
      int((raw["AQI"] > 500).sum()))
print("Done. Check the figures/ and results/ folders.")
