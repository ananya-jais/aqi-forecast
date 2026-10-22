"""
utils.py  -  shared helpers used by every script.

Contains:
  load_data()      build target + features from city_day.csv
  split_data()     chronological train / test / lockdown split
  build_pipeline() preprocessing + model in ONE sklearn Pipeline (no data leakage)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "city_day.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
for _p in (FIG_DIR, RES_DIR, MODEL_DIR):
    _p.mkdir(exist_ok=True)

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene"]          # Xylene dropped (>50% missing)
NUM_FEATURES = POLLUTANTS + ["AQI", "AQI_lag1", "AQI_3day_mean",
                             "month_sin", "month_cos"]
CAT_FEATURES = ["City"]
ALERT_THRESHOLD = 200      # "Poor" or worse


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def load_data():
    """Return a DataFrame with features for day t and targets for day t+1."""
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.drop(columns=["Xylene", "AQI_Bucket"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    # AQI scale ends at 500 -> cap anything above
    df["AQI"] = df["AQI"].clip(upper=500)

    g = df.groupby("City")
    # next-day target
    df["AQI_next"] = g["AQI"].shift(-1)
    df["Date_next"] = g["Date"].shift(-1)
    # previous day AQI (only valid if the previous row is exactly 1 day before)
    df["AQI_lag1"] = g["AQI"].shift(1)
    df["Date_prev"] = g["Date"].shift(1)
    df.loc[(df["Date"] - df["Date_prev"]).dt.days != 1, "AQI_lag1"] = np.nan
    # 3-day mean AQI (t, t-1, t-2)
    df["AQI_3day_mean"] = g["AQI"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    # season as sin / cos
    m = df["Date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)

    # keep only rows with a valid AQI today AND tomorrow, and exactly 1 day apart
    ok = (df["AQI"].notna() & df["AQI_next"].notna()
          & ((df["Date_next"] - df["Date"]).dt.days == 1))
    df = df[ok].copy()

    # classification label
    df["alert"] = (df["AQI_next"] > ALERT_THRESHOLD).astype(int)
    return df.reset_index(drop=True)


def split_data(df):
    """Chronological split. Train is sorted by DATE (needed for TimeSeriesSplit)."""
    train = df[df["Date"] < "2019-01-01"].sort_values("Date").reset_index(drop=True)
    test = df[(df["Date"] >= "2019-01-01") & (df["Date"] <= "2020-02-29")] \
        .reset_index(drop=True)
    lockdown = df[(df["Date"] >= "2020-03-25") & (df["Date"] <= "2020-06-30")] \
        .reset_index(drop=True)
    return train, test, lockdown


# ----------------------------------------------------------------------------
# 2. PREPROCESSING (all fitted on training data only, inside the pipeline)
# ----------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at the 1st / 99th percentile learned from TRAIN data."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor(use_pca=False):
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("winsor", Winsorizer()),
             ("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)))
    num_pipe = Pipeline(steps)
    return ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CAT_FEATURES),
    ])


def build_pipeline(model, extra_steps=None, use_pca=False):
    """Pipeline = preprocessing -> (optional extra steps) -> model.
    The final step is always called 'model' so grids look like model__alpha."""
    steps = [("prep", make_preprocessor(use_pca))]
    if extra_steps:
        steps += extra_steps
    steps.append(("model", model))
    return Pipeline(steps)


# ----------------------------------------------------------------------------
# 3. METRICS
# ----------------------------------------------------------------------------
def reg_metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}
