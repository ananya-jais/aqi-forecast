# Next-Day AQI Forecasting and Health-Alert Classification for Indian Cities

ML course assignment (B.Sc. (H) Computer Science, Sem V, University of Delhi).
SDG 3 (Good Health and Well-being) and SDG 11 (Sustainable Cities and Communities).

## Problem
1. Regression: forecast tomorrow's AQI from today's pollutant concentrations.
2. Classification: will tomorrow be "Poor" or worse (AQI > 200)?

## Data
Kaggle "Air Quality Data in India (2015-2020)", file `city_day.csv`
(https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india), compiled from CPCB. Licence: CC0.
Put the file at `data/city_day.csv`.

## How to run
```
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python 01_eda.py               # figures 1-3, missing-value table
python 02_regression.py        # 8 regressors, tables 2-3, lockdown, per-city (a few minutes)
python 03_classification.py    # 7 classifiers, table 4, ROC curves
python 04_pca_analysis.py      # PCA figure and table 5
```
All random seeds are 42. Outputs go to `figures/` and `results/`.

## Method (short)
- Chronological split: train 2015-2018, test Jan 2019 - Feb 2020, lockdown (25 Mar - 30 Jun 2020) held out as a stress test.
- Preprocessing inside one sklearn Pipeline, fitted on training data only: median imputation, winsorising at 1st/99th percentile, standardisation, one-hot city.
- Hyper-parameters by GridSearchCV with 5-fold forward-chaining TimeSeriesSplit.
- Baseline: persistence (tomorrow = today).

## Folder structure
```
utils.py              shared data loading, split, pipeline
01_eda.py ... 04_pca_analysis.py
data/                 city_day.csv
figures/              all plots used in the report
results/              all tables (CSV)
```

## Results

All results are on the held-out test set (1 Jan 2019 - 29 Feb 2020). Training: 2015-2018 (13,041 samples). Test: 8,374 samples. COVID lockdown (25 Mar - 30 Jun 2020, 2,369 samples) is a separate stress test. The full tables for all models are in `results/` and `results/summary.txt`.

### Next-day AQI regression

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Persistence baseline | 25.59 | 42.43 | 0.839 |
| Linear regression (OLS) | 24.55 | 36.90 | 0.879 |
| Lasso | 24.39 | 37.75 | 0.873 |
| Decision tree | 23.33 | 36.39 | 0.882 |
| SVR (RBF) | 23.69 | 34.96 | 0.891 |

SVR gave the lowest test RMSE, 17.6% below persistence. The strongest lasso predictors were today's AQI, PM2.5, the 3-day mean AQI and CO.

### Health alert classification (next-day AQI > 200)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Persistence baseline | 0.922 | 0.832 | 0.834 | 0.833 | - |
| Logistic regression | 0.938 | 0.873 | 0.860 | 0.866 | 0.981 |
| Decision tree | 0.927 | 0.791 | 0.931 | 0.855 | 0.974 |
| SVM (RBF) | 0.938 | 0.870 | 0.863 | 0.866 | 0.979 |
| MLP (64-32) | 0.937 | 0.865 | 0.865 | 0.865 | 0.981 |

Logistic regression and SVM tied for the best F1. The decision tree had the highest recall (0.931). For logistic regression, 273 of 1,944 real alert days were missed (14.0%).

### Robustness checks

- COVID lockdown: SVR RMSE 25.18 vs persistence 21.11, so the model was worse than persistence when emissions changed suddenly (distribution shift).
- Per-city: SVR beat persistence in 20 of 25 cities. It was worse in Kolkata, Coimbatore, Shillong, Kochi and Ernakulam, mostly cities with little or no training data.
- PCA (95% variance) did not improve results (OLS RMSE 36.90 -> 40.07).