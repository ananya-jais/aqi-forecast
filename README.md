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
(Fill in from results/*.csv after you run the scripts.)
