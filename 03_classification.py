"""
03_classification.py  -  Health alert: will tomorrow's AQI be > 200 ("Poor" or worse)?
Models: Logistic, Gaussian NB, Decision tree, kNN, Perceptron, SVM (RBF), MLP
Baseline: persistence (alert tomorrow if AQI today > 200)

Creates:
  results/table2_best_params_classification.csv
  results/table4_classification.csv
  results/confusion_matrix_best.csv
  figures/fig7_roc_curves.png
Run:  python 03_classification.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import (ALERT_THRESHOLD, CAT_FEATURES, FIG_DIR, NUM_FEATURES,
                   RANDOM_STATE, RES_DIR, build_pipeline, load_data, split_data)


def class_metrics(y, pred, score=None):
    d = {"Acc": accuracy_score(y, pred),
         "Prec": precision_score(y, pred, zero_division=0),
         "Recall": recall_score(y, pred),
         "F1": f1_score(y, pred)}
    d["AUC"] = roc_auc_score(y, score) if score is not None else np.nan
    return d


def get_score(model, X):
    """Continuous score for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    df = load_data()
    train, test, lockdown = split_data(df)
    feats = NUM_FEATURES + CAT_FEATURES
    X_tr, y_tr = train[feats], train["alert"]
    X_te, y_te = test[feats], test["alert"]
    print(f"alert rate: train={y_tr.mean():.1%}  test={y_te.mean():.1%}")

    rs = RANDOM_STATE
    models = {
        "Logistic regression": (build_pipeline(LogisticRegression(max_iter=2000)),
                                {"model__C": [0.01, 0.1, 1, 10],
                                 "model__class_weight": [None, "balanced"]}),
        "Gauss. Naive Bayes": (build_pipeline(GaussianNB()), {}),
        "Decision tree": (build_pipeline(DecisionTreeClassifier(random_state=rs)),
                          {"model__max_depth": [4, 8, 12],
                           "model__class_weight": [None, "balanced"]}),
        "k-NN": (build_pipeline(KNeighborsClassifier()),
                 {"model__n_neighbors": [5, 15, 31]}),
        "Perceptron": (build_pipeline(Perceptron(random_state=rs)), {}),
        "SVM (RBF)": (build_pipeline(SVC(kernel="rbf", random_state=rs)),
                      {"model__C": [0.1, 1, 10],
                       "model__class_weight": [None, "balanced"]}),
        "MLP (64-32)": (build_pipeline(MLPClassifier(hidden_layer_sizes=(64, 32),
                                                     early_stopping=True,
                                                     max_iter=500,
                                                     random_state=rs)),
                        {"model__alpha": [1e-4, 1e-2, 1]}),
    }

    cv = TimeSeriesSplit(n_splits=5)
    rows, params_rows, scores = [], [], {}

    # persistence baseline
    base_pred = (test["AQI"] > ALERT_THRESHOLD).astype(int)
    rows.append({"Model": "Persistence baseline", **class_metrics(y_te, base_pred)})

    fitted = {}
    for name, (pipe, grid) in models.items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        pred = gs.predict(X_te)
        sc = get_score(gs.best_estimator_, X_te)
        scores[name] = sc
        fitted[name] = gs.best_estimator_
        m = class_metrics(y_te, pred, sc)
        rows.append({"Model": name, **m})
        params_rows.append({"Model": name,
                            "Selected": {k.replace("model__", ""): v
                                         for k, v in gs.best_params_.items()}})
        print(f"{name:22s} F1={m['F1']:.3f}  Recall={m['Recall']:.3f}  "
              f"AUC={m['AUC']:.3f}  ({time.time() - t0:.0f}s)")

    table4 = pd.DataFrame(rows).round(3)
    table4.to_csv(RES_DIR / "table4_classification.csv", index=False)
    pd.DataFrame(params_rows).to_csv(
        RES_DIR / "table2_best_params_classification.csv", index=False)
    print("\n", table4.to_string(index=False))

    # ---- ROC curves --------------------------------------------------------
    plt.figure(figsize=(6, 5))
    for name, sc in scores.items():
        fpr, tpr, _ = roc_curve(y_te, sc)
        auc = roc_auc_score(y_te, sc)
        plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("ROC curves on the test set")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_roc_curves.png", dpi=200)
    plt.close()

    # ---- confusion matrix of best-F1 model ---------------------------------
    learned = table4[table4["Model"] != "Persistence baseline"]
    best_name = learned.loc[learned["F1"].idxmax(), "Model"]
    pred = fitted[best_name].predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    cm = pd.DataFrame({"TN": [tn], "FP": [fp], "FN": [fn], "TP": [tp]})
    cm.insert(0, "Model", best_name)
    cm.to_csv(RES_DIR / "confusion_matrix_best.csv", index=False)
    print(f"\nBest F1 model: {best_name}")
    print(cm.to_string(index=False))
    print(f"Missed alerts (FN) = {fn} of {fn + tp} real alert days "
          f"({fn / (fn + tp):.1%})")
    print("\nDone.")


if __name__ == "__main__":
    main()
