"""
07_alert_analysis.py  -  A closer look at the health-alert task (AQI > 200).

Questions answered:
  1. CALIBRATION: when the model says "70% chance of an alert", do ~70% of
     those days really turn out to be alert days?  (reliability diagram,
     Brier score, expected calibration error)
  2. THRESHOLD: the default cut-off 0.5 is not sacred. Because a missed alert
     is worse than a false alarm, we choose the threshold that minimises
     (COST_RATIO x missed alerts + false alarms). COST_RATIO = 5 is an
     ASSUMPTION (one missed alert is as bad as 5 false alarms); the script also
     shows ratios 1, 2 and 10 so you can see how the threshold moves.
     IMPORTANT: the threshold is chosen on a VALIDATION slice of the training
     years (last 20% of the training period), never on the test set.
  3. COST OF MISSED ALERTS: precision / recall / false alarms / missed alerts
     at each threshold on the test set.

Final classifier = the one selected by CV-F1 in 03_classification.py
(logistic regression).

Needs:   python 03_classification.py   (creates selected_classifier.txt)
Creates: results/table7_threshold_sweep.csv
         results/table8_calibration_bins.csv
         figures/fig8_alert_calibration_pr.png
Run:     python 07_alert_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from utils import (CAT_FEATURES, FIG_DIR, NUM_FEATURES, RES_DIR, build_pipeline,
                   load_data, split_data)

COST_RATIOS = [1, 2, 5, 10]   # 1 missed alert = r false alarms (assumption)
MAIN_RATIO = 5                # headline operating point
SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"threshold": round(t, 2),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "F1": f1_score(y, pred),
            "alerts_issued": int(pred.sum()),
            "FP (false alarms)": int(fp), "FN (missed alerts)": int(fn),
            "TP": int(tp), "TN": int(tn)}


def main():
    final = (RES_DIR / "selected_classifier.txt").read_text().strip()
    if final != "Logistic regression":
        print(f"NOTE: the CV-selected classifier is '{final}', but this script "
              f"analyses logistic regression (it gives probabilities).")

    df = load_data()
    train, test, _ = split_data(df)
    feats = NUM_FEATURES + CAT_FEATURES
    X_tr, y_tr = train[feats], train["alert"]
    X_te, y_te = test[feats], test["alert"]

    # ---- fit the final model (same grid as 03_classification.py) ------------
    pipe = build_pipeline(LogisticRegression(max_iter=2000))
    grid = {"model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"]}
    gs = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(5), scoring="f1", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    model = gs.best_estimator_
    print("Selected parameters:", {k.replace("model__", ""): v
                                   for k, v in gs.best_params_.items()})
    p_te = model.predict_proba(X_te)[:, 1]

    # ---- 1. calibration -----------------------------------------------------
    frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="uniform")
    bins = np.clip((p_te * 10).astype(int), 0, 9)
    counts = np.bincount(bins, minlength=10)
    counts = counts[counts > 0]               # bins that contain data (same order)
    cal = pd.DataFrame({"mean_predicted_prob": mean_pred,
                        "observed_alert_rate": frac_pos, "n": counts}).round(3)
    cal.to_csv(RES_DIR / "table8_calibration_bins.csv", index=False)
    brier = brier_score_loss(y_te, p_te)
    ece = float(np.sum(counts / counts.sum() * np.abs(frac_pos - mean_pred)))
    print(f"\nCalibration on test set: Brier score = {brier:.3f}, "
          f"expected calibration error (ECE) = {ece:.3f}")
    print(cal.to_string(index=False))

    # ---- 2. choose the threshold on a VALIDATION slice of training years ----
    cut = int(0.8 * len(train))               # train is sorted by date
    val_model = clone(model).fit(X_tr.iloc[:cut], y_tr.iloc[:cut])
    p_val = val_model.predict_proba(X_tr.iloc[cut:])[:, 1]
    y_val = y_tr.iloc[cut:]
    cands = np.arange(0.05, 0.96, 0.01)

    def val_cost(t, r):
        pred = (p_val >= t).astype(int)
        fn = int(((pred == 0) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        return r * fn + fp

    chosen_by_ratio = {r: float(min(cands, key=lambda t: val_cost(t, r)))
                       for r in COST_RATIOS}
    print("\nThreshold chosen on the validation slice, by cost ratio "
          "(1 missed alert = r false alarms):")
    for r, t in chosen_by_ratio.items():
        print(f"  r = {r:>2}  ->  threshold {t:.2f}")
    chosen = chosen_by_ratio[MAIN_RATIO]

    # ---- 3. thresholds on the test set ---------------------------------------
    rows = [metrics_at(y_te, p_te, t) for t in SWEEP]
    for r, t in chosen_by_ratio.items():
        row = metrics_at(y_te, p_te, t)
        row["threshold"] = f"{t:.2f} (cost ratio {r}:1, chosen on validation)"
        rows.append(row)
    sweep = pd.DataFrame(rows)
    sweep.round(3).to_csv(RES_DIR / "table7_threshold_sweep.csv", index=False)
    print("\nTest-set results by threshold:\n",
          sweep.round(3).to_string(index=False))

    default = metrics_at(y_te, p_te, 0.5)
    chosen_row = metrics_at(y_te, p_te, chosen)
    n_alert = int(y_te.sum())
    d_fn = default["FN (missed alerts)"] - chosen_row["FN (missed alerts)"]
    d_fp = chosen_row["FP (false alarms)"] - default["FP (false alarms)"]
    print(f"\nCost of missed alerts (test set, {n_alert} real alert days):")
    print(f"  threshold 0.50 : missed {default['FN (missed alerts)']} "
          f"({default['FN (missed alerts)'] / n_alert:.1%}), "
          f"false alarms {default['FP (false alarms)']}")
    print(f"  threshold {chosen:.2f} : missed {chosen_row['FN (missed alerts)']} "
          f"({chosen_row['FN (missed alerts)'] / n_alert:.1%}), "
          f"false alarms {chosen_row['FP (false alarms)']}")
    if d_fn > 0:
        print(f"  -> lowering the threshold catches {d_fn} more real alert days "
              f"at the price of {d_fp} more false alarms "
              f"(about {d_fp / d_fn:.1f} extra false alarms per extra alert caught).")

    # ---- figure: reliability diagram + precision-recall curve ----------------
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
    ax[0].plot(mean_pred, frac_pos, "o-", label="logistic regression")
    ax[0].set_xlabel("Predicted probability of alert")
    ax[0].set_ylabel("Observed share of alert days")
    ax[0].set_title(f"Calibration (Brier {brier:.3f}, ECE {ece:.3f})")
    ax[0].legend(fontsize=8)
    ax[1].plot(rec, prec, color="tab:blue")
    ax[1].scatter([default["recall"]], [default["precision"]], color="tab:red",
                  zorder=3, label="threshold 0.50")
    ax[1].scatter([chosen_row["recall"]], [chosen_row["precision"]], color="tab:green",
                  zorder=3, label=f"threshold {chosen:.2f} (cost ratio {MAIN_RATIO}:1)")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-recall curve (test set)")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_alert_calibration_pr.png", dpi=200)
    plt.close()
    print("\nDone.")


if __name__ == "__main__":
    main()