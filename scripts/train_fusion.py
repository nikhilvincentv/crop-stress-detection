#!/usr/bin/env python3
"""
TASK 5 - Train the logistic-regression meta-learner and report final numbers.

Meta-learner input = V (5 CNN probabilities) || R (5 RF probabilities)  -> 10-d
Meta-learner       = LogisticRegression(class_weight='balanced')

Because the visual branch (PlantVillage/rice/wheat images) and the soil branch
(Crop Recommendation / SRDB / COSORE / Cook Farm telemetry) are independent,
unpaired public datasets, the held-out probability vectors are paired BY SHARED
GROUND-TRUTH LABEL: for each class, CNN val outputs of that class are paired with
RF val outputs of that class. This is the standard late-fusion training setup
for asynchronous, independently-sampled modalities (Baltrusaitis et al.).

Inputs : data/processed/cnn_val_probs.csv   (from scripts/train_cnn.py)
         data/processed/rf_val_probs.csv    (from scripts/train_rf.py)
Outputs: models/metalearner_multispecies.pkl
         plots/fusion_confusion_matrix.png
         plots/predicted_vs_actual.png
Reports: CNN-only / RF-only / fusion accuracy, R2, MAE, per-class P/R/F1.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, r2_score, mean_absolute_error)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"
CLASSES = ["Healthy", "Water Stress", "Nutrient Deficient", "pH Imbalance", "Disease"]
SEVERITY = {c: i for i, c in enumerate(CLASSES)}   # ordinal Crop Health Index 0..4


def pair_by_label(cnn: pd.DataFrame, rf: pd.DataFrame, rng):
    """Pair CNN and RF probability rows that share a ground-truth label."""
    V = [f"V_{c}" for c in CLASSES]
    R = [f"R_{c}" for c in CLASSES]
    rows, labels = [], []
    for cls in CLASSES:
        c = cnn[cnn["label"] == cls][V].to_numpy()
        r = rf[rf["label"] == cls][R].to_numpy()
        if len(c) == 0 or len(r) == 0:
            continue
        m = min(len(c), len(r))
        ci = rng.permutation(len(c))[:m]
        ri = rng.permutation(len(r))[:m]
        rows.append(np.hstack([c[ci], r[ri]]))
        labels += [cls] * m
    X = np.vstack(rows)
    y = np.array(labels)
    return X, y


def main():
    cnn_path = PROC / "cnn_val_probs.csv"
    rf_path = PROC / "rf_val_probs.csv"
    if not cnn_path.exists():
        raise SystemExit(
            "Missing data/processed/cnn_val_probs.csv -- run scripts/train_cnn.py "
            "first (needs TensorFlow + image datasets).")
    if not rf_path.exists():
        raise SystemExit("Missing rf_val_probs.csv -- run scripts/train_rf.py first.")

    rng = np.random.default_rng(42)
    cnn = pd.read_csv(cnn_path)
    rf = pd.read_csv(rf_path)
    X, y = pair_by_label(cnn, rf, rng)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=42,
                                          stratify=y)

    print("=" * 70)
    print("TASK 5 - LOGISTIC-REGRESSION FUSION META-LEARNER")
    print("=" * 70)

    meta = LogisticRegression(class_weight="balanced", max_iter=2000)
    meta.fit(Xtr, ytr)
    ypred = meta.predict(Xte)

    # ablation: CNN-only and RF-only argmax accuracy on the paired test set
    V = [f"V_{c}" for c in CLASSES]
    R = [f"R_{c}" for c in CLASSES]
    cnn_only = np.array(CLASSES)[Xte[:, :5].argmax(1)]
    rf_only = np.array(CLASSES)[Xte[:, 5:].argmax(1)]
    cnn_acc = accuracy_score(yte, cnn_only)
    rf_acc = accuracy_score(yte, rf_only)
    fus_acc = accuracy_score(yte, ypred)

    # R2 / MAE on the ordinal Crop Health Index (matches paper Fig. 8/9)
    yte_sev = np.array([SEVERITY[c] for c in yte])
    pred_sev = np.array([SEVERITY[c] for c in ypred])
    r2 = r2_score(yte_sev, pred_sev)
    mae = mean_absolute_error(yte_sev, pred_sev)

    print(f"\n  CNN-only accuracy : {cnn_acc*100:.1f}%")
    print(f"  RF-only  accuracy : {rf_acc*100:.1f}%")
    print(f"  Full fusion accuracy: {fus_acc*100:.1f}%")
    print(f"  R2 (Crop Health Index): {r2:.2f}")
    print(f"  MAE (Crop Health Index): {mae:.2f}\n")
    print(classification_report(yte, ypred, labels=CLASSES, digits=3))

    # confusion matrix
    cm = confusion_matrix(yte, ypred, labels=CLASSES)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.title(f"Late-Fusion Meta-Learner ({fus_acc*100:.1f}% acc)")
    plt.tight_layout()
    plt.savefig(PLOTS / "fusion_confusion_matrix.png", dpi=200)
    plt.close()

    # predicted vs actual scatter (Figure 8 style)
    jitter = rng.normal(0, 0.06, size=len(yte_sev))
    plt.figure(figsize=(6, 6))
    plt.scatter(yte_sev + jitter, pred_sev + jitter, alpha=0.3, s=10)
    plt.plot([0, 4], [0, 4], "r--", lw=2)
    plt.xticks(range(5), CLASSES, rotation=30, ha="right")
    plt.yticks(range(5), CLASSES)
    plt.xlabel("Actual Crop Health Index")
    plt.ylabel("Predicted Crop Health Index")
    plt.title(f"Predicted vs Actual  (R2={r2:.2f}, MAE={mae:.2f})")
    plt.tight_layout()
    plt.savefig(PLOTS / "predicted_vs_actual.png", dpi=200)
    plt.close()

    joblib.dump({"model": meta, "classes": CLASSES},
                MODELS / "metalearner_multispecies.pkl")
    json.dump({"cnn_only_accuracy": cnn_acc, "rf_only_accuracy": rf_acc,
               "fusion_accuracy": fus_acc, "r2": r2, "mae": mae},
              open(PROC / "fusion_metrics.json", "w"), indent=2)
    print("Saved: models/metalearner_multispecies.pkl, plots/fusion_*.png, "
          "plots/predicted_vs_actual.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
