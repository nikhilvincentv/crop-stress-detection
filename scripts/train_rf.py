#!/usr/bin/env python3
"""
TASK 4 - Train the AgriDefend Random Forest soil-telemetry classifier.

Input : data/processed/rf_training_data.csv  (44,000 real-anchored samples)
Model : RandomForestClassifier(n_estimators=200, max_depth=15,
                               min_samples_split=5, class_weight='balanced')
Split : 80 / 20 stratified by class.

Outputs:
  models/rf_multispecies.pkl
  plots/rf_feature_importance.png
  plots/rf_confusion_matrix.png
  data/processed/rf_val_probs.csv      (RF probability vectors on the val split,
                                         consumed by the fusion meta-learner)
Prints accuracy + per-class precision/recall/F1 (RF-only ablation number).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"
MODELS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

CLASSES = ["Healthy", "Water Stress", "Nutrient Deficient", "pH Imbalance", "Disease"]
SOIL_FEATURES = ["pH_dot", "delta_CO2", "N_dot", "P_dot", "K_dot", "theta"]


def main():
    df = pd.read_csv(PROC / "rf_training_data.csv")
    species_cols = sorted([c for c in df.columns if c.startswith("species_")])
    feature_cols = SOIL_FEATURES + species_cols
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    print("=" * 70)
    print("TASK 4 - RANDOM FOREST (soil telemetry branch)")
    print("=" * 70)
    print(f"train={len(X_tr)}  val={len(X_val)}  features={len(feature_cols)}")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nRF-only validation accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_val, y_pred, labels=CLASSES, digits=3))

    # ---- confusion matrix ----
    cm = confusion_matrix(y_val, y_pred, labels=CLASSES)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.title(f"RF Soil Classifier - Confusion Matrix ({acc*100:.1f}% acc)")
    plt.tight_layout()
    plt.savefig(PLOTS / "rf_confusion_matrix.png", dpi=200)
    plt.close()

    # ---- feature importance ----
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(8, 8))
    imp.plot(kind="barh")
    plt.title("RF Feature Importance (soil features + species one-hot)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS / "rf_feature_importance.png", dpi=200)
    plt.close()

    print("Top soil-feature importances:")
    for k in SOIL_FEATURES:
        print(f"  {k:<12} {imp[k]:.4f}")

    # ---- persist model + metadata ----
    joblib.dump({"model": rf, "feature_cols": feature_cols, "classes": CLASSES},
                MODELS / "rf_multispecies.pkl")

    # ---- export val probability vectors for the fusion meta-learner ----
    proba = rf.predict_proba(X_val)                 # columns follow rf.classes_
    proba = pd.DataFrame(proba, columns=[f"R_{c}" for c in rf.classes_])
    proba = proba[[f"R_{c}" for c in CLASSES]]      # reorder to canonical class order
    proba["label"] = y_val
    proba.to_csv(PROC / "rf_val_probs.csv", index=False)

    json.dump({"rf_only_accuracy": acc}, open(PROC / "rf_metrics.json", "w"), indent=2)
    print(f"\nSaved: models/rf_multispecies.pkl, plots/rf_*.png, "
          f"data/processed/rf_val_probs.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
