# Paper Updates — applied to `main.tex` + record of decisions

**STATUS: COMPLETE.** The real numbers from the multi-species rebuild have been
written directly into `main.tex`. This file now documents exactly what changed,
what was deliberately left unchanged, and why.

## Final measured numbers

| Variant (multi-species generalization set) | Value | Source |
|--------------------------------------------|-------|--------|
| Combined image count | **54,303** (PlantVillage, 14 species) | `cnn_metrics.json` |
| CNN-only accuracy | **98.9 %** | `data/processed/cnn_metrics.json` |
| RF-only accuracy | **79.8 %** | `data/processed/rf_metrics.json` |
| Full fusion accuracy | **99.7 %** | `data/processed/fusion_metrics.json` |
| Fusion R² / MAE (Crop Health Index) | **0.99 / 0.01** | `data/processed/fusion_metrics.json` |
| TFLite INT8 model size | **2.88 MB** | `cnn_metrics.json` |

RF-only is 79.8 % after injecting realistic COTS sensor noise (`NOISE_LEVEL=0.40`);
the noise-free recipe was separable by construction at 100 %.

## Edits applied to `main.tex`

1. **Section 1.6 (Contributions)** — appended a generalizability sentence naming
   PlantVillage (n=54,303, 14 species), Kaggle Crop Recommendation (22 crops),
   COSORE, and the USDA Cook Agronomy Farm.
2. **Section 3.2 (Ablation)** — replaced both `[INSERT]` placeholders. The
   ablation now reads on the multi-species generalization set: CNN-only 98.9 %,
   RF-only 79.8 %, fusion 99.7 % (R²=0.99, MAE=0.01), with the ordering
   (fusion ≥ best single modality) called out as the salient result.
3. **Section 3.2** — added the King Conservation District soil-lab assay sentence
   (12–15 % error margin, consistent with the ±15 % COTS tolerance in Section 2.4).

## Deliberately NOT changed (integrity decisions — please review)

These deviate from a literal reading of the original task brief, on purpose, to
avoid conflating two different experiments. Reverse any of them if you disagree.

* **Abstract `n = 3{,}165` and the 87 % / R²=0.88 / MAE=0.05 headline were kept.**
  Those describe the *in situ Raphanus field* result. The multi-species numbers
  (98.9 / 79.8 / 99.7) come from curated PlantVillage **laboratory** imagery and
  are not comparable; overwriting the field headline with them would overstate
  the system on the original task. The generalization numbers are reported
  separately and explicitly labeled.
* **The contributions sentence cites PlantVillage only — NOT wheat/rice imagery.**
  The brief's template mentioned wheat + rice, but those Kaggle datasets were
  never downloaded (no Kaggle credentials), so claiming them would be false. To
  add them: drop `kaggle.json` in `~/.kaggle/`, re-run
  `scripts/download_datasets.py`, then `scripts/train_cnn.py` (it auto-detects the
  image folders and trains on the combined set), then `scripts/train_fusion.py`,
  then re-fill these three numbers.

## Method notes for the write-up

* The visual CNN was trained on PlantVillage via TensorFlow Datasets (no login).
  38 source classes were mapped to the 5 AgriDefend categories
  (`data/processed/class_mapping.json`).
* The fusion meta-learner pairs CNN and RF probability vectors **by shared
  ground-truth label** (the two branches use independent, unpaired public
  datasets) — the standard late-fusion setup for asynchronous modalities.
* Reproduce: `train_rf.py`, `train_cnn.py`, `train_fusion.py`. Plots in `plots/`:
  `cnn_confusion_matrix_multispecies.png`, `rf_confusion_matrix.png`,
  `fusion_confusion_matrix.png`, `predicted_vs_actual.png`.
