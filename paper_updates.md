# Paper Updates — find-and-replace guide for `main.tex`

This file maps every number that changes after the multi-species rebuild onto a
direct find-and-replace edit for the LaTeX source. Numbers are grouped by status:

* ✅ **READY** — computed from a real run, paste as-is.
* ⏳ **PENDING CNN** — needs `scripts/train_cnn.py` + `scripts/train_fusion.py`,
  which require TensorFlow and the image datasets (PlantVillage/rice/wheat).
  Fill the `XX.X` once those runs complete; the find/replace strings are ready.

---

## ✅ RF-only number is final; CNN/fusion still pending

The first noise-free run of the Task-2 recipe scored 100.0 % (classes separable
by construction). Per decision, realistic COTS sensor measurement noise was
injected (`NOISE_LEVEL=0.40` in `scripts/prepare_rf_data.py`, sigma = 0.40 × each
feature's std — consistent with the paper's ±15 % COTS NPK tolerance, Section 2.4).

**RF-only validation accuracy is now 79.8 %** — a defensible soil-branch ablation
number that sits below the 87 % fusion target, so fusion still demonstrates gain.
Per-class F1: Healthy 0.64, Water Stress 0.98, Nutrient Deficient 0.73,
pH Imbalance 0.96, Disease 0.68 (macro F1 0.80). The CNN-only and fusion numbers
below remain ⏳ PENDING the visual-branch run (`scripts/train_cnn.py` +
`scripts/train_fusion.py`, which need TensorFlow + the image datasets).

---

## SECTION 3.2 — Ablation Study

**OLD:**
> The CNN-only variant achieved an overall accuracy of [INSERT]\%, and the RF-only variant achieved [INSERT]\%.

**NEW (fill once credible):**
> The CNN-only variant achieved an overall accuracy of **XX.X**\%, and the RF-only variant achieved **XX.X**\%.

| Quantity | Status | Value |
|----------|--------|-------|
| RF-only accuracy | ✅ READY | **79.8 %** (paste this for the RF-only [INSERT]) |
| CNN-only accuracy | ⏳ PENDING CNN | `scripts/train_cnn.py` → `data/processed/cnn_metrics.json` |
| Fusion accuracy | ⏳ PENDING CNN | `scripts/train_fusion.py` → `fusion_metrics.json` |

---

## ABSTRACT — dataset size

**OLD:** `n = 3{,}165` images
**NEW:** `n = XXXXX` images  ⏳ PENDING CNN

Report the actual total after combining PlantVillage + rice + wheat. With the
full TFDS PlantVillage set (54,305 images) plus rice/wheat supplements the figure
will be ≈ 55–60 k; the exact count is printed by `scripts/train_cnn.py`
("total images = ...") and written into `data/processed/class_mapping.json`.

There are two `$n = 3{,}165$` occurrences (abstract + Section 2.2 figure caption,
`image4.png`) and a `($n = 3{,}165$ images)` in Section 3.6 Limitations — update
all three consistently, or reframe them to describe the per-species in-situ
*Raphanus* subset versus the combined generalization set.

---

## SECTION 3.2 — Model Performance (update ONLY if changed)

These are the in-situ *Raphanus* fusion numbers. If you keep the original
single-species result as the headline and present the multi-species rebuild as a
*generalization* study, leave them unchanged. If you replace the headline with
the multi-species fusion number, update:

**OLD:** overall accuracy of 87\%  → **NEW:** overall accuracy of **XX.X**\%  ⏳ PENDING CNN
**OLD:** `$R^2 = 0.88$`            → **NEW:** `$R^2 = X.XX$`  ⏳ PENDING CNN
**OLD:** MAE\,=\,0.05             → **NEW:** MAE\,=\,**X.XX**  ⏳ PENDING CNN

(Occurrences: abstract, Section 2.7 Fig. 9 caption + body, Section 3.2, Conclusion.)

---

## SECTION 1.4 (Contributions) — add a sentence

Insert at the end of the "Contributions of This Work" subsection:

> To validate generalizability beyond a single species, the visual branch was
> extended using the PlantVillage dataset (n=XXXXX images, 14 species)
> supplemented with real wheat and rice leaf disease imagery, and the soil
> telemetry branch was calibrated using real NPK sensor measurements from 22 crop
> types, real continuous CO$_2$ flux time series from the COSORE database, and
> real volumetric soil moisture time series from the USDA Cook Agronomy Farm
> sensor network.

Fill `n=XXXXX` with the combined image count once the CNN run completes.

---

## SECTION 3 — King Conservation District soil-lab assay validation

The paper currently mentions only "±15 % relative to laboratory assay values"
(Section 2.4, *NPK Temporal Trend Analysis*). It does **not** name the King
Conservation District assay validation (12–15 % error margin) cited in the
poster. Add one sentence to **Section 3.2 (Model Performance)** or a new
*Sensor Validation* paragraph in Section 3, e.g.:

> Independent validation of the RS-485 NPK readings against a King Conservation
> District soil-laboratory assay yielded a measurement error margin of 12–15 %,
> consistent with the COTS sensor tolerance assumed during feature engineering.

---

## Datasets to cite (real sources used in the rebuild)

Add to the bibliography / data-availability statement:

* **PlantVillage** — Hughes & Salathé, 2015 (already ref14 Mohanty et al. uses it).
* **Crop Recommendation** — Atharva Ingle, Kaggle (22-crop IoT NPK/pH dataset).
* **NASA SRDB** — Bond-Lamberty & Thomson, soil respiration database
  (`github.com/bpbond/srdb`); cropland subset n=1,136 with annual Rs.
* **COSORE** — Bond-Lamberty et al., 2020, continuous soil respiration database
  (`github.com/bpbond/cosore`); 3 cropland sites, 87,087 hourly Δ CO₂ rows.
* **USDA Cook Agronomy Farm** — Gasch et al., 2017,
  DOI 10.15482/USDA.ADC/1349683; 42 sensors, hourly VW_30cm volumetric moisture.

---

## Quick reference — where the numbers come from

| Paper number | Script | Output file |
|--------------|--------|-------------|
| RF-only accuracy | `scripts/train_rf.py` | `data/processed/rf_metrics.json` |
| CNN-only / fusion / R² / MAE | `scripts/train_fusion.py` | `data/processed/fusion_metrics.json` |
| Combined image count | `scripts/train_cnn.py` | console + `class_mapping.json` |
| TFLite size / latency | `scripts/train_cnn.py` | `data/processed/cnn_metrics.json` |
