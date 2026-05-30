#!/usr/bin/env python3
"""
TASK 3 - Retrain the visual MobileNetV2 CNN for multi-species crop stress.

Combines PlantVillage + rice + wheat leaf-disease imagery, maps every source
class into the five AgriDefend health categories, trains a transfer-learned
MobileNetV2, exports an INT8 TFLite model, and reports full metrics.

Requirements to RUN (not satisfied on the macOS dev box used for the RF branch):
  * tensorflow==2.13.0            (Pi/Linux) or tensorflow-macos (Apple Silicon)
  * image datasets present under data/plantvillage, data/rice, data/wheat
    - PlantVillage may instead be pulled via tensorflow_datasets ('plant_village')

Outputs:
  data/processed/class_mapping.json
  models/cnn_multispecies.h5
  models/agridefend_multispecies.tflite
  plots/cnn_confusion_matrix_multispecies.png
  data/processed/cnn_val_probs.csv   (CNN probability vectors for fusion)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"
for d in (PROC, MODELS, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["Healthy", "Water Stress", "Nutrient Deficient", "pH Imbalance", "Disease"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = (224, 224)
BATCH = 32
MAX_EPOCHS = 25


# ----------------------------------------------------------------------------
# Explicit, complete source-class -> AgriDefend-category mapping (Task 3 rules)
# ----------------------------------------------------------------------------
def map_class(raw_name: str) -> str:
    n = raw_name.lower()
    if "healthy" in n:
        return "Healthy"
    if any(k in n for k in ["blight", "wilt", "scorch", "drought"]):
        return "Water Stress"
    if any(k in n for k in ["spot", "chlorosis", "yellow", "mosaic"]):
        return "Nutrient Deficient"
    if any(k in n for k in ["mildew", "rust", "mold", "leaf_curl", "leaf curl", "curl"]):
        return "pH Imbalance"
    return "Disease"   # any remaining disease class


def build_index() -> pd.DataFrame:
    """Walk the image dataset folders -> DataFrame[path, source_class, label]."""
    rows = []
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for sub in ["plantvillage", "rice", "wheat"]:
        base = DATA / sub
        if not base.exists():
            continue
        for img in base.rglob("*"):
            if img.suffix in exts:
                source_class = img.parent.name
                rows.append({"path": str(img), "source_class": source_class,
                             "label": map_class(source_class)})
    return pd.DataFrame(rows)


def save_mapping(df: pd.DataFrame):
    mapping = (df.drop_duplicates("source_class")
                 .set_index("source_class")["label"].to_dict())
    with open(PROC / "class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print("COMPLETE CLASS MAPPING (source_class -> AgriDefend category):")
    for k in sorted(mapping):
        print(f"  {k:<45} -> {mapping[k]}")
    print(f"\n{len(mapping)} source classes -> 5 categories; total images = {len(df)}")
    return mapping


def main():
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("=" * 70)
    print("TASK 3 - MULTI-SPECIES MobileNetV2 CNN")
    print("=" * 70)

    df = build_index()
    if df.empty:
        raise SystemExit(
            "No images found under data/plantvillage, data/rice, data/wheat.\n"
            "Provide Kaggle credentials and re-run scripts/download_datasets.py,\n"
            "or load PlantVillage via tensorflow_datasets('plant_village').")
    save_mapping(df)

    df["y"] = df["label"].map(CLASS_IDX)
    tr, val = train_test_split(df, test_size=0.20, random_state=42, stratify=df["y"])

    # ---- data generators (augmentation matches paper + PNW overcast channel shift) ----
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25, horizontal_flip=True, vertical_flip=True,
        brightness_range=[0.7, 1.3], zoom_range=0.15,
        width_shift_range=0.10, height_shift_range=0.10,
        channel_shift_range=30.0,                      # NEW: overcast/low-light
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    def flow(gen, frame, shuffle):
        return gen.flow_from_dataframe(
            frame, x_col="path", y_col="label", target_size=IMG_SIZE,
            classes=CLASSES, class_mode="categorical", batch_size=BATCH,
            shuffle=shuffle)

    train_flow = flow(train_gen, tr, True)
    val_flow = flow(val_gen, val, False)

    # ---- model: MobileNetV2 backbone, all frozen except top 30 layers ----
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False,
                       weights="imagenet")
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(5, activation="softmax"),
    ])
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    cw = compute_class_weight("balanced", classes=np.arange(5), y=tr["y"].to_numpy())
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    cbs = [
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
        callbacks.EarlyStopping(patience=7, restore_best_weights=True,
                                monitor="val_loss"),
    ]
    model.fit(train_flow, validation_data=val_flow, epochs=MAX_EPOCHS,
              class_weight=class_weight, callbacks=cbs)

    model.save(MODELS / "cnn_multispecies.h5")

    # ---- evaluation ----
    val_flow.reset()
    probs = model.predict(val_flow)
    y_true = val_flow.classes
    y_pred = probs.argmax(1)
    print("\n" + classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.title("Multi-species CNN - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS / "cnn_confusion_matrix_multispecies.png", dpi=200)
    plt.close()

    # CNN val probability vectors for the fusion meta-learner
    cnn_probs = pd.DataFrame(probs, columns=[f"V_{c}" for c in CLASSES])
    cnn_probs["label"] = [CLASSES[i] for i in y_true]
    cnn_probs.to_csv(PROC / "cnn_val_probs.csv", index=False)

    # ---- INT8 TFLite conversion (100 random val images as representative set) ----
    rep_paths = val["path"].sample(min(100, len(val)), random_state=42).tolist()

    def rep_data():
        for p in rep_paths:
            img = tf.keras.utils.load_img(p, target_size=IMG_SIZE)
            arr = preprocess_input(np.expand_dims(
                tf.keras.utils.img_to_array(img), 0).astype("float32"))
            yield [arr]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()
    tfl_path = MODELS / "agridefend_multispecies.tflite"
    tfl_path.write_bytes(tflite)
    size_mb = len(tflite) / 1e6
    print(f"TFLite INT8 model: {size_mb:.2f} MB -> {tfl_path}")

    # ---- latency benchmark over 50 runs (this machine; Pi 5 paper target = 1.4 s) ----
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
    for _ in range(5):
        interp.set_tensor(inp["index"], dummy); interp.invoke()
    t0 = time.perf_counter()
    for _ in range(50):
        interp.set_tensor(inp["index"], dummy); interp.invoke()
        interp.get_tensor(out["index"])
    lat = (time.perf_counter() - t0) / 50
    print(f"Mean TFLite inference latency (this machine): {lat*1000:.1f} ms")

    json.dump({"tflite_mb": size_mb, "latency_s_devbox": lat},
              open(PROC / "cnn_metrics.json", "w"), indent=2)
    print("=" * 70)


if __name__ == "__main__":
    main()
