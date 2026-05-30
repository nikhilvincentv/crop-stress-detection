#!/usr/bin/env python3
"""
TASK 3 - Retrain the visual MobileNetV2 CNN for multi-species crop stress.

Two data sources are supported, auto-detected:
  (A) image folders under data/plantvillage, data/rice, data/wheat
      (the combined PlantVillage + rice + wheat set, when Kaggle imagery present)
  (B) TensorFlow Datasets 'plant_village' (no login) when the folders are empty.

Every source class is mapped into the five AgriDefend health categories, a
transfer-learned MobileNetV2 is trained, an INT8 TFLite model is exported, and
full metrics + the CNN validation probability vectors (for fusion) are written.

Outputs:
  data/processed/class_mapping.json
  models/cnn_multispecies.h5
  models/agridefend_multispecies.tflite
  plots/cnn_confusion_matrix_multispecies.png
  data/processed/cnn_val_probs.csv
  data/processed/cnn_metrics.json
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


def build_folder_index() -> pd.DataFrame:
    rows = []
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for sub in ["plantvillage", "rice", "wheat"]:
        base = DATA / sub
        if not base.exists():
            continue
        for img in base.rglob("*"):
            if img.suffix in exts:
                sc = img.parent.name
                rows.append({"path": str(img), "source_class": sc,
                             "label": map_class(sc)})
    return pd.DataFrame(rows)


def save_mapping(mapping: dict, n_images: int):
    with open(PROC / "class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print("COMPLETE CLASS MAPPING (source_class -> AgriDefend category):")
    for k in sorted(mapping):
        print(f"  {k:<45} -> {mapping[k]}")
    print(f"\n{len(mapping)} source classes -> 5 categories; total images = {n_images}")


def shared_finish(model, val_probs, y_true_idx, rep_images, n_images):
    """Evaluation, plots, TFLite INT8, latency, metrics -- common to both sources."""
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = val_probs.argmax(1)
    acc = float((y_pred == y_true_idx).mean())
    print(f"\nCNN-only validation accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_true_idx, y_pred, labels=range(5),
                                target_names=CLASSES, digits=3, zero_division=0))

    cm = confusion_matrix(y_true_idx, y_pred, labels=range(5))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.title(f"Multi-species CNN - Confusion Matrix ({acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(PLOTS / "cnn_confusion_matrix_multispecies.png", dpi=200)
    plt.close()

    cnn_probs = pd.DataFrame(val_probs, columns=[f"V_{c}" for c in CLASSES])
    cnn_probs["label"] = [CLASSES[i] for i in y_true_idx]
    cnn_probs.to_csv(PROC / "cnn_val_probs.csv", index=False)

    # ---- INT8 TFLite (100 representative val images) ----
    def rep_data():
        for im in rep_images:
            yield [np.expand_dims(im, 0).astype("float32")]

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

    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
    for _ in range(5):
        interp.set_tensor(inp["index"], dummy); interp.invoke()
    t0 = time.perf_counter()
    for _ in range(50):
        interp.set_tensor(inp["index"], dummy); interp.invoke(); interp.get_tensor(out["index"])
    lat = (time.perf_counter() - t0) / 50
    print(f"Mean TFLite inference latency (this machine): {lat*1000:.1f} ms")

    json.dump({"cnn_only_accuracy": acc, "tflite_mb": size_mb,
               "latency_s_devbox": lat, "n_images": n_images},
              open(PROC / "cnn_metrics.json", "w"), indent=2)
    print("=" * 70)


def build_model():
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.applications import MobileNetV2
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
    return model


# ----------------------------------------------------------------------------
def run_tfds():
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from sklearn.utils.class_weight import compute_class_weight

    print("Loading PlantVillage via TensorFlow Datasets (no login required)...")
    (tr_raw, val_raw), info = tfds.load(
        "plant_village", split=["train[:80%]", "train[80%:]"],
        as_supervised=True, with_info=True)
    names = info.features["label"].names
    mapping = {n: map_class(n) for n in names}
    n_total = int(info.splits["train"].num_examples)
    save_mapping(mapping, n_total)

    remap = tf.constant([CLASS_IDX[map_class(n)] for n in names], dtype=tf.int64)

    rot = layers.RandomRotation(25 / 360.0)
    zoom = layers.RandomZoom(0.15)
    trans = layers.RandomTranslation(0.10, 0.10)

    def decode(img, lbl):
        img = tf.image.resize(tf.cast(img, tf.float32), IMG_SIZE)
        return img, tf.gather(remap, lbl)

    def augment(img, lbl):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = img * tf.random.uniform([], 0.7, 1.3)            # brightness range
        img = img + tf.random.uniform([3], -30.0, 30.0)        # channel shift (overcast)
        img = tf.clip_by_value(img, 0.0, 255.0)
        img = img[None, ...]
        img = trans(zoom(rot(img, training=True), training=True), training=True)
        return tf.clip_by_value(img[0], 0.0, 255.0), lbl

    def to_model(img, lbl):
        return preprocess_input(img), tf.one_hot(lbl, 5)

    AUTO = tf.data.AUTOTUNE
    train_ds = (tr_raw.map(decode, AUTO).map(augment, AUTO).map(to_model, AUTO)
                .batch(BATCH).prefetch(AUTO))
    val_ds = (val_raw.map(decode, AUTO).map(to_model, AUTO).batch(BATCH).prefetch(AUTO))

    # label arrays (for class weights + evaluation) -- skip JPEG decode for speed
    remap_np = [CLASS_IDX[map_class(n)] for n in names]

    def labels_only(split):
        # skip JPEG decode (faster); SkipDecoding must target the 'image' feature
        # and therefore cannot be combined with as_supervised's (img,label) tuples.
        ds = tfds.load("plant_village", split=split,
                       decoders={"image": tfds.decode.SkipDecoding()})
        return np.array([remap_np[int(ex["label"])] for ex in ds])

    y_train = labels_only("train[:80%]")
    y_val = labels_only("train[80%:]")
    cw = compute_class_weight("balanced", classes=np.arange(5), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("class distribution (train):", np.bincount(y_train, minlength=5).tolist())

    model = build_model()
    cbs = [callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
           callbacks.EarlyStopping(patience=7, restore_best_weights=True,
                                   monitor="val_loss")]
    model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
              class_weight=class_weight, callbacks=cbs)
    model.save(MODELS / "cnn_multispecies.h5")

    val_probs = model.predict(val_ds)
    rep_imgs = []
    for img, _ in val_ds.unbatch().take(100):
        rep_imgs.append(img.numpy())
    shared_finish(model, val_probs, y_val[:len(val_probs)], rep_imgs, n_total)


def run_folders(df):
    import tensorflow as tf
    from tensorflow.keras import callbacks
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    mapping = (df.drop_duplicates("source_class").set_index("source_class")["label"].to_dict())
    save_mapping(mapping, len(df))
    df["y"] = df["label"].map(CLASS_IDX)
    tr, val = train_test_split(df, test_size=0.20, random_state=42, stratify=df["y"])

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input, rotation_range=25,
        horizontal_flip=True, vertical_flip=True, brightness_range=[0.7, 1.3],
        zoom_range=0.15, width_shift_range=0.10, height_shift_range=0.10,
        channel_shift_range=30.0)
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    def flow(gen, frame, shuffle):
        return gen.flow_from_dataframe(frame, x_col="path", y_col="label",
            target_size=IMG_SIZE, classes=CLASSES, class_mode="categorical",
            batch_size=BATCH, shuffle=shuffle)

    train_flow, val_flow = flow(train_gen, tr, True), flow(val_gen, val, False)
    cw = compute_class_weight("balanced", classes=np.arange(5), y=tr["y"].to_numpy())
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    model = build_model()
    cbs = [callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
           callbacks.EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss")]
    model.fit(train_flow, validation_data=val_flow, epochs=MAX_EPOCHS,
              class_weight=class_weight, callbacks=cbs)
    model.save(MODELS / "cnn_multispecies.h5")

    val_flow.reset()
    val_probs = model.predict(val_flow)
    y_val = val_flow.classes
    rep_paths = val["path"].sample(min(100, len(val)), random_state=42).tolist()
    rep_imgs = []
    for p in rep_paths:
        im = tf.keras.utils.img_to_array(tf.keras.utils.load_img(p, target_size=IMG_SIZE))
        rep_imgs.append(preprocess_input(im))
    shared_finish(model, val_probs, y_val[:len(val_probs)], rep_imgs, len(df))


def main():
    print("=" * 70)
    print("TASK 3 - MULTI-SPECIES MobileNetV2 CNN")
    print("=" * 70)
    df = build_folder_index()
    if not df.empty:
        print(f"Using image folders: {len(df)} images")
        run_folders(df)
    else:
        print("No local image folders found -> using TensorFlow Datasets PlantVillage")
        run_tfds()


if __name__ == "__main__":
    main()
