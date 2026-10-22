from pathlib import Path
import csv
import json
import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflowjs as tfjs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = ROOT / "docs"
EVAL_DIR = DOCS / "evaluation"
MODEL_DIR = DOCS / "model"
CLASSES = ["peacock", "sparrow"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", "5"))

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def count_images(folder: Path) -> int:
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in allowed) if folder.exists() else 0

def load_dataset(directory: Path, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

def main():
    train_dir = DATA / "train"
    for name in CLASSES:
        if count_images(train_dir / name) < 2:
            raise SystemExit(
                f"Need at least 2 images in data/train/{name}/. "
                "Add labelled images on GitHub and rerun the workflow."
            )

    DOCS.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Reserve 20% of the training folder for validation.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="validation",
    )
    fit_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="training",
    )

    autotune = tf.data.AUTOTUNE
    # Simple augmentation happens only during training, outside the exported model,
    # keeping browser conversion compatible with TensorFlow.js.
    fit_ds = fit_ds.map(
        lambda images, labels: (tf.image.random_flip_left_right(images), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    history = model.fit(
        fit_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Prefer a genuinely separate test folder. Otherwise report validation-only.
    test_dir = DATA / "test"
    test_counts = {name: count_images(test_dir / name) for name in CLASSES}
    has_test = all(test_counts[name] >= 1 for name in CLASSES)
    if has_test:
        eval_ds = load_dataset(test_dir, shuffle=False)
        eval_kind = "Independent test set (data/test)"
    else:
        eval_ds = val_ds
        eval_kind = "Validation-only (data/test needs images from both classes)"

    y_true = np.concatenate([labels.numpy() for _, labels in eval_ds], axis=0)
    probs = model.predict(eval_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASSES,
        output_dict=True, zero_division=0
    )

    with (EVAL_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({
            "evaluation_type": eval_kind,
            "accuracy": accuracy,
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "evaluated_images": int(len(y_true)),
            "classes": CLASSES,
            "confusion_matrix": cm.tolist(),
            "train_images": {name: count_images(train_dir / name) for name in CLASSES},
            "test_images": test_counts,
            "note": "Metrics describe this dataset only; they are not a guarantee on all real-world photos."
        }, f, indent=2)

    with (EVAL_DIR / "classification_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for name in CLASSES:
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"], row["f1-score"], row["support"]])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion-matrix.png", dpi=150)
    plt.close(fig)

    # Export a browser-compatible model. The preprocessing layer is included.
    if MODEL_DIR.exists():
        for p in MODEL_DIR.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                import shutil
                shutil.rmtree(p)
    tfjs.converters.save_keras_model(model, str(MODEL_DIR))

    # Human-readable summary for the Pages website.
    summary = {
        "evaluation_type": eval_kind,
        "accuracy": accuracy,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "evaluated_images": int(len(y_true)),
        "classes": CLASSES,
    }
    with (EVAL_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"TensorFlow.js model exported to {MODEL_DIR}")

if __name__ == "__main__":
    main()
