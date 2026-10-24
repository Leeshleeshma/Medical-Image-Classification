"""Shared data code for Group 5 (ICSI 535): chest X-ray pneumonia project.

Everyone loads the same images, the same splits and the same preprocessing from here,
so the dense network, the CNN and the ResNet are compared fairly.

    from xray_data import load_split, make_dataset

    X_train, y_train = load_split("train", PROCESSED_DIR, model="cnn")         # numpy arrays
    train_ds = make_dataset("train", PROCESSED_DIR, model="resnet", shuffle=True)  # tf.data pipeline

Splits are "train", "validation" and "test". Labels: 0 = NORMAL, 1 = PNEUMONIA.
"""
import re
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 224
SEED = 42
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# 1. Preprocessing: grayscale, pad to a square with black bars, resize to 224 x 224
# ---------------------------------------------------------------------------
def pad_and_resize(img, size=IMAGE_SIZE):
    """Grayscale PIL image -> size x size, keeping the shape (no stretching)."""
    img = img.convert("L")
    width, height = img.size
    side = max(width, height)
    square = Image.new("L", (side, side), 0)                      # black canvas
    square.paste(img, ((side - width) // 2, (side - height) // 2))  # X-ray in the middle
    return square.resize((size, size), Image.Resampling.LANCZOS)


def stretch_resize(img, size=IMAGE_SIZE):
    """The old way (squashes the image). Kept only for before/after pictures."""
    return img.convert("L").resize((size, size), Image.Resampling.LANCZOS)


def load_image(path, size=IMAGE_SIZE):
    with Image.open(path) as img:
        return np.asarray(pad_and_resize(img, size), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 2. Patient IDs from Kaggle file names
# ---------------------------------------------------------------------------
# PNEUMONIA: person1_bacteria_1.jpeg, person1_virus_6.jpeg      -> patient "person1"
# NORMAL:    IM-0115-0001.jpeg                                  -> patient "IM-0115"
#            NORMAL2-IM-0927-0001.jpeg (also ...-0001-0002)     -> patient "NORMAL2-IM-0927"
PATIENT_PATTERNS = [
    re.compile(r"^(person\d+)_(?:bacteria|virus)_\d+", re.IGNORECASE),
    re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+(?:-\d+)*$", re.IGNORECASE),
]


def patient_id_from_filename(filename):
    """Returns (patient_id, found). If no pattern matches, the file is its own patient."""
    stem = Path(filename).stem
    for pattern in PATIENT_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group(1), True
    return stem, False


# ---------------------------------------------------------------------------
# 3. Loading the saved splits (same .npz format as data/processed_224_gray)
# ---------------------------------------------------------------------------
def load_npz(name, processed_dir):
    """Raw uint8 images (N, 224, 224) and int64 labels, exactly as saved."""
    with np.load(Path(processed_dir) / f"{name}.npz") as data:
        return data["images"], data["labels"].astype(np.int64)


def _shape_for_model(images, model):
    """images: float32 (N, 224, 224) scaled 0-1."""
    if model == "dense":
        return images                                   # (N, 224, 224), like 02_dense_baseline
    if model == "cnn":
        return images[..., np.newaxis]                  # (N, 224, 224, 1), like 03_cnn_from_scratch
    raise ValueError("model must be 'dense', 'cnn' or 'resnet'")


def load_split(name, processed_dir, model="cnn"):
    """Numpy arrays for model.fit(): pixels 0-1, shaped for 'dense' or 'cnn'.

    For the ResNet use make_dataset(..., model="resnet"): 3-channel float copies of
    every image would use several GB of memory, so the pipeline converts batch by batch.
    """
    if model == "resnet":
        raise ValueError("Use make_dataset(name, processed_dir, model='resnet') for the ResNet.")
    images, labels = load_npz(name, processed_dir)
    return _shape_for_model(images.astype(np.float32) / 255.0, model), labels


def make_dataset(name, processed_dir, model="cnn", batch_size=32, shuffle=False,
                 seed=SEED, resnet_preprocess="resnet50"):
    """tf.data pipeline of (image, label) batches.

    model="dense"  -> (224, 224), pixels 0-1
    model="cnn"    -> (224, 224, 1), pixels 0-1
    model="resnet" -> (224, 224, 3): the gray channel copied 3 times, then ImageNet
                      normalization. resnet_preprocess="resnet50" for
                      keras.applications.ResNet50, "resnet50v2" for ResNet50V2.
    """
    import tensorflow as tf

    images, labels = load_npz(name, processed_dir)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)

    if model == "resnet":
        if resnet_preprocess == "resnet50":
            from keras.applications.resnet50 import preprocess_input
        elif resnet_preprocess == "resnet50v2":
            from keras.applications.resnet_v2 import preprocess_input
        else:
            raise ValueError("resnet_preprocess must be 'resnet50' or 'resnet50v2'")

        def to_input(x, y):
            x = tf.image.grayscale_to_rgb(tf.cast(x, tf.float32)[..., tf.newaxis])  # 0-255, 3 channels
            return preprocess_input(x), y
    elif model in ("dense", "cnn"):
        def to_input(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            return (x if model == "dense" else x[..., tf.newaxis]), y
    else:
        raise ValueError("model must be 'dense', 'cnn' or 'resnet'")

    ds = ds.batch(batch_size).map(to_input, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights(labels):
    """Balanced class weights, the same formula 03_cnn_from_scratch uses."""
    counts = np.bincount(labels, minlength=2)
    return {i: float(len(labels) / (2 * counts[i])) for i in range(2)}
