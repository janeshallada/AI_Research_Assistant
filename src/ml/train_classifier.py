"""
Stage 3, 4 & 5 of the ML pipeline: model architecture, training, evaluation,
and persistence for the document domain classifier.

Architecture
------------
TextVectorization -> Embedding -> GlobalAveragePooling1D -> Dense(128, relu)
-> Dropout(0.3) -> Dense(num_classes, softmax)

- TextVectorization is baked into the saved model so raw strings can be fed
  directly at inference time (no separate tokenizer object is required to
  reproduce vectorization, though we additionally persist the vocabulary via
  pickle as a fallback/inspection artifact, per the required project layout).
- Dropout(0.3) is used for regularization to reduce overfitting given the
  relatively small labelled dataset typical of a course/demo project.
- Loss: sparse_categorical_crossentropy (labels are integer-encoded).
- Optimizer: adam.

Usage:
    python -m src.ml.train_classifier
"""
import json
import logging
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from config.settings import settings
from src.ml.dataset_prep import load_dataset, prepare_train_val_split

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 64


def build_model(train_texts, num_classes: int) -> tf.keras.Model:
    vectorize_layer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LEN,
    )
    vectorize_layer.adapt(train_texts)

    model = models.Sequential([
        layers.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, vectorize_layer


def train_and_save(csv_path: str, epochs: int = 40, batch_size: int = 8) -> dict:
    df = load_dataset(csv_path)
    train_texts, val_texts, train_labels, val_labels, labels = prepare_train_val_split(df)

    num_classes = len(labels)
    model, vectorize_layer = build_model(train_texts, num_classes)

    train_texts_t = tf.constant(train_texts, dtype=tf.string)
    val_texts_t = tf.constant(val_texts, dtype=tf.string)
    train_labels_a = np.array(train_labels)
    val_labels_a = np.array(val_labels)

    history = model.fit(
        train_texts_t,
        train_labels_a,
        validation_data=(val_texts_t, val_labels_a),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    eval_loss, eval_acc = model.evaluate(val_texts_t, val_labels_a, verbose=0)
    logger.info("Validation accuracy: %.4f | loss: %.4f", eval_acc, eval_loss)

    # --- Persistence (Stage 5) ---
    model.save(settings.model_path)

    vocabulary = vectorize_layer.get_vocabulary()
    with open(settings.tokenizer_path, "wb") as f:
        pickle.dump({"vocabulary": vocabulary}, f)

    with open(settings.labels_path, "w") as f:
        json.dump(labels, f)

    return {
        "labels": labels,
        "val_accuracy": float(eval_acc),
        "val_loss": float(eval_loss),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }


if __name__ == "__main__":
    result = train_and_save(csv_path="./data/dataset/sample_training_data.csv")
    print(json.dumps(result, indent=2))
