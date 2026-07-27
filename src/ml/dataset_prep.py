"""
Stage 1 & 2 of the ML pipeline: Data preprocessing and feature preparation
for the TensorFlow document-classification model.

Loads the labelled CSV dataset (text,label columns), cleans it, and produces
train/validation splits plus the sorted label vocabulary used consistently
across training and inference.
"""
import re
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset CSV must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def get_label_list(df: pd.DataFrame) -> List[str]:
    return sorted(df["label"].unique().tolist())


def prepare_train_val_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[List[str], List[str], List[int], List[int], List[str]]:
    labels = get_label_list(df)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    df["label_idx"] = df["label"].map(label_to_idx)

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label_idx"]
    )
    return (
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        train_df["label_idx"].tolist(),
        val_df["label_idx"].tolist(),
        labels,
    )
