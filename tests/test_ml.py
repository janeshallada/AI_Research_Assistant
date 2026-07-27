import pandas as pd

from src.ml.dataset_prep import clean_text, load_dataset, get_label_list, prepare_train_val_split
from src.ml.predictor import DocumentClassifier


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, World! 123") == "hello world 123"


def test_load_dataset_from_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({
        "text": ["Deep learning paper about neural networks.", "Cloud native microservices architecture."],
        "label": ["Artificial Intelligence", "Cloud Computing"],
    }).to_csv(csv_path, index=False)

    df = load_dataset(str(csv_path))
    assert len(df) == 2
    assert set(get_label_list(df)) == {"Artificial Intelligence", "Cloud Computing"}


def test_prepare_train_val_split_shapes(tmp_path):
    csv_path = tmp_path / "data.csv"
    rows = []
    for label in ["Artificial Intelligence", "Cloud Computing"]:
        for i in range(6):
            rows.append({"text": f"{label} sample text number {i}", "label": label})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = load_dataset(str(csv_path))
    train_texts, val_texts, train_labels, val_labels, labels = prepare_train_val_split(df, test_size=0.3)
    assert len(train_texts) == len(train_labels)
    assert len(val_texts) == len(val_labels)
    assert len(labels) == 2


def test_classifier_predict_returns_none_when_untrained(tmp_path, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "model_path", str(tmp_path / "no_model.h5"))
    monkeypatch.setattr(settings, "labels_path", str(tmp_path / "no_labels.json"))
    DocumentClassifier._instance = None
    clf = DocumentClassifier.get_instance()
    category, confidence = clf.predict("some text")
    assert category is None
    assert confidence is None
