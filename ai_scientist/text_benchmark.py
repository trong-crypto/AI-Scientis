"""Bounded, reproducible classification of labeled free-text records."""

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


MAX_TEXT_ROWS = 6000
MAX_TEXT_LENGTH = 10000
OUTER_FOLDS = 5
SEED = 42
PRIMARY_METRIC = "macro_f1"


def _prepare_text(frame, target, text_column):
    if target == text_column or target not in frame or text_column not in frame:
        raise ValueError("Text and target must be distinct, existing columns.")
    if not pd.api.types.is_string_dtype(frame[text_column]) and frame[text_column].dtype != object:
        raise ValueError("The selected text column must contain free-text strings.")
    data = frame[[text_column, target]].copy()
    if data.isna().any().any():
        raise ValueError("Text classification needs nonmissing text and labels.")
    data[text_column] = data[text_column].astype(str).str.strip()
    data[target] = data[target].astype(str).str.strip()
    if (data[text_column].str.len().eq(0).any() or data[target].eq("").any()
            or data[text_column].str.len().gt(MAX_TEXT_LENGTH).any()):
        raise ValueError("Text or labels are empty, or a record exceeds the text-length limit.")
    if data.groupby(text_column)[target].nunique().gt(1).any():
        raise ValueError("Identical text has conflicting labels.")
    before = len(data)
    data = data.drop_duplicates(subset=[text_column]).reset_index(drop=True)
    labels = data[target].value_counts()
    if not 30 <= len(data) <= MAX_TEXT_ROWS or not 2 <= len(labels) <= 10:
        raise ValueError("Text classification needs 30–6000 distinct records and 2–10 classes.")
    if labels.min() < OUTER_FOLDS:
        raise ValueError("Each class needs at least five distinct records for held-out folds.")
    if data[text_column].str.len().median() < 8:
        raise ValueError("The chosen column does not contain enough text for TF-IDF.")
    return data, before - len(data), labels.to_dict()


def _metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_text_benchmark(frame, dataset, design, workspace_dir):
    """Fit TF-IDF only on training folds and export every held-out prediction."""
    target, text_column = dataset["target"], design["text_column"]
    data, duplicates_removed, class_counts = _prepare_text(frame, target, text_column)
    texts = data[text_column].to_numpy()
    labels = data[target].to_numpy()
    models = {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "tfidf_logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      max_features=8000, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "tfidf_naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      max_features=8000, sublinear_tf=True)),
            ("classifier", MultinomialNB()),
        ]),
    }
    predictions = {name: np.empty(len(data), dtype=object) for name in models}
    fold_ids = np.empty(len(data), dtype=int)
    fold_results = []
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    for number, (train, test) in enumerate(splitter.split(texts, labels), 1):
        fold_ids[test] = number
        per_fold = {}
        for name, model in models.items():
            model.fit(texts[train], labels[train])
            predicted = model.predict(texts[test])
            predictions[name][test] = predicted
            per_fold[name] = _metrics(labels[test], predicted)
        fold_results.append({"fold": number, "metrics": per_fold})
    scores = {name: _metrics(labels, predicted) for name, predicted in predictions.items()}
    baseline = scores["dummy_baseline"][PRIMARY_METRIC]
    proposed = scores["tfidf_logistic_regression"][PRIMARY_METRIC]
    gain = proposed - baseline
    output_dir = Path(workspace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "experiment_dataset.csv"
    data.to_csv(data_file, index=False)
    predictions_file = output_dir / "experiment_predictions.csv"
    pd.DataFrame({"row": np.arange(1, len(data) + 1), "outer_fold": fold_ids,
                  "observed": labels, **predictions}).to_csv(predictions_file, index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(scores)
    ax.barh([name.replace("_", " ") for name in names[::-1]],
            [scores[name][PRIMARY_METRIC] for name in names[::-1]],
            color=["#47768e", "#28668a", "#87929d"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Out-of-fold macro F1 (higher is better)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "experiment_plot.png", dpi=180)
    finally:
        plt.close(fig)
    return {
        "status": "completed", "experiment_type": "text_classification",
        "dataset": {**dataset, "rows": len(data), "task": "text_classification",
                    "text_column": text_column, "features": [text_column],
                    "duplicates_removed": duplicates_removed, "class_counts": class_counts,
                    "file": data_file.name,
                    "sha256": hashlib.sha256(data_file.read_bytes()).hexdigest()},
        "metrics": scores, "primary_metric": PRIMARY_METRIC,
        "improvement_over_baseline": gain,
        "protocol": {"outer_folds": OUTER_FOLDS, "seed": SEED,
                     "sklearn_version": sklearn.__version__,
                     "feature_extraction": "character TF-IDF n-grams (3–5), maximum 8000 features",
                     "fold_results": fold_results},
        "plot_file": "experiment_plot.png",
        "predictions_file": predictions_file.name,
        "limitations": ("Exploratory performance on this labeled collection. "
                        "Stratified random folds assume independent messages and can "
                        "overstate generalization if senders or campaigns overlap across "
                        "folds. The result does not establish causal effects or global novelty."),
    }
