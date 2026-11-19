"""Nested cross-validation adapted from PTHTTM's tabular experiment."""

import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
PRIMARY_METRIC = {"classification": "macro_f1", "regression": "mae"}


def _models(task):
    if task == "classification":
        return DummyClassifier(strategy="prior"), [
            ("logistic_regression", LogisticRegression(max_iter=1500, class_weight="balanced"),
             {"estimator__C": [0.1, 1, 10]}),
            ("random_forest", RandomForestClassifier(random_state=SEED, class_weight="balanced"),
             {"estimator__n_estimators": [50, 100]}),
            ("extra_trees", ExtraTreesClassifier(random_state=SEED, class_weight="balanced"),
             {"estimator__n_estimators": [50, 100]}),
        ]
    return DummyRegressor(strategy="mean"), [
        ("ridge", Ridge(), {"estimator__alpha": [0.1, 1, 10]}),
        ("random_forest", RandomForestRegressor(random_state=SEED),
         {"estimator__n_estimators": [50, 100]}),
        ("extra_trees", ExtraTreesRegressor(random_state=SEED),
         {"estimator__n_estimators": [50, 100]}),
    ]


def _prepare(frame, target, features, task):
    # Deduplicate complete source records, before selecting model features.
    # Discrete predictors can legitimately be identical for different websites.
    initial_size = len(frame)
    source = frame.drop_duplicates().reset_index(drop=True)
    data = source[[target, *features]].copy()
    if data[target].isna().any():
        raise ValueError("The target has missing values.")
    if len(data) < 30:
        raise ValueError("At least 30 distinct observations are required.")
    y = data.pop(target)
    groups = pd.factorize(pd.util.hash_pandas_object(data, index=False))[0]
    if len(np.unique(groups)) < 30:
        raise ValueError("The selected predictors distinguish fewer than 30 independent feature patterns.")
    if y.nunique() < 2:
        raise ValueError("The target must have at least two values.")
    if task == "classification":
        if y.nunique() > 50 or y.value_counts().min() < OUTER_FOLDS:
            raise ValueError("Classification needs 2–50 labels and at least five rows per label.")
        if min(len(np.unique(groups[y.eq(label).to_numpy()])) for label in y.unique()) < OUTER_FOLDS:
            raise ValueError("Each class needs at least five independent predictor patterns.")
        numeric_labels = pd.to_numeric(y, errors="coerce")
        y = (numeric_labels if numeric_labels.notna().all()
             and numeric_labels.nunique() == y.nunique() else y.astype(str))
    else:
        y = pd.to_numeric(y).astype(float)
        if not np.isfinite(y).all():
            raise ValueError("Regression requires a finite numeric target.")

    numeric = data.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    if any(data[column].isna().all() for column in features):
        raise ValueError("An approved predictor column is entirely missing.")
    if numeric:
        data[numeric] = data[numeric].astype(float)
        if np.isinf(data[numeric].to_numpy()).any():
            raise ValueError("A predictor contains infinite values.")
    for column in categorical:
        if data[column].nunique() > 100:
            raise ValueError("A categorical predictor exceeds 100 distinct values.")
        data[column] = data[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return data, y, numeric, categorical, initial_size - len(source), groups


def _metrics(task, observed, predicted):
    if task == "classification":
        return {"accuracy": float(accuracy_score(observed, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
                "macro_f1": float(f1_score(observed, predicted, average="macro", zero_division=0))}
    return {"mae": float(mean_absolute_error(observed, predicted)),
            "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
            "r2": float(r2_score(observed, predicted))}


def run_benchmark(frame, dataset, design, workspace_dir):
    """Evaluate the baseline and inner-CV model selection on shared outer folds."""
    task, target, features = design["task"], dataset["target"], design["features"]
    data, y, numeric, categorical, duplicate_count, groups = _prepare(
        frame, target, features, task)
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent",
                                                           keep_empty_features=True)),
                                  ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    classification = task == "classification"
    outer = (StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
             if classification else KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED))
    scoring = "f1_macro" if classification else "neg_mean_absolute_error"
    baseline, candidates = _models(task)
    observed = y.to_numpy()
    names = ["dummy_baseline", *(name for name, _, _ in candidates), "nested_selection"]
    predictions = {name: np.empty_like(observed) for name in names}
    fold_ids = np.empty(len(y), dtype=int)
    fold_results = []
    outer_splits = outer.split(data, y, groups) if classification else outer.split(data, y)
    for fold, (train, test) in enumerate(outer_splits, start=1):
        x_train, x_test, y_train = data.iloc[train], data.iloc[test], y.iloc[train]
        inner = (StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                     random_state=SEED)
                 if classification else KFold(n_splits=INNER_FOLDS, shuffle=True,
                                              random_state=SEED))
        inner_splits = list(inner.split(x_train, y_train, groups[train])
                            if classification else inner.split(x_train, y_train))
        fold_ids[test] = fold
        baseline.fit(x_train, y_train)
        predictions["dummy_baseline"][test] = baseline.predict(x_test)
        best_score = -np.inf
        tuning = {}
        for name, estimator, grid in candidates:
            search = GridSearchCV(Pipeline([("preprocess", preprocessing), ("estimator", estimator)]),
                                  grid, scoring=scoring, cv=inner_splits, error_score="raise")
            search.fit(x_train, y_train)
            predicted = search.predict(x_test)
            predictions[name][test] = predicted
            tuning[name] = {"score": float(search.best_score_), "parameters": search.best_params_}
            if search.best_score_ > best_score:
                best_score = search.best_score_
                selected = name
                predictions["nested_selection"][test] = predicted
        fold_results.append({
            "fold": fold, "selected_model": selected, "tuning": tuning,
            "metrics": {name: _metrics(task, observed[test], predictions[name][test])
                        for name in names},
        })

    metrics = {name: _metrics(task, observed, predictions[name]) for name in names}
    primary = PRIMARY_METRIC[task]
    baseline_value = metrics["dummy_baseline"][primary]
    selected_value = metrics["nested_selection"][primary]
    gain = (selected_value - baseline_value) if classification else (baseline_value - selected_value)
    if not all(np.isfinite(value) for scores in metrics.values() for value in scores.values()):
        raise ValueError("The benchmark produced a non-finite metric.")

    # Save the exact input table (including order) for reproducibility.
    data_file = "experiment_dataset.csv"
    data_path = Path(workspace_dir, data_file)
    export = data.copy()
    export[target] = y
    export.to_csv(data_path, index=False)
    output = pd.DataFrame({"row": np.arange(1, len(y) + 1), "outer_fold": fold_ids,
                           "feature_group": groups,
                           "observed": observed, **predictions})
    output.to_csv(Path(workspace_dir, "experiment_predictions.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        labels = [name.replace("_", " ") for name in names]
        ax.barh(labels[::-1], [metrics[name][primary] for name in names[::-1]],
                color=["#28668a"] * (len(names) - 1) + ["#788796"])
        ax.set_xlabel(primary.replace("_", " ") + (" (higher is better)" if classification
                                                   else " (lower is better)"))
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(workspace_dir, "experiment_plot.png"), dpi=180)
    finally:
        plt.close(fig)
    return {
        "status": "completed", "experiment_type": "tabular_benchmark",
        "dataset": {**dataset, "rows": len(y), "features": features, "task": task,
                    "duplicates_removed": duplicate_count, "file": data_file,
                    "class_counts": ({str(label): int(count) for label, count in
                                      y.value_counts().items()} if classification else {}),
                    "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()},
        "metrics": metrics, "primary_metric": primary, "improvement_over_baseline": gain,
        "protocol": {"outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "seed": SEED,
                     "split_strategy": ("stratified groups of identical predictor patterns"
                                        if classification else "shuffled row folds"),
                     "sklearn_version": sklearn.__version__, "fold_results": fold_results},
        "plot_file": "experiment_plot.png", "predictions_file": "experiment_predictions.csv",
        "limitations": ("Exploratory predictive performance on this dataset. "
                        + ("Grouped folds keep identical predictor patterns together but "
                           "cannot resolve unobserved source or time dependence. "
                           if classification else "Random folds assume independent observations. ")
                        + "The comparison does not establish causality, external validity "
                        "or scientific novelty."),
    }
