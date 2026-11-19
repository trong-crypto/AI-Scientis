"""Example starting experiment using scikit-learn's built-in Iris dataset."""

import json
import os

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
)
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

results = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred, average="macro")),
}
path = os.environ.get("AI_SCIENTIST_RESULTS_PATH", "results.json")
with open(path, "w", encoding="utf-8") as output:
    json.dump(results, output, indent=2)
