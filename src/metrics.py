"""Evaluation and reporting helpers for binary classification models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def threshold_search(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    start: float = 0.35,
    stop: float = 0.651,
    step: float = 0.001,
) -> tuple[float, float, pd.DataFrame]:
    """Find the probability threshold with the best accuracy.

    The default search grid matches the LightGBM notebook: thresholds from
    0.350 through 0.650 inclusive, using a 0.001 increment.
    """
    thresholds = np.arange(start, stop, step)
    scores = []

    for threshold in thresholds:
        pred_label = (probabilities >= threshold).astype(int)
        scores.append(accuracy_score(y_true, pred_label))

    scores_array = np.array(scores)
    best_idx = int(scores_array.argmax())
    best_threshold = float(thresholds[best_idx])
    best_score = float(scores_array[best_idx])

    results = pd.DataFrame({"threshold": thresholds, "accuracy": scores_array})
    return best_threshold, best_score, results


def print_classification_results(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    title: str,
    target_names: list[str] | None = None,
) -> None:
    """Print accuracy, classification report, and confusion matrix."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.5f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
