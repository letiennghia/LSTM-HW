"""Consistent binary metrics and persisted final evaluation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def compute_metrics(truth, predicted) -> dict:
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "precision": float(precision_score(truth, predicted, pos_label=1, zero_division=0)),
            "recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
            "f1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist()}


def export_test(directory: Path, frame, predicted, scores, score_type: str) -> dict:
    from .visualization import plot_confusion
    predicted = np.asarray(predicted)
    if len(predicted) != len(frame) or len(scores) != len(frame):
        raise ValueError("Prediction count does not match test reviews")
    metrics = compute_metrics(frame.label, predicted)
    save_json(directory / "test_metrics.json", metrics)
    pd.DataFrame({"review_id": frame.review_id, "true_label": frame.label, "predicted_label": predicted,
                  "score": scores, "score_type": score_type}).to_csv(directory / "test_predictions.csv", index=False)
    report = classification_report(frame.label, predicted, labels=[0, 1], target_names=["Negative", "Positive"], zero_division=0)
    (directory / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(metrics["confusion_matrix"], directory / "confusion_matrix.png")
    return metrics
