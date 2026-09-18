"""Evaluation metrics calculator for ticket triage models.

Computes accuracy, precision, recall, F1 (macro and weighted),
per-class classification report, and confusion matrix.
"""

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_classifier(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for multi-class classification."""
    acc = float(accuracy_score(y_true, y_pred))

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weight_prec, weight_rec, weight_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    class_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "model_name": model_name,
        "test_samples": len(y_true),
        "labels": list(labels),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weight_prec), 4),
        "weighted_recall": round(float(weight_rec), 4),
        "weighted_f1": round(float(weight_f1), 4),
        "classification_report": class_report,
        "confusion_matrix": cm.tolist(),
    }
