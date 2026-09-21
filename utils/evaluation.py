"""
Evaluation metrics for malware detection models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve,
    fbeta_score,
)
from typing import Dict, Any, Tuple
import time


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    train_time: float = 0.0,
    inference_time_ms: float = 0.0,
    model_size_mb: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a binary malware classifier.

    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Binary predictions
        y_proba: Probability estimates for class 1 (malware)
        model_name: Human-readable model name
        train_time: Training time in seconds
        inference_time_ms: Inference time per sample in milliseconds
        model_size_mb: Model file size in MB

    Returns:
        Dictionary of all metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n_total = len(y_true)

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # recall-weighted

    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    # Error rates
    fpr_overall = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_overall = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # Detection rate at 1% FPR
    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_proba)
    detection_at_1pct_fpr = _detection_rate_at_fpr(fpr_vals, tpr_vals, target_fpr=0.01)

    return {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "f2_score": round(f2, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "false_positive_rate": round(fpr_overall, 4),
        "false_negative_rate": round(fnr_overall, 4),
        "mcc": round(mcc, 4),
        "detection_at_1pct_fpr": round(detection_at_1pct_fpr, 4),
        "train_time_s": round(train_time, 2),
        "inference_time_ms": round(inference_time_ms, 4),
        "model_size_mb": round(model_size_mb, 3),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def _detection_rate_at_fpr(
    fpr_vals: np.ndarray,
    tpr_vals: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Interpolate TPR (detection rate) at a given FPR threshold."""
    if len(fpr_vals) < 2:
        return 0.0
    return float(np.interp(target_fpr, fpr_vals, tpr_vals))


def get_roc_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_proba)


def get_pr_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_true, y_proba)


def benchmark_inference_time(
    model,
    X_test: np.ndarray,
    n_repeats: int = 5,
) -> float:
    """
    Measure average inference time per sample in milliseconds.

    Args:
        model: Fitted sklearn-compatible model.
        X_test: Test features.
        n_repeats: Number of timing repetitions.

    Returns:
        Inference time in ms per sample.
    """
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = model.predict_proba(X_test)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg_time = np.mean(times)
    return (avg_time / len(X_test)) * 1000  # ms per sample


def metrics_to_dataframe(metrics_list: list) -> pd.DataFrame:
    """Convert a list of metric dicts to a comparison DataFrame."""
    df = pd.DataFrame(metrics_list)
    if "model_name" in df.columns:
        df = df.set_index("model_name")
    return df


def highlight_best_worst(df: pd.DataFrame) -> Any:
    """
    Return a styled DataFrame where best (green) and worst (red) cells are highlighted
    per metric column. Handles both maximize and minimize metrics.
    """
    minimize_metrics = {"false_positive_rate", "false_negative_rate",
                        "train_time_s", "inference_time_ms", "model_size_mb"}

    def _style_col(col):
        if col.name in minimize_metrics:
            best_idx = col.idxmin()
            worst_idx = col.idxmax()
        else:
            best_idx = col.idxmax()
            worst_idx = col.idxmin()
        styles = [""] * len(col)
        for i, idx in enumerate(col.index):
            if idx == best_idx:
                styles[i] = "background-color: #10B981; color: #000"
            elif idx == worst_idx:
                styles[i] = "background-color: #EF4444; color: #fff"
        return styles

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df.style.apply(_style_col, axis=0, subset=numeric_cols)
