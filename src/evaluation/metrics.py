import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.models import TemperatureScaler


METRIC_COLS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "precision",
    "malicious_recall",
    "auprc",
    "auroc",
    "ece",
    "brier_score",
    "fa_10k",
    "false_positives",
    "benign_count",
]
RESOURCE_COLS = ["model_size_mb", "peak_rss_mb", "latency_per_flow_sec", "throughput_flows_sec"]


def top_label_ece(y_true, y_proba, n_bins=15):
    probs = np.asarray(y_proba, dtype=float)
    y_true = np.asarray(y_true)
    if probs.ndim != 2 or probs.shape[0] != len(y_true) or len(y_true) == 0:
        return math.nan

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        right = conf <= bins[idx + 1] if idx == n_bins - 1 else conf < bins[idx + 1]
        mask = (conf >= bins[idx]) & right
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def brier_score(y_true, y_proba):
    probs = np.asarray(y_proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if probs.ndim != 2 or probs.shape[0] != len(y_true):
        return math.nan
    probs = np.clip(probs, 1e-12, 1.0)
    if probs.shape[1] == 2:
        return float(np.mean((probs[:, 1] - y_true) ** 2))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def safe_metric(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return math.nan


def compute_metrics(y_true, y_pred, y_proba, task="binary"):
    row = {
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "macro_f1": safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": safe_metric(f1_score, y_true, y_pred, average="weighted", zero_division=0),
        "mcc": safe_metric(matthews_corrcoef, y_true, y_pred),
        "ece": top_label_ece(y_true, y_proba),
        "brier_score": brier_score(y_true, y_proba),
    }

    probs = np.asarray(y_proba)
    if task == "binary" and probs.ndim == 2 and probs.shape[1] >= 2:
        benign = int((np.asarray(y_true) == 0).sum())
        false_positives = int(((np.asarray(y_pred) == 1) & (np.asarray(y_true) == 0)).sum())
        row.update(
            {
                "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
                "malicious_recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
                "auprc": safe_metric(average_precision_score, y_true, probs[:, 1]),
                "auroc": safe_metric(roc_auc_score, y_true, probs[:, 1])
                if len(np.unique(y_true)) > 1
                else math.nan,
                "fa_10k": (false_positives / benign) * 10000 if benign > 0 else math.nan,
                "false_positives": false_positives,
                "benign_count": benign,
            }
        )
    else:
        row.update(
            {
                "precision": math.nan,
                "malicious_recall": math.nan,
                "auprc": safe_metric(average_precision_score, y_true, probs, average="macro"),
                "auroc": safe_metric(roc_auc_score, y_true, probs, multi_class="ovr", average="macro")
                if len(np.unique(y_true)) > 1
                else math.nan,
                "fa_10k": math.nan,
                "false_positives": math.nan,
                "benign_count": math.nan,
            }
        )
    return row


def make_two_col_probs(p1):
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-8, 1 - 1e-8)
    return np.vstack([1 - p1, p1]).T


def fit_binary_calibrators(probs_val, y_val):
    p_val = np.clip(probs_val[:, 1], 1e-8, 1 - 1e-8)
    calibrators = {"uncalibrated": lambda probs: probs}
    try:
        platt = LogisticRegression(solver="lbfgs")
        platt.fit(p_val.reshape(-1, 1), y_val)
        calibrators["platt_sigmoid"] = lambda probs, platt=platt: make_two_col_probs(
            platt.predict_proba(np.clip(probs[:, 1], 1e-8, 1 - 1e-8).reshape(-1, 1))[:, 1]
        )
    except Exception as exc:
        print("Platt failed:", exc)
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val, y_val)
        calibrators["isotonic"] = lambda probs, iso=iso: make_two_col_probs(
            iso.transform(np.clip(probs[:, 1], 1e-8, 1 - 1e-8))
        )
    except Exception as exc:
        print("Isotonic failed:", exc)
    try:
        temp = TemperatureScaler().fit(probs_val, y_val)
        calibrators["temperature"] = lambda probs, temp=temp: temp.transform(probs)
    except Exception as exc:
        print("Temperature failed:", exc)
    return calibrators


def apply_threshold_policy(probs, threshold):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    low = conf < threshold
    pred2 = pred.copy()
    pred2[low] = 0
    coverage = float((~low).mean())
    return pred2, coverage
