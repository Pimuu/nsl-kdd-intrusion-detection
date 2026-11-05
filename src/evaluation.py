"""Metric computation.

Binary task (positive class = attack)
-------------------------------------
accuracy, precision, recall (= detection rate), F1, ROC-AUC, and the false
alarm rate FPR = FP / (FP + TN). In an operational IDS the false-alarm rate is
what analysts feel day to day, so it is reported explicitly.

Multi-class task (normal / DoS / Probe / R2L / U2R)
---------------------------------------------------
accuracy, macro-averaged precision / recall / F1 (each class counts equally,
so rare-but-severe U2R/R2L are not drowned out by DoS), weighted F1, macro
one-vs-rest ROC-AUC, and per-class recall.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config


def binary_metrics(y_true, y_pred, y_score) -> dict:
    """Metrics for normal(0) vs attack(1). ``y_score`` = P(attack)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # ROC-AUC is undefined for a constant score (majority baseline) -> 0.5.
    auc = roc_auc_score(y_true, y_score) if np.ptp(y_score) > 0 else 0.5
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def multiclass_metrics(y_true, y_pred, y_proba, classes) -> dict:
    """Metrics for the 5-way task. ``y_proba`` columns follow ``classes``."""
    labels = config.CATEGORY_ORDER
    # Re-order probability columns to the canonical class order.
    proba = np.zeros((len(y_true), len(labels)))
    for j, c in enumerate(classes):
        proba[:, labels.index(c)] = y_proba[:, j]
    # One-vs-rest ROC-AUC computed class by class (macro = unweighted mean over
    # the classes present in y_true). Constant scores (majority baseline) -> 0.5.
    y_true = np.asarray(y_true)
    aucs = []
    for k, c in enumerate(labels):
        pos = y_true == c
        if pos.all() or not pos.any():
            continue
        aucs.append(roc_auc_score(pos, proba[:, k]) if np.ptp(proba[:, k]) > 0 else 0.5)
    auc = float(np.mean(aucs)) if aucs else float("nan")
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "macro_roc_auc_ovr": auc,
    }
    per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for c, r in zip(labels, per_class):
        out[f"recall_{c}"] = r
    return out


def confusion(y_true, y_pred, labels) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
