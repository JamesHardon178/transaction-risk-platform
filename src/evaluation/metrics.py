"""评估指标模块。

约定：所有概率阈值默认 0.5。阈值寻优属于阶段三，
本阶段一律用默认阈值报告，保证三个模型可比。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


def evaluate_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """在给定阈值下计算全套指标。

    返回 dict：precision / recall / f1 / roc_auc / pr_auc /
    confusion_matrix(ndarray) / tn fp fn tp / n / n_pos。
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    # 只有真实标签同时含两类时 AUC 才有定义；单类时返回 nan 并注明
    if len(np.unique(y_true)) < 2:
        roc = pr = float("nan")
    else:
        roc = float(roc_auc_score(y_true, y_proba))
        pr = float(average_precision_score(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "confusion_matrix": cm,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def metrics_row(name: str, m: dict) -> dict:
    """把 evaluate_at_threshold 的 dict 压成一行表格记录。"""
    return {
        "模型": name,
        "Threshold": m["threshold"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
        "ROC-AUC": m["roc_auc"],
        "PR-AUC": m["pr_auc"],
        "TP": m["tp"],
        "FP": m["fp"],
        "FN": m["fn"],
        "TN": m["tn"],
    }
