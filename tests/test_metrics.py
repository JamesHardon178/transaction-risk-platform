"""评估指标模块的单元测试（手工可验算的小样本）。"""

import numpy as np
import pytest

from src.evaluation.metrics import evaluate_at_threshold, metrics_row


def test_perfect_classifier():
    y_true = [0, 0, 0, 1, 1]
    y_proba = [0.05, 0.1, 0.2, 0.9, 0.8]
    m = evaluate_at_threshold(y_true, y_proba, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 3)


def test_known_confusion_case():
    # 手工验算：阈值 0.5 时预测 [0,1,1,0,1]
    y_true = [0, 0, 1, 1, 1]
    y_proba = [0.2, 0.9, 0.9, 0.1, 0.7]
    m = evaluate_at_threshold(y_true, y_proba, threshold=0.5)
    # tp=2 (第3、5), fp=1 (第2), fn=1 (第4), tn=1 (第1)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_all_negative_predictions_zero_division_safe():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.4]
    m = evaluate_at_threshold(y_true, y_proba)
    assert m["tp"] == 0 and m["fp"] == 0
    assert m["precision"] == 0.0  # zero_division=0
    assert m["recall"] == 0.0


def test_single_class_auc_is_nan():
    m = evaluate_at_threshold([1, 1, 1], [0.2, 0.8, 0.6])
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])


def test_metrics_row_keys():
    m = evaluate_at_threshold([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    row = metrics_row("M", m)
    for k in ["模型", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "TP", "FP", "FN", "TN"]:
        assert k in row
