"""绘图函数的冒烟测试（Agg 后端，输出写入临时目录）。"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import src.evaluation.plots as plots  # noqa: E402


@pytest.fixture(autouse=True)
def _redirect_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "FIGURE_DIR", tmp_path)
    plots.setup_style()
    yield


def _two_class_proba(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    proba = np.clip(y * 0.7 + rng.random(n) * 0.3, 0, 1)
    return y, proba


def test_plot_roc_curves_accepts_pair_dict(tmp_path):
    y, p = _two_class_proba()
    out = plots.plot_roc_curves({"M1": (y, p), "M2": (y, 1 - p)}, "t", "t_roc.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_pr_curves(tmp_path):
    y, p = _two_class_proba(seed=1)
    out = plots.plot_pr_curves({"M1": (y, p)}, baseline_rate=float(y.mean()), title="t", filename="t_pr.png")
    assert out.exists()


def test_plot_confusion_matrices_and_comparison(tmp_path):
    y, p = _two_class_proba(seed=2)
    from src.evaluation.metrics import evaluate_at_threshold

    m = evaluate_at_threshold(y, p)
    out = plots.plot_confusion_matrices({"M1": m["confusion_matrix"]}, "t", "t_cm.png")
    assert out.exists()
    df = pd.DataFrame([metrics_row_dict(m)])
    out2 = plots.plot_model_comparison(df, ["Precision", "Recall", "F1"], "t", "t_bar.png")
    assert out2.exists()


def metrics_row_dict(m):
    return {"模型": "M1", "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"]}


def test_plot_threshold_curves(tmp_path):
    y, p = _two_class_proba(seed=3)
    out = plots.plot_threshold_curves(y, p, "M1", "t_thr.png")
    assert out.exists()
