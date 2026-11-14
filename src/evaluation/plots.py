"""评估绘图模块：ROC / PR / 混淆矩阵 / 指标对比。

统一中文字体与配色，所有图保存到 reports/figures/。
配色沿用阶段一：正常=蓝、欺诈=红。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

NORMAL_C = "#3465A4"   # 正常/蓝
FRAUD_C = "#CC4444"    # 欺诈/红
MODEL_COLORS = {
    "Logistic Regression": "#3465A4",
    "Random Forest": "#7A9A3B",
    "XGBoost": "#CC4444",
}


def setup_style() -> None:
    """统一中文字体与负号显示（Microsoft YaHei 在 Windows 可用）。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_style("whitegrid", rc={"font.sans-serif": ["Microsoft YaHei"]})


def _save(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


def plot_roc_curves(
    probas: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    filename: str,
) -> Path:
    """多条 ROC 曲线。probas: {模型名: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_proba) in probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, lw=1.8,
                label=f"{name}  (AUC={roc_auc_score(y_true, y_proba):.4f})",
                color=MODEL_COLORS.get(name))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="随机基线")
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR (Recall)")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, filename)
    plt.close(fig)
    return FIGURE_DIR / filename


def plot_pr_curves(
    probas: dict[str, tuple[np.ndarray, np.ndarray]],
    baseline_rate: float,
    title: str,
    filename: str,
) -> Path:
    """多条 PR 曲线，含随机基线（=正样本率）。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_proba) in probas.items():
        p, r, _ = precision_recall_curve(y_true, y_proba)
        # 与 metrics.py 保持同一口径：average_precision_score（阶梯求和），
        # 而非 auc(r, p) 梯形法——两者数值不同，绝不允许同名指标两套算法
        ax.plot(r, p, lw=1.8,
                label=f"{name}  (PR-AUC={average_precision_score(y_true, y_proba):.4f})",
                color=MODEL_COLORS.get(name))
    ax.axhline(baseline_rate, color="gray", ls="--", lw=0.9,
               label=f"随机基线 (正样本率={baseline_rate:.4f})")
    ax.set_xlabel("Recall（查全率）")
    ax.set_ylabel("Precision（查准率）")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, filename)
    plt.close(fig)
    return FIGURE_DIR / filename


def plot_confusion_matrices(
    cms: dict[str, np.ndarray],
    title: str,
    filename: str,
) -> Path:
    """横向排布的混淆矩阵热力图。cms: {模型名: 2x2 ndarray}"""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        # 按行归一化显示比例，格子标注原始计数
        row = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
            annot_kws={"fontsize": 10}, ax=ax,
        )
        for (i, j), v in np.ndenumerate(row):
            ax.text(j + 0.5, i + 0.78, f"{v:.2%}", ha="center",
                    va="center", fontsize=8, color="#CC4444" if i == 1 and j == 0 else "#555555")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")
        ax.set_xticklabels(["正常(0)", "欺诈(1)"])
        ax.set_yticklabels(["正常(0)", "欺诈(1)"], rotation=0)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    _save(fig, filename)
    plt.close(fig)
    return FIGURE_DIR / filename


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric_cols: list[str],
    title: str,
    filename: str,
) -> Path:
    """分组柱状图对比各模型指标。metrics_df 需含"模型"列。

    配色按"指标"区分（同一指标在不同模型间同色），
    避免与模型数量不一致时颜色重复造成误读。
    """
    palette = ["#3465A4", "#7A9A3B", "#CC4444", "#8064A2", "#4BACC6", "#F28E2B"]
    df = metrics_df.set_index("模型")[metric_cols]
    ax = df.plot.bar(figsize=(9, 5), width=0.78, color=palette[: len(metric_cols)])
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("得分")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(df.index, rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=1.5)
    ax.legend(fontsize=8, ncol=len(metric_cols), loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    _save(fig=ax.figure, filename=filename)
    plt.close(ax.figure)
    return FIGURE_DIR / filename


def plot_threshold_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    filename: str,
) -> Path:
    """单模型在不同阈值下的 Precision/Recall 曲线（为阶段三铺垫）。"""
    p, r, th = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(th, p[:-1], color=NORMAL_C, lw=1.8, label="Precision")
    ax.plot(th, r[:-1], color=FRAUD_C, lw=1.8, label="Recall")
    ax.set_xlabel("判定阈值")
    ax.set_ylabel("得分")
    ax.set_title(f"{model_name}：阈值 - Precision/Recall 关系", fontsize=12)
    ax.legend(fontsize=9)
    _save(fig, filename)
    plt.close(fig)
    return FIGURE_DIR / filename
