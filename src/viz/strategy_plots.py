"""策略可视化模块（阶段三）。

复用 ``src.evaluation.plots`` 的字体/配色/落盘逻辑，保证阶段一~三的图表风格一致。
所有函数返回图片路径，不强制 matplotlib 后端（便于 Notebook 内嵌与看板导出）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.plots import FRAUD_C, NORMAL_C, _save, setup_style

STRATEGY_COLORS = {
    "保守策略": "#CC4444",
    "平衡策略": "#7A9A3B",
    "激进策略": "#3465A4",
}


def plot_threshold_metric_curves(
    scan: pd.DataFrame,
    strategies: pd.DataFrame | None = None,
    title: str = "阈值 - 指标变化曲线",
    filename: str = "fig16_threshold_curves.png",
) -> Path:
    """Precision / Recall / F1 随阈值变化，并标注三策略所在位置。"""
    setup_style()
    d = scan.sort_values("threshold")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(d["threshold"], d["precision"], color=NORMAL_C, lw=1.8, label="Precision")
    ax.plot(d["threshold"], d["recall"], color=FRAUD_C, lw=1.8, label="Recall")
    ax.plot(d["threshold"], d["f1"], color="#7A9A3B", lw=1.8, ls="--", label="F1")

    if strategies is not None:
        for _, r in strategies.iterrows():
            ax.axvline(r["阈值"], color=STRATEGY_COLORS.get(r["策略名称"], "gray"),
                       ls=":", lw=1.2, alpha=0.9)
            ax.annotate(f"{r['策略名称']}\n{r['阈值']:.3f}",
                        xy=(r["阈值"], 0.04), xytext=(4, 0),
                        textcoords="offset points", fontsize=8,
                        color=STRATEGY_COLORS.get(r["策略名称"], "gray"),
                        rotation=90, va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("判定阈值")
    ax.set_ylabel("得分")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="center right")
    out = _save(fig, filename)
    plt.close(fig)
    return out


def plot_cost_curve(
    curves: dict[str, pd.DataFrame],
    best: tuple[str, float, float] | None = None,
    title: str = "阈值 - 总成本曲线",
    filename: str = "fig17_cost_curve.png",
) -> Path:
    """多成本假设下的 阈值-总成本 曲线。curves: {图例: DataFrame(含 threshold/cost)}"""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, df in curves.items():
        d = df.sort_values("threshold")
        ax.plot(d["threshold"], d["cost"], lw=1.8, label=label)

    if best is not None:
        label, th, cost = best
        ax.scatter([th], [cost], s=70, color=FRAUD_C, zorder=5,
                   label=f"最优 {label}：阈值 {th:.3f}，成本 {cost:,.0f}")

    ax.set_xlim(0, 1)
    ax.set_xlabel("判定阈值")
    ax.set_ylabel("实验假设总成本")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    out = _save(fig, filename)
    plt.close(fig)
    return out


def plot_strategy_comparison(
    strategies: pd.DataFrame,
    title: str = "三策略指标与成本对比",
    filename: str = "fig18_strategy_comparison.png",
) -> Path:
    """左：Precision/Recall/F1；右：总成本（双轴分开画，避免量纲混淆）。"""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    x = np.arange(len(strategies))
    w = 0.26
    for i, (col, color) in enumerate([("Precision", NORMAL_C), ("Recall", FRAUD_C), ("F1", "#7A9A3B")]):
        bars = ax.bar(x + (i - 1) * w, strategies[col], width=w, label=col, color=color, alpha=0.88)
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies["策略名称"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("得分")
    ax.set_title("分类指标", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper center")

    ax = axes[1]
    bars = ax.bar(strategies["策略名称"], strategies["总成本"],
                  color=[STRATEGY_COLORS.get(s, "#888") for s in strategies["策略名称"]])
    ax.bar_label(bars, fmt="%.0f", fontsize=8.5, padding=2)
    ax.set_ylabel("实验假设总成本")
    ax.set_title("总成本（漏判=100 / 误拦截=5 的演示假设）", fontsize=11, pad=34)
    # 顶部留出注释空间（两行文字在最高柱上方）
    ax.set_ylim(0, float(strategies["总成本"].max()) * 1.22)
    for i, r in strategies.reset_index().iterrows():
        ax.annotate(f"阈值 {r['阈值']:.3f}\nTP={r['TP']} FP={r['FP']} FN={r['FN']}",
                    xy=(i, r["总成本"]), xytext=(0, 26), textcoords="offset points",
                    ha="center", fontsize=7.5)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out = _save(fig, filename)
    plt.close(fig)
    return out


def plot_sensitivity_line(
    sens: pd.DataFrame,
    title: str = "成本敏感性：漏判/误拦截成本比 → 最优阈值",
    filename: str = "fig19_sensitivity_line.png",
) -> Path:
    """横轴取 log10(成本比)，左轴最优阈值、右轴该阈值下总成本。"""
    setup_style()
    d = sens.sort_values("ratio")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(d["ratio"], d["threshold"], "o-", color=FRAUD_C, lw=1.8, ms=6, label="最优阈值")
    for _, r in d.iterrows():
        ax.annotate(f"{r['threshold']:.3f}", (r["ratio"], r["threshold"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("单笔漏判成本 / 单笔误拦截成本（对数轴）")
    ax.set_ylabel("成本最优阈值", color=FRAUD_C)
    ax.tick_params(axis="y", labelcolor=FRAUD_C)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, which="both")

    ax2 = ax.twinx()
    ax2.plot(d["ratio"], d["cost"], "s--", color=NORMAL_C, lw=1.5, ms=5, alpha=0.85, label="该阈值下总成本")
    ax2.set_ylabel("总成本", color=NORMAL_C)
    ax2.tick_params(axis="y", labelcolor=NORMAL_C)

    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, fontsize=8.5, loc="upper left")
    ax.set_title(title, fontsize=12)
    out = _save(fig, filename)
    plt.close(fig)
    return out


def plot_sensitivity_heatmap(
    matrix: pd.DataFrame,
    title: str = "最优阈值热力图（漏判成本 × 误拦截成本）",
    filename: str = "fig20_sensitivity_heatmap.png",
) -> Path:
    """行=漏判成本、列=误拦截成本，颜色=最优阈值。"""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax,
                cbar_kws={"label": "成本最优阈值"}, linewidths=0.6, linecolor="white",
                annot_kws={"fontsize": 8})
    ax.set_xlabel("单笔误拦截成本")
    ax.set_ylabel("单笔漏判成本")
    ax.set_title(title, fontsize=12)
    out = _save(fig, filename)
    plt.close(fig)
    return out


def plot_val_test_delta(
    delta: pd.DataFrame,
    title: str = "验证集 vs 测试集：冻结策略表现差异",
    filename: str = "fig21_val_test_delta.png",
) -> Path:
    """分组条形：验证集/测试集在 Precision/Recall/F1/误拦截率/漏判率 上的对比。"""
    setup_style()
    d = delta[delta["指标"].isin(["Precision", "Recall", "F1", "误拦截率", "漏判率"])]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.36
    b1 = ax.bar(x - w / 2, d["验证集"], width=w, label="验证集（选阈值）", color=NORMAL_C)
    b2 = ax.bar(x + w / 2, d["测试集"], width=w, label="测试集（一次性评估）", color=FRAUD_C)
    ax.bar_label(b1, fmt="%.3f", fontsize=7.5, padding=1.5)
    ax.bar_label(b2, fmt="%.3f", fontsize=7.5, padding=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(d["指标"])
    ax.set_ylim(0, max(1.0, float(d[["验证集", "测试集"]].to_numpy().max()) * 1.18))
    ax.set_ylabel("数值")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    out = _save(fig, filename)
    plt.close(fig)
    return out
