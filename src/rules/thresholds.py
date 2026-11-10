"""阈值扫描模块。

设计要点
--------
1. **向量化**：不对每个阈值重复遍历全量样本。先把概率升序排列并累加
   正样本计数，再用 ``np.searchsorted`` 一次定位，复杂度 O(T log n)。
2. **口径统一**：判定规则固定为 ``proba >= threshold`` 判为欺诈，
   与阶段二 ``src/evaluation/metrics.evaluate_at_threshold`` 完全一致，
   避免"同一指标两套算法"。
3. **参数校验前置**：阈值必须落在 [0, 1]、成本必须非负有限值，非法输入
   直接抛 ValueError，不静默返回 NaN 让下游误判。

术语（与业务对齐）
------------------
- 误拦截率 = FP / (FP + TN)：正常交易被误判为欺诈的比例，衡量"骚扰客户"程度
- 漏判率   = FN / (FN + TP)：真实欺诈被放过的比例，衡量"风控失守"程度
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 题目指定的 13 个演示阈值（保留，便于人工对照）
SPEC_THRESHOLDS: tuple[float, ...] = (
    0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
)

SCAN_COLUMNS: list[str] = [
    "threshold", "tp", "fp", "tn", "fn",
    "precision", "recall", "f1",
    "fpr", "fnr", "n_pred_pos",
]


def build_threshold_grid(
    step: float = 0.001,
    extra: tuple[float, ...] = SPEC_THRESHOLDS,
) -> np.ndarray:
    """构造细粒度阈值网格：等步长 + 指定阈值 + 两个端点。

    参数
    ----
    step  : 步长，必须在 (0, 1] 内
    extra : 额外必须包含的阈值（题目指定的演示点）

    返回升序、去重、裁剪到 [0, 1] 的 float 数组。
    """
    if not isinstance(step, (int, float)) or not np.isfinite(step):
        raise ValueError(f"step 必须是有限数值，收到 {step!r}")
    if not (0 < step <= 1):
        raise ValueError(f"step 必须落在 (0, 1]，收到 {step}")

    for t in extra:
        if not (0 <= t <= 1):
            raise ValueError(f"额外阈值必须落在 [0, 1]，收到 {t}")

    n = int(round(1 / step))
    grid = np.linspace(0.0, 1.0, n + 1)
    grid = np.concatenate([grid, np.asarray(extra, dtype=float), [0.0, 1.0]])
    # 四舍五入到 6 位，避免 0.30000000000000004 这类浮点噪声进入报告
    return np.unique(np.round(np.clip(grid, 0.0, 1.0), 6))


def validate_inputs(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """校验标签与概率并统一成 ndarray。

    规则：
    - 长度必须一致且非空
    - 标签只能是 {0, 1}（不允许 -1、2 这类隐性错误标签）
    - 概率必须落在 [0, 1] 且无 NaN/Inf
    """
    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("y_true / y_proba 必须是一维数组")
    if len(y) != len(p):
        raise ValueError(f"长度不一致：y_true={len(y)}, y_proba={len(p)}")
    if len(y) == 0:
        raise ValueError("输入为空，无法计算")

    uniq = np.unique(y)
    if not set(uniq.tolist()) <= {0, 1}:
        raise ValueError(f"y_true 只能含 0/1，实际含 {uniq.tolist()}")
    if not np.all(np.isfinite(p)):
        raise ValueError("y_proba 含 NaN 或 Inf")
    if p.min() < 0 or p.max() > 1:
        raise ValueError(f"y_proba 必须落在 [0,1]，实际范围 [{p.min()}, {p.max()}]")

    return y.astype(int), p


def scan_thresholds(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    thresholds: np.ndarray | list[float] | None = None,
    cost_fn: float | None = None,
    cost_fp: float | None = None,
) -> pd.DataFrame:
    """扫描多个阈值，返回每个阈值下的混淆矩阵计数与派生指标。

    参数
    ----
    thresholds : 判定阈值序列；None 时用 build_threshold_grid() 默认网格
    cost_fn    : 单笔漏判成本；与 cost_fp 同时给出时附加 cost 列
    cost_fp    : 单笔误拦截成本

    返回
    ----
    DataFrame，列见 SCAN_COLUMNS（可能追加 cost）。
    """
    y, p = validate_inputs(y_true, y_proba)

    th = build_threshold_grid() if thresholds is None else np.asarray(thresholds, dtype=float)
    if th.ndim != 1 or len(th) == 0:
        raise ValueError("thresholds 必须是非空一维序列")
    if not np.all(np.isfinite(th)):
        raise ValueError("thresholds 含 NaN 或 Inf")
    if th.min() < 0 or th.max() > 1:
        raise ValueError(f"thresholds 必须落在 [0,1]，实际范围 [{th.min()}, {th.max()}]")

    if (cost_fn is None) != (cost_fp is None):
        raise ValueError("cost_fn 与 cost_fp 必须同时提供或同时省略")

    # ---- 向量化核心：升序概率 + 正样本前缀和 ----
    order = np.argsort(p, kind="mergesort")      # mergesort 稳定，结果可复现
    p_sorted = p[order]
    y_sorted = y[order]
    cum_pos = np.concatenate([[0], np.cumsum(y_sorted)])   # cum_pos[i] = 前 i 个样本中的正样本数

    n = len(y)
    n_pos = int(y_sorted.sum())
    # predicted negative = 概率严格小于阈值的样本数（proba >= th 判正）
    n_neg_pred = np.searchsorted(p_sorted, th, side="left")
    pos_below = cum_pos[n_neg_pred]              # 其中被漏掉的正样本 = FN

    fn = pos_below
    tp = n_pos - fn
    tn = n_neg_pred - fn
    fp = (n - n_neg_pred) - tp

    tp, fp, tn, fn = (a.astype(int) for a in (tp, fp, tn, fn))
    pred_pos = tp + fp
    precision = np.divide(tp, pred_pos, out=np.zeros(len(th)), where=pred_pos > 0)
    recall = np.divide(tp, n_pos, out=np.zeros(len(th)), where=n_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(len(th)), where=denom > 0)
    fpr = np.divide(fp, fp + tn, out=np.full(len(th), np.nan), where=(fp + tn) > 0)
    fnr = np.divide(fn, fn + tp, out=np.full(len(th), np.nan), where=(fn + tp) > 0)

    out = pd.DataFrame({
        "threshold": th,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "n_pred_pos": pred_pos.astype(int),
    })

    if cost_fn is not None:
        out["cost"] = total_cost(fn, fp, cost_fn, cost_fp)

    return out


def total_cost(
    n_fn: np.ndarray | int,
    n_fp: np.ndarray | int,
    cost_fn: float,
    cost_fp: float,
) -> np.ndarray:
    """总成本 = 漏判数量 × 单笔漏判成本 + 误拦截数量 × 单笔误拦截成本。

    参数校验：成本必须非负、有限；计数必须非负。
    """
    for name, v in (("cost_fn", cost_fn), ("cost_fp", cost_fp)):
        if not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
            raise ValueError(f"{name} 必须是有限数值，收到 {v!r}")
        if v < 0:
            raise ValueError(f"{name} 不能为负，收到 {v}")

    fn = np.asarray(n_fn, dtype=float)
    fp = np.asarray(n_fp, dtype=float)
    if np.any(fn < 0) or np.any(fp < 0):
        raise ValueError("漏判/误拦截数量不能为负")

    return fn * float(cost_fn) + fp * float(cost_fp)
