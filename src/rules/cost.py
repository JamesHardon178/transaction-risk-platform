"""成本模型与敏感性分析模块。

成本模型（**实验假设，非真实金融机构数据**）
--------------------------------------------
    总成本 = 漏判数量 × 单笔漏判成本 + 误拦截数量 × 单笔误拦截成本

- 漏判（FN）：真实欺诈被判为正常 → 资金损失 → 单笔成本通常远高于误拦截
- 误拦截（FP）：正常交易被判为欺诈 → 客户打扰/运营成本 → 单笔成本较低

两者比例决定"最优阈值往哪边滑"，这正是敏感性分析要展示的东西。
所有函数都强制校验参数：成本必须非负有限，比例必须为正。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.thresholds import scan_thresholds, total_cost

# 演示用实验假设（必须在报告/看板中显式标注为假设）
DEFAULT_COST_FN: float = 100.0
DEFAULT_COST_FP: float = 5.0

REQUIRED_COLS = ("threshold", "tp", "fp", "tn", "fn")


def _check_scan(scan: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in scan.columns]
    if missing:
        raise KeyError(f"扫描结果缺少列: {missing}")
    if len(scan) == 0:
        raise ValueError("扫描结果为空")
    if (scan["threshold"] < 0).any() or (scan["threshold"] > 1).any():
        raise ValueError("扫描结果含越界阈值")


def with_cost(
    scan: pd.DataFrame,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> pd.DataFrame:
    """给扫描结果追加 cost 列（不修改入参，返回副本）。"""
    _check_scan(scan)
    out = scan.copy()
    out["cost"] = total_cost(out["fn"].to_numpy(), out["fp"].to_numpy(), cost_fn, cost_fp)
    return out


def cost_curve(
    scan: pd.DataFrame,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> pd.DataFrame:
    """阈值 → 总成本 曲线数据（阈值升序）。"""
    out = with_cost(scan, cost_fn, cost_fp)
    return out.sort_values("threshold").reset_index(drop=True)


def optimal_row(
    scan: pd.DataFrame,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict:
    """使总成本最小的阈值行。

    并列最小成本时取**较高阈值**（成本相同则少打扰客户，符合业务直觉）。
    返回 dict，含 threshold/tp/fp/tn/fn/precision/recall/f1/cost。
    """
    curve = with_cost(scan, cost_fn, cost_fp)
    min_cost = curve["cost"].min()
    tied = curve[np.isclose(curve["cost"], min_cost)]
    row = tied.sort_values("threshold", ascending=False).iloc[0]
    return {k: (float(row[k]) if k not in ("tp", "fp", "tn", "fn") else int(row[k]))
            for k in ("threshold", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "cost")}


def total_cost_at(
    scan: pd.DataFrame,
    threshold: float,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """指定阈值的总成本（阈值必须在扫描网格内，避免静默插值）。"""
    _check_scan(scan)
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold 必须落在 [0,1]，收到 {threshold}")
    hit = scan[np.isclose(scan["threshold"], threshold, atol=1e-9)]
    if len(hit) == 0:
        raise ValueError(f"阈值 {threshold} 不在扫描网格内，请先把它加入网格")
    r = hit.iloc[0]
    return float(total_cost(int(r["fn"]), int(r["fp"]), cost_fn, cost_fp))


def sensitivity_scan(
    scan: pd.DataFrame,
    cost_fp: float = DEFAULT_COST_FP,
    fn_costs: tuple[float, ...] | list[float] | None = None,
) -> pd.DataFrame:
    """固定误拦截成本，扫描单笔漏判成本 → 观察最优阈值漂移。

    返回列：cost_fn / cost_fp / ratio / threshold / precision / recall / f1 / tp / fp / fn / cost
    ratio = cost_fn / cost_fp，便于直接讲"漏判比误拦截贵多少倍"。
    """
    _check_scan(scan)
    if not np.isfinite(cost_fp) or cost_fp <= 0:
        raise ValueError(f"cost_fp 必须为正有限值，收到 {cost_fp}")

    if fn_costs is None:
        # 覆盖"漏判几乎不痛"到"漏判极贵"两个极端
        fn_costs = (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
    if len(fn_costs) == 0:
        raise ValueError("fn_costs 不能为空")

    rows = []
    for c_fn in fn_costs:
        r = optimal_row(scan, cost_fn=float(c_fn), cost_fp=float(cost_fp))
        rows.append({
            "cost_fn": float(c_fn),
            "cost_fp": float(cost_fp),
            "ratio": float(c_fn) / float(cost_fp),
            **r,
        })
    return pd.DataFrame(rows)


def sensitivity_grid(
    scan: pd.DataFrame,
    fn_costs: tuple[float, ...] | list[float] | None = None,
    fp_costs: tuple[float, ...] | list[float] | None = None,
) -> pd.DataFrame:
    """二维敏感性：漏判成本 × 误拦截成本 的网格，返回 long-form。

    列：cost_fn / cost_fp / ratio / threshold / cost / recall / precision
    """
    _check_scan(scan)
    fn_costs = fn_costs or (1, 5, 10, 25, 50, 100, 250, 500, 1000)
    fp_costs = fp_costs or (1, 2, 5, 10, 20, 50)

    for name, grid in (("fn_costs", fn_costs), ("fp_costs", fp_costs)):
        if len(grid) == 0:
            raise ValueError(f"{name} 不能为空")
        if any((not np.isfinite(c)) or c <= 0 for c in grid):
            raise ValueError(f"{name} 必须全为正有限值")

    rows = []
    for c_fn in fn_costs:
        for c_fp in fp_costs:
            r = optimal_row(scan, cost_fn=float(c_fn), cost_fp=float(c_fp))
            rows.append({
                "cost_fn": float(c_fn),
                "cost_fp": float(c_fp),
                "ratio": float(c_fn) / float(c_fp),
                **r,
            })
    return pd.DataFrame(rows)


def grid_matrix(sens: pd.DataFrame, value: str = "threshold") -> pd.DataFrame:
    """把 long-form 敏感性结果转成 行=cost_fn、列=cost_fp 的矩阵（画热力图用）。"""
    if value not in ("threshold", "cost", "recall", "precision"):
        raise ValueError(f"不支持的取值列: {value}")
    return sens.pivot(index="cost_fn", columns="cost_fp", values=value)


def compare_at_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float],
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> pd.DataFrame:
    """对给定阈值列表逐个计算指标（供看板滑块/策略表复用）。"""
    scan = scan_thresholds(y_true, y_proba, thresholds=thresholds, cost_fn=cost_fn, cost_fp=cost_fp)
    return scan
