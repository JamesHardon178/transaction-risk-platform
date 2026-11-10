"""风险策略选择模块。

三种策略的**判定规则先于数据给定**（写成常量，不按验证集结果反调），
这样"策略"才是一个可复用的业务规则，而不是对 55 笔验证欺诈的事后拟合：

| 策略 | 规则 | 业务含义 |
|---|---|---|
| 保守 | 在 Recall ≥ 0.90 的阈值中取 Precision 最高者 | "宁可多拦，不可漏放"：先守住召回底线，再尽量减少打扰 |
| 平衡 | 取总成本最小的阈值 | 在给定成本假设下追求经济最优 |
| 激进 | 在 Precision ≥ 0.95 的阈值中取 Recall 最高者 | "不冤枉好人"：先保证拦截准确，宁可少抓 |

阈值一旦在验证集上选定即**冻结**，测试集只做一次评估，不回调。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.cost import DEFAULT_COST_FN, DEFAULT_COST_FP, with_cost
from src.rules.thresholds import validate_inputs

# 策略规则的目标值（业务政策，不是从数据里调出来的）
CONSERVATIVE_RECALL_TARGET: float = 0.90
AGGRESSIVE_PRECISION_TARGET: float = 0.95
PRIMARY_MODEL: str = "Random Forest"
COMPARISON_MODEL: str = "XGBoost"
BASELINE_MODEL: str = "Logistic Regression"

STRATEGY_ORDER = ("保守策略", "平衡策略", "激进策略")

STRATEGY_ROWS = [
    "策略名称", "判定规则", "阈值", "Precision", "Recall", "F1",
    "TP", "FP", "FN", "TN", "误拦截率", "漏判率", "总成本", "规则是否可达",
]


def _pick(scan: pd.DataFrame, mask: np.ndarray, key: str, ascending: bool) -> tuple[pd.Series | None, bool]:
    """在掩码子集里按 key 排序取第一条；返回 (行, 是否满足规则)。"""
    sub = scan[mask]
    if len(sub) == 0:
        return None, False
    return sub.sort_values(key, ascending=ascending).iloc[0], True


def select_by_recall_target(scan: pd.DataFrame, target: float = CONSERVATIVE_RECALL_TARGET) -> dict:
    """保守：满足 Recall ≥ target 的阈值中，取 Precision 最高（即阈值最高）。

    target 不可达时退化为"最大化 Recall"并标记 feasible=False。
    """
    if not (0 < target <= 1):
        raise ValueError(f"target 必须落在 (0,1]，收到 {target}")
    row, ok = _pick(scan, (scan["recall"] >= target).to_numpy(), "precision", ascending=False)
    if row is None:
        row, _ = _pick(scan, np.ones(len(scan), dtype=bool), "recall", ascending=False)
    return {"row": row, "feasible": ok, "rule": f"Recall ≥ {target:.0%} 下 Precision 最高"}


def select_by_precision_target(scan: pd.DataFrame, target: float = AGGRESSIVE_PRECISION_TARGET) -> dict:
    """激进：满足 Precision ≥ target 的阈值中，取 Recall 最高（即阈值最低）。"""
    if not (0 < target <= 1):
        raise ValueError(f"target 必须落在 (0,1]，收到 {target}")
    row, ok = _pick(scan, (scan["precision"] >= target).to_numpy(), "recall", ascending=False)
    if row is None:
        row, _ = _pick(scan, np.ones(len(scan), dtype=bool), "precision", ascending=False)
    return {"row": row, "feasible": ok, "rule": f"Precision ≥ {target:.0%} 下 Recall 最高"}


def select_by_min_cost(
    scan: pd.DataFrame,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict:
    """平衡：总成本最小；并列时取较高阈值（少打扰）。"""
    curve = with_cost(scan, cost_fn, cost_fp)
    min_cost = curve["cost"].min()
    row = curve[np.isclose(curve["cost"], min_cost)].sort_values("threshold", ascending=False).iloc[0]
    return {"row": row, "feasible": True,
            "rule": f"总成本最小（漏判={cost_fn:g} / 误拦截={cost_fp:g}）"}


def build_strategies(
    scan: pd.DataFrame,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    recall_target: float = CONSERVATIVE_RECALL_TARGET,
    precision_target: float = AGGRESSIVE_PRECISION_TARGET,
) -> pd.DataFrame:
    """生成三策略对比表（全部数字来自扫描结果，不手填）。"""
    if "cost" not in scan.columns:
        scan = with_cost(scan, cost_fn, cost_fp)

    picks = [
        ("保守策略", select_by_recall_target(scan, recall_target)),
        ("平衡策略", select_by_min_cost(scan, cost_fn, cost_fp)),
        ("激进策略", select_by_precision_target(scan, precision_target)),
    ]

    rows = []
    for name, res in picks:
        r = res["row"]
        rows.append({
            "策略名称": name,
            "判定规则": res["rule"],
            "阈值": round(float(r["threshold"]), 4),
            "Precision": round(float(r["precision"]), 4),
            "Recall": round(float(r["recall"]), 4),
            "F1": round(float(r["f1"]), 4),
            "TP": int(r["tp"]),
            "FP": int(r["fp"]),
            "FN": int(r["fn"]),
            "TN": int(r["tn"]),
            "误拦截率": round(float(r["fpr"]), 6),
            "漏判率": round(float(r["fnr"]), 6),
            "总成本": round(float(r["cost"]), 2),
            "规则是否可达": "是" if res["feasible"] else "否（已退化为极端策略）",
        })
    return pd.DataFrame(rows, columns=STRATEGY_ROWS)


def freeze_strategy(strategy_row: pd.Series | dict, model_name: str = PRIMARY_MODEL) -> dict:
    """把选定策略固化成可序列化 dict（写入 JSON / 供看板读取）。"""
    d = dict(strategy_row)
    th = float(d["阈值"])
    if not (0 <= th <= 1):
        raise ValueError(f"冻结阈值必须落在 [0,1]，收到 {th}")
    return {
        "model": model_name,
        "strategy": str(d["策略名称"]),
        "threshold": th,
        "frozen": True,
        "note": "阈值在验证集上确定后冻结；测试集仅用于一次性评估，严禁回调",
    }


def evaluate_frozen(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold: float,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict:
    """用冻结阈值在指定数据集上评估一次（测试集调用仅允许一次）。"""
    y, p = validate_inputs(y_true, y_proba)
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold 必须落在 [0,1]，收到 {threshold}")

    y_pred = (p >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cost = fn * cost_fn + fp * cost_fp

    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn), 6) if fp + tn else float("nan"),
        "fnr": round(fn / (fn + tp), 6) if fn + tp else float("nan"),
        "cost": round(float(cost), 2),
    }


def val_test_delta(val_metrics: dict, test_metrics: dict) -> pd.DataFrame:
    """验证集 vs 测试集表现差异表（绝对值 + 相对变化）。"""
    rows = []
    for k, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                     ("fpr", "误拦截率"), ("fnr", "漏判率"), ("cost", "实验假设总成本")):
        v, t = float(val_metrics[k]), float(test_metrics[k])
        rows.append({
            "指标": label,
            "验证集": round(v, 4),
            "测试集": round(t, 4),
            "绝对差": round(t - v, 4),
            "相对变化": f"{(t - v) / v:+.1%}" if v else "n/a",
        })
    return pd.DataFrame(rows)
