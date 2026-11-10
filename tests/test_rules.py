"""阶段三规则层测试：阈值扫描、成本模型、策略选择、边界与非法输入。

原则：
- 每个公开函数至少一条"正确性"（与 sklearn 口径对齐）+ 一条"非法输入"测试；
- 覆盖极端阈值（0 与 1）和并列成本的 tie-break 行为；
- 敏感性分析验证单调性（漏判成本越高 → 最优阈值不升高）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from src.rules.cost import (
    DEFAULT_COST_FN, DEFAULT_COST_FP, cost_curve, grid_matrix,
    optimal_row, sensitivity_scan, total_cost_at, with_cost,
)
from src.rules.strategies import (
    build_strategies, evaluate_frozen, freeze_strategy,
    select_by_min_cost, select_by_precision_target, select_by_recall_target,
)
from src.rules.thresholds import (
    SPEC_THRESHOLDS, build_threshold_grid, scan_thresholds, total_cost, validate_inputs,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- fixtures
@pytest.fixture()
def sample():
    """构造一个平衡度可控的合成样本集（含重复概率值，考验 searchsorted 边界）。"""
    n = 500
    y = np.zeros(n, dtype=int)
    y[:30] = 1                                   # 30 个正样本
    p = np.clip(RNG.normal(0.3, 0.25, n), 0, 1)
    p[y == 1] = np.clip(RNG.normal(0.7, 0.2, 30), 0, 1)
    p[10] = 0.5                                  # 人为制造同分，验证 >= 边界
    p[11] = 0.5
    return y, p


@pytest.fixture()
def sample_scan(sample):
    y, p = sample
    return scan_thresholds(y, p, cost_fn=DEFAULT_COST_FN, cost_fp=DEFAULT_COST_FP)


# ---------------------------------------------------------------- 网格
def test_grid_contains_spec_thresholds():
    grid = build_threshold_grid(step=0.001)
    for t in SPEC_THRESHOLDS:
        assert np.isclose(grid, t).any()
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert np.all(np.diff(grid) > 0)             # 严格升序、无重复


def test_grid_invalid_step_raises():
    with pytest.raises(ValueError):
        build_threshold_grid(step=0)
    with pytest.raises(ValueError):
        build_threshold_grid(step=-0.1)
    with pytest.raises(ValueError):
        build_threshold_grid(step=1.5)


def test_grid_invalid_extra_raises():
    with pytest.raises(ValueError):
        build_threshold_grid(extra=(1.2,))
    with pytest.raises(ValueError):
        build_threshold_grid(extra=(float("nan"),))


# ---------------------------------------------------------------- 输入校验
def test_validate_inputs_bad_labels(sample):
    y, p = sample
    y_bad = y.copy()
    y_bad[0] = 2
    with pytest.raises(ValueError, match="0/1"):
        validate_inputs(y_bad, p)


def test_validate_inputs_length_mismatch(sample):
    y, p = sample
    with pytest.raises(ValueError, match="长度不一致"):
        validate_inputs(y, p[:-1])


def test_validate_inputs_proba_out_of_range(sample):
    y, p = sample
    p_bad = p.copy()
    p_bad[0] = 1.5
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        validate_inputs(y, p_bad)


def test_validate_inputs_nan_proba(sample):
    y, p = sample
    p_bad = p.copy()
    p_bad[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_inputs(y, p_bad)


# ---------------------------------------------------------------- 阈值扫描正确性
def test_scan_matches_sklearn_at_05(sample):
    y, p = sample
    scan = scan_thresholds(y, p, thresholds=[0.5])
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    r = scan.iloc[0]
    assert (int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"])) == \
           (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])


def test_scan_extreme_thresholds(sample):
    y, p = sample
    scan = scan_thresholds(y, p, thresholds=[0.0, 1.0])
    r0, r1 = scan.iloc[0], scan.iloc[1]
    # 阈值 0：全部判欺诈
    assert r0["tp"] == y.sum() and r0["fn"] == 0 and r0["fp"] == (y == 0).sum()
    # 阈值 1：只有概率恰为 1 的样本被判欺诈（>= 规则）
    assert r1["tp"] == int(((p >= 1) & (y == 1)).sum())
    assert r1["fp"] == int(((p >= 1) & (y == 0)).sum())


def test_scan_duplicate_probabilities_boundary(sample):
    """概率恰等于阈值的样本必须判为欺诈（>= 规则）。"""
    y, p = sample
    scan = scan_thresholds(y, p, thresholds=[0.5])
    manual_pred = int((p >= 0.5).sum())
    assert int(scan.iloc[0]["tp"] + scan.iloc[0]["fp"]) == manual_pred


def test_scan_invalid_thresholds_raises(sample):
    y, p = sample
    with pytest.raises(ValueError):
        scan_thresholds(y, p, thresholds=[1.5])
    with pytest.raises(ValueError):
        scan_thresholds(y, p, thresholds=[np.nan])
    with pytest.raises(ValueError):
        scan_thresholds(y, p, thresholds=np.array([]))


def test_scan_requires_cost_pair(sample):
    y, p = sample
    with pytest.raises(ValueError, match="同时"):
        scan_thresholds(y, p, thresholds=[0.5], cost_fn=100)


def test_total_cost_negative_raises():
    with pytest.raises(ValueError):
        total_cost(3, 2, cost_fn=-1, cost_fp=5)
    with pytest.raises(ValueError):
        total_cost(3, 2, cost_fn=100, cost_fp=float("nan"))


# ---------------------------------------------------------------- 成本模型
def test_cost_curve_matches_manual(sample_scan):
    curve = cost_curve(sample_scan, cost_fn=100, cost_fp=5)
    assert np.all(np.diff(curve["threshold"]) > 0)
    manual = curve["fn"] * 100 + curve["fp"] * 5
    assert np.allclose(curve["cost"], manual)


def test_optimal_row_tie_break_prefers_higher_threshold():
    """成本并列时取更高阈值（少打扰客户）。"""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.8])
    scan = scan_thresholds(y, p, thresholds=[0.0, 0.2, 0.4, 0.7, 0.9])
    # 成本_fn=0 时：抓满全部欺诈的成本恒为 0，多个阈值并列
    best = optimal_row(scan, cost_fn=0.0, cost_fp=1.0)
    tied = scan[(scan["fn"] * 0 + scan["fp"] * 1.0) == 0.0]
    assert best["threshold"] == tied["threshold"].max()


def test_total_cost_at_missing_threshold_raises(sample_scan):
    with pytest.raises(ValueError, match="不在扫描网格"):
        total_cost_at(sample_scan, 0.12345)


def test_with_cost_does_not_mutate_input(sample_scan):
    before = sample_scan.copy()
    with_cost(sample_scan, 100, 5)
    pd.testing.assert_frame_equal(sample_scan, before)


# ---------------------------------------------------------------- 敏感性
def test_sensitivity_threshold_non_increasing(sample_scan):
    """漏判成本越贵 → 最优阈值不应升高（单调不增）。"""
    sens = sensitivity_scan(sample_scan, cost_fp=5.0,
                            fn_costs=(5, 20, 100, 500, 2000, 10000))
    th = sens.sort_values("cost_fn")["threshold"].to_numpy()
    assert np.all(np.diff(th) <= 1e-9)


def test_sensitivity_rejects_bad_cost(sample_scan):
    with pytest.raises(ValueError):
        sensitivity_scan(sample_scan, cost_fp=0.0)
    with pytest.raises(ValueError):
        sensitivity_scan(sample_scan, cost_fp=-3)


def test_grid_matrix_shape(sample_scan):
    sens = sensitivity_scan(sample_scan, cost_fp=5.0, fn_costs=(10, 100, 1000))
    m = grid_matrix(sens.assign(cost_fp=5.0), value="threshold")
    assert m.shape[0] == 3


# ---------------------------------------------------------------- 策略选择
def test_strategy_recall_target(sample_scan):
    res = select_by_recall_target(sample_scan, target=0.9)
    assert res["feasible"]
    assert res["row"]["recall"] >= 0.9
    # 是满足约束的阈值中 Precision 最高的
    ok = sample_scan[sample_scan["recall"] >= 0.9]
    assert res["row"]["precision"] == ok["precision"].max()


def test_strategy_precision_target(sample_scan):
    res = select_by_precision_target(sample_scan, target=0.95)
    assert res["row"]["precision"] >= 0.95
    ok = sample_scan[sample_scan["precision"] >= 0.95]
    assert res["row"]["recall"] == ok["recall"].max()


def test_strategy_target_infeasible_degrades(sample_scan):
    """不可达目标 → 退化为极端策略并标记 feasible=False。"""
    res = select_by_recall_target(sample_scan, target=1.0)
    if not res["feasible"]:
        assert res["row"]["recall"] == sample_scan["recall"].max()


def test_strategy_invalid_target_raises(sample_scan):
    with pytest.raises(ValueError):
        select_by_recall_target(sample_scan, target=0)
    with pytest.raises(ValueError):
        select_by_precision_target(sample_scan, target=1.5)


def test_build_strategies_table(sample_scan):
    table = build_strategies(sample_scan)
    assert list(table["策略名称"]) == ["保守策略", "平衡策略", "激进策略"]
    # 全部指标都在 [0,1] 内，成本非负
    for col in ("Precision", "Recall", "F1"):
        assert table[col].between(0, 1).all()
    assert (table["总成本"] >= 0).all()
    # 阈值一致地落在扫描网格里
    for th in table["阈值"]:
        assert np.isclose(sample_scan["threshold"], th, atol=1e-4).any()


def test_freeze_strategy_roundtrip():
    row = {"策略名称": "平衡策略", "阈值": 0.144}
    frozen = freeze_strategy(row, model_name="Random Forest")
    assert frozen["threshold"] == pytest.approx(0.144)
    assert frozen["frozen"] is True
    with pytest.raises(ValueError):
        freeze_strategy({"策略名称": "x", "阈值": 1.7})


def test_evaluate_frozen_matches_sklearn(sample):
    y, p = sample
    m = evaluate_frozen(y, p, threshold=0.5, cost_fn=100, cost_fp=5)
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    assert m["cost"] == cm[1, 0] * 100 + cm[0, 1] * 5


def test_evaluate_frozen_invalid_threshold(sample):
    y, p = sample
    with pytest.raises(ValueError):
        evaluate_frozen(y, p, threshold=1.4)


def test_select_by_min_cost_equals_optimal_row(sample_scan):
    a = select_by_min_cost(sample_scan, 100, 5)["row"]
    b = optimal_row(sample_scan, 100, 5)
    assert float(a["threshold"]) == pytest.approx(b["threshold"])


def test_cost_defaults_are_documented_assumptions():
    """默认成本必须与文档口径一致，防止悄悄改数。"""
    assert (DEFAULT_COST_FN, DEFAULT_COST_FP) == (100.0, 5.0)
