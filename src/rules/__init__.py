"""风险规则与策略模块：阈值扫描、成本模型、策略选择。"""

from src.rules.cost import (
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    cost_curve,
    grid_matrix,
    optimal_row,
    sensitivity_grid,
    sensitivity_scan,
    total_cost_at,
    with_cost,
)
from src.rules.strategies import (
    AGGRESSIVE_PRECISION_TARGET,
    CONSERVATIVE_RECALL_TARGET,
    build_strategies,
    evaluate_frozen,
    freeze_strategy,
    select_by_min_cost,
    select_by_precision_target,
    select_by_recall_target,
    val_test_delta,
)
from src.rules.thresholds import (
    SPEC_THRESHOLDS,
    build_threshold_grid,
    scan_thresholds,
    total_cost,
    validate_inputs,
)

__all__ = [
    "SPEC_THRESHOLDS", "build_threshold_grid", "scan_thresholds", "total_cost", "validate_inputs",
    "DEFAULT_COST_FN", "DEFAULT_COST_FP", "with_cost", "cost_curve", "optimal_row",
    "total_cost_at", "sensitivity_scan", "sensitivity_grid", "grid_matrix",
    "CONSERVATIVE_RECALL_TARGET", "AGGRESSIVE_PRECISION_TARGET", "build_strategies",
    "select_by_recall_target", "select_by_precision_target", "select_by_min_cost",
    "freeze_strategy", "evaluate_frozen", "val_test_delta",
]
