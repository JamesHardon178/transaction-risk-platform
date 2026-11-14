"""阶段三分析脚本：阈值扫描 → 策略选择 → 冻结 → 测试集一次性评估 → 敏感性分析。

约束（脚本内硬性保证）
----------------------
1. 阈值/策略/成本参数**只用验证集**（predictions_val_*.csv）确定；
2. 测试集（predictions_test_*.csv）只在 freeze 之后评估一次，不参与任何选择；
3. 成本参数是**实验假设**，写进所有产出物，不伪装成真实业务数据。

产出
----
reports/threshold_analysis.json     阈值/策略/敏感性的机器可读档案
reports/strategy_comparison.csv     三策略对比表（RF 主选 + XGB 对比）
reports/figures/fig16~fig21         策略图表
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.predictions import load_predictions                       # noqa: E402
from src.rules.cost import (                                            # noqa: E402
    DEFAULT_COST_FN, DEFAULT_COST_FP, grid_matrix,
    optimal_row, sensitivity_grid, sensitivity_scan, with_cost,
)
from src.rules.strategies import (                                       # noqa: E402
    AGGRESSIVE_PRECISION_TARGET, BASELINE_MODEL, COMPARISON_MODEL,
    CONSERVATIVE_RECALL_TARGET, PRIMARY_MODEL, build_strategies, evaluate_frozen, freeze_strategy, val_test_delta,
)
from src.rules.thresholds import (                                       # noqa: E402
    SPEC_THRESHOLDS, build_threshold_grid, scan_thresholds,
)
from src.viz import strategy_plots                                       # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
MODELS = {"rf": "Random Forest", "xgb": "XGBoost", "lr": "Logistic Regression"}


def spec_table(scan: pd.DataFrame) -> list[dict]:
    """把题目指定的 13 个阈值单独抽成表（便于人工对照）。"""
    rows = []
    for t in SPEC_THRESHOLDS:
        hit = scan[np.isclose(scan["threshold"], t, atol=1e-9)]
        if len(hit) == 0:
            continue
        r = hit.iloc[0]
        rows.append({
            "threshold": t,
            "tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]), "tn": int(r["tn"]),
            "precision": round(float(r["precision"]), 4),
            "recall": round(float(r["recall"]), 4),
            "f1": round(float(r["f1"]), 4),
            "fpr": round(float(r["fpr"]), 6),
            "fnr": round(float(r["fnr"]), 6),
        })
    return rows


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    grid = build_threshold_grid(step=0.001)
    print(f"[网格] {len(grid)} 个阈值，范围 [{grid.min():.3f}, {grid.max():.3f}]")

    # ---------- 1. 只读验证集，做全部扫描 ----------
    val, val_scan = {}, {}
    for abbr, name in MODELS.items():
        d = load_predictions("val", abbr)
        val[name] = d
        val_scan[name] = scan_thresholds(
            d["Class"].to_numpy(), d["predicted_probability"].to_numpy(),
            thresholds=grid, cost_fn=DEFAULT_COST_FN, cost_fp=DEFAULT_COST_FP,
        )
        print(f"[验证集] {name:20s} n={len(d):6d} 欺诈={int(d['Class'].sum()):3d} 扫描完成")

    rf_name = PRIMARY_MODEL
    rf_scan = val_scan[rf_name]

    # 成本最优阈值（仅用于说明"0.5 不是最优"）
    optimal = {name: optimal_row(s, DEFAULT_COST_FN, DEFAULT_COST_FP) for name, s in val_scan.items()}
    for name, r in optimal.items():
        print(f"[成本最优] {name:20s} 阈值={r['threshold']:.3f} 成本={r['cost']:.0f} "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']}")

    # ---------- 2. 三策略（RF 主选，XGB 对比）----------
    strat_rf = build_strategies(rf_scan, DEFAULT_COST_FN, DEFAULT_COST_FP)
    strat_xgb = build_strategies(val_scan[COMPARISON_MODEL], DEFAULT_COST_FN, DEFAULT_COST_FP)
    print("\n[三策略 · Random Forest]")
    print(strat_rf[["策略名称", "阈值", "Precision", "Recall", "F1", "TP", "FP", "FN", "总成本"]].to_string(index=False))

    # 冻结：平衡策略（成本最小）作为最终推荐；其余两策略阈值同样冻结后一并评估
    balanced = strat_rf.set_index("策略名称").loc["平衡策略"].to_dict()
    balanced["策略名称"] = "平衡策略"
    frozen = freeze_strategy(balanced, model_name=rf_name)
    print(f"\n[冻结] 模型={frozen['model']} 策略={frozen['strategy']} 阈值={frozen['threshold']}")

    frozen_thresholds = {r["策略名称"]: float(r["阈值"]) for _, r in strat_rf.iterrows()}

    # ---------- 3. 测试集：一次性评估（不回调）----------
    test = {name: load_predictions("test", abbr) for abbr, name in MODELS.items()}
    test_eval = {}
    for name, th in frozen_thresholds.items():
        m = evaluate_frozen(test[rf_name]["Class"].to_numpy(),
                            test[rf_name]["predicted_probability"].to_numpy(),
                            th, DEFAULT_COST_FN, DEFAULT_COST_FP)
        test_eval[name] = m
        print(f"[测试集] {name} 阈值={th:.3f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} TP={m['tp']} FP={m['fp']} FN={m['fn']} 成本={m['cost']:.0f}")

    val_eval = {}
    for name, th in frozen_thresholds.items():
        val_eval[name] = evaluate_frozen(val[rf_name]["Class"].to_numpy(),
                                         val[rf_name]["predicted_probability"].to_numpy(),
                                         th, DEFAULT_COST_FN, DEFAULT_COST_FP)
    delta = val_test_delta(val_eval["平衡策略"], test_eval["平衡策略"])
    print("\n[验证集 vs 测试集 · 平衡策略]")
    print(delta.to_string(index=False))

    # ---------- 4. 敏感性分析 ----------
    sens1d = sensitivity_scan(rf_scan, cost_fp=DEFAULT_COST_FP)
    sens2d = sensitivity_grid(
        rf_scan,
        fn_costs=(5, 10, 20, 50, 100, 200, 500, 1000, 2000),
        fp_costs=(1, 2, 5, 10, 20, 50),
    )
    print("\n[敏感性 · 1D] 漏判成本 → 最优阈值")
    print(sens1d[["cost_fn", "ratio", "threshold", "recall", "precision", "cost"]].to_string(index=False))

    # ---------- 5. 出图 ----------
    FIGS = {}
    FIGS["fig16"] = strategy_plots.plot_threshold_metric_curves(
        rf_scan, strat_rf,
        title="图 16  Random Forest：阈值 - Precision/Recall/F1 与三策略位置",
        filename="fig16_threshold_curves.png")
    curves = {
        f"漏判 {c:.0f} / 误拦截 {DEFAULT_COST_FP:.0f}": with_cost(rf_scan, c, DEFAULT_COST_FP)
        for c in (10, 50, 100, 500, 2000)
    }
    FIGS["fig17"] = strategy_plots.plot_cost_curve(
        curves, best=("平衡策略", frozen["threshold"], float(balanced["总成本"])),
        title="图 17  不同成本假设下的 阈值-总成本曲线（Random Forest，验证集）",
        filename="fig17_cost_curve.png")
    FIGS["fig18"] = strategy_plots.plot_strategy_comparison(
        strat_rf, title="图 18  三策略指标与成本对比（Random Forest，验证集）",
        filename="fig18_strategy_comparison.png")
    FIGS["fig19"] = strategy_plots.plot_sensitivity_line(
        sens1d, title="图 19  成本敏感性：漏判/误拦截成本比 → 成本最优阈值",
        filename="fig19_sensitivity_line.png")
    FIGS["fig20"] = strategy_plots.plot_sensitivity_heatmap(
        grid_matrix(sens2d, "threshold"),
        title="图 20  最优阈值热力图（行=漏判成本，列=误拦截成本）",
        filename="fig20_sensitivity_heatmap.png")
    FIGS["fig21"] = strategy_plots.plot_val_test_delta(
        delta, title="图 21  平衡策略在验证集与测试集上的表现差异",
        filename="fig21_val_test_delta.png")
    print("\n[图表]")
    for k, p in FIGS.items():
        print(f"  {k}: {p}")

    # ---------- 6. 落盘 ----------
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    analysis = {
        "generated_by": "scripts/build_stage3_analysis.py",
        "cost_assumptions": {
            "cost_fn": DEFAULT_COST_FN,
            "cost_fp": DEFAULT_COST_FP,
            "disclaimer": "实验假设，仅用于演示成本权衡，不代表任何真实金融机构的损失/运营成本",
        },
        "grid": {"step": 0.001, "n_thresholds": int(len(grid)),
                 "spec_thresholds": list(SPEC_THRESHOLDS)},
        "models": {
            "primary": PRIMARY_MODEL, "comparison": COMPARISON_MODEL, "baseline": BASELINE_MODEL,
        },
        "val_split": {
            "n": int(len(val[rf_name])), "n_pos": int(val[rf_name]["Class"].sum()),
        },
        "test_split": {
            "n": int(len(test[rf_name])), "n_pos": int(test[rf_name]["Class"].sum()),
        },
        "cost_optimal": optimal,
        "spec_thresholds_rf": spec_table(rf_scan),
        "strategies": {
            "Random Forest": strat_rf.to_dict(orient="records"),
            "XGBoost": strat_xgb.to_dict(orient="records"),
        },
        "frozen_strategy": frozen,
        "frozen_thresholds_all": frozen_thresholds,
        "val_metrics_frozen": val_eval,
        "test_metrics_frozen": test_eval,
        "val_test_delta_balanced": delta.to_dict(orient="records"),
        "sensitivity_1d": sens1d.to_dict(orient="records"),
        "sensitivity_2d": sens2d.to_dict(orient="records"),
        "figures": {k: str(v) for k, v in FIGS.items()},
        "policy_targets": {
            "conservative_recall_target": CONSERVATIVE_RECALL_TARGET,
            "aggressive_precision_target": AGGRESSIVE_PRECISION_TARGET,
            "note": "策略规则的目标值在查看验证集结果前即确定，不按结果反调",
        },
    }
    out_json = REPORTS / "threshold_analysis.json"
    out_json.write_text(json.dumps(_clean(analysis), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[落盘] {out_json}  ({out_json.stat().st_size:,} bytes)")

    comp = pd.concat(
        [strat_rf.assign(模型="Random Forest"), strat_xgb.assign(模型="XGBoost")],
        ignore_index=True,
    )[["模型", "策略名称", "判定规则", "阈值", "Precision", "Recall", "F1",
       "TP", "FP", "FN", "TN", "误拦截率", "漏判率", "总成本", "规则是否可达"]]
    out_csv = REPORTS / "strategy_comparison.csv"
    comp.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[落盘] {out_csv}")


if __name__ == "__main__":
    main()
