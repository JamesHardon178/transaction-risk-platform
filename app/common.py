"""Streamlit 平台公共层：统一加载阶段一~三的真实产物。

原则
----
1. 只读阶段产物（预测 CSV / 指标 JSON / 清洗后 parquet），不在看板里重训模型；
2. 所有数字要么直接来自落盘文件，要么由真实数据实时计算，**无任何硬编码指标**；
3. 缓存用 @st.cache_data，参数变化只触发相应重算。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

MODEL_OPTIONS = {
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "Logistic Regression": "lr",
}
MODEL_NAME_BY_ABBR = {v: k for k, v in MODEL_OPTIONS.items()}


@st.cache_data(show_spinner="加载预测结果…")
def load_pred(split: str, abbr: str) -> pd.DataFrame:
    path = PROCESSED / f"predictions_{split}_{abbr}.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少阶段二预测文件: {path}（请先执行 02_model_training.ipynb）")
    return pd.read_csv(path)


@st.cache_data(show_spinner="加载指标档案…")
def load_metrics_json() -> dict:
    path = REPORTS / "model_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="加载策略档案…")
def load_analysis_json() -> dict:
    path = REPORTS / "threshold_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少阶段三分析档案: {path}（请先运行 scripts/build_stage3_analysis.py）")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="加载清洗后数据…")
def load_clean_data() -> pd.DataFrame:
    path = PROCESSED / "transactions_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少清洗后数据: {path}（请先执行 01_data_exploration.ipynb）")
    return pd.read_parquet(path, columns=["Time", "Amount", "Class"])


@st.cache_data(show_spinner="加载策略对比表…")
def load_strategy_csv() -> pd.DataFrame:
    path = REPORTS / "strategy_comparison.csv"
    return pd.read_csv(path)


def cost_scan(y, proba, cost_fn: float, cost_fp: float, step: float = 0.001) -> pd.DataFrame:
    """阈值扫描 + 成本列（薄封装，便于缓存与复用）。"""
    from src.rules.thresholds import build_threshold_grid, scan_thresholds

    grid = build_threshold_grid(step=step)
    return scan_thresholds(y, proba, thresholds=grid, cost_fn=cost_fn, cost_fp=cost_fp)


def fmt_int(v) -> str:
    return f"{int(v):,}"
