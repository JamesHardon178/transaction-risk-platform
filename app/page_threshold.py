"""页面 3：风险阈值分析。滑块 + 实时指标 + 阈值曲线（全部由预测 CSV 实时计算）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import MODEL_OPTIONS, MODEL_NAME_BY_ABBR, fmt_int, load_pred
from src.rules.thresholds import build_threshold_grid, scan_thresholds


def render() -> None:
    st.header("🎚️ 风险阈值分析")
    st.caption("阈值含义：模型欺诈概率 ≥ 阈值 → 判为欺诈（拦截）。全部指标由预测文件实时计算")

    model_name = st.radio("模型", list(MODEL_OPTIONS), horizontal=True,
                          help="阶段二已按验证集 PR-AUC 选定 Random Forest 为最终模型")
    abbr = MODEL_OPTIONS[model_name]

    d = load_pred("val", abbr)
    y = d["Class"].to_numpy()
    p = d["predicted_probability"].to_numpy()
    n_fraud = int(y.sum())

    st.subheader("阈值滑块")
    threshold = st.slider(
        "判定阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.001,
        help="阶段二所有模型指标均基于默认阈值 0.5；拖动查看不同工作点",
    )

    scan = scan_thresholds(y, p, thresholds=[threshold])
    r = scan.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{r['precision']:.4f}", help="报警中真欺诈占比")
    c2.metric("Recall", f"{r['recall']:.4f}", help="欺诈被抓到的比例")
    c3.metric("F1", f"{r['f1']:.4f}")

    st.subheader(f"混淆矩阵（阈值 {threshold:.3f}，验证集 {fmt_int(len(d))} 笔 / {n_fraud} 笔欺诈）")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TP 抓获欺诈", fmt_int(r["tp"]), delta=f"漏判率 {r['fnr']:.2%}" if r['tp'] + r['fn'] else "n/a",
              delta_color="off")
    m2.metric("FP 误拦截", fmt_int(r["fp"]), delta=f"误拦截率 {r['fpr']:.2%}" if r['fp'] + r['tn'] else "n/a",
              delta_color="inverse")
    m3.metric("FN 漏判", fmt_int(r["fn"]), delta_color="inverse")
    m4.metric("TN 正常放行", fmt_int(r["tn"]))

    st.divider()
    st.subheader("阈值变化曲线")
    grid = build_threshold_grid(step=0.001)
    full = scan_thresholds(y, p, thresholds=grid)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full["threshold"], y=full["precision"],
                             mode="lines", name="Precision", line=dict(color="#3465A4")))
    fig.add_trace(go.Scatter(x=full["threshold"], y=full["recall"],
                             mode="lines", name="Recall", line=dict(color="#CC4444")))
    fig.add_trace(go.Scatter(x=full["threshold"], y=full["f1"],
                             mode="lines", name="F1", line=dict(color="#7A9A3B", dash="dash")))
    fig.add_vline(x=threshold, line_dash="dot", line_color="#333",
                  annotation_text=f"当前 {threshold:.3f}")
    fig.update_layout(xaxis_title="判定阈值", yaxis_title="得分", yaxis_range=[0, 1.02],
                      height=420, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    with st.expander("查看扫描数据表（每 0.05 一个采样点）"):
        sample = full[np.isclose(full["threshold"] % 0.05, 0, atol=1e-9)]
        st.dataframe(
            sample[["threshold", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "fpr", "fnr"]]
            .round(4),
            width="stretch",
        )

    st.markdown(
        f"""
        **当前工作点解读（{MODEL_NAME_BY_ABBR[abbr]}）**
        - 阈值越低 → 抓得越多（Recall ↑）但误拦截暴涨（FP ↑，Precision ↓）；
        - 阈值越高 → 报警更准（Precision ↑）但漏判增多（Recall ↓）；
        - "模型能力"（排序）与"工作点"（阈值）是两件事——本页面调的是后者。
        """
    )
