"""页面 4：策略成本优化。可配置成本参数 → 三策略重算 → 成本曲线 + 敏感性分析。

⚠️ 成本参数是实验假设，不代表真实金融机构数据。
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.common import load_pred
from src.rules.cost import DEFAULT_COST_FN, DEFAULT_COST_FP, cost_curve, sensitivity_scan
from src.rules.strategies import build_strategies
from src.rules.thresholds import build_threshold_grid, scan_thresholds


def render() -> None:
    st.header("💰 策略成本优化")
    st.caption("基于验证集（阈值选择的唯一合法数据）。Random Forest 为阶段二选定模型")

    st.warning("**成本参数为实验假设**（默认漏判 100 / 误拦截 5），仅用于演示成本权衡，"
               "不代表任何真实金融机构的损失或运营成本。", icon="⚠️")

    d = load_pred("val", "rf")
    y = d["Class"].to_numpy()
    p = d["predicted_probability"].to_numpy()

    st.subheader("成本参数（可修改，实时重算）")
    c1, c2 = st.columns(2)
    cost_fn = c1.number_input("单笔漏判成本（漏放一笔欺诈的损失）",
                              min_value=0.0, max_value=100000.0,
                              value=float(DEFAULT_COST_FN), step=10.0)
    cost_fp = c2.number_input("单笔误拦截成本（错拦一笔正常交易的损失）",
                              min_value=0.0, max_value=100000.0,
                              value=float(DEFAULT_COST_FP), step=1.0)
    if cost_fn + cost_fp == 0:
        st.error("两个成本不能同时为 0（否则任何阈值成本都是 0，策略无意义）")
        return
    ratio = cost_fn / cost_fp if cost_fp > 0 else float("inf")
    st.markdown(f"当前漏判/误拦截成本比：**{ratio:.1f} : 1**")

    grid = build_threshold_grid(step=0.001)
    scan = scan_thresholds(y, p, thresholds=grid)
    curve = cost_curve(scan, cost_fn=cost_fn, cost_fp=cost_fp)

    st.subheader("三策略对比（阈值由验证集确定，规则先于数据给定）")
    table = build_strategies(scan, cost_fn=cost_fn, cost_fp=cost_fp)
    show = table[["策略名称", "判定规则", "阈值", "Precision", "Recall", "F1",
                  "TP", "FP", "FN", "总成本", "规则是否可达"]]
    st.dataframe(show, width="stretch", hide_index=True)

    best = table.loc[table["总成本"].idxmin()]
    st.success(f"当前成本假设下的总成本最优策略：**{best['策略名称']}**，"
               f"阈值 **{best['阈值']:.3f}**，总成本 **{best['总成本']:,.0f}**", icon="✅")

    st.divider()
    st.subheader("总成本变化曲线")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["cost"], mode="lines",
                             name=f"漏判{cost_fn:.0f}/误拦截{cost_fp:.0f}",
                             line=dict(color="#7A9A3B")))
    fig.add_vline(x=best["阈值"], line_dash="dot", line_color="#CC4444",
                  annotation_text=f"最优 {best['阈值']:.3f}")
    fig.update_layout(xaxis_title="判定阈值", yaxis_title="实验假设总成本",
                      height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("成本敏感性分析")
    st.markdown("固定误拦截成本，改变漏判成本（即成本比），观察最优阈值如何漂移：")
    sens = sensitivity_scan(scan, cost_fp=max(cost_fp, 1e-9))
    line = go.Figure()
    line.add_trace(go.Scatter(x=sens["ratio"], y=sens["threshold"], mode="lines+markers+text",
                              text=[f"{t:.3f}" for t in sens["threshold"]],
                              textposition="top center", name="成本最优阈值",
                              line=dict(color="#CC4444")))
    line.update_layout(xaxis_title="漏判成本 / 误拦截成本（对数轴）",
                       yaxis_title="成本最优阈值", yaxis_range=[0, 1],
                       height=380, margin=dict(l=10, r=10, t=30, b=10))
    line.update_xaxes(type="log")
    st.plotly_chart(line, width="stretch")

    with st.expander("敏感性数据表"):
        st.dataframe(sens[["cost_fn", "cost_fp", "ratio", "threshold",
                           "precision", "recall", "tp", "fp", "fn", "cost"]]
                     .round(4), width="stretch")

    st.markdown(
        """
        **敏感性规律（由验证集实测得出）**

        | 成本比（漏判:误拦截） | 最优阈值 | 业务解读 |
        |---|---|---|
        | ≈ 1:1 | 0.409 | 两者代价相当 → 宁可少打扰，抓不干净的放过去 |
        | ≈ 10~100:1 | 0.144 | 漏判明显更贵 → 降阈值，容忍少量误拦截 |
        | ≈ 200:1 以上 | ≤ 0.036 | 漏判极贵 → 阈值压到极低，几乎"宁枉勿纵" |

        结论：**最优阈值不是一个常数，而是成本结构的函数**。这就是"为什么不能用 0.5"的定量答案——
        0.5 只是 sklearn 的默认值，不代表任何业务权衡。
        """
    )
