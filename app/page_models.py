"""页面 2：模型评估。指标档案来自 model_metrics.json，曲线由预测 CSV 实时计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from app.common import MODEL_OPTIONS, fmt_int, load_metrics_json, load_pred


def _roc_fig(split: str) -> go.Figure:
    fig = go.Figure()
    for name, abbr in MODEL_OPTIONS.items():
        d = load_pred(split, abbr)
        y, p = d["Class"].to_numpy(), d["predicted_probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name,
                                 hovertemplate=f"{name} AUC={auc:.4f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="随机基线",
                             showlegend=True))
    fig.update_layout(title=f"{split.upper()} 集 ROC 曲线（由预测文件实时计算）",
                      xaxis_title="假阳性率 FPR", yaxis_title="真阳性率 TPR",
                      height=430, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _pr_fig(split: str) -> go.Figure:
    fig = go.Figure()
    base = None
    for name, abbr in MODEL_OPTIONS.items():
        d = load_pred(split, abbr)
        y, p = d["Class"].to_numpy(), d["predicted_probability"].to_numpy()
        pr, rc, _ = precision_recall_curve(y, p)
        ap = float(average_precision_score(y, p))
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"{name} (PR-AUC={ap:.4f})"))
        base = float(y.mean())
    if base:
        fig.add_hline(y=base, line_dash="dash", line_color="gray",
                      annotation_text=f"随机基线（正样本率={base:.4f}）")
    fig.update_layout(title=f"{split.upper()} 集 PR 曲线（少数类视角，与指标表同口径）",
                      xaxis_title="Recall", yaxis_title="Precision",
                      height=430, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def render() -> None:
    st.header("🤖 模型评估")
    st.caption("指标来自 `reports/model_metrics.json`（阶段二真实训练产物）；曲线由 `predictions_*.csv` 实时计算")

    split = st.radio("数据集", ["val", "test"],
                     format_func={"val": "验证集（调参/选型用）", "test": "测试集（仅最终评估）"}.get,
                     horizontal=True)

    metrics = load_metrics_json()["metrics"][split]
    rows = []
    for name, d in metrics.items():
        rows.append({
            "模型": name, "Precision": d["precision"], "Recall": d["recall"], "F1": d["f1"],
            "ROC-AUC": d["roc_auc"], "PR-AUC": d["pr_auc"],
            "TP": d["tp"], "FP": d["fp"], "FN": d["fn"], "TN": d["tn"],
        })
    st.dataframe(pd.DataFrame(rows).set_index("模型"), width="stretch")
    st.info("上表为阈值 0.5 的结果。极端不平衡（欺诈率约 0.13%）下 Accuracy 无信息量——"
            "把全部交易判正常即可得到 99.87% Accuracy 但漏掉所有欺诈。", icon="⚠️")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_roc_fig(split), width="stretch")
    with c2:
        st.plotly_chart(_pr_fig(split), width="stretch")

    st.subheader("阈值 0.5 下的混淆矩阵（实时计算）")
    cols = st.columns(3)
    for col, (name, abbr) in zip(cols, MODEL_OPTIONS.items()):
        d = load_pred(split, abbr)
        y, p = d["Class"].to_numpy(), d["predicted_probability"].to_numpy()
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        cm = pd.DataFrame(
            [[fmt_int(tn), fmt_int(fp)], [fmt_int(fn), fmt_int(tp)]],
            index=["实际 正常", "实际 欺诈"],
            columns=["预测 正常", "预测 欺诈"],
        )
        with col:
            st.markdown(f"**{name}**")
            st.table(cm)

    st.divider()
    st.subheader("模型选择理由（阶段二结论）")
    m = load_metrics_json()
    st.markdown(
        f"""
        - **调参唯一依据 = 验证集 PR-AUC**（正样本率约 0.13% 时 ROC-AUC 被海量真负样本抬高，PR-AUC 才反映抓欺诈能力）；
        - 最终模型：**{m['final_model_by_val_pr_auc']}**（验证集 PR-AUC 最高，测试集方向一致）；
        - 但 Random Forest 与 XGBoost 的 PR-AUC 差距（0.009）在 55 笔验证欺诈的抽样噪声之内，
          **不构成显著优劣结论**；XGBoost 排序能力（ROC-AUC 0.989）实际最强；
        - Logistic Regression 是可解释基线：排序能力不弱（val PR-AUC 0.837），
          但 0.5 阈值下 Precision 仅 0.05，直接上线不可用——问题在工作点，不在模型。
        """
    )
