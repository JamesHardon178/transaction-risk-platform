"""页面 1：数据概览。数据来自 transactions_clean.parquet（真实清洗后数据）。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.common import fmt_int, load_clean_data


def render() -> None:
    st.header("📊 数据概览")
    st.caption("数据源：Kaggle Credit Card Fraud Detection → 清洗后数据 `transactions_clean.parquet`（原始 CSV 只读）")

    df = load_clean_data()
    n = len(df)
    n_fraud = int(df["Class"].sum())
    n_normal = n - n_fraud
    fraud_ratio = n_fraud / n
    # 建模口径：阶段二按全部 31 列去重 → 283,726。该数字取自 model_metrics.json
    # 的真实切分记录数之和（train+val+test），不能用本表 3 列做 drop_duplicates
    # （3 列重复 ≠ 全列重复，会得到错误的 279,950）
    from app.common import load_metrics_json
    splits = load_metrics_json()["splits"]
    n_dedup = int(sum(s["记录数"] for s in splits))
    n_fraud_dedup = int(sum(s["欺诈数"] for s in splits))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总交易数（原始）", fmt_int(n),
              help=f"建模前已去重为 {fmt_int(n_dedup)}（阶段二口径，欺诈 {fmt_int(n_fraud_dedup)} 笔）")
    c2.metric("正常交易", fmt_int(n_normal))
    c3.metric("欺诈交易", fmt_int(n_fraud))
    c4.metric("欺诈比例（原始）", f"{fraud_ratio:.4%}")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("交易金额分布")
        amt = df["Amount"]
        # 金额强右偏（偏度 16.98），直方图截取到 P99 保证可读，全部样本进入统计
        p99 = float(amt.quantile(0.99))
        fig = px.histogram(
            df, x="Amount", nbins=80, range_x=(0, p99),
            color_discrete_sequence=["#7A9A3B"],
            labels={"Amount": "交易金额（€）"},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          title=f"金额直方图（截取 0~P99={p99:,.0f}€ 显示，统计含全部 {fmt_int(n)} 笔）")
        st.plotly_chart(fig, width="stretch")

        box = px.box(df, x="Class", y="Amount", color="Class",
                     color_discrete_map={0: "#3465A4", 1: "#CC4444"},
                     category_orders={"Class": [0, 1]},
                     labels={"Class": "类别（0=正常 1=欺诈）", "Amount": "金额（€）"})
        box.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                          title="两类金额箱线图（含离群点， Fraud 中位数反而更低）")
        st.plotly_chart(box, width="stretch")

    with right:
        st.subheader("数据基本信息")
        info = pd.DataFrame({
            "项目": ["记录数", "字段数（本表）", "缺失值", "完全重复行", "Time 覆盖",
                     "金额最小值", "金额中位数", "金额均值", "金额最大值",
                     "金额 P99", "零金额交易数"],
            "值": [
                f"{fmt_int(n)}（去重后 {fmt_int(n_dedup)}）",
                "3",
                fmt_int(int(df.isna().sum().sum())),
                "0（建模前已去重；原始数据 1,081 行，见阶段一报告）",
                f"{df['Time'].min():.0f} ~ {df['Time'].max():.0f} 秒（≈ {df['Time'].max()/3600:.1f} 小时）",
                f"€ {df['Amount'].min():.2f}",
                f"€ {df['Amount'].median():.2f}",
                f"€ {df['Amount'].mean():.2f}",
                f"€ {df['Amount'].max():,.2f}",
                f"€ {df['Amount'].quantile(0.99):,.2f}",
                fmt_int(int((df["Amount"] == 0).sum())),
            ],
        })
        st.table(info)

        st.subheader("金额分布统计口径说明")
        st.markdown(
            f"""
            - 金额分布**强右偏**：均值 €{df['Amount'].mean():.2f} 远大于中位数 €{df['Amount'].median():.2f}；
            - 阶段一已证实：欺诈金额中位数（€9.25）**低于**正常交易（€22.00），
              "金额越大越可疑"的直觉在该数据集上不成立；
            - V1~V28 为 PCA 匿名特征，本页面仅展示 Time / Amount / Class 三个原始字段。
            """
        )
