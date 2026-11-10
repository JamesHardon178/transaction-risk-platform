"""页面 5：风险分析报告。最终推荐策略（冻结产物）+ 成本假设 + 局限性 + 结论。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.common import FIGURES, fmt_int, load_analysis_json, load_strategy_csv


def render() -> None:
    st.header("📋 风险分析报告")
    st.caption("最终推荐策略为冻结产物（`threshold_analysis.json`），测试集仅做过一次性评估")

    a = load_analysis_json()
    frozen = a["frozen_strategy"]
    cost = a["cost_assumptions"]

    st.subheader("1. 最终推荐策略")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("推荐模型", frozen["model"])
    c2.metric("推荐策略", frozen["strategy"])
    c3.metric("推荐阈值", f"{frozen['threshold']:.3f}")
    val_m = a["val_metrics_frozen"]["平衡策略"]
    test_m = a["test_metrics_frozen"]["平衡策略"]
    c4.metric("验证集总成本（实验假设）", f"{val_m['cost']:,.0f}")

    st.markdown(
        f"""
        **推荐理由**：在验证集上，平衡策略（成本最小化）以总成本 {val_m['cost']:,.0f} 优于
        保守策略 {a['val_metrics_frozen']['保守策略']['cost']:,.0f} 与激进策略
        {a['val_metrics_frozen']['激进策略']['cost']:,.0f}（成本假设：漏判 {cost['cost_fn']:.0f} / 误拦截 {cost['cost_fp']:.0f}）。
        阈值一旦确定即冻结，**未根据测试集结果做任何回调**。
        """
    )

    st.subheader("2. 冻结策略的验证集 vs 测试集表现（一次性评估）")
    rows = []
    for k, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                     ("tp", "TP"), ("fp", "FP"), ("fn", "FN"), ("cost", "实验假设总成本")):
        rows.append({"指标": label, "验证集": val_m[k], "测试集": test_m[k],
                     "差值": round(float(test_m[k]) - float(val_m[k]), 4)})
    st.dataframe(pd.DataFrame(rows).set_index("指标"), width="stretch")
    st.warning("测试集召回下降约 14 个百分点（0.891 → 0.750）：验证/测试各只有 55/52 笔欺诈，"
               "指标波动在抽样噪声范围内，同时提示时间分布上可能存在欺诈模式漂移。"
               "**测试集上激进策略成本（1,350）虽低于平衡策略（1,465），但不能据此回调阈值——"
               "否则就是用测试集调参，违反评估纪律。**", icon="⚠️")

    st.subheader("3. 三策略完整对比（含 XGBoost 对照）")
    st.dataframe(load_strategy_csv(), width="stretch", hide_index=True)

    st.subheader("4. 成本假设说明")
    st.markdown(
        f"""
        - 单笔漏判成本 **{cost['cost_fn']:.0f}** / 单笔误拦截成本 **{cost['cost_fp']:.0f}**；
        - 这两个数字是**演示用实验假设**，不代表任何真实金融机构的损失/运营成本；
        - 总成本公式：`总成本 = 漏判数量 × 漏判成本 + 误拦截数量 × 误拦截成本`；
        - 真实业务中漏判成本 ≈ 被盗刷金额 + 追偿成本，误拦截成本 ≈ 客户流失/客服/复核成本，
          两者随商户、客群、地区差异巨大——因此阶段三把参数做成**可配置**而非写死。
        """
    )

    st.subheader("5. 模型局限性")
    st.markdown(
        """
        1. **评估样本小**：验证集 55 笔 / 测试集 52 笔欺诈，指标置信区间宽，小差距不下强结论；
        2. **时间跨度仅 48 小时**：无法验证欺诈模式随时间漂移的稳健性；
        3. **V1~V28 为 PCA 匿名特征**：无法映射到具体业务字段，可解释性受限；
        4. **XGBoost 概率未校准**：scale_pos_weight 整体抬高概率，不能按字面理解为欺诈概率；
        5. **成本假设是实验性的**：推荐阈值依附于假设的成本结构，换业务须重新标定；
        6. **单一数据集**：结论不能外推到其他卡组织/地区/商户结构。
        """
    )

    st.subheader("6. 分析结论")
    st.markdown(
        """
        1. 阈值 0.5 不是最优工作点：Random Forest 在 0.5 的实验假设成本为 1,400，成本最优点 0.144 仅 735，**降低约 47%**；
        2. 最优阈值由成本结构决定：漏判/误拦截成本比从 1:1 拉到 200:1，最优阈值从 0.409 滑到 0.036；
        3. 三种策略对应三种业务目标：保守（召回底线 90%，漏判最少 5 笔但误拦 245 笔）、平衡（成本最小）、激进（精确底线 95%，几乎零打扰）；
        4. 线性基线 LR 的排序能力可用，但默认阈值下误拦 965 笔，说明**工作点选择与模型能力同样重要**；
        5. 本平台的全部数字均可由 `scripts/build_stage3_analysis.py` 与两个 Notebook 复现，无任何手工编造。
        """
    )

    st.divider()
    fig = FIGURES / "fig21_val_test_delta.png"
    if fig.exists():
        st.image(str(fig), caption="图 21  平衡策略在验证集与测试集上的表现差异")
    fig16 = FIGURES / "fig16_threshold_curves.png"
    if fig16.exists():
        st.image(str(fig16), caption="图 16  Random Forest 阈值-指标曲线与三策略位置")
