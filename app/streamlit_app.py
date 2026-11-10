"""交易风险阈值优化与策略分析平台（Streamlit 入口）。

五个页面：
1. 数据概览        —— 真实清洗后数据（parquet）
2. 模型评估        —— 阶段二指标档案 + 预测 CSV 实时曲线
3. 风险阈值分析    —— 滑块实时重算
4. 策略成本优化    —— 可配置成本参数 + 敏感性分析
5. 风险分析报告    —— 冻结策略与结论

运行：streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 保证 `src` 可导入（无论从哪个目录启动）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.page_models import render as render_models        # noqa: E402
from app.page_overview import render as render_overview    # noqa: E402
from app.page_report import render as render_report        # noqa: E402
from app.page_strategy import render as render_strategy    # noqa: E402
from app.page_threshold import render as render_threshold  # noqa: E402

st.set_page_config(
    page_title="交易风险识别与策略优化平台",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "数据概览": render_overview,
    "模型评估": render_models,
    "风险阈值分析": render_threshold,
    "策略成本优化": render_strategy,
    "风险分析报告": render_report,
}


def main() -> None:
    # 深链接支持：?page=模型评估 可直达指定页面（便于分享与演示）
    qp = st.query_params.get("page")
    pages = list(PAGES)
    default_index = pages.index(qp) if qp in pages else 0

    with st.sidebar:
        st.title("🛡️ 交易风险平台")
        st.caption("Kaggle Credit Card Fraud Detection · 三阶段分析")
        page = st.radio("页面", pages, index=default_index, label_visibility="collapsed")
        st.divider()
        st.markdown(
            "**数据与口径**\n"
            "- 原始数据只读：`data/raw/creditcard.csv`\n"
            "- 阈值/策略仅用验证集确定，测试集一次性评估\n"
            "- 成本参数均为实验假设"
        )
        st.caption(f"项目根目录：`{PROJECT_ROOT.name}`")

    PAGES[page]()


if __name__ == "__main__":
    main()
