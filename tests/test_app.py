"""Streamlit 平台的渲染测试：用官方 AppTest 框架真实执行五个页面。

要求：每个页面渲染无异常、核心内容出现、且数字来自真实产物。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


@pytest.fixture(scope="module")
def at():
    from streamlit.testing.v1 import AppTest
    t = AppTest.from_file(str(APP), default_timeout=300)
    t.run()
    return t


def _goto(at, page: str):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"页面 {page} 渲染异常: {at.exception}"


def test_sidebar_has_five_pages(at):
    labels = at.sidebar.radio[0].options
    assert labels == ["数据概览", "模型评估", "风险阈值分析", "策略成本优化", "风险分析报告"]


def test_page_overview_renders_real_numbers(at):
    _goto(at, "数据概览")
    metric_text = "\n".join(str(m.value) for m in at.metric)
    table_text = "\n".join(str(tb.value) for tb in at.table)
    # 原始 284,807 / 建模去重后 283,726（阶段一/二文档化的事实，作为回归锚点）
    assert "284,807" in metric_text, "总交易数应来自真实 parquet（原始口径）"
    assert "283,726" in table_text, "应同时展示建模去重口径（来自 model_metrics.json 切分记录数）"
    assert not at.exception


def test_page_models_renders_metrics_table(at):
    _goto(at, "模型评估")
    body = "\n".join(str(md.value) for md in at.markdown) + "\n".join(str(b.value) for b in at.button)
    assert "Random Forest" in body
    # 指标档案中的最终模型名应出现在页面
    final = json.loads((REPORTS / "model_metrics.json").read_text(encoding="utf-8"))["final_model_by_val_pr_auc"]
    assert final in body


def test_page_threshold_slider_updates(at):
    _goto(at, "风险阈值分析")
    assert at.slider, "页面 3 应包含阈值滑块"
    before = [str(m.value) for m in at.metric]
    at.slider[0].set_value(0.1).run()
    assert not at.exception
    after = [str(m.value) for m in at.metric]
    assert before != after, "滑块变化后指标应实时更新"


def test_page_strategy_renders_table(at):
    _goto(at, "策略成本优化")
    text = "\n".join(str(e.value) for e in at.markdown)
    df_text = "\n".join(str(d.value) for d in at.dataframe)
    assert "平衡策略" in text or "平衡策略" in df_text
    # "实验假设"必须显式出现（st.warning / st.caption / markdown 任一）
    surfaces = text + "\n".join(str(e.value) for e in at.caption)
    surfaces += "\n".join(str(e.value) for e in getattr(at, "warning", []))
    assert "实验假设" in surfaces, "成本页必须显式标注实验假设"
    assert not at.exception


def test_page_report_frozen_threshold(at):
    _goto(at, "风险分析报告")
    a = json.loads((REPORTS / "threshold_analysis.json").read_text(encoding="utf-8"))
    th = f"{a['frozen_strategy']['threshold']:.3f}"
    text = "\n".join(str(m.value) for m in at.metric)
    assert th in text, "报告页应展示冻结阈值"
    assert "实验假设" in "\n".join(str(md.value) for md in at.markdown)


def test_number_input_changes_cost(at):
    _goto(at, "策略成本优化")
    # 把漏判成本改为 500（误拦截保持默认），页面应正常重算
    for ni in at.number_input:
        if "漏判" in str(ni.label):
            ni.set_value(500.0).run()
            break
    assert not at.exception
