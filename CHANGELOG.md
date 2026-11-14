# 更新日志

本项目的所有重要变更都记录在此，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循粗略的语义化：`0.x` 阶段以"档案结构是否变化"为主要判断依据。

## [Unreleased]

### 新增
- `scripts/check_repo_health.py`：零依赖仓库自检——文档链接可达性、关键产物齐全度、
  `data/` 目录纪律、两份冻结档案的 sha256 指纹。
- `docs/development.md`：L0–L5 验证阶梯、提交规范、数据与数值纪律、真实踩坑清单。
- `pyproject.toml`：项目元数据与 ruff / pytest 配置（规则集 E4/E7/E9/F，行宽 120）。
- `.github/workflows/ci.yml`：lint / compile / test 三个并行 job；`requirements-test.txt` 提供最小测试依赖。

### 修复
- `src/data/predictions.py`、`src/evaluation/metrics.py`：类型注解引用了未在模块级导入的
  `np` / `pd`，在 `typing.get_type_hints()`、Sphinx、FastAPI 等解析注解的场景会抛 `NameError`。
  运行时因 `from __future__ import annotations` 而"看起来正常"，属于延迟暴露的缺陷。

### 变更
- 清理 8 处未使用导入与 1 处无占位符的 f-string（ruff F401/F541），不改任何行为与数值。

## [0.1.1] - 2026-11-10

### 变更
- README 补入 5 张关键图表（三模型对比、验证集 PR、阈值-成本曲线、三策略成本、成本比敏感性），
  并清理指向非公开文档的引用，保证克隆后所有引用均可解析。

## [0.1.0] - 2026-11-10

### 新增
- 阶段一：数据质量体检（6 项检查 → `reports/data_quality_report.md|json`）、EDA（7 图 + 1 交互图）、
  SQL 分析（SQLite 上 9 条查询，覆盖 CTE / RANK / LAG / NTILE / 窗口累加 / 手写中位数）。
- 阶段二：LR / RF / XGBoost 三模型，按 `Time` 70/15/15 时间序列切分，以验证集 PR-AUC 选型，
  产出 `reports/model_metrics.json` 与验证/测试集预测文件。
- 阶段三：0.001 步长阈值扫描（1,001 点）、漏判/误拦成本模型、三档业务策略与冻结评估、
  1D/2D 敏感性分析，产出 `reports/threshold_analysis.json` 与 `reports/strategy_comparison.csv`。
- Streamlit 五页看板（数据概览 / 模型评估 / 阈值分析 / 策略成本 / 风险报告），
  全部数字实时计算或读自落盘档案，无硬编码指标。
- `tests/`：58 项用例（0.1.0 交付时实测 58/58 通过）。

[Unreleased]: https://github.com/JamesHardon178/transaction-risk-platform/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/JamesHardon178/transaction-risk-platform/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/JamesHardon178/transaction-risk-platform/releases/tag/v0.1.0
