# 开发与验证指南

> 本文回答一个问题：**改完代码，怎么用最低成本确认"没改坏"？**
> 结论先行——按下面的验证阶梯从 L0 往上走，任何一级失败就停下来修，不要跳级。

## 1. 环境准备

项目依赖固定为 Python 3.13（Windows x64 实测；conda/pyenv 均可）。

```bash
# 完整环境（跑 Notebook、Streamlit、SQL 复现都需要）
python -m venv .venv
.venv\Scripts\activate            # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt

# 只跑静态检查 + 单元测试的最小环境
pip install -r requirements-test.txt
```

`.venv/`、`data/raw/`、`data/processed/` 均不入库（见 `.gitignore`），
所以**克隆一份干净仓库时没有任何交易数据**：这不是缺文件，是设计如此。

## 2. 验证阶梯

| 级别 | 命令 | 覆盖范围 | 需要什么 |
|---|---|---|---|
| **L0 编译** | `python -m compileall -q src app scripts tests` | 全部模块的语法与导入图 | 仅 Python |
| **L1 静态检查** | `ruff check .` | 未定义名（F821）、未使用导入（F401）、空 f-string（F541） | ruff |
| **L2 单元测试** | `pytest -q --ignore=tests/test_app.py` | 51 项纯合成数据用例（特征/指标/规则/绘图/数据准备） | `requirements-test.txt` |
| **L3 全量测试** | `pytest -q` | 58 项，额外含五页 Streamlit AppTest | 全量依赖 + `data/processed` |
| **L4 端到端复现** | Notebook 01 → 02 → `scripts/build_stage3_analysis.py` → `scripts/run_sql.py` | 指标、阈值、成本、SQL 结果与归档逐字节一致 | 全量依赖 + 原始 CSV |
| **L5 仓库自检** | `python scripts/check_repo_health.py` | 文档链接、关键产物、数据目录纪律、冻结档案指纹 | 仅 Python + git |

**为什么 L2 比 L3 少 7 项**：`tests/test_app.py` 通过官方 AppTest 真实渲染五个页面，
断言"总交易数 284,807"这类来自 `data/processed/transactions_clean.parquet` 的真实口径数字。
数据不入库 → 无数据环境下该文件必然失败，因此 CI 与本地快速验证都显式跳过它，
**而不是**把它改成 mock（mock 掉这层就失去"看板数字来自真实产物"的验证意义）。

## 3. 提交规范

采用 Conventional Commits，`type` 取值限于下表；`scope` 可选，用于标出影响范围。

| type | 用途 | 本项目示例 |
|---|---|---|
| `feat` | 新增能力（含脚本、模块） | `feat(scripts): 新增仓库自检脚本` |
| `fix` | 修缺陷 | `fix: 补齐类型注解引用的缺失导入` |
| `docs` | 只改文档 | `docs: 新增开发与验证指南` |
| `style` | 不改行为的代码整理 | `style: 清理未使用的导入` |
| `refactor` | 结构重组、行为不变 | — |
| `test` | 测试相关 | — |
| `chore` | 构建、配置、元数据 | `chore: 添加 pyproject.toml` |
| `ci` | CI/CD 配置 | `ci: 添加 GitHub Actions 工作流` |

三条硬要求：

1. **一个提交只做一件事**——代码整理与行为变更不能混在同一条里，否则 review 与回滚都会失真；
2. **提交信息写"为什么"**，正文解释动机、影响面与验证方式，而不是复述 diff；
3. **提交前至少过 L0 + L1**，涉及规则的改动再加 L2。

## 4. 数据与数值纪律

- `data/raw/`（原始 CSV，只读）与 `data/processed/`（parquet / SQLite）**一律不入库**：
  体积大，且原始字段虽已脱敏仍按准敏感数据处理。
- `reports/model_metrics.json` 与 `reports/threshold_analysis.json` 是**冻结档案**：
  数值只能由实验重跑产生，任何手工改动都视为破坏可复现性。
  用 `python scripts/check_repo_health.py` 打印的 sha256 短指纹做跨提交比对。
- 测试集只出现一次（最终评估），不参与模型选择、阈值寻优与策略调参。

## 5. 常见陷阱（真实踩过的）

| 现象 | 原因 | 处理 |
|---|---|---|
| `FileNotFoundError: predictions_val_rf.csv` | 阶段二 Notebook 未跑，预测文件不存在 | 先跑 `notebooks/02_model_training.ipynb`，或 L2 级跳过 `test_app.py` |
| AppTest 断言 `284,807` 失败 | 读成了去重后的 283,726 行口径 | 确认使用的是原始口径的清洗层数据，勿把去重结果覆盖回去 |
| 图表数量"少一张" | `fig08_hourly_interactive.html` 是交互图，不是 PNG | 全量为 20 PNG + 1 HTML = 21 |
| ruff 报 F821 但脚本能跑 | `from __future__ import annotations` 把注解变成字符串，运行时不求值 | 补模块级导入（参考 `cf5b008`），让 `typing.get_type_hints()` 也能解析 |
| CI 测试 job 失败但本地全绿 | CI 只装 `requirements-test.txt`，无 streamlit | CI 已 `--ignore=tests/test_app.py`；新增测试若依赖 streamlit 需同步改 CI |

## 6. 持续集成

`.github/workflows/ci.yml` 在推送到 `main` 与对 `main` 的 PR 上运行三个并行 job：

| job | 内容 | 失败含义 |
|---|---|---|
| `lint` | `ruff check`（规则取自 `pyproject.toml`） | 存在未定义名/未使用导入等确定性问题 |
| `compile` | `compileall src app scripts tests` | 存在语法错误或包结构断裂 |
| `test` | `pytest --ignore=tests/test_app.py` | 单元行为与约定不符 |

三个 job 都不接触私有数据，因此**任何人 fork 后都能复现绿灯**——这是公开仓库的基本礼貌。
