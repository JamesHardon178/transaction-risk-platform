# 基于交易数据的风险识别与策略优化分析平台

> 信用卡欺诈是典型的**极端类别不平衡**问题（欺诈仅占 0.17%）：
> 把所有交易判为"正常"就有 99.83% 的 Accuracy，却漏掉每一笔真实损失。
> 本项目从一份真实的两天交易流水出发，完成
> **数据体检 → 探索分析（EDA + SQL）→ 三模型对比 → 阈值-成本-策略优化 → 可视化交付** 的完整闭环，
> 并把"业务成本假设 → 最优工作点"的映射做成可配置、可复现的分析平台。

**项目状态：三阶段全部完成并通过最终验收**（pytest 58/58、两本 Notebook 从头执行零报错、SQL 结果逐字节复现、Streamlit 五页渲染正常）。

---

## 1. 项目背景与业务问题

- **数据**：[Kaggle · Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)（ULB 布鲁塞尔自由大学，2015），2013 年 9 月欧洲持卡人 48 小时内共 **284,807 笔**真实信用卡交易。
- **业务问题**：
  1. 哪些交易特征与欺诈相关？（金额、时段、V 特征分布）
  2. 如何在不平衡数据上可靠地识别欺诈？（模型选型与评估口径）
  3. 模型概率应该取什么阈值拦截？（漏判 vs 误拦截的成本权衡）
- **核心矛盾**：1 : 577.9 的类别不平衡 + 单一业务口径下的成本敏感决策。

## 2. 项目功能

| 功能 | 说明 |
|---|---|
| 数据质量体检 | 缺失/重复/类型/标签/时间范围 6 项检查，输出人读 + 机读报告 |
| EDA 分析 | 类别不平衡、金额分布、时段模式、Mann-Whitney 检验、相关性、关键特征对比（7 图） |
| SQL 分析 | SQLite 上 9 条查询：分组聚合、CTE、RANK/LAG/NTILE/窗口累加、SQL 手写中位数 |
| 建模与评估 | LR / RF / XGBoost 三模型，时间序列切分，PR-AUC 选型，过拟合检查 |
| 阈值优化 | 验证集上 0.001 步长扫描 1,001 点，成本最优阈值 + 三种业务策略 |
| 敏感性分析 | 1D（漏判成本→最优阈值）与 2D（漏判×误拦）热力图 |
| Streamlit 平台 | 五页交互看板，全部数字实时计算或读自落盘档案，无硬编码指标 |

## 3. 技术栈

- **语言/环境**：Python 3.13.7（项目内独立 `.venv`，Windows x64）
- **数据处理**：pandas 2.3.3、NumPy 2.3.5、SciPy 1.18.1、PyArrow（Parquet）
- **机器学习**：scikit-learn 1.9.1（LR/RF/标准化/指标）、xgboost 3.4.1
- **数据库**：SQLite（标准库 sqlite3，库文件 89.5 MB）
- **可视化**：Matplotlib、Seaborn、Plotly（交互图）
- **应用**：Streamlit 1.66.1（五页看板 + 官方 AppTest 渲染测试）
- **质量保障**：pytest 9.2.0（58 用例）、Jupyter nbconvert 从头执行验证
- 完整锁定见 `requirements.txt`（顶层）与 `requirements-lock.txt`（130 包全量）

## 4. 整体架构

```
data/raw/creditcard.csv (只读, 284,807×31)
        │
        ├── scripts/check_data_quality.py ──► reports/data_quality_report.md/.json
        ├── scripts/build_database.py ──────► data/processed/transactions.db (SQLite)
        │                                          │
        │                                          └── sql/*.sql + scripts/run_sql.py
        │                                                ──► reports/sql_results.md
        └── notebooks/01_data_exploration.ipynb (EDA, 7 图)
                ──► data/processed/transactions_clean.parquet (去重+派生标记)
                        │
                        └── notebooks/02_model_training.ipynb
                                │   src/data/preparation.py   去重/时间切分(70/15/15)
                                │   src/features/features.py  特征白名单(防泄漏)
                                │   src/models/train.py       LR/RF/XGB + 验证集网格选参
                                │   src/evaluation/*          指标与 ROC/PR/混淆矩阵图
                                └──► data/processed/predictions_{val,test}_{lr,rf,xgb}.csv
                                        │
                                        └── scripts/build_stage3_analysis.py
                                                │  src/rules/thresholds.py  阈值扫描
                                                │  src/rules/cost.py        成本模型/敏感性
                                                │  src/rules/strategies.py  三策略/冻结/一次性评估
                                                └──► reports/threshold_analysis.json
                                                      reports/strategy_comparison.csv
                                                      reports/figures/fig16~21
                                        │
                                        └── app/streamlit_app.py + page_*.py（五页看板）
```

模块级依赖与数据流详见 `docs/architecture.md`。

## 5. 数据来源与字段说明

| 字段 | 含义 | 说明 |
|---|---|---|
| `Time` | 距首笔交易的秒数（0~172,792） | **不是时间戳**，无日期与时区，只能做相对时段分析 |
| `V1`~`V28` | PCA 变换后的匿名特征 | 禁止赋予业务含义；原始字段（商户/卡种/地区）未公开 |
| `Amount` | 交易金额（欧元） | 唯一保留量纲的原始业务变量 |
| `Class` | 标签：1 = 欺诈，0 = 正常 | 依据 Kaggle 官方说明并用数据交叉验证（见 `docs/data_dictionary.md`） |

完整字段字典（含派生字段与处理口径）见 **`docs/data_dictionary.md`**。

## 6. 数据分析方法

1. **质量体检**（`scripts/check_data_quality.py`，只读原始数据）：零缺失、无类型问题；
   发现 1,081 行完全重复（0.38%，其中 19 笔欺诈）与 1,825 笔零金额交易。
2. **EDA**（Notebook 01，7 图 + 1 交互图）：
   - 金额：强右偏（偏度 16.98），欺诈中位数 9.25€ **低于**正常 22.00€（Mann-Whitney p=8.6e-06，效应量仅 -0.1159——显著 ≠ 可区分）；
   - 时段：相对第 2 时段欺诈率 1.7127%，为全局的约 10 倍；
   - 特征：V17/V14/V12/V10 最强（欺诈类整体左移 4~7 个标准差）；V1~V28 完全正交，但 Amount↔V2 相关 -0.531；
   - 负面结论：相邻交易间隔两类中位数均为 0，无区分力，**不进特征池**。
3. **SQL 分析**（SQLite，9 条查询，覆盖 CTE / RANK / LAG / NTILE / 窗口累加 / 手写中位数）：
   独立口径复核 EDA 结论，例如"金额十分位欺诈率"推翻"金额越大越可疑"的直觉。

## 7. 模型训练与评估

### 7.1 数据处理与切分

- 去重：仅删完全重复（Class 冲突检查按 30 个特征列分组，实测 0 组冲突），284,807 → **283,726 行 / 473 笔欺诈**；去重在切分**之前**，防重复样本泄漏；
- 切分：按 `Time` 升序 **70/15/15** 时间序列切分（train 198,608/366 欺诈、val 42,559/55、test 42,559/52），模拟"用过去训练、对未来评估"；禁用随机切分；
- **已声明的限制**：验证/测试各只有 55/52 笔欺诈，每笔影响 Recall 约 1.8~1.9pp，指标噪声大。

### 7.2 三模型与不平衡处理

| 模型 | 不平衡处理 | 关键设置 |
|---|---|---|
| Logistic Regression | `class_weight="balanced"` | StandardScaler 只在训练集 fit（Pipeline 内） |
| Random Forest | `class_weight="balanced_subsample"` | 400 棵 × depth 14（验证集网格选出） |
| XGBoost | `scale_pos_weight=541.6` | `eval_metric=aucpr` + 验证集 early stopping（93 轮） |

**不用 SMOTE**：V1-V28 是 PCA 空间坐标，线性插值不满足流形结构，且会破坏时间分布。

### 7.3 评估协议与真实结果

- 选型唯一依据 = **验证集 PR-AUC**（用 `average_precision_score`，不用梯形法）；测试集只在最终评估出现一次；
- 禁止只看 Accuracy：全判正常即 99.87%，但欺诈全漏。

验证集（阈值 0.5）：

| 模型 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.0502 | **0.9273** | 0.0952 | 0.9824 | 0.8367 |
| **Random Forest** | 1.0000 | 0.7455 | **0.8542** | 0.9664 | **0.8735** |
| XGBoost | 0.4138 | 0.8727 | 0.5614 | **0.9886** | 0.8646 |

测试集（一次性评估）：RF PR-AUC 0.7820 > XGB 0.7569 > LR 0.7101，排序方向一致；
val+test 合并口径（107 笔欺诈）复核同为 RF > XGB > LR。

**最终模型：Random Forest**。诚实说明：RF 与 XGB 的 PR-AUC 差距（0.0089）在 55 笔验证欺诈的抽样噪声内，不构成"显著更优"的统计结论。

## 8. 风险策略优化方法

- 阈值扫描：验证集上 0.001 步长 1,001 点（O(T log n) 向量化实现），外加 13 个演示阈值；
- **策略规则先于数据写死**（`src/rules/strategies.py` 常量），不按结果反调：
  | 策略 | 规则 | 验证集结果（RF） |
  |---|---|---|
  | 保守 | Recall ≥ 90% 下 Precision 最高 | 阈值 0.041，Recall 0.909，漏判仅 5 笔 |
  | 平衡 | 总成本最小 | **阈值 0.144，成本 735（最低）** |
  | 激进 | Precision ≥ 95% 下 Recall 最高 | 阈值 0.287，Precision 0.957，误拦仅 2 笔 |
- 冻结 **RF + 平衡策略 0.144** → 测试集**仅评估一次**，不得回调。

## 9. 成本模型

```
总成本 = 漏判数(FN) × 单笔漏判成本 + 误拦截数(FP) × 单笔误拦截成本
```

- **漏判 100 / 误拦截 5 是实验假设**（演示成本权衡用，不代表任何真实机构数据）；
- 该假设下：默认 0.5 阈值成本 1,400 → 成本最优 0.144 成本 735，**降低 47.5%**；
- 敏感性（实测，单调不增，pytest 断言保证）：成本比 1:1→1000:1 时最优阈值 0.409→0.017。
  **阈值不是常数，而是成本结构的函数**——任何固定阈值（包括 0.5）都隐含一个未声明的成本假设。

## 10. 项目运行方法

> **数据不入库**（原始 210 MB + 产物 169 MB，且受 Kaggle 数据使用条款约束）。
> 克隆后第一步先取数据，脚本会校验 284,807×31、492 笔欺诈三项指标；详见 `data/README.md`。

```bash
cd transaction-risk-platform

# 0) 获取原始数据（首次必做，之后自动跳过）
python scripts/download_data.py             # 需 kaggle CLI 或手动下载，见 data/README.md

# 1) 环境（已有 .venv 可跳过）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m ipykernel install --user --name transaction-risk --display-name "Python 3.13 (transaction-risk)"

# 2) 激活（Git Bash）：source .venv/Scripts/activate   （CMD：.venv\Scripts\activate.bat）

# 3) 数据流水线（原始数据已就位时按需重跑）
python scripts/check_data_quality.py        # 数据质量体检（只读原始 CSV）
python scripts/build_database.py            # 重建 SQLite 库
python scripts/run_sql.py                   # 执行 9 条 SQL → reports/sql_results.md

# 4) Notebook（Jupyter 内核选 transaction-risk）
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_training.ipynb

# 5) 阶段三分析（阈值扫描→策略→冻结→测试集一次性评估→敏感性，一键复现）
python scripts/build_stage3_analysis.py

# 6) 看板
streamlit run app/streamlit_app.py          # 浏览器打开 http://localhost:8501
# 支持 ?page=模型评估 深链接直达指定页面

# 7) 测试
pytest -q                                   # 58 个用例
```

> 阶段三的看板与脚本读取阶段二落盘的预测文件；从零复现需先跑完两本 Notebook。

## 11. 实际实验结果（全部可复现）

### 阶段一：数据与 EDA

- 284,807 笔 × 31 字段，零缺失；欺诈 492 笔（0.1727%），不平衡比 577.9:1
- 欺诈金额中位数 9.25€ < 正常 22.00€（p=8.6e-06，效应量 -0.1159）
- 相对第 2 时段欺诈率 1.7127%（约全局 10 倍）；>5,000€ 的 55 笔**零欺诈**
- 最强特征 V17/V14/V12/V10；负面结论：相邻交易间隔无区分力

### 阶段二：三模型（详见 `reports/stage2_model_report.md`）

- 去重 283,726 行 / 473 欺诈；时间切分 train 198,608(366) / val 42,559(55) / test 42,559(52)
- 验证集 PR-AUC：**RF 0.8735** > XGB 0.8646 > LR 0.8367（测试集 0.7820/0.7569/0.7101，方向一致）
- 过拟合检查：RF train-val 差 +0.109（已用 depth/min_samples_leaf 压制）、XGB +0.066 最轻

### 阶段三：阈值与策略（详见 `reports/stage3_strategy_report.md`）

- 成本最优阈值 **0.144**（成本 735），比默认 0.5（成本 1,400）**低 47.5%**
- 三策略：保守 0.041（R 0.909）/ 平衡 0.144（成本最小 735）/ 激进 0.287（P 0.957）
- 冻结 RF+平衡 0.144 → 测试集一次性评估：P 0.542 / R 0.750 / F1 0.629 / 成本 1,465（**比验证集 +99.3%**）
- 成本上升的解读：55/52 笔欺诈的抽样噪声 + 可能的时间漂移；按纪律**不回调**
- 敏感性：成本比 1:1→1000:1，最优阈值 0.409→0.017（单调不增）

### 验收结果（2026-11-08 实测）

| 检查项 | 结果 |
|---|---|
| pytest | **58/58 通过**（15.0s） |
| Notebook 01 | 从头执行 43 单元零报错，7 图内嵌 |
| Notebook 02 | 从头执行 30 单元零报错，7 图内嵌，指标与归档一致（确定性复现） |
| SQL | 9 条查询重跑，`sql_results.md` 逐字节一致（md5 相同） |
| 指标一致性 | 由预测文件独立复算 P/R/F1/ROC-AUC/PR-AUC，与 `model_metrics.json` 最大偏差 1.1e-16 |
| 阈值/成本复现 | 独立复算最优阈值 0.144/成本 735、三策略、冻结测试集评估（成本 1,465），与档案一致 |
| Streamlit | 五页全部渲染正常（截图见 `docs/screenshots/`），AppTest 7 项通过 |

## 12. 项目局限性

1. **成本参数是实验假设**：阈值 0.144 依附"漏判 100/误拦 5"；真实业务须用实际损失数据重标定（平台已做成可配置）；
2. **评估样本小**：验证/测试各 55/52 笔欺诈，指标置信区间宽，模型间小差距不可下强结论；测试集成本 +99.3% 无法排除抽样噪声；
3. **数据仅 48 小时**：无法验证欺诈模式漂移，上线需持续监控（如周度 Recall/FP 率）；
4. **特征匿名**：V1-V28 不可业务归因；无卡号/商户字段，无法构造同卡行为、商户集中度等强特征；
5. **XGBoost 概率未校准**：`scale_pos_weight` 整体抬高概率，跨模型直接比较概率尺度需谨慎（校准是可选改进）；
6. **单一数据集**：结论不能外推到其他卡组织/地区；
7. **未做 SHAP / 概率校准 / 交叉验证多折**：列为可选改进，未在本项目中实现。

## 13. 文档与诚信约束

- 文档：`docs/architecture.md`（模块关系）、`docs/data_dictionary.md`（字段字典）、`docs/demo_guide.md`（5 分钟演示流程）、`docs/resume_project.md`（简历项目描述）、`docs/interview_qa.md`（15 道面试题）、阶段报告 `reports/stage*.md`
- 诚信约束：
  1. 全部数字来自真实数据与实跑代码，不虚构、不修饰；
  2. 原始数据只读（`data/raw/`，系统级只读属性）；
  3. 测试集不参与任何选择：模型/参数/阈值只用验证集，测试集一次性评估；
  4. 成本参数与策略目标值均显式标注为假设；
  5. 每个结论可由 `notebooks/` 与 `scripts/` 从头复现。

## 14. 看板页面一览

| 页面 | 截图 | 内容 |
|---|---|---|
| 数据概览 | `docs/screenshots/01_overview.png` | 原始/去重双口径、金额分布、数据基本信息 |
| 模型评估 | `docs/screenshots/02_models.png` | 三模型指标表、ROC/PR 实时曲线、0.5 阈值混淆矩阵 |
| 风险阈值分析 | `docs/screenshots/03_threshold.png` | 阈值滑块实时重算 P/R/F1 与混淆矩阵、阈值曲线 |
| 策略成本优化 | `docs/screenshots/04_strategy.png` | 成本参数可改、三策略重算、成本曲线、敏感性 |
| 风险分析报告 | `docs/screenshots/05_report.png` | 冻结策略、验证/测试差异、成本假设、局限性 |
