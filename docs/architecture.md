# 架构说明：模块关系与数据流

> 本文说明项目各模块的职责、依赖方向与数据流转，帮助快速定位"某个数字是谁算出来的"。

## 1. 分层总览

项目分为五层，依赖方向**自上而下单向**（上层调用下层，下层不感知上层）：

```
┌─────────────────────────────────────────────────────────────┐
│  应用层  app/streamlit_app.py + app/page_*.py + app/common.py │
│          （只读产物 + 实时重算，不训练模型）                     │
├─────────────────────────────────────────────────────────────┤
│  编排层  notebooks/01、02 + scripts/build_stage3_analysis.py  │
│          （把下层模块串成端到端流水线，产出落盘档案）            │
├─────────────────────────────────────────────────────────────┤
│  规则层  src/rules/{thresholds, cost, strategies}.py          │
│  评估层  src/evaluation/{metrics, plots}.py                   │
│  可视化  src/viz/strategy_plots.py                             │
├─────────────────────────────────────────────────────────────┤
│  建模层  src/models/train.py（LR/RF/XGB + 验证集网格选参）      │
│  特征层  src/features/features.py（特征白名单，防目标泄漏）     │
├─────────────────────────────────────────────────────────────┤
│  数据层  src/data/{preparation, predictions}.py               │
│          scripts/{check_data_quality, build_database,          │
│                   run_sql}.py + sql/*.sql                      │
├─────────────────────────────────────────────────────────────┤
│  存储层  data/raw（只读）→ data/processed（parquet/SQLite/CSV） │
│          reports/*.json / *.csv / figures/（机器可读档案）      │
└─────────────────────────────────────────────────────────────┘
```

## 2. 数据流（从原始 CSV 到看板）

```
creditcard.csv (只读)
   │
   ├─[check_data_quality.py]──► data_quality_report.md / .json
   │
   ├─[build_database.py]──► transactions.db ──[run_sql.py + sql/*.sql]──► sql_results.md
   │
   └─[01_data_exploration.ipynb]──► transactions_clean.parquet（去重 + is_duplicate 等标记）
              │
              └─[02_model_training.ipynb]
                   ├─ preparation.py：dedup → sort by Time → 70/15/15 时间切分
                   ├─ features.py：make_xy（白名单 V1-28 + Amount + Time）
                   ├─ train.py：三模型 + 验证集网格选参（PR-AUC）
                   ├─ metrics.py / plots.py：指标 + fig09~15
                   └─ predictions.py：predictions_{val,test}_{lr,rf,xgb}.csv
                          │
                          └─[build_stage3_analysis.py]
                               ├─ thresholds.py：scan_thresholds（0.001 步长 1001 点）
                               ├─ cost.py：optimal_row / sensitivity_scan / sensitivity_grid
                               ├─ strategies.py：build_strategies → freeze_strategy
                               │                  → evaluate_frozen（测试集仅一次）
                               ├─ strategy_plots.py：fig16~21
                               └──► threshold_analysis.json + strategy_comparison.csv
                                          │
                                          └─[streamlit_app.py]──► 五页看板
```

## 3. 模块职责明细

### 数据层

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `src/data/preparation.py` | 去重、按 Time 排序切分 70/15/15、切分摘要 | Class 冲突检查按**非 Class 列**分组（否则全列重复永远检不出冲突）；纯函数 |
| `src/data/predictions.py` | 预测落盘/读取，schema 校验 | 固定 5 列：Time, Amount, Class, predicted_probability, predicted_label；提供 `load_predictions()` 供阶段三复用 |
| `scripts/check_data_quality.py` | 6 项质量体检 | 只读 `data/raw/`，不改写 |
| `scripts/build_database.py` | CSV → SQLite，显式 schema + 3 索引 | 表 `transactions(raw_row_id, time_sec, v1..v28, amount, class)` |
| `scripts/run_sql.py` | 执行 `sql/*.sql`，结果写 `sql_results.md` | 与 Notebook 同源数据，双口径互核 |

### 建模与评估层

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `src/features/features.py` | `FEATURES` 白名单 + `make_xy()` | 白名单显式排除 `Class`，防目标泄漏；不引入未验证的衍生特征 |
| `src/models/train.py` | 三模型训练函数 + 网格选参 | `RANDOM_STATE=42` 保证确定性；LR 的 StandardScaler 在 Pipeline 内只 fit 训练集；不平衡用代价加权，不用 SMOTE；主指标 `pr_auc` |
| `src/evaluation/metrics.py` | `evaluate_at_threshold` 等全套指标 | PR-AUC 用 `average_precision_score`（非梯形法），口径与 plots.py 一致 |
| `src/evaluation/plots.py` | ROC/PR/混淆矩阵/对比图/阈值曲线（fig09~15） | ROC 图例用 `roc_auc_score`、PR 图例用 `average_precision_score`，与 metrics 口径一致 |

### 规则层（阶段三核心）

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `src/rules/thresholds.py` | `scan_thresholds`：向量化阈值扫描 | O(T log n)（排序 + `np.searchsorted`）；判定规则 `proba >= threshold` 与阶段二完全一致；参数校验前置 |
| `src/rules/cost.py` | `total_cost` / `with_cost` / `optimal_row` / `sensitivity_scan` / `sensitivity_grid` | 成本 = FN×漏判成本 + FP×误拦成本；并列最优取较高阈值（少打扰）；成本参数必须非负有限 |
| `src/rules/strategies.py` | 三策略选择、`freeze_strategy`、`evaluate_frozen`、`val_test_delta` | **策略规则先于数据写成常量**（Recall≥0.90 / 成本最小 / Precision≥0.95）；测试集只允许评估一次 |
| `src/viz/strategy_plots.py` | fig16~21（阈值曲线/成本曲线/策略对比/敏感性线图与热力图/验证-测试差值） | 图例数字与 JSON 档案同源 |

### 应用层

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `app/streamlit_app.py` | 入口：侧边栏五页导航 + `?page=` 深链接 | 不含任何业务计算 |
| `app/common.py` | 统一数据加载（`@st.cache_data`） | 只读产物文件；`cost_scan()` 薄封装复用规则层 |
| `app/page_overview.py` | 数据概览：双口径计数、金额分布 | 读 parquet + model_metrics.json |
| `app/page_models.py` | 模型评估：指标表、ROC/PR 实时曲线 | 读 model_metrics.json + 预测 CSV |
| `app/page_threshold.py` | 阈值滑块实时重算 P/R/F1 与混淆矩阵 | 实时调用规则层扫描 |
| `app/page_strategy.py` | 成本参数可改、三策略重算、敏感性 | 与 page_threshold 共用扫描缓存 |
| `app/page_report.py` | 冻结策略、验证/测试差异、成本假设声明 | 读 threshold_analysis.json |

## 4. 关键设计决策

1. **档案即接口**：阶段间通过落盘文件解耦（predictions CSV、model_metrics.json、threshold_analysis.json），任何阶段都可以独立重跑、独立检查；
2. **口径单一**：判定规则 `proba >= threshold`、PR-AUC 算法、成本公式在 src 内只有一处定义，看板/脚本/报告全部复用；
3. **防泄漏三道闸**：特征白名单（features.py）→ 去重在切分前（preparation.py）→ 测试集只在冻结后评估一次（strategies.py）；
4. **假设显式化**：成本参数、策略目标值都是代码里的常量并写入 JSON 档案（`cost_assumptions`、`policy_targets`），报告与看板同步声明；
5. **确定性**：`RANDOM_STATE=42` + 固定网格 + 顺序切分，两本 Notebook 重跑指标与归档逐位一致（已实测）。

## 5. 测试映射

| 测试文件 | 覆盖 | 用例数 |
|---|---|---|
| `tests/test_preparation.py` | 去重/切分/Class 冲突检查 | 7 |
| `tests/test_features_predictions.py` | 特征白名单、预测 schema | 5 |
| `tests/test_metrics.py` | 指标计算（含 PR-AUC 口径） | 5 |
| `tests/test_plots.py` | 图表生成（Agg 落盘） | 4 |
| `tests/test_rules.py` | 阈值扫描/成本/策略/冻结/敏感性单调性 | 30 |
| `tests/test_app.py` | AppTest 真实渲染五页 + 交互 | 7 |
| **合计** | | **58** |

（用例数为 `pytest --collect-only` 实测值，2026-11-08。）
