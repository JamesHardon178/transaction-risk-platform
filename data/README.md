# data/ 目录说明

本目录**不包含数据文件**（原始数据 210 MB、处理产物 169 MB，超出仓库合理体积，
且需遵守 Kaggle 数据集的使用条款）。

## 获取原始数据

```bash
python scripts/download_data.py          # 自动尝试 Kaggle CLI
python scripts/download_data.py --check  # 只校验已有文件
```

脚本会校验：284,807 行 × 31 列、欺诈样本 492 笔——三项全对才继续。
手动下载地址：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 目录约定

| 路径 | 内容 | 是否入库 |
|---|---|---|
| `data/raw/creditcard.csv` | Kaggle 原始数据（只读） | 否 |
| `data/processed/transactions.db` | SQLite 库（由 `scripts/build_database.py` 生成） | 否 |
| `data/processed/transactions_clean.parquet` | 去重后数据 | 否 |
| `data/processed/predictions_{val,test}_{lr,rf,xgb}.csv` | 六份预测结果 | 否 |

## 从零复现的顺序

```bash
python scripts/download_data.py        # 1. 下载 + 校验
python scripts/check_data_quality.py   # 2. 数据质量报告
python scripts/build_database.py       # 3. 建 SQLite 库 + 去重
python scripts/run_sql.py              # 4. 9 条 SQL 分析 → reports/sql_results.md
# 5. Notebook 01 → 02（Jupyter 内核选 transaction-risk）
python scripts/build_stage3_analysis.py  # 6. 阈值/策略/敏感性 → reports/*.json
pytest -q                              # 7. 58 个用例（依赖前 4 步产物）
streamlit run app/streamlit_app.py     # 8. 五页看板
```

> 注意：`pytest` 中的多数用例会读取 `data/processed/` 下的产物，
> 未完成第 3–4 步时会出现文件缺失类报错，这是预期行为而非代码缺陷。
