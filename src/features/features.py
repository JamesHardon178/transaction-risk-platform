"""特征模块：明确建模用特征清单与 X/y 构造。

阶段二刻意保持特征集最小化：
- V1-V28（PCA 匿名特征）+ Amount + Time。
- Time 作为特征保留：它是"交易发生的相对秒数"，模型可以从中
  学到昼夜节律（阶段一已证实欺诈率与时段强相关）。
- 不引入任何阶段一结论之外的衍生特征，避免在本阶段引入未经
  验证的工程复杂度。
"""

from __future__ import annotations

import pandas as pd

TIME_COL = "Time"
AMOUNT_COL = "Amount"
TARGET_COL = "Class"

FEATURES: list[str] = [f"V{i}" for i in range(1, 29)] + [AMOUNT_COL, TIME_COL]


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """从 DataFrame 提取特征矩阵 X 与标签 y。

    只保留 FEATURES 列（显式白名单，防止把 Class 等目标列泄漏进特征）。
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"缺少特征列: {missing}")
    X = df[FEATURES].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y
