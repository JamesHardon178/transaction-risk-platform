"""数据准备模块：加载原始数据、重复分析、去重、时间顺序切分。

设计约束
--------
1. 原始数据只读：本模块从不写 data/raw/ 下的任何文件。
2. 切分严格按 Time 升序，训练在前、验证居中、测试在后，
   杜绝"用未来预测过去"的时间泄漏。
3. 所有函数是纯函数（输入 DataFrame -> 输出 DataFrame/dict），
   便于 pytest 单测。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 项目根目录：src/data/preparation.py -> parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

TARGET_COL = "Class"
TIME_COL = "Time"


def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """加载原始 CSV（只读）。

    Time/Amount 读为 float，Class 读为 int，其余 V1-V28 为 float。
    """
    df = pd.read_csv(path)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def analyze_duplicates(df: pd.DataFrame) -> dict:
    """分析完全重复记录。

    返回：
        n_dup_rows      : 参与重复的行总数（keep=False 口径，含首行）
        n_dup_groups    : 重复组数
        size_distribution: 组大小 -> 组数的映射
        n_class_conflict: 特征列完全相同但 Class 不同的行组数。
                          注意：对"全列重复"做该检查没有意义（全列相同
                          必然同标签），因此冲突检查在全部行上进行——
                          它回答的是"同一笔交易被打了不同标签"的问题。
        n_to_drop       : drop_duplicates 将删除的行数
    """
    dup_mask = df.duplicated(keep=False)
    dups = df.loc[dup_mask]
    group_sizes = dups.groupby(list(df.columns)).size()

    # Class 冲突：对全部行按非 Class 列分组，Class 是否唯一
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    conflict = df.groupby(feature_cols)[TARGET_COL].nunique()

    return {
        "n_dup_rows": int(len(dups)),
        "n_dup_groups": int(len(group_sizes)),
        "size_distribution": {int(k): int(v) for k, v in group_sizes.value_counts().items()},
        "n_class_conflict": int((conflict > 1).sum()),
        "n_to_drop": int(df.duplicated().sum()),
    }


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """仅删除完全重复记录（保留首次出现），不做任何其他修改。"""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 Time 升序做时间顺序切分。

    前 train_ratio 为训练集，接下来 val_ratio 为验证集，
    其余为测试集。索引重置，原 df 不被修改。

    注意：切分边界用整数下标 int(n*ratio)，保证三集互不重叠且并集=全集。
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio 必须小于 1")

    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df_sorted)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))

    train = df_sorted.iloc[:i_train].reset_index(drop=True)
    val = df_sorted.iloc[i_train:i_val].reset_index(drop=True)
    test = df_sorted.iloc[i_val:].reset_index(drop=True)
    return train, val, test


def summarize_splits(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """输出三集的记录数、欺诈数、欺诈率与时间范围。"""
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rows.append(
            {
                "数据集": name,
                "记录数": len(part),
                "欺诈数": int(part[TARGET_COL].sum()),
                "正常数": int((part[TARGET_COL] == 0).sum()),
                "欺诈率": part[TARGET_COL].mean(),
                "Time范围": f"[{part[TIME_COL].min():.0f}, {part[TIME_COL].max():.0f}]",
            }
        )
    return pd.DataFrame(rows)
