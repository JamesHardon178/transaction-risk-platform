"""数据准备模块的单元测试（纯合成数据，不依赖原始 CSV）。"""

import numpy as np
import pandas as pd
import pytest

from src.data.preparation import (
    analyze_duplicates,
    deduplicate,
    split_by_time,
    summarize_splits,
)


def _toy_df():
    return pd.DataFrame(
        {
            "Time": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "V1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "Class": [0, 1, 0, 0, 1, 0, 0],
        }
    )


class TestDuplicates:
    def test_deduplicate_keeps_first_occurrence(self):
        df = pd.concat([_toy_df(), _toy_df().iloc[[2, 5]]], ignore_index=True)
        assert len(df) == 9
        out = deduplicate(df)
        assert len(out) == 7
        # 保留的是首次出现
        assert out["Time"].tolist() == _toy_df()["Time"].tolist()

    def test_deduplicate_is_not_inplace(self):
        df = _toy_df()
        before = df.copy()
        deduplicate(df)
        pd.testing.assert_frame_equal(df, before)

    def test_analyze_duplicates_counts(self):
        df = pd.concat([_toy_df(), _toy_df().iloc[[2, 2, 5]]], ignore_index=True)
        info = analyze_duplicates(df)
        # iloc[[2,2,5]] 复制了 2 号行两次、5 号行一次
        assert info["n_to_drop"] == 3
        assert info["n_dup_groups"] == 2  # 2 号一组(3 行)、5 号一组(2 行)
        assert info["n_class_conflict"] == 0

    def test_class_conflict_detected(self):
        # 同一特征组合对应两个不同 Class -> 冲突
        df = _toy_df()
        conflict_row = df.iloc[[2]].copy()
        conflict_row["Class"] = 1
        df = pd.concat([df, conflict_row], ignore_index=True)
        info = analyze_duplicates(df)
        assert info["n_class_conflict"] == 1


class TestSplitByTime:
    def test_split_ordered_disjoint_complete(self):
        df = _toy_df()
        train, val, test = split_by_time(df, train_ratio=0.7, val_ratio=0.15)
        assert len(train) == int(7 * 0.7)
        assert len(train) + len(val) + len(test) == len(df)
        # 时间不重叠且递增
        assert train["Time"].max() <= val["Time"].min()
        assert val["Time"].max() <= test["Time"].min()
        # 每一集内部 Time 升序
        for part in (train, val, test):
            assert part["Time"].is_monotonic_increasing

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            split_by_time(_toy_df(), train_ratio=0.9, val_ratio=0.2)

    def test_summarize_splits_fields(self):
        train, val, test = split_by_time(_toy_df())
        s = summarize_splits(train, val, test)
        assert list(s["数据集"]) == ["train", "val", "test"]
        assert s["记录数"].sum() == 7
        assert s["欺诈数"].sum() == 2
