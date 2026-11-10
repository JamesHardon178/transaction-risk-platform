"""特征模块与预测保存模块的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from src.data.predictions import OUTPUT_COLUMNS, save_predictions
from src.features.features import FEATURES, make_xy


def _toy_df():
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["Class"] = (rng.random(n) < 0.2).astype(int)
    return df


class TestMakeXY:
    def test_shape_and_columns(self):
        df = _toy_df()
        X, y = make_xy(df)
        assert list(X.columns) == FEATURES
        assert len(X) == len(y) == len(df)
        assert set(y.unique()) <= {0, 1}

    def test_no_target_leakage(self):
        df = _toy_df()
        X, _ = make_xy(df)
        assert "Class" not in X.columns

    def test_missing_column_raises(self):
        df = _toy_df().drop(columns=["Amount"])
        with pytest.raises(KeyError):
            make_xy(df)


class TestSavePredictions:
    def test_schema_and_values(self, tmp_path):
        df = _toy_df()
        proba = np.linspace(0, 1, len(df))
        path = save_predictions(df, proba, "XGBoost", "val", out_dir=tmp_path)
        out = pd.read_csv(path)
        assert list(out.columns) == OUTPUT_COLUMNS
        assert len(out) == len(df)
        assert out["Class"].tolist() == df["Class"].tolist()
        # label 阈值逻辑
        expected = (proba >= 0.5).astype(int)
        assert out["predicted_label"].tolist() == expected.tolist()
        assert path.name == "predictions_val_xgb.csv"

    def test_row_mismatch_raises(self, tmp_path):
        df = _toy_df()
        with pytest.raises(ValueError):
            save_predictions(df, np.zeros(len(df) - 1), "XGBoost", "val", out_dir=tmp_path)
