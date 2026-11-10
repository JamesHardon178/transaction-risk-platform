"""预测结果持久化模块。

输出字段：Time / Amount / Class / predicted_probability / predicted_label。
验证集与测试集分别保存；文件名带模型缩写。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_ABBR = {
    "Logistic Regression": "lr",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}

OUTPUT_COLUMNS = ["Time", "Amount", "Class", "predicted_probability", "predicted_label"]


def save_predictions(
    df_source: pd.DataFrame,
    y_proba: pd.Series | np.ndarray,
    model_name: str,
    split_name: str,
    threshold: float = 0.5,
    out_dir: Path | str = PROCESSED_DIR,
) -> Path:
    """保存某个模型在某切分上的预测结果，返回写入路径。

    df_source 必须含 Time/Amount/Class 列，且行序与 y_proba 一一对应。
    """
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proba = np.asarray(y_proba, dtype=float)
    if len(proba) != len(df_source):
        raise ValueError("y_proba 与 df_source 行数不一致")

    out = df_source[["Time", "Amount", "Class"]].copy()
    out["predicted_probability"] = proba
    out["predicted_label"] = (proba >= threshold).astype(int)
    out = out[OUTPUT_COLUMNS]

    abbr = MODEL_ABBR.get(model_name, model_name.lower())
    path = out_dir / f"predictions_{split_name}_{abbr}.csv"
    out.to_csv(path, index=False)
    return path


def load_predictions(
    split_name: str,
    model_name: str,
    in_dir: Path | str = PROCESSED_DIR,
) -> pd.DataFrame:
    """读取阶段二落盘的预测文件（阶段三阈值分析的唯一输入）。

    参数
    ----
    split_name : "val" 或 "test"（训练集预测未落盘，也不允许用于阈值选择）
    model_name : 模型名或缩写（"Random Forest" / "rf" 均可）

    校验：文件存在、字段齐全、Class 只含 0/1、概率落在 [0,1]。
    """
    import numpy as np

    if split_name not in ("val", "test"):
        raise ValueError(f'split_name 只能是 "val" 或 "test"，收到 {split_name!r}')

    abbr = MODEL_ABBR.get(model_name, model_name)
    path = Path(in_dir) / f"predictions_{split_name}_{abbr}.csv"
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在: {path}（请先执行阶段二 Notebook）")

    df = pd.read_csv(path)
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"预测文件缺少字段 {missing}: {path}")

    y = df["Class"].to_numpy()
    p = df["predicted_probability"].to_numpy(dtype=float)
    if len(df) == 0:
        raise ValueError(f"预测文件为空: {path}")
    if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError(f"预测概率越界或含非法值: {path}")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError(f"Class 含非 0/1 取值: {np.unique(y)}")
    return df

