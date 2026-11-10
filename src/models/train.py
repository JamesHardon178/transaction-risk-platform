"""模型训练模块：逻辑回归 / 随机森林 / XGBoost。

不平衡处理策略（全部只在训练集内部生效）：
- LogisticRegression : class_weight="balanced"（损失函数中按类频率反比加权）
- RandomForest       : class_weight="balanced_subsample"（每棵树的 bootstrap 内加权）
- XGBoost            : scale_pos_weight = 负样本数 / 正样本数

不使用 SMOTE：48 小时数据的时间结构会被插值出的"合成交易"破坏，
且过采样对树模型增益有限、显著增加训练成本。本阶段用代价加权，
阶段三如需对比重采样，再在训练集内部实验。

超参选择只用验证集（PR-AUC 为主指标），测试集不参与任何选择。
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42

# 评估/调参主指标：PR-AUC（average_precision）。
# 在 0.17% 正样本率下，ROC-AUC 会因大量真负样本而显得虚高，
# PR-AUC 对少数类查准/查全更敏感。
PRIMARY_METRIC = "pr_auc"


def train_logreg(X_train: pd.DataFrame, y_train: pd.Series, C: float = 1.0) -> Pipeline:
    """逻辑回归基线。StandardScaler 只在训练集上 fit（Pipeline 保证）。

    max_iter=3000 保证在标准化后的 30 维数据上充分收敛；
    solver=lbfgs 适合中小规模稠密特征。
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_leaf: int = 5,
) -> RandomForestClassifier:
    """随机森林。

    参数以"控内存、控过拟合"为先：
    - max_depth=12 + min_samples_leaf=5 抑制对噪声样本的深拟合；
    - n_jobs=-1 并行训练。
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    n_estimators: int = 1000,
) -> tuple[XGBClassifier, dict]:
    """XGBoost，用验证集 early stopping 自动选轮数。

    - scale_pos_weight 处理 578:1 的类不平衡；
    - eval_metric=aucpr 与主指标一致；
    - early_stopping_rounds=50：验证集 PR-AUC 连续 50 轮不提升即停，
      best_iteration 即为"以验证集为依据"选出的轮数。
    - subsample/colsample_bytree < 1 引入随机性防过拟合。

    返回 (拟合好的模型, 训练元信息 dict)。
    """
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    meta = {
        "scale_pos_weight": spw,
        "best_iteration": int(model.best_iteration),
        "best_val_aucpr": float(model.best_score),
    }
    return model, meta


# ---------------------------------------------------------------- 调参框架

def select_by_validation(
    fit_fn,  # callable(**params) -> fitted model
    param_grid: list[dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    primary_metric_fn,  # callable(y_true, y_proba) -> float
) -> tuple[object, dict, pd.DataFrame]:
    """在小参数网格上训练，按验证集主指标选最优。

    只接触训练集与验证集；测试集永远不会传进本函数。
    返回 (最优模型, 最优参数, 全部试验记录 DataFrame)。
    """
    records = []
    best_model, best_params, best_score = None, None, -1.0
    for params in param_grid:
        model = fit_fn(X_train, y_train, **params)
        proba = model.predict_proba(X_val)[:, 1]
        score = float(primary_metric_fn(y_val, proba))
        records.append({**params, "val_score": score})
        if score > best_score:
            best_model, best_params, best_score = model, params, score
    return best_model, best_params, pd.DataFrame(records)
