"""阶段一 · 数据质量检查

对 data/raw/creditcard.csv 做只读体检，输出机器可读的 JSON 与人工可读的 Markdown。
本脚本不修改原始数据，不做任何"顺手清洗"，只负责如实报告问题。

用法：
    .venv/Scripts/python.exe scripts/check_data_quality.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "creditcard.csv"
RAW_ZIP = ROOT / "data" / "raw" / "creditcardfraud.zip"
OUT_MD = ROOT / "reports" / "data_quality_report.md"
OUT_JSON = ROOT / "reports" / "data_quality.json"

V_COLS = [f"V{i}" for i in range(1, 29)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> dict:
    if not RAW_CSV.exists():
        raise SystemExit(f"原始文件不存在：{RAW_CSV}（请先下载 Kaggle 数据集）")

    df = pd.read_csv(RAW_CSV)

    # ---------- 1. 文件层 ----------
    file_info = {
        "csv_path": str(RAW_CSV.relative_to(ROOT)).replace("\\", "/"),
        "csv_size_bytes": RAW_CSV.stat().st_size,
        "csv_sha256": sha256(RAW_CSV),
        "zip_size_bytes": RAW_ZIP.stat().st_size if RAW_ZIP.exists() else None,
        "zip_sha256": sha256(RAW_ZIP) if RAW_ZIP.exists() else None,
    }

    # ---------- 2. 结构层 ----------
    structure = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtype_counts": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "non_numeric_cols": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
    }

    # ---------- 3. 缺失 ----------
    na = df.isna().sum()
    missing = {
        "total_missing_cells": int(na.sum()),
        "cols_with_missing": {c: int(v) for c, v in na.items() if v > 0},
        "missing_ratio": float(na.sum() / df.size),
    }

    # ---------- 4. 重复 ----------
    dup_mask = df.duplicated(keep="first")
    dup = {
        "full_duplicate_rows": int(dup_mask.sum()),
        "full_duplicate_ratio": float(dup_mask.mean()),
        "duplicates_that_are_fraud": int(df.loc[dup_mask, "Class"].sum()),
        "fraud_rows_after_dedup": int(df.loc[~dup_mask, "Class"].sum()),
    }

    # 说明：Time 是每笔交易独有的秒级时间戳，完全相同的 Time 极可能是同一条记录的重复
    dup["duplicate_rows_share_time_with_original"] = int(
        df.loc[dup_mask, "Time"].isin(df.loc[~dup_mask, "Time"]).sum()
    )

    # ---------- 5. 标签 ----------
    vc = df["Class"].value_counts().sort_index()
    label = {
        "unique_values": sorted(int(v) for v in df["Class"].unique()),
        "count_0": int(vc.get(0, 0)),
        "count_1": int(vc.get(1, 0)),
        "ratio_0": float(vc.get(0, 0) / len(df)),
        "ratio_1": float(vc.get(1, 0) / len(df)),
        "imbalance_ratio": float(vc.get(0, 1) / max(vc.get(1, 1), 1)),
    }

    # ---------- 6. Time ----------
    t = df["Time"]
    ts = t.sort_values()
    diffs = ts.diff().dropna()
    time_info = {
        "min_seconds": float(t.min()),
        "max_seconds": float(t.max()),
        "span_hours": float((t.max() - t.min()) / 3600),
        "is_monotonic_increasing": bool(t.is_monotonic_increasing),
        "duplicated_timestamps": int(t.duplicated().sum()),
        "seconds_with_multiple_tx": int((df.groupby("Time").size() > 1).sum()),
        "max_tx_in_one_second": int(df.groupby("Time").size().max()),
        "zero_diff_count": int((diffs == 0).sum()),
    }

    # ---------- 7. Amount ----------
    a = df["Amount"]
    q1, q3 = a.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    amount = {
        "min": float(a.min()),
        "p25": float(q1),
        "median": float(a.median()),
        "mean": float(a.mean()),
        "p75": float(q3),
        "p99": float(a.quantile(0.99)),
        "p999": float(a.quantile(0.999)),
        "max": float(a.max()),
        "std": float(a.std()),
        "zero_amount_count": int((a == 0).sum()),
        "zero_amount_ratio": float((a == 0).mean()),
        "iqr_upper_fence": float(upper),
        "outliers_by_iqr": int((a > upper).sum()),
        "outlier_ratio_by_iqr": float((a > upper).mean()),
        "skewness": float(a.skew()),
        "kurtosis": float(a.kurtosis()),
    }

    # ---------- 8. V1-V28（PCA 特征）----------
    v_desc = df[V_COLS].describe().T
    v_info = {
        "n_features": len(V_COLS),
        "abs_mean_max": float(v_desc["mean"].abs().max()),
        "mean_range": [float(v_desc["mean"].min()), float(v_desc["mean"].max())],
        "std_range": [float(v_desc["std"].min()), float(v_desc["std"].max())],
        "abs_min_of_data": float(df[V_COLS].abs().min().min()),
        "abs_max_of_data": float(df[V_COLS].abs().max().max()),
        "n_features_with_zero_variance": int((v_desc["std"] == 0).sum()),
    }

    # ---------- 9. 结论式判定 ----------
    issues = []
    if missing["total_missing_cells"] == 0:
        issues.append("无缺失值，无需插补。")
    else:
        issues.append(f"存在 {missing['total_missing_cells']} 个缺失单元格，需处理。")
    if dup["full_duplicate_rows"] > 0:
        issues.append(
            f"存在 {dup['full_duplicate_rows']} 行完全重复记录（{dup['full_duplicate_ratio']:.4%}），"
            f"其中 {dup['duplicates_that_are_fraud']} 行的 Class=1。建模前需去重，"
            "否则同一条记录可能同时落入训练集与验证集，造成指标虚高。"
        )
    if amount["zero_amount_count"] > 0:
        issues.append(
            f"存在 {amount['zero_amount_count']} 笔金额为 0 的交易（{amount['zero_amount_ratio']:.3%}）。"
            "数据集未提供交易类型字段，无法证实这些 0 元交易的业务性质（可能是余额查询、预授权或冲正），"
            "因此不做业务归因，仅在特征工程中单独标记为 `is_zero_amount`，避免与真实小额交易混同。"
        )
    if time_info["span_hours"] < 72:
        issues.append(
            f"Time 仅覆盖 {time_info['span_hours']:.0f} 小时（约 2 天），"
            "无法做季节性/星期效应分析，且 Time 是相对秒数而非真实时钟，不能还原具体时刻。"
        )
    issues.append(
        "V1–V28 为 PCA 变换后的匿名特征，仅保留数值信息，不可赋予业务含义；"
        "其背后的原始字段（商户、卡种、地区等）未公开，因此无法做业务归因。"
    )
    issues.append(
        f"类别极不平衡：欺诈占比 {label['ratio_1']:.4%}（1:{label['imbalance_ratio']:.0f}）。"
        "后续建模禁止使用 Accuracy 作为主指标。"
    )

    result = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": file_info,
        "structure": structure,
        "missing": missing,
        "duplicates": dup,
        "label": label,
        "time": time_info,
        "amount": amount,
        "v_features": v_info,
        "issues": issues,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_markdown(result), encoding="utf-8")

    print_markdown(render_markdown(result))
    print(f"\n[写出] {OUT_MD.relative_to(ROOT)}")
    print(f"[写出] {OUT_JSON.relative_to(ROOT)}")
    return result


def render_markdown(r: dict) -> str:
    f, s, m, d, lb, t, am, v = (
        r["file"], r["structure"], r["missing"], r["duplicates"],
        r["label"], r["time"], r["amount"], r["v_features"],
    )
    lines = [
        "# 数据质量检查报告",
        "",
        f"> 生成时间：{r['generated_at']}　|　脚本：`scripts/check_data_quality.py`　|　数据：`{f['csv_path']}`",
        "",
        "## 1. 文件指纹",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| CSV 大小 | {f['csv_size_bytes']:,} bytes（{f['csv_size_bytes']/1024/1024:.2f} MB） |",
        f"| CSV SHA256 | `{f['csv_sha256']}` |",
        f"| ZIP 大小 | {f['zip_size_bytes']:,} bytes |" if f["zip_size_bytes"] else "| ZIP 大小 | - |",
        f"| ZIP SHA256 | `{f['zip_sha256']}` |" if f["zip_sha256"] else "| ZIP SHA256 | - |",
        "",
        "## 2. 数据结构",
        "",
        f"- 记录数：**{s['n_rows']:,}**　列数：**{s['n_cols']}**",
        f"- 字段清单：`{'`, `'.join(s['columns'])}`",
        f"- 数据类型分布：{s['dtype_counts']}",
        f"- 非数值列：{s['non_numeric_cols'] or '无'}",
        "",
        "## 3. 缺失值",
        "",
        f"- 缺失单元格总数：**{m['total_missing_cells']}**（占比 {m['missing_ratio']:.4%}）",
        f"- 含缺失的列：{m['cols_with_missing'] or '无'}",
        "",
        "## 4. 重复记录",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 完全重复行数 | {d['full_duplicate_rows']:,} |",
        f"| 重复占比 | {d['full_duplicate_ratio']:.4%} |",
        f"| 其中 Class=1（欺诈） | {d['duplicates_that_are_fraud']} |",
        f"| 去重后欺诈笔数 | {d['fraud_rows_after_dedup']} |",
        "",
        "## 5. 标签分布（Class）",
        "",
        "| Class | 含义 | 笔数 | 占比 |",
        "|---|---|---|---|",
        f"| 0 | 正常交易 | {lb['count_0']:,} | {lb['ratio_0']:.4%} |",
        f"| 1 | 欺诈交易 | {lb['count_1']:,} | {lb['ratio_1']:.4%} |",
        "",
        f"**不平衡比：正常 : 欺诈 ≈ {lb['imbalance_ratio']:.1f} : 1**",
        "",
        "## 6. Time 字段",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 最小值 | {t['min_seconds']:.0f} 秒 |",
        f"| 最大值 | {t['max_seconds']:.0f} 秒 |",
        f"| 覆盖时长 | {t['span_hours']:.2f} 小时 |",
        f"| 是否单调递增 | {t['is_monotonic_increasing']} |",
        f"| 重复时间戳数 | {t['duplicated_timestamps']:,} |",
        f"| 出现多笔交易的秒数 | {t['seconds_with_multiple_tx']:,} |",
        f"| 单秒最大交易笔数 | {t['max_tx_in_one_second']} |",
        "",
        "## 7. Amount 字段",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 最小 / 中位数 / 均值 / 最大 | {am['min']:.2f} / {am['median']:.2f} / {am['mean']:.2f} / {am['max']:.2f} |",
        f"| 标准差 | {am['std']:.2f} |",
        f"| P99 / P99.9 | {am['p99']:.2f} / {am['p999']:.2f} |",
        f"| 零金额笔数 | {am['zero_amount_count']:,}（{am['zero_amount_ratio']:.3%}） |",
        f"| IQR 上界（Q3+1.5IQR） | {am['iqr_upper_fence']:.2f} |",
        f"| IQR 判定离群笔数 | {am['outliers_by_iqr']:,}（{am['outlier_ratio_by_iqr']:.3%}） |",
        f"| 偏度 / 峰度 | {am['skewness']:.3f} / {am['kurtosis']:.3f} |",
        "",
        "## 8. V1–V28（PCA 匿名特征）",
        "",
        f"- 特征数：{v['n_features']}",
        f"- 均值范围：[{v['mean_range'][0]:.2e}, {v['mean_range'][1]:.2e}]（近似 0，符合 PCA 中心化）",
        f"- 标准差范围：[{v['std_range'][0]:.4f}, {v['std_range'][1]:.4f}]（多数近似 1，符合 PCA 白化）",
        f"- 取值绝对值范围：[{v['abs_min_of_data']:.4f}, {v['abs_max_of_data']:.2f}]",
        f"- 零方差特征数：{v['n_features_with_zero_variance']}",
        "",
        "## 9. 问题清单与处理决定",
        "",
    ]
    for i, s_ in enumerate(r["issues"], 1):
        lines.append(f"{i}. {s_}")
    lines.append("")
    return "\n".join(lines)


def print_markdown(md: str) -> None:
    print(md)


if __name__ == "__main__":
    main()
