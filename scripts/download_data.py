"""下载并校验 Kaggle Credit Card Fraud Detection 原始数据。

数据来源：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
（ULB 布鲁塞尔自由大学机器学习小组，2015；2013 年 9 月欧洲持卡人 48 小时交易）

用法：
    python scripts/download_data.py            # 自动尝试 Kaggle CLI
    python scripts/download_data.py --check    # 只校验已有文件

前置条件（二选一）：
    1. 安装 kaggle CLI 并配置 API Token：
         pip install kaggle
         把 https://www.kaggle.com/settings 下载的 kaggle.json 放到 ~/.kaggle/
    2. 手动从数据集页面下载 creditcardfraud.zip，解压后把 creditcard.csv
       放到 data/raw/creditcard.csv

校验口径（与项目文档一致）：
    284,807 行 × 31 列，Class=1（欺诈）共 492 笔
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "creditcard.csv"
DATASET = "mlg-ulb/creditcardfraud"
EXPECTED_ROWS = 284_807
EXPECTED_COLS = 31
EXPECTED_FRAUD = 492


def validate(path: Path) -> bool:
    """按行数/列数/欺诈数三项校验原始数据完整性。"""
    import pandas as pd

    if not path.exists():
        print(f"[FAIL] 文件不存在：{path}")
        return False

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[INFO] 文件：{path}（{size_mb:.1f} MB）")

    df = pd.read_csv(path)
    rows, cols = df.shape
    fraud = int(df["Class"].sum()) if "Class" in df.columns else -1

    print(f"[INFO] 形状：{rows} 行 × {cols} 列 | 欺诈：{fraud} 笔")

    ok = True
    if (rows, cols) != (EXPECTED_ROWS, EXPECTED_COLS):
        print(f"[FAIL] 形状应为 {EXPECTED_ROWS}×{EXPECTED_COLS}")
        ok = False
    if fraud != EXPECTED_FRAUD:
        print(f"[FAIL] 欺诈样本数应为 {EXPECTED_FRAUD}")
        ok = False
    if ok:
        print("[OK] 校验通过，可以运行数据流水线：")
        print("     python scripts/check_data_quality.py")
        print("     python scripts/build_database.py")
        print("     python scripts/run_sql.py")
    return ok


def try_kaggle_cli() -> bool:
    """尝试通过 Kaggle CLI 下载（需已配置 API Token）。"""
    if shutil.which("kaggle") is None:
        print("[INFO] 未找到 kaggle 命令，跳过自动下载")
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"]
    print(f"[INFO] 执行：{' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[WARN] Kaggle CLI 下载失败，通常是未配置 API Token：")
        print(result.stderr.strip()[:500])
        return False

    # CLI 可能把 zip 解压到子目录，做一次兜底查找
    for candidate in RAW_DIR.rglob("creditcard.csv"):
        if candidate != CSV_PATH:
            shutil.move(str(candidate), str(CSV_PATH))
        return True

    zip_candidates = list(RAW_DIR.glob("*.zip"))
    for zpath in zip_candidates:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(RAW_DIR)
        print(f"[INFO] 已解压：{zpath.name}")
        return CSV_PATH.exists()

    return False


def print_manual_steps() -> None:
    print(
        "\n手动下载步骤：\n"
        "  1. 打开 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "  2. 点击 Download（需登录 Kaggle 账号，免费）\n"
        "  3. 解压得到 creditcard.csv\n"
        f"  4. 放到 {CSV_PATH}\n"
        "  5. 重新运行：python scripts/download_data.py --check\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="下载并校验原始交易数据")
    parser.add_argument("--check", action="store_true", help="只校验现有文件，不下载")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        return 0 if validate(CSV_PATH) else 1

    if CSV_PATH.exists():
        print("[INFO] 已存在原始数据，直接校验")
        return 0 if validate(CSV_PATH) else 1

    if try_kaggle_cli() and CSV_PATH.exists():
        return 0 if validate(CSV_PATH) else 1

    print_manual_steps()
    return 1


if __name__ == "__main__":
    sys.exit(main())
