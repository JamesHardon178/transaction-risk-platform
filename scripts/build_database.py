"""阶段一 · 构建 SQLite 分析库

从只读的 data/raw/creditcard.csv 导入 data/processed/transactions.db，
建立显式 schema 与索引，便于后续 SQL 分析（CTE / 窗口函数）。

设计说明
--------
1. 库中只存**原始字段**（time_sec / v1..v28 / amount / class），
   不做任何派生列。小时、金额区间等一律在 SQL 里现算，
   保证"数据入库一次、口径在查询处可见"。
2. 保留 raw_row_id（原始 CSV 的行号，0 起），便于把 SQL 结果回溯到 CSV 具体行。
3. class 字段含义：0 = 正常交易，1 = 欺诈交易（依据 Kaggle 数据集说明与数据分布核实）。
4. 每次重建都会覆盖 data/processed/transactions.db，但绝不触碰 data/raw/。

用法：
    .venv/Scripts/python.exe scripts/build_database.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "creditcard.csv"
DB_PATH = ROOT / "data" / "processed" / "transactions.db"

V_COLS = [f"V{i}" for i in range(1, 29)]

DDL = """
DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    raw_row_id INTEGER PRIMARY KEY,   -- 原始 CSV 行号（0 起，不含表头）
    time_sec   REAL    NOT NULL,      -- Time：距首笔交易的秒数
    {v_cols},
    amount     REAL    NOT NULL,      -- Amount：交易金额
    class      INTEGER NOT NULL CHECK (class IN (0, 1))  -- 0=正常, 1=欺诈
);
""".format(v_cols=",\n    ".join(f"{c.lower():<10s} REAL" for c in V_COLS))

INDEXES = [
    "CREATE INDEX idx_tx_class  ON transactions(class);",
    "CREATE INDEX idx_tx_time   ON transactions(time_sec);",
    "CREATE INDEX idx_tx_amount ON transactions(amount);",
]


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"原始文件不存在：{RAW_CSV}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    print(f"[读取] {RAW_CSV.relative_to(ROOT)}")
    df = pd.read_csv(RAW_CSV)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"time": "time_sec"})
    df.insert(0, "raw_row_id", range(len(df)))

    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(DDL)
        for stmt in INDEXES:
            con.execute(stmt)
        df.to_sql("transactions", con, if_exists="append", index=False, chunksize=50_000)
        con.commit()

        print("\n[校验]")
        n = con.execute("SELECT COUNT(*) FROM transactions;").fetchone()[0]
        print(f"  行数            : {n:,}（CSV 行数 {len(df):,}）")
        n0, n1 = con.execute(
            "SELECT SUM(class=0), SUM(class=1) FROM transactions;"
        ).fetchone()
        print(f"  class=0 / =1    : {n0:,} / {n1:,}")
        print(f"  欺诈占比        : {n1 / n:.4%}")
        s = con.execute("SELECT ROUND(SUM(amount), 2) FROM transactions;").fetchone()[0]
        print(f"  金额合计        : {s:,.2f}")
        tmin, tmax = con.execute("SELECT MIN(time_sec), MAX(time_sec) FROM transactions;").fetchone()
        print(f"  time_sec 范围   : {tmin:.0f} ~ {tmax:.0f}")
        nn = con.execute(
            "SELECT COUNT(*) FROM transactions WHERE amount IS NULL OR time_sec IS NULL;"
        ).fetchone()[0]
        print(f"  关键字段空值    : {nn}")
        print("\n[表结构]")
        for row in con.execute("PRAGMA table_info(transactions);").fetchall():
            print(f"  {row[1]:<10s} {row[2]}")
        idx = con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='transactions';"
        ).fetchall()
        print(f"  索引: {[r[0] for r in idx]}")
    finally:
        con.close()

    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n[写出] {DB_PATH.relative_to(ROOT)}（{size_mb:.1f} MB）")


if __name__ == "__main__":
    main()
