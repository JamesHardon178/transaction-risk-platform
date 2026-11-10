"""阶段一 · 执行 SQL 分析文件并导出真实结果

读取 sql/*.sql，按 `-- >>> 标题` 分块，逐条在 transactions.db 上执行，
把每条 SQL 原文与真实执行结果写成 reports/sql_results.md。

约定：
- 每个 SQL 块以 `-- >>> 标题` 开头，语句以分号结束，一个块只放一条语句
- 不打印/不写入手工构造的任何数字，全部来自 sqlite3 实际执行

用法：
    .venv/Scripts/python.exe scripts/run_sql.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "transactions.db"
SQL_DIR = ROOT / "sql"
OUT_MD = ROOT / "reports" / "sql_results.md"


def parse_sql_file(path: Path) -> list[tuple[str, str]]:
    """按 `-- >>> 标题` 把文件切成 [(title, sql), ...]"""
    text = path.read_text(encoding="utf-8")
    blocks = text.split("-- >>> ")[1:]
    out = []
    for b in blocks:
        lines = b.split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:])
        body = body[: body.rindex(";")].strip()
        out.append((title, body + ";"))
    return out


def to_markdown_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    """不依赖 tabulate 的轻量 Markdown 表格渲染"""
    if df.empty:
        return "_（无结果行）_"
    show = df.head(max_rows)
    cols = [str(c) for c in show.columns]

    def fmt(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        if isinstance(v, float):
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.2f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_共 {len(df):,} 行，此处仅显示前 {max_rows} 行_")
    else:
        lines.append(f"\n_共 {len(df):,} 行_")
    return "\n".join(lines)


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"数据库不存在：{DB_PATH}（请先运行 scripts/build_database.py）")

    con = sqlite3.connect(DB_PATH)
    out = [
        "# SQL 分析结果（真实执行输出）",
        "",
        f"> 生成脚本：`scripts/run_sql.py`　|　数据库：`{DB_PATH.relative_to(ROOT).as_posix()}`　"
        f"|　SQLite：{sqlite3.sqlite_version}",
        ">",
        "> 本文件所有表格均由 SQLite 实际执行后导出，未做任何人工修改。",
        "",
    ]
    total = 0
    try:
        for sql_file in sorted(SQL_DIR.glob("*.sql")):
            out.append(f"## {sql_file.name}")
            out.append("")
            print(f"\n===== {sql_file.name} =====")
            for title, sql in parse_sql_file(sql_file):
                df = pd.read_sql_query(sql, con)
                total += 1
                print(f"  [{total:02d}] {title}  →  {len(df)} 行")
                out.append(f"### {title}")
                out.append("")
                out.append("```sql")
                out.append(sql.strip())
                out.append("```")
                out.append("")
                out.append(to_markdown_table(df))
                out.append("")
    finally:
        con.close()

    OUT_MD.write_text("\n".join(out), encoding="utf-8")
    print(f"\n共执行 {total} 条查询，结果写入 {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
