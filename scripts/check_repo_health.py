#!/usr/bin/env python
"""仓库自检脚本：不需要安装任何第三方依赖，用标准库跑完。

检查四类问题（全部是"仓库长得像个项目"的最小必要条件）：

1. **文档链接**：README.md 与 docs/*.md 中的本地图片/文件链接是否指向真实存在的路径
   （跳过 http(s)、mailto、以及纯 `#锚点`）。失效链接是公开仓库最常见的不专业信号。
2. **关键产物**：reports/ 下被 README、看板、测试依赖的档案文件与图表是否齐全。
   这些文件是"结论可追溯"的凭证，缺一个就意味着某段叙述失去依据。
3. **数据目录纪律**：data/raw 与 data/processed 下不允许有被 git 跟踪的文件
   （原始交易数据体积大且含准敏感字段，只允许 .gitkeep 占位）。
4. **冻结数值完整性**：打印 reports/model_metrics.json 与 threshold_analysis.json 的
   sha256 短指纹，便于跨提交人工比对"实验结论是否被无意改动"。

用法：
    python scripts/check_repo_health.py          # 有人读输出，发现问题退出码 1
    python scripts/check_repo_health.py --quiet   # 只输出结论行（供 CI / 定时任务使用）
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"

# Markdown 链接目标：只匹配 `](target)` 这一段。
# 不写成 `[text](target)` 是因为徽章形如 [![alt](img)](url)——方括号里再套方括号，
# 一次性匹配整个链接会漏掉外层目标；只认右括号前的目标串则两类写法都能覆盖。
LINK_PATTERN = re.compile(r"\]\(\s*<?([^)\s>]+)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")

# 被 README 正文、Streamlit 看板或测试引用的档案；缺任何一个都算结构性问题
REQUIRED_ARTIFACTS = [
    "reports/model_metrics.json",
    "reports/threshold_analysis.json",
    "reports/strategy_comparison.csv",
    "reports/data_quality.json",
    "reports/data_quality_report.md",
    "reports/stage1_report.md",
    "reports/stage2_model_report.md",
    "reports/stage3_strategy_report.md",
    "reports/sql_results.md",
    "docs/architecture.md",
    "docs/data_dictionary.md",
    "docs/demo_guide.md",
]

# README 第 15 节声明"共 21 张"（20 张静态 PNG + 1 张交互 HTML），与磁盘实际数量做一致性校验
EXPECTED_FIGURE_COUNT = 21

# 冻结的实验数值文件：内容随实验冻结，指纹变化需要人工确认
FROZEN_FILES = [
    "reports/model_metrics.json",
    "reports/threshold_analysis.json",
]

# data/ 下只允许这两类文件被跟踪：占位符与目录说明（真正的数据文件一律不入库）
DATA_ALLOWED = {".gitkeep", "README.md"}


def iter_markdown_files() -> list[Path]:
    files = [PROJECT_ROOT / "README.md"]
    docs = PROJECT_ROOT / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.glob("*.md")))
    return [f for f in files if f.is_file()]


def check_links() -> list[str]:
    """返回失效链接描述列表；空列表代表全部可达。"""
    problems: list[str] = []
    checked = 0
    for md_path in iter_markdown_files():
        text = md_path.read_text(encoding="utf-8", errors="replace")
        seen: set[str] = set()
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip("<>").split("#", 1)[0]
            if not target or target.startswith(SKIP_PREFIXES) or target in seen:
                continue
            seen.add(target)
            checked += 1
            resolved = (md_path.parent / target).resolve()
            if not resolved.exists():
                problems.append(f"{md_path.relative_to(PROJECT_ROOT)} -> {target}（不存在）")
    print(f"  文档链接      : 扫描 {checked} 条本地链接，失效 {len(problems)} 条")
    return problems


def check_artifacts() -> list[str]:
    missing = [rel for rel in REQUIRED_ARTIFACTS if not (PROJECT_ROOT / rel).exists()]
    problems = [f"缺少必需产物: {rel}" for rel in missing]

    figures_dir = REPORTS / "figures"
    figures = [p for p in figures_dir.iterdir() if p.is_file() and p.name != ".gitkeep"]
    if len(figures) != EXPECTED_FIGURE_COUNT:
        problems.append(
            f"图表数量不符: 期望 {EXPECTED_FIGURE_COUNT} 个（20 PNG + 1 交互 HTML），实际 {len(figures)} 个"
        )
    print(f"  关键产物      : 缺 {len(missing)} 项，图表 {len(figures)} 个（期望 {EXPECTED_FIGURE_COUNT}）")

    for rel in FROZEN_FILES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # 冻结档案被损坏比数值漂移更严重
            problems.append(f"冻结档案不是合法 JSON: {rel}（{exc.msg}）")
    return problems


def check_data_hygiene() -> list[str]:
    """用 git ls-files 判断跟踪状态：比遍历磁盘更准确（磁盘允许有本地数据）。"""
    try:
        out = subprocess.run(
            ["git", "ls-files", "data"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  数据目录纪律  : 跳过（git 不可用：{exc}）")
        return []

    offenders = [
        line for line in out.splitlines() if line.strip() and Path(line).name not in DATA_ALLOWED
    ]
    print(f"  数据目录纪律  : data/ 下被跟踪文件 {len(out.splitlines())} 个，违规 {len(offenders)} 个")
    return [f"data/ 下有不应入库的文件被跟踪: {rel}" for rel in offenders]


def report_fingerprints() -> None:
    for rel in FROZEN_FILES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"  冻结指纹      : {rel} sha256:{digest}")


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    if not quiet:
        print(f"仓库自检：{PROJECT_ROOT}")

    problems: list[str] = []
    problems += check_links()
    problems += check_artifacts()
    problems += check_data_hygiene()
    if not quiet:
        report_fingerprints()

    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("\n结论：全部通过（链接可达、产物齐全、数据目录干净、冻结档案可解析）。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
