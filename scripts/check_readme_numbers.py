"""README 数字溯源自检：验证「实测数据」表里每行的数字都能在它引用的报告里找到.

用法: python scripts/check_readme_numbers.py
失败返回 1（可用于 CI）；只做"数字是否出现在对应报告里"这一件事，不做语义判断。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spxh.console import ensure_utf8  # noqa: E402
ensure_utf8()
README = ROOT / "README.md"

_NUM = re.compile(r"\d+(?:\.\d+)?(?:e-?\d+)?")


def main() -> int:
    text = README.read_text(encoding="utf-8")
    start = text.find("## 实测数据")
    end = text.find("## 项目结构")
    if start < 0 or end < 0:
        print("README 里找不到「实测数据」或「项目结构」章节")
        return 1
    rows = [line for line in text[start:end].splitlines()
            if line.startswith("|") and not re.match(r"^\|\s*-+", line) and "里程碑 | 指标" not in line]
    checked = hit = 0
    problems: list[str] = []
    for row in rows:
        cells = [cell.strip() for cell in row.split("|") if cell.strip()]
        if len(cells) < 4:
            continue
        measure, link = cells[2], cells[3]
        match = re.search(r"\(([^)]+)\)", link)
        if not match:
            continue
        target = ROOT / match.group(1)
        if not target.exists():
            problems.append("链接不存在: " + match.group(1))
            continue
        report = target.read_text(encoding="utf-8")
        numbers = [n for n in _NUM.findall(measure) if len(n) > 1]
        if not numbers:
            continue
        checked += 1
        if any(n in report for n in numbers):
            hit += 1
        else:
            problems.append("{}：数字 {} 在 {} 里找不到".format(cells[0], numbers[:4], match.group(1)))
    print("=" * 92)
    print("README 数字溯源：抽查 {} 行，命中 {} 行".format(checked, hit))
    print("=" * 92)
    for item in problems:
        print("  [FAIL] " + item)
    ok = bool(checked) and not problems
    print("")
    print("  [{}] 「实测数据」表里的数字都能在它引用的报告里找到（实测 {}/{}）".format(
        "PASS" if ok else "FAIL", hit, checked))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
