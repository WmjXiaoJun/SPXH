"""控制台编码：让中文输出在非 UTF-8 环境（如 en-US Windows 的 cp1252）下也能正常打印.

背景：Python 在输出被重定向时使用本地编码，en-US Windows 上是 cp1252，
`print("中文")` 会直接抛 UnicodeEncodeError 崩掉（CI 的 windows-latest 就是这么挂的，
见 docs/开源检查清单.md §9.8）。中文 Windows 是 cp936，恰好不出问题，所以本地不易发现。

`ensure_utf8()` 把 stdout / stderr 切到 UTF-8；已经设置 `SPXH_NO_UTF8=1` 时不做任何事。
"""
from __future__ import annotations

import os
import sys
from typing import Any


def ensure_utf8(force: bool = False) -> None:
    """把标准输出/错误切到 UTF-8，失败则静默放过（绝不因为编码问题让程序崩掉）."""
    if os.environ.get("SPXH_NO_UTF8"):
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        current = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if not force and current == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - 平台差异，尽力而为
            pass


__all__ = ["ensure_utf8"]
