"""M5 帧同步与载荷提取：Barker-13 前导 + 帧头解析 + CRC 校验."""
from spxh.core.framesync.extract import FrameExtraction, extract_frames
from spxh.core.framesync.preamble import (
    correlate_bits,
    correlate_symbols,
    preamble_pattern,
    preamble_symbols,
)

__all__ = [
    "FrameExtraction",
    "extract_frames",
    "preamble_pattern",
    "preamble_symbols",
    "correlate_symbols",
    "correlate_bits",
]
