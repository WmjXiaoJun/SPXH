"""前导相关：从解调后的符号/比特流里找到 Barker-13 的起点，并**解出相位模糊**.

为什么这一步能解决相位模糊
--------------------------
Barker-13 是已知序列，把它按当前调制映射成期望符号后与接收符号做归一化复数相关：
* 相关峰的位置给出帧起始（前导在帧首，峰值 13:1 远高于旁瓣）；
* 相关峰的**复数相位**就是整段符号相对期望星座的旋转角 —— 因为接收端载波环只会
  锁到某个对称点（BPSK 180 度、QPSK 90 度、8PSK 45 度），这个相位正好等于模糊角。
  把它旋回去，模糊就被**前导**解掉了，不再需要拿真值去搜旋转。

FSK 是恒模非相干检测，没有相位模糊，所以在**比特域**做相关（前导比特 ±1 直接匹配）。
"""
from __future__ import annotations

import numpy as np

from spxh.core.framing.barker import BARKER_LEN, barker_bits
from spxh.core.synth import mapping
from spxh.core.types import FRAME_MAGIC, MOD_BITS_PER_SYMBOL, MOD_CODES

__all__ = [
    "preamble_pattern",
    "preamble_symbols",
    "detection_bits",
    "detection_symbols",
    "correlate_symbols",
    "correlate_bits",
]


def preamble_pattern() -> np.ndarray:
    """Barker-13 的 ±1 形式，符号约定与星座映射一致：比特 0 -> +1，比特 1 -> -1.

    注意不能写成"1 -> +1"：M0 的 BPSK 星座把比特 0 映射到 +1、比特 1 映射到 -1
    （见 synth/mapping.py），这里必须同约定，否则比特域相关的峰值会整体反号
    （实测正相关峰变 -1、"相位模糊"也会判反）。
    """
    bits = barker_bits(BARKER_LEN)
    return np.where(bits == 1, -1.0, 1.0)


def preamble_symbols(modulation: str) -> tuple[np.ndarray, int]:
    """返回 (期望前导符号, 完整符号数).

    只取能被 bits_per_symbol 整除的部分：QPSK 下 13 位前导只有前 6 个完整符号
    （第 7 个符号里混了帧头第 0 位），因此相关长度是 floor(13/bps)。
    """
    mod = mapping.normalize_modulation(modulation)
    bps = int(MOD_BITS_PER_SYMBOL[mod])
    count = BARKER_LEN // bps
    if count < 2:
        raise ValueError("调制阶数过高，前导不足 2 个完整符号：{}".format(mod))
    bits = barker_bits(BARKER_LEN)[: count * bps]
    indices = mapping.bits_to_indices(bits, mod)
    return mapping.indices_to_points(indices, mod), count


def detection_bits(modulation: str) -> np.ndarray:
    """**更长的已知前缀**：Barker-13 + MAGIC(0xA5) + 调制码(4 bit) = 25 bit.

    帧头前 1.5 字节其实是已知的：byte0 恒为 MAGIC，byte1 的高 4 bit 是调制码
    （调制类型由 /api/analyze 或分类器给出）。把这 25 bit 一起当模板，
    检测积分长度差不多翻倍（QPSK：6 -> 12 个符号）—— 这是**不改线格式**就能拿到的
    低信噪比检测增益。真正的帧起点仍然由解析时的 13 位 Barker 逐位核对确认。
    """
    mod = mapping.normalize_modulation(modulation)
    barker = barker_bits(BARKER_LEN)
    magic = np.array([(FRAME_MAGIC >> (7 - i)) & 1 for i in range(8)], dtype=np.uint8)
    code = MOD_CODES[mod] & 0x0F
    mod_bits = np.array([(code >> (3 - i)) & 1 for i in range(4)], dtype=np.uint8)
    return np.concatenate([barker, magic, mod_bits]).astype(np.uint8)


def detection_symbols(modulation: str) -> tuple[np.ndarray, int]:
    """把检测比特模板映射成期望符号（只取完整符号）."""
    mod = mapping.normalize_modulation(modulation)
    bps = int(MOD_BITS_PER_SYMBOL[mod])
    bits = detection_bits(mod)
    count = bits.size // bps
    if count < 2:
        return preamble_symbols(mod)
    indices = mapping.bits_to_indices(bits[: count * bps], mod)
    return mapping.indices_to_points(indices, mod), count


def correlate_symbols(symbols: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """归一化复数滑动相关（返回复数序列，模为相关系数 0~1）.

    c[i] = sum_j conj(p_j) * y[i+j] / (||p|| * ||y[i:i+L]||)
    """
    y = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    p = np.asarray(pattern, dtype=np.complex128).reshape(-1)
    length = p.size
    if y.size < length:
        return np.zeros(0, dtype=np.complex128)
    windows = np.lib.stride_tricks.sliding_window_view(y, length)
    numerator = windows @ np.conj(p)
    norm_p = float(np.sqrt(np.sum(np.abs(p) ** 2)))
    norm_y = np.sqrt(np.sum(np.abs(windows) ** 2, axis=1))
    denominator = norm_p * np.maximum(norm_y, 1e-12)
    return numerator / denominator


def correlate_bits(bits: np.ndarray, pattern: np.ndarray | None = None) -> np.ndarray:
    """比特域归一化相关（FSK 用，也用于帧头前导的二次核对）.

    返回实数序列，取值 -1~1；峰值处应接近 +1。
    """
    b = np.asarray(bits, dtype=np.float64).reshape(-1)
    pm = preamble_pattern() if pattern is None else np.asarray(pattern, dtype=np.float64).reshape(-1)
    length = pm.size
    if b.size < length:
        return np.zeros(0, dtype=np.float64)
    signed = 1.0 - 2.0 * b
    windows = np.lib.stride_tricks.sliding_window_view(signed, length)
    denominator = float(np.sum(pm ** 2))
    return (windows @ pm) / max(denominator, 1e-12)
