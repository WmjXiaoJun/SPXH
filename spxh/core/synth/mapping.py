"""调制映射：比特 <-> 星座/音调索引 <-> 星座点.

约定
----
* 所有星座均归一化到**单位平均符号能量**（mean |s|^2 = 1），因此 SNR 定义在各调制间可比。
* PSK/QAM 的星座表按比特组（MSB-first）顺序排列，排列本身已是 Gray 编码。
* FSK 无星座点，索引即音调序号；4FSK 的两个比特到音调序号使用 Gray 映射。
"""
from __future__ import annotations

import numpy as np

from spxh.core.types import MOD_BITS_PER_SYMBOL, is_fsk, normalize_modulation

__all__ = [
    "bin_to_gray",
    "gray_to_bin",
    "constellation",
    "num_points",
    "bits_per_symbol",
    "bits_to_indices",
    "indices_to_bits",
    "indices_to_points",
    "points_to_indices",
    "min_distance",
]


def bin_to_gray(value):
    arr = np.asarray(value, dtype=np.int64)
    return arr ^ (arr >> 1)


def gray_to_bin(value):
    arr = np.asarray(value, dtype=np.int64).copy()
    result = arr.copy()
    shift = 1
    while np.any((arr >> shift) > 0):
        result = result ^ (arr >> shift)
        shift += 1
    return result


def _bpsk() -> np.ndarray:
    return np.asarray([1.0 + 0j, -1.0 + 0j], dtype=np.complex128)


def _qpsk() -> np.ndarray:
    return (np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2.0))


def _psk8() -> np.ndarray:
    # 相位序号 p 上的比特标签取 bin_to_gray(p)，等价于「标签 i 落在相位 gray_to_bin(i)」。
    # 这样相邻相位（含 315 度 -> 0 度的环绕）恰好相差 1 比特。
    idx = np.arange(8)
    return np.exp(2j * np.pi * gray_to_bin(idx) / 8.0).astype(np.complex128)


def _qam16() -> np.ndarray:
    points = np.zeros(16, dtype=np.complex128)
    for i in range(16):
        b0 = (i >> 3) & 1
        b1 = (i >> 2) & 1
        b2 = (i >> 1) & 1
        b3 = i & 1
        level_i = 2 * int(bin_to_gray((b0 << 1) | b1)) - 3
        level_q = 2 * int(bin_to_gray((b2 << 1) | b3)) - 3
        points[i] = level_i + 1j * level_q
    return points / np.sqrt(10.0)


_CONSTELLATIONS: dict[str, np.ndarray] = {
    "bpsk": _bpsk(),
    "qpsk": _qpsk(),
    "8psk": _psk8(),
    "16qam": _qam16(),
}

_FSK_POINTS = {"2fsk": 2, "4fsk": 4}
_FSK_TONE_GRAY = {"2fsk": False, "4fsk": True}


def bits_per_symbol(modulation: str) -> int:
    return MOD_BITS_PER_SYMBOL[normalize_modulation(modulation)]


def num_points(modulation: str) -> int:
    mod = normalize_modulation(modulation)
    if mod in _FSK_POINTS:
        return _FSK_POINTS[mod]
    return int(_CONSTELLATIONS[mod].size)


def constellation(modulation: str) -> np.ndarray:
    """返回星座点（按比特组 MSB-first 顺序）；FSK 无星座点，抛 ValueError."""
    mod = normalize_modulation(modulation)
    if mod not in _CONSTELLATIONS:
        raise ValueError("modulation " + mod + " has no constellation (FSK uses tones)")
    return _CONSTELLATIONS[mod].copy()


def min_distance(modulation: str) -> float:
    points = constellation(modulation)
    diff = np.abs(points[:, None] - points[None, :])
    diff[diff == 0] = np.inf
    return float(np.min(diff))


def _bit_matrix(bits, bps: int) -> np.ndarray:
    arr = np.asarray(bits).reshape(-1)
    if arr.size % bps != 0:
        raise ValueError("bit count " + str(arr.size) + " is not a multiple of " + str(bps))
    if arr.size and np.any((arr != 0) & (arr != 1)):
        raise ValueError("bits must be 0/1")
    return arr.astype(np.int64).reshape(-1, bps)


def bits_to_indices(bits, modulation: str) -> np.ndarray:
    """比特流 -> 星座/音调索引."""
    mod = normalize_modulation(modulation)
    bps = MOD_BITS_PER_SYMBOL[mod]
    mat = _bit_matrix(bits, bps)
    weights = (1 << np.arange(bps - 1, -1, -1)).astype(np.int64)
    raw = mat @ weights
    if _FSK_TONE_GRAY.get(mod, False):
        return (raw ^ (raw >> 1)).astype(np.int64)
    return raw.astype(np.int64)


def indices_to_bits(indices, modulation: str) -> np.ndarray:
    """星座/音调索引 -> 比特流（bits_to_indices 的逆）."""
    mod = normalize_modulation(modulation)
    bps = MOD_BITS_PER_SYMBOL[mod]
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    values = gray_to_bin(idx) if _FSK_TONE_GRAY.get(mod, False) else idx
    if np.any(values < 0) or np.any(values >= (1 << bps)):
        raise ValueError("index out of range for " + mod)
    shifts = np.arange(bps - 1, -1, -1)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8).reshape(-1)


def indices_to_points(indices, modulation: str) -> np.ndarray:
    """星座索引 -> 复数星座点；FSK 抛 ValueError."""
    mod = normalize_modulation(modulation)
    if mod not in _CONSTELLATIONS:
        raise ValueError("modulation " + mod + " has no constellation points (FSK uses tones)")
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    table = _CONSTELLATIONS[mod]
    if np.any(idx < 0) or np.any(idx >= table.size):
        raise ValueError("index out of range for " + mod)
    return table[idx].copy()


def points_to_indices(points, modulation: str) -> np.ndarray:
    """复数星座点 -> 最近星座索引（最大似然硬判决）."""
    mod = normalize_modulation(modulation)
    table = constellation(mod)
    rx = np.asarray(points, dtype=np.complex128).reshape(-1)
    dist = np.abs(rx[:, None] - table[None, :]) ** 2
    return np.argmin(dist, axis=1).astype(np.int64)
