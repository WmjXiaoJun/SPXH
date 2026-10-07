"""根升余弦（RRC）成型滤波与匹配滤波.

taps 归一到单位能量（sum h^2 = 1），因此成型后平均符号能量等于星座平均能量；
配合 matched_filter 在 k*sps + (len(taps)-1) 处采样即为最佳采样点。
"""
from __future__ import annotations

import numpy as np

__all__ = ["rrc_taps", "upsample", "pulse_shape", "matched_filter", "filter_delay", "symbol_sample_indices"]


def _rrc_response(t: np.ndarray, beta: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if beta <= 0.0:
        return np.sinc(t)
    out = np.zeros_like(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(np.pi * t * (1.0 - beta)) + 4.0 * beta * t * np.cos(np.pi * t * (1.0 + beta))
        den = np.pi * t * (1.0 - (4.0 * beta * t) ** 2)
        safe = np.abs(den) > 1e-12
        out = np.where(safe, num / np.where(safe, den, 1.0), 0.0)
    out = np.where(np.isclose(t, 0.0), 1.0 - beta + 4.0 * beta / np.pi, out)
    singular = np.isclose(np.abs(t), 1.0 / (4.0 * beta))
    if np.any(singular):
        val = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            - (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
        out = np.where(singular, val, out)
    return out


def rrc_taps(sps: int, span_symbols: int = 8, rolloff: float = 0.35) -> np.ndarray:
    """生成单位能量的 RRC 滤波器（长度 span_symbols*sps + 1，奇数）."""
    sps = int(sps)
    span_symbols = int(span_symbols)
    if sps < 2:
        raise ValueError("sps must be >= 2")
    if span_symbols < 1:
        raise ValueError("span_symbols must be >= 1")
    if not (0.0 <= rolloff <= 1.0):
        raise ValueError("rolloff must be in [0, 1]")
    half = span_symbols * sps // 2
    t = (np.arange(-half, half + 1, dtype=np.float64)) / float(sps)
    taps = _rrc_response(t, float(rolloff))
    energy = float(np.sum(taps ** 2))
    if energy <= 0:
        raise ValueError("degenerate RRC taps")
    return (taps / np.sqrt(energy)).astype(np.float64)


def upsample(symbols, sps: int) -> np.ndarray:
    """符号 -> 插零上采样（每个符号占 sps 个采样点）."""
    sps = int(sps)
    arr = np.asarray(symbols).reshape(-1)
    out = np.zeros(arr.size * sps, dtype=np.complex128)
    out[::sps] = arr
    return out


def filter_delay(taps) -> float:
    """滤波器群延迟（采样点）."""
    return (len(np.asarray(taps).reshape(-1)) - 1) / 2.0


def pulse_shape(symbols, sps: int, rolloff: float = 0.35, span_symbols: int = 8, taps=None):
    """成型滤波，返回 (波形, 群延迟采样数)."""
    if taps is None:
        taps = rrc_taps(sps, span_symbols=span_symbols, rolloff=rolloff)
    waveform = np.convolve(upsample(symbols, sps), taps).astype(np.complex128)
    return waveform, filter_delay(taps)


def matched_filter(x, taps) -> np.ndarray:
    """匹配滤波（RRC 对称，故直接卷积）."""
    return np.convolve(np.asarray(x).reshape(-1), np.asarray(taps).reshape(-1)).astype(np.complex128)


def symbol_sample_indices(num_symbols: int, sps: int, delay: float, offset: int = 0):
    """成型 + 匹配滤波之后，各符号的最佳采样点索引（k*sps + 2*delay + offset）."""
    return (np.arange(num_symbols) * int(sps) + int(round(2.0 * delay)) + int(offset)).astype(np.int64)
