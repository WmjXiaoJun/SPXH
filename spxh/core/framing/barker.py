"""Barker-13 前导序列与滑动相关.

Barker-13 = 1 1 1 1 1 0 0 1 1 0 1 0 1（+1 记 1，-1 记 0）。
非周期自相关旁瓣幅值不超过 1，即主瓣:旁瓣 = 13:1，低信噪比下仍有清晰相关峰。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "BARKER_13",
    "BARKER_PM",
    "BARKER_LEN",
    "barker_bits",
    "barker_pm",
    "aperiodic_autocorrelation",
    "correlate_bits",
    "find_sync",
    "verify_barker_property",
]

_BARKER_PM: dict[int, tuple[int, ...]] = {
    2: (1, -1),
    3: (1, 1, -1),
    4: (1, 1, -1, 1),
    5: (1, 1, 1, -1, 1),
    7: (1, 1, 1, -1, -1, 1, -1),
    11: (1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1),
    13: (1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1),
}

BARKER_PM: dict[int, np.ndarray] = {n: np.asarray(seq, dtype=np.int8) for n, seq in _BARKER_PM.items()}
BARKER_LEN = 13
BARKER_13 = np.asarray(_BARKER_PM[13], dtype=np.int8)


def barker_pm(n: int = 13) -> np.ndarray:
    """返回 +/-1 形式的 Barker 序列."""
    if n not in BARKER_PM:
        raise ValueError("Barker length " + str(n) + " not supported (available: " + str(sorted(BARKER_PM)) + ")")
    return BARKER_PM[n].copy()


def barker_bits(n: int = 13) -> np.ndarray:
    """返回 0/1 形式的 Barker 序列（-1 -> 0）."""
    return ((barker_pm(n) + 1) // 2).astype(np.uint8)


def aperiodic_autocorrelation(seq) -> np.ndarray:
    """非周期自相关（含负延迟），滞后 0 处取最大值."""
    s = np.asarray(seq, dtype=np.float64)
    n = s.size
    lags = np.arange(-(n - 1), n)
    out = np.zeros(lags.size, dtype=np.float64)
    for i, lag in enumerate(lags):
        if lag >= 0:
            out[i] = float(np.dot(s[: n - lag], s[lag:]))
        else:
            out[i] = float(np.dot(s[-lag:], s[: n + lag]))
    return out


def verify_barker_property(n: int = 13) -> bool:
    """校验非周期自相关旁瓣幅值不超过 1."""
    ac = aperiodic_autocorrelation(barker_pm(n))
    center = ac.size // 2
    if ac[center] != n:
        return False
    sidelobes = np.delete(ac, center)
    return bool(np.max(np.abs(sidelobes)) <= 1.0)


def correlate_bits(bits, n: int = 13, normalized: bool = True) -> np.ndarray:
    """对 0/1 比特流做滑动相关，输出每个起点处的相关值（长度 = len(bits)-n+1）."""
    arr = np.asarray(bits).reshape(-1).astype(np.int8)
    ref = barker_pm(n).astype(np.int8)
    if arr.size < n:
        return np.zeros(0, dtype=np.float64)
    pm = (arr * 2 - 1).astype(np.float64)
    corr = np.correlate(pm, ref.astype(np.float64), mode="valid")
    if normalized:
        corr = corr / float(n)
    return corr


def find_sync(bits, n: int = 13, threshold: float = 0.8, min_gap: int = 0) -> list[int]:
    """在比特流中寻找相关峰位置（返回起点索引列表）."""
    corr = correlate_bits(bits, n=n, normalized=True)
    if corr.size == 0:
        return []
    hits: list[int] = []
    order = np.argsort(-corr)
    taken = np.zeros(corr.size, dtype=bool)
    for idx in order:
        if corr[idx] < threshold:
            break
        if taken[idx]:
            continue
        if min_gap > 0 and any(abs(int(idx) - h) < min_gap for h in hits):
            continue
        hits.append(int(idx))
        lo = max(0, int(idx) - min_gap)
        hi = min(corr.size, int(idx) + min_gap + 1)
        taken[lo:hi] = True
    return sorted(hits)
