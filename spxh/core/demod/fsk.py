"""FSK 非相干解调：逐符号音调相关 + 能量型软信息.

FSK 是恒模信号，没有星座形状，也不需要载波相位跟踪（这正是它相对 PSK 的优势）。
每个符号区间内对 M 个候选音调做相关，取最大能量者判为该音调；
软信息用能量差（近似 LLR），符号约定同样满足"正数表示比特 0"。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from spxh.core.synth import mapping

__all__ = ["tone_frequencies", "fsk_demodulate", "fsk_llr"]


def tone_frequencies(modulation: str, symbol_rate: float, modulation_index: float | None = None) -> np.ndarray:
    mod = mapping.normalize_modulation(modulation)
    if mod not in ("2fsk", "4fsk"):
        raise ValueError("tone_frequencies 只适用于 FSK")
    points = mapping.num_points(mod)
    index = 0.5 if mod == "4fsk" else 1.0
    if modulation_index is not None:
        index = float(modulation_index)
    return (np.arange(points) - (points - 1) / 2.0) * index * float(symbol_rate)


def fsk_demodulate(
    signal,
    symbol_rate: float,
    cfo_hz: float = 0.0,
    modulation: str = "2fsk",
    modulation_index: float | None = None,
    sps: int | None = None,
) -> dict[str, Any]:
    """逐符号音调相关，返回索引、能量矩阵与判决符号."""
    samples = np.asarray(signal.samples).reshape(-1).astype(np.complex128)
    fs = float(signal.sample_rate)
    if sps is None:
        sps = int(round(fs / float(symbol_rate)))
    sps = max(2, int(sps))
    tones = tone_frequencies(modulation, symbol_rate, modulation_index=modulation_index)

    count = max(0, (samples.size - sps) // sps + 1)
    if count < 4:
        raise ValueError("记录太短，无法做 FSK 解调")
    n = np.arange(sps, dtype=np.float64)
    basis = np.exp(-2j * np.pi * (tones[:, None] + float(cfo_hz)) * n[None, :] / fs)  # (M, sps)

    windows = samples[: count * sps].reshape(count, sps)
    # basis[m, n] = exp(-j2*pi*f_m*n/fs)，相关直接用 basis（**不能再取共轭**，
    # 否则等价于用 -f_m 去相关，2FSK 会得到完全相反的判决）
    correlations = windows @ basis.T                   # (count, M)
    energies = np.abs(correlations) ** 2
    indices = np.argmax(energies, axis=1)
    symbols = correlations[np.arange(count), indices]
    return {
        "indices": indices.astype(np.int64),
        "energies": energies,
        "correlations": correlations,
        "symbols": symbols,
        "tones_hz": tones,
        "sps": sps,
        "num_symbols": count,
    }


def fsk_llr(energies: np.ndarray, modulation: str, sigma2: float) -> tuple[np.ndarray, np.ndarray]:
    """能量型软信息：返回 (llr, 硬判决比特).

    对每个比特位取"该位=0 的音调里的最大能量"与"该位=1 的最大能量"之差作为 LLR，
    正数表示比特 0 更可能。
    """
    mod = mapping.normalize_modulation(modulation)
    points = mapping.num_points(mod)
    bps = mapping.bits_per_symbol(mod)
    labels = mapping.indices_to_bits(np.arange(points), mod).reshape(points, bps)
    energy = np.asarray(energies, dtype=np.float64)
    sigma = max(float(sigma2), 1e-12)

    llr = np.zeros((energy.shape[0], bps), dtype=np.float64)
    for bit in range(bps):
        zeros = labels[:, bit] == 0
        ones = ~zeros
        e0 = np.max(energy[:, zeros], axis=1)
        e1 = np.max(energy[:, ones], axis=1)
        llr[:, bit] = (e0 - e1) / sigma
    indices = np.argmax(energy, axis=1)
    decision_bits = labels[indices].reshape(-1).astype(np.uint8)
    return np.clip(llr, -60.0, 60.0).reshape(-1), decision_bits
