"""软判决：max-log 比特 LLR.

符号约定（全项目统一）
--------------------
    LLR = log( P(bit=0 | y) / P(bit=1 | y) )
**正数表示比特 0 更可能**，负数是比特 1。这与信道译码教材的常见约定一致，
下游 Viterbi/LDPC（M4）直接用这个符号。

max-log 近似
------------
    LLR_p = ( min_{s: bit_p=1} |y-s|^2 - min_{s: bit_p=0} |y-s|^2 ) / sigma^2
其中 sigma^2 是**判决点上的复噪声方差**。AGC 把符号功率归一到 1 之后，
匹配滤波输出的 Es/N0 = 1/sigma^2，所以 sigma^2 = 10^(-EsN0/10)。
这一步必须用 M1 估计出来的 SNR —— 这也正是 M1 要输出 SNR 的原因之一。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from spxh.core.synth import mapping

__all__ = ["symbol_noise_variance", "max_log_llr", "rotation_symmetry_order"]


def symbol_noise_variance(es_n0_db: float, minimum: float = 1e-6) -> float:
    """由 Es/N0（dB）得到归一化判决点上的复噪声方差."""
    return float(max(10.0 ** (-float(es_n0_db) / 10.0), minimum))


def rotation_symmetry_order(modulation: str) -> int:
    """星座的旋转模糊阶数（BPSK 180 度、QPSK/16QAM 90 度、8PSK 45 度、FSK 无）."""
    mod = mapping.normalize_modulation(modulation)
    return {"bpsk": 2, "qpsk": 4, "8psk": 8, "16qam": 4}.get(mod, 1)


def _bit_labels(modulation: str) -> np.ndarray:
    mod = mapping.normalize_modulation(modulation)
    points = mapping.num_points(mod)
    bps = mapping.bits_per_symbol(mod)
    indices = np.arange(points)
    return mapping.indices_to_bits(indices, mod).reshape(points, bps)


def max_log_llr(samples: np.ndarray, modulation: str, sigma2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (llr 扁平数组, 硬判决比特, 硬判决星座索引)."""
    mod = mapping.normalize_modulation(modulation)
    if mod in ("2fsk", "4fsk"):
        raise ValueError("FSK 用 fsk_llr（非相干能量检测），不走最大似然欧氏距离")
    constellation = mapping.constellation(mod)
    labels = _bit_labels(mod)                      # (points, bps)
    bps = labels.shape[1]
    y = np.asarray(samples, dtype=np.complex128).reshape(-1)

    distances = np.abs(y[:, None] - constellation[None, :]) ** 2      # (N, points)
    indices = np.argmin(distances, axis=1)
    decision_bits = labels[indices].reshape(-1).astype(np.uint8)

    sigma = max(float(sigma2), 1e-12)
    llr = np.zeros((y.size, bps), dtype=np.float64)
    for bit in range(bps):
        ones = labels[:, bit] == 1
        zeros = ~ones
        d1 = np.min(distances[:, ones], axis=1)
        d0 = np.min(distances[:, zeros], axis=1)
        llr[:, bit] = (d1 - d0) / sigma
    llr = np.clip(llr, -60.0, 60.0)
    return llr.reshape(-1), decision_bits, indices.astype(np.int64)
