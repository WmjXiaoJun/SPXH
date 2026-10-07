"""原始 I/Q 高阶累积量与峰度.

采用 Swami & Sadler 的复累积量定义（零均值、功率归一化后）：

    M20 = E[x^2]            M21 = E[|x|^2]
    M40 = E[x^4]            M42 = E[|x|^4]          M63 = E[|x|^6]
    C40 = M40 - 3*M20^2
    C42 = M42 - |M20|^2 - 2*M21^2
    C63 = M63 - 9*|M41|*M21 - 6*M21^3

理论值（归一化：|C40|/M21^2、|C42|/M21^2、|C63|/M21^3）：

    调制      |C40|   |C42|   |C63|
    BPSK      2.00    2.00    16.0
    QPSK      1.00    1.00     4.0
    8PSK      0.00    1.00     0.0
    16QAM     0.68    0.68     2.08
    FSK       0.00    1.00     0.0    （恒模、圆对称，与 8PSK 在累积量上不可分）

注意：脉冲成型带来的 ISI 会让实测值偏离理论值，所以分类器必须用同一条合成链路
（同样的 sps/滚降/成型）生成的样本训练，不能直接套理论门限。
"""
from __future__ import annotations

import numpy as np

__all__ = ["cumulant_features", "CUMULANT_THEORY"]


CUMULANT_THEORY: dict[str, dict[str, float]] = {
    "bpsk": {"c40": 2.00, "c42": 2.00, "c63": 16.0},
    "qpsk": {"c40": 1.00, "c42": 1.00, "c63": 4.0},
    "8psk": {"c40": 0.00, "c42": 1.00, "c63": 0.0},
    "16qam": {"c40": 0.68, "c42": 0.68, "c63": 2.08},
    "2fsk": {"c40": 0.00, "c42": 1.00, "c63": 0.0},
    "4fsk": {"c40": 0.00, "c42": 1.00, "c63": 0.0},
}


def cumulant_features(x: np.ndarray) -> dict[str, float]:
    """返回归一化后的累积量与峰度."""
    samples = np.asarray(x).reshape(-1).astype(np.complex128)
    if samples.size < 64:
        raise ValueError("need at least 64 samples for cumulants")
    centered = samples - samples.mean()
    power = float(np.mean(np.abs(centered) ** 2))
    if power <= 0:
        return {"cum_c40": 0.0, "cum_c42": 0.0, "cum_c63": 0.0, "cum_kurtosis": 0.0}
    norm = centered / np.sqrt(power)

    m20 = complex(np.mean(norm ** 2))
    m21 = float(np.mean(np.abs(norm) ** 2))
    m40 = complex(np.mean(norm ** 4))
    m41 = complex(np.mean(norm ** 3 * np.conj(norm)))
    m42 = float(np.mean(np.abs(norm) ** 4))
    m63 = float(np.mean(np.abs(norm) ** 6))

    c40 = m40 - 3.0 * m20 ** 2
    c42 = m42 - abs(m20) ** 2 - 2.0 * m21 ** 2
    c63 = m63 - 9.0 * abs(m41) * m21 - 6.0 * m21 ** 3
    kurtosis = float(np.mean(np.abs(norm) ** 4) / max(m21 ** 2, 1e-12) - 2.0)

    return {
        "cum_c40": float(abs(c40) / max(m21 ** 2, 1e-12)),
        "cum_c42": float(abs(c42) / max(m21 ** 2, 1e-12)),
        "cum_c63": float(abs(c63) / max(m21 ** 3, 1e-12)),
        "cum_kurtosis": kurtosis,
    }
