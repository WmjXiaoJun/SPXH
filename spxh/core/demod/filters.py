"""插值、AGC 与分数延迟（接收链的零件）."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

__all__ = ["FarrowInterpolator", "agc_normalize", "fractional_delay"]


class FarrowInterpolator:
    """三次拉格朗日插值器：在任意分数位置取样本.

    节点取 index-1, index, index+1, index+2 四个点，插值位置 x = index + mu (mu ∈ [0,1))，
    用显式拉格朗日基写成 mu 的三次多项式（这就是 Farrow 结构）：
        p(mu) = y0*L0 + y1*L1 + y2*L2 + y3*L3
        L0 = -mu(mu-1)(mu-2)/6, L1 = (mu+1)(mu-1)(mu-2)/2
        L2 = -(mu+1)mu(mu-2)/2, L3 = (mu+1)mu(mu-1)/6
    展开后常数项系数和恰为 1、其余幂次系数和为 0，保证 mu=0 时恰好取到 y1。
    """

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = np.asarray(samples, dtype=np.complex128).reshape(-1)

    def __call__(self, position: float) -> complex:
        index = int(np.floor(position))
        mu = float(position - index)
        if index < 1 or index + 2 >= self.samples.size:
            return complex(self.samples[int(np.clip(index, 0, self.samples.size - 1))])
        y0, y1, y2, y3 = (
            self.samples[index - 1],
            self.samples[index],
            self.samples[index + 1],
            self.samples[index + 2],
        )
        mu2 = mu * mu
        mu3 = mu2 * mu
        a0 = y1
        a1 = -y0 / 3.0 - y1 / 2.0 + y2 - y3 / 6.0
        a2 = y0 / 2.0 - y1 + y2 / 2.0
        a3 = -y0 / 6.0 + y1 / 2.0 - y2 / 2.0 + y3 / 6.0
        return complex(a0 + a1 * mu + a2 * mu2 + a3 * mu3)


def agc_normalize(samples: np.ndarray, reference: float = 1.0) -> tuple[np.ndarray, float]:
    """按 RMS 归一化，返回 (归一化样本, 增益)."""
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2))) if x.size else 0.0
    if rms <= 0:
        return x.copy(), 1.0
    gain = float(reference) / rms
    return x * gain, gain


def fractional_delay(samples: np.ndarray, delay_samples: float, numtaps: int = 33) -> np.ndarray:
    """用分数延迟 FIR 给信号加一个可控的定时偏移（用于测试定时环）."""
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if delay_samples == 0:
        return x.copy()
    taps = int(numtaps) | 1
    half = (taps - 1) / 2.0
    n = np.arange(taps) - half
    kernel = np.sinc(n - float(delay_samples)) * np.hamming(taps)
    kernel = kernel / np.sum(kernel)
    return lfilter(kernel, [1.0], x)
