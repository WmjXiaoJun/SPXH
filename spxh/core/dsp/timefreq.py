"""时频分析：STFT 瀑布图数据（供可视化与时频特征使用）."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.signal import stft as scipy_stft

from spxh.core.types import Signal

__all__ = ["Waterfall", "stft_waterfall"]


@dataclass
class Waterfall:
    """STFT 幅度瀑布（dB），频率轴已 fftshift 到升序 [-fs/2, fs/2)."""

    times: np.ndarray
    freqs: np.ndarray
    magnitude_db: np.ndarray
    sample_rate: float
    nperseg: int

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.freqs.size), int(self.times.size))

    def frequency_profile(self, f_low: float, f_high: float) -> np.ndarray:
        """某个频段内能量随时间的演化（线性幅度）."""
        mask = (self.freqs >= f_low) & (self.freqs <= f_high)
        linear = 10.0 ** (self.magnitude_db[mask] / 20.0)
        return np.sum(linear, axis=0)

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "sample_rate": self.sample_rate,
            "nperseg": self.nperseg,
            "duration_s": float(self.times[-1] - self.times[0]) if self.times.size > 1 else 0.0,
            "freq_range": [float(self.freqs[0]), float(self.freqs[-1])],
        }


def stft_waterfall(
    x,
    sample_rate: Optional[float] = None,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
    reference: float = 1.0,
) -> Waterfall:
    """计算 STFT 幅度瀑布（dB，相对 reference）."""
    if isinstance(x, Signal):
        samples = x.samples
        fs = float(x.sample_rate) if sample_rate is None else float(sample_rate)
    else:
        samples = np.asarray(x).reshape(-1)
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing raw samples")
        fs = float(sample_rate)
    nperseg = int(min(max(16, nperseg), samples.size))
    if noverlap is None:
        noverlap = nperseg * 3 // 4
    freqs, times, zxx = scipy_stft(
        samples,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=int(noverlap),
        detrend="constant",
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    order = np.argsort(freqs)
    magnitude = np.abs(zxx)[order, :]
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12) / max(reference, 1e-12))
    return Waterfall(
        times=np.asarray(times, dtype=np.float64),
        freqs=np.asarray(freqs)[order],
        magnitude_db=magnitude_db.astype(np.float64),
        sample_rate=fs,
        nperseg=nperseg,
    )
