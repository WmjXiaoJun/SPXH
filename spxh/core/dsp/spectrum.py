"""功率谱估计：Welch PSD、噪声底、谱峰、占用带宽.

全部使用**双侧**功率谱密度（f 从 -fs/2 到 fs/2，单位 功率/Hz），
满足 sum(psd) * df = mean(|x|^2)，因此谱域功率与波形功率可直接对账。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.signal import welch as scipy_welch

from spxh.core.types import Signal

__all__ = [
    "PsdResult",
    "welch_psd",
    "estimate_noise_floor",
    "parabolic_peak",
    "smooth_psd",
    "psd_peak",
    "occupied_bandwidth",
    "band_power",
]


def _as_samples(x) -> tuple[np.ndarray, Optional[float]]:
    if isinstance(x, Signal):
        return x.samples, float(x.sample_rate)
    return np.asarray(x).reshape(-1), None


@dataclass
class PsdResult:
    """双侧 PSD 及其派生量."""

    freqs: np.ndarray
    psd: np.ndarray
    sample_rate: float
    nperseg: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0]) if self.freqs.size > 1 else float(self.sample_rate)

    def total_power(self) -> float:
        """sum(psd)*df，应等于 mean(|x|^2)."""
        return float(np.sum(self.psd) * self.df)

    def noise_floor(self, edge_fraction: float = 0.1) -> float:
        return estimate_noise_floor(self.freqs, self.psd, edge_fraction=edge_fraction)

    def peak(self, fmin: Optional[float] = None, fmax: Optional[float] = None, smooth_bins: int = 5) -> dict[str, Any]:
        return psd_peak(self.freqs, self.psd, fmin=fmin, fmax=fmax, smooth_bins=smooth_bins)

    def occupied_band(self, fraction: float = 0.99, smooth_bins: int = 5, noise_floor: Optional[float] = None) -> dict[str, Any]:
        peak = self.peak(smooth_bins=smooth_bins)
        return occupied_bandwidth(
            self.freqs, self.psd, peak_index=int(peak["index"]), fraction=fraction, noise_floor=noise_floor
        )

    def residual(self, smooth_bins: int = 5, noise_floor: Optional[float] = None) -> np.ndarray:
        """扣掉噪声底并裁剪到非负的残余谱（信号成分）."""
        floor = self.noise_floor() if noise_floor is None else float(noise_floor)
        return np.maximum(smooth_psd(self.psd, smooth_bins) - floor, 0.0)

    def band_power(self, f_low: float, f_high: float) -> float:
        return band_power(self.freqs, self.psd, f_low, f_high)


def welch_psd(
    x,
    sample_rate: Optional[float] = None,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    nfft: Optional[int] = None,
    detrend: str = "constant",
) -> PsdResult:
    """Welch 平均周期图（双侧 PSD，频率升序）."""
    samples, fs = _as_samples(x)
    if sample_rate is not None:
        fs = float(sample_rate)
    if fs is None:
        raise ValueError("sample_rate is required when passing raw samples")
    if samples.size < 16:
        raise ValueError("need at least 16 samples for PSD")
    if nperseg is None:
        nperseg = int(min(samples.size, max(64, 1 << int(np.floor(np.log2(max(samples.size // 8, 64)))))))
    nperseg = int(min(nperseg, samples.size))
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = scipy_welch(
        samples,
        fs=float(fs),
        window=window,
        nperseg=nperseg,
        noverlap=int(noverlap),
        nfft=nfft,
        detrend=detrend,
        return_onesided=False,
        scaling="density",
    )
    order = np.argsort(freqs)
    return PsdResult(
        freqs=np.asarray(freqs)[order],
        psd=np.asarray(psd)[order],
        sample_rate=float(fs),
        nperseg=nperseg,
        meta={"window": window, "noverlap": int(noverlap), "nfft": int(nfft or nperseg), "num_samples": int(samples.size)},
    )


def estimate_noise_floor(freqs: np.ndarray, psd: np.ndarray, edge_fraction: float = 0.1) -> float:
    """用频带两端（各占 edge_fraction）的均值估计噪声功率谱密度.

    对白噪声该均值是无偏的（不像中位数会低估）。前提是信号占用带宽没有铺满
    奈奎斯特带 —— 在 sps >= 4 的常见配置下成立；若占用带宽触到边缘，
    analyze 会给出告警而不是默默返回一个偏高的底噪。
    """
    freqs = np.asarray(freqs)
    psd = np.asarray(psd, dtype=np.float64)
    n = psd.size
    if n < 8:
        return float(np.mean(psd))
    k = max(1, int(round(n * float(edge_fraction))))
    edges = np.concatenate([psd[:k], psd[-k:]])
    return float(np.mean(edges))


def smooth_psd(psd: np.ndarray, bins: int = 5) -> np.ndarray:
    """滑动平均（边缘用 replicate padding），用于抑制谱线抖动."""
    bins = int(bins)
    if bins <= 1:
        return np.asarray(psd, dtype=np.float64)
    arr = np.asarray(psd, dtype=np.float64)
    if arr.size < bins:
        return arr.copy()
    kernel = np.ones(bins, dtype=np.float64) / bins
    padded = np.pad(arr, (bins // 2, bins - 1 - bins // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parabolic_peak(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float]:
    """对 (index-1, index, index+1) 做抛物线插值，返回 (更精细的横坐标, 峰值).

    在 log 域插值更接近谱峰的真实形状；y 需为正。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    index = int(index)
    if index <= 0 or index >= y.size - 1:
        return float(x[index]), float(y[index])
    y0, y1, y2 = np.log(np.maximum(y[index - 1 : index + 2], 1e-300))
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-15:
        return float(x[index]), float(y[index])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    step = float(x[index + 1] - x[index])
    peak_value = float(np.exp(y1 - 0.25 * (y0 - y2) * delta))
    return float(x[index] + delta * step), peak_value


def psd_peak(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    smooth_bins: int = 5,
) -> dict[str, Any]:
    """在给定频段内找谱峰（抛物线插值 + 噪声底对比）."""
    freqs = np.asarray(freqs, dtype=np.float64)
    smoothed = smooth_psd(psd, smooth_bins)
    mask = np.ones(freqs.size, dtype=bool)
    if fmin is not None:
        mask &= freqs >= float(fmin)
    if fmax is not None:
        mask &= freqs <= float(fmax)
    if not np.any(mask):
        raise ValueError("empty frequency range for peak search")
    indices = np.flatnonzero(mask)
    local = indices[int(np.argmax(smoothed[indices]))]
    peak_freq, peak_value = parabolic_peak(freqs, smoothed, local)
    floor = estimate_noise_floor(freqs, psd)
    return {
        "index": int(local),
        "frequency": peak_freq,
        "value": peak_value,
        "noise_floor": floor,
        "snr_db": float(10.0 * np.log10(max(peak_value, 1e-300) / max(floor, 1e-300))),
        "bin_frequency": float(freqs[local]),
    }


def occupied_bandwidth(
    freqs: np.ndarray,
    psd: np.ndarray,
    peak_index: int,
    fraction: float = 0.99,
    noise_floor: Optional[float] = None,
    smooth_bins: int = 5,
) -> dict[str, Any]:
    """以谱峰为中心、按功率占比展开的占用带宽.

    先**平滑**再扣噪声底，然后按到峰值的频率距离排序、累加残余谱功率直到
    fraction * 总残余功率。平滑这一步不能省：未平滑时噪声在底噪上下的起伏会被
    裁成"信号"，低信噪比下能把 1350 Hz 的信号带宽撑到 6000 Hz。
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = smooth_psd(np.asarray(psd, dtype=np.float64), smooth_bins)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    peak_index = int(np.clip(peak_index, 0, psd.size - 1))
    if noise_floor is None:
        noise_floor = estimate_noise_floor(freqs, psd)
    residual = np.maximum(psd - float(noise_floor), 0.0)
    if float(np.sum(residual)) <= 0.0:
        residual = psd.copy()
        psd = residual
        total = float(np.sum(psd))
    else:
        # 噪声去偏：裁剪后的噪声残余在噪声区仍是正的（半个分布被裁掉），
        # 用频带两端估计每个 bin 的平均残余并整体扣除，否则低信噪比下
        # 这些残余会把占用带宽从 1350 Hz 撑到 6000 Hz。
        edge_bins = max(1, int(round(residual.size * 0.1)))
        noise_residual = np.concatenate([residual[:edge_bins], residual[-edge_bins:]])
        bias_per_bin = float(np.mean(noise_residual))
        corrected = float(np.sum(residual)) - bias_per_bin * residual.size
        total = corrected if corrected > 0 else float(np.sum(residual))
        psd = residual
    if total <= 0:
        raise ValueError("psd must be non-negative")
    distances = np.abs(np.arange(psd.size) - peak_index)
    order = np.argsort(distances, kind="stable")
    cumulative = np.cumsum(psd[order])
    cutoff = int(np.searchsorted(cumulative, fraction * total)) + 1
    selected = order[:cutoff]
    lo = int(selected.min())
    hi = int(selected.max())
    band_power_value = float(np.sum(psd[lo : hi + 1]))
    return {
        "f_low": float(freqs[lo]),
        "f_high": float(freqs[hi]),
        "bandwidth": float(freqs[hi] - freqs[lo]),
        "center": float(0.5 * (freqs[lo] + freqs[hi])),
        "captured_fraction": float(band_power_value / total),
        "index_low": lo,
        "index_high": hi,
    }


def band_power(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    """频段 [f_low, f_high] 内的功率（对 PSD 做矩形积分）."""
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    if freqs.size < 2:
        return 0.0
    df = float(freqs[1] - freqs[0])
    lo, hi = (float(f_low), float(f_high)) if f_low <= f_high else (float(f_high), float(f_low))
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(psd[mask]) * df)
