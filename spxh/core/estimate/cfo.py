"""载频（频偏）估计.

三种互补的谱域估计量，取中位数做稳健融合：

1. centroid    残余谱（扣噪声底 + 平滑）的功率质心。对 PSK/QAM 的对称谱最稳。
2. symmetry    谱对称性匹配：A(k) = sum_i r[i] * r[2k-i]，在真实载频处取极大。
               对多音（FSK）谱只要音调位置对称就成立，不受各音调强弱不均影响。
3. edge_mid    残余谱在阈值处的左右边界中点（对 FSK 音调位置敏感）。

为什么不直接取谱峰：RRC 成型谱是**平顶**的，峰位在平顶上随机漂移，
用峰位当载频估计会引入几十 Hz 的随机误差。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.signal import fftconvolve

from spxh.core.dsp.spectrum import (
    PsdResult,
    estimate_noise_floor,
    occupied_bandwidth,
    parabolic_peak,
    psd_peak,
    smooth_psd,
    welch_psd,
)
from spxh.core.estimate.types import Estimate
from spxh.core.types import Signal

__all__ = [
    "residual_spectrum",
    "cfo_centroid",
    "cfo_symmetry",
    "cfo_edge_midpoint",
    "estimate_cfo",
]


def residual_spectrum(psd: PsdResult, smooth_bins: int = 9, noise_floor: Optional[float] = None) -> np.ndarray:
    """扣掉噪声底并裁剪到非负的残余谱（即信号成分）."""
    floor = psd.noise_floor() if noise_floor is None else float(noise_floor)
    return np.maximum(smooth_psd(psd.psd, smooth_bins) - floor, 0.0)


def cfo_centroid(freqs: np.ndarray, residual: np.ndarray) -> float:
    total = float(np.sum(residual))
    if total <= 0:
        return 0.0
    return float(np.sum(np.asarray(freqs) * residual) / total)


def cfo_symmetry(freqs: np.ndarray, residual: np.ndarray) -> float:
    """谱对称中心：最大化 sum_i r[i] r[2k-i] 的 k 对应的频率."""
    r = np.asarray(residual, dtype=np.float64)
    if r.size < 8 or float(np.sum(r)) <= 0:
        return 0.0
    conv = fftconvolve(r, r, mode="full")
    folded = conv[0::2][: r.size]
    index = int(np.argmax(folded))
    refined, _ = parabolic_peak(np.asarray(freqs, dtype=np.float64), np.maximum(folded, 1e-300), index)
    return float(refined)


def cfo_edge_midpoint(freqs: np.ndarray, residual: np.ndarray, threshold: float = 0.25) -> float:
    """残余谱在 peak*threshold 处的左右边界中点."""
    r = np.asarray(residual, dtype=np.float64)
    peak = float(np.max(r)) if r.size else 0.0
    if peak <= 0:
        return 0.0
    indices = np.flatnonzero(r >= threshold * peak)
    if indices.size < 2:
        return float(freqs[int(np.argmax(r))])
    return float(0.5 * (freqs[indices[0]] + freqs[indices[-1]]))


def estimate_cfo(
    signal: Optional[Signal] = None,
    psd: Optional[PsdResult] = None,
    smooth_bins: int = 9,
    nperseg: Optional[int] = None,
    nfft: Optional[int] = None,
) -> Estimate:
    """稳健融合三种谱域载频估计，返回带置信度与证据的估计值."""
    if psd is None:
        if signal is None:
            raise ValueError("either signal or psd is required")
        nperseg = nperseg or 1024
        nfft = nfft or int(nperseg * 8)
        psd = welch_psd(signal, nperseg=nperseg, nfft=nfft)
    freqs = np.asarray(psd.freqs, dtype=np.float64)
    floor = psd.noise_floor()
    residual = residual_spectrum(psd, smooth_bins=smooth_bins, noise_floor=floor)

    candidates = {
        "centroid": cfo_centroid(freqs, residual),
        "symmetry": cfo_symmetry(freqs, residual),
        "edge_midpoint": cfo_edge_midpoint(freqs, residual),
    }
    values = np.asarray(list(candidates.values()), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return Estimate("cfo", 0.0, "Hz", 0.0, "none", {"reason": "no signal energy above noise floor"})
    fused = float(np.median(values))
    spread = float(np.max(values) - np.min(values))

    peak = psd_peak(freqs, psd.psd, smooth_bins=smooth_bins)
    band = occupied_bandwidth(freqs, psd.psd, peak_index=int(peak["index"]), fraction=0.99, noise_floor=floor)
    bandwidth = max(float(band["bandwidth"]), freqs[1] - freqs[0])

    # 置信度：候选一致性（离散度相对于带宽） x 谱峰显著度
    agreement = float(np.clip(1.0 - spread / (0.05 * bandwidth), 0.0, 1.0))
    prominence = float(np.clip((float(peak["snr_db"]) - 3.0) / 20.0, 0.0, 1.0))
    confidence = float(np.clip(0.7 * agreement + 0.3 * prominence, 0.0, 1.0))

    return Estimate(
        name="cfo",
        value=fused,
        unit="Hz",
        confidence=confidence,
        method="median(centroid, symmetry, edge_midpoint)",
        evidence={
            "candidates": candidates,
            "spread_hz": spread,
            "occupied_bandwidth_hz": bandwidth,
            "band": {"f_low": band["f_low"], "f_high": band["f_high"], "center": band["center"]},
            "noise_floor_psd": floor,
            "peak_snr_db": peak["snr_db"],
            "agreement": agreement,
            "prominence": prominence,
        },
    )
