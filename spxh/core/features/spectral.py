"""频谱域特征：带宽、平坦度、峰噪比、滚降估计、谱偏度."""
from __future__ import annotations

import numpy as np

from spxh.core.dsp.spectrum import PsdResult, occupied_bandwidth, smooth_psd

__all__ = ["spectral_features"]


def _rolloff_estimate(psd: PsdResult, symbol_rate: float, residual: np.ndarray, peak_index: int) -> float:
    """由 -20 dB 带宽估计滚降系数.

    RRC 谱在 |f| < (1-alpha)/2T 内是平顶，到 (1+alpha)/2T 处滚降到零；
    取比平顶低 20 dB 的连续边界当作 (1+alpha)/2T，于是 alpha ≈ B_20dB/Rs - 1。
    """
    peak_value = float(residual[peak_index]) if residual.size else 0.0
    if peak_value <= 0 or symbol_rate <= 0:
        return 0.0
    threshold = 0.01 * peak_value  # -20 dB
    left = peak_index
    while left > 0 and residual[left] >= threshold:
        left -= 1
    right = peak_index
    while right < residual.size - 1 and residual[right] >= threshold:
        right += 1
    bandwidth = float(psd.freqs[right] - psd.freqs[left])
    return float(np.clip(bandwidth / symbol_rate - 1.0, 0.0, 1.0))


def spectral_features(psd: PsdResult, symbol_rate: float) -> dict[str, float]:
    floor = psd.noise_floor()
    smoothed = smooth_psd(psd.psd, 9)
    residual = np.maximum(smoothed - floor, 0.0)
    peak_index = int(np.argmax(residual))
    peak_value = float(residual[peak_index])
    band = occupied_bandwidth(psd.freqs, psd.psd, peak_index=peak_index, fraction=0.99, noise_floor=floor)

    mask = (psd.freqs >= band["f_low"]) & (psd.freqs <= band["f_high"])
    inband = np.maximum(psd.psd[mask], 1e-300)
    arithmetic = float(np.mean(inband))
    geometric = float(np.exp(np.mean(np.log(inband))))
    flatness = float(np.clip(geometric / max(arithmetic, 1e-300), 0.0, 1.0))

    weights = residual / max(float(np.sum(residual)), 1e-300)
    freqs = np.asarray(psd.freqs, dtype=np.float64)
    mean_f = float(np.sum(weights * freqs))
    var_f = float(np.sum(weights * (freqs - mean_f) ** 2))
    skew = float(np.sum(weights * (freqs - mean_f) ** 3) / (var_f ** 1.5)) if var_f > 0 else 0.0

    bandwidth_over_rs = float(band["bandwidth"] / symbol_rate) if symbol_rate > 0 else 0.0
    return {
        "spec_bw_over_rs": bandwidth_over_rs,
        "spec_flatness": flatness,
        "spec_peak_snr_db": float(10.0 * np.log10(max(peak_value, 1e-300) / max(floor, 1e-300))),
        "spec_rolloff_est": _rolloff_estimate(psd, symbol_rate, residual, peak_index),
        "spec_skew": float(np.clip(skew, -10.0, 10.0)),
    }
