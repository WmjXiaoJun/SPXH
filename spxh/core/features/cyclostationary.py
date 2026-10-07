"""循环平稳域特征：复用 M1 的循环剖面.

对每个候选符号倍频 k*Rs 检查是否存在显著谱线，并统计剖面的峰值性与熵。
FSK 恒模、没有包络起伏，这一域是把它和 8PSK 区分开的关键（两者累积量几乎相同）。
"""
from __future__ import annotations

import numpy as np

from spxh.core.estimate.symbol_rate import cyclic_profile, detect_peaks

__all__ = ["cyclostationary_features"]


def cyclostationary_features(signal, symbol_rate: float, harmonics: int = 4) -> dict[str, float]:
    if symbol_rate <= 0:
        return {
            "cyc_line_prominence_db": 0.0,
            "cyc_harmonic_count": 0.0,
            "cyc_peak_to_median_db": 0.0,
            "cyc_entropy": 0.0,
            "cyc_line_count": 0.0,
        }
    samples = np.asarray(signal.samples).reshape(-1)
    fs = float(signal.sample_rate)
    alpha, profile = cyclic_profile(samples, fs)
    db = 10.0 * np.log10(np.maximum(profile, 1e-300))

    fmax = float(min(alpha[-1], (harmonics + 0.5) * symbol_rate))
    peaks = detect_peaks(alpha, profile, fmin=0.25 * symbol_rate, fmax=max(fmax, 0.5 * symbol_rate))

    tolerance = max(15.0, 0.01 * symbol_rate)
    harmonic_count = 0
    line_prominence = 0.0
    for k in range(1, int(harmonics) + 1):
        target = k * symbol_rate
        near = [p for p in peaks if abs(p["frequency"] - target) <= tolerance]
        if near:
            harmonic_count += 1
            if k == 1:
                line_prominence = float(max(p["level_db"] for p in near))

    band = (alpha >= 0.5 * symbol_rate) & (alpha <= 1.5 * symbol_rate)
    median_db = float(np.median(db[band])) if np.any(band) else float(np.median(db))
    peak_db = float(np.max(db[band])) if np.any(band) else median_db

    window = (alpha >= 0.05 * symbol_rate) & (alpha <= fmax)
    segment = np.maximum(profile[window], 0.0)
    total = float(np.sum(segment))
    if total > 0 and segment.size > 1:
        probabilities = segment / total
        probabilities = probabilities[probabilities > 0]
        entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(segment.size))
    else:
        entropy = 0.0

    return {
        "cyc_line_prominence_db": line_prominence,
        "cyc_harmonic_count": float(harmonic_count),
        "cyc_peak_to_median_db": float(peak_db - median_db),
        "cyc_entropy": float(np.clip(entropy, 0.0, 1.0)),
        "cyc_line_count": float(len(peaks)),
    }
