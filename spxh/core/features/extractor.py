"""五域 23 维特征提取（M2 对外唯一入口）.

    from spxh.core.features import extract_features
    features = extract_features(signal)          # 内部会先跑 M1 的 analyze_signal

特征顺序**固定**（FEATURE_NAMES），训练与推理必须一致；模型文件里会保存这份名单做校验。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from spxh.core.dsp import stft_waterfall, welch_psd
from spxh.core.estimate import AnalyzeResult, analyze_signal
from spxh.core.features.constellation import constellation_features, recover_symbols
from spxh.core.features.cumulant import cumulant_features
from spxh.core.features.cyclostationary import cyclostationary_features
from spxh.core.features.spectral import spectral_features
from spxh.core.features.timefreq import timefreq_features
from spxh.core.features.types import FeatureSet
from spxh.core.types import Signal

__all__ = ["FEATURE_NAMES", "FEATURE_GROUPS", "FeatureConfig", "FeatureSet", "extract_features"]

FEATURE_NAMES: tuple[str, ...] = (
    # 原始 I/Q 累积量（4）
    "cum_c40",
    "cum_c42",
    "cum_c63",
    "cum_kurtosis",
    # 频谱（5）
    "spec_bw_over_rs",
    "spec_flatness",
    "spec_peak_snr_db",
    "spec_rolloff_est",
    "spec_skew",
    # STFT 时频 / 包络（3）
    "tf_env_var",
    "tf_env_cv",
    "tf_centroid_std",
    # 星座（6）
    "const_evm_percent",
    "const_cluster_count",
    "const_dispersion",
    "const_radial_spread",
    "const_phase_symmetry_order",
    "const_phase_symmetry_strength",
    # 循环平稳（5）
    "cyc_line_prominence_db",
    "cyc_harmonic_count",
    "cyc_peak_to_median_db",
    "cyc_entropy",
    "cyc_line_count",
)

FEATURE_GROUPS: dict[str, list[int]] = {
    "cumulant": [0, 1, 2, 3],
    "spectral": [4, 5, 6, 7, 8],
    "timefreq": [9, 10, 11],
    "constellation": [12, 13, 14, 15, 16, 17],
    "cyclostationary": [18, 19, 20, 21, 22],
}

if len(FEATURE_NAMES) != 23:
    raise RuntimeError("FEATURE_NAMES 必须是 23 维，当前 " + str(len(FEATURE_NAMES)))


@dataclass
class FeatureConfig:
    """特征提取假设（必须与训练时一致，模型文件里会保存）."""

    rolloff_assumed: float = 0.35
    span_symbols: int = 16
    nperseg: int = 1024
    nfft: Optional[int] = None
    waterfall_nperseg: int = 256
    max_symbols: int = 1200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_features(
    signal: Signal,
    estimates: Optional[AnalyzeResult] = None,
    config: Optional[FeatureConfig] = None,
) -> FeatureSet:
    """提取五域 23 维特征."""
    config = config or FeatureConfig()
    warnings: list[str] = []
    if estimates is None:
        estimates = analyze_signal(signal, nperseg=config.nperseg, nfft=config.nfft)

    symbol_rate = float(estimates.symbol_rate.value)
    if not np.isfinite(symbol_rate) or symbol_rate <= 0:
        symbol_rate = float(signal.sample_rate) / 8.0
        warnings.append("符号速率估计无效，退回 fs/8 作为星座/循环特征的假设")
    cfo_hz = float(estimates.cfo.value) if np.isfinite(estimates.cfo.value) else 0.0

    psd = welch_psd(signal, nperseg=int(config.nperseg), nfft=config.nfft)
    values: dict[str, float] = {}

    # 先做盲符号恢复：累积量与星座特征都必须建立在**符号速率采样点**上，
    # 直接对过采样波形求高阶累积量会因为中心极限效应趋近 0（实测 BPSK 的 |C40| 只有 0.007，
    # 而符号速率采样下应接近 2）。
    points = recover_symbols(
        signal,
        cfo_hz=cfo_hz,
        symbol_rate=symbol_rate,
        rolloff=float(config.rolloff_assumed),
        span_symbols=int(config.span_symbols),
        max_symbols=int(config.max_symbols),
    )
    values.update(cumulant_features(points))
    values.update(spectral_features(psd, symbol_rate))

    waterfall = stft_waterfall(signal, nperseg=int(config.waterfall_nperseg))
    values.update(timefreq_features(waterfall, symbol_rate, samples=signal.samples))
    values.update(constellation_features(points, symbol_rate))
    values.update(cyclostationary_features(signal, symbol_rate))

    ordered = np.asarray([float(values.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float64)
    evidence = {
        "symbol_rate": symbol_rate,
        "cfo_hz": cfo_hz,
        "es_n0_db": float(estimates.snr.value),
        "snr_confidence": float(estimates.snr.confidence),
        "symbol_rate_confidence": float(estimates.symbol_rate.confidence),
        "cfo_confidence": float(estimates.cfo.confidence),
        "occupied_bandwidth_hz": float(estimates.occupied.get("bandwidth", 0.0)),
        "num_symbols_used": int(points.size),
        "config": config.to_dict(),
        "feature_version": "m2-23dim-v1",
    }
    return FeatureSet(
        names=list(FEATURE_NAMES),
        values=ordered,
        groups={k: list(v) for k, v in FEATURE_GROUPS.items()},
        evidence=evidence,
        warnings=warnings + list(estimates.warnings),
    )
