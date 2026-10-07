"""M1 分析阶段：把 PSD / 载频 / 符号速率 / SNR 估计串成一次调用.

这是 M1 的对外出口：analyze_signal(signal) -> AnalyzeResult，
既给人看（summary），也给机器用（to_dict），并且能与合成真值直接对账
（compare_with_truth）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from spxh.core.dsp.spectrum import PsdResult, occupied_bandwidth, psd_peak, welch_psd
from spxh.core.estimate.cfo import estimate_cfo
from spxh.core.estimate.snr import estimate_snr
from spxh.core.estimate.symbol_rate import estimate_symbol_rate
from spxh.core.estimate.types import Estimate
from spxh.core.types import GroundTruth, Signal, jsonable

__all__ = ["AnalyzeResult", "analyze_signal", "compare_with_truth"]


@dataclass
class AnalyzeResult:
    sample_rate: float
    num_samples: int
    duration_s: float
    noise_floor_psd: float
    total_power: float
    peak: dict[str, Any]
    occupied: dict[str, Any]
    cfo: Estimate
    symbol_rate: Estimate
    snr: Estimate
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def sps(self) -> float:
        if self.symbol_rate.value <= 0:
            return float("nan")
        return float(self.sample_rate / self.symbol_rate.value)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(
            {
                "sample_rate": self.sample_rate,
                "num_samples": self.num_samples,
                "duration_s": self.duration_s,
                "noise_floor_psd": self.noise_floor_psd,
                "total_power": self.total_power,
                "peak": self.peak,
                "occupied": self.occupied,
                "cfo": self.cfo.to_dict(),
                "symbol_rate": self.symbol_rate.to_dict(),
                "snr": self.snr.to_dict(),
                "sps": self.sps,
                "warnings": list(self.warnings),
                "meta": self.meta,
            }
        )

    def summary(self) -> str:
        lines = [
            "采样率 {:.1f} Hz | {} 采样点 | {:.3f} s".format(self.sample_rate, self.num_samples, self.duration_s),
            "噪声底 PSD {:.3e} | 总功率 {:.4g}".format(self.noise_floor_psd, self.total_power),
            "占用带宽 {:.1f} Hz  [{:.1f}, {:.1f}] 中心 {:.1f} Hz".format(
                self.occupied.get("bandwidth", float("nan")),
                self.occupied.get("f_low", float("nan")),
                self.occupied.get("f_high", float("nan")),
                self.occupied.get("center", float("nan")),
            ),
            self.cfo.describe(),
            self.symbol_rate.describe(),
            self.snr.describe() + (" (每符号采样点 {:.2f})".format(self.sps) if np.isfinite(self.sps) else ""),
        ]
        if self.warnings:
            lines.append("告警: " + "; ".join(self.warnings))
        return "\n".join(lines)


def analyze_signal(
    signal: Signal,
    psd: Optional[PsdResult] = None,
    nperseg: Optional[int] = 1024,
    nfft: Optional[int] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    harmonics: int = 4,
    band_fraction: float = 0.99,
) -> AnalyzeResult:
    """一次调用完成 M1 的全部估计."""
    if psd is None:
        nperseg = int(nperseg or 1024)
        psd = welch_psd(signal, nperseg=nperseg, nfft=int(nfft or nperseg * 8))

    floor = psd.noise_floor()
    peak = psd_peak(psd.freqs, psd.psd)
    occupied = occupied_bandwidth(
        psd.freqs, psd.psd, peak_index=int(peak["index"]), fraction=band_fraction, noise_floor=floor
    )

    cfo = estimate_cfo(psd=psd)
    # 物理约束：占用带宽 B 与符号速率同量级（PSK/QAM 约 B/(1+alpha)，MFSK 约 B/((M-1)h+2)），
    # 因此 Rs 不可能远小于 B。用 0.25*B 作为搜索下界，可以排除循环剖面低频谱底的伪峰
    # （实测 4FSK 在 10 dB 时会把 977 Hz 判成 149 Hz）。
    bandwidth = float(occupied.get("bandwidth", 0.0))
    symbol_rate = estimate_symbol_rate(
        signal,
        fmin=fmin,
        fmax=fmax,
        harmonics=harmonics,
        bandwidth_hint=bandwidth if bandwidth > 0 else None,
    )
    snr = estimate_snr(signal, symbol_rate=symbol_rate.value, psd=psd)

    warnings: list[str] = []
    nyquist = 0.5 * psd.sample_rate
    edge = 0.98 * nyquist
    if occupied["f_low"] <= -edge or occupied["f_high"] >= edge:
        warnings.append("占用带宽触到奈奎斯特边缘，噪声底估计可能偏高")
    if cfo.confidence < 0.35:
        warnings.append("载频估计置信度低（{:.2f}）".format(cfo.confidence))
    if symbol_rate.confidence < 0.35:
        warnings.append("符号速率估计置信度低（{:.2f}）".format(symbol_rate.confidence))
    if symbol_rate.value > 0 and signal.sample_rate / symbol_rate.value < 2.5:
        warnings.append("估计出的每符号采样点少于 2.5，符号速率可能被高估")
    if signal.duration < 0.05:
        warnings.append("记录过短（{:.3f} s），估计方差大".format(signal.duration))
    num_segments = max(1, (signal.num_samples - psd.nperseg) // max(psd.nperseg // 2, 1) + 1)
    if num_segments < 4:
        warnings.append(
            "Welch 平均段数仅 {}（nperseg={}，记录 {} 点），谱估计方差偏大".format(
                num_segments, psd.nperseg, signal.num_samples
            )
        )

    return AnalyzeResult(
        sample_rate=signal.sample_rate,
        num_samples=signal.num_samples,
        duration_s=signal.duration,
        noise_floor_psd=float(floor),
        total_power=float(psd.total_power()),
        peak=peak,
        occupied=occupied,
        cfo=cfo,
        symbol_rate=symbol_rate,
        snr=snr,
        warnings=warnings,
        meta={
            "psd_nperseg": psd.nperseg,
            "psd_df_hz": psd.df,
            "estimator": "spxh-m1",
        },
    )


def compare_with_truth(result: AnalyzeResult, truth: GroundTruth) -> dict[str, Any]:
    """把估计结果与合成真值对账（M1 验收的核心接口）."""
    sps_true = float(truth.sps)
    sps_est = result.sps
    cfo_err = result.cfo.value - float(truth.cfo_hz)
    sr_err = result.symbol_rate.value - float(truth.symbol_rate)
    sr_rel = sr_err / float(truth.symbol_rate) if truth.symbol_rate else float("nan")
    snr_err = result.snr.value - float(truth.snr_db)
    return jsonable(
        {
            "modulation": truth.modulation,
            "cfo_true_hz": float(truth.cfo_hz),
            "cfo_est_hz": result.cfo.value,
            "cfo_error_hz": cfo_err,
            "symbol_rate_true_hz": float(truth.symbol_rate),
            "symbol_rate_est_hz": result.symbol_rate.value,
            "symbol_rate_error_hz": sr_err,
            "symbol_rate_rel_error": sr_rel,
            "sps_true": sps_true,
            "sps_est": sps_est,
            "es_n0_true_db": float(truth.snr_db),
            "es_n0_est_db": result.snr.value,
            "es_n0_error_db": snr_err,
            "cfo_confidence": result.cfo.confidence,
            "symbol_rate_confidence": result.symbol_rate.confidence,
            "snr_confidence": result.snr.confidence,
            "warnings": list(result.warnings),
        }
    )
