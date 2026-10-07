"""盲信噪比估计.

主估计器（谱方法）
------------------
复高斯白噪声的双侧功率谱密度是常数 N0 = sigma^2 / fs，因此

    P_total   = mean(|x|^2) = sum(PSD) * df
    P_noise   = N0 * fs                （全带噪声功率）
    P_signal  = P_total - P_noise
    Es/N0     = P_signal / (N0 * Rs)

推导：成型后 P_signal = Es/sps（每个符号的能量摊到 sps 个采样点），
N0*Rs = (sigma^2/fs)*Rs = sigma^2/sps，故比值恰为 Es/sigma^2 = Es/N0。
只需要 Rs（由上一步符号速率估计给出）与噪声底，不需要调制类型。

交叉校验（M2M4）
----------------
M2 = E|x|^2、M4 = E|x|^4，对恒模信号 S = sqrt(2*M2^2 - M4)、N = M2 - S。
它给出的是**波形**信噪比（Es/N0 / sps），且对成型信号（包络起伏）会偏低，
因此只作为交叉校验记录在证据里，不参与融合。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from spxh.core.dsp.spectrum import PsdResult, occupied_bandwidth, psd_peak, welch_psd
from spxh.core.estimate.types import Estimate
from spxh.core.types import Signal

__all__ = ["estimate_snr_spectral", "estimate_snr_m2m4", "estimate_snr"]


def _split_power(psd: PsdResult, noise_floor: Optional[float] = None) -> dict[str, float]:
    n0 = psd.noise_floor() if noise_floor is None else float(noise_floor)
    total = psd.total_power()
    noise = float(n0 * psd.sample_rate)
    signal = max(total - noise, 0.0)
    return {"noise_floor_psd": n0, "total_power": total, "noise_power": noise, "signal_power": signal}


def estimate_snr_spectral(
    signal: Optional[Signal] = None,
    psd: Optional[PsdResult] = None,
    symbol_rate: Optional[float] = None,
    nperseg: Optional[int] = None,
    nfft: Optional[int] = None,
) -> Estimate:
    """谱域盲 SNR 估计（给定 Rs 时输出 Es/N0，否则输出波形带内 SNR）."""
    if psd is None:
        if signal is None:
            raise ValueError("either signal or psd is required")
        nperseg = nperseg or 1024
        nfft = nfft or int(nperseg * 8)
        psd = welch_psd(signal, nperseg=nperseg, nfft=nfft)
    parts = _split_power(psd)
    n0 = parts["noise_floor_psd"]
    if n0 <= 0:
        return Estimate("snr", float("inf"), "dB", 0.0, "spectral", {"reason": "noise floor is zero"})

    waveform_snr_db = 10.0 * np.log10(max(parts["signal_power"], 1e-300) / max(parts["noise_power"], 1e-300))
    evidence: dict[str, Any] = dict(parts)
    evidence["snr_waveform_db"] = waveform_snr_db

    peak = psd_peak(psd.freqs, psd.psd)
    band = occupied_bandwidth(psd.freqs, psd.psd, peak_index=int(peak["index"]), fraction=0.99, noise_floor=n0)
    evidence["band"] = {"f_low": band["f_low"], "f_high": band["f_high"], "bandwidth": band["bandwidth"]}

    if symbol_rate is None or symbol_rate <= 0:
        return Estimate(
            name="snr_waveform",
            value=waveform_snr_db,
            unit="dB",
            confidence=0.3,
            method="spectral(total-noise, without symbol rate)",
            evidence=evidence,
        )

    es_n0 = parts["signal_power"] / (n0 * float(symbol_rate))
    es_n0_db = float(10.0 * np.log10(max(es_n0, 1e-300)))
    evidence["symbol_rate"] = float(symbol_rate)
    evidence["es_n0_linear"] = es_n0

    # 置信度：峰值显著度 + 占用带宽是否触到奈奎斯特边缘（会影响噪声底估计）
    prominence = float(np.clip((float(peak["snr_db"]) - 3.0) / 20.0, 0.0, 1.0))
    edge_limit = 0.98 * 0.5 * psd.sample_rate
    touches_edge = bool(band["f_low"] <= -edge_limit or band["f_high"] >= edge_limit)
    edge_factor = 0.6 if touches_edge else 1.0
    confidence = float(np.clip((0.4 + 0.6 * prominence) * edge_factor, 0.0, 1.0))

    return Estimate(
        name="es_n0",
        value=es_n0_db,
        unit="dB",
        confidence=confidence,
        method="spectral(P_signal=(P_total-N0*fs), Es/N0=P_signal/(N0*Rs))",
        evidence=evidence,
    )


def estimate_snr_m2m4(signal: Optional[Signal] = None, samples: Optional[np.ndarray] = None) -> dict[str, Any]:
    """M2M4 盲 SNR（波形信噪比，仅对恒模信号无偏）."""
    if signal is not None:
        x = np.asarray(signal.samples).reshape(-1)
    elif samples is not None:
        x = np.asarray(samples).reshape(-1)
    else:
        raise ValueError("signal or samples is required")
    m2 = float(np.mean(np.abs(x) ** 2))
    m4 = float(np.mean(np.abs(x) ** 4))
    discriminant = 2.0 * m2 * m2 - m4
    if discriminant <= 0:
        return {"valid": False, "reason": "2*M2^2 - M4 <= 0", "m2": m2, "m4": m4}
    signal_power = float(np.sqrt(discriminant))
    noise_power = float(m2 - signal_power)
    if noise_power <= 0:
        return {
            "valid": False,
            "reason": "estimated noise power <= 0",
            "m2": m2,
            "m4": m4,
            "signal_power": signal_power,
            "snr_waveform_db": float("inf"),
        }
    return {
        "valid": True,
        "m2": m2,
        "m4": m4,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "snr_waveform_db": float(10.0 * np.log10(signal_power / noise_power)),
    }


def estimate_snr(
    signal: Signal,
    symbol_rate: Optional[float] = None,
    psd: Optional[PsdResult] = None,
    nperseg: Optional[int] = None,
    nfft: Optional[int] = None,
) -> Estimate:
    """融合估计：谱方法为主，M2M4 作为证据交叉校验."""
    estimate = estimate_snr_spectral(
        signal=signal, psd=psd, symbol_rate=symbol_rate, nperseg=nperseg, nfft=nfft
    )
    m2m4 = estimate_snr_m2m4(signal=signal)
    estimate.evidence["m2m4"] = m2m4
    if symbol_rate and symbol_rate > 0:
        sps = float(signal.sample_rate) / float(symbol_rate)
        if m2m4.get("valid"):
            estimate.evidence["m2m4_es_n0_db"] = float(m2m4["snr_waveform_db"] + 10.0 * np.log10(max(sps, 1e-12)))
        estimate.evidence["sps_used"] = sps
    if estimate.method.startswith("spectral(total-noise"):
        estimate.evidence["note"] = "未提供符号速率，只能给出波形带内 SNR"
    return estimate
