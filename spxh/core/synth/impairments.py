"""信道损伤注入：AWGN、载波频偏、相位噪声、IQ 不平衡、DC 偏置.

SNR 定义
--------
X = 10*log10(Es/N0)，其中 Es 为**每个符号波形的能量**（PSK/QAM 单位能量成型后 Es=1；
FSK 恒模且每符号 sps 个采样点，Es=sps）。复高斯噪声方差 sigma^2 = Es / 10^(X/10)，
实部虚部各分一半。该定义与 sps、调制方式无关，可直接对应接收端匹配滤波后的 Es/N0。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

__all__ = [
    "complex_noise_variance",
    "add_awgn",
    "apply_cfo",
    "apply_phase_noise",
    "apply_iq_imbalance",
    "apply_dc_offset",
    "measure_snr_db",
    "average_power",
]


def average_power(x) -> float:
    arr = np.asarray(x).reshape(-1)
    return float(np.mean(np.abs(arr) ** 2))


def complex_noise_variance(es_per_symbol: float, snr_db: float) -> float:
    """由 Es/N0（dB）得到复噪声方差 sigma^2 = E|n|^2."""
    return float(es_per_symbol) / (10.0 ** (float(snr_db) / 10.0))


def add_awgn(x, snr_db: float, es_per_symbol: float = 1.0, rng=None, return_sigma: bool = False):
    """加入复高斯白噪声."""
    arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    sigma2 = complex_noise_variance(es_per_symbol, snr_db)
    if rng is None:
        rng = np.random.default_rng()
    noise = np.sqrt(sigma2 / 2.0) * (rng.standard_normal(arr.size) + 1j * rng.standard_normal(arr.size))
    out = arr + noise
    if return_sigma:
        return out, sigma2
    return out


def apply_cfo(x, sample_rate: float, cfo_hz: float) -> np.ndarray:
    """载波频偏（基带等效：乘以 exp(j2*pi*f*t)）."""
    arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    if cfo_hz == 0.0:
        return arr.copy()
    n = np.arange(arr.size, dtype=np.float64)
    return arr * np.exp(2j * np.pi * float(cfo_hz) * n / float(sample_rate))


def apply_phase_noise(x, sample_rate: float, rms_rad: float, bandwidth_hz=None, rng=None) -> np.ndarray:
    """一阶 OU（低通白噪声）相位噪声，稳态 RMS 为 rms_rad.

    phi[n] = a*phi[n-1] + sqrt(1-a^2)*w[n]，a = exp(-2*pi*B/fs)，
    3 dB 带宽 B 默认取 fs/1000。
    """
    arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    rms = float(rms_rad)
    if rms <= 0.0:
        return arr.copy()
    if rng is None:
        rng = np.random.default_rng()
    fs = float(sample_rate)
    bw = float(bandwidth_hz) if bandwidth_hz is not None else fs / 1000.0
    bw = max(min(bw, fs / 2.0), 1e-9)
    a = float(np.exp(-2.0 * np.pi * bw / fs))
    w = rng.standard_normal(arr.size)
    phi = rms * lfilter([np.sqrt(1.0 - a * a)], [1.0, -a], w)
    return arr * np.exp(1j * phi)


def apply_iq_imbalance(x, gain_db: float = 0.0, phase_deg: float = 0.0) -> np.ndarray:
    """IQ 幅相不平衡：y = alpha*x + beta*conj(x)."""
    arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    if gain_db == 0.0 and phase_deg == 0.0:
        return arr.copy()
    g = 10.0 ** (float(gain_db) / 20.0)
    phi = np.deg2rad(float(phase_deg))
    alpha = (1.0 + g * np.exp(1j * phi)) / 2.0
    beta = (1.0 - g * np.exp(1j * phi)) / 2.0
    return alpha * arr + beta * np.conj(arr)


def apply_dc_offset(x, offset=0j) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    if offset == 0:
        return arr.copy()
    return arr + complex(offset)


def measure_snr_db(signal, reference) -> float:
    """用无噪参考信号估计实际 SNR（dB）."""
    sig = np.asarray(signal, dtype=np.complex128).reshape(-1)
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    if sig.size != ref.size:
        raise ValueError("signal and reference must have the same length")
    noise = sig - ref
    p_ref = average_power(ref)
    p_noise = average_power(noise)
    if p_noise <= 0:
        return float("inf")
    return float(10.0 * np.log10(p_ref / p_noise))
