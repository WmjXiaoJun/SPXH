"""星座域特征：盲恢复符号点后看几何结构.

恢复流程（不需要真值）：
  1. 用 M1 估计的载频把基带搬到零频；
  2. 用一个 RRC 匹配滤波（滚降默认 0.35，属于已知假设）；
  3. 按估计的符号速率在 k*(fs/Rs) 处插值采样；
  4. **分块定时相位搜索**：每 256 个符号在 ±0.3 符号内找一个让平均功率最大的相位。
     M1 的符号速率误差（约 1e-4）在几千个符号上会累积成定时漂移，不分块校正星座会被拖花；
  5. **全局残余载频去旋**：见 estimate_residual_cfo；
  6. 只取前 max_symbols 个符号（默认 1200），进一步限制残余漂移。

严格的最佳定时/载波跟踪是 M3 的工作；这里是让 M2 的累积量与星座特征可用的最小闭环。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from spxh.core.synth.pulse import matched_filter, rrc_taps

__all__ = [
    "recover_symbols",
    "constellation_features",
    "detect_cluster_centers",
    "estimate_residual_cfo",
]


def estimate_residual_cfo(
    points: np.ndarray,
    symbol_rate: float,
    orders=(2, 4, 8),
    max_offset_ratio: float = 0.03,
    nfft: Optional[int] = None,
):
    """从**符号采样点**上估计残余载频（M 次方 + 周期图峰值搜索）.

    为什么必须做：M1 的粗估误差约 1% Rs（1 kHz 符号速率下约 10 Hz），在 0.3 s 记录上
    会累积几十弧度相位，把星座转成一个圆环 —— 实测 QPSK 的 |C40| 会从理论 1.0 掉到 0.03、
    星座簇数从 4 涨到 11。C42 只依赖幅度所以不受影响，但 C40 需要相干积累。

    原理：PSK/QAM 星座具有 M 重旋转对称，s^M 的相位是 M 倍星座朝向加上 M 倍残余频偏
    造成的线性相位；因此 |FFT(s^M)| 的峰值位置就是 M*Δf。
    对 M∈{2,4,8} 各测一次，取峰最显著（peak/median 最大）的那个作为估计。

    比"分块相位 + 解缠"稳健得多：后者在残余 10 Hz、块长 64 符号时每块漂 187 度，
    远超 M 次方相位的 45 度解缠上限，必然解错。

    返回 (residual_cfo_hz, prominence)。
    """
    pts = np.asarray(points, dtype=np.complex128).reshape(-1)
    n = pts.size
    if n < 64 or symbol_rate <= 0:
        return 0.0, 0.0
    if nfft is None:
        nfft = int(2 ** int(np.ceil(np.log2(max(n * 8, 4096)))))
    window = np.hanning(n)
    freqs_all = np.fft.fftfreq(int(nfft), d=1.0 / float(symbol_rate))

    best_residual = 0.0
    best_prominence = 0.0
    for order in orders:
        order = int(order)
        if order not in (2, 4, 8):
            continue
        spectrum = np.abs(np.fft.fft((pts ** order) * window, int(nfft)))
        limit = float(order) * float(max_offset_ratio) * float(symbol_rate)
        mask = np.abs(freqs_all) <= limit
        if not np.any(mask):
            continue
        band = spectrum[mask]
        band_freqs = freqs_all[mask]
        median = float(np.median(band))
        if median <= 0:
            continue
        index = int(np.argmax(band))
        if 0 < index < band.size - 1:
            y0, y1, y2 = band[index - 1], band[index], band[index + 1]
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            peak_freq = float(band_freqs[index] + delta * (band_freqs[index + 1] - band_freqs[index]))
        else:
            peak_freq = float(band_freqs[index])
        prominence = float(band[index] / median)
        if prominence > best_prominence:
            best_prominence = prominence
            best_residual = peak_freq / float(order)
    return float(best_residual), best_prominence


def recover_symbols(
    signal,
    cfo_hz: float,
    symbol_rate: float,
    rolloff: float = 0.35,
    span_symbols: int = 16,
    max_symbols: int = 1200,
    timing_block: int = 256,
    timing_span: float = 0.3,
    timing_steps: int = 13,
    residual_correction: bool = True,
) -> np.ndarray:
    """盲恢复符号点（CFO 校正 + 匹配滤波 + 分块定时搜索 + 插值采样 + 残余载频去旋）."""
    if symbol_rate <= 0:
        raise ValueError("symbol_rate must be positive")
    samples = np.asarray(signal.samples).reshape(-1).astype(np.complex128)
    fs = float(signal.sample_rate)
    index = np.arange(samples.size, dtype=np.float64)
    corrected = samples * np.exp(-2j * np.pi * float(cfo_hz) * index / fs)

    sps_design = max(2, int(round(fs / symbol_rate)))
    taps = rrc_taps(sps_design, span_symbols=span_symbols, rolloff=rolloff)
    filtered = matched_filter(corrected, taps)
    delay = (taps.size - 1) / 2.0

    step = fs / float(symbol_rate)
    total = int(min(max_symbols, np.floor((filtered.size - 2 * delay - 1) / step)))
    if total < 16:
        raise ValueError("signal too short for blind constellation recovery")
    base_positions = 2.0 * delay + np.arange(total, dtype=np.float64) * step
    grid = np.arange(filtered.size, dtype=np.float64)
    real = filtered.real
    imag = filtered.imag

    offsets = np.linspace(-timing_span, timing_span, int(timing_steps))
    recovered = np.empty(total, dtype=np.complex128)
    block = max(16, int(timing_block))
    filled = total
    for start in range(0, total, block):
        stop = min(start + block, total)
        positions = base_positions[start:stop]
        best_offset = 0.0
        best_power = -1.0
        for offset in offsets:
            candidate = positions + offset
            candidate = candidate[candidate < filtered.size - 1]
            if candidate.size == 0:
                continue
            sample_real = np.interp(candidate, grid, real)
            sample_imag = np.interp(candidate, grid, imag)
            power = float(np.mean(sample_real ** 2 + sample_imag ** 2))
            if power > best_power:
                best_power = power
                best_offset = float(offset)
        candidate = base_positions[start:stop] + best_offset
        candidate = candidate[candidate < filtered.size - 1]
        recovered[start : start + candidate.size] = (
            np.interp(candidate, grid, real) + 1j * np.interp(candidate, grid, imag)
        )
        if candidate.size < stop - start:
            filled = start + candidate.size
            break
    recovered = recovered[:filled]

    if residual_correction:
        residual, _strength = estimate_residual_cfo(recovered, symbol_rate)
        if residual != 0.0:
            index = np.arange(recovered.size, dtype=np.float64)
            recovered = recovered * np.exp(-2j * np.pi * residual * index / float(symbol_rate))
    return recovered


def detect_cluster_centers(points: np.ndarray, bins: int = 48, threshold: float = 0.15, extent: float = 2.2) -> np.ndarray:
    """网格密度峰值法检测星座簇中心（不依赖 sklearn）."""
    pts = np.asarray(points, dtype=np.complex128).reshape(-1)
    if pts.size < 32:
        return np.zeros((0, 2))
    radius = float(np.sqrt(np.mean(np.abs(pts) ** 2)))
    if radius <= 0:
        return np.zeros((0, 2))
    normalized = pts / radius
    hist, edges = np.histogramdd(
        np.column_stack([normalized.real, normalized.imag]),
        bins=int(bins),
        range=[[-extent, extent], [-extent, extent]],
    )
    if hist.max() <= 0:
        return np.zeros((0, 2))
    density = hist / hist.max()
    padded = np.pad(density, 1, mode="constant")
    smoothed = np.zeros_like(density)
    for di in range(3):
        for dj in range(3):
            smoothed += padded[di : di + density.shape[0], dj : dj + density.shape[1]]
    smoothed /= 9.0
    centers: list[tuple[float, float]] = []
    nx, ny = smoothed.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            value = smoothed[i, j]
            if value < threshold:
                continue
            window = smoothed[max(0, i - 2) : i + 3, max(0, j - 2) : j + 3]
            if value >= window.max() - 1e-12:
                centers.append(
                    (
                        0.5 * (edges[0][i] + edges[0][i + 1]),
                        0.5 * (edges[1][j] + edges[1][j + 1]),
                    )
                )
    return np.asarray(centers, dtype=np.float64) if centers else np.zeros((0, 2))


def constellation_features(points: np.ndarray, symbol_rate: float) -> dict[str, float]:
    pts = np.asarray(points, dtype=np.complex128).reshape(-1)
    if pts.size < 32:
        return {
            "const_evm_percent": 0.0,
            "const_cluster_count": 0.0,
            "const_dispersion": 0.0,
            "const_radial_spread": 0.0,
            "const_phase_symmetry_order": 0.0,
            "const_phase_symmetry_strength": 0.0,
        }
    radius = float(np.sqrt(np.mean(np.abs(pts) ** 2)))
    normalized = pts / max(radius, 1e-300)
    centers = detect_cluster_centers(normalized)

    if centers.shape[0] > 0:
        distances = np.abs(normalized[:, None] - (centers[:, 0] + 1j * centers[:, 1])[None, :])
        nearest = np.min(distances, axis=1)
        dispersion = float(np.mean(nearest))
        evm = float(100.0 * np.sqrt(np.mean(nearest ** 2)))
        cluster_count = float(centers.shape[0])
    else:
        dispersion = 0.0
        evm = 0.0
        cluster_count = 0.0

    magnitudes = np.abs(normalized)
    radial_spread = float(np.std(magnitudes) / max(np.mean(magnitudes), 1e-300))
    phases = np.angle(normalized)
    best_order = 0.0
    best_strength = 0.0
    for order in (2, 4, 8):
        strength = float(np.abs(np.mean(np.exp(1j * order * phases))))
        if strength > best_strength:
            best_strength = strength
            best_order = float(order)

    return {
        "const_evm_percent": evm,
        "const_cluster_count": cluster_count,
        "const_dispersion": dispersion,
        "const_radial_spread": radial_spread,
        "const_phase_symmetry_order": best_order,
        "const_phase_symmetry_strength": best_strength,
    }
