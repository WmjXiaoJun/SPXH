"""谱相关（SSCA）：把"这是不是通信信号、符号速率多少、载频偏多少"变成一张看得见的图.

原理
----
通信信号是**循环平稳**的：它的自相关在时间上周期变化，因此在"频移 f - 循环频率 alpha"
的双频平面上有结构 —— 具体来说，成型脉冲的信号在 alpha = k * 符号速率 的竖线上有峰，
而载频偏移体现在 f = CFO 的横线上。白噪声没有这种结构（只在 alpha = 0 有能量）。
所以这张图既是特征来源，也是"为什么这么判"的直接证据。

实现
----
用条带谱相关算法（SSCA）：把信号切成 K 个 L 点帧，各自做加窗 FFT 得到 X_k(f)，则

    S_x^alpha(f) = (1/K) * sum_k X_k(f + alpha/2) * conj(X_k(f - alpha/2))

alpha 取 fs/L 的整数倍（频率分辨率），实现上就是频谱上左右各移 shift = alpha*L/fs 个 bin。
边缘 |m| < |shift| 的区域没有定义（频谱不够宽），置为无效（用 null 传出去），不假装有值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spxh.core.types import jsonable

__all__ = ["SSCAResult", "compute_ssca", "find_surface_peaks"]


@dataclass
class SSCAResult:
    sample_rate: float
    nfft: int
    freqs: np.ndarray              # (F,) 频移轴 Hz
    alphas: np.ndarray             # (A,) 循环频率轴 Hz
    surface_db: np.ndarray         # (A, F) dB，峰值归一到 0
    min_db: float
    max_db: float
    peaks: list[dict[str, Any]] = field(default_factory=list)
    symbol_rate_hz: float | None = None
    cfo_hz: float | None = None
    warnings: list[str] = field(default_factory=list)

    def alpha_profile(self) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(invalid="ignore"):
            profile = np.nanmax(self.surface_db, axis=1)
        return self.alphas, profile

    def freq_profile(self) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(invalid="ignore"):
            profile = np.nanmax(self.surface_db, axis=0)
        return self.freqs, profile

    def to_dict(self, max_points: int = 200) -> dict[str, Any]:
        alphas, alpha_profile = self.alpha_profile()
        freqs, freq_profile = self.freq_profile()
        alpha_idx, freq_idx, surface = _downsample_2d(self.surface_db, max_points)
        return jsonable(
            {
                "sample_rate": self.sample_rate,
                "nfft": int(self.nfft),
                "freqs": freqs.tolist(),
                "alphas": alphas.tolist(),
                "surface_db": [[None if not np.isfinite(v) else float(v) for v in row] for row in surface],
                "surface_freqs": freqs[freq_idx].tolist(),
                "surface_alphas": alphas[alpha_idx].tolist(),
                "min_db": float(self.min_db),
                "max_db": float(self.max_db),
                "peaks": self.peaks,
                "alpha_profile": {"alpha_hz": alphas.tolist(), "level_db": [None if not np.isfinite(v) else float(v) for v in alpha_profile]},
                "freq_profile": {"freq_hz": freqs.tolist(), "level_db": [None if not np.isfinite(v) else float(v) for v in freq_profile]},
                "features": {
                    "symbol_rate_hz": self.symbol_rate_hz,
                    "cfo_hz": self.cfo_hz,
                    "cfo_ridge_hz": self.cfo_hz,
                    "note": "alpha 轴峰 = 符号速率及其谐波（精确）；f 轴 alpha!=0 脊线位置 ~ 载频偏移（分辨率 fs/nfft，仅供解释，精确值看 M1 的参数估计）",
                },
                "warnings": list(self.warnings),
            }
        )


def _downsample_2d(surface: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = surface.shape
    row_stride = max(1, int(np.ceil(rows / float(max_points))))
    col_stride = max(1, int(np.ceil(cols / float(max_points))))
    row_idx = np.arange(0, rows, row_stride)
    col_idx = np.arange(0, cols, col_stride)
    return row_idx, col_idx, surface[np.ix_(row_idx, col_idx)]


def compute_ssca(
    samples: np.ndarray,
    sample_rate: float,
    nfft: int = 256,
    alpha_max: float | None = None,
    symbol_rate_hint: float | None = None,
    overlap: float = 0.5,
    min_db: float = -55.0,
) -> SSCAResult:
    """计算双频平面 S_x^alpha(f)（dB，峰值归一）。

    alpha_max 默认取 2 x 符号速率（或 2 x fs/8），足以覆盖符号速率及其一到二次谐波。
    """
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    fs = float(sample_rate)
    nfft = int(max(32, nfft))
    if x.size < nfft * 2:
        raise ValueError("记录太短：至少需要 2 个 {} 点帧".format(nfft))
    if alpha_max is None:
        base = float(symbol_rate_hint) if symbol_rate_hint else fs / 8.0
        alpha_max = 2.2 * base
    alpha_max = float(min(alpha_max, fs / 2.0))

    hop = max(1, int(round(nfft * (1.0 - float(overlap)))))
    starts = np.arange(0, x.size - nfft + 1, hop)
    if starts.size < 2:
        starts = np.array([0, max(0, x.size - nfft)])
    window = np.hanning(nfft)
    frames = np.stack([x[s : s + nfft] * window for s in starts])
    spectra = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)     # (K, L)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

    resolution = fs / nfft
    # 注意：SSCA 里两路频谱各偏 alpha/2，所以 bin 位移 shift 对应的循环频率是
    # alpha = 2 * shift * 分辨率 —— 写成 shift * 分辨率会把符号速率读出一半（踩过）。
    max_shift = int(np.floor((alpha_max / 2.0) / resolution))
    shifts = np.arange(-max_shift, max_shift + 1)
    alphas = 2.0 * shifts * resolution

    surface = np.full((shifts.size, nfft), np.nan, dtype=np.complex128)
    for i, shift in enumerate(shifts):
        if shift == 0:
            product = spectra * np.conj(spectra)
        else:
            left = np.roll(spectra, shift, axis=1)
            right = np.roll(spectra, -shift, axis=1)
            product = left * np.conj(right)
            # 循环移位会在两端引入"绕回来"的假点：把无效区标成 NaN
            product[:, : max(0, shift)] = np.nan
            product[:, nfft - max(0, shift) :] = np.nan
        with np.errstate(invalid="ignore", all="ignore"):
            valid = np.isfinite(product)
            counts = valid.sum(axis=0)
            totals = np.where(valid, product, 0).sum(axis=0)
            surface[i] = np.where(counts > 0, totals / np.maximum(counts, 1), complex(np.nan, np.nan))

    magnitude = np.abs(surface)
    peak = float(np.nanmax(magnitude)) if np.any(np.isfinite(magnitude)) else 0.0
    if peak <= 0:
        raise ValueError("谱相关平面全为零（信号可能为空或全零）")
    db = 20.0 * np.log10(np.maximum(magnitude, peak * 1e-12) / peak)
    db = np.maximum(db, min_db)

    result = SSCAResult(
        sample_rate=fs,
        nfft=nfft,
        freqs=freqs,
        alphas=alphas,
        surface_db=db,
        min_db=float(min_db),
        max_db=0.0,
    )
    result.peaks = find_surface_peaks(result, limit=8)

    # 特征：alpha 轴上（排除 0）最强的峰 = 符号速率及谐波；f 轴上最强峰 = 载频偏移
    alphas_all, alpha_profile = result.alpha_profile()
    valid_alpha = np.isfinite(alpha_profile) & (alphas_all > resolution * 0.5)
    if np.any(valid_alpha):
        best = int(np.argmax(np.where(valid_alpha, alpha_profile, -np.inf)))
        result.symbol_rate_hz = float(alphas_all[best])
    # CFO：用 alpha != 0 的脊线位置（= 载频）而不是 alpha=0 那行（= 普通功率谱）。
    # RRC 成型谱是平顶的，功率谱峰值/重心都会被平坦区带偏（实测偏 30~60 Hz）；
    # 而循环谱脊线在 f 轴上正好落在载频上。分辨率就是 fs/nfft，所以它只作
    # **解释**用 —— 精确的载频估计仍然由 M1 给（相对误差 < 1% Rs）。
    freqs_all, _freq_profile = result.freq_profile()
    ridge_row = None
    if result.symbol_rate_hz is not None:
        ridge_row = int(np.argmin(np.abs(alphas_all - abs(result.symbol_rate_hz))))
    if ridge_row is not None:
        profile = db[ridge_row]
        if np.any(np.isfinite(profile)):
            peak_bin = int(np.nanargmax(profile))
            width = 4
            lo = max(0, peak_bin - width)
            hi = min(profile.size, peak_bin + width + 1)
            local = np.clip(profile[lo:hi], -120.0, 0.0)
            weights = np.power(10.0, np.where(np.isfinite(local), local, -120.0) / 20.0)
            if np.sum(weights) > 0:
                result.cfo_hz = float(np.sum(freqs_all[lo:hi] * weights) / np.sum(weights))
    result.warnings = list(_ssca_warnings(result, sample_rate=fs, frames=int(starts.size)))
    return result


def _ssca_warnings(result: SSCAResult, sample_rate: float, frames: int) -> list[str]:
    warnings: list[str] = []
    if frames < 8:
        warnings.append("帧数只有 {}：循环谱估计会很噪，建议加长记录".format(frames))
    if result.symbol_rate_hz is not None and result.symbol_rate_hz >= sample_rate / 2.0 - 1e-9:
        warnings.append("循环频率接近奈奎斯特，符号速率估计可能落到别名上")
    return warnings


def find_surface_peaks(result: SSCAResult, limit: int = 8, neighbourhood: int = 2) -> list[dict[str, Any]]:
    """在双频平面上找显著局部峰（带非极大抑制）."""
    surface = result.surface_db
    rows, cols = surface.shape
    peaks: list[tuple[float, int, int]] = []
    for i in range(rows):
        lo_r, hi_r = max(0, i - neighbourhood), min(rows, i + neighbourhood + 1)
        for j in range(cols):
            value = surface[i, j]
            if not np.isfinite(value):
                continue
            lo_c, hi_c = max(0, j - neighbourhood), min(cols, j + neighbourhood + 1)
            block = surface[lo_r:hi_r, lo_c:hi_c]
            if not np.any(np.isfinite(block)):
                continue
            if value < np.nanmax(block) - 1e-9:
                continue
            peaks.append((float(value), i, j))
    peaks.sort(key=lambda item: -item[0])
    out: list[dict[str, Any]] = []
    for value, i, j in peaks:
        alpha = float(result.alphas[i])
        freq = float(result.freqs[j])
        if any(abs(alpha - p["alpha_hz"]) < 2 * (result.alphas[1] - result.alphas[0]) for p in out):
            continue
        out.append({"alpha_hz": alpha, "freq_hz": freq, "level_db": value})
        if len(out) >= limit:
            break
    return out
