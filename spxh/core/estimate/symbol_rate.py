"""符号速率估计.

两种互补的循环特征 + 基于"谱线谐波结构"的基频判定：

* 循环自相关（cyclic）：z[n] = mean_tau x[n+tau] conj(x[n])，谱线出现在 alpha = k*Rs。
  对六类调制都有效 —— FSK 是恒模、没有包络起伏，只有这项能给出强谱线。
* 包络（envelope）：z[n] = |x[n]|^2 - mean，对 PSK/QAM 的成型信号很强，
  对恒模的 FSK 几乎无效（实测 ~10 dB 线，而循环法有 22~42 dB）。

基频判定（关键）：先在剖面上检测**显著谱线**（局部极大且高出中位水平 min_prominence_db），
再用"谁的整数倍能解释最多谱线"来判定基频（score = sum 1/k），最后用最小二乘
f0 = sum(k_i * p_i) / sum(k_i^2) 把若干次谐波的峰值位置合并起来提高精度。

这里**不能**用"谐波 dB 求和 + 网格搜索"：噪声底的 dB 也在 0 附近，低频候选因为能塞进
更多谐波项反而得分更高，会把符号速率一路拉到搜索下界（实测 4FSK 直接错 95%）。
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d

from spxh.core.estimate.types import Estimate
from spxh.core.types import Signal

__all__ = [
    "cyclic_profile",
    "envelope_profile",
    "harmonic_score",
    "line_prominence_db",
    "detect_peaks",
    "select_fundamental",
    "estimate_symbol_rate",
]

DEFAULT_TAUS: tuple[int, ...] = (1, 2, 4, 8)
MAX_PROFILE_FFT = 1 << 18


def _fft_profile(
    z: np.ndarray,
    fs: float,
    nfft: Optional[int] = None,
    complex_input: bool = False,
    max_segments: int = 8,
    min_segment: int = 1024,
    overlap: float = 0.5,
):
    """分段平均的加窗零填充 FFT，得到 (alpha, |Z|^2)，只保留 alpha >= 0.

    为什么要分段：单段循环周期图的方差极大，短记录（几百个符号）下低频谱底里会冒出
    比真实符号速率线还强的伪峰（实测 0.33 s 的 QPSK 直接把 1000 Hz 判成 152 Hz）。
    按 50% 重叠分成最多 8 段平均 |Z|^2，可把伪峰压低一个量级，代价是频率分辨率
    退化为 fs/seg_len（默认每段至少 1024 点，仍远细于符号速率）。
    """
    z = np.asarray(z)
    n = z.size
    z = z - z.mean()
    if nfft is None:
        nfft = int(2 ** int(np.ceil(np.log2(max(n * 4, 1024)))))
    nfft = int(min(max(nfft, 1024), MAX_PROFILE_FFT))
    if max_segments > 1 and n > min_segment:
        segment = int(min(max(min_segment, n // int(max_segments)), n))
    else:
        segment = n
    hop = max(1, int(segment * (1.0 - float(overlap))))
    starts = list(range(0, max(n - segment + 1, 1), hop))[: int(max_segments)] or [0]
    window = np.hanning(segment)
    spectrum_length = nfft if complex_input else (nfft // 2 + 1)
    accumulator = np.zeros(spectrum_length, dtype=np.float64)
    for start in starts:
        piece = z[start : start + segment]
        if piece.size < segment:
            piece = z[-segment:]
        piece = piece - piece.mean()
        windowed = piece * window
        if complex_input:
            spectrum = np.fft.fft(windowed, nfft)
        else:
            spectrum = np.fft.rfft(np.real(windowed), nfft)
        accumulator += np.abs(spectrum) ** 2
    freqs = np.fft.fftfreq(nfft, 1.0 / fs) if complex_input else np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = freqs >= 0.0
    return np.asarray(freqs[keep], dtype=np.float64), (accumulator / float(len(starts)))[keep]


def cyclic_profile(x, fs: float, taus: Sequence[int] = DEFAULT_TAUS, nfft: Optional[int] = None):
    """循环自相关剖面：alpha 轴上的谱线强度."""
    samples = np.asarray(x).reshape(-1)
    taus = tuple(int(t) for t in taus if int(t) > 0 and int(t) < samples.size // 4)
    if not taus:
        raise ValueError("taus must contain at least one lag smaller than a quarter of the record")
    length = samples.size - max(taus)
    accumulator = np.zeros(length, dtype=np.complex128)
    for tau in taus:
        accumulator += samples[tau : tau + length] * np.conj(samples[:length])
    accumulator /= float(len(taus))
    return _fft_profile(accumulator, fs, nfft=nfft, complex_input=True)


def envelope_profile(x, fs: float, nfft: Optional[int] = None):
    """包络（|x|^2）剖面：alpha 轴上的谱线强度."""
    samples = np.asarray(x).reshape(-1)
    return _fft_profile(np.abs(samples) ** 2, fs, nfft=nfft, complex_input=False)


def _to_db(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    reference = float(np.median(profile)) if profile.size else 0.0
    if reference <= 0:
        reference = float(np.max(profile)) if profile.size and np.max(profile) > 0 else 1.0
    return 10.0 * np.log10(np.maximum(profile, 1e-300) / max(reference, 1e-300))


def _parabolic_db(alpha: np.ndarray, db: np.ndarray, index: int) -> float:
    if index <= 0 or index >= db.size - 1:
        return float(alpha[index])
    y0, y1, y2 = db[index - 1], db[index], db[index + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(alpha[index])
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return float(alpha[index] + delta * (alpha[index + 1] - alpha[index]))


def detect_peaks(
    alpha: np.ndarray,
    profile: np.ndarray,
    fmin: float,
    fmax: float,
    min_prominence_db: float = 8.0,
    max_peaks: int = 16,
    baseline_window_hz: float = 200.0,
) -> list[dict[str, float]]:
    """检测显著谱线：局部极大 + 高出**局部基线** min_prominence_db，峰值位置做抛物线插值.

    关键点：基线必须取**局部**（滑动平均）而不是全局中位数。循环剖面的低频谱底带有
    明显的 1/f 抬升，用全局中位数做基线时，几十~几百 Hz 的伪峰会被算成十几 dB 的"强线"，
    真实符号速率线反而被淹没（实测 0.33 s 记录的 QPSK 会把 1000 Hz 判成 152 Hz）。
    减掉滑动平均基线等价于在 alpha 轴上做一次白化。
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    db = _to_db(profile)
    if alpha.size > 3:
        df = float(alpha[1] - alpha[0])
        window = int(max(3, round(float(baseline_window_hz) / max(df, 1e-9))))
        if window % 2 == 0:
            window += 1
        window = int(min(window, db.size if db.size % 2 == 1 else db.size - 1))
        if window >= 3:
            db = db - uniform_filter1d(db, size=window, mode="nearest")
    mask = (alpha >= fmin) & (alpha <= fmax)
    if not np.any(mask):
        return []
    middle = db[1:-1]
    is_local = (middle >= db[:-2]) & (middle >= db[2:])
    indices = np.flatnonzero(is_local) + 1
    indices = indices[mask[indices]]
    if indices.size == 0:
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        indices = np.asarray([candidates[int(np.argmax(db[candidates]))]])
    order = np.argsort(-db[indices])
    peaks: list[dict[str, float]] = []
    for position in order:
        i = int(indices[position])
        prominence = float(db[i])
        if prominence < min_prominence_db:
            continue
        peaks.append(
            {
                "frequency": _parabolic_db(alpha, db, i),
                "alpha": float(alpha[i]),
                "level_db": float(db[i]),
                "prominence_db": prominence,
            }
        )
        if len(peaks) >= max_peaks:
            break
    peaks.sort(key=lambda p: p["frequency"])
    return peaks


def select_fundamental(
    peaks: Sequence[dict[str, float]],
    harmonics: int = 4,
    min_tolerance_hz: float = 15.0,
    tolerance_ratio: float = 0.01,
    level_drop_db: float = 18.0,
    candidate_drop_db: float = 8.0,
) -> dict[str, Any]:
    """判定基频：以**最强谱线**为锚，只考察它的整数分之一候选。

    为什么不遍历所有候选峰：谱线以下的低频谱底里存在大量高出中位水平 20 dB 的
    伪峰（窗口泄漏/循环统计的 1/f 成分），遍历式"谁能解释最多峰"会把基频
    判到 60~100 Hz 这类伪峰上（实测符号速率直接错 90%）。

    锚定法：真实基频 f0 必然满足 anchor = k*f0（anchor 是某个谐波），所以候选只有
    anchor/k；再用"能否解释更多谱线"选 k，得分相同时取**更大的频率**（少切分优先）。
    """
    if not peaks:
        return {"frequency": float("nan"), "score": 0.0, "matched": [], "reason": "no peaks"}
    anchor = max(peaks, key=lambda p: float(p["level_db"]))
    level_floor = float(anchor["level_db"]) - float(level_drop_db)
    # 候选门限要更严：真实基频应当本身就是一条**强**谱线；
    # 只用 18 dB 的宽松门限会让低频谱底里的伪峰（比锚低 11~13 dB）被当成 anchor/3、anchor/4。
    candidate_floor = float(anchor["level_db"]) - float(candidate_drop_db)

    candidates: list[float] = []
    for k in range(1, int(harmonics) + 1):
        target = float(anchor["frequency"]) / float(k)
        tolerance = max(float(min_tolerance_hz), float(tolerance_ratio) * target)
        nearest = min(peaks, key=lambda p: abs(float(p["frequency"]) - target))
        if abs(float(nearest["frequency"]) - target) <= tolerance and float(nearest["level_db"]) >= candidate_floor:
            candidates.append(float(nearest["frequency"]))
    if not candidates:
        candidates = [float(anchor["frequency"])]

    best: dict[str, Any] = {"frequency": float("nan"), "score": -1.0, "matched": []}
    for f0 in candidates:
        if f0 <= 0:
            continue
        matched = [(1, f0)]
        score = 1.0
        for m in range(2, int(harmonics) + 1):
            target = m * f0
            tolerance = max(float(min_tolerance_hz), float(tolerance_ratio) * target)
            near = [
                p
                for p in peaks
                if abs(float(p["frequency"]) - target) <= tolerance and float(p["level_db"]) >= level_floor
            ]
            if not near:
                # 谐波必须**连续**出现：否则"anchor/k"这种切分只要碰到锚本身
                # （m=k）就能白拿分，会把基频判到 anchor/3、anchor/4。
                break
            nearest = min(near, key=lambda p: abs(float(p["frequency"]) - target))
            matched.append((m, float(nearest["frequency"])))
            score += 1.0 / m
        entry = {"frequency": f0, "score": score, "matched": matched}
        if score > best["score"] + 1e-9 or (
            abs(score - best["score"]) <= 1e-9 and f0 > best.get("frequency", float("-inf"))
        ):
            best = entry
    matched = best.get("matched", [])
    if len(matched) >= 2:
        ks = np.asarray([m[0] for m in matched], dtype=np.float64)
        fs_ = np.asarray([m[1] for m in matched], dtype=np.float64)
        refined = float(np.sum(ks * fs_) / np.sum(ks * ks))
    else:
        refined = float(best["frequency"])
    best["frequency_seed"] = float(best["frequency"])
    best["frequency"] = refined
    best["num_harmonics"] = len(matched)
    best["anchor_hz"] = float(anchor["frequency"])
    best["anchor_level_db"] = float(anchor["level_db"])
    best["level_floor_db"] = level_floor
    best["candidate_floor_db"] = candidate_floor
    return best


def harmonic_score(alpha: np.ndarray, profile: np.ndarray, f0: float, harmonics: int = 4) -> tuple[float, int]:
    """谐波加权和（对显著谱线计数，供外部校验用）：score = sum_k [有谱线] / k."""
    peaks = detect_peaks(alpha, profile, fmin=max(f0 * 0.5, 1e-6), fmax=float(np.max(alpha)), min_prominence_db=6.0)
    if not peaks:
        return 0.0, 0
    selected = select_fundamental(peaks, harmonics=harmonics)
    # 仅当 selected 的基频与询问频率一致时返回其得分
    if abs(selected.get("frequency_seed", -1.0) - f0) > max(15.0, 0.01 * f0):
        return 0.0, 0
    return float(selected["score"]), int(selected["num_harmonics"])


def line_prominence_db(alpha: np.ndarray, profile: np.ndarray, f0: float, tol_hz: float = 25.0) -> float:
    """给定频率处的谱线相对局部中位水平的 dB 强度."""
    db = _to_db(profile)
    alpha = np.asarray(alpha, dtype=np.float64)
    band = np.abs(alpha - float(f0)) <= tol_hz
    if not np.any(band):
        return float("-inf")
    peak = float(np.max(db[band]))
    outside = (np.abs(alpha - float(f0)) > 3.0 * tol_hz) & (np.abs(alpha - float(f0)) < 8.0 * tol_hz)
    baseline = float(np.median(db[outside])) if np.any(outside) else float(np.median(db))
    return peak - baseline


def estimate_symbol_rate(
    signal: Optional[Signal] = None,
    samples: Optional[np.ndarray] = None,
    sample_rate: Optional[float] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    harmonics: int = 4,
    taus: Sequence[int] = DEFAULT_TAUS,
    nfft: Optional[int] = None,
    min_prominence_db: float = 8.0,
    bandwidth_hint: Optional[float] = None,
    bandwidth_floor_ratio: float = 0.25,
) -> Estimate:
    """估计符号速率（Hz），返回带置信度与证据的估计值.

    物理约束：占用带宽 B 与符号速率同量级（PSK/QAM 约 B/(1+alpha)，MFSK 约 B/((M-1)h+2)），
    因此 Rs 不可能远小于 B。用 bandwidth_floor_ratio * B 作为搜索下界可以排除循环剖面
    低频谱底的伪峰（实测 4FSK 在 10 dB 时会把 977 Hz 判成 149 Hz）。
    bandwidth_hint 为 None 时内部自己估一个带宽（多花一次 Welch PSD）。
    """
    if signal is not None:
        x = np.asarray(signal.samples).reshape(-1)
        fs = float(signal.sample_rate)
    else:
        if samples is None or sample_rate is None:
            raise ValueError("either signal, or samples with sample_rate is required")
        x = np.asarray(samples).reshape(-1)
        fs = float(sample_rate)
    if x.size < 256:
        raise ValueError("need at least 256 samples for symbol-rate estimation")

    lo = float(fmin) if fmin is not None else fs / 200.0
    hi = float(fmax) if fmax is not None else 0.45 * fs

    bandwidth = None
    try:
        if bandwidth_hint is not None:
            bandwidth = float(bandwidth_hint)
        else:
            from spxh.core.dsp.spectrum import welch_psd

            psd = welch_psd(np.asarray(x).reshape(-1), sample_rate=fs, nperseg=int(min(1024, x.size)))
            bandwidth = float(psd.occupied_band()["bandwidth"])
    except Exception:  # noqa: BLE001 - 带宽提示只是先验约束，失败就退回默认区间
        bandwidth = None
    bandwidth_floor = float(bandwidth_floor_ratio) * bandwidth if bandwidth else 0.0
    if bandwidth_floor > lo:
        lo = bandwidth_floor
    if not 0 < lo < hi:
        raise ValueError("invalid frequency range for symbol-rate search")

    alpha_c, prof_c = cyclic_profile(x, fs, taus=taus, nfft=nfft)
    alpha_e, prof_e = envelope_profile(x, fs, nfft=nfft)

    peaks_c = detect_peaks(alpha_c, prof_c, lo, hi, min_prominence_db=min_prominence_db)
    peaks_e = detect_peaks(alpha_e, prof_e, lo, hi, min_prominence_db=min_prominence_db)
    best_cyclic = select_fundamental(peaks_c, harmonics=harmonics) if peaks_c else None
    best_envelope = select_fundamental(peaks_e, harmonics=harmonics) if peaks_e else None

    def _pack(best, peaks):
        if best is None:
            return None
        level = float(np.max([p["prominence_db"] for p in peaks])) if peaks else 0.0
        return {
            "frequency": float(best["frequency"]),
            "frequency_seed": float(best["frequency_seed"]),
            "score": float(best["score"]),
            "num_harmonics": int(best["num_harmonics"]),
            "best_prominence_db": level,
            "peaks": [{"frequency": round(p["frequency"], 4), "prominence_db": round(p["prominence_db"], 3)} for p in peaks[:8]],
        }

    cyc = _pack(best_cyclic, peaks_c)
    env = _pack(best_envelope, peaks_e)

    if cyc is None and env is None:
        subset = alpha_c[(alpha_c >= lo) & (alpha_c <= hi)]
        db_c = _to_db(prof_c)[(alpha_c >= lo) & (alpha_c <= hi)]
        return Estimate(
            name="symbol_rate",
            value=float(subset[int(np.argmax(db_c))]) if subset.size else 0.0,
            unit="Hz",
            confidence=0.0,
            method="no-prominent-line",
            evidence={"fmin": lo, "fmax": hi, "reason": "no spectral line above threshold"},
        )

    # 只有一路给出谱线时不算"分歧"，但也不能算交叉验证通过：取 0.6 的部分一致性。
    # FSK 恒模时包络法本来就没有线，这是预期行为而不是矛盾。
    if cyc is None:
        chosen, method, agreement = env, "envelope", 0.6
    elif env is None:
        chosen, method, agreement = cyc, "cyclic", 0.6
    else:
        rel_diff = abs(cyc["frequency"] - env["frequency"]) / max(cyc["frequency"], 1e-12)
        agreement = float(np.clip(1.0 - rel_diff / 0.01, 0.0, 1.0))
        if rel_diff <= 0.005:
            chosen, method = cyc, "cyclic+envelope(agree)"
        else:
            chosen = cyc if cyc["best_prominence_db"] >= env["best_prominence_db"] - 3.0 else env
            method = "cyclic+envelope(disagree)"

    prominence_score = float(np.clip((chosen["best_prominence_db"] - 6.0) / 24.0, 0.0, 1.0))
    # 两法分歧时不给谐波加分，并且把显著度按最低一致性打折 —— 分歧本身就是"不可信"的信号
    harmonic_bonus = 0.15 if (chosen["num_harmonics"] >= 2 and agreement > 0.0) else 0.0
    confidence = float(np.clip(0.6 * prominence_score * max(agreement, 0.15) + 0.4 * agreement + harmonic_bonus, 0.0, 1.0))

    return Estimate(
        name="symbol_rate",
        value=float(chosen["frequency"]),
        unit="Hz",
        confidence=confidence,
        method=method,
        evidence={
            "cyclic": cyc,
            "envelope": env,
            "agreement": agreement,
            "fmin": lo,
            "fmax": hi,
            "bandwidth_hz": bandwidth,
            "bandwidth_floor_hz": bandwidth_floor,
            "taus": list(taus),
            "harmonics": int(harmonics),
        },
    )
