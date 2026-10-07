"""物理规则回退（确定性、可复现、无需训练数据）.

当机器学习置信度不足或样本被判为分布外时，用这套规则给出兜底结论。
它不是"更弱的模型"，而是另一种保证：结论只依赖物理量（累积量、径向结构、
频谱形状、包络统计、瞬时频率直方图），因此**可复现**且**失效可解释**。

关键设计：**多证据投票 + 证据不足就拒绝**。
低信噪比下几乎所有物理量都会退化（实测 0 dB 时包络变异系数对六类调制都是 0.52、
占用带宽比都落进 1.0~2.2），此时任何"硬判"都是在编答案。规则只在证据足够时给出结论，
否则返回 None，由门控标记为 reject —— 这与方案里"宁可说不知道"的原则一致。

用到的物理事实：

* 恒模性：FSK 是真正的恒模（包络变异系数在高信噪比下 ~0.19，只来自噪声），
  而 RRC 成型的 PSK/QAM 包络起伏大（0.34~0.44）；
* 频谱形状：RRC 成型信号的滚降估计约 0.3；FSK 不是 RRC 成型（估计贴近 1.0），
  占用带宽/符号速率比约 2.9（2FSK）与 3.4（4FSK），而 PSK/QAM 只有 1.2~1.5；
* 旋转对称阶数：BPSK=2、QPSK/16QAM=4、8PSK=8；
* 累积量：|C40| 理论值 BPSK=2、QPSK=1、16QAM=0.68、8PSK/FSK≈0；
* 瞬时频率：FSK 有 2 个（2FSK）或 4 个（4FSK）离散音调。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = [
    "PhysicalDecision",
    "physical_classify",
    "count_if_tones",
    "instantaneous_frequency",
    "envelope_cv",
]


@dataclass
class PhysicalDecision:
    modulation: Optional[str]
    confidence: float
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modulation": self.modulation,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "scores": {k: round(float(v), 4) for k, v in self.scores.items()},
            "detail": self.detail,
        }


def instantaneous_frequency(signal, cfo_hz: float = 0.0) -> np.ndarray:
    """瞬时频率（Hz，已去掉估计的载频）."""
    samples = np.asarray(signal.samples).reshape(-1).astype(np.complex128)
    fs = float(signal.sample_rate)
    index = np.arange(samples.size, dtype=np.float64)
    corrected = samples * np.exp(-2j * np.pi * float(cfo_hz) * index / fs)
    phase = np.unwrap(np.angle(corrected))
    return np.diff(phase) * fs / (2.0 * np.pi)


def envelope_cv(samples) -> float:
    """包络变异系数 std(|x|)/mean(|x|)（恒模信号接近 0）."""
    envelope = np.abs(np.asarray(samples).reshape(-1))
    if envelope.size == 0:
        return 0.0
    mean = float(np.mean(envelope))
    return float(np.std(envelope) / mean) if mean > 0 else 0.0


def count_if_tones(signal, cfo_hz: float, symbol_rate: float, max_tones: int = 6) -> int:
    """瞬时频率直方图的峰个数（FSK 的"音调数"）.

    阈值取 0.5、峰间最小间隔取 0.35*Rs：2FSK(h=1) 的音调间隔是 Rs、
    4FSK(h=0.5) 是 0.5*Rs，都能分开；而 PSK/QAM 的瞬时频率只是过渡期的一团，
    不会被误数成多音。
    """
    if symbol_rate <= 0:
        return 0
    inst = instantaneous_frequency(signal, cfo_hz=cfo_hz)
    if inst.size < 64:
        return 0
    limit = 1.2 * symbol_rate
    clipped = inst[np.abs(inst) <= limit]
    if clipped.size < 64:
        return 0
    hist, edges = np.histogram(clipped, bins=64, range=(-limit, limit))
    if hist.max() <= 0:
        return 0
    density = hist.astype(np.float64) / hist.max()
    smoothed = np.convolve(density, np.array([1.0, 2.0, 1.0]) / 4.0, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    minimum_separation = 0.35 * symbol_rate
    peaks: list[float] = []
    for index in range(1, smoothed.size - 1):
        if smoothed[index] < 0.5:
            continue
        if smoothed[index] < smoothed[index - 1] or smoothed[index] <= smoothed[index + 1]:
            continue
        frequency = float(centers[index])
        if all(abs(frequency - existing) > minimum_separation for existing in peaks):
            peaks.append(frequency)
    return int(min(len(peaks), int(max_tones)))


def _gauss(value: float, center: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return float(np.exp(-0.5 * ((value - center) / sigma) ** 2))


def physical_classify(
    signal,
    features,
    cfo_hz: Optional[float] = None,
    symbol_rate: Optional[float] = None,
) -> PhysicalDecision:
    """基于物理量的确定性判别；证据不足时返回 modulation=None."""
    evidence = features.evidence
    cfo = float(evidence.get("cfo_hz", 0.0) if cfo_hz is None else cfo_hz)
    rs = float(evidence.get("symbol_rate", 0.0) if symbol_rate is None else symbol_rate)
    if rs <= 0:
        return PhysicalDecision(None, 0.0, "符号速率无效，物理规则无法工作", {}, {})

    c40 = float(features["cum_c40"])
    c42 = float(features["cum_c42"])
    radial = float(features["const_radial_spread"])
    clusters = float(features["const_cluster_count"])
    rolloff = float(features["spec_rolloff_est"])
    bandwidth_ratio = float(features["spec_bw_over_rs"])
    order = float(features["const_phase_symmetry_order"])
    env_cv = float(features["tf_env_cv"]) if "tf_env_cv" in features.names else envelope_cv(signal.samples)
    centroid_std = float(features["tf_centroid_std"])

    tones = count_if_tones(signal, cfo, rs) if (rolloff > 0.6 or bandwidth_ratio > 2.0) else 0

    # 多证据投票：至少两条成立才判 FSK 家族
    fsk_votes = 0
    fsk_votes += 1 if rolloff > 0.75 else 0
    fsk_votes += 1 if bandwidth_ratio > 2.0 else 0
    fsk_votes += 1 if env_cv < 0.32 else 0
    fsk_votes += 1 if centroid_std > 0.10 else 0

    detail: dict[str, Any] = {
        "cum_c40": round(c40, 4),
        "cum_c42": round(c42, 4),
        "radial_spread": round(radial, 4),
        "cluster_count": round(clusters, 1),
        "rolloff_est": round(rolloff, 4),
        "bandwidth_over_rs": round(bandwidth_ratio, 4),
        "envelope_cv": round(env_cv, 4),
        "phase_symmetry_order": round(order, 1),
        "if_tones": tones,
        "fsk_votes": fsk_votes,
    }

    scores: dict[str, float] = {}
    if fsk_votes >= 3:
        # 2FSK(h=1) 与 4FSK(h=0.5) 的两个独立证据：
        #   占用带宽/符号速率比  20 dB 实测 2.87 vs 3.40（10 dB 时 2.62 vs 3.73，仍然可分）
        #   瞬时频率音调数       2 vs 4（低信噪比下会退化，所以只作为第二票）
        ratio_2fsk = _gauss(bandwidth_ratio, 2.9, 0.45)
        ratio_4fsk = _gauss(bandwidth_ratio, 3.4, 0.55)
        tone_2fsk = 1.0 if tones == 2 else (0.4 if tones == 0 else 0.15)
        tone_4fsk = 1.0 if tones >= 4 else (0.5 if tones == 3 else 0.15)
        base = 0.35
        scores["2fsk"] = base + 0.9 * ratio_2fsk + 0.6 * tone_2fsk
        scores["4fsk"] = base + 0.9 * ratio_4fsk + 0.6 * tone_4fsk
        reason = "FSK 家族（{} 条证据：滚降 {:.2f} / 带宽比 {:.2f} / 包络变异 {:.2f} / 频率游走 {:.2f}）；带宽比判 2FSK vs 4FSK，瞬时频率检测到 {} 个音调".format(
            fsk_votes, rolloff, bandwidth_ratio, env_cv, centroid_std, tones
        )
    else:
        constant_modulus = radial < 0.20 or env_cv < 0.30
        modulation_shape = 1.0 if constant_modulus else 0.35
        scores["bpsk"] = _gauss(c40, 2.0, 0.45) * modulation_shape * (1.0 if order == 2 else 0.5)
        scores["qpsk"] = _gauss(c40, 1.0, 0.35) * modulation_shape * (1.0 if order == 4 else 0.6)
        scores["8psk"] = _gauss(c40, 0.0, 0.30) * _gauss(clusters, 8.0, 3.0) * modulation_shape
        scores["16qam"] = _gauss(c40, 0.68, 0.30) * _gauss(clusters, 16.0, 6.0) * (1.0 if radial > 0.20 else 0.4)
        reason = "非 FSK（{} 条证据）：|C40|={:.2f}、径向离散度={:.2f}、星座簇数={:.0f}、对称阶数={:.0f}、包络变异={:.2f}".format(
            fsk_votes, c40, radial, clusters, order, env_cv
        )

    total = float(sum(scores.values()))
    if total <= 0:
        return PhysicalDecision(None, 0.0, "所有物理打分均为 0", scores, detail)
    normalized = {k: v / total for k, v in scores.items()}
    best = max(normalized, key=lambda key: normalized[key])
    confidence = float(normalized[best])
    # 0.55：实测把门限从 0.55 提到 0.70 只会让规则"少答"（10 dB 命中率 0.556 -> 0.417），
    # 精度几乎不涨（回退样本准确率 0.19 -> 0.22）。物理量本身在低信噪比下不可分，
    # 靠调门限救不回来，所以选覆盖面更好的 0.55，把"不确定"交给 gate=reject 表达。
    if confidence < 0.55:
        return PhysicalDecision(
            None,
            confidence,
            reason + "；最高分 {:.2f} 未过门限，规则拒绝给结论".format(confidence),
            normalized,
            detail,
        )
    return PhysicalDecision(best, confidence, reason, normalized, detail)
