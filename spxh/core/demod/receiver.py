"""M3 接收链：定时同步 -> 载波跟踪 -> 判决 -> 软判决 LLR -> 误码对账.

与 M2 里"为算特征临时凑的"盲恢复相比，这条链是真正的接收机结构：

    AGC -> RRC 匹配滤波 -> Gardner 定时环（Farrow 插值）->
    判决导向载波环（二阶 PLL）-> 星座判决 / FSK 音调检测 -> max-log LLR

相位模糊（BPSK 180 度、QPSK/16QAM 90 度、8PSK 45 度）是**调制本身的性质**，真实接收机靠
已知前导（本项目的 Barker-13 帧头，M5）或差分编码解决。这里在"与真值对账"时提供
模糊 + 符号滑动联合搜索，仅用于评估，不掩盖问题：结果里会报出用的是哪个旋转。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from spxh.core.demod.filters import agc_normalize
from spxh.core.demod.fsk import fsk_demodulate, fsk_llr
from spxh.core.demod.llr import max_log_llr, rotation_symmetry_order, symbol_noise_variance
from spxh.core.demod.sync import CarrierRecovery, LoopConfig, TimingRecovery
from spxh.core.estimate import AnalyzeResult, analyze_signal
from spxh.core.synth import mapping
from spxh.core.synth.pulse import matched_filter, rrc_taps
from spxh.core.types import MOD_BITS_PER_SYMBOL, GroundTruth, Signal, jsonable

__all__ = ["DemodResult", "demodulate", "evaluate_against_truth", "DEFAULT_ROLLOFF"]

DEFAULT_ROLLOFF = 0.35
DEFAULT_SPAN_SYMBOLS = 16


@dataclass
class DemodResult:
    modulation: str
    modulation_source: str
    symbol_rate: float
    cfo_hz: float
    snr_db: float
    symbols: np.ndarray
    symbol_indices: np.ndarray
    bits: np.ndarray
    llr: np.ndarray
    evm_percent: Optional[float] = None
    timing_metric: float = 0.0
    carrier_metric: float = 0.0
    timing_errors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phase_errors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    freq_estimates: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ber: Optional[dict[str, Any]] = None
    #: 每个符号对应的采样位置（输入信号的索引），用于把信号侧突发区间映射到符号/比特
    symbol_samples: np.ndarray = field(default_factory=lambda: np.zeros(0))
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def num_symbols(self) -> int:
        return int(self.symbols.size)

    @property
    def num_bits(self) -> int:
        return int(self.bits.size)

    def trace_arrays(self, points: int = 400) -> dict[str, list[float]]:
        """把三条环路观测量降采样到 <= points 点，供前端画图."""
        timing = np.asarray(self.timing_errors, dtype=np.float64)
        phase = np.asarray(self.phase_errors, dtype=np.float64)
        freq = np.asarray(self.freq_estimates, dtype=np.float64)
        count = max(timing.size, phase.size, freq.size)
        if count == 0:
            return {"index": [], "timing_error": [], "phase_error_rad": [], "freq_estimate_hz": []}
        stride = max(1, int(np.ceil(count / max(1, int(points)))))
        index = np.arange(0, count, stride, dtype=np.int64)
        return {
            "index": index.tolist(),
            "timing_error": _pick(timing, index).tolist(),
            "phase_error_rad": _pick(phase, index).tolist(),
            "freq_estimate_hz": _pick(freq, index).tolist(),
        }

    def to_dict(self, max_symbols: int = 2000, trace_points: int = 400, bit_head: int = 128) -> dict[str, Any]:
        traces = self.trace_arrays(points=trace_points)
        symbols = self.symbols[: int(max_symbols)]
        bits = self.bits[: int(bit_head)]
        llr_sample = self.llr[: int(bit_head)]
        return jsonable(
            {
                "modulation": self.modulation,
                "modulation_source": self.modulation_source,
                "symbol_rate": self.symbol_rate,
                "cfo_hz": self.cfo_hz,
                "lock": {
                    # 三态：前端优先用 state（0.7 以上锁定、0.4~0.7 临界、以下失锁）
                    "state": _lock_state(self.timing_metric, self.carrier_metric),
                    "timing": bool(self.timing_metric >= 0.5),
                    "carrier": bool(self.carrier_metric >= 0.5),
                    "timing_metric": self.timing_metric,
                    "carrier_metric": self.carrier_metric,
                    "snr_db": self.snr_db,
                    "locked_symbols": int(self.symbols.size),
                },
                "evm_percent": self.evm_percent,
                "symbols": {"count": int(symbols.size), "i": symbols.real.tolist(), "q": symbols.imag.tolist()},
                "traces": traces,
                "bits": {
                    "count": int(self.bits.size),
                    "preview": "".join(str(int(b)) for b in bits),
                    "head": [int(b) for b in bits],
                },
                "llr": {
                    "count": int(self.llr.size),
                    "mean_abs": float(np.mean(np.abs(self.llr))) if self.llr.size else 0.0,
                    "min": float(np.min(self.llr)) if self.llr.size else 0.0,
                    "max": float(np.max(self.llr)) if self.llr.size else 0.0,
                    "sample": [float(v) for v in llr_sample],
                },
                "ber": self.ber,
                "warnings": list(self.warnings),
                "evidence": jsonable(self.evidence),
            }
        )


def _lock_state(timing_metric: float, carrier_metric: float) -> str:
    """把两条环路的收敛度合成一个三态锁定状态（前端徽标直接用）."""
    worst = min(float(timing_metric), float(carrier_metric))
    if worst >= 0.7:
        return "locked"
    if worst >= 0.4:
        return "marginal"
    return "unlocked"


def _pick(array: np.ndarray, index: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return np.zeros(index.size, dtype=np.float64)
    clipped = np.clip(index, 0, array.size - 1)
    return np.asarray(array, dtype=np.float64)[clipped]


def _rotation_candidates(modulation: str) -> list[float]:
    order = rotation_symmetry_order(modulation)
    if order <= 1:
        return [0.0]
    return [2.0 * np.pi * k / order for k in range(order)]


def evaluate_against_truth(
    symbols: np.ndarray,
    modulation: str,
    truth: GroundTruth,
    sigma2: float = 1.0,
    max_shift_symbols: int = 16,
) -> Optional[dict[str, Any]]:
    """与真值对账：模糊旋转 + 符号滑动联合搜索后给出 BER/SER.

    只用于评估（真实接收机用帧头解决模糊），返回里会明确写出用了哪个旋转与滑动。
    """
    mod = mapping.normalize_modulation(modulation)
    if mod in ("2fsk", "4fsk"):
        return None
    bps = MOD_BITS_PER_SYMBOL[mod]
    y = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    truth_bits = np.asarray(truth.bits, dtype=np.uint8).reshape(-1)
    if y.size < 16 or truth_bits.size < 16:
        return None

    limit_symbols = int(min(y.size, max(16, 20000 // max(1, bps))))
    y = y[:limit_symbols]

    best: Optional[dict[str, Any]] = None
    for rotation in _rotation_candidates(mod):
        rotated = y * np.exp(1j * rotation)
        indices = mapping.points_to_indices(rotated, mod)
        bits = mapping.indices_to_bits(indices, mod)
        for shift in range(-int(max_shift_symbols), int(max_shift_symbols) + 1):
            offset = shift * bps
            if offset >= 0:
                a = bits[offset:]
                b = truth_bits[: a.size]
            else:
                a = bits[:offset] if offset != 0 else bits[:0]
                b = truth_bits[-offset : -offset + a.size]
            count = int(min(a.size, b.size))
            if count < 16:
                continue
            errors = int(np.sum(a[:count] != b[:count]))
            rate = errors / count
            if best is None or rate < best["bit_error_rate"]:
                best = {
                    "bit_errors": errors,
                    "compared_bits": count,
                    "bit_error_rate": float(rate),
                    "symbol_error_rate": float(rate),
                    "ambiguity_rotation_deg": float(np.degrees(rotation)),
                    "symbol_shift": int(shift),
                }
    return best


def demodulate(
    signal: Signal,
    analysis: Optional[AnalyzeResult] = None,
    modulation: Optional[str] = None,
    modulation_source: str = "provided",
    rolloff: float = DEFAULT_ROLLOFF,
    span_symbols: int = DEFAULT_SPAN_SYMBOLS,
    burst_regions: Optional[list[tuple[int, int]]] = None,
    blank_bursts: bool = True,
    config: Optional[LoopConfig] = None,
    truth: Optional[GroundTruth] = None,
    classifier=None,
    model_dir: str = "models",
) -> DemodResult:
    """完整接收链，返回符号/比特/LLR/环路观测量/误码统计."""
    warnings: list[str] = []
    estimates = analysis or analyze_signal(signal)
    symbol_rate = float(estimates.symbol_rate.value)
    if not np.isfinite(symbol_rate) or symbol_rate <= 0:
        raise ValueError("符号速率估计无效，无法解调")

    mod = mapping.normalize_modulation(modulation) if modulation else None
    if mod is None:
        from spxh.core.classify import classify_signal

        payload = classify_signal(signal, classifier=classifier, model_dir=model_dir, estimates=estimates)
        mod = mapping.normalize_modulation(payload["modulation"]) if payload.get("modulation") else None
        modulation_source = "classifier"
        if payload.get("gate") != "ml":
            warnings.append("调制识别门控状态为 {}，解调结论仅供参考".format(payload.get("gate")))
        if mod is None:
            raise ValueError("无法确定调制类型（分类器拒绝给出结论）")
    cfo = float(estimates.cfo.value) if np.isfinite(estimates.cfo.value) else 0.0
    snr_db = float(estimates.snr.value)
    if estimates.symbol_rate.confidence < 0.35:
        warnings.append("符号速率估计置信度偏低（{:.2f}），定时环可能锁不住".format(estimates.symbol_rate.confidence))
    if estimates.snr.confidence < 0.35:
        warnings.append("SNR 估计置信度偏低（{:.2f}），LLR 的尺度不可靠".format(estimates.snr.confidence))

    fs = float(signal.sample_rate)
    sps = fs / symbol_rate
    sigma2 = symbol_noise_variance(snr_db)
    loops = config or LoopConfig()

    if mod in ("2fsk", "4fsk"):
        result = fsk_demodulate(signal, symbol_rate=symbol_rate, cfo_hz=cfo, modulation=mod)
        llr, bits = fsk_llr(result["energies"], mod, sigma2)
        indices = result["indices"]
        payload = DemodResult(
            modulation=mod,
            modulation_source=modulation_source,
            symbol_rate=symbol_rate,
            cfo_hz=cfo,
            snr_db=snr_db,
            symbols=result["symbols"],
            symbol_indices=indices,
            bits=bits,
            llr=llr,
            evm_percent=None,
            timing_metric=1.0,
            carrier_metric=1.0,
            timing_errors=np.zeros(max(0, indices.size - 1)),
            phase_errors=np.zeros(indices.size),
            freq_estimates=np.zeros(indices.size),
            warnings=warnings,
            evidence={
                "receiver": "spxh-m3-fsk",
                "sps": float(result["sps"]),
                "tones_hz": result["tones_hz"].tolist(),
                "sigma2": sigma2,
                "num_symbols": int(indices.size),
            },
        )
        if truth is not None:
            payload.ber = _fsk_ber(indices, mod, truth)
        return payload

    design_sps = max(2, int(round(sps)))
    samples = np.asarray(signal.samples, dtype=np.complex128).reshape(-1).copy()
    # 抗突发：先把干扰/静默段**置零**（blanking）。不置零的话，强干扰的能量会被匹配滤波
    # 与 AGC 带进环路，把定时/载波相位一起拽走 —— 损伤就从"一段"扩散成"整帧"（M12-C 实测）。
    burst_regions = [(int(a), int(b)) for a, b in (burst_regions or []) if int(b) > int(a)]
    blanked = 0
    if blank_bursts and burst_regions:
        taper = max(1, design_sps // 2)
        for start, end in burst_regions:
            start = max(0, start)
            end = min(samples.size, end)
            if end <= start:
                continue
            samples[start:end] = 0.0
            blanked += end - start
            # 两端做短线性过渡，避免硬切边引入的频谱泄漏反过来影响同步环
            for offset in range(taper):
                if start - 1 - offset >= 0:
                    samples[start - 1 - offset] *= (offset + 1) / (taper + 1)
                if end + offset < samples.size:
                    samples[end + offset] *= (offset + 1) / (taper + 1)
    index = np.arange(samples.size, dtype=np.float64)
    coarse_corrected = samples * np.exp(-2j * np.pi * cfo * index / fs)
    taps = rrc_taps(design_sps, span_symbols=int(span_symbols), rolloff=float(rolloff))
    filtered = matched_filter(coarse_corrected, taps)
    filtered, agc_gain = agc_normalize(filtered)

    # 从匹配滤波器峰值附近起步（2*delay），避开滤波器暂态零区
    filter_delay = (taps.size - 1) / 2.0
    start_position = TimingRecovery.default_start_position(sps, filter_delay=filter_delay)
    timing = TimingRecovery(sps, loops).run(
        filtered, start_position=start_position, freeze_regions=burst_regions
    )
    if timing.symbols.size < 16:
        raise ValueError("定时环输出符号过少（记录太短或环路发散）")
    # 载波环按**符号**冻结：符号位置落在突发区间内就不更新环路，只按当前相位/频率滑行
    freeze_mask = None
    frozen_symbols = 0
    if burst_regions and timing.symbol_positions.size:
        positions = np.asarray(timing.symbol_positions, dtype=np.float64)
        freeze_mask = np.zeros(positions.size, dtype=bool)
        for start, end in burst_regions:
            freeze_mask |= (positions >= start) & (positions < end)
        frozen_symbols = int(np.count_nonzero(freeze_mask))
    carrier = CarrierRecovery(mod, symbol_rate, loops).run(
        timing.symbols, coarse_freq_hz=cfo, freeze_mask=freeze_mask
    )

    # 符号级增益归一化：AGC 是在**滤波器输出**上做的，含暂态与噪声，符号样本的
    # 实际功率会有百分之几的偏差；而 LLR 的尺度（sigma^2）与 EVM 都假设判决点功率为 1。
    recovered = np.asarray(carrier.symbols, dtype=np.complex128)
    if recovered.size:
        scale = float(np.sqrt(np.mean(np.abs(recovered) ** 2)))
        if np.isfinite(scale) and scale > 0:
            recovered = recovered / scale

    llr, bits, symbol_indices = max_log_llr(recovered, mod, sigma2)
    ideal = mapping.indices_to_points(symbol_indices, mod)
    error = recovered - ideal
    reference = float(np.sqrt(np.mean(np.abs(ideal) ** 2))) or 1.0
    evm = float(100.0 * np.sqrt(np.mean(np.abs(error) ** 2)) / reference)

    if timing.metric < 0.5:
        warnings.append("定时环收敛度偏低（{:.2f}）".format(timing.metric))
    if carrier.metric < 0.5:
        warnings.append("载波环收敛度偏低（{:.2f}）".format(carrier.metric))

    result = DemodResult(
        modulation=mod,
        modulation_source=modulation_source,
        symbol_rate=symbol_rate,
        cfo_hz=cfo,
        snr_db=snr_db,
        symbols=recovered,
        symbol_indices=symbol_indices,
        bits=bits,
        llr=llr,
        evm_percent=evm,
        timing_metric=float(timing.metric),
        carrier_metric=float(carrier.metric),
        timing_errors=timing.timing_errors,
        phase_errors=carrier.phase_errors,
        freq_estimates=carrier.freq_estimates,
        # 定时环记录的是"滤波后"的位置；AGC/匹配滤波都不改变时间轴长度，
        # 因此它就是原信号上的采样索引（供突发区间映射用）。
        symbol_samples=np.asarray(timing.symbol_positions, dtype=np.float64),
        warnings=warnings,
        evidence={
            "receiver": "spxh-m3-psk-qam",
            "sps": float(sps),
            "rolloff": float(rolloff),
            "span_symbols": int(span_symbols),
            "sigma2": sigma2,
            "agc_gain": float(agc_gain),
            "loop_config": loops.to_dict(),
            "burst_regions": [[int(a), int(b)] for a, b in burst_regions],
            "blanked_samples": int(blanked),
            "frozen_symbols": int(frozen_symbols),
            "estimated_freq_hz": float(carrier.final_freq_hz),
        },
    )
    if truth is not None:
        result.ber = evaluate_against_truth(recovered, mod, truth, sigma2)
    return result


def _fsk_ber(indices: np.ndarray, modulation: str, truth: GroundTruth) -> Optional[dict[str, Any]]:
    mod = mapping.normalize_modulation(modulation)
    bits = mapping.indices_to_bits(indices, mod)
    truth_bits = np.asarray(truth.bits, dtype=np.uint8).reshape(-1)
    if bits.size < 16 or truth_bits.size < 16:
        return None
    bps = MOD_BITS_PER_SYMBOL[mod]
    best: Optional[dict[str, Any]] = None
    for shift in range(-16, 17):
        offset = shift * bps
        if offset >= 0:
            a = bits[offset:]
            b = truth_bits[: a.size]
        else:
            a = bits[:offset]
            b = truth_bits[-offset : -offset + a.size]
        count = int(min(a.size, b.size))
        if count < 16:
            continue
        errors = int(np.sum(a[:count] != b[:count]))
        rate = errors / count
        if best is None or rate < best["bit_error_rate"]:
            best = {
                "bit_errors": errors,
                "compared_bits": count,
                "bit_error_rate": float(rate),
                "symbol_error_rate": float(rate),
                "ambiguity_rotation_deg": 0.0,
                "symbol_shift": int(shift),
            }
    return best
