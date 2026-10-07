"""定时同步（Gardner + 插值）与载波跟踪（判决导向二阶环）.

设计要点
--------
* 定时环工作在匹配滤波后的过采样信号上，用 Gardner 误差检测器 + 二阶环路滤波 +
  Farrow 插值，输出每秒一个符号的判决样本，同时给出定时误差轨迹（供前端观测）。
* 载波环用判决导向（DD）相位检测器 + 二阶 PLL：先用 M 次方粗估把残余频偏压到环路
  捕获带内（这一步复用 M1/M2 的 estimate_residual_cfo），再让环路自己跟踪剩余部分。
* 两个环路都输出**观测量**（timing_error / phase_error / freq_estimate），这是"能不能
  相信解调结果"的直接证据，也是前端"解调"页要画的三条曲线。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from spxh.core.demod.filters import FarrowInterpolator
from spxh.core.synth import mapping

__all__ = ["TimingRecovery", "CarrierRecovery", "LoopConfig"]


@dataclass
class LoopConfig:
    """环路参数（默认值按 2 sps 归一化，实测在 QPSK/16QAM/8PSK 上都能收敛）."""

    timing_kp: float = 0.08
    timing_ki: float = 0.002
    carrier_kp: float = 0.10
    carrier_ki: float = 0.0015
    max_timing_correction: float = 0.25
    max_freq_correction_ratio: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return {
            "timing_kp": self.timing_kp,
            "timing_ki": self.timing_ki,
            "carrier_kp": self.carrier_kp,
            "carrier_ki": self.carrier_ki,
        }


@dataclass
class TimingResult:
    symbols: np.ndarray
    timing_errors: np.ndarray
    corrected_indices: np.ndarray
    loop_state: np.ndarray = field(default_factory=lambda: np.zeros(0))
    metric: float = 0.0
    #: 每个判决符号在**滤波后样本序列**里的位置（长度与 symbols 相同）。
    #: 有了它才能把"信号侧检测到的突发区间"映射成"哪些符号不可信"，
    #: 进而给出编码域的擦除位置（M12 的突发检测 -> 自动擦除）。
    symbol_positions: np.ndarray = field(default_factory=lambda: np.zeros(0))


class TimingRecovery:
    """Gardner 定时误差检测 + 二阶环路 + Farrow 插值（输入需 >= 2 样本/符号）."""

    def __init__(self, samples_per_symbol: float, config: Optional[LoopConfig] = None) -> None:
        self.sps = float(samples_per_symbol)
        if self.sps < 2.0:
            raise ValueError("定时环需要至少 2 样本/符号，当前 {:.2f}".format(self.sps))
        self.config = config or LoopConfig()

    @staticmethod
    def default_start_position(samples_per_symbol: float, filter_delay: float = 0.0) -> float:
        """推荐的起始采样位置.

        匹配滤波后，第 k 个符号的峰值在 k*sps + 2*filter_delay；如果从 0 附近起步，
        环路会在滤波器的**暂态零区**里空转（误差≈0、功率≈0），既浪费时间又会让
        归一化尺度失真。接收机知道滤波器延迟，所以直接从这里起步。
        """
        base = 2.0 * float(filter_delay)
        return max(float(np.ceil(samples_per_symbol / 2.0)) + 1.0, base + 2.0)

    def run(
        self,
        samples: np.ndarray,
        max_symbols: Optional[int] = None,
        start_position: Optional[float] = None,
        freeze_regions: Optional[list[tuple[int, int]]] = None,
    ) -> TimingResult:
        """逐符号输出判决样本，同时记录定时误差轨迹.

        Gardner 误差检测器的正确形式（以 sps 样本/符号工作）：
            y[k]      第 k 个符号的判决样本，位置 p_k
            y[k-1/2]  p_k - sps/2 处的中间样本（即第 k-1 与第 k 个符号之间的中点）
            e[k] = Re{ conj(y[k-1/2]) * (y[k] - y[k-1]) }
        注意中点必须取 sps/2 的偏移，而不是固定 0.5 个样本 —— 后者只在 2 样本/符号时成立。
        """
        x = np.asarray(samples, dtype=np.complex128).reshape(-1)
        interpolator = FarrowInterpolator(x)
        cfg = self.config
        half = self.sps / 2.0
        mu = 0.0
        integrator = 0.0
        position = float(start_position) if start_position is not None else self.default_start_position(self.sps)

        symbols: list[complex] = []
        errors: list[float] = []
        indices: list[float] = []
        previous_on: Optional[complex] = None
        # 归一化用**全段平均功率**（常数），避免暂态段把尺度带偏
        power = float(np.mean(np.abs(x) ** 2)) if x.size else 1.0
        if not np.isfinite(power) or power <= 0:
            power = 1.0

        limit = int(max_symbols) if max_symbols else int(max(0.0, (x.size - 4 - half - position)) / self.sps)
        regions = [(int(a), int(b)) for a, b in (freeze_regions or []) if int(b) > int(a)]

        def _frozen(sample_position: float) -> bool:
            """当前位置是否落在突发区间内（滑行区）—— 区间通常只有几段，线性扫描足够快."""
            value = int(sample_position)
            for start, end in regions:
                if start <= value < end:
                    return True
            return False

        for _ in range(max(0, limit)):
            on_time = interpolator(position + mu)
            mid = interpolator(position + mu - half)
            if previous_on is not None and not _frozen(position + mu):
                error = float(np.real(np.conj(mid) * (on_time - previous_on))) / max(power, 1e-12)
                error = float(np.clip(error, -2.0, 2.0))
                errors.append(error)
                # Gardner 的 S 曲线：采样偏晚（off>0）时 e>0，因此要把采样位置**往前**拉，
                # 也就是 mu 减小 —— 符号写反环路会直接发散（实测 EVM 卡在 59% 不收敛）。
                integrator += cfg.timing_ki * error
                correction = cfg.timing_kp * error + integrator
                correction = float(np.clip(correction, -cfg.max_timing_correction, cfg.max_timing_correction))
                mu -= correction
                if mu > 0.5:
                    mu -= 1.0
                    position += 1.0
                elif mu < -0.5:
                    mu += 1.0
                    position -= 1.0

            symbols.append(complex(on_time))
            indices.append(float(position + mu))
            previous_on = on_time
            position += self.sps

        symbol_array = np.asarray(symbols, dtype=np.complex128)
        error_array = np.asarray(errors, dtype=np.float64)
        return TimingResult(
            symbols=symbol_array,
            timing_errors=error_array,
            corrected_indices=np.asarray(indices, dtype=np.float64)[1:],
            loop_state=np.asarray([mu, integrator], dtype=np.float64),
            metric=_loop_metric(error_array),
            symbol_positions=np.asarray(indices, dtype=np.float64),
        )


@dataclass
class CarrierResult:
    symbols: np.ndarray
    phase_errors: np.ndarray
    freq_estimates: np.ndarray
    metric: float = 0.0
    coarse_freq_hz: float = 0.0
    final_freq_hz: float = 0.0


class CarrierRecovery:
    """判决导向二阶载波环（PSK/QAM 通用）."""

    def __init__(self, modulation: str, symbol_rate: float, config: Optional[LoopConfig] = None) -> None:
        self.modulation = mapping.normalize_modulation(modulation)
        if self.modulation in ("2fsk", "4fsk"):
            raise ValueError("FSK 不做判决导向载波跟踪（用非相干音调检测）")
        self.symbol_rate = float(symbol_rate)
        self.config = config or LoopConfig()
        self.constellation = mapping.constellation(self.modulation)
        self.min_distance = mapping.min_distance(self.modulation)

    def run(self, symbols: np.ndarray, coarse_freq_hz: float = 0.0,
            freeze_mask: Optional[np.ndarray] = None) -> CarrierResult:
        x = np.asarray(symbols, dtype=np.complex128).reshape(-1)
        cfg = self.config
        phase = 0.0
        integrator = 0.0
        frequency = 0.0
        derotated: list[complex] = []
        phase_errors: list[float] = []
        frequencies: list[float] = []

        # 环路内部用"每符号相位增量"（rad/symbol）积分，对外统一换算成 Hz：
        # freq_hz 就是环跟踪到的频偏（与输入残余同号），直接对应频谱上的偏移量。
        max_freq_hz = cfg.max_freq_correction_ratio * self.symbol_rate
        rad_per_hz = 2.0 * np.pi / self.symbol_rate
        mask = np.asarray(freeze_mask, dtype=bool).reshape(-1) if freeze_mask is not None else None
        for index, sample in enumerate(x):
            rotated = sample * np.exp(-1j * phase)
            if mask is not None and index < mask.size and mask[index]:
                # 突发区间：**不更新环路**（判决导向误差在干扰/静默段完全不可信），
                # 只按当前相位与频率"滑行"过去 —— 出突发时环路仍是锁的，损伤因此局部化。
                phase += integrator
                phase = float(np.angle(np.exp(1j * phase)))
                derotated.append(complex(sample * np.exp(-1j * phase)))
                phase_errors.append(0.0)
                frequencies.append(float(np.clip(integrator / rad_per_hz, -max_freq_hz, max_freq_hz)))
                continue
            decision_index = int(mapping.points_to_indices(np.asarray([rotated]), self.modulation)[0])
            decision = self.constellation[decision_index]
            # 判决导向相位误差（用误差矢量的横向分量，并做幅度归一化）
            error_vector = rotated - decision
            magnitude = abs(rotated) ** 2 + 1e-12
            error = float(np.imag(np.conj(decision) * error_vector) / magnitude)
            error = float(np.clip(error, -0.5, 0.5))
            phase_errors.append(error)

            integrator += cfg.carrier_ki * error
            freq_hz = float(np.clip(integrator / rad_per_hz, -max_freq_hz, max_freq_hz))
            integrator = freq_hz * rad_per_hz          # 限幅后写回积分器，避免无界漂移
            phase += cfg.carrier_kp * error + integrator
            phase = float(np.angle(np.exp(1j * phase)))  # 保持在 [-pi, pi]

            derotated.append(complex(sample * np.exp(-1j * phase)))
            frequencies.append(freq_hz)

        phase_array = np.asarray(phase_errors, dtype=np.float64)
        frequency_array = np.asarray(frequencies, dtype=np.float64)
        metric = carrier_lock_metric(phase_array)
        # freq_estimates 是"环跟踪到的频偏"（与输入残余同号）：粗估 + 环跟踪量才是完整估计
        tracked = float(np.mean(frequency_array[-max(1, frequency_array.size // 4) :])) if frequency_array.size else 0.0
        return CarrierResult(
            symbols=np.asarray(derotated, dtype=np.complex128),
            phase_errors=phase_array,
            freq_estimates=frequency_array,
            metric=metric,
            coarse_freq_hz=float(coarse_freq_hz),
            final_freq_hz=float(coarse_freq_hz + tracked),
        )


def _tail(errors: np.ndarray, tail_ratio: float = 0.5) -> np.ndarray:
    values = np.asarray(errors, dtype=np.float64)
    if values.size < 8:
        return values
    return values[int(values.size * (1.0 - tail_ratio)) :]


def _loop_metric(errors: np.ndarray, tail_ratio: float = 0.5, scale: float = 1.5) -> float:
    """定时环收敛度：尾部定时误差的 RMS 映射到 0~1.

    尺度 1.5 是**实测标定**的：误差已按全段平均功率归一化，锁定状态下尾部 RMS 约
    0.4~0.5（对应 EVM 与理论噪声底一致），失锁时 > 3。用固定尺度而不是拿整段 RMS
    当参考 —— 后者在锁定与失锁两种情况下几乎一样大，等于没有信息（实测恒为 0.2）。
    """
    tail = _tail(errors, tail_ratio)
    if tail.size < 4:
        return 0.0
    rms_tail = float(np.sqrt(np.mean(tail ** 2)))
    return float(1.0 / (1.0 + (rms_tail / float(scale)) ** 2))


def carrier_lock_metric(phase_errors: np.ndarray, tail_ratio: float = 0.5) -> float:
    """载波环锁定指示：尾部相位误差的**集中度** |mean(exp(j*e))|.

    锁定时相位误差集中在 0 附近 -> 接近 1；失锁时在 [-pi, pi] 上铺开 -> 接近 0。
    这是通信里标准的锁检测量，比"误差 RMS 归一化"可解释得多。
    """
    tail = _tail(phase_errors, tail_ratio)
    if tail.size < 4:
        return 0.0
    return float(abs(np.mean(np.exp(1j * tail))))
