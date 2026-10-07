"""调制器：比特/索引 -> 复数基带波形.

PSK/QAM：星座映射 + RRC 成型滤波。
FSK：连续相位（CPFSK），每符号 sps 个采样点，音调间隔 h*Rs；
     2FSK 默认 h=1.0（音调间隔 Rs，Sunde FSK，正交），4FSK 默认 h=0.5（间隔 Rs/2，正交）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spxh.core.synth import mapping
from spxh.core.synth.pulse import pulse_shape, rrc_taps
from spxh.core.types import MOD_BITS_PER_SYMBOL, normalize_modulation

__all__ = ["ModParams", "Modulated", "default_fsk_mod_index", "fsk_tone_hz", "modulate_indices", "modulate_bits"]


def default_fsk_mod_index(modulation: str) -> float:
    mod = normalize_modulation(modulation)
    if mod == "2fsk":
        return 1.0
    if mod == "4fsk":
        return 0.5
    raise ValueError("default_fsk_mod_index only applies to FSK, got " + mod)


@dataclass(frozen=True)
class ModParams:
    modulation: str
    symbol_rate: float
    sample_rate: float
    rolloff: float = 0.35
    span_symbols: int = 16
    fsk_mod_index: Optional[float] = None

    def __post_init__(self) -> None:
        mod = normalize_modulation(self.modulation)
        object.__setattr__(self, "modulation", mod)
        symbol_rate = float(self.symbol_rate)
        sample_rate = float(self.sample_rate)
        object.__setattr__(self, "symbol_rate", symbol_rate)
        object.__setattr__(self, "sample_rate", sample_rate)
        if symbol_rate <= 0 or sample_rate <= 0:
            raise ValueError("symbol_rate and sample_rate must be positive")
        ratio = sample_rate / symbol_rate
        sps = int(round(ratio))
        if abs(ratio - sps) > 1e-6:
            raise ValueError("sample_rate must be an integer multiple of symbol_rate (got ratio " + str(ratio) + ")")
        if sps < 2:
            raise ValueError("samples per symbol must be >= 2")
        if not (0.0 <= float(self.rolloff) <= 1.0):
            raise ValueError("rolloff must be in [0, 1]")
        if int(self.span_symbols) < 1:
            raise ValueError("span_symbols must be >= 1")
        if mod in ("2fsk", "4fsk"):
            index = self.fsk_mod_index
            object.__setattr__(self, "fsk_mod_index", default_fsk_mod_index(mod) if index is None else float(index))
            if float(object.__getattribute__(self, "fsk_mod_index")) <= 0:
                raise ValueError("fsk_mod_index must be positive")

    @property
    def sps(self) -> int:
        return int(round(self.sample_rate / self.symbol_rate))

    @property
    def bits_per_symbol(self) -> int:
        return MOD_BITS_PER_SYMBOL[self.modulation]

    @property
    def fsk_index(self) -> float:
        if self.fsk_mod_index is None:
            raise ValueError("modulation " + self.modulation + " has no FSK modulation index")
        return float(self.fsk_mod_index)

    def to_dict(self) -> dict:
        return {
            "modulation": self.modulation,
            "symbol_rate": self.symbol_rate,
            "sample_rate": self.sample_rate,
            "sps": self.sps,
            "rolloff": float(self.rolloff),
            "span_symbols": int(self.span_symbols),
            "fsk_mod_index": None if self.fsk_mod_index is None else float(self.fsk_mod_index),
        }


@dataclass
class Modulated:
    waveform: np.ndarray
    indices: np.ndarray
    points: Optional[np.ndarray]
    tone_hz: Optional[np.ndarray]
    delay_samples: float
    params: ModParams

    @property
    def num_symbols(self) -> int:
        return int(self.indices.size)

    @property
    def num_samples(self) -> int:
        return int(self.waveform.size)


def fsk_tone_hz(indices, params: ModParams) -> np.ndarray:
    """索引 -> 每符号音调频率（Hz，相对基带中心）."""
    mod = normalize_modulation(params.modulation)
    m = mapping.num_points(mod)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return (idx.astype(np.float64) - (m - 1) / 2.0) * params.fsk_index * params.symbol_rate


def modulate_indices(indices, params: ModParams) -> Modulated:
    """星座/音调索引 -> 基带波形."""
    mod = normalize_modulation(params.modulation)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError("indices must not be empty")
    n_points = mapping.num_points(mod)
    if np.any(idx < 0) or np.any(idx >= n_points):
        raise ValueError("index out of range for " + mod + " (0.." + str(n_points - 1) + ")")

    if mod in ("2fsk", "4fsk"):
        tones = fsk_tone_hz(idx, params)
        freqs = np.repeat(tones, params.sps)
        phase = np.cumsum(2.0 * np.pi * freqs / params.sample_rate)
        waveform = np.exp(1j * phase).astype(np.complex128)
        return Modulated(
            waveform=waveform,
            indices=idx,
            points=None,
            tone_hz=tones,
            delay_samples=0.0,
            params=params,
        )

    points = mapping.indices_to_points(idx, mod)
    taps = rrc_taps(params.sps, span_symbols=params.span_symbols, rolloff=params.rolloff)
    waveform, delay = pulse_shape(points, params.sps, taps=taps)
    return Modulated(
        waveform=waveform,
        indices=idx,
        points=points,
        tone_hz=None,
        delay_samples=float(delay),
        params=params,
    )


def modulate_bits(bits, params: ModParams) -> Modulated:
    """比特流 -> 基带波形（比特数必须是每符号比特数的整数倍）."""
    indices = mapping.bits_to_indices(bits, params.modulation)
    return modulate_indices(indices, params)
