"""端到端信号生成器：载荷 -> 帧 -> 调制 -> 损伤 -> 波形 + 真值.

这是 M0 的核心：**每个生成信号都带完整真值**（比特、符号索引、星座点、帧边界、
CRC 值、全部参数），后续每一级都能用真值量化误差。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from spxh.core.fec.codec import coded_length_bits
from spxh.core.framing.format import build_frame, frame_length_bits
from spxh.core.synth import mapping
from spxh.core.synth.impairments import (
    add_awgn,
    apply_cfo,
    apply_dc_offset,
    apply_iq_imbalance,
    apply_phase_noise,
)
from spxh.core.synth.modulators import ModParams, Modulated, modulate_bits
from spxh.core.types import (
    MOD_BITS_PER_SYMBOL,
    FrameRecord,
    GroundTruth,
    Signal,
    normalize_modulation,
)

__all__ = ["SynthConfig", "synthesize", "es_per_symbol"]

GENERATOR_TAG = "spxh-m0"


@dataclass
class SynthConfig:
    """合成信号配置（可 JSON 序列化，可复现）."""

    modulation: str = "qpsk"
    symbol_rate: float = 1000.0
    sps: int = 8
    sample_rate: Optional[float] = None
    snr_db: float = 20.0
    cfo_hz: float = 0.0
    phase_noise_rms_rad: float = 0.0
    phase_noise_bw_hz: Optional[float] = None
    rolloff: float = 0.35
    span_symbols: int = 16
    fsk_mod_index: Optional[float] = None
    payload_len: int = 64
    num_frames: int = 3
    payloads: Optional[list] = None
    fec: str = "none"
    interleave: bool = False
    iq_gain_db: float = 0.0
    iq_phase_deg: float = 0.0
    dc_offset: complex = 0j
    seed: int = 0
    keep_clean: bool = True

    def resolved_sample_rate(self) -> float:
        if self.sample_rate is not None:
            return float(self.sample_rate)
        return float(int(self.sps) * float(self.symbol_rate))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["payloads"] = None if self.payloads is None else [bytes(p).hex() for p in self.payloads]
        data["dc_offset"] = [float(self.dc_offset.real), float(self.dc_offset.imag)]
        data["sample_rate_effective"] = self.resolved_sample_rate()
        return data

    def validate(self) -> None:
        mod = normalize_modulation(self.modulation)
        if mod in ("2fsk", "4fsk") and self.sps < 2:
            raise ValueError("sps must be >= 2")
        if int(self.sps) < 2:
            raise ValueError("sps must be >= 2")
        if float(self.symbol_rate) <= 0:
            raise ValueError("symbol_rate must be positive")
        if self.sample_rate is not None and abs(self.sample_rate / self.symbol_rate - round(self.sample_rate / self.symbol_rate)) > 1e-6:
            raise ValueError("sample_rate must be an integer multiple of symbol_rate")
        # 编码/交织在 M4/M6 已接入：这里只做参数自洽性检查（长度必须由
        # (payload_len, fec, interleave) 唯一确定，否则接收端算不出帧长）
        try:
            coded_length_bits(int(self.payload_len), str(self.fec), bool(self.interleave))
        except ValueError as exc:
            raise ValueError("fec/interleave 与 payload_len 不匹配：" + str(exc)) from exc
        if int(self.num_frames) < 1:
            raise ValueError("num_frames must be >= 1")
        # 允许负 SNR：方案要求 −5 至 +35 dB 可配，低信噪比（含负值）是 AMC 的关键工况
        if not (-60.0 <= float(self.snr_db) <= 200.0):
            raise ValueError("snr_db out of range")
        if float(self.phase_noise_rms_rad) < 0:
            raise ValueError("phase_noise_rms_rad must be >= 0")


def es_per_symbol(modulated: Modulated) -> float:
    """每个符号波形的能量 Es.

    PSK/QAM：单位能量成型后 Es = 星座平均能量（= 1）。
    FSK：恒模且每符号 sps 个采样点，Es = sps * 平均功率。
    """
    if modulated.points is not None:
        return float(np.mean(np.abs(modulated.points) ** 2))
    return float(modulated.params.sps * np.mean(np.abs(modulated.waveform) ** 2))


def _default_payloads(cfg: SynthConfig, rng: np.random.Generator) -> list[bytes]:
    length = int(cfg.payload_len)
    if length < 0 or length > 0xFFFF:
        raise ValueError("payload_len out of range: " + str(length))
    return [
        rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()
        for _ in range(int(cfg.num_frames))
    ]


def synthesize(cfg: SynthConfig) -> tuple[Signal, GroundTruth]:
    """生成 (接收信号, 真值)."""
    cfg.validate()
    mod = normalize_modulation(cfg.modulation)
    fs = cfg.resolved_sample_rate()
    rng = np.random.default_rng(int(cfg.seed))

    payloads = [bytes(p) for p in cfg.payloads] if cfg.payloads is not None else _default_payloads(cfg, rng)
    if len(payloads) < 1:
        raise ValueError("payloads must not be empty")

    frame_bit_blocks: list[np.ndarray] = []
    frame_plan: list[tuple[Any, bytes, int, int]] = []
    bit_cursor = 0
    for payload in payloads:
        bits_k, header, crc = build_frame(payload, mod, fec=cfg.fec, interleave=cfg.interleave)
        frame_plan.append((header, payload, crc, bit_cursor))
        frame_bit_blocks.append(bits_k)
        bit_cursor += int(bits_k.size)

    bits = np.concatenate(frame_bit_blocks).astype(np.uint8)
    bps = MOD_BITS_PER_SYMBOL[mod]
    pad_bits = (-bits.size) % bps
    if pad_bits:
        padding = rng.integers(0, 2, size=pad_bits, dtype=np.uint8)
        bits = np.concatenate([bits, padding]).astype(np.uint8)

    params = ModParams(
        modulation=mod,
        symbol_rate=float(cfg.symbol_rate),
        sample_rate=fs,
        rolloff=float(cfg.rolloff),
        span_symbols=int(cfg.span_symbols),
        fsk_mod_index=cfg.fsk_mod_index,
    )
    modulated = modulate_bits(bits, params)
    clean = modulated.waveform

    tx = apply_cfo(clean, fs, float(cfg.cfo_hz))
    tx = apply_phase_noise(tx, fs, float(cfg.phase_noise_rms_rad), cfg.phase_noise_bw_hz, rng=rng)
    tx = apply_iq_imbalance(tx, float(cfg.iq_gain_db), float(cfg.iq_phase_deg))
    tx = apply_dc_offset(tx, cfg.dc_offset)

    es = es_per_symbol(modulated)
    rx, sigma2 = add_awgn(tx, float(cfg.snr_db), es_per_symbol=es, rng=rng, return_sigma=True)

    delay = float(modulated.delay_samples)
    sps = params.sps
    frames: list[FrameRecord] = []
    for index, (header, payload, crc, start_bit) in enumerate(frame_plan):
        num_bits = frame_length_bits(len(payload), cfg.fec, cfg.interleave)
        start_symbol = start_bit // bps
        num_symbols = -(-num_bits // bps)
        frames.append(
            FrameRecord(
                index=index,
                header=header,
                payload=payload,
                crc=int(crc),
                start_bit=int(start_bit),
                num_bits=int(num_bits),
                start_symbol=int(start_symbol),
                num_symbols=int(num_symbols),
                start_sample=int(start_symbol * sps),
                num_samples=int(num_symbols * sps),
            )
        )

    truth = GroundTruth(
        modulation=mod,
        symbol_rate=float(cfg.symbol_rate),
        sample_rate=fs,
        sps=int(sps),
        snr_db=float(cfg.snr_db),
        cfo_hz=float(cfg.cfo_hz),
        phase_noise_rms_rad=float(cfg.phase_noise_rms_rad),
        rolloff=float(cfg.rolloff),
        span_symbols=int(params.span_symbols),
        fsk_mod_index=float(params.fsk_mod_index) if params.fsk_mod_index is not None else 0.0,
        seed=int(cfg.seed),
        num_symbols=int(modulated.num_symbols),
        num_bits=int(bits.size),
        pad_bits=int(pad_bits),
        filter_delay_samples=delay,
        noise_variance=float(sigma2),
        frames=frames,
        extras={
            "generator": GENERATOR_TAG,
            "fec": cfg.fec,
            "interleave": bool(cfg.interleave),
            "es_per_symbol": es,
            "noise_model": "complex AWGN, sigma^2 = Es/10^(snr_db/10), I/Q 均分",
            "snr_definition": "10*log10(Es/N0)",
            "bits_per_symbol": bps,
            "sample_index_convention": "采样索引基于上采样符号栅格：符号 k 占 [k*sps, (k+1)*sps)；波形数组相对该栅格有 +filter_delay_samples 的偏移",
            "payloads_hex": [p.hex() for p in payloads],
            "config": cfg.to_dict(),
        },
        bits=bits,
        symbol_indices=modulated.indices,
        symbols=modulated.points,
        tone_hz=modulated.tone_hz,
        clean=clean.copy() if cfg.keep_clean else None,
    )

    signal = Signal(
        samples=rx,
        sample_rate=fs,
        center_freq=None,
        meta={
            "source": "spxh.synth",
            "datatype": "synthetic",
            "modulation": mod,
            "symbol_rate": float(cfg.symbol_rate),
            "sps": int(sps),
            "snr_db": float(cfg.snr_db),
            "cfo_hz": float(cfg.cfo_hz),
            "noise_variance": float(sigma2),
            "seed": int(cfg.seed),
            "generator": GENERATOR_TAG,
        },
    )
    return signal, truth
