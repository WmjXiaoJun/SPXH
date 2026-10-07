"""理想参考接收端，仅用于验证 M0 合成链是否正确.

它不是生产解调器：不做定时同步、不做载波恢复、不做均衡。
适用前提是「无频偏、无相噪、无关调、采样相位已知」——
这正是 M0 真值自检需要的条件；真实接收链在 M3 实现。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from spxh.core.synth import mapping
from spxh.core.synth.modulators import default_fsk_mod_index
from spxh.core.synth.pulse import matched_filter, rrc_taps
from spxh.core.types import GroundTruth, Signal

__all__ = [
    "receive_psk_qam",
    "receive_fsk",
    "receive",
    "symbol_points_psk_qam",
    "evm_percent",
    "ber",
]


def _require_psk_qam(truth: GroundTruth) -> None:
    if truth.modulation in ("2fsk", "4fsk"):
        raise ValueError("receive_psk_qam does not apply to " + truth.modulation)


def _resolve_span(truth: GroundTruth, span_symbols: Optional[int]) -> int:
    if span_symbols is not None:
        return int(span_symbols)
    return int(getattr(truth, "span_symbols", 16) or 16)


def symbol_points_psk_qam(signal: Signal, truth: GroundTruth, span_symbols: Optional[int] = None) -> np.ndarray:
    """匹配滤波 + 理想采样，返回每符号复数值.

    接收端必须使用与发端**完全相同**的 RRC 抽头，否则群延迟不一致会直接采错点。
    """
    _require_psk_qam(truth)
    taps = rrc_taps(truth.sps, span_symbols=_resolve_span(truth, span_symbols), rolloff=truth.rolloff)
    filtered = matched_filter(signal.samples, taps)
    start = int(round(2.0 * truth.filter_delay_samples))
    samples = filtered[start:: truth.sps][: truth.num_symbols]
    if samples.size < truth.num_symbols:
        raise ValueError("signal too short for " + str(truth.num_symbols) + " symbols")
    return samples


def receive_psk_qam(signal: Signal, truth: GroundTruth, span_symbols: Optional[int] = None) -> np.ndarray:
    samples = symbol_points_psk_qam(signal, truth, span_symbols=span_symbols)
    indices = mapping.points_to_indices(samples, truth.modulation)
    return mapping.indices_to_bits(indices, truth.modulation)


def receive_fsk(signal: Signal, truth: GroundTruth) -> np.ndarray:
    """逐符号音调相关（相干、理想定时）."""
    if truth.modulation not in ("2fsk", "4fsk"):
        raise ValueError("receive_fsk only applies to FSK, got " + truth.modulation)
    sps = int(truth.sps)
    fs = float(truth.sample_rate)
    m = mapping.num_points(truth.modulation)
    h = truth.fsk_mod_index or default_fsk_mod_index(truth.modulation)
    tones = (np.arange(m) - (m - 1) / 2.0) * h * truth.symbol_rate
    n = np.arange(sps)
    basis = np.exp(-2j * np.pi * np.outer(tones, n) / fs)
    x = np.asarray(signal.samples)
    count = min(truth.num_symbols, x.size // sps)
    indices = np.zeros(count, dtype=np.int64)
    for k in range(count):
        seg = x[k * sps : (k + 1) * sps]
        indices[k] = int(np.argmax(np.abs(basis @ seg)))
    return mapping.indices_to_bits(indices, truth.modulation)


def receive(signal: Signal, truth: GroundTruth) -> np.ndarray:
    if truth.modulation in ("2fsk", "4fsk"):
        return receive_fsk(signal, truth)
    return receive_psk_qam(signal, truth)


def ber(reference_bits, received_bits) -> float:
    ref = np.asarray(reference_bits).reshape(-1)
    got = np.asarray(received_bits).reshape(-1)
    n = min(ref.size, got.size)
    if n == 0:
        return float("nan")
    return float(np.mean(ref[:n] != got[:n]))


def evm_percent(signal: Signal, truth: GroundTruth, span_symbols: Optional[int] = None) -> float:
    """EVM（%）：匹配滤波后采样点相对理想星座点的 RMS 误差."""
    samples = symbol_points_psk_qam(signal, truth, span_symbols=span_symbols)
    ideal = mapping.indices_to_points(truth.symbol_indices[: samples.size], truth.modulation)
    error = samples - ideal
    ref = np.sqrt(np.mean(np.abs(ideal) ** 2))
    if ref <= 0:
        return float("nan")
    return float(100.0 * np.sqrt(np.mean(np.abs(error) ** 2)) / ref)
