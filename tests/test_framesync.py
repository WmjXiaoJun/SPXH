"""M5 测试：Barker 前导相关、相位模糊求解、帧同步与载荷提取."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.demod.filters import fractional_delay
from spxh.core.framing.barker import barker_bits
from spxh.core.framesync import correlate_bits, correlate_symbols, extract_frames, preamble_pattern, preamble_symbols
from spxh.core.synth import SynthConfig, synthesize
from spxh.core.types import MODULATIONS, Signal


def _case(modulation: str, snr_db: float = 30.0, seed: int = 11, frames: int = 4, payload: int = 64):
    return synthesize(
        SynthConfig(
            modulation=modulation,
            symbol_rate=1000.0,
            sps=8,
            snr_db=snr_db,
            cfo_hz=37.0,
            num_frames=frames,
            payload_len=payload,
            seed=seed,
            keep_clean=True,
        )
    )


def test_preamble_pattern_uses_constellation_convention():
    # 比特 0 -> +1、比特 1 -> -1（与 M0 的 BPSK 星座一致）
    pattern = preamble_pattern()
    bits = barker_bits(13)
    assert np.array_equal(pattern, np.where(bits == 1, -1.0, 1.0))
    assert pattern.size == 13


@pytest.mark.parametrize("modulation,expected", (("bpsk", 13), ("qpsk", 6), ("8psk", 4), ("16qam", 3)))
def test_preamble_symbols_count(modulation, expected):
    points, count = preamble_symbols(modulation)
    assert count == expected
    assert points.size == expected


def test_correlate_symbols_finds_position_and_phase():
    points, _ = preamble_symbols("qpsk")
    rng = np.random.default_rng(0)
    noise = 0.3 * (rng.standard_normal(40) + 1j * rng.standard_normal(40))
    tail = 0.3 * (rng.standard_normal(60) + 1j * rng.standard_normal(60))
    y = np.concatenate([noise, points * np.exp(1j * np.pi / 2), tail])
    corr = correlate_symbols(y, points)
    peak = int(np.argmax(np.abs(corr)))
    assert peak == 40
    assert abs(corr[peak]) > 0.99
    assert abs(np.degrees(np.angle(corr[peak])) - 90.0) < 1e-6


def test_correlate_bits_sign_convention():
    bits = np.zeros(40, dtype=np.uint8)
    bits[5:18] = barker_bits(13)
    corr = correlate_bits(bits)
    assert int(np.argmax(corr)) == 5
    assert abs(corr[5] - 1.0) < 1e-9
    inverted = bits.copy()
    inverted[5:18] = 1 - inverted[5:18]
    assert abs(correlate_bits(inverted)[5] + 1.0) < 1e-9


@pytest.mark.parametrize("modulation", ("bpsk", "qpsk", "2fsk"))
def test_extract_frames_recovers_payload(modulation):
    signal, truth = _case(modulation, snr_db=30.0)
    injected = Signal(samples=fractional_delay(signal.samples, 0.35), sample_rate=signal.sample_rate)
    result = extract_frames(injected, modulation=modulation, truth=truth)
    summary = result.summary
    assert summary["frames"] >= 3
    if modulation in ("bpsk", "qpsk"):
        assert summary["frames_crc_ok"] == summary["frames"]
        assert summary["payload_bytes_exact"] == summary["payload_bytes_total"]
    assert result.locked


def test_extract_frames_resolves_phase_ambiguity_without_truth():
    """把整段信号转 90 度，M5 必须靠前导自己把模糊解出来（truth 只用于最后对账）"""
    signal, truth = _case("qpsk", snr_db=30.0)
    rotated = Signal(samples=signal.samples * np.exp(1j * np.pi / 2), sample_rate=signal.sample_rate)
    result = extract_frames(rotated, modulation="qpsk", truth=truth)
    assert result.summary["frames_crc_ok"] == result.summary["frames"] > 0
    assert abs(result.ambiguity_rotation_deg) % 90.0 < 1e-6


def test_extract_frames_to_dict_contract():
    signal, truth = _case("qpsk", snr_db=30.0)
    payload = extract_frames(signal, modulation="qpsk", truth=truth).to_dict()
    assert set(payload) == {"modulation", "modulation_source", "symbol_rate", "cfo_hz", "sync", "frames", "summary", "warnings", "evidence"}
    sync = payload["sync"]
    assert sync["preamble"] == "barker13"
    assert sync["preamble_bits"] == "".join(str(int(b)) for b in barker_bits(13))
    assert set(sync["correlation"]) == {"index", "value", "stride"}
    assert len(sync["correlation"]["index"]) == len(sync["correlation"]["value"])
    assert payload["frames"] and set(payload["frames"][0]) >= {"index", "start_symbol", "header", "crc", "payload"}
    assert payload["summary"]["frames"] >= 1
    # 契约要求：数值不能有 NaN/Infinity
    import json

    text = json.dumps(payload, ensure_ascii=False)
    assert "NaN" not in text and "Infinity" not in text


def test_extract_frames_short_signal_does_not_fabricate_frames():
    """记录过短时必须"明说没帧"，而不是硬凑出一帧."""
    signal, _ = _case("qpsk", snr_db=30.0)
    clipped = Signal(samples=signal.samples[:600], sample_rate=signal.sample_rate)
    try:
        result = extract_frames(clipped, modulation="qpsk")
    except ValueError:
        return          # 明确报错同样可接受
    assert result.summary["frames"] == 0
    assert result.warnings
