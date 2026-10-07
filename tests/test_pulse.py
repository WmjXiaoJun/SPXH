"""RRC 成型/匹配滤波测试."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.synth.pulse import (
    filter_delay,
    matched_filter,
    pulse_shape,
    rrc_taps,
    symbol_sample_indices,
    upsample,
)


@pytest.mark.parametrize("rolloff", [0.0, 0.22, 0.35, 0.5, 1.0])
def test_rrc_taps_unit_energy_and_symmetry(rolloff):
    taps = rrc_taps(8, span_symbols=8, rolloff=rolloff)
    assert taps.size == 8 * 8 + 1
    assert abs(float(np.sum(taps ** 2)) - 1.0) < 1e-12
    assert np.allclose(taps, taps[::-1], atol=1e-12)


def test_rrc_peak_at_center():
    taps = rrc_taps(4, span_symbols=6, rolloff=0.35)
    center = taps.size // 2
    assert center == int(np.argmax(taps))


def test_upsample_places_symbols_on_grid():
    symbols = np.array([1 + 1j, -1 + 0j, 0 + 1j])
    out = upsample(symbols, 4)
    assert out.size == 12
    assert np.array_equal(out[::4], symbols)
    assert np.count_nonzero(out) == 3


def test_pulse_shape_preserves_single_symbol_energy_exactly():
    single = np.zeros(64, dtype=np.complex128)
    single[32] = 1 + 1j
    waveform, _ = pulse_shape(single, 8, rolloff=0.35, span_symbols=16)
    assert abs(float(np.sum(np.abs(waveform) ** 2)) - 2.0) < 1e-9


def test_pulse_shape_multi_symbol_energy_is_approximately_preserved():
    # 多符号时总能量含符号间相关项，只能近似守恒（有限长 RRC 的固有性质）
    rng = np.random.default_rng(4)
    symbols = (rng.standard_normal(64) + 1j * rng.standard_normal(64)) / np.sqrt(2)
    waveform, delay = pulse_shape(symbols, 8, rolloff=0.35, span_symbols=16)
    energy = float(np.sum(np.abs(waveform) ** 2))
    reference = float(np.sum(np.abs(symbols) ** 2))
    assert abs(energy - reference) / reference < 0.01
    assert delay == filter_delay(rrc_taps(8, span_symbols=16, rolloff=0.35))


def test_matched_filter_recovers_symbols_without_noise():
    rng = np.random.default_rng(5)
    from spxh.core.synth import mapping

    indices = rng.integers(0, 16, size=200)
    points = mapping.indices_to_points(indices, "16qam")
    taps = rrc_taps(8, span_symbols=16, rolloff=0.35)
    waveform, delay = pulse_shape(points, 8, taps=taps)
    received = matched_filter(waveform, taps)
    positions = symbol_sample_indices(points.size, 8, delay)
    recovered = received[positions]
    # 有限长 RRC 存在约 0.3% RMS 的截断 ISI（非 bug），判决必须全部正确
    assert np.max(np.abs(recovered - points)) < 0.02
    assert np.array_equal(mapping.points_to_indices(recovered, "16qam"), indices)
