"""符号速率估计测试."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.estimate.symbol_rate import (
    cyclic_profile,
    detect_peaks,
    envelope_profile,
    estimate_symbol_rate,
    select_fundamental,
)
from spxh.core.types import MODULATIONS, Signal


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_symbol_rate_accuracy_for_all_modulations(synth_case, modulation):
    signal, truth = synth_case(modulation=modulation, snr_db=20.0, symbol_rate=1234.5)
    estimate = estimate_symbol_rate(signal)
    relative = abs(estimate.value - truth.symbol_rate) / truth.symbol_rate
    assert relative < 2e-3, (modulation, estimate.value, truth.symbol_rate, estimate.evidence)
    assert estimate.confidence > 0.2


@pytest.mark.parametrize("modulation", ("qpsk", "16qam", "2fsk", "4fsk"))
def test_symbol_rate_accuracy_at_10db(synth_case, modulation):
    signal, truth = synth_case(modulation=modulation, snr_db=10.0, symbol_rate=977.3)
    estimate = estimate_symbol_rate(signal)
    relative = abs(estimate.value - truth.symbol_rate) / truth.symbol_rate
    if modulation == "4fsk":
        # 已知局限：4FSK(h=0.5) 在 10 dB（波形 SNR 仅约 1 dB）时符号速率循环特征只有 ~5 dB，
        # 低于噪声伪峰。此时必须"知道自己在猜"：要么估准，要么置信度低到下游不会采信。
        assert relative < 5e-3 or estimate.confidence < 0.35, (estimate.value, estimate.confidence)
    else:
        assert relative < 5e-3, (modulation, estimate.value, truth.symbol_rate)


def test_low_snr_estimates_are_flagged_not_silently_wrong(synth_case):
    signal, truth = synth_case(modulation="16qam", snr_db=0.0, num_frames=4, payload_len=64)
    estimate = estimate_symbol_rate(signal)
    relative = abs(estimate.value - truth.symbol_rate) / truth.symbol_rate
    assert relative < 5e-3 or estimate.confidence < 0.35


def test_cyclic_profile_has_line_at_symbol_rate(synth_case):
    signal, truth = synth_case(modulation="qpsk", snr_db=20.0)
    alpha, profile = cyclic_profile(signal.samples, signal.sample_rate)
    peaks = detect_peaks(alpha, profile, 50.0, 0.45 * signal.sample_rate, min_prominence_db=8.0)
    frequencies = np.asarray([p["frequency"] for p in peaks])
    assert np.any(np.abs(frequencies - truth.symbol_rate) < 0.01 * truth.symbol_rate)


def test_envelope_profile_is_weak_for_constant_modulus_fsk(synth_case):
    """FSK 恒模、没有包络起伏：包络法的谱线强度应显著低于循环法."""
    signal, truth = synth_case(modulation="2fsk", snr_db=20.0)
    alpha_c, profile_c = cyclic_profile(signal.samples, signal.sample_rate)
    alpha_e, profile_e = envelope_profile(signal.samples, signal.sample_rate)
    peaks_c = detect_peaks(alpha_c, profile_c, 50.0, 0.45 * signal.sample_rate)
    peaks_e = detect_peaks(alpha_e, profile_e, 50.0, 0.45 * signal.sample_rate)
    best_c = max((p["level_db"] for p in peaks_c), default=float("-inf"))
    best_e = max((p["level_db"] for p in peaks_e), default=float("-inf"))
    assert best_c > best_e


def test_select_fundamental_prefers_true_fundamental_over_harmonic():
    peaks = [
        {"frequency": 500.0, "alpha": 500.0, "level_db": 20.0, "prominence_db": 20.0},
        {"frequency": 1000.0, "alpha": 1000.0, "level_db": 22.0, "prominence_db": 22.0},
        {"frequency": 1500.0, "alpha": 1500.0, "level_db": 14.0, "prominence_db": 14.0},
    ]
    best = select_fundamental(peaks, harmonics=4)
    assert abs(best["frequency"] - 500.0) < 1.0
    assert best["num_harmonics"] >= 3


def test_select_fundamental_does_not_over_subdivide():
    peaks = [
        {"frequency": 1000.0, "alpha": 1000.0, "level_db": 30.0, "prominence_db": 30.0},
        {"frequency": 250.0, "alpha": 250.0, "level_db": 24.0, "prominence_db": 24.0},
    ]
    best = select_fundamental(peaks, harmonics=4)
    assert abs(best["frequency"] - 1000.0) < 1.0


def test_estimate_symbol_rate_reports_low_confidence_on_noise():
    rng = np.random.default_rng(17)
    noise = (rng.standard_normal(40000) + 1j * rng.standard_normal(40000)) / np.sqrt(2)
    estimate = estimate_symbol_rate(samples=noise, sample_rate=8000.0)
    assert estimate.confidence < 0.35


def test_estimate_symbol_rate_requires_input():
    with pytest.raises(ValueError):
        estimate_symbol_rate()
    with pytest.raises(ValueError):
        estimate_symbol_rate(samples=np.zeros(1024, dtype=np.complex128))
