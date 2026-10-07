"""盲 SNR 估计测试."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.estimate import estimate_snr, estimate_snr_m2m4, estimate_snr_spectral
from spxh.core.types import MODULATIONS, Signal


@pytest.mark.parametrize("modulation", MODULATIONS)
@pytest.mark.parametrize("snr_db", [10.0, 20.0])
def test_spectral_snr_matches_truth_with_true_symbol_rate(synth_case, modulation, snr_db):
    signal, truth = synth_case(modulation=modulation, snr_db=snr_db)
    estimate = estimate_snr_spectral(signal, symbol_rate=truth.symbol_rate)
    assert abs(estimate.value - truth.snr_db) < 0.5, (modulation, snr_db, estimate.value)
    assert estimate.name == "es_n0"
    assert estimate.unit == "dB"


def test_spectral_snr_with_estimated_symbol_rate(synth_case):
    from spxh.core.estimate import estimate_symbol_rate

    signal, truth = synth_case(modulation="qpsk", snr_db=10.0)
    symbol_rate = estimate_symbol_rate(signal)
    estimate = estimate_snr(signal, symbol_rate=symbol_rate.value)
    assert abs(estimate.value - truth.snr_db) < 0.6
    assert "m2m4" in estimate.evidence
    assert estimate.evidence["sps_used"] == pytest.approx(truth.sps, rel=0.01)


def test_spectral_snr_without_symbol_rate_returns_waveform_snr(synth_case):
    signal, truth = synth_case(modulation="qpsk", snr_db=20.0)
    estimate = estimate_snr_spectral(signal)
    assert estimate.name == "snr_waveform"
    # 波形 SNR 与 Es/N0 相差 10*log10(sps)
    assert estimate.value == pytest.approx(truth.snr_db - 10.0 * np.log10(truth.sps), abs=0.5)


def test_m2m4_is_accurate_for_constant_modulus_fsk(synth_case):
    signal, truth = synth_case(modulation="2fsk", snr_db=20.0)
    result = estimate_snr_m2m4(signal)
    assert result["valid"] is True
    waveform_snr = result["snr_waveform_db"]
    expected = truth.snr_db - 10.0 * np.log10(truth.sps)
    assert abs(waveform_snr - expected) < 1.5


def test_m2m4_is_biased_for_shaped_qam(synth_case):
    """成型信号的包络起伏会让 M2M4 偏低 —— 这是已知偏差，记录在证据里而不是参与融合."""
    signal, truth = synth_case(modulation="16qam", snr_db=20.0)
    result = estimate_snr_m2m4(signal)
    assert result["valid"] is True
    expected = truth.snr_db - 10.0 * np.log10(truth.sps)
    assert result["snr_waveform_db"] < expected


def test_snr_of_pure_noise_is_very_low():
    rng = np.random.default_rng(23)
    noise = (rng.standard_normal(60000) + 1j * rng.standard_normal(60000)) / np.sqrt(2)
    estimate = estimate_snr_spectral(Signal(samples=noise, sample_rate=8000.0), symbol_rate=1000.0)
    assert estimate.value < 3.0


def test_snr_requires_input():
    with pytest.raises(ValueError):
        estimate_snr_spectral()
