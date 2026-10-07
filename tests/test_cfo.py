"""载频估计测试."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.dsp import welch_psd
from spxh.core.estimate import estimate_cfo
from spxh.core.types import Signal, MODULATIONS


def _estimate(signal, nfft=8192):
    psd = welch_psd(signal, nperseg=1024, nfft=nfft)
    return estimate_cfo(psd=psd)


def test_cfo_of_pure_tone():
    fs = 8000.0
    f0 = -417.3
    t = np.arange(60000) / fs
    tone = np.exp(2j * np.pi * f0 * t)
    estimate = _estimate(Signal(samples=tone, sample_rate=fs))
    assert abs(estimate.value - f0) < 1.0
    assert estimate.confidence > 0.6


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_cfo_accuracy_for_all_modulations(synth_case, modulation):
    signal, truth = synth_case(modulation=modulation, snr_db=20.0, symbol_rate=1000.0, cfo_hz=37.0)
    estimate = _estimate(signal)
    assert abs(estimate.value - truth.cfo_hz) <= 0.015 * truth.symbol_rate
    assert 0.0 <= estimate.confidence <= 1.0


def test_cfo_evidence_records_candidates(synth_case):
    signal, truth = synth_case(cfo_hz=-215.5)
    estimate = _estimate(signal)
    candidates = estimate.evidence["candidates"]
    assert set(candidates) == {"centroid", "symmetry", "edge_midpoint"}
    assert abs(estimate.value - truth.cfo_hz) <= 0.02 * truth.symbol_rate


def test_cfo_confidence_drops_on_noise_only():
    rng = np.random.default_rng(11)
    noise = (rng.standard_normal(30000) + 1j * rng.standard_normal(30000)) / np.sqrt(2)
    estimate = _estimate(Signal(samples=noise, sample_rate=8000.0))
    assert estimate.confidence < 0.5


def test_cfo_requires_signal_or_psd():
    with pytest.raises(ValueError):
        estimate_cfo()
