"""DSP 基础层测试：Welch PSD、噪声底、谱峰、占用带宽、STFT."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.dsp import stft_waterfall, welch_psd
from spxh.core.dsp.spectrum import occupied_bandwidth, parabolic_peak, psd_peak, smooth_psd
from spxh.core.types import Signal


def test_psd_total_power_matches_waveform_power(synth_case):
    signal, _ = synth_case()
    psd = welch_psd(signal, nperseg=1024)
    assert psd.total_power() == pytest.approx(signal.power, rel=0.02)


def test_psd_frequency_axis_is_two_sided_and_ascending(synth_case):
    signal, _ = synth_case()
    psd = welch_psd(signal, nperseg=1024, nfft=8192)
    assert psd.freqs[0] == pytest.approx(-signal.sample_rate / 2, abs=1e-9)
    assert np.all(np.diff(psd.freqs) > 0)
    assert psd.df == pytest.approx(signal.sample_rate / 8192)


def test_noise_floor_of_white_noise_matches_density():
    rng = np.random.default_rng(3)
    sigma2 = 0.01
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(200000) + 1j * rng.standard_normal(200000))
    signal = Signal(samples=noise, sample_rate=8000.0)
    psd = welch_psd(signal, nperseg=1024)
    expected = sigma2 / signal.sample_rate
    assert psd.noise_floor() == pytest.approx(expected, rel=0.1)


def test_psd_peak_finds_tone_with_sub_bin_accuracy():
    fs = 8000.0
    n = 40000
    f0 = 321.7
    t = np.arange(n) / fs
    tone = np.exp(2j * np.pi * f0 * t)
    psd = welch_psd(Signal(samples=tone, sample_rate=fs), nperseg=1024, nfft=8192)
    peak = psd.peak()
    assert abs(peak["frequency"] - f0) < 0.2


def test_parabolic_peak_interpolates_symmetric_peak():
    x = np.arange(11, dtype=np.float64)
    y = np.exp(-((x - 5.4) ** 2) / 2.0)
    value, peak = parabolic_peak(x, y, 5)
    assert abs(value - 5.4) < 0.05
    assert peak > 0


def test_occupied_bandwidth_is_stable_across_snr(synth_case):
    widths = []
    for snr in (20.0, 10.0):
        signal, truth = synth_case(snr_db=snr)
        psd = welch_psd(signal, nperseg=1024, nfft=8192)
        band = psd.occupied_band(noise_floor=psd.noise_floor())
        widths.append(band["bandwidth"])
        expected = truth.symbol_rate * (1.0 + truth.rolloff)
        assert 0.5 * expected < band["bandwidth"] < 1.5 * expected
    assert abs(widths[0] - widths[1]) / widths[0] < 0.3


def test_occupied_bandwidth_uses_noise_floor_to_reject_noise():
    rng = np.random.default_rng(5)
    noise = (rng.standard_normal(40000) + 1j * rng.standard_normal(40000)) / np.sqrt(2)
    fs = 8000.0
    tone = np.exp(2j * np.pi * 500.0 * np.arange(40000) / fs)
    signal = Signal(samples=tone + 0.5 * noise, sample_rate=fs)
    psd = welch_psd(signal, nperseg=1024, nfft=8192)
    band = psd.occupied_band(noise_floor=psd.noise_floor(), fraction=0.6)
    assert band["bandwidth"] < 200.0


def test_band_power_sums_density():
    freqs = np.linspace(-4000, 4000, 4001, endpoint=False)
    psd = np.ones_like(freqs) * 1e-6
    from spxh.core.dsp.spectrum import band_power

    power = band_power(freqs, psd, -1000.0, 1000.0)
    # 2000 Hz 带宽、密度 1e-6 功率/Hz -> 总功率 2e-3
    assert power == pytest.approx(2e-3, rel=0.02)


def test_smooth_psd_keeps_shape_and_length():
    arr = np.arange(50, dtype=np.float64)
    smoothed = smooth_psd(arr, 5)
    assert smoothed.size == arr.size
    assert np.all(np.diff(smoothed) >= -1e-9)


def test_stft_waterfall_shape_and_time_tracking(synth_case):
    signal, _ = synth_case(num_frames=4, payload_len=64)
    waterfall = stft_waterfall(signal, nperseg=256)
    assert waterfall.shape[0] == waterfall.freqs.size
    assert waterfall.magnitude_db.shape == waterfall.shape
    assert waterfall.times[0] >= 0.0
    assert waterfall.summary()["duration_s"] > 0.0
