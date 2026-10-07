"""M3 接收链测试：插值、定时环、载波环、LLR、FSK、整链 BER."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.demod import demodulate
from spxh.core.demod.filters import FarrowInterpolator, agc_normalize, fractional_delay
from spxh.core.demod.fsk import fsk_demodulate
from spxh.core.demod.llr import max_log_llr, rotation_symmetry_order, symbol_noise_variance
from spxh.core.demod.receiver import evaluate_against_truth
from spxh.core.demod.sync import CarrierRecovery, LoopConfig, TimingRecovery
from spxh.core.synth import SynthConfig, mapping, synthesize
from spxh.core.synth.pulse import matched_filter, rrc_taps
from spxh.core.types import MODULATIONS, Signal

SPS = 8
SYMBOL_RATE = 1000.0


def _case(modulation: str, snr_db: float = 20.0, frames: int = 8, payload: int = 128, cfo_hz: float = 37.0, seed: int = 11):
    return synthesize(
        SynthConfig(
            modulation=modulation,
            symbol_rate=SYMBOL_RATE,
            sps=SPS,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            num_frames=frames,
            payload_len=payload,
            seed=seed,
            keep_clean=True,
        )
    )


def test_farrow_interpolates_complex_exponential():
    """三次拉格朗日插值的误差是 O((2*pi*f)^4)，0.037 周/样本时约 3e-5 —— 远低于噪声底."""
    n = np.arange(200)
    x = np.exp(2j * np.pi * 0.037 * n)
    interpolator = FarrowInterpolator(x)
    for position in (10.25, 20.5, 50.75, 100.1):
        assert abs(interpolator(position) - np.exp(2j * np.pi * 0.037 * position)) < 1e-4
    low = np.exp(2j * np.pi * 0.005 * n)
    low_interp = FarrowInterpolator(low)
    assert abs(low_interp(30.5) - np.exp(2j * np.pi * 0.005 * 30.5)) < 1e-7


def test_fractional_delay_changes_phase_but_not_power():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)) / np.sqrt(2)
    delayed = fractional_delay(x, 0.35)
    assert delayed.size == x.size
    assert abs(float(np.mean(np.abs(delayed) ** 2)) - float(np.mean(np.abs(x) ** 2))) < 0.05


def test_timing_loop_locks_with_injected_delay():
    # 这条只测定时环，所以不加频偏（否则星座在旋转，EVM 无意义）
    signal, truth = _case("qpsk", snr_db=20.0, cfo_hz=0.0)
    injected = fractional_delay(signal.samples, 0.35)
    taps = rrc_taps(SPS, span_symbols=16, rolloff=0.35)
    filtered = agc_normalize(matched_filter(injected, taps))[0]
    delay = (taps.size - 1) / 2.0
    result = TimingRecovery(float(SPS), LoopConfig()).run(
        filtered, start_position=TimingRecovery.default_start_position(float(SPS), filter_delay=delay)
    )
    assert result.metric > 0.5, result.metric
    indices = mapping.points_to_indices(result.symbols, "qpsk")
    ideal = mapping.indices_to_points(indices, "qpsk")
    evm = 100.0 * np.sqrt(np.mean(np.abs(result.symbols - ideal) ** 2)) / np.sqrt(np.mean(np.abs(ideal) ** 2))
    # 20 dB 的纯噪声 EVM 约 10%，定时环锁定后不应显著超过它
    assert evm < 20.0, evm


def test_carrier_loop_tracks_residual_frequency():
    signal, _ = _case("qpsk", snr_db=20.0, cfo_hz=0.0)
    # 故意留下 +12 Hz 残余频偏
    fs = float(signal.sample_rate)
    n = np.arange(signal.samples.size)
    residual = signal.samples * np.exp(2j * np.pi * 12.0 * n / fs)
    taps = rrc_taps(SPS, span_symbols=16, rolloff=0.35)
    filtered = agc_normalize(matched_filter(residual, taps))[0]
    delay = (taps.size - 1) / 2.0
    timing = TimingRecovery(float(SPS)).run(
        filtered, start_position=TimingRecovery.default_start_position(float(SPS), filter_delay=delay)
    )
    carrier = CarrierRecovery("qpsk", SYMBOL_RATE).run(timing.symbols)
    assert carrier.metric > 0.5, carrier.metric
    # freq_estimates 是环跟踪到的频偏（与输入残余同号），应当收敛到 +12 Hz 附近
    tail = carrier.freq_estimates[-carrier.freq_estimates.size // 4 :]
    assert float(np.mean(tail)) > 5.0, float(np.mean(tail))
    assert abs(float(np.mean(tail)) - 12.0) < 3.0, float(np.mean(tail))


@pytest.mark.parametrize("modulation", ("bpsk", "qpsk", "2fsk", "4fsk"))
def test_demodulate_zero_ber_at_20db(modulation):
    signal, truth = _case(modulation, snr_db=20.0)
    injected = Signal(samples=fractional_delay(signal.samples, 0.35), sample_rate=signal.sample_rate)
    result = demodulate(injected, modulation=modulation, truth=truth)
    assert result.ber is not None
    assert result.ber["bit_error_rate"] <= 1e-3, (modulation, result.ber)


@pytest.mark.parametrize("modulation", ("8psk", "16qam"))
def test_demodulate_low_ber_at_20db_for_higher_orders(modulation):
    signal, truth = _case(modulation, snr_db=20.0)
    injected = Signal(samples=fractional_delay(signal.samples, 0.35), sample_rate=signal.sample_rate)
    result = demodulate(injected, modulation=modulation, truth=truth)
    assert result.ber is not None
    # 更高阶星座对残余定时/相位抖动更敏感，20 dB 下留出更宽的容差（见 M3 报告"已知局限"）
    assert result.ber["bit_error_rate"] <= 0.05, (modulation, result.ber)


def test_demodulate_reports_lock_and_traces():
    signal, truth = _case("qpsk", snr_db=20.0)
    result = demodulate(signal, modulation="qpsk", truth=truth)
    payload = result.to_dict(max_symbols=100, trace_points=50)
    assert payload["lock"]["timing"] is True
    assert payload["lock"]["carrier"] is True
    assert len(payload["symbols"]["i"]) == min(100, result.num_symbols)
    assert set(payload["traces"]) == {"index", "timing_error", "phase_error_rad", "freq_estimate_hz"}
    assert len(payload["traces"]["index"]) <= 50
    assert payload["bits"]["count"] == result.num_bits
    assert isinstance(payload["llr"]["mean_abs"], float)


def test_llr_sign_matches_hard_decision():
    signal, _ = _case("qpsk", snr_db=20.0)
    result = demodulate(signal, modulation="qpsk")
    hard = result.bits.astype(np.int64)
    signs = (result.llr < 0).astype(np.int64)
    assert np.array_equal(signs, hard)


def test_llr_magnitude_grows_with_snr():
    means = []
    for snr_db in (5.0, 15.0):
        signal, _ = _case("qpsk", snr_db=snr_db)
        result = demodulate(signal, modulation="qpsk")
        means.append(float(np.mean(np.abs(result.llr))))
    assert means[0] < means[1]


def test_symbol_noise_variance_and_rotation_order():
    assert abs(symbol_noise_variance(10.0) - 0.1) < 1e-12
    assert rotation_symmetry_order("bpsk") == 2
    assert rotation_symmetry_order("qpsk") == 4
    assert rotation_symmetry_order("8psk") == 8
    assert rotation_symmetry_order("16qam") == 4


def test_max_log_llr_is_consistent_with_nearest_point():
    constellation = mapping.constellation("qpsk")
    sigma2 = symbol_noise_variance(20.0)
    llr, bits, indices = max_log_llr(constellation, "qpsk", sigma2)
    # 无噪声时每个星座点的硬判决应回到自己，且 LLR 符号与比特一致
    assert np.array_equal(indices, np.arange(4))
    expected = mapping.indices_to_bits(np.arange(4), "qpsk")
    assert np.array_equal(bits, expected)
    assert np.all((llr < 0) == (expected == 1))


def test_evaluate_against_truth_handles_rotation_ambiguity():
    signal, truth = _case("qpsk", snr_db=25.0)
    result = demodulate(signal, modulation="qpsk")
    rotated = result.symbols * np.exp(1j * np.pi / 2)   # QPSK 的 90 度模糊
    metrics = evaluate_against_truth(rotated, "qpsk", truth, symbol_noise_variance(25.0))
    assert metrics is not None
    assert metrics["bit_error_rate"] <= 1e-3
    assert abs(metrics["ambiguity_rotation_deg"] - 270.0) < 1e-6 or abs(metrics["ambiguity_rotation_deg"] - 90.0) < 1e-6


def test_fsk_demodulate_recovers_tones():
    for modulation, expected_symbols in (("2fsk", 2), ("4fsk", 4)):
        signal, truth = _case(modulation, snr_db=30.0)
        result = fsk_demodulate(signal, symbol_rate=SYMBOL_RATE, cfo_hz=37.0, modulation=modulation)
        assert result["indices"].size == truth.num_symbols
        assert len(result["tones_hz"]) == expected_symbols


def test_demodulate_requires_valid_symbol_rate():
    signal, _ = _case("qpsk", snr_db=20.0)
    from spxh.core.estimate import AnalyzeResult, Estimate

    analyze = AnalyzeResult(
        sample_rate=signal.sample_rate,
        num_samples=signal.num_samples,
        duration_s=signal.duration,
        noise_floor_psd=1e-6,
        total_power=1e-2,
        peak={},
        occupied={},
        cfo=Estimate("cfo", 0.0, "Hz"),
        symbol_rate=Estimate("symbol_rate", -1.0, "Hz"),
        snr=Estimate("es_n0", 10.0, "dB"),
    )
    with pytest.raises(ValueError):
        demodulate(signal, analysis=analyze, modulation="qpsk")
