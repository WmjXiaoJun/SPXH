"""M12-C 测试：突发检测（塌陷/压制）+ 自动擦除的"不帮倒忙"性质."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.dsp.burst import detect_bursts, detect_bursts_dict
from spxh.core.framesync import extract_frames
from spxh.core.synth import SynthConfig, synthesize
from spxh.core.types import Signal


def _signal(snr_db: float = 20.0, frames: int = 6, fec: str = "rs"):
    return synthesize(
        SynthConfig(modulation="qpsk", symbol_rate=1000.0, sps=8, snr_db=snr_db, cfo_hz=37.0,
                    num_frames=frames, payload_len=60, fec=fec, seed=11, keep_clean=True)
    )


def test_clean_and_noisy_signals_have_no_bursts():
    """干净信号与纯噪声背景都不能虚报 —— 虚报会白白吃掉纠错能力."""
    signal, _ = _signal()
    assert detect_bursts(signal.samples, sample_rate=signal.sample_rate, sps=8.0) == []
    samples = np.asarray(signal.samples, dtype=np.complex128)
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(samples.size) + 1j * rng.standard_normal(samples.size)) * 0.05
    assert detect_bursts(samples + noise, sample_rate=signal.sample_rate, sps=8.0) == []


def test_blanking_is_detected_as_collapse():
    signal, _ = _signal()
    samples = np.asarray(signal.samples, dtype=np.complex128).copy()
    center, half = samples.size // 2, 12 * 8 // 2
    samples[center - half: center + half] = 0.0
    regions = detect_bursts(samples, sample_rate=signal.sample_rate, sps=8.0)
    assert len(regions) >= 1
    hit = min(regions, key=lambda r: abs(r.start_sample - (center - half)))
    assert hit.kind == "collapse"
    assert hit.depth_db < -40.0                     # 置零应该是极深的塌陷
    assert hit.start_sample <= center - half <= hit.end_sample   # 位置要覆盖真实区间
    assert hit.confidence > 0.5


def test_interference_is_detected_as_jam():
    """压制（功率抬升）是另一类突发：只找塌陷的检测器会完全漏掉它."""
    signal, _ = _signal()
    samples = np.asarray(signal.samples, dtype=np.complex128).copy()
    center, half = samples.size // 2, 12 * 8 // 2
    span = slice(center - half, center + half)
    local = float(np.sqrt(np.mean(np.abs(samples[span]) ** 2))) or 1.0
    rng = np.random.default_rng(17)
    samples[span] = (rng.standard_normal(span.stop - span.start)
                     + 1j * rng.standard_normal(span.stop - span.start)) * local * 4.0
    regions = detect_bursts(samples, sample_rate=signal.sample_rate, sps=8.0)
    kinds = {region.kind for region in regions}
    assert "jam" in kinds
    jam = next(region for region in regions if region.kind == "jam")
    assert jam.depth_db > 3.0                       # 抬升量（正 dB）


def test_report_dict_shape():
    signal, _ = _signal()
    payload = detect_bursts_dict(signal.samples, sample_rate=signal.sample_rate, sps=8.0)
    assert payload["burst_count"] == 0
    assert payload["bursts"] == []
    assert "rule" in payload


def test_extract_with_auto_erasures_never_worse_than_plain():
    """核心性质：自动擦除（CRC 仲裁 + 预算含 0）**不会比不标擦除更差**."""
    signal, truth = _signal(snr_db=9.0, frames=6)
    samples = np.asarray(signal.samples, dtype=np.complex128).copy()
    center, half = samples.size // 2, 20 * 8 // 2
    span = slice(center - half, center + half)
    local = float(np.sqrt(np.mean(np.abs(samples[span]) ** 2))) or 1.0
    rng = np.random.default_rng(17)
    samples[span] = (rng.standard_normal(span.stop - span.start)
                     + 1j * rng.standard_normal(span.stop - span.start)) * local * 3.0
    injected = Signal(samples=samples, sample_rate=signal.sample_rate)

    plain = extract_frames(injected, modulation="qpsk", truth=truth, blind_fec=True)
    auto = extract_frames(injected, modulation="qpsk", truth=truth, blind_fec=True,
                          erasure_decoding=True, auto_erasures=True)
    assert auto.summary["frames_crc_ok"] >= plain.summary["frames_crc_ok"]
    # 检测到的突发要写进证据表，前端/代理才看得到
    assert isinstance(auto.evidence.get("burst_ranges"), list)
    assert auto.evidence.get("auto_erasures") is True


def test_auto_erasures_do_not_break_clean_links():
    """干净链路开自动擦除也不该有任何变化（没有突发就不标擦除）."""
    signal, truth = _signal(snr_db=20.0, frames=6)
    injected = Signal(samples=signal.samples, sample_rate=signal.sample_rate)
    plain = extract_frames(injected, modulation="qpsk", truth=truth, blind_fec=True)
    auto = extract_frames(injected, modulation="qpsk", truth=truth, blind_fec=True,
                          erasure_decoding=True, auto_erasures=True)
    assert auto.summary["frames"] == plain.summary["frames"]
    assert auto.summary["frames_crc_ok"] == plain.summary["frames_crc_ok"]
    assert auto.evidence.get("burst_ranges") == []


def test_timing_loop_freezes_inside_burst_region():
    """突发区间内定时环**不更新**（滑行），出突发后仍应锁定；区间外行为不变."""
    from spxh.core.demod.sync import TimingRecovery, LoopConfig

    rng = np.random.default_rng(3)
    n = 4000
    sps = 8.0
    # 造一段恒模信号（QPSK 符号），中间挖一个洞
    symbols = (rng.integers(0, 4, size=n // int(sps)) * 2 + 1).astype(np.complex128)
    symbols = np.exp(1j * np.pi * (symbols.real - 1) / 2)
    samples = np.repeat(symbols, int(sps))
    hole = slice(1500, 1700)
    damaged = samples.copy()
    damaged[hole] = 0.0

    loop = TimingRecovery(sps, LoopConfig())
    plain = loop.run(damaged, start_position=TimingRecovery.default_start_position(sps))
    frozen = loop.run(damaged, start_position=TimingRecovery.default_start_position(sps),
                      freeze_regions=[(hole.start, hole.stop)])
    # 两者输出的符号数应一致；冻结版在洞内不更新 mu，因此洞外的定时误差轨迹更平稳
    assert frozen.symbols.size == plain.symbols.size
    assert frozen.symbol_positions.size == frozen.symbols.size
    tail = slice(int(frozen.timing_errors.size * 0.8), None)
    assert float(np.std(frozen.timing_errors[tail])) <= float(np.std(plain.timing_errors[tail])) + 1e-6


def test_carrier_loop_coasts_on_freeze_mask():
    """载波环在冻结符号上只滑行（不更新相位/频率），相位轨迹连续."""
    from spxh.core.demod.sync import CarrierRecovery, LoopConfig

    rng = np.random.default_rng(5)
    symbols = np.exp(1j * (np.pi / 4 + rng.integers(0, 4, size=400) * np.pi / 2)) * (
        0.9 + 0.1 * rng.random(400))
    mask = np.zeros(400, dtype=bool)
    mask[150:180] = True
    loop = CarrierRecovery("qpsk", 1000.0, LoopConfig())
    plain = loop.run(symbols)
    coased = loop.run(symbols, freeze_mask=mask)
    assert coased.symbols.size == plain.symbols.size
    # 冻结段的相位误差记为 0（明确表示"这段不可信、环路没更新"）
    assert float(np.max(np.abs(coased.phase_errors[150:180]))) == 0.0
    # 冻结期间频率保持不变
    assert float(np.std(coased.freq_estimates[151:180])) < 1e-9


def test_demodulate_blanks_and_freezes_burst():
    """demodulate 收到突发区间时要真的置零并冻结（证据表里能查到）."""
    from spxh.core.demod import demodulate

    signal, _ = _signal(snr_db=20.0, frames=6)
    samples = np.asarray(signal.samples, dtype=np.complex128).copy()
    center, half = samples.size // 2, 10 * 8 // 2
    span = slice(center - half, center + half)
    local = float(np.sqrt(np.mean(np.abs(samples[span]) ** 2))) or 1.0
    rng = np.random.default_rng(11)
    samples[span] = (rng.standard_normal(span.stop - span.start)
                     + 1j * rng.standard_normal(span.stop - span.start)) * local * 4.0
    regions = [(center - half - 8, center + half + 8)]
    result = demodulate(Signal(samples=samples, sample_rate=signal.sample_rate),
                        modulation="qpsk", burst_regions=regions, blank_bursts=True)
    assert result.evidence.get("blanked_samples", 0) > 0
    assert result.evidence.get("frozen_symbols", 0) > 0
    # 关闭置零时不应改动样本（对照）
    untouched = demodulate(Signal(samples=samples, sample_rate=signal.sample_rate),
                           modulation="qpsk", burst_regions=regions, blank_bursts=False)
    assert untouched.evidence.get("blanked_samples", 0) == 0
