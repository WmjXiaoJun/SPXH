"""端到端合成测试：参考接收端零误码、真值一致性、可复现性."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.framing import parse_frame
from spxh.core.synth import SynthConfig, synthesize
from spxh.core.types import MODULATIONS, GroundTruth
from spxh.tools.reference_rx import ber, evm_percent, receive

HIGH_SNR_MODS = MODULATIONS


@pytest.mark.parametrize("modulation", HIGH_SNR_MODS)
def test_reference_receiver_has_zero_errors_at_high_snr(modulation):
    signal, truth = synthesize(
        SynthConfig(modulation=modulation, snr_db=30.0, num_frames=2, payload_len=32, seed=2026)
    )
    received = receive(signal, truth)
    assert received.size >= truth.bits.size
    assert ber(truth.bits, received) == 0.0


@pytest.mark.parametrize("modulation", HIGH_SNR_MODS)
def test_frames_are_contiguous_and_crc_valid(modulation):
    signal, truth = synthesize(SynthConfig(modulation=modulation, snr_db=30.0, num_frames=3, payload_len=24))
    offset = 0
    for frame in truth.frames:
        assert frame.start_bit == offset
        assert frame.start_symbol == offset // truth.extras["bits_per_symbol"]
        result = parse_frame(truth.bits[offset:])
        assert result["ok"] is True
        assert result["payload"] == frame.payload
        assert result["crc_actual"] == frame.crc
        offset += frame.num_bits
    assert offset <= truth.num_bits
    assert truth.num_bits - offset == truth.pad_bits


def test_bit_and_symbol_counts_are_consistent():
    signal, truth = synthesize(SynthConfig(modulation="16qam", snr_db=25.0, num_frames=2, payload_len=40))
    bps = 4
    assert truth.num_bits % bps == 0
    assert truth.num_symbols == truth.num_bits // bps
    assert truth.bits.size == truth.num_bits
    assert truth.symbol_indices.size == truth.num_symbols
    assert truth.symbols is not None and truth.symbols.size == truth.num_symbols
    assert signal.num_samples == truth.num_symbols * truth.sps + 2 * int(truth.filter_delay_samples)


def test_fsk_truth_exposes_tones_without_constellation():
    signal, truth = synthesize(SynthConfig(modulation="4fsk", snr_db=25.0, num_frames=1, payload_len=8))
    assert truth.symbols is None
    assert truth.tone_hz is not None
    assert truth.tone_hz.size == truth.num_symbols
    expected = (np.arange(4) - 1.5) * 0.5 * truth.symbol_rate
    assert np.allclose(np.unique(np.round(truth.tone_hz, 6)), np.round(expected, 6))


def test_generation_is_reproducible():
    cfg = SynthConfig(modulation="qpsk", snr_db=12.0, seed=777, cfo_hz=3.0, phase_noise_rms_rad=0.02)
    a_signal, a_truth = synthesize(cfg)
    b_signal, b_truth = synthesize(cfg)
    assert np.array_equal(a_signal.samples, b_signal.samples)
    assert np.array_equal(a_truth.bits, b_truth.bits)
    assert np.array_equal(a_truth.clean, b_truth.clean)


def test_seed_changes_payloads_and_noise():
    a_signal, a_truth = synthesize(SynthConfig(modulation="qpsk", seed=1, num_frames=1, payload_len=16))
    b_signal, b_truth = synthesize(SynthConfig(modulation="qpsk", seed=2, num_frames=1, payload_len=16))
    assert a_truth.frames[0].payload != b_truth.frames[0].payload or not np.array_equal(a_signal.samples, b_signal.samples)


def test_noise_variance_matches_es_over_n0():
    for modulation, snr_db in (("qpsk", 10.0), ("2fsk", 10.0), ("16qam", 5.0)):
        signal, truth = synthesize(SynthConfig(modulation=modulation, snr_db=snr_db, num_frames=2, payload_len=64, seed=9))
        measured = float(np.mean(np.abs(signal.samples - truth.clean) ** 2))
        es = truth.extras["es_per_symbol"]
        assert abs(10.0 * np.log10(es / measured) - snr_db) < 0.15


def test_fsk_es_scales_with_sps():
    _, truth = synthesize(SynthConfig(modulation="2fsk", snr_db=10.0, sps=8, num_frames=1, payload_len=8))
    assert abs(truth.extras["es_per_symbol"] - 8.0) < 1e-9


def test_ber_decreases_with_snr():
    errors = []
    for snr in (0.0, 10.0, 20.0):
        signal, truth = synthesize(SynthConfig(modulation="16qam", snr_db=snr, num_frames=2, payload_len=128, seed=31))
        errors.append(ber(truth.bits, receive(signal, truth)))
    assert errors[0] > errors[1] > errors[2]


def test_evm_decreases_with_snr():
    evms = []
    for snr in (10.0, 20.0, 30.0):
        signal, truth = synthesize(SynthConfig(modulation="qpsk", snr_db=snr, num_frames=2, payload_len=64, seed=32))
        evms.append(evm_percent(signal, truth))
    assert evms[0] > evms[1] > evms[2]


def test_fec_and_interleave_are_now_supported():
    """M6 起 fec/interleave 真正接入帧结构（不再是 NotImplementedError）."""
    for fec in ("conv", "rs", "ccsds", "ldpc"):
        payload_len = 60 if fec == "ccsds" else (64 if fec == "ldpc" else 64)
        signal, truth = synthesize(
            SynthConfig(modulation="qpsk", fec=fec, payload_len=payload_len, num_frames=1, seed=3)
        )
        assert signal.num_samples > 0
        assert truth.frames[0].header.fec == fec
        assert truth.frames[0].num_bits > payload_len * 8   # 编码后帧更长
    _, truth_i = synthesize(SynthConfig(modulation="qpsk", interleave=True, payload_len=64, num_frames=1, seed=3))
    assert truth_i.frames[0].header.interleave is True


def test_ccsds_pads_to_interleave_depth():
    """CCSDS 交织深度要求整除，不足时补零；任意长度都应可用."""
    for payload_len in (60, 61, 63):
        _, truth = synthesize(SynthConfig(modulation="qpsk", fec="ccsds", payload_len=payload_len, num_frames=1, seed=3))
        assert truth.frames[0].header.payload_len == payload_len


def test_out_of_range_fec_payload_length_raises():
    """超出编码器能力时必须明确报错，而不是静默生成"声称已编码"的波形."""
    with pytest.raises(ValueError):
        synthesize(SynthConfig(modulation="qpsk", fec="ldpc", payload_len=200, num_frames=1))
    with pytest.raises(ValueError):
        synthesize(SynthConfig(modulation="qpsk", fec="ccsds", payload_len=1400, num_frames=1))


def test_ground_truth_save_load_roundtrip(tmp_path):
    signal, truth = synthesize(SynthConfig(modulation="8psk", snr_db=18.0, num_frames=2, payload_len=12, seed=5))
    paths = truth.save(tmp_path / "case")
    assert paths["truth_json"].endswith(".truth.json")
    loaded = GroundTruth.load(tmp_path / "case")
    assert np.array_equal(loaded.bits, truth.bits)
    assert np.array_equal(loaded.symbol_indices, truth.symbol_indices)
    assert loaded.frames[1].payload == truth.frames[1].payload
    assert loaded.extras["config"]["modulation"] == "8psk"
    assert np.allclose(loaded.clean, truth.clean)


def test_cfo_and_phase_noise_are_recorded_and_degrade_ber():
    clean_signal, clean_truth = synthesize(SynthConfig(modulation="qpsk", snr_db=40.0, num_frames=2, payload_len=64, seed=6))
    impaired_signal, impaired_truth = synthesize(
        SynthConfig(modulation="qpsk", snr_db=40.0, num_frames=2, payload_len=64, seed=6, cfo_hz=12.0)
    )
    assert impaired_truth.cfo_hz == 12.0
    # 理想参考接收端不补偿频偏，因此误码会显著上升 —— 这正是 M1/M3 要解决的问题
    assert ber(impaired_truth.bits, receive(impaired_signal, impaired_truth)) > ber(
        clean_truth.bits, receive(clean_signal, clean_truth)
    )
