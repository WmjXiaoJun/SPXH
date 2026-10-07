"""M6 编码链路测试：帧结构与编码器对接、交织单位、端到端 CRC."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.fec.codec import coded_length_bits, decode_payload, encode_payload, interleave_unit_bits
from spxh.core.framing.format import build_frame, frame_length_bits, parse_frame
from spxh.core.framesync import extract_frames
from spxh.core.synth import SynthConfig, synthesize


@pytest.mark.parametrize("fec,payload", (("none", 64), ("conv", 64), ("rs", 64), ("ccsds", 60), ("ldpc", 64)))
@pytest.mark.parametrize("interleave", (False, True))
def test_coded_frame_noiseless_roundtrip(fec, payload, interleave):
    rng = np.random.default_rng(0)
    info = bytes(rng.integers(0, 256, size=payload, dtype=np.uint8))
    bits, header, crc = build_frame(info, "qpsk", fec=fec, interleave=interleave)
    assert bits.size == frame_length_bits(payload, fec, interleave)
    assert header.fec == fec and header.interleave is interleave
    parsed = parse_frame(bits)
    assert parsed["ok"], parsed["reason"]
    assert bytes(parsed["payload"]) == info
    assert parsed["crc_ok"] is True


@pytest.mark.parametrize("fec,payload", (("conv", 64), ("rs", 64), ("ccsds", 60), ("ldpc", 64)))
def test_codec_payload_roundtrip_soft(fec, payload):
    rng = np.random.default_rng(1)
    info = bytes(rng.integers(0, 256, size=payload, dtype=np.uint8))
    for interleave in (False, True):
        bits = encode_payload(info, fec, interleave)
        assert bits.size == coded_length_bits(payload, fec, interleave)
        llr = np.where(bits == 0, 6.0, -6.0)
        data, meta = decode_payload(bits, fec, payload, interleave, llr=llr)
        assert data == info
        assert meta["fec"] == fec


def test_interleave_unit_matches_code_type():
    # 卷积/LDPC 是比特级码；RS/CCSDS 是符号级码（按字节打散）
    assert interleave_unit_bits("conv") == 1
    assert interleave_unit_bits("ldpc") == 1
    assert interleave_unit_bits("rs") == 8
    assert interleave_unit_bits("ccsds") == 8


def test_bit_interleaving_would_break_rs():
    """用比特交织会把 32 比特突发摊成 32 个符号错误（> t=8），这正是改成字节交织的原因."""
    from spxh.core.fec.interleave import block_deinterleave, block_interleave
    from spxh.core.fec.reed_solomon import ReedSolomon

    rs = ReedSolomon(n=76, k=60)
    rng = np.random.default_rng(2)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)
    bits = np.unpackbits(codeword, bitorder="big")
    rows = int(np.ceil(bits.size / 16.0))
    padded = np.zeros(rows * 16, dtype=np.uint8)
    padded[: bits.size] = bits
    interleaved = block_interleave(padded, rows, 16)
    interleaved[:32] ^= 1                                   # 32 比特突发
    damaged = block_deinterleave(interleaved, rows, 16)[: bits.size]
    received = np.packbits(damaged, bitorder="big")
    _, _, failed = rs.decode(received)
    assert failed is True                                   # 比特交织：不可纠


def test_rs_byte_interleaving_recovers_burst():
    """按字节交织：32 比特突发 = 4 个字节错误 < t=8，可纠."""
    rng = np.random.default_rng(3)
    info = bytes(rng.integers(0, 256, size=64, dtype=np.uint8))
    bits, _, _ = build_frame(info, "qpsk", fec="rs", interleave=True)
    corrupted = np.asarray(bits, dtype=np.uint8).copy()
    start = 13 + 32 + 100
    corrupted[start : start + 32] ^= 1
    parsed = parse_frame(corrupted)
    assert parsed["ok"]
    assert bytes(parsed["payload"]) == info
    assert parsed["fec"]["uncorrectable"] is False


@pytest.mark.parametrize("fec,payload", (("conv", 64), ("rs", 64), ("ccsds", 60), ("ldpc", 64)))
def test_end_to_end_coded_link(fec, payload):
    """合成 -> 解调 -> 帧同步 -> 译码 -> CRC -> 载荷（20 dB）."""
    signal, truth = synthesize(
        SynthConfig(modulation="qpsk", symbol_rate=1000.0, sps=8, snr_db=20.0, cfo_hz=37.0,
                    num_frames=4, payload_len=payload, fec=fec, seed=11, keep_clean=True)
    )
    result = extract_frames(signal, modulation="qpsk", truth=truth)
    summary = result.summary
    assert summary["frames"] >= 2
    assert summary["frames_crc_ok"] == summary["frames"]
    assert summary["payload_bytes_exact"] == summary["payload_bytes_total"]


def test_ccsds_accepts_arbitrary_length_by_padding():
    """CCSDS 级联的交织深度要求信息字节数可整除；不足时在信息尾部补零，任意长度都能用."""
    for payload_len in (60, 61, 63):
        signal, truth = synthesize(
            SynthConfig(modulation="qpsk", fec="ccsds", payload_len=payload_len, num_frames=1, snr_db=20.0, seed=3)
        )
        assert signal.num_samples > 0
        assert truth.frames[0].header.payload_len == payload_len


def test_synthesize_rejects_out_of_range_fec_length():
    """超出编码器能力时必须明确报错，而不是静默生成"声称已编码"的波形."""
    with pytest.raises(ValueError):
        synthesize(SynthConfig(modulation="qpsk", fec="ldpc", payload_len=200, num_frames=1, snr_db=20.0))
    with pytest.raises(ValueError):
        synthesize(SynthConfig(modulation="qpsk", fec="ccsds", payload_len=1400, num_frames=1, snr_db=20.0))
