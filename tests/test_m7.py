"""M7 测试：盲编码识别、RS 擦除译码、帧头假设重写."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.fec.reed_solomon import ReedSolomon
from spxh.core.framing.format import build_frame, parse_frame, frame_length_bits
from spxh.core.framesync.extract import _BLIND_HYPOTHESES


def _tamper_fec_field(bits: np.ndarray) -> np.ndarray:
    """把帧头 byte1 的 fec 字段（帧比特 25..27）与交织位（28）打成 0：谎报"无编码"."""
    out = np.asarray(bits, dtype=np.uint8).copy()
    out[25] = 0
    out[26] = 0
    out[27] = 0
    out[28] = 0
    return out


@pytest.mark.parametrize("fec", ("conv", "rs", "ccsds", "ldpc"))
def test_blind_code_detection_recovers_after_header_tampering(fec):
    rng = np.random.default_rng(7)
    info = bytes(rng.integers(0, 256, size=60, dtype=np.uint8))
    bits, _header, _crc = build_frame(info, "qpsk", fec=fec)
    tampered = _tamper_fec_field(bits)

    # 按被篡改的帧头去解：必然失败（它说自己是无编码）
    assert not parse_frame(tampered)["ok"]

    # 盲搜索：对每个 (编码, 交织) 假设重写帧头里的 fec 位再算 CRC
    hit = None
    for hypothesis_fec, hypothesis_inter in _BLIND_HYPOTHESES:
        candidate = parse_frame(tampered, force_fec=hypothesis_fec, force_interleave=hypothesis_inter)
        if candidate["ok"] and bytes(candidate["payload"]) == info:
            hit = (hypothesis_fec, hypothesis_inter)
            break
    assert hit is not None, fec
    assert hit[0] == fec


def test_force_fec_rewrites_header_bits_for_crc():
    """盲模式的关键：CRC 必须按**假设重写后的帧头**算，否则永远对不上."""
    rng = np.random.default_rng(8)
    info = bytes(rng.integers(0, 256, size=60, dtype=np.uint8))
    bits, header, _crc = build_frame(info, "qpsk", fec="ldpc")
    # 帧头里的调制字段不变，只把 fec 假设改掉：错误假设不应通过
    wrong = parse_frame(bits, force_fec="rs")
    assert not wrong["ok"]
    right = parse_frame(bits, force_fec="ldpc")
    assert right["ok"]
    assert bytes(right["payload"]) == info


def test_rs_erasure_decoding_doubles_capacity():
    rs = ReedSolomon(n=76, k=60)
    rng = np.random.default_rng(9)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)

    # t=8：9 个未知错误不可纠，但 9 个**已知位置**可以
    positions = np.sort(rng.choice(rs.n, size=9, replace=False))
    damaged = codeword.copy()
    damaged[positions] ^= rng.integers(1, 256, size=9, dtype=np.uint8)
    _, _, failed = rs.decode(damaged)
    assert failed is True
    fixed, corrected, failed2 = rs.decode(damaged, erasures=positions.tolist())
    assert failed2 is False and corrected == 9
    assert np.array_equal(fixed[: rs.k], info)

    # 2t = 16 个擦除仍然可以
    positions16 = np.sort(rng.choice(rs.n, size=16, replace=False))
    damaged16 = codeword.copy()
    damaged16[positions16] ^= rng.integers(1, 256, size=16, dtype=np.uint8)
    fixed16, _, failed16 = rs.decode(damaged16, erasures=positions16.tolist())
    assert failed16 is False
    assert np.array_equal(fixed16[: rs.k], info)

    # 17 个擦除超出能力，必须如实报不可纠
    positions17 = np.sort(rng.choice(rs.n, size=17, replace=False))
    damaged17 = codeword.copy()
    damaged17[positions17] ^= rng.integers(1, 256, size=17, dtype=np.uint8)
    _, _, failed17 = rs.decode(damaged17, erasures=positions17.tolist())
    assert failed17 is True


def test_rs_erasure_with_unknown_errors_is_recovered():
    """M8 起：擦除标得不全（还漏了一个未知错误）也能救回来 —— 2*1 + 4 = 6 <= 2t。

    这正是"误差 + 擦除联合译码"的价值：纯擦除译码在这种情况下只能报不可纠。
    """
    rs = ReedSolomon(n=76, k=60)
    rng = np.random.default_rng(10)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)
    damaged = codeword.copy()
    erasures = [3, 10, 20, 30]
    for position in erasures:
        damaged[position] ^= 0x5A
    damaged[70] ^= 0x33                     # 一个未标注的错误
    decoded, corrected, failed = rs.decode(damaged, erasures=erasures)
    assert failed is False
    assert corrected == 5                   # 4 个擦除 + 1 个未知错误
    assert np.array_equal(decoded[: rs.k], info)


def test_rs_joint_decoding_reports_uncorrectable_beyond_capacity():
    """超出 2 x 错误 + 擦除 <= 2t 时必须报不可纠，不许给出可疑结果."""
    rs = ReedSolomon(n=76, k=60)
    rng = np.random.default_rng(11)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)
    damaged = codeword.copy()
    erasures = [3, 10, 20, 30]
    for position in erasures:
        damaged[position] ^= 0x5A
    unknown = [50, 51, 52, 53, 54, 55, 56]  # 2*7 + 4 = 18 > 16
    for position in unknown:
        damaged[position] ^= 0x33
    _, _, failed = rs.decode(damaged, erasures=erasures)
    assert failed is True


def test_frame_length_consistent_with_crc_inside_code():
    """CRC 进编码后，帧长必须由 (payload_len, fec, interleave) 唯一决定."""
    for fec, payload in (("none", 60), ("conv", 60), ("rs", 60), ("ccsds", 60), ("ldpc", 60)):
        bits, _header, _crc = build_frame(bytes(payload), "qpsk", fec=fec)
        assert bits.size == frame_length_bits(payload, fec, False)
        parsed = parse_frame(bits)
        assert parsed["ok"], (fec, parsed["reason"])
