"""M9 测试：帧头 CRC-8 自校验、载荷长度盲恢复、报告层（离线确定性）."""
from __future__ import annotations

import json

import numpy as np
import pytest

from spxh.core.framing.crc import crc8
from spxh.core.framing.format import (
    HEADER_BITS,
    PREAMBLE_BITS,
    build_frame,
    header_crc_ok,
    parse_frame,
)
from spxh.core.framesync.extract import _blind_header_candidates
from spxh.core.report import build_prompt, narrate, render_offline


def test_crc8_matches_standard_vector():
    assert crc8(b"123456789") == 0xF4


def test_header_is_five_bytes_with_crc8():
    for fec in ("none", "conv", "rs", "ccsds", "ldpc"):
        bits, header, _crc = build_frame(bytes(60), "qpsk", fec=fec)
        hb = np.packbits(
            np.asarray(bits[PREAMBLE_BITS : PREAMBLE_BITS + HEADER_BITS], dtype=np.uint8), bitorder="big"
        ).tobytes()
        assert len(hb) == 5
        assert header_crc_ok(hb)
        # 改一个 bit，CRC 必须报警
        tampered = bytearray(hb)
        tampered[2] ^= 0x01
        assert not header_crc_ok(bytes(tampered))


@pytest.mark.parametrize("fec", ("none", "conv", "rs", "ccsds", "ldpc"))
def test_blind_length_recovery_after_header_damage(fec):
    """把载荷长度字段打错：声明解析必然失败，靠 CRC-8 过滤的假设搜索恢复."""
    rng = np.random.default_rng(3)
    info = bytes(rng.integers(0, 256, size=60, dtype=np.uint8))
    bits, _header, _crc = build_frame(info, "qpsk", fec=fec)
    tampered = np.asarray(bits, dtype=np.uint8).copy()
    # byte2 的 bit0-1（帧比特 30、31）与 byte3 的 bit0（帧比特 38）
    tampered[30] ^= 1
    tampered[31] ^= 1
    tampered[38] ^= 1
    assert not parse_frame(tampered)["ok"]

    candidates = _blind_header_candidates(tampered, 0, limit=12)
    assert candidates, "CRC-8 过滤后应给出候选"
    recovered = None
    for spec in candidates:
        parsed = parse_frame(
            tampered,
            force_fec=spec["fec"],
            force_interleave=spec["interleave"],
            force_payload_len=spec["payload_len"],
        )
        if parsed["ok"] and bytes(parsed["payload"]) == info:
            recovered = spec
            break
    assert recovered is not None
    assert recovered["fec"] == fec
    assert recovered["payload_len"] == 60


def test_header_crc_does_not_break_normal_parsing():
    info = bytes(range(60))
    bits, _header, _crc = build_frame(info, "qpsk", fec="rs")
    parsed = parse_frame(bits)
    assert parsed["ok"]
    assert parsed["header_crc_ok"] is True
    assert bytes(parsed["payload"]) == info


def test_offline_report_is_deterministic_and_evidence_backed():
    path = "data/demo/qpsk_snr+10.0_000.sigmf-meta"
    first = narrate(path, provider="offline")
    second = narrate(path, provider="offline")
    assert first["mode"] == "offline"
    assert json.dumps(first["sections"], ensure_ascii=False) == json.dumps(second["sections"], ensure_ascii=False)
    assert first["headline"]
    assert 0.0 <= float(first["confidence"]) <= 1.0
    ids = {section["id"] for section in first["sections"]}
    assert {"signal", "modulation"} <= ids
    for section in first["sections"]:
        assert isinstance(section["evidence"], list)
    # 提示词必须包含"不许编造数值"的约束与实际证据
    assert "绝不允许编造" in build_prompt(first["evidence"])


def test_report_survives_missing_stages():
    """证据不全时也要产出结构化报告（而不是抛异常）."""
    report = render_offline({"path": "x", "signal": [], "classification": [], "frame": [],
                             "cyclostationary": [], "pipeline_errors": [{"stage": "frame", "error": "boom"}]})
    assert report["mode"] == "offline"
    assert report["uncertainties"]
    assert report["sections"] == [] or isinstance(report["sections"], list)
