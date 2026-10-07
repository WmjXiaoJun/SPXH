"""帧结构打包/解析测试."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.framing.format import (
    HEADER_BYTES,
    PREAMBLE_BITS,
    build_frame,
    frame_length_bits,
    header_bytes,
    parse_frame,
    parse_header_bytes,
    verify_frame,
)
from spxh.core.types import MODULATIONS, Header


def test_header_bytes_roundtrip():
    header = Header(modulation="8psk", fec="rs", interleave=True, payload_len=4096)
    raw = header_bytes(header)
    assert len(raw) == HEADER_BYTES
    assert raw[0] == 0xA5
    parsed = parse_header_bytes(raw)
    assert parsed.modulation == "8psk"
    assert parsed.fec == "rs"
    assert parsed.interleave is True
    assert parsed.payload_len == 4096


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_frame_build_parse_roundtrip(modulation):
    payload = bytes(range(256)) * 2
    bits, header, crc = build_frame(payload, modulation)
    assert header.modulation == modulation
    assert bits.size == frame_length_bits(len(payload))
    result = parse_frame(bits)
    assert result["ok"] is True
    assert result["payload"] == payload
    assert result["crc_actual"] == crc
    assert result["bits_consumed"] == bits.size
    assert np.array_equal(bits[:PREAMBLE_BITS], np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8))


def test_empty_payload_is_valid():
    bits, header, _ = build_frame(b"", "qpsk")
    assert header.payload_len == 0
    result = parse_frame(bits)
    assert result["ok"] is True
    assert result["payload"] == b""


def test_single_bit_flip_is_detected():
    bits, _, _ = build_frame(b"SPXH" * 8, "qpsk")
    for position in (0, PREAMBLE_BITS, 20, 60, bits.size - 1):
        corrupted = bits.copy()
        corrupted[position] ^= 1
        ok, reason = verify_frame(corrupted)
        assert not ok, "flip at " + str(position) + " should be rejected"
        assert reason


def test_bad_magic_is_reported():
    bits, _, _ = build_frame(b"abc", "qpsk")
    corrupted = bits.copy()
    corrupted[PREAMBLE_BITS] ^= 1
    result = parse_frame(corrupted)
    assert result["ok"] is False
    assert "magic" in result["reason"] or "header" in result["reason"]


def test_payload_length_field_is_used_for_truncation():
    bits, _, _ = build_frame(b"x" * 32, "16qam")
    truncated = bits[: bits.size - 8]
    result = parse_frame(truncated)
    assert result["ok"] is False
    assert "truncated" in result["reason"]


def test_consecutive_frames_parse_with_offsets():
    payloads = [b"first", b"second-frame", b""]
    chunks = [build_frame(p, "qpsk")[0] for p in payloads]
    stream = np.concatenate(chunks)
    offset = 0
    for index, payload in enumerate(payloads):
        result = parse_frame(stream[offset:])
        assert result["ok"] is True
        assert result["payload"] == payload
        offset += result["bits_consumed"]
    assert offset == stream.size


def test_payload_too_long_rejected():
    with pytest.raises(ValueError):
        build_frame(b"x" * 70000, "qpsk")
