"""比特/字节打包测试."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.bits import bits_to_bytes, bits_to_int, bytes_to_bits, hexdump, int_to_bits


def test_bytes_bits_roundtrip_msb_first():
    data = bytes([0xA5, 0x00, 0xFF, 0x5A])
    bits = bytes_to_bits(data)
    assert bits.tolist()[:8] == [1, 0, 1, 0, 0, 1, 0, 1]
    assert bits_to_bytes(bits) == data


def test_bits_to_bytes_requires_byte_multiple():
    with pytest.raises(ValueError):
        bits_to_bytes([1, 0, 1])


def test_int_bits_roundtrip():
    for width in (1, 4, 8, 16, 32):
        for value in (0, 1, (1 << width) - 1):
            bits = int_to_bits(value, width)
            assert bits.size == width
            assert bits_to_int(bits) == value


def test_int_bits_range_check():
    with pytest.raises(ValueError):
        int_to_bits(256, 8)


def test_bits_reject_non_binary():
    with pytest.raises(ValueError):
        bits_to_bytes(np.array([0, 2, 0, 0, 0, 0, 0, 0]))


def test_hexdump_contains_ascii_and_offset():
    text = hexdump(b"SPXH\x00\x01")
    assert "00000000" in text
    assert "SPXH" in text
