"""CRC-16-CCITT 测试（含标准校验向量）."""
from __future__ import annotations

import pytest

from spxh.core.framing.crc import check_crc16, crc16_ccitt, crc16_ccitt_hex


def test_standard_check_vector():
    # CRC-16/CCITT-FALSE 的标准向量
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_empty_input_is_init_value():
    assert crc16_ccitt(b"") == 0xFFFF


def test_crc_changes_with_any_single_bit():
    base = bytes(range(32))
    reference = crc16_ccitt(base)
    for byte_index in range(len(base)):
        for bit in range(8):
            mutated = bytearray(base)
            mutated[byte_index] ^= 1 << bit
            assert crc16_ccitt(bytes(mutated)) != reference


def test_check_crc16_roundtrip():
    body = b"SPXH payload"
    crc = crc16_ccitt(body)
    assert check_crc16(body + crc.to_bytes(2, "big"))
    assert not check_crc16(body + b"\x00\x00")


def test_hex_helper():
    assert crc16_ccitt_hex(b"123456789") == "29b1"
