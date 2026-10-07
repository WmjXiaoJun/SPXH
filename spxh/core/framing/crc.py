"""CRC-16-CCITT（即 CRC-16/CCITT-FALSE）.

参数：poly=0x1021, init=0xFFFF, refin=false, refout=false, xorout=0x0000。
标准校验向量：CRC16("123456789") == 0x29B1。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "CRC16_POLY",
    "CRC16_INIT",
    "CRC16_XOROUT",
    "CRC8_POLY",
    "crc16_ccitt",
    "crc16_ccitt_hex",
    "check_crc16",
    "crc8",
]

CRC16_POLY = 0x1021
CRC16_INIT = 0xFFFF
CRC16_XOROUT = 0x0000

_TABLE: list[int] = []


def _build_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = (byte << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return table


_TABLE = _build_table()


def crc16_ccitt(data: bytes, init: int = CRC16_INIT, xorout: int = CRC16_XOROUT) -> int:
    """对字节序列计算 CRC-16-CCITT，返回 16 位整数."""
    crc = init & 0xFFFF
    for byte in bytes(data):
        crc = ((crc << 8) & 0xFFFF) ^ _TABLE[((crc >> 8) ^ byte) & 0xFF]
    return (crc ^ xorout) & 0xFFFF


def crc16_ccitt_hex(data: bytes) -> str:
    return "{:04x}".format(crc16_ccitt(data))


CRC8_POLY = 0x07          # CRC-8/SMBUS（poly=x^8+x^2+x+1, init=0x00, 无反转）
_CRC8_TABLE: list[int] = []


def _build_crc8_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = ((crc << 1) ^ CRC8_POLY) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
        table.append(crc)
    return table


_CRC8_TABLE = _build_crc8_table()


def crc8(data: bytes, init: int = 0x00) -> int:
    """CRC-8：给**帧头**做自校验用（1 字节开销，能挡住 255/256 的随机错误）."""
    crc = init & 0xFF
    for byte in bytes(data):
        crc = _CRC8_TABLE[(crc ^ byte) & 0xFF]
    return crc & 0xFF


def check_crc16(data_with_crc: bytes) -> bool:
    """校验「数据 + 2 字节大端 CRC」整体."""
    if len(data_with_crc) < 3:
        return False
    body = data_with_crc[:-2]
    tail = int.from_bytes(data_with_crc[-2:], "big")
    return crc16_ccitt(body) == tail
