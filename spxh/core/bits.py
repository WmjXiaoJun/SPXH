"""比特/字节打包工具（统一 MSB-first 约定）."""
from __future__ import annotations

import numpy as np

__all__ = [
    "bytes_to_bits",
    "bits_to_bytes",
    "int_to_bits",
    "bits_to_int",
    "hexdump",
]


def _as_bit_array(bits) -> np.ndarray:
    arr = np.asarray(bits)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    if arr.dtype == np.uint8 and arr.ndim == 1:
        return arr
    flat = arr.reshape(-1).astype(np.int64)
    if np.any((flat != 0) & (flat != 1)):
        raise ValueError("bits must contain only 0/1")
    return flat.astype(np.uint8)


def bytes_to_bits(data: bytes, msb_first: bool = True) -> np.ndarray:
    """字节序列 -> 比特数组（默认高位优先）."""
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    bits = np.unpackbits(raw)
    if not msb_first:
        bits = bits.reshape(-1, 8)[:, ::-1].reshape(-1)
    return bits.astype(np.uint8)


def bits_to_bytes(bits, msb_first: bool = True) -> bytes:
    """比特数组 -> 字节序列（长度必须是 8 的整数倍）."""
    arr = _as_bit_array(bits)
    if arr.size % 8 != 0:
        raise ValueError("bit length must be a multiple of 8, got " + str(arr.size))
    mat = arr.reshape(-1, 8)
    if not msb_first:
        mat = mat[:, ::-1]
    return np.packbits(mat.reshape(-1)).tobytes()


def int_to_bits(value: int, width: int, msb_first: bool = True) -> np.ndarray:
    """非负整数 -> 定宽比特数组（高位优先）."""
    if width <= 0:
        raise ValueError("width must be positive")
    if value < 0 or value >= (1 << width):
        raise ValueError("value " + str(value) + " does not fit in " + str(width) + " bits")
    bits = np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)
    if not msb_first:
        bits = bits[::-1].copy()
    return bits


def bits_to_int(bits, msb_first: bool = True) -> int:
    """定宽比特数组 -> 整数（高位优先）."""
    arr = _as_bit_array(bits)
    if arr.size == 0:
        return 0
    if not msb_first:
        arr = arr[::-1]
    value = 0
    for b in arr:
        value = (value << 1) | int(b)
    return value


def hexdump(data: bytes, width: int = 16) -> str:
    """标准十六进制转储（载荷查看用）."""
    lines = []
    for offset in range(0, len(data), width):
        chunk = bytes(data[offset:offset + width])
        hex_part = " ".join("{:02x}".format(b) for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append("{:08x}  {:<{hw}}  |{}|".format(offset, hex_part, ascii_part, hw=width * 3 - 1))
    return "\n".join(lines) if lines else "(empty)"
