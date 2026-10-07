"""CCSDS 级联码：外码 RS(255,223) + 内码卷积(K=7, r=1/2) + 符号交织.

链序（与 CCSDS 131.0-B 同构，细节简化见下）：

    信息字节 -> RS 编码（I 路，每路一个 RS(255,223) 码字）
             -> 符号交织（深度 I 的轮转发牌，等价于对 I 个码字做转置）
             -> 每符号 8 比特（MSB first）
             -> 卷积编码（K=7, r=1/2, 133/171）

简化说明：CCSDS 还规定了对偶基（dual basis）符号表示与随机化、以及具体的交织器实现；
本实现只保留**结构性**的部分（级联顺序、符号交织深度、标准卷积码），
因此它给出的是正确的编码增益量级与正确的软判决接口，但不是"逐比特可互操作"的 CCSDS 实现。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spxh.core.fec.reed_solomon import ReedSolomon
from spxh.core.fec.viterbi import ConvolutionalCode

__all__ = ["CCSDSCode", "symbol_interleave", "symbol_deinterleave"]


def symbol_interleave(codewords: np.ndarray, depth: int) -> np.ndarray:
    """深度 depth 的符号交织：把 depth 个码字按符号轮转发牌（转置）."""
    words = np.asarray(codewords, dtype=np.uint8)
    if words.ndim != 2 or words.shape[0] != depth:
        raise ValueError("需要形状为 (depth, n) 的码字矩阵")
    return words.T.reshape(-1).copy()


def symbol_deinterleave(stream: np.ndarray, depth: int, n: int) -> np.ndarray:
    data = np.asarray(stream, dtype=np.uint8).reshape(-1)
    if data.size != depth * n:
        raise ValueError("长度与 depth*n 不符")
    return data.reshape(n, depth).T.copy()


@dataclass
class CCSDSCode:
    interleave: int = 5
    n: int = 255
    k: int = 223
    primitive: int = 0x11D

    def __post_init__(self) -> None:
        self.interleave = int(self.interleave)
        if self.interleave < 1:
            raise ValueError("交织深度必须为正")
        self.rs = ReedSolomon(n=int(self.n), k=int(self.k), primitive=int(self.primitive))
        self.conv = ConvolutionalCode()

    @property
    def info_bytes(self) -> int:
        return self.interleave * self.rs.k

    @property
    def rate(self) -> float:
        return (self.rs.k / self.rs.n) * self.conv.rate

    def encode(self, info: bytes | bytearray | np.ndarray) -> np.ndarray:
        data = np.frombuffer(bytes(info), dtype=np.uint8) if not isinstance(info, np.ndarray) else info.astype(np.uint8)
        if data.size != self.info_bytes:
            raise ValueError("信息长度必须是 {} 字节，当前 {}".format(self.info_bytes, data.size))
        codewords = np.stack(
            [self.rs.encode(data[i * self.rs.k : (i + 1) * self.rs.k]) for i in range(self.interleave)]
        )
        interleaved = symbol_interleave(codewords, self.interleave)
        bits = np.unpackbits(interleaved.astype(np.uint8), bitorder="big")
        return self.conv.encode(bits)

    def decode(self, llr, soft: bool = True, terminate: bool = True) -> dict:
        values = np.asarray(llr, dtype=np.float64).reshape(-1)
        if soft:
            bits = self.conv.decode_soft(values, terminate=terminate)
        else:
            bits = self.conv.decode_hard((values < 0).astype(np.uint8), terminate=terminate)
        expected_bits = self.interleave * self.rs.n * 8
        if bits.size < expected_bits:
            raise ValueError("译码输出的比特数不足：{} < {}".format(bits.size, expected_bits))
        packed = np.packbits(bits[:expected_bits], bitorder="big")
        matrix = symbol_deinterleave(packed, self.interleave, self.rs.n)
        info_blocks = []
        corrected = 0
        uncorrectable = 0
        for row in matrix:
            decoded, fixed, failed = self.rs.decode(row)
            corrected += int(fixed)
            uncorrectable += int(failed)
            info_blocks.append(decoded[: self.rs.k])
        return {
            "info": np.concatenate(info_blocks),
            "corrected_symbols": corrected,
            "uncorrectable_codewords": uncorrectable,
            "conv_bits": bits,
        }
