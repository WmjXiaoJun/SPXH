"""M4 信道编码：交织、Viterbi、Reed-Solomon、CCSDS 级联、LDPC."""
from spxh.core.fec.ccsds import CCSDSCode, symbol_deinterleave, symbol_interleave
from spxh.core.fec.interleave import (
    block_deinterleave,
    block_interleave,
    convolutional_deinterleave,
    convolutional_interleave,
)
from spxh.core.fec.ldpc import LDPCCode
from spxh.core.fec.reed_solomon import GF256, ReedSolomon, rs_correctable_errors
from spxh.core.fec.viterbi import ConvolutionalCode, viterbi_decode_hard, viterbi_decode_soft

__all__ = [
    "block_interleave",
    "block_deinterleave",
    "convolutional_interleave",
    "convolutional_deinterleave",
    "symbol_interleave",
    "symbol_deinterleave",
    "ConvolutionalCode",
    "viterbi_decode_hard",
    "viterbi_decode_soft",
    "GF256",
    "ReedSolomon",
    "rs_correctable_errors",
    "CCSDSCode",
    "LDPCCode",
]
