"""M4 信道编码测试：交织、Viterbi、Reed-Solomon、CCSDS 级联、LDPC."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.fec import (
    CCSDSCode,
    ConvolutionalCode,
    GF256,
    LDPCCode,
    ReedSolomon,
    block_deinterleave,
    block_interleave,
    convolutional_deinterleave,
    convolutional_interleave,
    symbol_deinterleave,
    symbol_interleave,
)
from spxh.core.fec.viterbi import viterbi_decode_hard, viterbi_decode_soft


# ------------------------------------------------------------------ 交织
def test_block_interleave_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=15 * 136, dtype=np.uint8)
    interleaved = block_interleave(bits, 15, 136)
    assert not np.array_equal(interleaved, bits)
    assert np.array_equal(block_deinterleave(interleaved, 15, 136), bits)


def test_block_interleave_spreads_burst():
    bits = np.zeros(200, dtype=np.uint8)
    bits[:20] = 1                       # 连续 20 个错误
    interleaved = block_interleave(bits, 10, 20).reshape(20, 10)
    # 交织后每个"行"（对应原来的列）里错误被摊开
    assert interleaved.sum() == 20
    assert int(np.max(interleaved.sum(axis=1))) <= 2


def test_convolutional_interleave_roundtrip():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=500, dtype=np.uint8)
    for depth, step in ((5, 1), (5, 4), (8, 2)):
        interleaved = convolutional_interleave(bits, depth, step)
        # 冲刷段长度 = step * depth * (depth-1) / 2
        assert interleaved.size == bits.size + step * depth * (depth - 1) // 2
        assert np.array_equal(convolutional_deinterleave(interleaved, depth, step, length=bits.size), bits)


def test_symbol_interleave_roundtrip():
    rng = np.random.default_rng(2)
    words = rng.integers(0, 256, size=(5, 255), dtype=np.uint8)
    stream = symbol_interleave(words, 5)
    assert stream.size == 5 * 255
    assert np.array_equal(symbol_deinterleave(stream, 5, 255), words)


# ------------------------------------------------------------------ Viterbi
def test_viterbi_encoder_matches_reference():
    code = ConvolutionalCode(constraint=3, polys=(0o7, 0o5))
    info = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    manual = []
    register = 0
    for bit in list(info) + [0, 0]:
        register = ((register << 1) | int(bit)) & 0b111
        manual.append(bin(register & 0o7).count("1") & 1)
        manual.append(bin(register & 0o5).count("1") & 1)
    assert code.encode(info).tolist() == manual


def test_viterbi_noiseless_roundtrip():
    code = ConvolutionalCode()
    rng = np.random.default_rng(3)
    info = rng.integers(0, 2, size=500, dtype=np.uint8)
    coded = code.encode(info)
    assert coded.size == (info.size + 6) * 2
    assert np.array_equal(code.decode_hard(coded), info)
    llr = np.where(coded == 0, 5.0, -5.0)
    assert np.array_equal(code.decode_soft(llr), info)


def test_viterbi_corrects_errors():
    code = ConvolutionalCode()
    rng = np.random.default_rng(4)
    info = rng.integers(0, 2, size=400, dtype=np.uint8)
    coded = code.encode(info)
    noisy = coded.copy()
    noisy[rng.choice(coded.size, size=12, replace=False)] ^= 1
    assert np.array_equal(code.decode_hard(noisy), info)


def test_viterbi_soft_beats_hard_at_low_snr():
    code = ConvolutionalCode()
    rng = np.random.default_rng(5)
    sigma = 0.9
    hard_errors = 0
    soft_errors = 0
    total = 0
    for _ in range(6):
        info = rng.integers(0, 2, size=800, dtype=np.uint8)
        coded = code.encode(info)
        y = (1.0 - 2.0 * coded) + rng.normal(0.0, sigma, size=coded.size)
        llr = 2.0 * y / sigma ** 2
        hard_errors += int(np.sum(code.decode_hard((y < 0).astype(np.uint8)) != info))
        soft_errors += int(np.sum(code.decode_soft(llr) != info))
        total += info.size
    assert hard_errors > 0
    assert soft_errors <= hard_errors


def test_viterbi_helpers_use_default_code():
    bits = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    coded = ConvolutionalCode().encode(bits)
    assert np.array_equal(viterbi_decode_hard(coded), bits)
    llr = np.where(coded == 0, 4.0, -4.0)
    assert np.array_equal(viterbi_decode_soft(llr), bits)


# --------------------------------------------------------------- Reed-Solomon
def test_gf256_tables():
    gf = GF256()
    assert all(gf.exp[gf.log[value]] == value for value in range(1, 256))
    assert gf.mul(2, 128) == 29          # 0x100 ^ 0x11D
    assert gf.div(gf.mul(37, 91), 37) == 91


def test_rs_clean_codeword_has_zero_syndromes():
    rs = ReedSolomon()
    rng = np.random.default_rng(6)
    codeword = rs.encode(rng.integers(0, 256, size=rs.k, dtype=np.uint8))
    assert not any(rs.syndromes(codeword))
    decoded, corrected, failed = rs.decode(codeword)
    assert corrected == 0 and not failed


@pytest.mark.parametrize("errors", (1, 8, 16))
def test_rs_corrects_up_to_t_symbols(errors):
    rs = ReedSolomon()
    rng = np.random.default_rng(7)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)
    noisy = codeword.copy()
    positions = rng.choice(rs.n, size=errors, replace=False)
    noisy[positions] ^= rng.integers(1, 256, size=errors, dtype=np.uint8)
    decoded, corrected, failed = rs.decode(noisy)
    assert not failed
    assert corrected == errors
    assert np.array_equal(decoded[: rs.k], info)


def test_rs_reports_uncorrectable_beyond_t():
    rs = ReedSolomon()
    rng = np.random.default_rng(8)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)
    noisy = codeword.copy()
    positions = rng.choice(rs.n, size=rs.t + 1, replace=False)
    noisy[positions] ^= rng.integers(1, 256, size=rs.t + 1, dtype=np.uint8)
    _, _, failed = rs.decode(noisy)
    assert failed is True


def test_rs_shortened_code():
    rs = ReedSolomon(n=100, k=68)
    rng = np.random.default_rng(9)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)
    noisy = codeword.copy()
    positions = rng.choice(rs.n, size=8, replace=False)
    noisy[positions] ^= 0x5A
    decoded, corrected, failed = rs.decode(noisy)
    assert not failed and corrected == 8
    assert np.array_equal(decoded[: rs.k], info)


# ------------------------------------------------------------------- CCSDS
def test_ccsds_noiseless_roundtrip():
    code = CCSDSCode(interleave=5)
    rng = np.random.default_rng(10)
    info = rng.integers(0, 256, size=code.info_bytes, dtype=np.uint8)
    coded = code.encode(info)
    assert coded.size == code.interleave * code.n * 8 * 2 + 12   # 含卷积码结尾
    llr = np.where(coded == 0, 8.0, -8.0)
    result = code.decode(llr)
    assert np.array_equal(result["info"], info)
    assert result["uncorrectable_codewords"] == 0


def test_ccsds_rate_and_interleaver_objective():
    code = CCSDSCode(interleave=5)
    assert code.info_bytes == 5 * 223
    assert 0.43 < code.rate < 0.44


# -------------------------------------------------------------------- LDPC
def test_ldpc_code_parameters():
    code = LDPCCode(n=1152, dv=3, dc=6, seed=2026)
    assert code.n == 1152
    assert code.m <= 576
    assert code.k == code.n - code.m
    assert code.check_vars.shape[0] == code.m
    # 信息列 + 校验列 恰好铺满
    assert code._info_cols.size + code._check_cols.size == code.n


def test_ldpc_encoding_satisfies_parity():
    code = LDPCCode(n=1152, dv=3, dc=6, seed=2026)
    rng = np.random.default_rng(11)
    info = rng.integers(0, 2, size=code.k, dtype=np.uint8)
    codeword = code.encode(info)
    assert code.parity_ok(codeword)
    assert np.array_equal(codeword[code._info_cols], info)


def test_ldpc_noiseless_and_noisy_decode():
    code = LDPCCode(n=1152, dv=3, dc=6, seed=2026)
    rng = np.random.default_rng(12)
    info = rng.integers(0, 2, size=code.k, dtype=np.uint8)
    codeword = code.encode(info)
    assert np.array_equal(code.decode(np.where(codeword == 0, 6.0, -6.0))["info"], info)
    sigma = 1.0 / np.sqrt(2 * 10 ** 0.4)
    y = (1.0 - 2.0 * codeword) + rng.normal(0.0, sigma, size=codeword.size)
    result = code.decode(2.0 * y / sigma ** 2)
    assert result["converged"]
    assert np.array_equal(result["info"], info)


def test_ldpc_decode_requires_correct_length():
    code = LDPCCode(n=1152, dv=3, dc=6, seed=2026)
    with pytest.raises(ValueError):
        code.decode(np.zeros(10))
