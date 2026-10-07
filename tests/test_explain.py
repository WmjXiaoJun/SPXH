"""M8 测试：RS 误差+擦除联合译码、SSCA 谱相关."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.explain import compute_ssca, find_surface_peaks
from spxh.core.fec.reed_solomon import ReedSolomon
from spxh.core.synth import SynthConfig, synthesize


def _damaged(codeword, erasures, errors, rng):
    word = codeword.copy()
    for position in list(erasures) + list(errors):
        word[position] ^= int(rng.integers(1, 256))
    return word


@pytest.mark.parametrize("n_erasure,n_error,should_work", ((0, 8, True), (2, 7, True), (4, 6, True),
                                                           (8, 4, True), (16, 0, True), (4, 7, False)))
def test_rs_joint_error_erasure_capacity(n_erasure, n_error, should_work):
    """能力约束：2 x 未知错误 + 擦除 <= nsym = 2t."""
    rs = ReedSolomon(n=76, k=60)
    rng = np.random.default_rng(21)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)
    erasures = list(rng.choice(rs.n, size=n_erasure, replace=False)) if n_erasure else []
    remaining = [p for p in range(rs.n) if p not in erasures]
    errors = list(rng.choice(remaining, size=n_error, replace=False)) if n_error else []
    received = _damaged(codeword, erasures, errors, rng)
    decoded, corrected, failed = rs.decode(received, erasures=erasures)
    if should_work:
        assert failed is False
        assert np.array_equal(decoded[: rs.k], info)
    else:
        assert failed is True


def test_rs_joint_decoding_never_silently_miscorrects():
    """超出能力时必须报不可纠；擦除标得不准也要报，而不是给出可疑结果."""
    rs = ReedSolomon(n=76, k=60)
    rng = np.random.default_rng(22)
    info = rng.integers(0, 256, size=rs.k, dtype=np.uint8)
    codeword = rs.encode(info)
    erasures = [3, 11, 19, 27]
    errors = [40, 41, 42, 43, 44, 45, 46, 47]     # 2*8 + 4 = 20 > 16
    received = _damaged(codeword, erasures, errors, rng)
    _, _, failed = rs.decode(received, erasures=erasures)
    assert failed is True


def test_ssca_finds_symbol_rate_and_plausible_cfo():
    signal, truth = synthesize(
        SynthConfig(modulation="qpsk", symbol_rate=1000.0, sps=8, snr_db=20.0, cfo_hz=120.0,
                    num_frames=6, payload_len=64, seed=11, keep_clean=True)
    )
    result = compute_ssca(signal.samples, signal.sample_rate, nfft=256, symbol_rate_hint=1000.0)
    assert result.surface_db.shape == (result.alphas.size, result.freqs.size)
    # alpha 轴峰应正好落在符号速率上（分辨率 fs/nfft = 31.25 Hz）
    assert result.symbol_rate_hz is not None
    assert abs(result.symbol_rate_hz - 1000.0) <= 31.25 + 1e-6
    # 载频偏移由 alpha!=0 脊线给出，分辨率一个 bin
    assert result.cfo_hz is not None
    assert abs(result.cfo_hz - 120.0) <= 40.0


def test_ssca_surface_has_null_edges_and_peaks():
    signal, _ = synthesize(
        SynthConfig(modulation="qpsk", symbol_rate=1000.0, sps=8, snr_db=15.0, cfo_hz=0.0,
                    num_frames=4, payload_len=64, seed=3, keep_clean=True)
    )
    result = compute_ssca(signal.samples, signal.sample_rate, nfft=128)
    payload = result.to_dict(max_points=32)
    surface = payload["surface_db"]
    assert any(value is None for row in surface for value in row)   # 循环移位的无效区
    assert payload["peaks"], "至少应给出一个峰"
    assert set(payload["features"]) >= {"symbol_rate_hz", "cfo_hz", "note"}
    assert len(payload["alpha_profile"]["level_db"]) == len(payload["alphas"])
    assert len(payload["freq_profile"]["level_db"]) == len(payload["freqs"])


def test_ssca_rejects_short_records():
    with pytest.raises(ValueError):
        compute_ssca(np.zeros(64, dtype=np.complex128), 8000.0, nfft=256)


def test_ssca_peaks_are_suppressed_neighbourhoods():
    signal, _ = synthesize(
        SynthConfig(modulation="2fsk", symbol_rate=1000.0, sps=8, snr_db=20.0, cfo_hz=0.0,
                    num_frames=4, payload_len=64, seed=5, keep_clean=True)
    )
    result = compute_ssca(signal.samples, signal.sample_rate, nfft=256, symbol_rate_hint=1000.0)
    peaks = find_surface_peaks(result, limit=5)
    alphas = [p["alpha_hz"] for p in peaks]
    assert len(alphas) == len(set(alphas)), "非极大抑制后 alpha 不应重复"
