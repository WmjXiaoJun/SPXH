"""损伤注入测试."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.synth.impairments import (
    add_awgn,
    apply_cfo,
    apply_dc_offset,
    apply_iq_imbalance,
    apply_phase_noise,
    complex_noise_variance,
    measure_snr_db,
)


def test_complex_noise_variance_definition():
    assert abs(complex_noise_variance(1.0, 10.0) - 0.1) < 1e-15
    assert abs(complex_noise_variance(8.0, 0.0) - 8.0) < 1e-15


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_add_awgn_matches_target(snr_db):
    rng = np.random.default_rng(11)
    x = np.ones(200000, dtype=np.complex128)
    y, sigma2 = add_awgn(x, snr_db, es_per_symbol=1.0, rng=rng, return_sigma=True)
    measured = float(np.mean(np.abs(y - x) ** 2))
    assert abs(10.0 * np.log10(1.0 / measured) - snr_db) < 0.1
    assert abs(sigma2 - 10 ** (-snr_db / 10.0)) < 1e-12


def test_cfo_zero_is_identity_and_nonzero_shifts_phase():
    x = np.ones(64, dtype=np.complex128)
    assert np.array_equal(apply_cfo(x, 1000.0, 0.0), x)
    shifted = apply_cfo(x, 1000.0, 125.0)
    step = np.angle(shifted[1] / shifted[0])
    assert abs(step - 2 * np.pi * 125.0 / 1000.0) < 1e-12


def test_phase_noise_rms_is_as_requested():
    rng = np.random.default_rng(12)
    x = np.ones(400000, dtype=np.complex128)
    y = apply_phase_noise(x, 1e6, 0.1, bandwidth_hz=1e4, rng=rng)
    phase = np.unwrap(np.angle(y))
    assert abs(float(np.std(phase)) - 0.1) < 0.01


def test_iq_imbalance_identity_and_distortion():
    rng = np.random.default_rng(13)
    x = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) / np.sqrt(2)
    assert np.array_equal(apply_iq_imbalance(x, 0.0, 0.0), x)
    assert not np.allclose(apply_iq_imbalance(x, 1.0, 5.0), x)

    # 用基向量做确定性校验：y = alpha*x + beta*conj(x)
    #   x = 1  -> y = alpha + beta；x = j -> y = j*(alpha - beta)
    # 纯相位不平衡 (g=1)：alpha+beta = 1，alpha-beta = e^{j*phi} -> 功率严格守恒
    basis = np.array([1.0 + 0j, 1j])
    phase_only = apply_iq_imbalance(basis, 0.0, 10.0)
    assert np.allclose(np.abs(phase_only) ** 2, 1.0, atol=1e-12)

    # 纯幅度不平衡 (phi=0)：alpha+beta = 1，alpha-beta = g -> 第二项功率 g^2
    gain_only = apply_iq_imbalance(basis, 1.0, 0.0)
    assert abs(abs(gain_only[0]) ** 2 - 1.0) < 1e-12
    assert abs(abs(gain_only[1]) ** 2 - 10.0 ** 0.1) < 1e-12


def test_dc_offset_and_measure_snr():
    x = np.ones(1024, dtype=np.complex128)
    assert np.allclose(apply_dc_offset(x, 1 + 1j), x + 1 + 1j)
    rng = np.random.default_rng(14)
    noisy = add_awgn(x, 10.0, rng=rng)
    assert abs(measure_snr_db(noisy, x) - 10.0) < 0.3
