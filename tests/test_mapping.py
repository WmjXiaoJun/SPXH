"""调制映射测试：往返、单位能量、Gray 性质、最小距离."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.synth import mapping
from spxh.core.types import MODULATIONS

PSK_QAM = ("bpsk", "qpsk", "8psk", "16qam")


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_bits_indices_roundtrip(modulation):
    bps = mapping.bits_per_symbol(modulation)
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=bps * 257, dtype=np.uint8)
    indices = mapping.bits_to_indices(bits, modulation)
    assert indices.size == 257
    assert indices.max() < mapping.num_points(modulation)
    assert np.array_equal(mapping.indices_to_bits(indices, modulation), bits)


@pytest.mark.parametrize("modulation", PSK_QAM)
def test_unit_average_symbol_energy(modulation):
    points = mapping.constellation(modulation)
    assert abs(float(np.mean(np.abs(points) ** 2)) - 1.0) < 1e-12


@pytest.mark.parametrize("modulation", PSK_QAM)
def test_gray_mapping_nearest_neighbours_differ_by_one_bit(modulation):
    points = mapping.constellation(modulation)
    bps = mapping.bits_per_symbol(modulation)
    dist = np.abs(points[:, None] - points[None, :])
    np.fill_diagonal(dist, np.inf)
    min_dist = dist.min()
    values = np.arange(points.size)
    bits = ((values[:, None] >> np.arange(bps - 1, -1, -1)[None, :]) & 1)
    for i in range(points.size):
        neighbours = np.flatnonzero(np.isclose(dist[i], min_dist, atol=1e-12))
        assert neighbours.size >= 1
        for j in neighbours:
            assert np.sum(bits[i] != bits[j]) == 1, (modulation, i, int(j))


def test_min_distance_matches_theory():
    assert abs(mapping.min_distance("bpsk") - 2.0) < 1e-12
    assert abs(mapping.min_distance("qpsk") - np.sqrt(2.0)) < 1e-12
    assert abs(mapping.min_distance("8psk") - 2.0 * np.sin(np.pi / 8)) < 1e-12
    assert abs(mapping.min_distance("16qam") - 2.0 / np.sqrt(10.0)) < 1e-12


@pytest.mark.parametrize("modulation", PSK_QAM)
def test_ml_decision_is_exact_without_noise(modulation):
    indices = np.arange(mapping.num_points(modulation)).repeat(5)
    points = mapping.indices_to_points(indices, modulation)
    assert np.array_equal(mapping.points_to_indices(points, modulation), indices)


def test_fsk_points_are_rejected():
    with pytest.raises(ValueError):
        mapping.constellation("2fsk")
    with pytest.raises(ValueError):
        mapping.indices_to_points(np.array([0, 1]), "4fsk")


def test_gray_helpers_inverse():
    values = np.arange(64)
    assert np.array_equal(mapping.gray_to_bin(mapping.bin_to_gray(values)), values)
