"""Barker-13 序列与滑动相关测试."""
from __future__ import annotations

import numpy as np

from spxh.core.framing.barker import (
    BARKER_13,
    aperiodic_autocorrelation,
    barker_bits,
    barker_pm,
    correlate_bits,
    find_sync,
    verify_barker_property,
)


def test_barker13_bit_pattern_matches_article():
    assert barker_bits(13).tolist() == [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    assert BARKER_13.tolist() == [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]


def test_all_barker_lengths_have_sidelobe_at_most_one():
    for n in (2, 3, 4, 5, 7, 11, 13):
        assert verify_barker_property(n), "barker " + str(n)


def test_autocorrelation_peak_equals_length():
    ac = aperiodic_autocorrelation(barker_pm(13))
    center = ac.size // 2
    assert ac[center] == 13
    assert np.max(np.abs(np.delete(ac, center))) <= 1


def test_correlation_peak_is_one_and_located():
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, size=200, dtype=np.uint8)
    stream = np.concatenate([payload[:50], barker_bits(13), payload[50:]])
    corr = correlate_bits(stream, n=13)
    peak = int(np.argmax(corr))
    assert peak == 50
    assert abs(corr[peak] - 1.0) < 1e-12


def test_find_sync_locates_all_frames():
    rng = np.random.default_rng(1)
    frame = np.concatenate([barker_bits(13), rng.integers(0, 2, size=64, dtype=np.uint8)])
    stream = np.concatenate([frame, frame, frame])
    hits = find_sync(stream, n=13, threshold=0.9, min_gap=40)
    assert hits == [0, 77, 154]
