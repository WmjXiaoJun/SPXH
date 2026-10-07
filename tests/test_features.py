"""M2 特征工程测试."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.features import FEATURE_GROUPS, FEATURE_NAMES, FeatureConfig, FeatureSet, extract_features
from spxh.core.synth import SynthConfig, synthesize


def test_feature_names_and_groups_are_consistent():
    assert len(FEATURE_NAMES) == 23
    assert len(set(FEATURE_NAMES)) == 23
    covered = sorted(index for indices in FEATURE_GROUPS.values() for index in indices)
    assert covered == list(range(23))
    assert set(FEATURE_GROUPS) == {"cumulant", "spectral", "timefreq", "constellation", "cyclostationary"}


def test_extraction_is_deterministic(synth_case):
    signal, _ = synth_case(modulation="qpsk", snr_db=20.0)
    first = extract_features(signal)
    second = extract_features(signal)
    assert np.array_equal(first.vector(), second.vector())
    assert first.names == list(FEATURE_NAMES)


def test_evidence_carries_estimates_and_config(synth_case):
    signal, truth = synth_case(modulation="16qam", snr_db=20.0)
    feature_set = extract_features(signal)
    assert abs(feature_set.evidence["symbol_rate"] - truth.symbol_rate) / truth.symbol_rate < 5e-3
    assert abs(feature_set.evidence["es_n0_db"] - truth.snr_db) < 1.0
    assert feature_set.evidence["config"]["rolloff_assumed"] == FeatureConfig().rolloff_assumed
    assert feature_set.evidence["num_symbols_used"] > 0


@pytest.mark.parametrize(
    "modulation,expected",
    [("bpsk", (1.4, 2.6)), ("qpsk", (0.6, 1.4)), ("16qam", (0.4, 0.95))],
)
def test_cumulant_c40_matches_theory_order(synth_case, modulation, expected):
    """符号速率采样上的 |C40| 应接近理论值（BPSK 2 / QPSK 1 / 16QAM 0.68）."""
    signal, _ = synth_case(modulation=modulation, snr_db=20.0)
    value = extract_features(signal)["cum_c40"]
    assert expected[0] <= value <= expected[1], (modulation, value)


def test_cumulants_are_small_for_8psk(synth_case):
    signal, _ = synth_case(modulation="8psk", snr_db=20.0)
    assert extract_features(signal)["cum_c40"] < 0.4


@pytest.mark.parametrize("modulation,low,high", [("qpsk", 3, 6), ("8psk", 6, 12), ("16qam", 12, 24)])
def test_cluster_count_tracks_constellation_size(synth_case, modulation, low, high):
    signal, _ = synth_case(modulation=modulation, snr_db=20.0)
    clusters = extract_features(signal)["const_cluster_count"]
    assert low <= clusters <= high, (modulation, clusters)


def test_radial_spread_separates_psk_from_qam(synth_case):
    qpsk_signal, _ = synth_case(modulation="qpsk", snr_db=20.0)
    qam_signal, _ = synth_case(modulation="16qam", snr_db=20.0)
    assert extract_features(qam_signal)["const_radial_spread"] > extract_features(qpsk_signal)["const_radial_spread"]


def test_fsk_is_detected_as_not_rrc_shaped(synth_case):
    fsk_signal, _ = synth_case(modulation="2fsk", snr_db=20.0)
    psk_signal, _ = synth_case(modulation="qpsk", snr_db=20.0)
    assert extract_features(fsk_signal)["spec_rolloff_est"] > 0.7
    assert extract_features(psk_signal)["spec_rolloff_est"] < 0.6


def test_feature_set_zeroes_non_finite_values():
    values = np.array([1.0, np.nan, np.inf, 2.0])
    feature_set = FeatureSet(names=["a", "b", "c", "d"], values=values, groups={"all": [0, 1, 2, 3]})
    assert np.all(np.isfinite(feature_set.values))
    assert feature_set.warnings


def test_feature_set_rejects_length_mismatch():
    with pytest.raises(ValueError):
        FeatureSet(names=["a", "b"], values=[1.0], groups={})


def test_as_dict_is_json_serializable(synth_case):
    import json

    signal, _ = synth_case()
    payload = extract_features(signal).as_dict()
    json.dumps(payload)
    assert len(payload["values"]) == 23
    assert payload["values_by_name"]["cum_c40"] == payload["values"][0]
